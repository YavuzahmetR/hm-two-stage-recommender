import json
import random

import numpy as np
import polars as pl
import torch
from torch import nn

from prepare_two_tower import OUTPUT_DIR
from train_two_tower import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    NEGATIVES,
    SEED,
    TEMPERATURE,
    TwoTower,
    sample_negatives,
)
from train_v6 import DATA_DIR, PROCESSED_DIR, VALIDATION_DATE


CHECKPOINT_EPOCHS = {5, 10, 15}
MAX_EPOCHS = max(CHECKPOINT_EPOCHS)
RETRIEVAL_BATCH_SIZE = 256
CANDIDATE_COUNT = 150
TOWER_QUOTAS = [10, 25, 50]

REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"
EXPERIMENT_DIR = OUTPUT_DIR / "id_experiment"


def load_evaluation(users):
    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{VALIDATION_DATE}_v6.parquet"
    ).select("customer_id", "actual_items")

    candidates = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v6.parquet"
    )

    if candidates["as_of"].unique().to_list() != [VALIDATION_DATE]:
        raise ValueError("Unexpected validation date.")

    baseline = (
        candidates
        .sort(
            [
                "customer_id",
                "repeat_rank",
                "variant_rank",
                "covisit_rank",
                "popularity_rank",
                "article_id",
            ],
            nulls_last=True,
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").alias("base_items"))
    )

    if set(actual["customer_id"]) != set(baseline["customer_id"]):
        raise ValueError("Evaluation customer sets do not match.")

    evaluation = (
        actual
        .join(
            baseline,
            on="customer_id",
            how="left",
            validate="1:1",
        )
        .join(
            users,
            on="customer_id",
            how="left",
            validate="1:1",
        )
        .sort("customer_id")
    )

    for row in evaluation.iter_rows(named=True):
        if not row["actual_items"]:
            raise ValueError("Empty actual purchase list.")

        base = row["base_items"]
        if base is None or len(base) != 150 or len(set(base)) != 150:
            raise ValueError("Expected 150 unique baseline candidates.")

    return evaluation


@torch.inference_mode()
def retrieve(model, evaluation, article_ids, device):
    model.eval()

    known = evaluation.filter(pl.col("user_idx").is_not_null())
    item_indices = torch.arange(
        len(article_ids),
        dtype=torch.long,
        device=device,
    )
    item_vectors = model.encode_items(item_indices)

    lookup = {}
    saved_rows = []

    for start in range(0, known.height, RETRIEVAL_BATCH_SIZE):
        batch = known.slice(start, RETRIEVAL_BATCH_SIZE)

        user_indices = torch.tensor(
            batch["user_idx"].to_list(),
            dtype=torch.long,
            device=device,
        )

        scores = model.encode_users(user_indices) @ item_vectors.T

        if not torch.isfinite(scores).all().item():
            raise ValueError("Non-finite retrieval scores.")

        indices = torch.argsort(
            scores,
            dim=1,
            descending=True,
            stable=True,
        )[:, :CANDIDATE_COUNT]

        similarities = torch.gather(
            scores,
            dim=1,
            index=indices,
        ).cpu().tolist()

        for customer_id, item_indices, item_scores in zip(
            batch["customer_id"].to_list(),
            indices.cpu().tolist(),
            similarities,
        ):
            recommendations = [
                article_ids[index] for index in item_indices
            ]
            lookup[customer_id] = recommendations
            saved_rows.append(
                {
                    "customer_id": customer_id,
                    "tower_items": recommendations,
                    "tower_scores": item_scores,
                }
            )

    return lookup, saved_rows


def evaluate_retrieval(evaluation, lookup, epoch):
    records = []

    for row in evaluation.iter_rows(named=True):
        customer_id = row["customer_id"]
        actual = set(row["actual_items"])
        base = row["base_items"]
        base_set = set(base)
        has_history = row["user_idx"] is not None
        tower = lookup.get(customer_id, [])

        if has_history and len(tower) != CANDIDATE_COUNT:
            raise ValueError("Missing retrieval results.")

        methods = [
            ("Baseline v6", base),
            ("Tower only + fallback", tower if has_history else base),
        ]

        for quota in TOWER_QUOTAS:
            combined = list(
                dict.fromkeys(
                    base[:CANDIDATE_COUNT - quota]
                    + tower[:quota]
                    + base
                )
            )[:CANDIDATE_COUNT]

            methods.append((f"Combined tower={quota}", combined))

        for method, predicted in methods:
            predicted_set = set(predicted)

            if (
                len(predicted) != CANDIDATE_COUNT
                or len(predicted_set) != CANDIDATE_COUNT
            ):
                raise ValueError("Invalid candidate list.")

            hits = len(actual & predicted_set)

            records.append(
                {
                    "method": method,
                    "has_history": has_history,
                    "recall": hits / len(actual),
                    "hit_rate": float(hits > 0),
                    "new_candidates": len(predicted_set - base_set),
                    "gained_hits": len(actual & (predicted_set - base_set)),
                    "lost_hits": len(actual & (base_set - predicted_set)),
                }
            )

    results = pl.DataFrame(records)
    summaries = []

    for segment, frame in [
        ("all", results),
        ("history_28d", results.filter(pl.col("has_history"))),
        ("no_history_28d", results.filter(~pl.col("has_history"))),
    ]:
        if frame.is_empty():
            continue

        summaries.append(
            frame
            .group_by("method")
            .agg(
                pl.len().alias("customers"),
                pl.col("recall").mean().alias("candidate_recall_at_150"),
                pl.col("hit_rate").mean().alias("candidate_hit_rate_at_150"),
                pl.col("new_candidates").mean().alias("mean_new_candidates"),
                pl.col("gained_hits").sum(),
                pl.col("lost_hits").sum(),
            )
            .with_columns(
                pl.lit(epoch).alias("epoch"),
                pl.lit(segment).alias("segment"),
            )
            .select(
                "epoch",
                "segment",
                "method",
                "customers",
                "candidate_recall_at_150",
                "candidate_hit_rate_at_150",
                "mean_new_candidates",
                "gained_hits",
                "lost_hits",
            )
        )

    return pl.concat(summaries)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    rng = np.random.default_rng(SEED)
    device = torch.device("cuda")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    if metadata["as_of"] != str(VALIDATION_DATE):
        raise ValueError("Unexpected training snapshot date.")

    users = pl.read_parquet(
        OUTPUT_DIR / "users.parquet"
    ).sort("user_idx")

    items = pl.read_parquet(
        OUTPUT_DIR / "items.parquet"
    ).sort("item_idx")

    pairs = pl.read_parquet(OUTPUT_DIR / "interactions.parquet")

    n_users = metadata["n_users"]
    n_items = metadata["n_items"]

    for frame, index, id_column, expected in [
        (users, "user_idx", "customer_id", n_users),
        (items, "item_idx", "article_id", n_items),
    ]:
        if (
            frame.height != expected
            or frame[id_column].n_unique() != expected
            or not np.array_equal(
                frame[index].to_numpy(),
                np.arange(expected),
            )
        ):
            raise ValueError("Invalid ID mapping.")

    user_ids = pairs["user_idx"].to_numpy().astype(np.int64)
    item_ids = pairs["item_idx"].to_numpy().astype(np.int64)

    if len(user_ids) == 0 or len(user_ids) != metadata["n_pairs"]:
        raise ValueError("Invalid interaction count.")

    if not (
        0 <= user_ids.min() <= user_ids.max() < n_users
        and 0 <= item_ids.min() <= item_ids.max() < n_items
    ):
        raise ValueError("Invalid interaction indices.")

    if n_items < CANDIDATE_COUNT:
        raise ValueError("Not enough items for retrieval.")

    known_keys = np.unique(user_ids * n_items + item_ids)

    if len(known_keys) != len(user_ids):
        raise ValueError("Duplicate training pairs.")

    if (np.bincount(user_ids, minlength=n_users) >= n_items).any():
        raise ValueError("A user has no available negative items.")

    evaluation = load_evaluation(users)
    article_ids = items["article_id"].to_list()

    model = TwoTower(n_users, n_items, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    reports = []
    training_history = []

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Training pairs: {len(user_ids):,}")
    print(f"Validation customers: {evaluation.height:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(user_ids))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_users = user_ids[indices]
            positives = item_ids[indices]

            negatives = sample_negatives(
                batch_users,
                n_items,
                known_keys,
                rng,
            )
            candidate_items = np.concatenate(
                [positives[:, None], negatives],
                axis=1,
            )

            users_tensor = torch.as_tensor(
                batch_users,
                dtype=torch.long,
                device=device,
            )
            items_tensor = torch.as_tensor(
                candidate_items,
                dtype=torch.long,
                device=device,
            )
            targets = torch.zeros(
                len(indices),
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(users_tensor, items_tensor), targets)

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += loss.item() * len(indices)
            seen += len(indices)

        average_loss = total_loss / seen
        training_history.append(
            {"epoch": epoch, "training_loss": average_loss}
        )

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Training loss: {average_loss:.6f}",
            flush=True,
        )

        pl.DataFrame(training_history).write_csv(
            REPORT_DIR / "two_tower_id_training.csv"
        )

        if epoch not in CHECKPOINT_EPOCHS:
            continue

        torch.save(
            {
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": EMBEDDING_DIM,
                "temperature": TEMPERATURE,
                "epochs": epoch,
                "seed": SEED,
                "as_of": metadata["as_of"],
                "lookback_days": metadata["lookback_days"],
                "negatives": NEGATIVES,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
            },
            EXPERIMENT_DIR / f"model_epoch_{epoch}.pt",
        )

        print(f"Evaluating epoch {epoch}...", flush=True)

        lookup, retrieval_rows = retrieve(
            model,
            evaluation,
            article_ids,
            device,
        )

        if retrieval_rows:
            pl.DataFrame(retrieval_rows).write_parquet(
                EXPERIMENT_DIR / f"retrieval_epoch_{epoch}.parquet"
            )

        report = evaluate_retrieval(evaluation, lookup, epoch)
        reports.append(report)

        pl.concat(reports).write_csv(
            REPORT_DIR / "two_tower_id_experiments.csv"
        )

        with pl.Config(tbl_cols=9, tbl_width_chars=180):
            print(
                report
                .filter(pl.col("segment") == "all")
                .sort("candidate_recall_at_150", descending=True)
            )

        del lookup, retrieval_rows

    final_report = pl.concat(reports)

    print("\nFinal comparison:")
    with pl.Config(
        tbl_rows=20,
        tbl_cols=9,
        tbl_width_chars=180,
        float_precision=6,
    ):
        print(
            final_report
            .filter(pl.col("segment") == "all")
            .sort("candidate_recall_at_150", descending=True)
        )

    print("\nSaved two_tower_id_experiments.csv")
    print("Saved two_tower_id_training.csv")
    print(f"Saved model checkpoints and retrieval lists to {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()