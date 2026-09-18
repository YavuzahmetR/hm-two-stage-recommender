import json

import numpy as np
import polars as pl
import torch

from prepare_two_tower import OUTPUT_DIR
from train_two_tower import TwoTower
from train_v6 import DATA_DIR, PROCESSED_DIR, VALIDATION_DATE


K = 150
BATCH_SIZE = 256

REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"


def merge_candidates(base_items, tower_items):
    merged = list(
        dict.fromkeys(
            base_items[:100] + tower_items[:50] + base_items
        )
    )
    return merged[:K]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    metadata = json.loads(
        (OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    checkpoint = torch.load(
        OUTPUT_DIR / "two_tower_v1.pt",
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint["as_of"] != str(VALIDATION_DATE)
        or metadata["as_of"] != str(VALIDATION_DATE)
    ):
        raise ValueError("Unexpected two-tower snapshot date.")

    users = pl.read_parquet(
        OUTPUT_DIR / "users.parquet"
    ).sort("user_idx")

    items = pl.read_parquet(
        OUTPUT_DIR / "items.parquet"
    ).sort("item_idx")

    for frame, index, id_column, expected in [
        (users, "user_idx", "customer_id", checkpoint["n_users"]),
        (items, "item_idx", "article_id", checkpoint["n_items"]),
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

    if items.height < K:
        raise ValueError("Not enough products for retrieval.")

    model = TwoTower(
        checkpoint["n_users"],
        checkpoint["n_items"],
        checkpoint["embedding_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{VALIDATION_DATE}_v6.parquet"
    ).select("customer_id", "actual_items").sort("customer_id")

    candidates = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v6.parquet"
    )

    if candidates["as_of"].unique().to_list() != [VALIDATION_DATE]:
        raise ValueError("Unexpected candidate snapshot date.")

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

    known = evaluation.filter(pl.col("user_idx").is_not_null())
    article_ids = items["article_id"].to_list()

    tower_lookup = {}
    retrieval_rows = []

    print(f"Device: {device}")
    print(f"Evaluation customers: {evaluation.height}")
    print(f"Customers known to model: {known.height}")
    print(f"Retrieval products: {items.height}")

    with torch.inference_mode():
        item_indices = torch.arange(
            items.height,
            dtype=torch.long,
            device=device,
        )
        item_vectors = model.encode_items(item_indices)

        for start in range(0, known.height, BATCH_SIZE):
            batch = known.slice(start, BATCH_SIZE)

            user_indices = torch.tensor(
                batch["user_idx"].to_list(),
                dtype=torch.long,
                device=device,
            )

            user_vectors = model.encode_users(user_indices)
            scores = user_vectors @ item_vectors.T

            if not torch.isfinite(scores).all().item():
                raise ValueError("Non-finite retrieval scores.")

            top_indices = torch.argsort(
                scores,
                dim=1,
                descending=True,
                stable=True,
            )[:, :K]

            top_scores = torch.gather(
                scores,
                dim=1,
                index=top_indices,
            ).cpu().tolist()

            top_indices = top_indices.cpu().tolist()

            for customer_id, indices, similarities in zip(
                batch["customer_id"].to_list(),
                top_indices,
                top_scores,
            ):
                recommendations = [
                    article_ids[index] for index in indices
                ]

                tower_lookup[customer_id] = recommendations
                retrieval_rows.append(
                    {
                        "customer_id": customer_id,
                        "tower_items": recommendations,
                        "tower_scores": similarities,
                    }
                )

    records = []

    for row in evaluation.iter_rows(named=True):
        customer_id = row["customer_id"]
        actual_items = set(row["actual_items"])
        base_items = row["base_items"]
        has_history = row["user_idx"] is not None

        if not actual_items:
            raise ValueError("Empty actual purchase list.")

        if len(base_items) != K or len(set(base_items)) != K:
            raise ValueError("Expected 150 unique baseline candidates.")

        tower_items = tower_lookup.get(customer_id, [])

        if has_history and len(tower_items) != K:
            raise ValueError("Missing two-tower recommendations.")

        standalone = tower_items if has_history else base_items
        combined = merge_candidates(base_items, tower_items)
        base_set = set(base_items)

        for method, predicted in [
            ("Baseline v6", base_items),
            ("Two-tower + fallback", standalone),
            ("Baseline + two-tower", combined),
        ]:
            if len(predicted) != K or len(set(predicted)) != K:
                raise ValueError("Expected 150 unique candidates.")

            predicted_set = set(predicted)
            hits = len(actual_items & predicted_set)

            records.append(
                {
                    "customer_id": customer_id,
                    "has_history": has_history,
                    "method": method,
                    "candidate_recall_at_150": hits / len(actual_items),
                    "candidate_hit_rate_at_150": float(hits > 0),
                    "new_candidates": len(predicted_set - base_set),
                    "gained_hits": len(
                        actual_items & (predicted_set - base_set)
                    ),
                    "lost_hits": len(
                        actual_items & (base_set - predicted_set)
                    ),
                }
            )

    results = pl.DataFrame(records)
    summaries = []

    for segment, frame in [
        ("all", results),
        ("history_28d", results.filter(pl.col("has_history"))),
        ("no_history_28d", results.filter(~pl.col("has_history"))),
    ]:
        summary = (
            frame
            .group_by("method")
            .agg(
                pl.len().alias("customers"),
                pl.col("candidate_recall_at_150").mean(),
                pl.col("candidate_hit_rate_at_150").mean(),
                pl.col("new_candidates").mean().alias(
                    "mean_new_candidates"
                ),
                pl.col("gained_hits").sum().alias("gained_hits"),
                pl.col("lost_hits").sum().alias("lost_hits"),
            )
            .with_columns(pl.lit(segment).alias("segment"))
            .select(
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
        summaries.append(summary)

    report = pl.concat(summaries)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_csv(REPORT_DIR / "two_tower_v1_retrieval.csv")

    if retrieval_rows:
        pl.DataFrame(retrieval_rows).write_parquet(
            OUTPUT_DIR / "validation_retrieval_v1.parquet"
        )

    with pl.Config(
        tbl_rows=20,
        tbl_cols=8,
        tbl_width_chars=180,
        float_precision=6,
    ):
        print(report)

    print("\nSaved two_tower_v1_retrieval.csv")
    print("Saved retrieval lists for customers known to the model.")


if __name__ == "__main__":
    main()