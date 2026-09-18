import argparse
import json
import random
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from experiment_two_tower import evaluate_retrieval, load_evaluation
from popularity_baseline import DATA_DIR
from train_two_tower import NEGATIVES, sample_negatives


SEED = 42
HISTORY_DAYS = 28
HISTORY_SIZE = 20
BATCH_SIZE = 1024
RETRIEVAL_BATCH_SIZE = 256
EPOCHS = 15
CHECKPOINT_EPOCHS = {5, 10, 15}
LEARNING_RATE = 0.001
TEMPERATURE = 0.1
VALIDATION_DATE = date(2020, 9, 9)

REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"


class FeatureTwoTower(nn.Module):
    def __init__(self, n_items, n_types, n_colors, item_features):
        super().__init__()

        self.register_buffer(
            "item_features",
            torch.as_tensor(item_features, dtype=torch.long),
        )

        self.item_embedding = nn.Embedding(
            n_items + 1, 32, padding_idx=0
        )
        self.type_embedding = nn.Embedding(
            n_types, 8, padding_idx=0
        )
        self.color_embedding = nn.Embedding(
            n_colors, 8, padding_idx=0
        )

        self.item_tower = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.user_tower = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def encode_items(self, item_ids):
        attributes = self.item_features[item_ids]

        features = torch.cat(
            [
                self.item_embedding(item_ids),
                self.type_embedding(attributes[..., 0]),
                self.color_embedding(attributes[..., 1]),
            ],
            dim=-1,
        )

        return F.normalize(self.item_tower(features), dim=-1)

    def encode_users(self, histories):
        vectors = self.encode_items(histories)
        mask = histories.ne(0).unsqueeze(-1)

        pooled = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        return F.normalize(self.user_tower(pooled), dim=-1)

    def forward(self, histories, candidate_items):
        user_vectors = self.encode_users(histories)
        item_vectors = self.encode_items(candidate_items)

        return (
            user_vectors.unsqueeze(1) * item_vectors
        ).sum(dim=-1) / TEMPERATURE


def encode_attribute(values):
    vocabulary = {
        value: index + 1
        for index, value in enumerate(
            sorted({value for value in values if value is not None})
        )
    }

    encoded = np.array(
        [vocabulary.get(value, 0) for value in values],
        dtype=np.int64,
    )

    return encoded, vocabulary


def prepare_data(as_of, output_dir):
    history_start = as_of - timedelta(days=HISTORY_DAYS)

    history = (
        pl.scan_csv(
            DATA_DIR / "transactions_train.csv",
            schema_overrides={
                "customer_id": pl.String,
                "article_id": pl.String,
            },
            try_parse_dates=True,
        )
        .filter(
            (pl.col("t_dat") >= history_start)
            & (pl.col("t_dat") < as_of)
        )
        .select("customer_id", "article_id", "t_dat")
        .drop_nulls()
        .unique()
        .collect()
    )

    if history.is_empty():
        raise ValueError("Empty history window.")

    if (
        history["t_dat"].min() < history_start
        or history["t_dat"].max() >= as_of
    ):
        raise ValueError("Invalid history dates.")

    users = (
        history.select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index("user_idx")
    )
    items = (
        history.select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index("item_idx")
    )

    articles = (
        pl.scan_csv(
            DATA_DIR / "articles.csv",
            schema_overrides={"article_id": pl.String},
        )
        .select(
            "article_id",
            pl.col("product_type_no").cast(pl.String),
            pl.col("colour_group_code").cast(pl.String),
        )
        .collect()
    )

    if articles["article_id"].n_unique() != articles.height:
        raise ValueError("Duplicate article metadata.")

    if items.join(articles, on="article_id", how="anti").height:
        raise ValueError("Missing article metadata.")

    catalog = items.join(
        articles,
        on="article_id",
        how="left",
        validate="1:1",
    ).sort("item_idx")

    types, type_vocabulary = encode_attribute(
        catalog["product_type_no"].to_list()
    )
    colors, color_vocabulary = encode_attribute(
        catalog["colour_group_code"].to_list()
    )

    item_features = np.zeros((items.height + 1, 2), dtype=np.int64)
    item_features[1:, 0] = types
    item_features[1:, 1] = colors

    indexed = (
        history
        .join(users, on="customer_id", validate="m:1")
        .join(items, on="article_id", validate="m:1")
        .select("user_idx", "item_idx", "t_dat")
    )

    if indexed.height != history.height:
        raise ValueError("Rows changed during ID mapping.")

    known_pairs = indexed.select("user_idx", "item_idx").unique()
    pair_users = known_pairs["user_idx"].to_numpy().astype(np.int64)
    pair_items = known_pairs["item_idx"].to_numpy().astype(np.int64)
    known_keys = np.sort(pair_users * items.height + pair_items)

    counts = np.bincount(pair_users, minlength=users.height)
    if (counts >= items.height).any():
        raise ValueError("A user has no available negative items.")

    daily = (
        indexed
        .group_by(["user_idx", "t_dat"])
        .agg(pl.col("item_idx").sort())
        .sort(["user_idx", "t_dat"])
    )

    train_histories = np.zeros(
        (indexed.height, HISTORY_SIZE), dtype=np.int32
    )
    train_users = np.empty(indexed.height, dtype=np.int64)
    train_targets = np.empty(indexed.height, dtype=np.int64)

    inference_histories = np.zeros(
        (users.height, HISTORY_SIZE), dtype=np.int32
    )

    current_user = None
    recent = deque(maxlen=HISTORY_SIZE)
    last_day = None
    sample_count = 0

    for row in daily.iter_rows(named=True):
        user_idx = row["user_idx"]
        target_day = row["t_dat"]
        day_items = row["item_idx"]

        if current_user != user_idx:
            current_user = user_idx
            recent = deque(maxlen=HISTORY_SIZE)
            last_day = None

        if recent:
            if last_day is None or last_day >= target_day:
                raise ValueError("Target leaked into history.")

            previous_items = list(recent)

            for target in day_items:
                train_histories[
                    sample_count, :len(previous_items)
                ] = previous_items
                train_users[sample_count] = user_idx
                train_targets[sample_count] = target
                sample_count += 1

        # Add today's items only after building today's examples.
        recent.extend(item + 1 for item in day_items)
        last_day = target_day

        inference_histories[user_idx] = 0
        inference_histories[user_idx, :len(recent)] = list(recent)

    if sample_count == 0:
        raise ValueError("No chronological training examples.")

    train_histories = train_histories[:sample_count].copy()
    train_users = train_users[:sample_count].copy()
    train_targets = train_targets[:sample_count].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    users.write_parquet(output_dir / "users.parquet")
    items.write_parquet(output_dir / "items.parquet")
    np.save(output_dir / "item_features.npy", item_features)
    np.save(output_dir / "user_histories.npy", inference_histories)

    metadata = {
        "as_of": str(as_of),
        "history_start": str(history_start),
        "history_days": HISTORY_DAYS,
        "history_size": HISTORY_SIZE,
        "n_users": users.height,
        "n_items": items.height,
        "n_types": len(type_vocabulary) + 1,
        "n_colors": len(color_vocabulary) + 1,
        "training_examples": sample_count,
        "type_vocabulary": type_vocabulary,
        "color_vocabulary": color_vocabulary,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"History: [{history_start}, {as_of})")
    print(f"Users: {users.height:,}")
    print(f"Items: {items.height:,}")
    print(f"Chronological training examples: {sample_count:,}")
    print(
        "Users contributing training examples: "
        f"{len(np.unique(train_users)):,}"
    )

    return (
        metadata,
        users,
        items,
        item_features,
        inference_histories,
        train_histories,
        train_users,
        train_targets,
        known_keys,
    )


@torch.inference_mode()
def retrieve(model, evaluation, items, histories, device):
    model.eval()

    known = evaluation.filter(pl.col("user_idx").is_not_null())
    article_ids = items["article_id"].to_list()

    item_ids = torch.arange(
        1, items.height + 1, dtype=torch.long, device=device
    )
    item_vectors = model.encode_items(item_ids)

    lookup = {}
    saved_rows = []

    for start in range(0, known.height, RETRIEVAL_BATCH_SIZE):
        batch = known.slice(start, RETRIEVAL_BATCH_SIZE)
        user_indices = batch["user_idx"].to_numpy().astype(np.int64)

        history_tensor = torch.as_tensor(
            histories[user_indices],
            dtype=torch.long,
            device=device,
        )

        scores = model.encode_users(history_tensor) @ item_vectors.T

        if not torch.isfinite(scores).all().item():
            raise ValueError("Non-finite retrieval scores.")

        top_indices = torch.argsort(
            scores,
            dim=1,
            descending=True,
            stable=True,
        )[:, :150]

        top_scores = torch.gather(
            scores, dim=1, index=top_indices
        ).cpu().tolist()

        for customer_id, indices, similarities in zip(
            batch["customer_id"].to_list(),
            top_indices.cpu().tolist(),
            top_scores,
        ):
            recommendations = [article_ids[index] for index in indices]
            lookup[customer_id] = recommendations
            saved_rows.append(
                {
                    "customer_id": customer_id,
                    "tower_items": recommendations,
                    "tower_scores": similarities,
                }
            )

    return lookup, saved_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        choices=[
            "2020-08-19",
            "2020-08-26",
            "2020-09-02",
            "2020-09-09",
        ],
        default="2020-09-09",
    )
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    rng = np.random.default_rng(SEED)
    device = torch.device("cuda")

    output_dir = (
        DATA_DIR.parent.parent / "two_tower_features" / str(as_of)
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (
        metadata,
        users,
        items,
        item_features,
        inference_histories,
        train_histories,
        train_users,
        train_targets,
        known_keys,
    ) = prepare_data(as_of, output_dir)

    if items.height < 150:
        raise ValueError("Not enough retrieval items.")

    model = FeatureTwoTower(
        metadata["n_items"],
        metadata["n_types"],
        metadata["n_colors"],
        item_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    evaluation = (
        load_evaluation(users)
        if as_of == VALIDATION_DATE
        else None
    )

    reports = []
    training_history = []

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        "Parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_targets))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]

            negatives = sample_negatives(
                train_users[indices],
                metadata["n_items"],
                known_keys,
                rng,
            )

            candidate_items = np.concatenate(
                [train_targets[indices, None], negatives],
                axis=1,
            ) + 1

            histories_tensor = torch.as_tensor(
                train_histories[indices],
                dtype=torch.long,
                device=device,
            )
            items_tensor = torch.as_tensor(
                candidate_items,
                dtype=torch.long,
                device=device,
            )
            targets = torch.zeros(
                len(indices), dtype=torch.long, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(histories_tensor, items_tensor), targets
            )

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            total_loss += loss.item() * len(indices)
            seen += len(indices)

        average_loss = total_loss / seen
        training_history.append(
            {"epoch": epoch, "training_loss": average_loss}
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | Loss: {average_loss:.6f}",
            flush=True,
        )

        pl.DataFrame(training_history).write_csv(
            REPORT_DIR / f"feature_tower_training_{as_of}.csv"
        )

        if epoch not in CHECKPOINT_EPOCHS:
            continue

        torch.save(
            {
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "config": {
                    "n_items": metadata["n_items"],
                    "n_types": metadata["n_types"],
                    "n_colors": metadata["n_colors"],
                },
                "as_of": str(as_of),
                "epoch": epoch,
                "history_days": HISTORY_DAYS,
                "history_size": HISTORY_SIZE,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "negatives": NEGATIVES,
                "learning_rate": LEARNING_RATE,
            },
            output_dir / f"model_epoch_{epoch}.pt",
        )

        if evaluation is None:
            continue

        print(f"Evaluating epoch {epoch}...", flush=True)

        lookup, retrieval_rows = retrieve(
            model, evaluation, items, inference_histories, device
        )

        if retrieval_rows:
            pl.DataFrame(retrieval_rows).write_parquet(
                output_dir / f"retrieval_epoch_{epoch}.parquet"
            )

        report = evaluate_retrieval(evaluation, lookup, epoch)
        reports.append(report)

        pl.concat(reports).write_csv(
            REPORT_DIR / "feature_tower_retrieval.csv"
        )

        with pl.Config(tbl_cols=9, tbl_width_chars=180):
            print(
                report.filter(pl.col("segment") == "all")
                .sort("candidate_recall_at_150", descending=True)
            )

    if reports:
        print("\nFinal feature-tower comparison:")
        with pl.Config(
            tbl_rows=20,
            tbl_cols=9,
            tbl_width_chars=180,
            float_precision=6,
        ):
            print(
                pl.concat(reports)
                .filter(pl.col("segment") == "all")
                .sort("candidate_recall_at_150", descending=True)
            )

    print(f"\nSaved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()