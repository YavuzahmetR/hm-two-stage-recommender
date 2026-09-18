import gc
import json
import subprocess
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from lightgbm import Booster, LGBMRanker

from covisitation import build_neighbors
from feature_two_tower import FeatureTwoTower, retrieve
from popularity_baseline import DATA_DIR
from ranking_dataset import build_user_item_features
from train_ranker import evaluate_predictions
from train_v6 import (
    FEATURES as BASE_FEATURES,
    PROCESSED_DIR,
    TRAIN_DATES,
    VALIDATION_DATE,
    get_predictions,
)


TOWER_EPOCH = 5
TOWER_QUOTA = 25
CANDIDATE_COUNT = 150

FEATURES = BASE_FEATURES + ["tower_score", "tower_rank"]

TOWER_ROOT = DATA_DIR.parent.parent / "two_tower_features"
REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"
SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_tower(as_of):
    checkpoint_path = (
        TOWER_ROOT / str(as_of) / f"model_epoch_{TOWER_EPOCH}.pt"
    )

    if checkpoint_path.exists():
        print(f"Using saved tower: {as_of}", flush=True)
        return

    if as_of == VALIDATION_DATE:
        raise FileNotFoundError(
            "Expected the existing validation epoch-5 checkpoint."
        )

    print(f"\nTraining weekly tower: {as_of}", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import feature_two_tower as ft; "
                "ft.EPOCHS = 5; "
                "ft.CHECKPOINT_EPOCHS = {5}; "
                "ft.main()"
            ),
            "--as-of",
            str(as_of),
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)


def load_tower(as_of, device):
    folder = TOWER_ROOT / str(as_of)

    checkpoint = torch.load(
        folder / f"model_epoch_{TOWER_EPOCH}.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = json.loads(
        (folder / "metadata.json").read_text(encoding="utf-8")
    )

    if (
        checkpoint["as_of"] != str(as_of)
        or metadata["as_of"] != str(as_of)
        or checkpoint["epoch"] != TOWER_EPOCH
        or checkpoint["history_days"] != 28
    ):
        raise ValueError("Unexpected tower checkpoint.")

    users = pl.read_parquet(
        folder / "users.parquet"
    ).sort("user_idx")
    items = pl.read_parquet(
        folder / "items.parquet"
    ).sort("item_idx")

    histories = np.load(
        folder / "user_histories.npy", allow_pickle=False
    )
    item_features = np.load(
        folder / "item_features.npy", allow_pickle=False
    )

    for frame, index, id_column, expected in [
        (users, "user_idx", "customer_id", metadata["n_users"]),
        (items, "item_idx", "article_id", metadata["n_items"]),
    ]:
        if (
            frame.height != expected
            or frame[id_column].n_unique() != expected
            or not np.array_equal(
                frame[index].to_numpy(), np.arange(expected)
            )
        ):
            raise ValueError("Invalid tower ID mapping.")

    if histories.shape != (
        users.height,
        checkpoint["history_size"],
    ):
        raise ValueError("Unexpected history matrix shape.")

    if (
        histories.min() < 0
        or histories.max() > items.height
        or not (histories != 0).any(axis=1).all()
    ):
        raise ValueError("Invalid customer histories.")

    if checkpoint["config"]["n_items"] != items.height:
        raise ValueError("Tower item count does not match.")

    model = FeatureTwoTower(
        **checkpoint["config"],
        item_features=item_features,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if not np.array_equal(
        model.item_features.cpu().numpy(), item_features
    ):
        raise ValueError("Checkpoint metadata does not match.")

    model = model.to(device).eval()
    return model, users, items, histories


@torch.inference_mode()
def add_tower_scores(dataset, model, users, items, histories, device):
    dataset = dataset.sort(["customer_id", "article_id"])

    customers = (
        dataset.select("customer_id")
        .unique(maintain_order=True)["customer_id"]
        .to_list()
    )

    groups = (
        dataset.group_by("customer_id", maintain_order=True)
        .len()["len"]
        .to_numpy()
    )
    if not (groups == CANDIDATE_COUNT).all():
        raise ValueError("Expected 150 candidates per customer.")

    user_lookup = dict(
        users.select("customer_id", "user_idx").iter_rows()
    )
    item_lookup = dict(
        items.select("article_id", "item_idx").iter_rows()
    )

    item_positions = np.array(
        [
            item_lookup[item]
            for item in dataset["article_id"].to_list()
        ],
        dtype=np.int64,
    ).reshape(-1, CANDIDATE_COUNT)

    user_positions = np.array(
        [user_lookup.get(customer, -1) for customer in customers],
        dtype=np.int64,
    )

    item_ids = torch.arange(
        1, items.height + 1, dtype=torch.long, device=device
    )
    item_vectors = model.encode_items(item_ids)

    scores = np.full(
        (len(customers), CANDIDATE_COUNT),
        np.nan,
        dtype=np.float32,
    )

    known_rows = np.flatnonzero(user_positions >= 0)

    for start in range(0, len(known_rows), 256):
        rows = known_rows[start:start + 256]

        history_tensor = torch.as_tensor(
            histories[user_positions[rows]],
            dtype=torch.long,
            device=device,
        )
        candidate_indices = torch.as_tensor(
            item_positions[rows],
            dtype=torch.long,
            device=device,
        )

        user_vectors = model.encode_users(history_tensor)
        candidate_vectors = item_vectors[candidate_indices]

        batch_scores = (
            user_vectors.unsqueeze(1) * candidate_vectors
        ).sum(dim=-1)

        if not torch.isfinite(batch_scores).all().item():
            raise ValueError("Non-finite tower scores.")

        scores[rows] = batch_scores.cpu().numpy()

    return dataset.with_columns(
        pl.Series("tower_score", scores.reshape(-1)).fill_nan(None)
    )


def build_snapshot(as_of, transactions, articles, device):
    base = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{as_of}_v6.parquet"
    )
    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{as_of}_v6.parquet"
    ).select("customer_id", "actual_items")

    if base["as_of"].unique().to_list() != [as_of]:
        raise ValueError("Unexpected baseline snapshot date.")

    if set(base["customer_id"]) != set(actual["customer_id"]):
        raise ValueError("Baseline and actual customer sets differ.")

    ordered_base = (
        base.sort(
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

    recent_lookup = dict(
        base.filter(pl.col("repeat_rank").is_not_null())
        .sort(["customer_id", "repeat_rank", "article_id"])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id"))
        .select("customer_id", "article_id")
        .iter_rows()
    )

    model, users, items, histories = load_tower(as_of, device)

    evaluation = (
        actual
        .join(
            ordered_base,
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

    tower_lookup, retrieval_rows = retrieve(
        model, evaluation, items, histories, device
    )
    del retrieval_rows

    history = (
        transactions.filter(
            (pl.col("t_dat") >= as_of - timedelta(days=28))
            & (pl.col("t_dat") < as_of)
            & pl.col("article_id").is_not_null()
        )
        .select("customer_id", "article_id", "t_dat")
        .collect()
    )

    catalog = (
        history.group_by("article_id")
        .agg(
            pl.len().alias("item_popularity_28d"),
            (pl.col("t_dat") >= as_of - timedelta(days=7))
            .sum()
            .alias("item_popularity_7d"),
        )
        .join(articles, on="article_id", how="left", validate="1:1")
    )

    if catalog["product_type_no"].null_count():
        raise ValueError("Missing product types.")

    popular = (
        catalog.filter(pl.col("item_popularity_7d") > 0)
        .sort(
            ["item_popularity_7d", "article_id"],
            descending=[True, False],
        )
        ["article_id"].head(150).to_list()
    )
    popularity_ranks = {
        item: rank for rank, item in enumerate(popular, 1)
    }

    customer_ids = actual["customer_id"].to_list()

    preferences = (
        history.filter(pl.col("customer_id").is_in(customer_ids))
        .join(articles, on="article_id", how="left", validate="m:1")
        .group_by(["customer_id", "product_type_no"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (
                pl.col("count")
                / pl.col("count").sum().over("customer_id")
            ).alias("category_affinity")
        )
        .select("customer_id", "product_type_no", "category_affinity")
    )

    user_item = build_user_item_features(
        transactions, as_of, customer_ids
    )
    neighbors = build_neighbors(transactions, as_of=as_of)

    schema = {
        "customer_id": pl.String,
        "article_id": pl.String,
        "as_of": pl.Date,
        "label": pl.Int8,
        "candidate_order": pl.Int32,
        "tower_rank": pl.Int32,
        "popularity_rank": pl.Int32,
        "covisit_score": pl.Int32,
    }

    batches = []
    pending = []

    for row in evaluation.iter_rows(named=True):
        customer = row["customer_id"]
        original = row["base_items"]

        if len(original) != 150 or len(set(original)) != 150:
            raise ValueError("Invalid baseline candidate list.")

        tower = tower_lookup.get(customer, [])
        if row["user_idx"] is not None and len(tower) != 150:
            raise ValueError("Missing tower retrieval.")

        merged = list(
            dict.fromkeys(
                original[:150 - TOWER_QUOTA]
                + tower[:TOWER_QUOTA]
                + original
            )
        )[:150]

        recent = recent_lookup.get(customer, [])
        recent_set = set(recent)
        covisit_scores = Counter()

        for source in recent:
            for target, count in neighbors.get(source, []):
                if target not in recent_set:
                    covisit_scores[target] += count

        tower_ranks = {
            item: rank for rank, item in enumerate(tower, 1)
        }
        relevant = set(row["actual_items"])

        for position, item in enumerate(merged, 1):
            pending.append(
                (
                    customer,
                    item,
                    as_of,
                    int(item in relevant),
                    position,
                    tower_ranks.get(item),
                    popularity_ranks.get(item),
                    covisit_scores.get(item, 0),
                )
            )

        if len(pending) >= 15000:
            batches.append(
                pl.DataFrame(pending, schema=schema, orient="row")
            )
            pending = []

    if pending:
        batches.append(
            pl.DataFrame(pending, schema=schema, orient="row")
        )

    dataset = (
        pl.concat(batches)
        .join(
            base.select(
                "customer_id",
                "article_id",
                "repeat_rank",
                "variant_rank",
                "covisit_rank",
            ),
            on=["customer_id", "article_id"],
            how="left",
            validate="1:1",
        )
        .join(
            user_item,
            on=["customer_id", "article_id"],
            how="left",
            validate="1:1",
        )
        .join(
            catalog,
            on="article_id",
            how="left",
            validate="m:1",
        )
        .join(
            preferences,
            on=["customer_id", "product_type_no"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("user_item_purchase_count_28d").fill_null(0),
            pl.col("category_affinity").fill_null(0.0),
            pl.col("variant_rank")
            .is_not_null()
            .cast(pl.Int8)
            .alias("from_variant"),
        )
        .drop("product_type_no")
    )

    if dataset.height != actual.height * 150:
        raise ValueError("Unexpected merged candidate count.")

    if dataset["item_popularity_28d"].null_count():
        raise ValueError("Missing item popularity features.")

    # Existing candidates must retain their original feature values.
    keys = ["customer_id", "article_id"]
    common_old = (
        base.join(dataset.select(keys), on=keys, how="semi")
        .sort(keys)
    )
    common_new = (
        dataset.join(base.select(keys), on=keys, how="semi")
        .sort(keys)
    )

    for feature in BASE_FEATURES + ["label"]:
        old_values = common_old[feature].cast(pl.Float64).to_numpy()
        new_values = common_new[feature].cast(pl.Float64).to_numpy()

        if not np.allclose(old_values, new_values, equal_nan=True):
            raise ValueError(f"Existing feature changed: {feature}")

    dataset = add_tower_scores(
        dataset, model, users, items, histories, device
    )

    output = PROCESSED_DIR / f"candidates_{as_of}_v7.parquet"
    dataset.write_parquet(output)

    print(
        f"Saved {as_of}: {dataset.height:,} rows, "
        f"{dataset['label'].sum():,} positives",
        flush=True,
    )

    del model
    torch.cuda.empty_cache()
    return dataset


def predictions_from_scores(frame, scores):
    return (
        frame.select("customer_id", "article_id")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pipeline.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dates = [*TRAIN_DATES, VALIDATION_DATE]

    for as_of in dates:
        ensure_tower(as_of)

    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )
    articles = pl.read_csv(
        DATA_DIR / "articles.csv",
        schema_overrides={"article_id": pl.String},
    ).select("article_id", "product_type_no")

    if articles["article_id"].n_unique() != articles.height:
        raise ValueError("Duplicate article metadata.")

    for as_of in dates:
        output = PROCESSED_DIR / f"candidates_{as_of}_v7.parquet"

        if output.exists():
            print(f"Using saved V7 candidates: {as_of}", flush=True)
            continue

        print(f"\nBuilding integrated snapshot: {as_of}", flush=True)
        dataset = build_snapshot(as_of, transactions, articles, device)
        del dataset
        gc.collect()

    frames = []

    for as_of in TRAIN_DATES:
        frame = pl.read_parquet(
            PROCESSED_DIR / f"candidates_{as_of}_v7.parquet"
        )
        if frame["as_of"].unique().to_list() != [as_of]:
            raise ValueError("Unexpected training snapshot date.")

        frames.append(
            frame.select("as_of", "customer_id", "article_id", "label", *FEATURES)
        )

    train = pl.concat(frames).sort(
        ["as_of", "customer_id", "article_id"]
    )
    del frames, frame

    groups = (
        train.group_by(["as_of", "customer_id"], maintain_order=True)
        .len()["len"].to_numpy()
    )

    if groups.sum() != train.height or not (groups == 150).all():
        raise ValueError("Invalid ranking groups.")

    x_train = train.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()
    y_train = train["label"].to_numpy()

    print(f"\nTraining ranker: {train.height:,} rows", flush=True)

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )
    model.fit(
        x_train,
        y_train,
        group=groups,
        feature_name=FEATURES,
    )
    model.booster_.save_model(
        str(PROCESSED_DIR / "ranker_v7.txt")
    )

    del train, x_train, y_train
    gc.collect()

    original = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v6.parquet"
    )
    merged = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v7.parquet"
    )
    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{VALIDATION_DATE}_v6.parquet"
    ).select("customer_id", "actual_items")

    if merged["as_of"].unique().to_list() != [VALIDATION_DATE]:
        raise ValueError("Unexpected validation date.")

    old_model = Booster(
        model_file=str(PROCESSED_DIR / "ranker_v6_tuned.txt")
    )

    records = []

    rule_metrics = evaluate_predictions(
        actual, get_predictions(original)
    )
    records.append({"method": "Variant rule V6", **rule_metrics})

    for name, frame, booster in [
        ("Tuned V6 / original candidates", original, old_model),
        ("Tuned V6 / merged candidates", merged, old_model),
        ("V7 / merged candidates + tower features", merged, model.booster_),
    ]:
        matrix = frame.select(
            pl.col(booster.feature_name()).cast(pl.Float32)
        ).to_numpy()

        predictions = predictions_from_scores(
            frame, booster.predict(matrix)
        )
        metrics = evaluate_predictions(actual, predictions)
        records.append({"method": name, **metrics})

    report = pl.DataFrame(records).sort(
        "map_at_12", descending=True
    )
    report.write_csv(REPORT_DIR / "integration_v7.csv")

    importance = pl.DataFrame(
        {
            "feature": FEATURES,
            "gain": model.feature_importances_,
        }
    ).sort("gain", descending=True)
    importance.write_csv(REPORT_DIR / "integration_v7_importance.csv")

    config = {
        "tower_epoch": TOWER_EPOCH,
        "tower_quota": TOWER_QUOTA,
        "candidate_count": CANDIDATE_COUNT,
        "features": FEATURES,
        "train_dates": [str(day) for day in TRAIN_DATES],
        "validation_date": str(VALIDATION_DATE),
        "ranker_params": {
            "n_estimators": 40,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "reg_lambda": 1.0,
        },
    }
    (REPORT_DIR / "integration_v7_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    with pl.Config(
        tbl_rows=20,
        tbl_width_chars=160,
        float_precision=6,
    ):
        print("\nFinal ranking comparison:")
        print(report)
        print("\nFeature importance:")
        print(importance)

    print("\nSaved ranker_v7.txt and integration_v7 reports.")


if __name__ == "__main__":
    main()