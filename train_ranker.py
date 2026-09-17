import polars as pl
from lightgbm import LGBMRanker

from popularity_baseline import (
    DATA_DIR,
    VALIDATION_START,
    average_precision_at_k,
    build_popularity,
)


FEATURES = [
    "repeat_rank",
    "covisit_rank",
    "popularity_rank",
    "covisit_score",
    "user_item_purchase_count_28d",
    "user_item_recency_days",
]

PROCESSED_DIR = DATA_DIR.parent.parent / "processed"


def evaluate_predictions(
    actual: pl.DataFrame,
    predictions: pl.DataFrame,
) -> dict[str, float]:
    prediction_lookup = dict(predictions.iter_rows())

    recall_sum = 0.0
    hit_customers = 0
    ap_sum = 0.0

    if actual.is_empty():
        raise ValueError("No customers to evaluate.")

    for customer_id, actual_items in actual.iter_rows():
        predicted = prediction_lookup.get(customer_id, [])

        if len(predicted) != 12 or len(set(predicted)) != 12:
            raise ValueError("Expected 12 unique recommendations.")

        relevant = set(actual_items)
        hits = len(relevant.intersection(predicted))

        recall_sum += hits / len(relevant)
        hit_customers += int(hits > 0)
        ap_sum += average_precision_at_k(actual_items, predicted)

    return {
        "recall_at_12": recall_sum / actual.height,
        "hit_rate_at_12": hit_customers / actual.height,
        "map_at_12": ap_sum / actual.height,
    }

def evaluate_baselines(
    validation: pl.DataFrame,
    actual: pl.DataFrame,
) -> None:
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    popular_items = build_popularity(
        transactions,
        as_of=VALIDATION_START,
        k=12,
    )["article_id"].to_list()

    recent_by_customer = (
        validation
        .filter(pl.col("repeat_rank").is_not_null())
        .sort(["customer_id", "repeat_rank", "article_id"])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").alias("recent_items"))
        .select("customer_id", "recent_items")
    )

    recent_lookup = dict(recent_by_customer.iter_rows())

    popularity_rows = []
    repeat_rows = []

    for customer_id in actual["customer_id"].to_list():
        popularity_rows.append(
            {
                "customer_id": customer_id,
                "predictions": popular_items,
            }
        )

        recent_items = recent_lookup.get(customer_id, [])

        repeat_predictions = list(
            dict.fromkeys(recent_items + popular_items)
        )[:12]

        repeat_rows.append(
            {
                "customer_id": customer_id,
                "predictions": repeat_predictions,
            }
        )

    candidate_order = (
        validation
        .sort(
            [
                "customer_id",
                "repeat_rank",
                "covisit_rank",
                "popularity_rank",
                "article_id",
            ],
            nulls_last=True,
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
        .select("customer_id", "predictions")
    )

    baseline_predictions = {
        "Popularity": pl.DataFrame(popularity_rows),
        "Repeat + popularity": pl.DataFrame(repeat_rows),
        "Candidate order": candidate_order,
    }

    for name, predictions in baseline_predictions.items():
        metrics = evaluate_predictions(actual, predictions)

        print(f"\n{name}")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")


def evaluate_training(
    model: LGBMRanker,
    train: pl.DataFrame,
) -> None:
    actual = pl.read_parquet(
        PROCESSED_DIR / "train_actual_v4.parquet"
    ).select("customer_id", "actual_items")

    features = train.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()

    scores = model.booster_.predict(features)

    model_predictions = (
        train
        .select("customer_id", "article_id")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
        .select("customer_id", "predictions")
    )

    rule_predictions = (
        train
        .sort(
            [
                "customer_id",
                "repeat_rank",
                "covisit_rank",
                "popularity_rank",
                "article_id",
            ],
            nulls_last=True,
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
        .select("customer_id", "predictions")
    )

    methods = {
        "Candidate order": rule_predictions,
        "LightGBM v3": model_predictions,
    }

    print(f"\nTraining evaluation: {actual.height} customers")

    for name, predictions in methods.items():
        metrics = evaluate_predictions(actual, predictions)

        print(f"\nTraining - {name}")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")

def main():
    train = pl.read_parquet(
        PROCESSED_DIR / "train_candidates_v4.parquet"
    ).sort(["customer_id", "article_id"])

    validation = pl.read_parquet(
        PROCESSED_DIR / "validation_candidates_v4.parquet"
    )

    actual = pl.read_parquet(
        PROCESSED_DIR / "validation_actual_v4.parquet"
    ).select("customer_id", "actual_items")

    train_groups = (
        train.group_by("customer_id", maintain_order=True)
        .len()["len"]
        .to_numpy()
    )

    if train_groups.sum() != train.height:
        raise ValueError("Group sizes do not match the row count.")

    x_train = train.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()

    y_train = train["label"].to_numpy()

    x_validation = validation.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )

    print("Training ranker...")

    model.fit(
        x_train,
        y_train,
        group=train_groups,
        feature_name=FEATURES,
    )

    scores = model.booster_.predict(x_validation)

    predictions = (
        validation
        .select("customer_id", "article_id")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
        .select("customer_id", "predictions")
    )

    metrics = evaluate_predictions(actual, predictions)

    print(f"\nEvaluated customers: {actual.height}")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nFeature importance:")

    for name, importance in zip(FEATURES, model.feature_importances_):
        print(f"{name}: {importance:.2f}")

    model.booster_.save_model(
         str(PROCESSED_DIR / "ranker_v4.txt")
    )

    print("\nModel saved.")
    evaluate_baselines(validation, actual)
    evaluate_training(model, train)


if __name__ == "__main__":
    main()