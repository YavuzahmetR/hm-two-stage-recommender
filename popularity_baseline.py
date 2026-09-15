from datetime import date, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/hm")
VALIDATION_START = date(2020, 9, 9)


def build_popularity(
    transactions: pl.LazyFrame,
    as_of: date,
    lookback_days: int = 7,
    k: int = 12,
) -> pl.DataFrame:
    window_start = as_of - timedelta(days=lookback_days)

    return (
        transactions.filter(
            (pl.col("t_dat") >= window_start)
            & (pl.col("t_dat") < as_of)
            & pl.col("article_id").is_not_null()
        )
        .group_by("article_id")
        .agg(pl.len().alias("purchase_count"))
        .sort(["purchase_count", "article_id"], descending=[True, False])
        .head(k)
        .collect()
    )


def evaulate_recall(
    transactions: pl.LazyFrame,
    recommandations: list[str],
    target_start: date,
) -> pl.DataFrame:
    target_end = target_start + timedelta(days=7)

    customer_scores = (
        transactions.filter(
            (pl.col("t_dat") >= target_start)
            & (pl.col("t_dat") < target_end)
            & pl.col("customer_id").is_not_null()
            & pl.col("article_id").is_not_null()
        )
        .select("customer_id", "article_id")
        .unique()
        .with_columns(
            pl.col("article_id").is_in(recommandations).alias("is_hit")
        )
        .group_by("customer_id")
        .agg(
            pl.len().alias("actual_count"),
            pl.col("is_hit").sum().alias("hit_count"),
        )
        .with_columns(
            (pl.col("hit_count") / pl.col("actual_count")).alias("recall")
        )
    )

    return customer_scores.select(
        pl.len().alias("evaulated_customers"),
        pl.col("recall").mean().alias("recall_at_12"),
        (pl.col("hit_count") > 0).mean().alias("hit_rate_at_12"),
    ).collect()


def average_presicion_at_k(
        actual: list[str],
        predicted: list[str],
        k: int = 12
) -> float:
    if k<= 0:
        raise ValueError("k must be positive")

    actual_items = set(actual)

    if not actual_items:
        raise ValueError("Actual items must not be empty.")

    seen = set()
    hits = 0
    presicion_sum = 0.0

    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in actual_items and article_id not in seen:
            hits += 1
            presicion_sum += hits / rank
        seen.add(article_id)

    return presicion_sum / min(len(actual_items), k)


def evaluate_map(
        transactions: pl.LazyFrame,
        recommendations: list[str],
        target_start: date
) -> float:

    target_end = target_start + timedelta(days=7)

    actual_by_customer = (
        transactions.filter(
            (pl.col("t_dat") >= target_start)
            & (pl.col("t_dat") < target_end)
            & pl.col("customer_id").is_not_null()
            & pl.col("article_id").is_not_null()
        )
        .group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("actual_items"))
        .collect()
    )

    if actual_by_customer.is_empty():
        raise ValueError("No customers found in the target period.")

    scores = [
        average_presicion_at_k(actual, recommendations)
        for actual in actual_by_customer["actual_items"].to_list()
    ]

    return sum(scores) / len(scores)

def main():
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
    )

    if popular_items.height != 12:
        raise ValueError("Expected 12 unique recommendations.")

    print("Top 12 popular products:")
    with pl.Config(tbl_rows=12):
        print(popular_items)

    recommendations = popular_items["article_id"].to_list()
    print("\nRecommendation list:")
    print(recommendations)

    metrics = evaulate_recall(
        transactions,
        recommendations,
        target_start=VALIDATION_START,
    )

    print("\nValidation metrics:")
    print(metrics)

    map_score = evaluate_map(
        transactions,
        recommendations,
        target_start=VALIDATION_START
    )
    print(f"\nValidation MAP@12: {map_score:.6f}")

if __name__ == "__main__":
    main()