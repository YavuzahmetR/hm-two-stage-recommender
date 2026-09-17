from datetime import timedelta
from pathlib import Path

import polars as pl

from covisitation import build_neighbors
from popularity_baseline import (
    DATA_DIR,
    VALIDATION_START,
    average_precision_at_k,
    build_popularity,
)
from ranking_dataset import build_candidate_rows


def main():
    processed_dir = DATA_DIR.parent.parent / "processed"

    actual = pl.read_parquet(
        processed_dir / "validation_actual_v4.parquet"
    ).select("customer_id", "actual_items")

    customer_ids = actual["customer_id"].to_list()

    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    purchases = (
        transactions
        .filter(
            (pl.col("t_dat") >= VALIDATION_START - timedelta(days=90))
            & (pl.col("t_dat") < VALIDATION_START)
            & pl.col("customer_id").is_in(customer_ids)
            & pl.col("article_id").is_not_null()
        )
        .group_by(["customer_id", "article_id"])
        .agg(pl.col("t_dat").max().alias("last_purchase"))
        .collect()
    )

    lookups = {}

    for days in [28, 90]:
        recent = (
            purchases
            .filter(
                pl.col("last_purchase")
                >= VALIDATION_START - timedelta(days=days)
            )
            .sort(
                ["customer_id", "last_purchase", "article_id"],
                descending=[False, True, False],
            )
            .group_by("customer_id")
            .agg(pl.col("article_id").head(12).alias("recent_items"))
            .select("customer_id", "recent_items")
        )

        lookups[days] = dict(recent.iter_rows())

    recovered_customers = len(
        set(lookups[90]) - set(lookups[28])
    )

    print(f"Customers with 28-day history: {len(lookups[28])}")
    print(f"Customers with 90-day history: {len(lookups[90])}")
    print(f"Customers gaining history: {recovered_customers}")

    popular_pool = build_popularity(
        transactions,
        as_of=VALIDATION_START,
        k=150,
    )["article_id"].to_list()

    neighbors = build_neighbors(
        transactions,
        as_of=VALIDATION_START,
    )

    records = []

    for days in [28, 90]:
        print(f"\nEvaluating {days}-day history...")

        for customer_id, actual_items in actual.iter_rows():
            relevant = set(actual_items)

            if not relevant:
                raise ValueError("Empty actual purchase list.")

            rows = build_candidate_rows(
                customer_id=customer_id,
                recent_items=lookups[days].get(customer_id, []),
                popular_pool=popular_pool,
                neighbors=neighbors,
                k=150,
            )

            candidates = [row["article_id"] for row in rows]

            if len(candidates) != 150 or len(set(candidates)) != 150:
                raise ValueError("Expected 150 unique candidates.")

            predicted = candidates[:12]

            candidate_hits = len(relevant.intersection(candidates))
            prediction_hits = len(relevant.intersection(predicted))

            segment = (
                "history_28d"
                if customer_id in lookups[28]
                else "no_history_28d"
            )

            metrics = {
                "history_days": days,
                "candidate_recall_150": candidate_hits / len(relevant),
                "candidate_hit_rate_150": float(candidate_hits > 0),
                "recall_12": prediction_hits / len(relevant),
                "hit_rate_12": float(prediction_hits > 0),
                "map_12": average_precision_at_k(
                    actual_items,
                    predicted,
                ),
            }

            for population in ["all", segment]:
                records.append(
                    {"segment": population, **metrics}
                )

    metric_columns = [
        "candidate_recall_150",
        "candidate_hit_rate_150",
        "recall_12",
        "hit_rate_12",
        "map_12",
    ]

    report = (
        pl.DataFrame(records)
        .group_by(["segment", "history_days"])
        .agg(
            pl.len().alias("customers"),
            *[pl.col(name).mean() for name in metric_columns],
        )
        .sort(["segment", "history_days"])
    )

    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "history_comparison.csv"
    report.write_csv(output_path)

    with pl.Config(
        tbl_rows=10,
        tbl_cols=8,
        tbl_width_chars=180,
        float_precision=6,
    ):
        print(report)

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()