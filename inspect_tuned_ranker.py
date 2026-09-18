import polars as pl
from lightgbm import Booster

from popularity_baseline import average_precision_at_k
from train_ranker import evaluate_predictions
from train_v6 import (
    DATA_DIR,
    PROCESSED_DIR,
    VALIDATION_DATE,
    get_predictions,
)


REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"


def main():
    candidates = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v6.parquet"
    ).sort(["customer_id", "article_id"])

    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{VALIDATION_DATE}_v6.parquet"
    ).select("customer_id", "actual_items")

    model = Booster(
        model_file=str(PROCESSED_DIR / "ranker_v6_tuned.txt")
    )

    features = candidates.select(
        pl.col(model.feature_name()).cast(pl.Float32)
    ).to_numpy()

    model_predictions = (
        candidates
        .select("customer_id", "article_id")
        .with_columns(pl.Series("score", model.predict(features)))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
    )

    rule_predictions = get_predictions(candidates)

    history = (
        candidates
        .group_by("customer_id")
        .agg(
            pl.col("repeat_rank")
            .is_not_null()
            .any()
            .alias("has_history")
        )
    )

    customer_data = (
        actual
        .join(
            history,
            on="customer_id",
            how="left",
            validate="1:1",
        )
        .join(
            rule_predictions.rename({"predictions": "rule_items"}),
            on="customer_id",
            how="left",
            validate="1:1",
        )
        .join(
            model_predictions.rename({"predictions": "model_items"}),
            on="customer_id",
            how="left",
            validate="1:1",
        )
        .sort("customer_id")
    )

    for column in ["has_history", "rule_items", "model_items"]:
        if customer_data[column].null_count():
            raise ValueError(f"Missing values in {column}.")

    segments = [
        ("all", customer_data),
        (
            "history_28d",
            customer_data.filter(pl.col("has_history")),
        ),
        (
            "no_history_28d",
            customer_data.filter(~pl.col("has_history")),
        ),
    ]

    records = []

    for segment, frame in segments:
        if frame.is_empty():
            continue

        segment_actual = frame.select("customer_id", "actual_items")

        for method, column in [
            ("Variant rule", "rule_items"),
            ("LightGBM tuned", "model_items"),
        ]:
            predictions = frame.select(
                "customer_id",
                pl.col(column).alias("predictions"),
            )
            metrics = evaluate_predictions(segment_actual, predictions)

            records.append(
                {
                    "segment": segment,
                    "method": method,
                    "customers": frame.height,
                    **metrics,
                }
            )

    report = pl.DataFrame(records)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_csv(REPORT_DIR / "tuned_segments.csv")

    with pl.Config(
        tbl_rows=20,
        tbl_width_chars=140,
        float_precision=6,
    ):
        print("\nSegment comparison:")
        print(report)

    comparisons = []

    for row in customer_data.iter_rows(named=True):
        rule_ap = average_precision_at_k(
            row["actual_items"],
            row["rule_items"],
            k=12,
        )
        model_ap = average_precision_at_k(
            row["actual_items"],
            row["model_items"],
            k=12,
        )

        comparisons.append(
            {
                **row,
                "rule_ap": rule_ap,
                "model_ap": model_ap,
                "difference": model_ap - rule_ap,
            }
        )

    tolerance = 1e-12

    improved = [
        row for row in comparisons
        if row["difference"] > tolerance
    ]
    worsened = [
        row for row in comparisons
        if row["difference"] < -tolerance
    ]
    tied = len(comparisons) - len(improved) - len(worsened)

    print("\nCustomer-level AP@12 comparison:")
    print(f"Improved: {len(improved)}")
    print(f"Worsened: {len(worsened)}")
    print(f"Tied: {tied}")

    examples = [
        (
            "Improved with recent history",
            next(
                (row for row in improved if row["has_history"]),
                None,
            ),
        ),
        (
            "Worsened",
            next(iter(worsened), None),
        ),
        (
            "Changed ranking without recent history",
            next(
                (
                    row for row in comparisons
                    if not row["has_history"]
                    and row["rule_items"] != row["model_items"]
                ),
                None,
            ),
        ),
    ]

    for title, row in examples:
        print(f"\nExample: {title}")

        if row is None:
            print("No matching example.")
            continue

        print(f"Customer: {row['customer_id'][:12]}...")
        print(f"Recent history: {row['has_history']}")
        print(f"Actual products: {sorted(set(row['actual_items']))}")
        print(f"Rule AP@12: {row['rule_ap']:.6f}")
        print(f"Model AP@12: {row['model_ap']:.6f}")

        actual_items = set(row["actual_items"])

        detail = pl.DataFrame(
            {
                "position": list(range(1, 13)),
                "rule_item": row["rule_items"],
                "rule_hit": [
                    item in actual_items for item in row["rule_items"]
                ],
                "model_item": row["model_items"],
                "model_hit": [
                    item in actual_items for item in row["model_items"]
                ],
            }
        )

        with pl.Config(tbl_rows=12, tbl_width_chars=120):
            print(detail)

    print("\nSaved tuned_segments.csv")


if __name__ == "__main__":
    main()