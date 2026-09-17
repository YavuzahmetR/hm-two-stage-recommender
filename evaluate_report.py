from collections import defaultdict
from pathlib import Path

import polars as pl
from lightgbm import Booster

from popularity_baseline import (
    DATA_DIR,
    VALIDATION_START,
    average_precision_at_k,
    build_popularity,
)


VERSION = "v4"
PROCESSED_DIR = DATA_DIR.parent.parent / "processed"


def main():
    candidates = pl.read_parquet(
        PROCESSED_DIR / f"validation_candidates_{VERSION}.parquet"
    )

    actual = pl.read_parquet(
        PROCESSED_DIR / f"validation_actual_{VERSION}.parquet"
    ).select("customer_id", "actual_items")

    if actual.is_empty():
        raise ValueError("No customers to evaluate.")

    if actual["customer_id"].n_unique() != actual.height:
        raise ValueError("Duplicate customers in actual data.")

    if set(candidates["customer_id"]) != set(actual["customer_id"]):
        raise ValueError("Candidate and actual customer sets differ.")

    if candidates["as_of"].null_count() > 0:
        raise ValueError("Missing snapshot dates.")

    if candidates["as_of"].unique().to_list() != [VALIDATION_START]:
        raise ValueError("Unexpected validation snapshot date.")

    actual_lookup = dict(actual.iter_rows())

    model = Booster(
        model_file=str(PROCESSED_DIR / f"ranker_{VERSION}.txt")
    )

    features = candidates.select(
        pl.col(model.feature_name()).cast(pl.Float32)
    ).to_numpy()

    candidates = candidates.with_columns(
        pl.Series("model_score", model.predict(features))
    )

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

    grouped = (
        candidates
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
        .agg(
            pl.col("article_id"),
            pl.col("repeat_rank"),
            pl.col("model_score"),
        )
        .select(
            "customer_id",
            "article_id",
            "repeat_rank",
            "model_score",
        )
    )

    totals = defaultdict(lambda: [0, 0.0, 0.0, 0.0])

    for customer_id, items, repeat_ranks, scores in grouped.iter_rows():
        relevant = set(actual_lookup[customer_id])

        if not relevant:
            raise ValueError("Empty actual purchase list.")

        if len(items) != 150 or len(set(items)) != 150:
            raise ValueError("Expected 150 unique candidates.")

        recent_items = [
            item
            for item, rank in zip(items, repeat_ranks)
            if rank is not None
        ]

        segment = (
            "history_28d" if recent_items else "no_history_28d"
        )

        model_items = [
            item
            for item, score in sorted(
                zip(items, scores),
                key=lambda pair: (-pair[1], pair[0]),
            )[:12]
        ]

        predictions = {
            "Popularity": popular_items,
            "Repeat+pop": list(
                dict.fromkeys(recent_items + popular_items)
            )[:12],
            "Candidate order": items[:12],
            f"LightGBM {VERSION}": model_items,
        }

        measurements = {}

        for name, predicted in predictions.items():
            if len(predicted) != 12 or len(set(predicted)) != 12:
                raise ValueError("Expected 12 unique predictions.")

            hits = len(relevant.intersection(predicted))

            measurements[(name, 12)] = (
                hits / len(relevant),
                float(hits > 0),
                average_precision_at_k(list(relevant), predicted),
            )

        candidate_hits = len(relevant.intersection(items))
        best_hits = min(candidate_hits, 12)

        measurements[("Candidates", 150)] = (
            candidate_hits / len(relevant),
            float(candidate_hits > 0),
            0.0,
        )

        measurements[("Oracle bound", 12)] = (
            best_hits / len(relevant),
            float(best_hits > 0),
            best_hits / min(len(relevant), 12),
        )

        for population in ["all", segment]:
            for (method, cutoff), values in measurements.items():
                total = totals[(population, method, cutoff)]
                total[0] += 1

                for index, value in enumerate(values, start=1):
                    total[index] += value

    rows = []

    for (segment, method, cutoff), total in sorted(totals.items()):
        count, recall_sum, hit_sum, ap_sum = total

        rows.append(
            {
                "segment": segment,
                "method": method,
                "k": cutoff,
                "customers": count,
                "recall": recall_sum / count,
                "hit_rate": hit_sum / count,
                "map": (
                    None if method == "Candidates"
                    else ap_sum / count
                ),
            }
        )

    report = pl.DataFrame(rows)

    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"validation_{VERSION}.csv"
    report.write_csv(output_path)

    with pl.Config(
        tbl_rows=30,
        tbl_cols=7,
        tbl_width_chars=150,
        float_precision=6,
    ):
        print(report)

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()