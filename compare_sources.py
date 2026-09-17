from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import polars as pl

from popularity_baseline import (
    DATA_DIR,
    VALIDATION_START,
    average_precision_at_k,
)


def main():
    processed = DATA_DIR.parent.parent / "processed"
    start_7d = VALIDATION_START - timedelta(days=7)
    start_28d = VALIDATION_START - timedelta(days=28)

    actual = pl.read_parquet(
        processed / "validation_actual_v4.parquet"
    ).select("customer_id", "actual_items")

    baseline = pl.read_parquet(
        processed / "validation_candidates_v4.parquet"
    )

    actual_lookup = dict(actual.iter_rows())
    customer_ids = actual["customer_id"].to_list()

    if not customer_ids:
        raise ValueError("No validation customers.")

    if set(baseline["customer_id"]) != set(customer_ids):
        raise ValueError("Customer sets do not match.")

    articles = (
        pl.read_csv(
            DATA_DIR / "articles.csv",
            schema_overrides={
                "article_id": pl.String,
                "product_code": pl.String,
            },
        )
        .select("article_id", "product_code", "product_type_no")
    )

    if articles["article_id"].n_unique() != articles.height:
        raise ValueError("Article metadata contains duplicate IDs.")

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
            (pl.col("t_dat") >= start_28d)
            & (pl.col("t_dat") < VALIDATION_START)
            & pl.col("article_id").is_not_null()
        )
        .select("customer_id", "article_id", "t_dat")
        .collect()
    )

    item_stats = history.group_by("article_id").agg(
        pl.len().alias("item_popularity_28d"),
        (pl.col("t_dat") >= start_7d)
        .sum()
        .alias("item_popularity_7d"),
    )

    catalog = item_stats.join(
        articles,
        on="article_id",
        how="left",
        validate="1:1",
    )

    missing_metadata = catalog.filter(
        pl.col("product_code").is_null()
        | pl.col("product_type_no").is_null()
    ).height

    if missing_metadata:
        raise ValueError(
            f"Missing metadata for {missing_metadata} observed products."
        )

    preferences = (
        history
        .filter(pl.col("customer_id").is_in(customer_ids))
        .join(
            articles.select("article_id", "product_type_no"),
            on="article_id",
            how="left",
            validate="m:1",
        )
        .group_by(["customer_id", "product_type_no"])
        .agg(pl.len().alias("type_purchase_count"))
        .with_columns(
            (
                pl.col("type_purchase_count")
                / pl.col("type_purchase_count").sum().over("customer_id")
            ).alias("category_affinity")
        )
    )

    top_types = (
        preferences
        .sort(
            ["customer_id", "type_purchase_count", "product_type_no"],
            descending=[False, True, False],
        )
        .group_by("customer_id")
        .agg(pl.col("product_type_no").head(2).alias("types"))
        .select("customer_id", "types")
    )

    type_lookup = dict(top_types.iter_rows())
    code_lookup = dict(
        articles.select("article_id", "product_code").iter_rows()
    )

    weekly_items = (
        catalog
        .filter(pl.col("item_popularity_7d") > 0)
        .sort(
            ["item_popularity_7d", "article_id"],
            descending=[True, False],
        )
    )

    weekly_counts = dict(
        weekly_items.select(
            "article_id", "item_popularity_7d"
        ).iter_rows()
    )

    category_pools = defaultdict(list)
    variant_pools = defaultdict(list)

    for item, product_code, product_type in weekly_items.select(
        "article_id", "product_code", "product_type_no"
    ).iter_rows():
        category_pools[product_type].append(item)
        variant_pools[product_code].append(item)

    grouped = (
        baseline
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
        .agg(pl.col("article_id"), pl.col("repeat_rank"))
        .select("customer_id", "article_id", "repeat_rank")
    )

    def select_source(pool, excluded):
        available = set(pool) - excluded
        return sorted(
            available,
            key=lambda item: (-weekly_counts[item], item),
        )[:20]

    records = []

    for customer_id, base_items, repeat_ranks in grouped.iter_rows():
        recent = [
            item
            for item, rank in zip(base_items, repeat_ranks)
            if rank is not None
        ]
        recent_set = set(recent)
        base_set = set(base_items)
        relevant = set(actual_lookup[customer_id])

        if not relevant:
            raise ValueError("Empty actual purchase list.")

        category_pool = [
            item
            for product_type in type_lookup.get(customer_id, [])
            for item in category_pools[product_type]
        ]

        product_codes = {
            code_lookup[item]
            for item in recent
            if item in code_lookup
        }

        variant_pool = [
            item
            for product_code in product_codes
            for item in variant_pools[product_code]
        ]

        source_a = select_source(category_pool, recent_set)
        source_b = select_source(variant_pool, recent_set)

        experiments = {
            "Reference": [],
            "A_category": source_a,
            "B_variants": source_b,
            "A+B": source_a + source_b,
        }

        segment = "history_28d" if recent else "no_history_28d"

        for name, additions in experiments.items():
            candidates = list(
                dict.fromkeys(recent + additions + base_items)
            )[:150]

            if len(candidates) != 150:
                raise ValueError("Expected 150 unique candidates.")

            predicted = candidates[:12]
            candidate_hits = len(relevant.intersection(candidates))
            hits = len(relevant.intersection(predicted))

            metrics = {
                "experiment": name,
                "candidate_recall_150": candidate_hits / len(relevant),
                "candidate_hit_rate_150": float(candidate_hits > 0),
                "recall_12": hits / len(relevant),
                "hit_rate_12": float(hits > 0),
                "map_12": average_precision_at_k(
                    list(relevant), predicted
                ),
                "new_candidates": len(set(candidates) - base_set),
            }

            for population in ["all", segment]:
                records.append({"segment": population, **metrics})

    metric_columns = [
        "candidate_recall_150",
        "candidate_hit_rate_150",
        "recall_12",
        "hit_rate_12",
        "map_12",
        "new_candidates",
    ]

    report = (
        pl.DataFrame(records)
        .group_by(["segment", "experiment"])
        .agg(
            pl.len().alias("customers"),
            *[pl.col(name).mean() for name in metric_columns],
        )
        .sort(["segment", "experiment"])
    )

    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    report.write_csv(output_dir / "source_comparison.csv")

    catalog.write_parquet(
        processed / "validation_item_stats.parquet"
    )
    preferences.write_parquet(
        processed / "validation_category_preferences.parquet"
    )

    with pl.Config(
        tbl_rows=20,
        tbl_width_chars=150,
        float_precision=6,
    ):
        print(
            report.filter(pl.col("segment") == "all").select(
                "experiment",
                "candidate_recall_150",
                "candidate_hit_rate_150",
                "map_12",
                "new_candidates",
            )
        )

    print("\nSaved reports/metrics/source_comparison.csv")


if __name__ == "__main__":
    main()