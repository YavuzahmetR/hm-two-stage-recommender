from collections import Counter, defaultdict
from datetime import date, timedelta

import polars as pl

from covisitation import build_neighbors, get_covisitation_candidates
from popularity_baseline import DATA_DIR
from ranking_dataset import build_user_item_features


PROCESSED_DIR = DATA_DIR.parent.parent / "processed"


def build_snapshot(transactions, articles, as_of, base, actual):
    customer_ids = actual["customer_id"].to_list()
    start_7d = as_of - timedelta(days=7)

    history = (
        transactions
        .filter(
            (pl.col("t_dat") >= as_of - timedelta(days=28))
            & (pl.col("t_dat") < as_of)
            & pl.col("article_id").is_not_null()
        )
        .select("customer_id", "article_id", "t_dat")
        .collect()
    )

    catalog = (
        history
        .group_by("article_id")
        .agg(
            pl.len().alias("item_popularity_28d"),
            (pl.col("t_dat") >= start_7d)
            .sum()
            .alias("item_popularity_7d"),
        )
        .join(articles, on="article_id", how="left", validate="1:1")
    )

    if catalog.filter(
        pl.col("product_code").is_null()
        | pl.col("product_type_no").is_null()
    ).height:
        raise ValueError("Missing product metadata.")

    weekly = (
        catalog
        .filter(pl.col("item_popularity_7d") > 0)
        .sort(
            ["item_popularity_7d", "article_id"],
            descending=[True, False],
        )
    )

    popular = weekly["article_id"].head(150).to_list()
    popularity_ranks = {
        item: rank for rank, item in enumerate(popular, start=1)
    }

    counts = dict(
        weekly.select("article_id", "item_popularity_7d").iter_rows()
    )
    codes = dict(
        articles.select("article_id", "product_code").iter_rows()
    )

    variant_pools = defaultdict(list)

    for item, code in weekly.select(
        "article_id", "product_code"
    ).iter_rows():
        variant_pools[code].append(item)

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
        .agg(pl.len().alias("count"))
        .with_columns(
            (
                pl.col("count")
                / pl.col("count").sum().over("customer_id")
            ).alias("category_affinity")
        )
        .select("customer_id", "product_type_no", "category_affinity")
    )

    recent_lookup = dict(
        base
        .filter(pl.col("repeat_rank").is_not_null())
        .sort(["customer_id", "repeat_rank", "article_id"])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id"))
        .select("customer_id", "article_id")
        .iter_rows()
    )

    neighbors = build_neighbors(transactions, as_of=as_of)

    schema = {
        "customer_id": pl.String,
        "article_id": pl.String,
        "as_of": pl.Date,
        "label": pl.Int8,
        "repeat_rank": pl.Int32,
        "variant_rank": pl.Int32,
        "covisit_rank": pl.Int32,
        "popularity_rank": pl.Int32,
        "covisit_score": pl.Int32,
    }

    batches = []
    pending = []

    for customer_id, purchased in actual.select(
        "customer_id", "actual_items"
    ).iter_rows():
        recent = recent_lookup.get(customer_id, [])
        recent_set = set(recent)
        relevant = set(purchased)

        variant_pool = {
            item
            for source in recent
            for item in variant_pools[codes[source]]
            if item not in recent_set
        }

        variants = sorted(
            variant_pool,
            key=lambda item: (-counts[item], item),
        )[:20]

        related = get_covisitation_candidates(recent, neighbors, k=50)

        scores = Counter()

        for source in recent:
            for target, count in neighbors.get(source, []):
                if target not in recent_set:
                    scores[target] += count

        ranks = {
            "repeat_rank": {
                item: rank for rank, item in enumerate(recent, 1)
            },
            "variant_rank": {
                item: rank for rank, item in enumerate(variants, 1)
            },
            "covisit_rank": {
                item: rank for rank, item in enumerate(related, 1)
            },
            "popularity_rank": popularity_ranks,
        }

        candidates = list(
            dict.fromkeys(recent + variants + related + popular)
        )[:150]

        if len(candidates) != 150:
            raise ValueError("Expected 150 unique candidates.")

        for item in candidates:
            pending.append(
                {
                    "customer_id": customer_id,
                    "article_id": item,
                    "as_of": as_of,
                    "label": int(item in relevant),
                    **{
                        name: mapping.get(item)
                        for name, mapping in ranks.items()
                    },
                    "covisit_score": scores.get(item, 0),
                }
            )

        if len(pending) >= 15000:
            batches.append(pl.DataFrame(pending, schema=schema))
            pending = []

    if pending:
        batches.append(pl.DataFrame(pending, schema=schema))

    dataset = pl.concat(batches)
    original_count = dataset.height

    user_item = build_user_item_features(
        transactions, as_of, customer_ids
    )

    dataset = (
        dataset
        .join(
            user_item,
            on=["customer_id", "article_id"],
            how="left",
            validate="m:1",
        )
        .join(
            catalog.select(
                "article_id",
                "product_type_no",
                "item_popularity_7d",
                "item_popularity_28d",
            ),
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

    if dataset.height != original_count:
        raise ValueError("Feature joins changed the row count.")

    if dataset.select(
        pl.col("item_popularity_28d").is_null().any()
    ).item():
        raise ValueError("Missing item statistics.")

    return dataset


def main():
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
        schema_overrides={
            "article_id": pl.String,
            "product_code": pl.String,
        },
    ).select("article_id", "product_code", "product_type_no")

    if articles["article_id"].n_unique() != articles.height:
        raise ValueError("Duplicate article metadata.")

    snapshots = [
        (date(2020, 8, 19), "train_2020-08-19_v4", "actual_2020-08-19_v4"),
        (date(2020, 8, 26), "train_2020-08-26_v4", "actual_2020-08-26_v4"),
        (date(2020, 9, 2), "train_candidates_v4", "train_actual_v4"),
        (date(2020, 9, 9), "validation_candidates_v4", "validation_actual_v4"),
    ]

    for as_of, candidate_file, actual_file in snapshots:
        print(f"\nBuilding V6: {as_of}")

        base = pl.read_parquet(
            PROCESSED_DIR / f"{candidate_file}.parquet"
        )
        actual = pl.read_parquet(
            PROCESSED_DIR / f"{actual_file}.parquet"
        )

        if base["as_of"].unique().to_list() != [as_of]:
            raise ValueError("Unexpected snapshot date.")

        dataset = build_snapshot(
            transactions, articles, as_of, base, actual
        )

        if dataset.height != actual.height * 150:
            raise ValueError("Unexpected candidate row count.")

        dataset.write_parquet(
            PROCESSED_DIR / f"candidates_{as_of}_v6.parquet"
        )
        actual.write_parquet(
            PROCESSED_DIR / f"actual_{as_of}_v6.parquet"
        )

        print(f"Customers: {actual.height}")
        print(f"Rows: {dataset.height}")
        print(f"Positive rows: {dataset['label'].sum()}")

        del base, actual, dataset


if __name__ == "__main__":
    main()