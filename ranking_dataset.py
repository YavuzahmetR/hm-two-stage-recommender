from datetime import date, timedelta
from hashlib import sha256

import polars as pl

from covisitation import get_covisitation_candidates, build_neighbors

from popularity_baseline import DATA_DIR, build_popularity

def build_candidate_rows(
    customer_id: str,
    recent_items: list[str],
    popular_pool: list[str],
    neighbors: dict[str, list[tuple[str, int]]],
    k: int = 150,
) -> list[dict]:
    related_items = get_covisitation_candidates(
        recent_items,
        neighbors,
        k=50,
    )

    repeat_ranks = {
        article_id: rank
        for rank, article_id in enumerate(recent_items, start=1)
    }

    covisit_ranks = {
        article_id: rank
        for rank, article_id in enumerate(related_items, start=1)
    }

    popularity_ranks = {
        article_id: rank
        for rank, article_id in enumerate(popular_pool, start=1)
    }

    candidates = list(
        dict.fromkeys(recent_items + related_items + popular_pool)
    )[:k]

    if len(candidates) != k:
        raise ValueError(f"Expected {k} unique candidates.")

    rows = []

    for article_id in candidates:
        rows.append(
            {
                "customer_id": customer_id,
                "article_id": article_id,
                "repeat_rank": repeat_ranks.get(article_id),
                "covisit_rank": covisit_ranks.get(article_id),
                "popularity_rank": popularity_ranks.get(article_id),
            }
        )

    return rows



def build_training_snapshot(
    transactions: pl.LazyFrame,
    as_of: date,
    max_customers: int = 5000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_end = as_of + timedelta(days=7)

    actual_by_customer = (
        transactions
        .filter(
            (pl.col("t_dat") >= as_of)
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

    selected_ids = sorted(
        actual_by_customer["customer_id"].to_list(),
        key=lambda value: (sha256(value.encode()).digest(), value),
    )[:max_customers]

    selected_actual = (
        actual_by_customer
        .filter(pl.col("customer_id").is_in(selected_ids))
        .sort("customer_id")
    )

    recent_purchases = (
        transactions
        .filter(
            (pl.col("t_dat") >= as_of - timedelta(days=28))
            & (pl.col("t_dat") < as_of)
            & pl.col("customer_id").is_in(selected_ids)
            & pl.col("article_id").is_not_null()
        )
        .group_by(["customer_id", "article_id"])
        .agg(pl.col("t_dat").max().alias("last_purchase"))
        .sort(
            ["customer_id", "last_purchase", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id")
        .agg(pl.col("article_id").head(12).alias("recent_items"))
        .collect()
    )

    recent_lookup = dict(recent_purchases.iter_rows())

    popular_pool = build_popularity(
        transactions,
        as_of=as_of,
        k=150,
    )["article_id"].to_list()

    neighbors = build_neighbors(transactions, as_of=as_of)

    schema = {
        "customer_id": pl.String,
        "article_id": pl.String,
        "repeat_rank": pl.Int32,
        "covisit_rank": pl.Int32,
        "popularity_rank": pl.Int32,
        "label": pl.Int8,
        "as_of": pl.Date,
    }

    batches = []
    pending_rows = []
    customers_without_hits = 0

    for customer_id, actual in selected_actual.iter_rows():
        rows = build_candidate_rows(
            customer_id=customer_id,
            recent_items=recent_lookup.get(customer_id, []),
            popular_pool=popular_pool,
            neighbors=neighbors,
        )

        actual_items = set(actual)

        for row in rows:
            row["label"] = int(row["article_id"] in actual_items)
            row["as_of"] = as_of

        customers_without_hits += int(
            not any(row["label"] for row in rows)
        )

        pending_rows.extend(rows)

        if len(pending_rows) >= 15000:
            batches.append(pl.DataFrame(pending_rows, schema=schema))
            pending_rows = []

    if pending_rows:
        batches.append(pl.DataFrame(pending_rows, schema=schema))

    dataset = pl.concat(batches)

    print(f"Selected customers: {selected_actual.height}")
    print(f"Candidate rows: {dataset.height}")
    print(f"Positive rows: {dataset['label'].sum()}")
    print(f"Customers without positive candidates: {customers_without_hits}")

    return dataset, selected_actual


def main():
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    training_start = date(2020, 9, 2)

    dataset, actual = build_training_snapshot(
        transactions,
        as_of=training_start,
    )

    output_dir = DATA_DIR.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.write_parquet(output_dir / "train_candidates.parquet")
    actual.write_parquet(output_dir / "train_actual.parquet")

    print("\nTraining files saved.")


if __name__ == "__main__":
    main()        