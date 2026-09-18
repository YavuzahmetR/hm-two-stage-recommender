import json
from datetime import date, timedelta

import polars as pl

from popularity_baseline import DATA_DIR


AS_OF = date(2020, 9, 9)
LOOKBACK_DAYS = 28

OUTPUT_DIR = (
    DATA_DIR.parent.parent
    / "two_tower"
    / AS_OF.isoformat()
)


def main():
    history_start = AS_OF - timedelta(days=LOOKBACK_DAYS)

    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    interactions = (
        transactions
        .filter(
            (pl.col("t_dat") >= history_start)
            & (pl.col("t_dat") < AS_OF)
        )
        .select("customer_id", "article_id", "t_dat")
        .drop_nulls()
        .group_by(["customer_id", "article_id"])
        .agg(
            pl.len().alias("purchase_count"),
            pl.col("t_dat").min().alias("first_purchase"),
            pl.col("t_dat").max().alias("last_purchase"),
        )
        .collect()
    )

    if interactions.is_empty():
        raise ValueError("No interactions found.")

    if (
        interactions["first_purchase"].min() < history_start
        or interactions["last_purchase"].max() >= AS_OF
    ):
        raise ValueError("Interactions are outside the history window.")

    users = (
        interactions
        .select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index("user_idx")
    )

    items = (
        interactions
        .select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index("item_idx")
    )

    pairs = (
        interactions
        .join(
            users,
            on="customer_id",
            how="left",
            validate="m:1",
        )
        .join(
            items,
            on="article_id",
            how="left",
            validate="m:1",
        )
        .select(
            "user_idx",
            "item_idx",
            "purchase_count",
            "last_purchase",
        )
        .sort(["user_idx", "item_idx"])
    )

    if pairs.height != interactions.height:
        raise ValueError("Row count changed during ID mapping.")

    if (
        pairs["user_idx"].null_count()
        or pairs["item_idx"].null_count()
    ):
        raise ValueError("Missing ID mappings.")

    if pairs.select("user_idx", "item_idx").unique().height != pairs.height:
        raise ValueError("Duplicate user-item pairs.")

    user_counts = pairs.group_by("user_idx").len()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    users.write_parquet(OUTPUT_DIR / "users.parquet")
    items.write_parquet(OUTPUT_DIR / "items.parquet")
    pairs.write_parquet(OUTPUT_DIR / "interactions.parquet")

    metadata = {
        "as_of": AS_OF.isoformat(),
        "history_start": history_start.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "n_users": users.height,
        "n_items": items.height,
        "n_pairs": pairs.height,
        "n_transactions": int(pairs["purchase_count"].sum()),
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"History window: [{history_start}, {AS_OF})")
    print(f"Users: {users.height:,}")
    print(f"Items: {items.height:,}")
    print(f"Unique user-item pairs: {pairs.height:,}")
    print(f"Purchase rows: {metadata['n_transactions']:,}")
    print(
        "Users with one unique item: "
        f"{user_counts.filter(pl.col('len') == 1).height:,}"
    )
    print("\nInteraction preview:")
    print(pairs.head(5))
    print(f"\nSaved files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()