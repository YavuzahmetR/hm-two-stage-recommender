from pathlib import Path
import polars as pl

DATA_DIR = Path("data/raw/hm")

def main():
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id" : pl.String,
            "article_id" : pl.String
        },
        try_parse_dates=True
    )

    print("Transaction Schema:")
    print(transactions.collect_schema())

    print("\nFirst five Transactions:")
    print(transactions.head(5).collect())

    summary = transactions.select(
        pl.len().alias("row_count"),
        pl.col("t_dat").min().alias("first_date"),
        pl.col("t_dat").max().alias("last_date")
    ).collect()

    print("\nTransaction Summary:")
    print(summary)

    for filename, id_column in [
        ("articles.csv", "article_id"),
        ("customers.csv", "customer_id")
    ]:
        table = pl.scan_csv(
            DATA_DIR / filename,
            schema_overrides={id_column: pl.String}
        )

        print(f"\n{filename} schema:")
        print(table.collect_schema())

        print(f"\n{filename} preview:")
        print(table.head(3).collect())

if __name__ == "__main__":
    main()