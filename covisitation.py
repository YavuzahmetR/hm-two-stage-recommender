from collections import Counter, defaultdict
from datetime import date, timedelta

import polars as pl

from popularity_baseline import DATA_DIR, VALIDATION_START

def build_neighbors(
        transactions: pl.LazyFrame,
        as_of: date,
        lookback_days: int = 7,
        max_basket_size: int = 20,
        top_k: int = 20,
) -> dict[str, list[tuple[str,int]]]:
    baskets = (
        transactions.filter(
            (pl.col("t_dat") >= as_of - timedelta(days=lookback_days))
            & (pl.col("t_dat") < as_of)
            & pl.col("customer_id").is_not_null()
            & pl.col("article_id").is_not_null()
        )
        .group_by(["customer_id", "t_dat"])
        .agg(pl.col("article_id").unique().alias("items"))
        .collect()
    )

    pair_counts = defaultdict(Counter)
    used_baskets = 0
    skipped_large_baskets = 0

    for items in baskets["items"].to_list():
        if len(items) > max_basket_size:
            skipped_large_baskets += 1
            continue

        if len(items) < 2:
            continue

        used_baskets += 1

        for source in items:
            for target in items:
                if source != target:
                    pair_counts[source][target] += 1

    neighbors = {}

    for source, counts in pair_counts.items():
        ranked = sorted(
            counts.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        neighbors[source] = ranked[:top_k]

    print(f"Used baskets: {used_baskets}")
    print(f"Skipped large baskets: {skipped_large_baskets}")
    print(f"Products with neighbors: {len(neighbors)}")

    return neighbors


def get_covisitation_candidates(
    recent_items: list[str],
    neighbors: dict[str, list[tuple[str, int]]],
    k: int = 50,
) -> list[str]:
    scores = Counter()
    recent_set = set(recent_items)

    for source in recent_items:
        for target, count in neighbors.get(source, []):
            if target not in recent_set:
                scores[target] += count

    ranked = sorted(
        scores.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )

    return [article_id for article_id, _ in ranked[:k]]





def main():
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    neighbors = build_neighbors(
        transactions,
        as_of=VALIDATION_START,
    )

    if not neighbors:
        raise ValueError("No product neighbors found.")

    example_item = sorted(neighbors)[0]

    print(f"\nExample product: {example_item}")
    print("Neighbors:", neighbors[example_item][:5])


if __name__ == "__main__":
    main()
