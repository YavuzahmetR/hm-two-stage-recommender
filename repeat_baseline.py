from datetime import timedelta

import polars as pl

from covisitation import build_neighbors, get_covisitation_candidates

from ranking_dataset import build_candidate_rows

from popularity_baseline import(
    DATA_DIR,
    VALIDATION_START,
    average_presicion_at_k,
    build_popularity
)


def build_recommendations(
        recent_items: list[str],
        popular_items: list[str],
        k: int = 12
) -> list[str]:
    combined = recent_items + popular_items
    return list(dict.fromkeys(combined))[:k]  # Remove recurring product - keep the sorting(1- recent, 2-popular) that's why we dont use set.

def evaluate_candidate_recall(
        actual_by_customer: pl.DataFrame,
        recent_lookup: dict[str,list[str]],
        popular_items: list[str],
        k:int
) -> float:
    recall_sum = 0.0

    for customer_id, actual in actual_by_customer.iter_rows():
        candidates = build_recommendations(
            recent_lookup.get(customer_id, []),
            popular_items,
            k=k
        )

        if len(candidates) != k:
            raise ValueError(f"Expected {k} candidates.")

        actual_items = set(actual)
        hits = len(actual_items.intersection(candidates))
        recall_sum += hits / len(actual_items)

    return recall_sum / actual_by_customer.height
        

def evaluate_candidate_sources(
    actual_by_customer: pl.DataFrame,
    recent_lookup: dict[str, list[str]],
    popular_pool: list[str],
    neighbors: dict[str, list[tuple[str, int]]],
    k: int = 150,
) -> None:
    baseline_sum = 0.0
    enriched_sum = 0.0
    improved_customers = 0
    worsened_customers = 0

    for customer_id, actual in actual_by_customer.iter_rows():
        recent_items = recent_lookup.get(customer_id, [])

        baseline = build_recommendations(
            recent_items,
            popular_pool,
            k=k,
        )

        related_items = get_covisitation_candidates(
            recent_items,
            neighbors,
            k=50,
        )

        enriched = build_recommendations(
            recent_items + related_items,
            popular_pool,
            k=k,
        )

        if len(baseline) != k or len(enriched) != k:
            raise ValueError(f"Expected {k} candidates per source mix.")

        actual_items = set(actual)
        baseline_hits = len(actual_items.intersection(baseline))
        enriched_hits = len(actual_items.intersection(enriched))

        baseline_sum += baseline_hits / len(actual_items)
        enriched_sum += enriched_hits / len(actual_items)

        improved_customers += int(enriched_hits > baseline_hits)
        worsened_customers += int(enriched_hits < baseline_hits)

    customer_count = actual_by_customer.height

    if customer_count == 0:
        raise ValueError("No customers found in the target period.")

    baseline_recall = baseline_sum / customer_count
    enriched_recall = enriched_sum / customer_count

    print(f"\nBaseline Recall@{k}: {baseline_recall:.6f}")
    print(f"With covisitation Recall@{k}: {enriched_recall:.6f}")
    print(f"Recall difference: {enriched_recall - baseline_recall:+.6f}")
    print(f"Customers with more hits: {improved_customers}")
    print(f"Customers with fewer hits: {worsened_customers}")



def main():
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id" : pl.String,
            "article_id": pl.String
        },
        try_parse_dates=True
    )

    popular_pool = build_popularity(
        transactions,
        as_of=VALIDATION_START,
        k=150,
    )["article_id"].to_list()

    popular_items = popular_pool[:12]

    history_start = VALIDATION_START - timedelta(days=28)
    target_end = VALIDATION_START + timedelta(days=7)

    recent_purchases = (
        transactions.filter(
            (pl.col("t_dat") >= history_start)
            & (pl.col("t_dat") < VALIDATION_START)
            & pl.col("customer_id").is_not_null()
            & pl.col("article_id").is_not_null()
        ).group_by(["customer_id", "article_id"])
        .agg(pl.col("t_dat").max().alias("last_purchase"))
        .sort(
            ["customer_id","last_purchase","article_id"],
            descending=[False,True,False]
        ).group_by("customer_id")
        .agg(pl.col("article_id").head(12).alias("recent_items"))
        .collect()
    )

    recent_lookup = dict(recent_purchases.iter_rows())

    print(f"Customers with recent history: {len(recent_lookup)}")
    print(recent_purchases.head(3))

    actual_by_customer = (
        transactions
        .filter(
            (pl.col("t_dat") >= VALIDATION_START)
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

    recall_sum = 0.0
    hit_customers = 0
    ap_sum = 0.0

    customers_with_history = 0
    changed_lists = 0

    for customer_id, actual in actual_by_customer.iter_rows():
        recent_items = recent_lookup.get(customer_id, [])
        predicted = build_recommendations(
            recent_items,
            popular_items
        )

        customers_with_history += int(bool(recent_items))
        changed_lists += int(predicted != popular_items)

        if customers_with_history == 1 and recent_items:
            print("Recent items:", recent_items)
            print("Personalized recommendations:", predicted)

        if len(predicted) != 12 or len(set(predicted)) != 12:
            raise ValueError("Expected 12 unique recommendations.")

        actual_items = set(actual)
        hits = len(actual_items.intersection(predicted))

        recall_sum += hits / len(actual_items)
        hit_customers += int(hits > 0)
        ap_sum += average_presicion_at_k(actual, predicted)

    customer_count = actual_by_customer.height

    print(f"Evaluated customers: {customer_count}")
    print(f"Validation Recall@12: {recall_sum / customer_count:.6f}")
    print(f"Validation HitRate@12: {hit_customers / customer_count:.6f}")
    print(f"Validation MAP@12: {ap_sum / customer_count:.6f}")


    print(f"Evaluated customers with history: {customers_with_history}")
    print(f"Lists different from popularity: {changed_lists}")

    print("\nCandidate evaluation:")

    for k in [50, 100, 150]:
        recall = evaluate_candidate_recall(
            actual_by_customer,
            recent_lookup,
            popular_pool,
            k=k,
        )
        print(f"Candidate Recall@{k}: {recall:.6f}")

    neighbors = build_neighbors(
        transactions,
        as_of=VALIDATION_START,
    )

    evaluate_candidate_sources(
        actual_by_customer,
        recent_lookup,
        popular_pool,
        neighbors,
    )

    example_customer = next(
        customer_id
        for customer_id in sorted(
            actual_by_customer["customer_id"].to_list()
        )
        if customer_id in recent_lookup
    )

    example_rows = build_candidate_rows(
        customer_id=example_customer,
        recent_items=recent_lookup[example_customer],
        popular_pool=popular_pool,
        neighbors=neighbors,
    )

    preview = pl.DataFrame(example_rows)

    with pl.Config(tbl_width_chars=120, tbl_rows=15):
        print(preview.select(pl.exclude("customer_id")).head(15))


if __name__ == "__main__":
    main()
    