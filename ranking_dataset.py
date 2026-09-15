from covisitation import get_covisitation_candidates

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