import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import Booster

from build_v6 import build_snapshot
from popularity_baseline import average_precision_at_k, build_popularity
from train_ranker import evaluate_predictions
from train_v6 import FEATURES, get_predictions


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "hm"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "metrics"

MODEL_PATH = PROCESSED_DIR / "ranker_v6_tuned.txt"
MODEL_NAME = "Frozen tuned V6"

TEST_START = date(2020, 9, 16)
TEST_END = date(2020, 9, 23)
CUSTOMER_LIMIT = 10000


def scan_transactions():
    return pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = Booster(
        model_str=MODEL_PATH.read_text(encoding="utf-8")
    )

    if (
        model.feature_name() != FEATURES
        or model.current_iteration() != 40
    ):
        raise ValueError(
            "Expected the selected 11-feature, 40-tree V6 model."
        )

    return model


def check_metrics():
    examples = [
        (["a"], ["a"], 1.0),
        (["a"], ["x", "a"], 0.5),
        (["a", "b"], ["a", "x", "b"], 5 / 6),
        (["a"], ["x"], 0.0),
        (["a"], ["a", "a"], 1.0),
    ]

    for actual, predicted, expected in examples:
        score = average_precision_at_k(actual, predicted)

        if not np.isclose(score, expected):
            raise ValueError("AP metric check failed.")


def freeze_selection():
    source_files = [
        "finish_project.py",
        "build_v6.py",
        "ranking_dataset.py",
        "covisitation.py",
        "popularity_baseline.py",
        "train_ranker.py",
        "train_v6.py",
    ]

    manifest = {
        "selected_method": MODEL_NAME,
        "model_sha256": file_hash(MODEL_PATH),
        "test_start": str(TEST_START),
        "test_end_exclusive": str(TEST_END),
        "max_customers": CUSTOMER_LIMIT,
        "sampling": (
            "SHA256(customer_id), then customer_id; "
            "active test customers"
        ),
        "refit_before_test": False,
        "training_target_weeks": [
            "2020-08-19",
            "2020-08-26",
            "2020-09-02",
        ],
        "validation_start": "2020-09-09",
        "features": FEATURES,
        "source_sha256": {
            name: file_hash(ROOT / name)
            for name in source_files
        },
    }

    path = REPORT_DIR / "final_selection.json"

    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))

        if previous != manifest:
            raise ValueError(
                "Frozen model, protocol or source changed. "
                "Review the existing final selection before proceeding."
            )
    else:
        path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def build_candidates(transactions, customer_ids):
    past = transactions.filter(pl.col("t_dat") < TEST_START)

    recent = (
        past
        .filter(
            (pl.col("t_dat") >= TEST_START - timedelta(days=28))
            & pl.col("customer_id").is_in(customer_ids)
            & pl.col("article_id").is_not_null()
        )
        .group_by(["customer_id", "article_id"])
        .agg(pl.col("t_dat").max().alias("last_purchase"))
        .sort(
            ["customer_id", "last_purchase", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12))
        .collect()
    )

    seed_rows = [
        (customer_id, item, rank)
        for customer_id, items in recent.iter_rows()
        for rank, item in enumerate(items, start=1)
    ]

    seeds = pl.DataFrame(
        seed_rows,
        schema={
            "customer_id": pl.String,
            "article_id": pl.String,
            "repeat_rank": pl.Int32,
        },
        orient="row",
    )

    # Candidate generation receives no target purchases.
    request = pl.DataFrame(
        {
            "customer_id": pl.Series(
                customer_ids,
                dtype=pl.String,
            ),
            "actual_items": pl.Series(
                [[] for _ in customer_ids],
                dtype=pl.List(pl.String),
            ),
        }
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

    candidates = build_snapshot(
        past,
        articles,
        TEST_START,
        seeds,
        request,
    )

    if candidates["label"].sum() != 0:
        raise ValueError(
            "Prediction-only snapshot contains positive labels."
        )

    candidates = (
        candidates
        .drop("label")
        .sort(["customer_id", "article_id"])
    )

    counts = (
        candidates
        .group_by("customer_id")
        .len()["len"]
    )

    unique_pairs = candidates.select(
        "customer_id", "article_id"
    ).unique().height

    if (
        candidates.height != len(customer_ids) * 150
        or not (counts == 150).all()
        or unique_pairs != candidates.height
        or set(candidates["customer_id"]) != set(customer_ids)
    ):
        raise ValueError("Invalid candidate groups.")

    recency = candidates["user_item_recency_days"].drop_nulls()

    if len(recency) and (
        recency.min() < 1 or recency.max() > 28
    ):
        raise ValueError("Invalid historical recency.")

    return candidates


def predict(model, candidates):
    matrix = candidates.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()

    scores = model.predict(matrix)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores.")

    predictions = (
        candidates
        .select("customer_id", "article_id")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(
            pl.col("article_id").head(12).alias("predictions")
        )
        .select("customer_id", "predictions")
    )

    for _, items in predictions.iter_rows():
        if len(items) != 12 or len(set(items)) != 12:
            raise ValueError(
                "Expected 12 unique recommendations."
            )

    return predictions


def run_test(model):
    transactions = scan_transactions()

    actual = (
        transactions
        .filter(
            (pl.col("t_dat") >= TEST_START)
            & (pl.col("t_dat") < TEST_END)
            & pl.col("customer_id").is_not_null()
            & pl.col("article_id").is_not_null()
        )
        .group_by("customer_id")
        .agg(
            pl.col("article_id")
            .unique()
            .alias("actual_items")
        )
        .collect()
    )

    if actual.height < CUSTOMER_LIMIT:
        raise ValueError(
            "Expected at least 10,000 active test customers."
        )

    selected_ids = sorted(
        actual["customer_id"].to_list(),
        key=lambda value: (
            hashlib.sha256(value.encode()).digest(),
            value,
        ),
    )[:CUSTOMER_LIMIT]

    actual = (
        actual
        .filter(pl.col("customer_id").is_in(selected_ids))
        .sort("customer_id")
    )

    print(f"Selected test customers: {actual.height:,}")
    print("Building historical candidates...", flush=True)

    candidates = build_candidates(
        transactions, selected_ids
    )

    selected_predictions = predict(model, candidates)

    popular_items = build_popularity(
        transactions,
        as_of=TEST_START,
        k=12,
    )["article_id"].to_list()

    popularity_predictions = pl.DataFrame(
        {
            "customer_id": actual["customer_id"],
            "predictions": [
                popular_items for _ in range(actual.height)
            ],
        }
    )

    methods = [
        ("Popularity", popularity_predictions),
        ("Variant rule", get_predictions(candidates)),
        (MODEL_NAME, selected_predictions),
    ]

    history_ids = (
        candidates
        .filter(pl.col("repeat_rank").is_not_null())
        ["customer_id"]
        .unique()
        .to_list()
    )

    segments = [
        ("all", actual),
        (
            "history_28d",
            actual.filter(
                pl.col("customer_id").is_in(history_ids)
            ),
        ),
        (
            "no_history_28d",
            actual.filter(
                ~pl.col("customer_id").is_in(history_ids)
            ),
        ),
    ]

    records = []

    for segment, cohort in segments:
        if cohort.is_empty():
            continue

        for name, predictions in methods:
            metrics = evaluate_predictions(cohort, predictions)

            records.append(
                {
                    "segment": segment,
                    "method": name,
                    "customers": cohort.height,
                    **metrics,
                }
            )

    actual_lookup = {
        customer_id: set(items)
        for customer_id, items in actual.iter_rows()
    }

    recall_values = []
    hit_values = []

    candidate_lists = (
        candidates
        .group_by("customer_id")
        .agg(pl.col("article_id"))
    )

    for customer_id, items in candidate_lists.iter_rows():
        relevant = actual_lookup[customer_id]
        hits = len(relevant & set(items))

        recall_values.append(hits / len(relevant))
        hit_values.append(hits > 0)

    audit = {
        "customers": actual.height,
        "candidate_rows": candidates.height,
        "candidate_recall_at_150": float(
            np.mean(recall_values)
        ),
        "candidate_hit_rate_at_150": float(
            np.mean(hit_values)
        ),
        "checks_passed": [
            "AP sanity examples",
            "Frozen model and source hashes",
            "Candidate builder receives only pre-test transactions",
            "Candidate builder receives empty target lists",
            "150 unique candidates per customer",
            "Historical recency bounds",
            "Finite prediction scores",
            "12 unique recommendations per customer",
        ],
    }

    actual.write_parquet(
        PROCESSED_DIR / "final_test_actual.parquet"
    )
    selected_predictions.write_parquet(
        PROCESSED_DIR / "final_test_predictions.parquet"
    )

    (REPORT_DIR / "final_test_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    report = pl.DataFrame(records)
    report.write_csv(REPORT_DIR / "final_test.csv")

    print(
        "\nCandidate Recall@150: "
        f"{audit['candidate_recall_at_150']:.6f}"
    )
    print(
        "Candidate HitRate@150: "
        f"{audit['candidate_hit_rate_at_150']:.6f}"
    )

    return report


def recommend(model, customer_id):
    manifest_path = REPORT_DIR / "final_selection.json"

    if manifest_path.exists():
        frozen = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if frozen["model_sha256"] != file_hash(MODEL_PATH):
            raise ValueError("Selected model has changed.")

    cache = PROCESSED_DIR / "final_test_predictions.parquet"
    result = None

    if cache.exists():
        query = pl.scan_parquet(cache)

        if customer_id is None:
            result = query.head(1).collect()
        else:
            result = (
                query
                .filter(pl.col("customer_id") == customer_id)
                .collect()
            )

    if result is None or result.is_empty():
        if customer_id is None:
            raise ValueError(
                "Run evaluate first or provide --customer-id."
            )

        candidates = build_candidates(
            scan_transactions(),
            [customer_id],
        )
        result = predict(model, candidates)

    row = result.row(0, named=True)

    print(
        json.dumps(
            {"as_of": str(TEST_START), **row},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["evaluate", "recommend"],
    )
    parser.add_argument("--customer-id")
    args = parser.parse_args()

    check_metrics()
    model = load_model()

    if args.command == "recommend":
        recommend(model, args.customer_id)
        return

    if args.customer_id is not None:
        parser.error(
            "--customer-id is only supported by recommend."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    freeze_selection()

    report_path = REPORT_DIR / "final_test.csv"

    if report_path.exists():
        print("Using saved test results; no reevaluation.")
        report = pl.read_csv(report_path)
    else:
        print(
            "Evaluating frozen V6 on September 16-22, 2020...",
            flush=True,
        )
        report = run_test(model)

    with pl.Config(
        tbl_rows=12,
        tbl_width_chars=140,
        float_precision=6,
    ):
        print("\nFinal test results:")
        print(report)

    print("\nFinal test results saved.")
    print("Run: python finish_project.py recommend")


if __name__ == "__main__":
    main()