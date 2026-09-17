from datetime import date

import polars as pl
from lightgbm import LGBMRanker

from popularity_baseline import DATA_DIR
from ranking_dataset import build_training_snapshot
from train_ranker import FEATURES, evaluate_predictions


PROCESSED_DIR = DATA_DIR.parent.parent / "processed"
TRAIN_DATES = [
    date(2020, 8, 19),
    date(2020, 8, 26),
    date(2020, 9, 2),
]


def get_predictions(
    candidates: pl.DataFrame,
    model=None,
) -> pl.DataFrame:
    if model is None:
        ranked = candidates.sort(
            [
                "customer_id",
                "repeat_rank",
                "covisit_rank",
                "popularity_rank",
                "article_id",
            ],
            nulls_last=True,
        )
    else:
        features = candidates.select(
            pl.col(FEATURES).cast(pl.Float32)
        ).to_numpy()

        ranked = (
            candidates
            .with_columns(
                pl.Series("score", model.booster_.predict(features))
            )
            .sort(
                ["customer_id", "score", "article_id"],
                descending=[False, True, False],
            )
        )

    return (
        ranked
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
        .select("customer_id", "predictions")
    )


def main():
    transactions = pl.scan_csv(
        DATA_DIR / "transactions_train.csv",
        schema_overrides={
            "customer_id": pl.String,
            "article_id": pl.String,
        },
        try_parse_dates=True,
    )

    snapshots = []

    for as_of in TRAIN_DATES:
        print(f"\nPreparing training week: {as_of}")

        if as_of == date(2020, 9, 2):
            candidates = pl.read_parquet(
                PROCESSED_DIR / "train_candidates_v4.parquet"
            )
            actual = pl.read_parquet(
                PROCESSED_DIR / "train_actual_v4.parquet"
            )
        else:
            candidates, actual = build_training_snapshot(
                transactions,
                as_of=as_of,
                max_customers=5000,
            )

            candidates.write_parquet(
                PROCESSED_DIR / f"train_{as_of}_v4.parquet"
            )
            actual.write_parquet(
                PROCESSED_DIR / f"actual_{as_of}_v4.parquet"
            )

        if candidates["as_of"].unique().to_list() != [as_of]:
            raise ValueError("Unexpected training snapshot date.")

        snapshots.append((as_of, candidates, actual))

    train = pl.concat(
        [candidates for _, candidates, _ in snapshots]
    ).sort(["as_of", "customer_id", "article_id"])

    groups = (
        train
        .group_by(["as_of", "customer_id"], maintain_order=True)
        .len()["len"]
        .to_numpy()
    )

    if groups.sum() != train.height or not (groups == 150).all():
        raise ValueError("Invalid ranking groups.")

    x_train = train.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )

    print(f"\nTraining rows: {train.height}")
    print(f"Training customer-week groups: {len(groups)}")

    model.fit(
        x_train,
        train["label"].to_numpy(),
        group=groups,
        feature_name=FEATURES,
    )

    model.booster_.save_model(
        str(PROCESSED_DIR / "ranker_v5.txt")
    )

    del train, x_train

    validation = pl.read_parquet(
        PROCESSED_DIR / "validation_candidates_v4.parquet"
    )
    validation_actual = pl.read_parquet(
        PROCESSED_DIR / "validation_actual_v4.parquet"
    )

    evaluations = [
        (f"train_{as_of}", candidates, actual)
        for as_of, candidates, actual in snapshots
    ]
    evaluations.append(
        ("validation", validation, validation_actual)
    )

    records = []

    for split, candidates, actual in evaluations:
        actual = actual.select("customer_id", "actual_items")

        for name, predictor in [
            ("Candidate order", None),
            ("LightGBM v5", model),
        ]:
            predictions = get_predictions(candidates, predictor)
            metrics = evaluate_predictions(actual, predictions)

            records.append(
                {
                    "split": split,
                    "method": name,
                    "customers": actual.height,
                    **metrics,
                }
            )

    report = pl.DataFrame(records)

    output_dir = DATA_DIR.parent.parent.parent / "reports" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    report.write_csv(output_dir / "multiweek_v5.csv")

    with pl.Config(
        tbl_rows=20,
        tbl_width_chars=140,
        float_precision=6,
    ):
        print(report)

    print("\nSaved ranker_v5.txt and multiweek_v5.csv")


if __name__ == "__main__":
    main()