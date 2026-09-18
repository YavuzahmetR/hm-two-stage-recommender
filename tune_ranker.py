import gc
import json

import numpy as np
import polars as pl
from lightgbm import Booster, LGBMRanker, early_stopping

from train_ranker import evaluate_predictions
from train_v6 import (
    DATA_DIR,
    FEATURES,
    PROCESSED_DIR,
    TRAIN_DATES,
    VALIDATION_DATE,
    get_predictions,
)


TRIALS = [
    ("leaves7", 7, 50, 1.0, 0.05),
    ("leaves15", 15, 50, 1.0, 0.05),
    ("leaves31", 31, 50, 1.0, 0.05),
    ("leaves7_regularized", 7, 200, 10.0, 0.05),
    ("leaves15_regularized", 15, 200, 10.0, 0.05),
    ("leaves31_regularized", 31, 200, 10.0, 0.05),
    ("leaves15_slow", 15, 100, 5.0, 0.03),
    ("leaves31_slow", 31, 100, 5.0, 0.03),
]

REPORT_DIR = DATA_DIR.parent.parent.parent / "reports" / "metrics"


def get_groups(frame, columns):
    groups = (
        frame
        .group_by(columns, maintain_order=True)
        .len()["len"]
        .to_numpy()
    )

    if groups.sum() != frame.height or not (groups == 150).all():
        raise ValueError("Expected 150 contiguous rows per ranking group.")

    return groups


def predictions_from_scores(candidates, scores):
    return (
        candidates
        .select("customer_id", "article_id")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["customer_id", "score", "article_id"],
            descending=[False, True, False],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(12).alias("predictions"))
    )


def make_map_metric(validation, actual):
    if actual["customer_id"].n_unique() != actual.height:
        raise ValueError("Actual customers must be unique.")

    customers = validation.select("customer_id").unique(
        maintain_order=True
    )

    if set(customers["customer_id"]) != set(actual["customer_id"]):
        raise ValueError("Validation customer sets do not match.")

    aligned = customers.join(
        actual.select("customer_id", "actual_items"),
        on="customer_id",
        how="left",
        validate="1:1",
        maintain_order="left",
    )

    actual_counts = np.array(
        [
            len(set(items))
            for items in aligned["actual_items"].to_list()
        ],
        dtype=np.int32,
    )

    if (actual_counts == 0).any():
        raise ValueError("Expected non-empty actual purchase lists.")

    labels = validation["label"].to_numpy().reshape(-1, 150)
    denominators = np.minimum(actual_counts, 12)
    positions = np.arange(1, 13)

    def full_map_at_12(y_true, scores):
        score_matrix = scores.reshape(-1, 150)

        order = np.argsort(
            -score_matrix,
            axis=1,
            kind="stable",
        )[:, :12]

        hits = np.take_along_axis(labels, order, axis=1)
        precision = np.cumsum(hits, axis=1) / positions
        ap = (precision * hits).sum(axis=1) / denominators

        return "full_map_at_12", float(ap.mean()), True

    return full_map_at_12


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    for as_of in TRAIN_DATES:
        frame = pl.read_parquet(
            PROCESSED_DIR / f"candidates_{as_of}_v6.parquet"
        )

        if frame["as_of"].unique().to_list() != [as_of]:
            raise ValueError("Unexpected training snapshot date.")

        frames.append(frame)

    train = pl.concat(frames).sort(
        ["as_of", "customer_id", "article_id"]
    )
    del frames, frame

    validation = pl.read_parquet(
        PROCESSED_DIR / f"candidates_{VALIDATION_DATE}_v6.parquet"
    ).sort(["customer_id", "article_id"])

    actual = pl.read_parquet(
        PROCESSED_DIR / f"actual_{VALIDATION_DATE}_v6.parquet"
    ).select("customer_id", "actual_items")

    if validation["as_of"].unique().to_list() != [VALIDATION_DATE]:
        raise ValueError("Unexpected validation snapshot date.")

    if (
        validation.select("customer_id", "article_id")
        .unique()
        .height
        != validation.height
    ):
        raise ValueError("Duplicate validation candidates.")

    train_groups = get_groups(train, ["as_of", "customer_id"])
    validation_groups = get_groups(validation, ["customer_id"])

    x_train = train.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()
    y_train = train["label"].to_numpy()

    x_validation = validation.select(
        pl.col(FEATURES).cast(pl.Float32)
    ).to_numpy()
    y_validation = validation["label"].to_numpy()

    map_metric = make_map_metric(validation, actual)

    print(f"Training rows: {train.height}")
    print(f"Training groups: {len(train_groups)}")
    print(f"Validation customers: {actual.height}")

    del train
    gc.collect()

    records = []

    rule_predictions = get_predictions(validation)
    rule_metrics = evaluate_predictions(actual, rule_predictions)
    records.append({"method": "Variant rule", **rule_metrics})

    original = Booster(
        model_file=str(PROCESSED_DIR / "ranker_v6.txt")
    )

    if original.feature_name() != FEATURES:
        raise ValueError("V6 model feature order does not match.")

    original_predictions = predictions_from_scores(
        validation,
        original.predict(x_validation),
    )
    original_metrics = evaluate_predictions(
        actual,
        original_predictions,
    )
    records.append({"method": "LightGBM v6", **original_metrics})

    del original, original_predictions, rule_predictions

    best_score = -1.0
    best_name = None

    for index, trial in enumerate(TRIALS, start=1):
        name, leaves, min_samples, regularization, rate = trial

        params = {
            "objective": "lambdarank",
            "metric": "None",
            "n_estimators": 600,
            "learning_rate": rate,
            "num_leaves": leaves,
            "min_child_samples": min_samples,
            "reg_lambda": regularization,
            "random_state": 42,
            "n_jobs": 4,
            "deterministic": True,
            "force_col_wise": True,
            "importance_type": "gain",
            "verbosity": -1,
        }

        print(f"\nTrial {index}/{len(TRIALS)}: {name}")

        model = LGBMRanker(**params)
        model.fit(
            x_train,
            y_train,
            group=train_groups,
            feature_name=FEATURES,
            eval_set=[(x_validation, y_validation)],
            eval_group=[validation_groups],
            eval_metric=map_metric,
            callbacks=[
                early_stopping(
                    stopping_rounds=40,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )

        scores = model.booster_.predict(
            x_validation,
            num_iteration=model.best_iteration_,
        )
        predictions = predictions_from_scores(validation, scores)
        metrics = evaluate_predictions(actual, predictions)

        callback_map = map_metric(y_validation, scores)[1]
        if not np.isclose(
            callback_map,
            metrics["map_at_12"],
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("MAP calculations do not match.")

        records.append(
            {
                "method": name,
                "num_leaves": leaves,
                "min_child_samples": min_samples,
                "reg_lambda": regularization,
                "learning_rate": rate,
                "best_iteration": model.best_iteration_,
                **metrics,
            }
        )

        print(
            f"Best iteration: {model.best_iteration_} | "
            f"MAP@12: {metrics['map_at_12']:.6f}"
        )

        if metrics["map_at_12"] > best_score:
            best_score = metrics["map_at_12"]
            best_name = name

            model.booster_.save_model(
                str(PROCESSED_DIR / "ranker_v6_tuned.txt"),
                num_iteration=model.best_iteration_,
            )

            metadata = {
                "trial": name,
                "params": params,
                "best_iteration": model.best_iteration_,
                "features": FEATURES,
                "validation_date": str(VALIDATION_DATE),
                "metrics": metrics,
            }

            (REPORT_DIR / "tuning_v6_best.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

        (
            pl.from_dicts(records, infer_schema_length=None)
            .sort("map_at_12", descending=True)
            .write_csv(REPORT_DIR / "tuning_v6.csv")
        )

        del model, predictions, scores
        gc.collect()

    report = pl.from_dicts(records, infer_schema_length=None).sort(
        "map_at_12",
        descending=True,
    )

    with pl.Config(tbl_rows=20, tbl_width_chars=140, float_precision=6):
        print(
            report.select(
                "method",
                "best_iteration",
                "recall_at_12",
                "hit_rate_at_12",
                "map_at_12",
            )
        )

    print(f"\nBest tuning trial: {best_name}")
    print(f"MAP difference vs V6: {best_score - original_metrics['map_at_12']:+.6f}")
    print(f"MAP difference vs rule: {best_score - rule_metrics['map_at_12']:+.6f}")
    print("Saved tuning_v6.csv, tuning_v6_best.json and ranker_v6_tuned.txt")


if __name__ == "__main__":
    main()