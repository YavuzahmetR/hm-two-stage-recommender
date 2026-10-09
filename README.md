# H&M Two-Stage Recommender

## Overview

An offline fashion recommendation project using H&M purchase history and product metadata. It generates 150 candidates per customer and ranks them with LightGBM to produce 12 recommendations. ID-based and feature-based two-tower experiments examine whether neural candidates improve retrieval and final ranking.

The frozen V6 ranker was selected on validation before the final test. **The variant rule won on the final test week.** Recorded metrics and model-selection reports are included in `reports/metrics/`.

## Scope and limitations

- Local offline evaluation on sampled active purchasers, not Kaggle leaderboard or online business results.
- One final test week; results do not describe all customers or future periods.
- Product metadata is treated as static. Customer profile fields and images are not used by the final pipeline.
- “No history” means no purchases during the previous 28 days, **not necessarily a new customer**.
- Validation was used repeatedly in development; its best score is not an independent estimate.
- Original data, frozen model, and customer-level caches are not distributed with Git. Complete training/inference reproduction requires those artifacts and a compatible environment.
- Original dependency versions were not fully recorded. The requirements files are installation lists, not a historical lockfile.
- Image-based retrieval is a separate project using the same article IDs.

## Architecture and workflow

```text
Recent purchases + product variants + co-visitation + popularity
                              |
                    150 unique candidates
                              |
                    LightGBM LambdaRank
                              |
                      12 recommendations
```

Candidate sources, in their original priority order:

1. Recent purchases: up to 12 distinct products from the previous 28 days.
2. Product variants: up to 20 products sharing a product code with recent purchases.
3. Co-visitation: up to 50 related products from customer-day baskets in the previous week.
4. Popularity: products from the previous seven days fill the remaining positions.

Deduplication preserves order. Ties use explicit product-ID ordering where the original algorithm does so. Previously purchased products remain eligible because repeat purchases are valid recommendation targets.

The selected LightGBM V6 model uses 11 features, in this order:

```text
repeat_rank
covisit_rank
popularity_rank
covisit_score
user_item_purchase_count_28d
user_item_recency_days
variant_rank
from_variant
item_popularity_7d
item_popularity_28d
category_affinity
```

Selected settings: LambdaRank, 40 trees, 31 leaves, learning rate 0.05, minimum child samples 50, L2 regularization 1.0. Parameters and iteration count were selected by validation MAP@12.

Two neural retrieval approaches were also evaluated:

- ID-based: separate customer and product embeddings.
- Feature-based: item ID, product type, and color; customer vectors are built from up to 20 historical purchase events. Training histories exclude the target day. Negative samples exclude products purchased in the historical snapshot.

The integrated experiment combines the existing first 125 candidates with the first 25 neural candidates, deduplicates, and backfills to 150. Separate five-epoch models were trained for each ranking snapshot, using transactions before its target week. V7 adds neural similarity and retrieval rank as features. It was not selected because validation performance was lower than V6.

## Project layout

```text
run.py                         # Compatible root command entry point
scripts/run.py                 # Explicit command-to-module routing
src/hm_recommender/
  data/                        # Original inspection and snapshot preparation
  baselines/                   # Popularity and recent-purchase rules
  candidates/                  # Co-visitation neighbors
  ranking/                     # LightGBM training, tuning, integration
  retrieval/                   # ID-based and feature-based two-tower experiments
  evaluation/                  # Comparisons, archived results, historical CLI
  paths.py                     # Shared paths and Unicode-safe model I/O
  provenance.py                # Existing source-relocation checker
reports/metrics/               # Unmodified aggregate historical records
requirements.txt               # Core packages; not a historical lock
requirements-neural.txt        # Optional neural packages
```

Local `data/` holds raw data, processed datasets, models, and caches. Only root `/data/` is Git-ignored; the source package `src/hm_recommender/data/` is tracked. Source modules are tracked separately from local data and model artifacts.

## Data and artifacts

Download data from the [H&M Personalized Fashion Recommendations competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and place CSVs in `data/raw/hm/`. The final pipeline uses `transactions_train.csv` and `articles.csv`; the optional inspection command also opens `customers.csv`.

Models and customer-level outputs remain excluded from Git. Important local paths:

- `data/processed/ranker_v6_tuned.txt`: original frozen selected model.
- `data/processed/final_test_predictions.parquet`: historical recommendation cache.
- `data/processed/final_test_actual.parquet`: historical customer target lists.
- `data/two_tower/2020-09-09/`: ID-based retrieval artifacts.
- `data/two_tower_features/<snapshot-date>/`: feature-based retrieval artifacts.

| Recorded file | Contents |
|---|---|
| `tuning_v6.csv`, `tuning_v6_best.json` | Parameter comparisons and selected settings |
| `two_tower_id_experiments.csv`, `two_tower_id_training.csv` | ID-based retrieval experiments and training history |
| `feature_tower_retrieval.csv`, `feature_tower_training_<date>.csv` | Feature-based comparisons and training histories |
| `integration_v7.csv`, `integration_v7_config.json`, `integration_v7_importance.csv` | Integration results, settings, feature importance |
| `multiweek_v5.csv`, `multiweek_v6.csv`, `validation_v4.csv` | Earlier ranker comparisons |
| `history_comparison.csv`, `source_comparison.csv`, `tuned_segments.csv` | Validation analysis |
| `two_tower_v1_retrieval.csv` | Original ID-tower coverage comparison |
| `final_selection.json` | Frozen selection, original hashes, evaluation protocol |
| `final_test.csv` | Overall and segment-level test metrics |
| `final_test_audit.json` | Candidate coverage and original recorded checks |

Hashes in `final_selection.json` belong to the original source and model. Import relocation and comments change current source hashes without changing archived provenance. Preparation modules come from the original Git history at `cdf6995`.
The required frozen ranker SHA-256 is
`61b630395022068798c8206fdc4bf89ec96a21cc7569d00d654da0db7a7609b7`.

## Setup

Reading the archive needs only Python. For the core offline pipeline, create an environment and install:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Neural experiments additionally require PyTorch:

```bash
python -m pip install -r requirements-neural.txt
```

Install a PyTorch build suitable for your GPU. The historical experiments used PyTorch 2.6.0+cu124 on an NVIDIA GeForce RTX 3050 Laptop GPU. The neural training commands retain their original CUDA requirement.

## Usage

Run from the repository root. Root `run.py` remains compatible; `python scripts/run.py ...` reaches the same dispatcher. The dispatcher resolves the repository path even when the terminal starts elsewhere.

### Read archived results from a fresh clone

```bash
python run.py archive_results
python run.py archive_results --json
python run.py --help
```

The archive command reads the recorded CSV and JSON files without importing Polars, LightGBM, or PyTorch. JSON metric values stay exact CSV strings. It reports current file hashes; it does not execute a model, verify absent artifacts, or rerun the experiment.

### Reproduce historical training when data and environment are available

```bash
python run.py ranking_dataset
python run.py train_multiweek
python run.py build_v6
python run.py train_v6
python run.py tune_ranker
```

These commands build training snapshots and perform the validation-based parameter search. They write local artifacts and experiment reports, so use a separate reproduction checkout if preserving the distributed archive. They do not tune on the test week. Retraining can produce a different model hash and does not replace the original frozen selection automatically.

The four preparation modules missing from the published reorganization were recovered from Git history. The source now contains them; raw data and model artifacts are still needed for full execution.

### Validate and display the original archive with the frozen model

```bash
python run.py finish_project evaluate
```

This existing command verifies the original model hash and test settings, then displays the saved report without reevaluation. It requires the exact original model, `final_selection.json`, and `final_test.csv`. It loads the ML packages, unlike the new archive-only command.

### Generate historical recommendations

```bash
python run.py finish_project recommend
python run.py finish_project recommend --customer-id YOUR_CUSTOMER_ID
```

The first form reads an example from the local prediction cache. The second uses a cached result when available; otherwise it rebuilds candidates from raw data and scores them with the original frozen model. Both require that model. Historical cutoff: **September 16, 2020**.

Customers without recent history receive popularity candidates ranked with the available item features. This is an offline demonstration, not a live retail service.

This repository has no HTTP server. Recommendation JSON contains `as_of`,
`customer_id` and twelve ordered article IDs in `predictions`.

If the cache is missing, the message “Run evaluate first” does not create one:
`evaluate` only reads recorded reports. Supply the original prediction cache,
or a customer ID together with the frozen model and raw data. Missing or invalid
assets may produce CLI errors or tracebacks. Rescoring recorded final predictions
also requires `data/processed/final_test_actual.parquet`.

## Evaluation protocol

| Split | Target period | Customers |
|---|---|---:|
| Training | Three weeks starting August 19, August 26, and September 2, 2020 | 5,000 per week |
| Validation | September 9–15, 2020 | 10,000 |
| Original final test | September 16–22, 2020 | 10,000 |

Sampling sorts SHA256 hashes of customer IDs, then the IDs, separately among customers purchasing in each target period. Candidates and features use only transactions before the prediction date. Customers with zero relevant items in their candidate pool remain in evaluation.

Training has 2,250,000 rows across 15,000 customer-week queries. A customer may appear in several weeks. The selected model was frozen before the original test and evaluated without refitting or further parameter tuning.

- **MAP@12**, the selection metric: average per-customer AP@12. AP uses unique actual purchases, ignores duplicate predicted hits, and divides by `min(number of unique actual purchases, 12)`.
- **Recall@12**: fraction of unique actual purchases retrieved, averaged per customer.
- **HitRate@12**: fraction of customers with at least one correct recommendation.
- **Candidate Recall@150**: coverage before ranking; it does not measure top-12 order quality.

“Held out” describes the original experiment. These results have since been inspected; reusing this test week does not create a new unseen evaluation.

## Results and interpretation

### Recorded validation results

All methods use the same 10,000 validation customers.

| Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Variant rule | 0.057929 | 0.112300 | 0.025981 |
| Tuned LightGBM V6 | 0.056209 | 0.108100 | 0.026760 |
| Tuned V6 with two-tower candidates | 0.056209 | 0.108100 | 0.026762 |
| Retrained ranker with two-tower features | 0.048978 | 0.094900 | 0.023365 |

The variant rule orders recent purchases, variants, co-visitation, and popularity without a learned ranker. Tuned V6 with its original sources was selected before the original test. Neural additions produced a negligible MAP@12 difference, so they were excluded from the final selection.

| Candidate pool | Validation Recall@150 |
|---|---:|
| Original V6 candidates | 0.175006 |
| V6 combined with feature-based two-tower candidates | 0.176286 |

The small increase in candidate coverage did not establish a useful ranking improvement. Lower neural training loss also did not reliably identify the best retrieval checkpoint.

### Recorded final test results

| Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Popularity | 0.026280 | 0.068000 | 0.008686 |
| Variant rule | 0.061043 | 0.112600 | 0.028687 |
| Frozen tuned V6 | 0.054247 | 0.099200 | 0.027216 |

The rule won on this test week. The original selection was retained; there was no retrospective choice of the test winner as the selected model.

| Coverage metric | Value |
|---|---:|
| Candidate Recall@150 | 0.188256 |
| Candidate HitRate@150 | 0.357700 |

### Recorded results by recent purchase history

| Customer group | Customers | Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---|---:|---:|---:|
| `history_28d` | 4,514 | Variant rule | 0.101285 | 0.167036 | 0.052276 |
| `history_28d` | 4,514 | Frozen V6 | 0.100800 | 0.171910 | 0.052982 |
| `no_history_28d` | 5,486 | Variant rule | 0.027932 | 0.067809 | 0.009278 |
| `no_history_28d` | 5,486 | Frozen V6 | 0.015943 | 0.039373 | 0.006016 |

V6 slightly improved MAP and HitRate for customers with recent history. Its weaker no-recent-history performance explains the overall disadvantage. This is an error-analysis finding, not a routing policy tuned on the test set. Absence of 28-day history alone does not establish true new-user cold start.
