# H&M Two-Stage Recommender

An offline fashion recommendation project using H&M purchase history and product metadata.

The system generates 150 candidate products per customer, then ranks them with LightGBM to produce 12 recommendations. The project also includes ID-based and feature-based two-tower retrieval experiments.

## Pipeline

```text
Recent purchases + product variants + co-visitation + popularity
                              |
                    150 unique candidates
                              |
                    LightGBM LambdaRank
                              |
                      12 recommendations
```

Candidate sources:

- **Recent purchases:** up to 12 distinct products purchased in the previous 28 days.
- **Product variants:** up to 20 products sharing a product code with recent purchases.
- **Co-visitation:** up to 50 related products from customer-day baskets in the previous week.
- **Popularity:** popular products from the previous seven days fill the remaining positions.

Candidates are deduplicated before ranking.

## Data

The dataset comes from the [H&M Personalized Fashion Recommendations competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Place the downloaded CSV files under:

```text
data/raw/hm/
```

The final system uses `transactions_train.csv` and `articles.csv`. Customer profile fields and product images are not used.

Raw data, model files, and customer-level outputs are excluded from Git.

## Evaluation setup

| Split | Target period | Customers |
|---|---|---:|
| Training | Three weeks starting August 19, August 26, and September 2, 2020 | 5,000 per week |
| Validation | September 9–15, 2020 | 10,000 |
| Test | September 16–22, 2020 | 10,000 |

Customers are sampled deterministically by sorting the SHA256 hash of their IDs. Sampling is performed separately among customers who purchased during each target period.

Candidates and features use only transactions before the prediction date. Customers with no relevant candidates remain in the evaluation.

Training contains 2,250,000 rows grouped into 15,000 customer-week queries.

The main metric is **MAP@12**. Recall@12 and HitRate@12 are also reported:

- **Recall@12:** the fraction of actual purchases retrieved, averaged per customer.
- **HitRate@12:** the fraction of customers with at least one correct recommendation.
- **MAP@12:** rewards correct products appearing earlier in the recommendation list.
- **Candidate Recall@150:** measures coverage before ranking.

AP@12 uses unique actual purchases and divides by the smaller of the number of actual purchases and 12.

## Ranker

The selected LightGBM model uses 11 features:

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

Selected settings:

- Objective: LambdaRank
- Trees: 40
- Leaves: 31
- Learning rate: 0.05
- Minimum child samples: 50
- L2 regularization: 1.0

Parameters and the number of iterations were selected using validation MAP@12.

## Validation results

All methods below use the same 10,000 validation customers.

| Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Variant rule | 0.057929 | 0.112300 | 0.025981 |
| Tuned LightGBM V6 | 0.056209 | 0.108100 | 0.026760 |
| Tuned V6 with two-tower candidates | 0.056209 | 0.108100 | 0.026762 |
| Retrained ranker with two-tower features | 0.048978 | 0.094900 | 0.023365 |

I selected **tuned V6 with its original candidate sources** before evaluating the test week.

Adding neural candidates produced a negligible change in MAP@12. The additional retrieval pipeline was therefore not included in the final selection.

## Held-out test results

The selected model was frozen and evaluated without refitting or further parameter tuning.

| Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Popularity | 0.026280 | 0.068000 | 0.008686 |
| Variant rule | 0.061043 | 0.112600 | 0.028687 |
| Frozen tuned V6 | 0.054247 | 0.099200 | 0.027216 |

The rule-based method outperformed the selected ranker on this test week. The final report keeps that result rather than selecting a different method after inspecting the test.

Candidate coverage:

| Metric | Value |
|---|---:|
| Candidate Recall@150 | 0.188256 |
| Candidate HitRate@150 | 0.357700 |

### Results by recent purchase history

“Without history” means no purchases in the previous 28 days, not necessarily a new customer.

| Customer group | Customers | Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---|---:|---:|---:|
| With history | 4,514 | Variant rule | 0.101285 | 0.167036 | 0.052276 |
| With history | 4,514 | Frozen V6 | 0.100800 | 0.171910 | 0.052982 |
| Without history | 5,486 | Variant rule | 0.027932 | 0.067809 | 0.009278 |
| Without history | 5,486 | Frozen V6 | 0.015943 | 0.039373 | 0.006016 |

The ranker slightly improved MAP and HitRate for customers with recent history. Its weaker results for customers without recent history account for the overall disadvantage against the rule-based method.

This observation is an error-analysis finding, not a new policy tuned on the test set.

## Two-tower experiments

Two neural retrieval approaches were evaluated:

1. **ID-based:** separate customer and product embeddings.
2. **Feature-based:** product ID, product type, and color representations, with customer representations built from historical purchases.

The feature-based model uses up to 20 historical purchase events. Training histories exclude the target day. Previously purchased products remain eligible recommendations because repeat purchases are valid targets.

Negative samples exclude products the customer purchased within the historical snapshot.

The feature-based retrieval experiment improved candidate coverage slightly:

| Candidate pool | Validation Recall@150 |
|---|---:|
| Original V6 candidates | 0.175006 |
| V6 combined with feature-based two-tower candidates | 0.176286 |

The combined pool used the existing first 125 candidates and the first 25 neural candidates, followed by deduplication and backfilling to 150.

Separate five-epoch models were trained for each historical ranking snapshot. Their training data ended before the corresponding target week.

The integrated V7 ranker used the neural similarity score and retrieval rank as additional features. Its validation performance was lower than V6, so it was not selected.

## Run the project

The core pipeline uses Python, NumPy, Polars, and LightGBM.

```bash
python -m pip install numpy polars lightgbm scikit-learn
```

The neural experiments additionally require PyTorch. They were run with PyTorch 2.6.0+cu124 on an NVIDIA GeForce RTX 3050 Laptop GPU.

Run commands from the repository root.

### Reproduce the selected ranker

```bash
python ranking_dataset.py
python train_multiweek.py
python build_v6.py
python train_v6.py
python tune_ranker.py
```

These scripts prepare the training snapshots and repeat the validation-based parameter search. They do not tune against the test week.

### Evaluate the frozen model

```bash
python finish_project.py evaluate
```

The script records the model and source hashes before evaluation. Subsequent runs reuse the saved report and reject changes to the frozen selection.

### Generate recommendations

Display an example from the saved predictions:

```bash
python finish_project.py recommend
```

Generate recommendations for a specific customer:

```bash
python finish_project.py recommend --customer-id YOUR_CUSTOMER_ID
```

The command uses the historical cutoff **September 16, 2020**. It is an offline demonstration, not a live retail service.

Cached customers return immediately. Other requests rebuild historical candidates and score them with the frozen model. Customers without recent history receive popularity-based candidates ranked using available item features.

## Reports

Aggregate experiment outputs are stored in `reports/metrics/`.

| File | Contents |
|---|---|
| `tuning_v6.csv` | LightGBM parameter comparisons |
| `two_tower_id_experiments.csv` | ID-based retrieval experiments |
| `feature_tower_retrieval.csv` | Feature-based retrieval comparisons |
| `integration_v7.csv` | Ranker and two-tower integration results |
| `final_selection.json` | Frozen model hash and evaluation protocol |
| `final_test.csv` | Overall and segment-level test metrics |
| `final_test_audit.json` | Candidate coverage and evaluation checks |

Model artifacts and customer-level predictions remain under `data/`.

## Lessons and limitations

- Better candidate recall did not guarantee better top-12 recommendations.
- A lower neural training loss did not reliably identify the best retrieval checkpoint.
- Rule-based recommendations remained competitive and won on the final test week.
- Recent-history and no-history customers behaved differently.
- Validation was used repeatedly during development, so its best score is not an independent estimate.
- Evaluation covers sampled active purchasers and one final week, not all customers or an online setting.
- Product metadata is treated as static.
- Dependency versions are not fully locked.
- Checks cover metric examples, historical input boundaries, candidate uniqueness, recency bounds, and prediction validity. They are not an exhaustive test suite.
- FastAPI, live deployment, and image-based retrieval are outside this version's scope.

Image-based cold-start retrieval is planned as a separate project using the same article IDs.