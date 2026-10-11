# Results and evaluation

These are recorded historical measurements. Reading this file or running
`archive_results` does not rerun training or evaluation. Scores below are rounded
to six decimals; exact final values remain in `metrics/final_test.csv`.

## Evaluation conditions

| Split | Target purchases | Sample size |
|---|---|---:|
| Training | Three weeks starting August 19, August 26 and September 2, 2020 | 5,000 customer-week queries per week |
| Validation | September 9–15, 2020 | 10,000 customers |
| Final test | September 16–22, 2020 | 10,000 customers |

Customers are selected by a fixed deterministic ordering of IDs among purchasers
in each target period. The training dataset has 2,250,000 rows: 150 candidates for
each of 15,000 customer-week queries. A customer can appear in more than one week.
Candidates and features use transactions before the prediction date. Product
metadata is treated as static. Customers with no relevant candidate stay in evaluation.

- **MAP@12:** average customer AP@12; relevant hits earlier in the list contribute more.
  AP divides by `min(unique actual purchases, 12)` and ignores duplicate predicted hits.
- **Recall@12:** fraction of unique actual purchases recovered, averaged over customers.
- **HitRate@12:** fraction of customers with at least one correct suggestion.
- **Candidate Recall@150:** coverage of the first-stage pool, before ranking.

All are fractions, not percentages. They describe a sampled offline task.
The historical test was held out for the original selection; it is no longer unseen.

## Validation decisions

| Candidate-source experiment | Candidate Recall@150 | Rule MAP@12 |
|---|---:|---:|
| Original sources, 28-day history | 0.171662 | 0.025604 |
| Original sources, 90-day history | 0.171835 | 0.024750 |
| Add category candidates | 0.172135 | 0.023896 |
| Add product variants | 0.175006 | 0.025981 |
| Add category and variant candidates | 0.175443 | 0.023879 |

The 28-day history and variant candidates were retained. Higher candidate coverage
alone did not imply better ordering of the final twelve recommendations.

| Ranker or rule | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Popularity baseline | 0.020860 | 0.056300 | 0.006018 |
| Recent purchases + popularity | 0.048944 | 0.101300 | 0.023633 |
| Initial LightGBM (V4) | 0.048546 | 0.095800 | 0.022481 |
| Multiweek LightGBM (V5) | 0.050099 | 0.094400 | 0.023116 |
| LightGBM with variants (V6), before tuning | 0.055300 | 0.107300 | 0.025326 |
| Variant rule | 0.057929 | 0.112300 | 0.025981 |
| Selected tuned V6 | 0.056209 | 0.108100 | 0.026760 |
| Selected V6 + neural candidates | 0.056209 | 0.108100 | 0.026762 |
| Retrained V7 + neural features | 0.048978 | 0.094900 | 0.023365 |

Earlier models use earlier candidate designs; this table is the development path,
not a comparison isolating the effect of one ranker parameter. V5 and V6 learn from
three weekly snapshots rather than one. V6 tuning selected 40 trees, 31 leaves,
learning rate 0.05, minimum child samples 50 and L2 regularization 1.0, with seed 42.
The 11 features represent source ranks/scores, 28-day purchase count and recency,
variant membership, 7/28-day item popularity and customer category affinity.

## Why neural retrieval was not selected

The ID-based two-tower learns separate customer and product embeddings. Its best
recorded combined candidate coverage was 0.175171, using epoch 15 and ten neural
candidates. The feature-based tower also uses product type/color; a customer vector
uses up to twenty past purchase events. Histories exclude the target day, and
negative sampling excludes historical purchases.

The feature-based epoch-five experiment combined the first 125 existing candidates
with 25 neural candidates, then deduplicated and backfilled to 150. It reached
Recall@150 of 0.176286 versus 0.175006. Separate tower models were trained before
each ranking snapshot. Adding candidates barely changed V6 MAP@12; adding tower
similarity/rank and retraining V7 made it worse. Lower neural loss did not reliably
identify the best retrieval checkpoint. The final model therefore uses no tower.

## Final test

| Method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Popularity | 0.026280 | 0.068000 | 0.008686 |
| Variant rule | **0.061043** | **0.112600** | **0.028687** |
| Frozen tuned V6 | 0.054247 | 0.099200 | 0.027216 |

The test pool has 1,500,000 candidate rows, Candidate Recall@150 of 0.188256 and
Candidate HitRate@150 of 0.357700. The selected V6 was not refitted or retuned on
the test week. The rule's win is reported without replacing that original choice.

| Customer group | Customers | Variant-rule MAP@12 | V6 MAP@12 |
|---|---:|---:|---:|
| Purchases in previous 28 days | 4,514 | 0.052276 | **0.052982** |
| No purchases in previous 28 days | 5,486 | **0.009278** | 0.006016 |

V6's weaker performance without recent purchase history explains the overall gap.
This is an observed failure case, not a hybrid routing rule chosen after testing.
No recent history does not establish true new-customer cold start.

## What the three data files do

- `metrics/final_test.csv`: exact overall and customer-group scores.
- `metrics/final_test_audit.json`: customer count and first-stage candidate coverage.
- `metrics/final_selection.json`: the selected model's feature order, time split and
  file identity; the loader uses it to reject a different model for this historical demo.

Training commands can generate their own detailed experiment CSVs. They are local
outputs; the public result summary focuses on the decisions and final measurements.
