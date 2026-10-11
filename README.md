# H&M Two-Stage Recommender

An offline recommendation project: use a customer's past purchases to suggest
12 clothing products. The first stage finds 150 possible products; the second
stage uses LightGBM to rank them. Simple rules provide the comparison baselines.

The main finding: the trained ranker improved validation MAP@12, but the simpler
variant rule performed better on the final test week. Neural retrieval was also
tried and did not provide a useful improvement, so it is outside the final pipeline.

## How it works

```text
Past purchases + product information
                |
150 candidate products: repeats, variants, co-visitation, popularity
                |
LightGBM ranks the candidates
                |
12 ordered product IDs per customer
```

- **Repeats:** products bought in the previous 28 days, up to 12.
- **Variants:** products sharing a product code with those purchases, up to 20.
- **Co-visitation:** products bought together in customer-day baskets, up to 50.
- **Popularity:** products popular in the previous seven days fill the remaining slots.

Duplicates are removed while preserving source order. The ranker uses 11 features
describing these sources, purchase counts, recency and product-category affinity.
Previous purchases remain eligible because customers can buy the same item again.

## Results

These are recorded offline results on 10,000 purchasers in each target week.
They are not an online sales result or a Kaggle leaderboard score.

| Method | Validation MAP@12 | Final test MAP@12 |
|---|---:|---:|
| Variant rule, without a trained ranker | 0.025981 | **0.028687** |
| Selected LightGBM ranker (V6) | **0.026760** | 0.027216 |
| LightGBM with neural candidates | 0.026762 | Not evaluated |
| Retrained ranker with neural features | 0.023365 | Not evaluated |

MAP@12 rewards relevant products near the top of the list and averages this over
customers. These values are ranking scores, not classification accuracy.
The V6 model was selected on validation and frozen before the final test.
We retain that original choice rather than choose a new winner after seeing test results.

See [results and evaluation conditions](reports/RESULTS.md) for the time split,
Recall/HitRate, customer groups and rejected experiments.

## Try the recorded results

This works with Python alone, immediately after cloning; no dataset or model is needed:

```bash
python run.py archive_results
```

It prints the recorded final scores for popularity, the variant rule and LightGBM,
including customer groups and candidate coverage. The table above is a rounded
preview. This does not run inference. `python run.py --help` lists the commands.

## Run with data and a model

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On Linux/macOS, use `.venv/bin/python` instead. Download the
[H&M competition data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
and put `transactions_train.csv` and `articles.csv` in `data/raw/hm/`.
The inspection command additionally needs `customers.csv`.

The historical demo needs the original selected model at
`data/processed/ranker_v6_tuned.txt`. With that model and the raw data, run:

```bash
.venv\Scripts\python run.py finish_project recommend --customer-id YOUR_CUSTOMER_ID
```

It prints JSON containing the customer ID, cutoff date `2020-09-16` and twelve
ordered article IDs. Without a customer ID, it reads an example from the local
`data/processed/final_test_predictions.parquet` cache. Those large artifacts are
not supplied in Git; a fresh clone can read the recorded results but cannot perform
this historical model demo. There is no web app or HTTP API in this repository.

To reproduce training: run `ranking_dataset`, `train_multiweek`, `build_v6`,
`train_v6`, then `tune_ranker` through `python run.py COMMAND`. These commands
require the raw data and write local models and reports. A retrained model is a new
run; it is not automatically accepted as the original frozen model.
Historical dependency versions were not fully recorded.

## Project layout

```text
run.py, scripts/run.py       command entry points
src/hm_recommender/          data, candidates, ranking and evaluation code
  retrieval/                optional two-tower experiments
reports/RESULTS.md          explained results
reports/metrics/            three files used by the recorded-result commands
requirements.txt           packages for the LightGBM pipeline
requirements-neural.txt    optional PyTorch packages; not needed by the final model
```

The optional neural commands require a CUDA-compatible PyTorch installation and
`python -m pip install -r requirements-neural.txt`. They learn customer/product
vectors to propose candidates; they do not replace the selected final ranker.

## Limits

Only sampled active purchasers and one final test week were evaluated. A customer
with no purchases in the previous 28 days is not necessarily a new customer.
Validation was used repeatedly, and the historical test has since been inspected.
Customer profiles and images are not used; image retrieval is a separate project.
