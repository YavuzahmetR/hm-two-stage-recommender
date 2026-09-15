# H&M Two-Stage Recommender

Learning project for fashion recommendations using H&M transaction data.

## Current progress

- Transaction schema and date inspection
- Recent-popularity baseline
- Repeat-purchase baseline with popularity backfill
- Co-visitation candidate generation
- Recall@12, HitRate@12, MAP@12, and candidate recall evaluation
- Candidate feature rows for a future ranking model

## Validation results

Validation period: September 9–15, 2020.
Candidates and features use only earlier transactions.

| Recommendation method | Recall@12 | HitRate@12 | MAP@12 |
|---|---:|---:|---:|
| Recent popularity | 0.020905 | 0.056513 | 0.006604 |
| Repeat + popularity | 0.048607 | 0.100362 | 0.024508 |

| Candidate sources | Recall@150 |
|---|---:|
| Repeat + popularity | 0.160765 |
| Repeat + co-visitation + popularity | 0.171625 |

Results are from local runs on 72,019 customers with purchases
during the validation week. They are not final test results.

## Data

Download the CSV files from:
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

Place them under `data/raw/hm/`. Raw data is excluded from Git.

## Status

Work in progress. LightGBM ranking, two-tower retrieval, and FastAPI
serving are not implemented yet.