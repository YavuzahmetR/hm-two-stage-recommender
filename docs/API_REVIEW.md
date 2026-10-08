# CLI outputs and verification

Updated: **2026-10-09**. This repository provides an offline CLI. It has no FastAPI application, HTTP endpoint or API response model.

## Read archived results without model/data downloads

From the repository root:

```bash
python run.py --help
python run.py archive_results
python run.py archive_results --json
```

This path uses standard Python. JSON fields:

| Field | Meaning |
|---|---|
| `status` | `historical_recorded_not_recomputed`: reads the archive without running a new experiment |
| `selection` | Original model selection/protocol from `final_selection.json` |
| `audit` | Historical candidate coverage and recorded checks from `final_test_audit.json` |
| `metrics` | `final_test.csv` rows; values retain CSV precision as strings |
| `current_file_sha256` | Hashes of the three report files read; not proof of scientific validity |

Historical `checks_passed` does not mean the command reruns those checks. README tables can be compared with source reports; actual inference requires the assets below.

## Frozen model and cached recommendations

Install core dependencies and supply:

```text
data/processed/ranker_v6_tuned.txt
data/processed/final_test_predictions.parquet
```

The first file is the model; the second contains recorded 12-item recommendations. Reports are included under `reports/metrics/`. Model SHA-256 must match `final_selection.json`:
`61b630395022068798c8206fdc4bf89ec96a21cc7569d00d654da0db7a7609b7`.
The expected eleven features/order and forty trees are also checked.

```bash
python run.py finish_project evaluate
python run.py finish_project recommend
python run.py finish_project recommend --customer-id YOUR_CUSTOMER_ID
```

`evaluate` validates the original model/settings and shows recorded reports; it does not reproduce the final test. Argument-free `recommend` uses the first cached customer. An explicit customer is looked up in the cache first. JSON contains `as_of`, `customer_id` and ordered `predictions` with twelve article IDs. The cutoff is the historical `2020-09-16`; IDs retain string representation.

## Inference outside the cache

In addition to the frozen model, supply:

```text
data/raw/hm/transactions_train.csv
data/raw/hm/articles.csv
```

Candidates are generated from transactions before the cutoff and ranked into the same JSON format. Customers without recent history receive popularity-based candidates. Reading raw CSVs and generating candidates per invocation is not an HTTP latency/capacity benchmark.

## Known limitations

- The model, raw data and customer caches are not distributed. Archive reading works independently; recommendations require the specified assets.
- Missing/invalid files produce CLI errors or tracebacks; there are no HTTP status codes or standardized HTTP error responses.
- Without a cache or customer ID, existing guidance says “Run evaluate first”. That command does not build a cache. Supply the original cache, or raw data plus a customer ID and model.
- Rescoring historical final predictions also requires `data/processed/final_test_actual.parquet`. This is a historical-output check, not a new unseen test.
- An HTTP integration would require a separate interface and input/output contract.

Namespace/import, CLI, preserved-report and real-artifact comparisons are recorded in [CLEANUP_RESULTS.json](CLEANUP_RESULTS.json) and [VERIFICATION_RESULTS.json](VERIFICATION_RESULTS.json).
