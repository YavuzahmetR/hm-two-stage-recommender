"""Read recorded aggregate results without loading models or rerunning evaluation."""
import argparse
import csv
import json

from hm_recommender.paths import ROOT


REPORT_DIR = ROOT / "reports" / "metrics"


def read_archived_results(report_dir=REPORT_DIR):
    """Return recorded results and protocol; metric strings stay exact."""
    report_path = report_dir / "final_test.csv"
    with report_path.open(encoding="utf-8", newline="") as file:
        metrics = list(csv.DictReader(file))

    selection = json.loads(
        (report_dir / "final_selection.json").read_text(encoding="utf-8")
    )
    # File identity is needed by the model loader, not by the results summary.
    selection.pop("model_sha256", None)
    audit = json.loads(
        (report_dir / "final_test_audit.json").read_text(encoding="utf-8")
    )

    return {
        "status": "historical_recorded_not_recomputed",
        "selection": selection,
        "audit": audit,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Show exact records as JSON.")
    args = parser.parse_args()
    archive = read_archived_results()

    if args.json:
        print(json.dumps(archive, indent=2))
        return

    print("Archived final test results; no model execution or reevaluation.")
    print((REPORT_DIR / "final_test.csv").read_text(encoding="utf-8"), end="")
    print("\nRecorded candidate coverage:")
    print(json.dumps(archive["audit"], indent=2))


if __name__ == "__main__":
    main()
