"""Verify the documented source relocation without reopening the held-out test."""
import hashlib
import json


def verify_source_relocation(previous, current, reports):
    path = reports / "source_layout_migration.json"
    report = reports / "final_test.csv"
    if not path.is_file() or not report.is_file():
        return False
    migration = json.loads(path.read_text(encoding="utf-8"))
    old_protocol = {key: value for key, value in previous.items() if key != "source_sha256"}
    new_protocol = {key: value for key, value in current.items() if key != "source_sha256"}
    return (
        old_protocol == new_protocol
        and migration["previous_source_sha256"] == previous["source_sha256"]
        and migration["source_sha256"] == current["source_sha256"]
        and migration["model_sha256"] == current["model_sha256"]
        and migration["test_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    )
