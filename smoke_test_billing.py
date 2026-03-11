import argparse
import json
import os
import sys
from datetime import datetime

import requests

from smoke_test_generated_registry import run_generated_registry_smoke_tests
from smoke_test_registry_rules import run_registry_smoke_tests


def _post_upload(base_url: str, file_path: str) -> dict:
    endpoint = f"{base_url.rstrip('/')}/upload_csv"
    dataset_name = f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"dataset_name": dataset_name}
        res = requests.post(endpoint, files=files, data=data, timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"Upload failed ({res.status_code}): {res.text}")
    body = res.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise RuntimeError(f"Upload succeeded but dataset_id missing: {json.dumps(body)[:1000]}")
    return body


def _ask_sql(base_url: str, dataset_id: str, question: str, analyze_all_rows: bool = False, max_rows: int = 1000) -> dict:
    endpoint = f"{base_url.rstrip('/')}/ask_sql"
    payload = {
        "dataset_id": dataset_id,
        "question": question,
        "analyze_all_rows": analyze_all_rows,
        "max_rows": max_rows,
    }
    res = requests.post(endpoint, json=payload, timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"ask_sql failed ({res.status_code}) for '{question}': {res.text}")
    return res.json()


def _to_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).strip().replace(",", "")
    if not txt:
        return 0.0
    return float(txt)


def _get_event_ids(rows: list[dict]) -> set[str]:
    out = set()
    for r in rows:
        v = r.get("event_id")
        if v is None:
            continue
        out.add(str(v).strip())
    return out


def run_smoke_tests(base_url: str, file_path: str) -> int:
    print(f"Using API: {base_url}")
    print(f"Using workbook: {file_path}")

    upload_body = _post_upload(base_url, file_path)
    dataset_id = str(upload_body["dataset_id"])
    print(f"Uploaded dataset_id: {dataset_id}")
    print(
        "Upload generated rules: "
        f"count={int(upload_body.get('generated_registry_rule_count') or 0)}"
    )
    upload_qa = upload_body.get("upload_qa") or {}
    if upload_qa.get("status") != "passed":
        print("\nFAILURES:")
        print(
            "- Upload QA failed before smoke checks could proceed: "
            f"status={upload_qa.get('status')} failures={json.dumps(upload_qa.get('failures') or [])[:800]}"
        )
        return 1
    print(
        "Upload QA: "
        f"generated_checks={int(((upload_qa.get('generated_registry') or {}).get('checked_rule_count') or 0))} "
        f"curated_checks={int(((upload_qa.get('curated_registry') or {}).get('checked_rule_count') or 0))}"
    )

    failures: list[str] = []

    # 1) Uninvoiced completed operational events
    q1 = "Which completed operational events have not yet been invoiced?"
    r1 = _ask_sql(base_url, dataset_id, q1)
    rows1 = r1.get("rows") or []
    got_ids = _get_event_ids(rows1)
    expected_ids = {"EVT-1002", "EVT-1006", "EVT-1007", "EVT-1010"}
    if got_ids != expected_ids:
        failures.append(
            "Test1 failed: uninvoiced event IDs mismatch. "
            f"expected={sorted(expected_ids)} got={sorted(got_ids)}"
        )
    else:
        print("PASS Test1: uninvoiced completed events")

    # 2) Total unbilled value for completed operational events
    q2 = "What is the total value of completed operational events that are not invoiced?"
    r2 = _ask_sql(base_url, dataset_id, q2, analyze_all_rows=True)
    rows2 = r2.get("rows") or []
    total_value = 0.0
    if rows2:
        first = rows2[0]
        total_value = _to_float(first.get("total_value")) or _to_float(first.get("total_unbilled_value"))
    if abs(total_value - 1655.0) > 0.0001:
        failures.append(f"Test2 failed: total_value expected=1655 got={total_value}")
    else:
        print("PASS Test2: unbilled total value")

    # 3) Invoiced completed operational events should return invoice_amount values
    q3 = "Give me completed operational events that are invoiced with invoice amount"
    r3 = _ask_sql(base_url, dataset_id, q3)
    rows3 = r3.get("rows") or []
    if not rows3:
        failures.append("Test3 failed: invoiced events query returned 0 rows")
    else:
        bad_rows = [r for r in rows3 if _to_float(r.get("invoice_amount")) <= 0]
        if bad_rows:
            failures.append(
                "Test3 failed: one or more rows have non-positive invoice_amount. "
                f"sample_bad_row={json.dumps(bad_rows[0])[:400]}"
            )
        else:
            print("PASS Test3: invoiced events include invoice_amount")

    print("\n--- Registry route checks ---")
    registry_exit_code = run_registry_smoke_tests(base_url, dataset_id)
    if registry_exit_code != 0:
        failures.append(f"Registry smoke tests failed with exit code {registry_exit_code}")

    print("\n--- Generated registry route checks ---")
    generated_registry_exit_code = run_generated_registry_smoke_tests(
        base_url,
        dataset_id=dataset_id,
        min_generated_rules=1,
    )
    if generated_registry_exit_code != 0:
        failures.append(
            f"Generated registry smoke tests failed with exit code {generated_registry_exit_code}"
        )

    print("\n--- SQL snapshots ---")
    print("Test1 SQL:")
    print(r1.get("sql", "<missing>"))
    print("\nTest2 SQL:")
    print(r2.get("sql", "<missing>"))
    print("\nTest3 SQL:")
    print(r3.get("sql", "<missing>"))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"- {f}")
        return 1

    print("\nAll smoke tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run billing SQL smoke tests against the local FastAPI backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8094", help="FastAPI base URL")
    parser.add_argument(
        "--file",
        default="Billing_Reconciliation_Sample_Datasets.xlsx",
        help="Path to test Excel workbook",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"Workbook not found: {file_path}")
        return 2

    try:
        return run_smoke_tests(args.base_url, file_path)
    except Exception as e:
        print(f"Smoke test execution failed: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
