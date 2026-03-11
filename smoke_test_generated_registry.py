import argparse
import json
import os
from datetime import datetime

import requests


def _get(base_url: str, path: str) -> object:
    res = requests.get(f"{base_url.rstrip('/')}/{path.lstrip('/')}", timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"GET {path} failed ({res.status_code}): {res.text}")
    return res.json()


def _post(base_url: str, path: str, payload: dict) -> dict:
    res = requests.post(f"{base_url.rstrip('/')}/{path.lstrip('/')}", json=payload, timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"POST {path} failed ({res.status_code}): {res.text}")
    return res.json()


def _post_upload(base_url: str, file_path: str) -> dict:
    endpoint = f"{base_url.rstrip('/')}/upload_csv"
    dataset_name = f"generated_registry_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(file_path, "rb") as handle:
        files = {"file": (os.path.basename(file_path), handle)}
        data = {"dataset_name": dataset_name}
        res = requests.post(endpoint, files=files, data=data, timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"Upload failed ({res.status_code}): {res.text}")
    body = res.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        raise RuntimeError(f"Upload succeeded but dataset_id missing: {json.dumps(body)[:1000]}")
    return body


def _find_dataset(base_url: str, dataset_id: str) -> dict:
    datasets = _get(base_url, "/datasets") or []
    for dataset in datasets:
        if str((dataset or {}).get("id")) == str(dataset_id):
            return dataset
    raise RuntimeError(f"Dataset not found in /datasets: {dataset_id}")


def _extract_generated_rules(dataset: dict) -> list[dict]:
    relationships = (dataset or {}).get("relationships") or {}
    rules = relationships.get("generated_registry_rules") or []
    return [rule for rule in rules if isinstance(rule, dict)]


def _ask_sql(base_url: str, dataset_id: str, question: str, max_rows: int = 1000) -> dict:
    return _post(
        base_url,
        "/ask_sql",
        {
            "dataset_id": dataset_id,
            "question": question,
            "max_rows": max_rows,
            "analyze_all_rows": False,
        },
    )


def run_generated_registry_smoke_tests(
    base_url: str,
    *,
    dataset_id: str | None = None,
    file_path: str | None = None,
    min_generated_rules: int = 1,
    max_cases: int | None = None,
) -> int:
    failures: list[str] = []
    upload_body: dict | None = None

    if not dataset_id:
        if not file_path:
            raise RuntimeError("Provide either --dataset-id or --file")
        upload_body = _post_upload(base_url, file_path)
        dataset_id = str(upload_body["dataset_id"])
        print(f"Uploaded dataset_id: {dataset_id}")

    dataset = _find_dataset(base_url, dataset_id)
    generated_rules = _extract_generated_rules(dataset)

    upload_generated_count = None
    if upload_body is not None:
        upload_generated_count = int(upload_body.get("generated_registry_rule_count") or 0)
        upload_questions = upload_body.get("generated_registry_questions") or []
        upload_qa = upload_body.get("upload_qa") or {}
        upload_qa_generated = upload_qa.get("generated_registry") or {}
        upload_qa_curated = upload_qa.get("curated_registry") or {}
        print(
            "Upload response: "
            f"generated_registry_rule_count={upload_generated_count} "
            f"generated_registry_questions={len(upload_questions)}"
        )
        if upload_qa.get("status") != "passed":
            failures.append(
                "Upload QA did not pass: "
                f"status={upload_qa.get('status')} failures={json.dumps(upload_qa.get('failures') or [])[:800]}"
            )
        else:
            print(
                "PASS Upload QA: "
                f"generated_checks={int(upload_qa_generated.get('checked_rule_count') or 0)} "
                f"curated_checks={int(upload_qa_curated.get('checked_rule_count') or 0)}"
            )

    persisted_count = len(generated_rules)
    if persisted_count < min_generated_rules:
        failures.append(
            f"Expected at least {min_generated_rules} generated registry rules, found {persisted_count}"
        )
    else:
        print(f"PASS Generated rules persisted: {persisted_count}")

    if upload_generated_count is not None and upload_generated_count != persisted_count:
        failures.append(
            "Upload response generated_registry_rule_count does not match persisted generated rule count: "
            f"upload={upload_generated_count} persisted={persisted_count}"
        )

    cases = generated_rules
    if max_cases is not None:
        cases = cases[: max(0, int(max_cases))]

    if not cases:
        if failures:
            print("\nFAILURES:")
            for failure in failures:
                print(f"- {failure}")
            return 1
        print("No generated rules to validate.")
        return 0

    for index, rule in enumerate(cases, start=1):
        sample_question = str(rule.get("sample_question") or "").strip()
        expected_intent = str(rule.get("intent") or "").strip()
        if not sample_question:
            failures.append(f"generated_rule_{index}: missing sample_question")
            continue
        if not expected_intent:
            failures.append(f"generated_rule_{index}: missing intent")
            continue

        response = _ask_sql(base_url, dataset_id, sample_question)
        route_source = response.get("route_source")
        route_intent = response.get("route_intent")
        sql = str(response.get("sql") or "").strip()
        rows_returned = int(response.get("rows_returned") or 0)

        if route_source != "dataset_generated_registry_sql_template":
            failures.append(
                f"generated_rule_{index}: expected route_source=dataset_generated_registry_sql_template got {route_source}"
            )
            continue
        if route_intent != expected_intent:
            failures.append(
                f"generated_rule_{index}: expected route_intent={expected_intent} got {route_intent}"
            )
            continue
        if not sql:
            failures.append(f"generated_rule_{index}: ask_sql returned empty sql")
            continue
        if "dataset_id" not in sql.lower():
            failures.append(f"generated_rule_{index}: returned sql is missing dataset_id filter")
            continue

        print(
            f"PASS generated_rule_{index}: intent={route_intent} rows={rows_returned} "
            f"question={json.dumps(sample_question)}"
        )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll generated registry smoke tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run smoke tests for dataset-generated registry rules created during upload."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8094", help="FastAPI base URL")
    parser.add_argument(
        "--file",
        default="Billing_Reconciliation_Sample_Datasets.xlsx",
        help="Workbook to upload when --dataset-id is not provided",
    )
    parser.add_argument("--dataset-id", help="Existing dataset_id to validate instead of uploading a new file")
    parser.add_argument(
        "--min-generated-rules",
        type=int,
        default=1,
        help="Minimum number of generated rules expected on the dataset",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional cap on how many generated rules to exercise",
    )
    args = parser.parse_args()

    file_path = None
    if not args.dataset_id:
        file_path = os.path.abspath(args.file)
        if not os.path.exists(file_path):
            print(f"Workbook not found: {file_path}")
            return 2

    try:
        return run_generated_registry_smoke_tests(
            args.base_url,
            dataset_id=args.dataset_id,
            file_path=file_path,
            min_generated_rules=max(0, args.min_generated_rules),
            max_cases=args.max_cases,
        )
    except Exception as exc:
        print(f"Generated registry smoke test execution failed: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())