import argparse
import json
import sys

import requests


EXPECTED_INTENTS = {
    "ops_events_contracts_ratecard_customer_summary",
    "ops_events_invoices_customer_summary",
    "ops_events_invoices_customer_variance_summary",
    "ops_events_invoices_uninvoiced_by_customer",
    "ops_events_invoices_invoice_coverage_by_customer",
    "ops_events_invoices_billed_unbilled_status_by_customer",
}


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


def _assert_required_keys(row: dict, required_keys: list[str], label: str, failures: list[str]) -> None:
    missing = [key for key in required_keys if key not in row]
    if missing:
        failures.append(f"{label}: missing keys {missing} in sample row {json.dumps(row)[:400]}")


def run_registry_smoke_tests(base_url: str, dataset_id: str) -> int:
    failures: list[str] = []

    rules = _get(base_url, "/rules_registry")
    intents = {str(rule.get("intent")) for rule in rules if isinstance(rule, dict)}
    missing_intents = sorted(EXPECTED_INTENTS - intents)
    if missing_intents:
        failures.append(f"Registry missing expected intents: {missing_intents}")
    else:
        print("PASS Registry intents present")

    cases = [
        {
            "label": "contracts_ratecard_summary",
            "question": "Join Ops_Events and Contracts_RateCard on contract_id = contract_id and summarize key metrics by customer.",
            "expected_intent": "ops_events_contracts_ratecard_customer_summary",
            "min_rows": 1,
            "required_keys": ["customer", "event_count", "contract_count", "total_quantity", "total_expected_amount"],
        },
        {
            "label": "customer_summary",
            "question": "Join Ops_Events and Invoices_Synthetic on event_id = event_id and summarize key metrics by customer.",
            "expected_intent": "ops_events_invoices_customer_summary",
            "min_rows": 1,
            "required_keys": ["customer", "event_count", "invoice_count", "total_expected_amount", "total_billed_amount"],
        },
        {
            "label": "variance_summary",
            "question": "Show expected vs billed variance by customer for Ops_Events and Invoices_Synthetic.",
            "expected_intent": "ops_events_invoices_customer_variance_summary",
            "min_rows": 1,
            "required_keys": ["customer", "total_expected_amount", "total_billed_amount", "total_value_gap"],
        },
        {
            "label": "uninvoiced_by_customer",
            "question": "Show uninvoiced events by customer.",
            "min_rows": 0,
            "expected_intent": "ops_events_invoices_uninvoiced_by_customer",
            "required_keys": ["customer", "uninvoiced_event_count", "total_uninvoiced_expected_amount"],
        },
        {
            "label": "coverage_by_customer",
            "question": "Show invoice coverage summary by customer.",
            "expected_intent": "ops_events_invoices_invoice_coverage_by_customer",
            "min_rows": 1,
            "required_keys": ["customer", "event_count", "invoice_count", "uncovered_event_count"],
        },
        {
            "label": "billed_vs_unbilled",
            "question": "Show billed vs unbilled summary by customer.",
            "expected_intent": "ops_events_invoices_billed_unbilled_status_by_customer",
            "min_rows": 1,
            "required_keys": ["customer", "billed_event_count", "unbilled_event_count", "total_billed_amount", "total_unbilled_expected_amount"],
        },
    ]

    for case in cases:
        payload = {
            "dataset_id": dataset_id,
            "question": case["question"],
            "max_rows": 1000,
            "analyze_all_rows": False,
        }
        response = _post(base_url, "/ask_sql", payload)
        route_source = response.get("route_source")
        route_intent = response.get("route_intent")
        rows = response.get("rows") or []
        rows_returned = int(response.get("rows_returned") or 0)

        if route_source != "rules_registry_sql_template":
            failures.append(
                f"{case['label']}: expected route_source=rules_registry_sql_template got {route_source}"
            )
            continue
        if route_intent != case["expected_intent"]:
            failures.append(
                f"{case['label']}: expected route_intent={case['expected_intent']} got {route_intent}"
            )
            continue
        if rows_returned < case["min_rows"]:
            failures.append(
                f"{case['label']}: expected at least {case['min_rows']} rows got {rows_returned}"
            )
            continue
        if rows:
            _assert_required_keys(rows[0], case["required_keys"], case["label"], failures)

        sample = json.dumps(rows[0], default=str)[:300] if rows else "<no rows>"
        print(
            f"PASS {case['label']}: intent={route_intent} rows={rows_returned} sample={sample}"
        )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll registry smoke tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registry routing smoke tests against the local FastAPI backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8094", help="FastAPI base URL")
    parser.add_argument("--dataset-id", required=True, help="Existing dataset_id to query against")
    args = parser.parse_args()

    try:
        return run_registry_smoke_tests(args.base_url, args.dataset_id)
    except Exception as exc:
        print(f"Registry smoke test execution failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())