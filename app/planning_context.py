from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re


def normalize_name(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or "").strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")


def normalize_text(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _looks_numeric(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    text = re.sub(r"[$€£¥₹,%]", "", text).replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text))


def _looks_date(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text)
        or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text)
        or re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text)
    )


def _is_identifier_column(column_name: str) -> bool:
    normalized = normalize_name(column_name)
    tokens = set(normalized.split("_"))
    return bool(tokens & {"id", "key", "code", "number", "num", "identifier"})


METRIC_SOURCE_CANDIDATES: dict[str, list[str]] = {
    "quantity": ["quantity", "qty", "units", "unit"],
    "unit_price": ["unit_price", "unitprice", "price_per_unit", "price"],
    "sales": ["sales", "sales_amount", "net_sales", "total_sales"],
    "cost": ["cost", "total_cost", "unit_cost", "landed_cost", "expense", "expenses"],
    "revenue": ["revenue", "total_price", "sales_amount", "net_sales", "total_amount", "amount"],
    "expected_amount": ["expected_amount", "expected_amt", "amount_due", "expected_value"],
    "actual_amount": ["actual_amount", "actual_amt", "billed_amount", "amount_billed", "invoice_amount"],
    "stock_qty": ["stock_qty", "stock_quantity", "inventory_qty", "qty_on_hand", "on_hand_qty"],
}


DERIVED_METRIC_RULES: list[dict[str, Any]] = [
    {
        "targets": ["revenue", "sales", "order_value"],
        "expression": ("quantity", "*", "unit_price"),
    },
    {
        "targets": ["profit"],
        "expression": ("revenue", "-", "cost"),
    },
    {
        "targets": ["value_gap", "variance"],
        "expression": ("expected_amount", "-", "actual_amount"),
    },
    {
        "targets": ["inventory_value"],
        "expression": ("stock_qty", "*", "unit_price"),
    },
    {
        "targets": ["avg_price"],
        "expression": ("revenue", "/", "quantity"),
    },
]


def _normalized_numeric_lookup(numeric_columns: list[str]) -> dict[str, str]:
    return {normalize_name(column_name): str(column_name) for column_name in (numeric_columns or [])}


def _resolve_metric_reference(
    reference_name: str,
    numeric_columns: list[str],
    derived_metrics: dict[str, str] | None = None,
) -> str | None:
    normalized_lookup = _normalized_numeric_lookup(numeric_columns)
    for candidate in METRIC_SOURCE_CANDIDATES.get(reference_name, [reference_name]):
        resolved = normalized_lookup.get(normalize_name(candidate))
        if resolved:
            return normalize_name(resolved)
    for candidate in METRIC_SOURCE_CANDIDATES.get(reference_name, [reference_name]):
        matching_metric = next(
            (
                metric_name
                for metric_name in (derived_metrics or {})
                if normalize_name(metric_name) == normalize_name(candidate)
            ),
            None,
        )
        if matching_metric:
            return str(matching_metric)
    return None


def detect_derived_metrics_by_sheet(
    columns_by_sheet: dict[str, list[str]] | dict[str, set[str]],
    *,
    numeric_columns_by_sheet: dict[str, list[str]] | None = None,
    column_types_by_sheet: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    derived_metrics_by_sheet: dict[str, dict[str, str]] = {}
    numeric_columns_by_sheet = numeric_columns_by_sheet or {}
    column_types_by_sheet = column_types_by_sheet or {}

    for sheet_name, raw_columns in (columns_by_sheet or {}).items():
        if sheet_name in numeric_columns_by_sheet:
            numeric_columns = [str(column_name) for column_name in (numeric_columns_by_sheet.get(sheet_name) or [])]
        else:
            declared_types = column_types_by_sheet.get(str(sheet_name)) or {}
            numeric_columns = []
            for column_name in raw_columns or []:
                column_text = str(column_name)
                declared_type = str(declared_types.get(column_text) or "").lower()
                normalized = normalize_name(column_text)
                if declared_type == "numeric" or any(token in normalized for token in ["amount", "price", "revenue", "sales", "profit", "qty", "quantity", "cost", "value"]):
                    numeric_columns.append(column_text)

        derived_metrics: dict[str, str] = {}
        for rule in DERIVED_METRIC_RULES:
            left_reference, operator, right_reference = rule["expression"]
            left_operand = _resolve_metric_reference(left_reference, numeric_columns, derived_metrics)
            right_operand = _resolve_metric_reference(right_reference, numeric_columns, derived_metrics)
            if not left_operand or not right_operand:
                continue
            for metric_name in rule["targets"]:
                derived_metrics[str(metric_name)] = f"{left_operand} {operator} {right_operand}"

        if derived_metrics:
            derived_metrics_by_sheet[str(sheet_name)] = derived_metrics

    return derived_metrics_by_sheet


def flatten_derived_metrics_by_sheet(derived_metrics_by_sheet: dict[str, dict[str, str]] | None) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for sheet_metrics in (derived_metrics_by_sheet or {}).values():
        for metric_name, formula in (sheet_metrics or {}).items():
            flattened.setdefault(str(metric_name), str(formula))
    return flattened


@dataclass(slots=True)
class PlanningContext:
    dataset_id: str
    sheets: list[str]
    columns_by_sheet: dict[str, list[str]]
    numeric_columns_by_sheet: dict[str, list[str]]
    categorical_columns_by_sheet: dict[str, list[str]]
    identifier_columns_by_sheet: dict[str, list[str]]
    date_columns_by_sheet: dict[str, list[str]]
    semantic_roles: dict[str, dict[str, str]]
    joins: list[dict[str, Any]]
    sampled_categorical_values_by_sheet: dict[str, dict[str, list[str]]]
    derived_metrics_by_sheet: dict[str, dict[str, str]] = field(default_factory=dict)
    cleaning_report: dict[str, Any] = field(default_factory=dict)


_CONTEXT_CACHE: dict[str, tuple[str, PlanningContext]] = {}


def _fingerprint_context(
    dataset_id: str,
    columns_by_sheet: dict[str, list[str]],
    semantic_roles: dict[str, dict[str, str]],
    sampled_categorical_values_by_sheet: dict[str, dict[str, list[str]]],
    derived_metrics_by_sheet: dict[str, dict[str, str]],
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "columns_by_sheet": columns_by_sheet,
        "semantic_roles": semantic_roles,
        "sampled_categorical_values_by_sheet": sampled_categorical_values_by_sheet,
        "derived_metrics_by_sheet": derived_metrics_by_sheet,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _sampled_values_from_rows(sheet_rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, set[str]]] = {}
    for row in sheet_rows or []:
        if not isinstance(row, dict):
            continue
        sheet_name = str(row.get("sheet_name") or "").strip()
        row_json = row.get("row_json")
        if not sheet_name or not isinstance(row_json, dict):
            continue
        per_sheet = collected.setdefault(sheet_name, {})
        for column_name, value in row_json.items():
            normalized_value = normalize_text(value)
            if not normalized_value or len(normalized_value) < 2:
                continue
            values = per_sheet.setdefault(str(column_name), set())
            if len(values) < 100:
                values.add(normalized_value)

    sampled: dict[str, dict[str, list[str]]] = {}
    for sheet_name, per_sheet in collected.items():
        next_sheet: dict[str, list[str]] = {}
        for column_name, values in per_sheet.items():
            if 1 < len(values) <= 50:
                next_sheet[column_name] = sorted(values)
        if next_sheet:
            sampled[sheet_name] = next_sheet
    return sampled


def _infer_column_buckets(
    columns_by_sheet: dict[str, list[str]],
    column_types: dict[str, str],
    sampled_categorical_values_by_sheet: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    numeric_columns_by_sheet: dict[str, list[str]] = {}
    categorical_columns_by_sheet: dict[str, list[str]] = {}
    identifier_columns_by_sheet: dict[str, list[str]] = {}
    date_columns_by_sheet: dict[str, list[str]] = {}

    for sheet_name, columns in columns_by_sheet.items():
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        identifier_columns: list[str] = []
        date_columns: list[str] = []
        sampled_values = sampled_categorical_values_by_sheet.get(sheet_name) or {}

        for column_name in columns:
            normalized = normalize_name(column_name)
            declared_type = str(column_types.get(column_name) or "").lower()
            sample_values = sampled_values.get(column_name) or []
            if declared_type == "numeric" or any(token in normalized for token in ["amount", "price", "revenue", "sales", "profit", "qty", "quantity", "cost", "value"]):
                numeric_columns.append(column_name)
            elif declared_type == "date" or normalized.endswith("date"):
                date_columns.append(column_name)
            elif _is_identifier_column(column_name):
                identifier_columns.append(column_name)
            elif sample_values:
                sample_numeric_ratio = sum(1 for value in sample_values if _looks_numeric(value)) / max(len(sample_values), 1)
                sample_date_ratio = sum(1 for value in sample_values if _looks_date(value)) / max(len(sample_values), 1)
                if sample_numeric_ratio >= 0.8:
                    numeric_columns.append(column_name)
                elif sample_date_ratio >= 0.8:
                    date_columns.append(column_name)
                else:
                    categorical_columns.append(column_name)
            else:
                categorical_columns.append(column_name)

        numeric_columns_by_sheet[sheet_name] = numeric_columns
        categorical_columns_by_sheet[sheet_name] = categorical_columns
        identifier_columns_by_sheet[sheet_name] = identifier_columns
        date_columns_by_sheet[sheet_name] = date_columns

    return numeric_columns_by_sheet, categorical_columns_by_sheet, identifier_columns_by_sheet, date_columns_by_sheet


def build_planning_context(
    dataset_id: str,
    cols: list[str],
    col_types: dict[str, str],
    relationships_payload: dict[str, Any] | None,
    sheet_rows: list[dict[str, Any]] | None,
) -> PlanningContext:
    relationships_payload = relationships_payload or {}
    semantic_roles = relationships_payload.get("semantic_roles") or {}
    joins = list(relationships_payload.get("relationships") or [])
    sheet_names = [str(sheet) for sheet in list(relationships_payload.get("tables") or []) if str(sheet).strip()]
    sampled_values = _sampled_values_from_rows(sheet_rows or [])

    columns_by_sheet: dict[str, list[str]] = {}
    for row in sheet_rows or []:
        if not isinstance(row, dict):
            continue
        sheet_name = str(row.get("sheet_name") or "").strip()
        row_json = row.get("row_json")
        if not sheet_name or not isinstance(row_json, dict):
            continue
        existing = columns_by_sheet.setdefault(sheet_name, [])
        for column_name in row_json.keys():
            column_text = str(column_name)
            if column_text not in existing:
                existing.append(column_text)

    if not columns_by_sheet and sheet_names and cols:
        base_sheet = sheet_names[0]
        columns_by_sheet[base_sheet] = [str(column) for column in cols]
    elif len(columns_by_sheet) == 1 and cols:
        base_sheet = next(iter(columns_by_sheet.keys()))
        current = set(columns_by_sheet.get(base_sheet) or [])
        for column_name in cols:
            if column_name not in current:
                columns_by_sheet[base_sheet].append(str(column_name))

    if not sheet_names:
        sheet_names = list(columns_by_sheet.keys())

    numeric_columns_by_sheet, categorical_columns_by_sheet, identifier_columns_by_sheet, date_columns_by_sheet = _infer_column_buckets(
        columns_by_sheet,
        col_types or {},
        sampled_values,
    )
    derived_metrics_by_sheet = detect_derived_metrics_by_sheet(
        columns_by_sheet,
        numeric_columns_by_sheet=numeric_columns_by_sheet,
    )
    payload_derived_metrics = relationships_payload.get("derived_metrics_by_sheet") or {}
    if isinstance(payload_derived_metrics, dict):
        for sheet_name, sheet_metrics in payload_derived_metrics.items():
            if not isinstance(sheet_metrics, dict):
                continue
            derived_metrics_by_sheet[str(sheet_name)] = {
                str(metric_name): str(formula)
                for metric_name, formula in sheet_metrics.items()
                if str(metric_name).strip() and str(formula).strip()
            }

    fingerprint = _fingerprint_context(dataset_id, columns_by_sheet, semantic_roles, sampled_values, derived_metrics_by_sheet)
    cached = _CONTEXT_CACHE.get(dataset_id)
    if cached and cached[0] == fingerprint:
        return cached[1]

    context = PlanningContext(
        dataset_id=dataset_id,
        sheets=sheet_names,
        columns_by_sheet=columns_by_sheet,
        numeric_columns_by_sheet=numeric_columns_by_sheet,
        categorical_columns_by_sheet=categorical_columns_by_sheet,
        identifier_columns_by_sheet=identifier_columns_by_sheet,
        date_columns_by_sheet=date_columns_by_sheet,
        semantic_roles={str(sheet): {str(role): str(column) for role, column in (roles or {}).items()} for sheet, roles in semantic_roles.items()},
        joins=joins,
        sampled_categorical_values_by_sheet=sampled_values,
        derived_metrics_by_sheet=derived_metrics_by_sheet,
        cleaning_report=relationships_payload.get("cleaning_report") or {},
    )
    _CONTEXT_CACHE[dataset_id] = (fingerprint, context)
    return context