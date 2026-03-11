from __future__ import annotations

from collections import Counter
from typing import Any
import re

from app.planning_context import PlanningContext, normalize_name


def _is_pre_aggregated_sheet(context: PlanningContext, target_sheet: str | None) -> bool:
    if not target_sheet:
        return False
    sheet_name = normalize_name(target_sheet)
    if "summary" in sheet_name or "aggregate" in sheet_name:
        return True
    numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
    for column_name in numeric_columns:
        normalized = normalize_name(column_name)
        tokens = set(normalized.split("_"))
        if normalized in {"count", "row_count", "matched_rows", "total", "summary"}:
            return True
        if tokens & {"count", "total", "sum", "avg", "average", "min", "max", "gap", "value", "summary"}:
            return True
    return False


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = re.sub(r"[$€£¥₹,%]", "", text).replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    number = float(text)
    return -number if negative and number > 0 else number


def _round_number(value: float) -> float:
    return round(float(value), 4)


def _column_types_for_sheet(context: PlanningContext, target_sheet: str) -> dict[str, str]:
    numeric_columns = set(context.numeric_columns_by_sheet.get(target_sheet) or [])
    date_columns = set(context.date_columns_by_sheet.get(target_sheet) or [])
    identifier_columns = set(context.identifier_columns_by_sheet.get(target_sheet) or [])
    categorical_columns = set(context.categorical_columns_by_sheet.get(target_sheet) or [])

    inferred: dict[str, str] = {}
    for column_name in context.columns_by_sheet.get(target_sheet) or []:
        if column_name in numeric_columns:
            inferred[column_name] = "numeric"
        elif column_name in date_columns:
            inferred[column_name] = "date"
        elif column_name in identifier_columns:
            inferred[column_name] = "identifier"
        elif column_name in categorical_columns:
            inferred[column_name] = "categorical"
        else:
            inferred[column_name] = "text"
    return inferred


def build_sheet_profile(
    target_sheet: str,
    context: PlanningContext,
    sheet_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    sheet_name = str(target_sheet or "").strip()
    if not sheet_name:
        raise ValueError("target_sheet is required")

    matching_rows = [
        row.get("row_json")
        for row in list(sheet_rows or [])
        if str((row or {}).get("sheet_name") or "").strip() == sheet_name and isinstance((row or {}).get("row_json"), dict)
    ]
    column_names = list(context.columns_by_sheet.get(sheet_name) or [])
    inferred_column_types = _column_types_for_sheet(context, sheet_name)
    pre_aggregated_sheet = _is_pre_aggregated_sheet(context, sheet_name)

    numeric_summaries: list[dict[str, Any]] = []
    for column_name in context.numeric_columns_by_sheet.get(sheet_name) or []:
        numeric_values = [_to_float(row_json.get(column_name)) for row_json in matching_rows]
        numeric_values = [value for value in numeric_values if value is not None]
        if not numeric_values:
            continue
        numeric_summaries.append(
            {
                "column": column_name,
                "non_null_count": len(numeric_values),
                "min": _round_number(min(numeric_values)),
                "max": _round_number(max(numeric_values)),
                "avg": _round_number(sum(numeric_values) / len(numeric_values)),
            }
        )

    categorical_summaries: list[dict[str, Any]] = []
    categorical_candidates = [
        column_name
        for column_name in column_names
        if inferred_column_types.get(column_name) in {"categorical", "identifier", "text"}
    ]
    for column_name in categorical_candidates:
        values = [str(row_json.get(column_name) or "").strip() for row_json in matching_rows]
        values = [value for value in values if value]
        if not values:
            continue
        counts = Counter(values)
        categorical_summaries.append(
            {
                "column": column_name,
                "distinct_count": len(counts),
                "top_values": [value for value, _count in counts.most_common(5)],
            }
        )

    return {
        "sheet_name": sheet_name,
        "row_count": len(matching_rows),
        "column_count": len(column_names),
        "column_names": column_names,
        "inferred_column_types": inferred_column_types,
        "appears_pre_aggregated": pre_aggregated_sheet,
        "numeric_field_summaries": numeric_summaries,
        "categorical_field_summaries": categorical_summaries,
        "profile_summary": {
            "notable_numeric_fields": [summary["column"] for summary in numeric_summaries[:5]],
            "notable_categorical_fields": [summary["column"] for summary in categorical_summaries[:5]],
        },
    }