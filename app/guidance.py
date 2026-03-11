from __future__ import annotations

from typing import Any

from app.planning_context import PlanningContext


def _normalized_roles_for_sheet(context: PlanningContext, target_sheet: str | None) -> set[str]:
    if not target_sheet:
        return set()
    role_map = context.semantic_roles.get(target_sheet) or {}
    return {str(role).strip().lower() for role in role_map.keys() if str(role).strip()}


def _normalized_columns_for_sheet(context: PlanningContext, target_sheet: str | None) -> set[str]:
    if not target_sheet:
        return set()
    return {str(column).strip().lower() for column in (context.columns_by_sheet.get(target_sheet) or []) if str(column).strip()}


def _dataset_aware_suggestions(context: PlanningContext, target_sheet: str | None) -> list[str]:
    roles = _normalized_roles_for_sheet(context, target_sheet)
    columns = _normalized_columns_for_sheet(context, target_sheet)

    is_exception_like = bool(
        {"exception_type", "recommended_action", "value_gap", "expected_amount", "billed_amount", "billing_lag_days"} & roles
        or {"exception_type", "recommended_action", "value_gap", "expected_amount", "billed_amount", "billing_lag_days"} & columns
    )
    if is_exception_like:
        return [
            "Try: Show exception summary by exception_type",
            "Try: Show customer with recommended action Invoice Now",
            "Try: Show customer, event_id, value_gap, and billing_lag_days",
            "Try: Show total value_gap by customer",
        ]

    is_sales_like = bool(
        {"region", "product", "revenue", "quantity", "date"} & roles
        or {"region", "product", "totalprice", "amount", "quantity", "date"} & columns
    )
    if is_sales_like:
        return [
            "Try: Sum total_price where Product = Chair and Region = East",
            "Try: Show total price by Product where Region = East",
            "Try: Show rows where Region = East",
        ]

    return [
        "Try: Show a sheet-wise summary using detected semantic roles.",
        "Try: Show first 20 rows from the most relevant sheet.",
        "Try: Show trend by year for key numeric metrics.",
    ]


def build_planner_guidance(
    question: str,
    context: PlanningContext,
    *,
    target_sheet: str | None,
    warnings: list[str],
    extracted_filters: list[dict[str, Any]],
) -> dict[str, Any]:
    likely_columns: list[str] = []
    if target_sheet:
        likely_columns.extend((context.columns_by_sheet.get(target_sheet) or [])[:10])
    ambiguous_values = [warning for warning in warnings if "could map" in warning.lower()]
    filter_rephrasings: list[str] = []
    for filter_spec in extracted_filters[:2]:
        column = str(filter_spec.get("column") or "").strip()
        value = filter_spec.get("value")
        if column and value not in (None, ""):
            filter_rephrasings.append(f"Try: Show rows where {column} = {value}")

    safe_rephrasings = [
        *filter_rephrasings,
        *_dataset_aware_suggestions(context, target_sheet),
    ]

    return {
        "message": "The planner could not map the question confidently enough to a deterministic QuerySpec.",
        "likely_sheet": target_sheet,
        "likely_columns": likely_columns[:8],
        "ambiguous_values": ambiguous_values,
        "warnings": warnings,
        "suggested_questions": safe_rephrasings[:5],
    }