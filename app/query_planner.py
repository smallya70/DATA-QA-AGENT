from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import re

from app.business_intents import (
    BUSINESS_SYNONYMS,
    best_or_worst_ranking_aggregation_for_question,
    best_or_worst_ranking_direction,
    bottom_n_aggregation_for_question,
    CanonicalIntent,
    detect_canonical_intent,
    family_matches_column_or_role,
    has_distinct_intent,
    has_group_by_intent,
    matched_intent_keywords,
    role_matches_question,
    semantic_aliases,
    top_n_aggregation_for_question,
)
from app.filter_extractor import extract_filters
from app.guidance import build_planner_guidance
from app.planning_context import PlanningContext, normalize_name
from app.query_classifier import QueryClassification, classify_query
from app.query_spec import QuerySpec
from app.value_mapper import ValueMapping, map_values_to_columns


MEASURE_PRIORITY = {
    "sales": ["total_price", "sales_amount", "net_sales", "total_amount", "amount", "revenue"],
    "price": ["total_price", "unit_price", "price", "amount", "total_amount"],
    "revenue": ["revenue", "total_price", "sales_amount", "net_sales", "amount"],
    "amount": ["amount", "total_amount", "total_price", "sales_amount", "revenue", "value_gap", "expected_amount", "billed_amount"],
    "cost": ["cost", "total_cost", "unit_cost", "landed_cost", "expense", "expenses"],
    "value": ["amount", "total_amount", "total_price", "sales_amount", "revenue", "value_gap", "expected_amount", "billed_amount"],
    "value_gap": ["value_gap", "variance", "gap"],
    "profit": ["profit", "gross_profit", "net_profit"],
    "order_value": ["order_value", "total_price", "sales_amount", "net_sales", "total_amount", "amount"],
    "inventory_value": ["inventory_value"],
    "quantity": ["quantity", "qty", "units"],
    "count": ["*"],
}

ROW_COUNT_TERMS = {"row", "rows", "record", "records"}
DISTINCT_DIMENSION_TERMS = {"product", "products", "country", "countries", "region", "regions", "customer", "customers"}
DERIVED_METRIC_ALIASES = {
    "revenue": ["revenue"],
    "profit": ["profit"],
    "avg_price": ["avg price", "average price", "unit price"],
}
METRIC_ALIAS_TO_CANONICAL = {
    "sales": "revenue",
    "turnover": "revenue",
    "gross_sales": "revenue",
    "gross sales": "revenue",
    "variance": "value_gap",
    "margin": "margin_pct",
}
CANONICAL_METRIC_PHRASES = {
    "revenue": ["revenue", "sales", "turnover", "gross sales"],
    "order_value": ["order value"],
    "profit": ["profit"],
    "value_gap": ["value gap", "variance"],
    "inventory_value": ["inventory value"],
    "amount": ["amount"],
    "cost": ["cost"],
    "quantity": ["quantity", "qty", "units"],
    "expected_amount": ["expected amount"],
    "actual_amount": ["actual amount", "billed amount", "invoice amount", "amount billed"],
    "margin_pct": ["margin"],
}
GENERIC_METRIC_REQUEST_PHRASES = {"total", "value", "overall", "combined", "aggregate"}


@dataclass(slots=True)
class QueryPlan:
    pattern_id: str
    query_type: str
    canonical_intent: str
    answer_type: str
    query_shape: str
    target_sheet: str | None
    aggregation: str | None = None
    metric_column: str | None = None
    metric_sheet: str | None = None
    columns: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    partition_by: list[str] = field(default_factory=list)
    rank_dimension: str | None = None
    order_direction: str = "desc"
    requested_top_n: int | None = None
    time_grain: str | None = None
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    auto_corrections: list[str] = field(default_factory=list)
    validation_result: str = "pending"
    validation_issues: list[str] = field(default_factory=list)
    selected_columns_exist: bool = False
    pre_aggregated_sheet: bool = False
    requested_projection_columns: list[str] = field(default_factory=list)
    matched_existing_columns: list[str] = field(default_factory=list)
    aggregation_override_skipped_reason: str | None = None
    skip_aggregate_pattern_matching: bool = False
    intent_family: str | None = None
    metric_formula: str | None = None
    requested_metric_name: str | None = None
    canonical_metric_name: str | None = None
    resolved_metric_type: str | None = None


@dataclass(slots=True)
class PlannerResult:
    query_spec: QuerySpec | None
    query_shape: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    target_sheet: str | None = None
    extracted_filters: list[dict[str, Any]] = field(default_factory=list)
    inferred_value_mappings: list[dict[str, Any]] = field(default_factory=list)
    classifier: QueryClassification | None = None
    guidance: dict[str, Any] | None = None
    degraded_mode: bool = False
    canonical_intent: str = "unsupported"
    answer_type: str = "guidance"
    candidate_sheets: list[dict[str, Any]] = field(default_factory=list)
    selected_metric: dict[str, Any] | None = None
    candidate_metrics: list[dict[str, Any]] = field(default_factory=list)
    selected_dimensions: list[str] = field(default_factory=list)
    query_type: str = "aggregate"
    pattern_id: str | None = None
    matched_intent_keywords: dict[str, list[str]] = field(default_factory=dict)
    auto_corrections: list[str] = field(default_factory=list)
    validation_result: str = "not_run"
    fallback_reason: str | None = None
    requested_dimension: str | None = None
    requested_metric: str | None = None
    resolved_dimension: str | None = None
    resolved_metric: str | None = None
    missing_dimension_reason: str | None = None
    missing_metric_reason: str | None = None
    fallback_blocked_reason: str | None = None
    available_dimensions: list[str] = field(default_factory=list)
    available_metrics: list[str] = field(default_factory=list)


def _role_for_column(context: PlanningContext, sheet_name: str, column_name: str) -> str | None:
    for role_name, mapped_column in (context.semantic_roles.get(sheet_name) or {}).items():
        if str(mapped_column) == str(column_name):
            return str(role_name)
    return None


def _column_for_roles(context: PlanningContext, sheet_name: str | None, role_names: list[str]) -> str | None:
    if not sheet_name:
        return None
    role_map = context.semantic_roles.get(sheet_name) or {}
    for role_name in role_names:
        mapped_column = str(role_map.get(role_name) or "").strip()
        if mapped_column:
            return mapped_column
    return None


def _is_status_like_column_name(column_name: str) -> bool:
    return bool({"status", "state", "stage", "exception"} & set(normalize_name(column_name).split("_")))


def _augment_list_dimensions(question: str, context: PlanningContext, target_sheet: str | None, dimensions: list[str]) -> list[str]:
    if not target_sheet:
        return dimensions

    q_text = str(question or "").lower()
    if "invoice" not in q_text and "bill" not in q_text:
        return dimensions

    preferred_columns = [
        _column_for_roles(context, target_sheet, ["invoice_id"]),
        _column_for_roles(context, target_sheet, ["invoice_date"]),
        _column_for_roles(context, target_sheet, ["customer_id"]),
        _column_for_roles(context, target_sheet, ["invoice_status", "exception_type", "status"]),
        _column_for_roles(context, target_sheet, ["expected_amount"]),
        _column_for_roles(context, target_sheet, ["billed_amount"]),
        _column_for_roles(context, target_sheet, ["value_gap"]),
    ]

    ordered_dimensions: list[str] = []
    for column_name in preferred_columns + list(dimensions):
        normalized = str(column_name or "").strip()
        if normalized and normalized not in ordered_dimensions:
            ordered_dimensions.append(normalized)
    return ordered_dimensions


def _build_measure_alias(aggregation: str, column_name: str) -> str:
    if aggregation == "count" and column_name == "*":
        return "row_count"
    return f"{aggregation}_{normalize_name(column_name or 'value')}"


def _requested_top_n(question: str) -> tuple[int | None, str | None]:
    match = re.search(r"\b(top|bottom)\s+(\d+)\b", str(question or "").lower())
    if not match:
        return None, None
    return max(1, int(match.group(2))), ("desc" if match.group(1) == "top" else "asc")


def _requested_row_preview_limit(question: str, max_rows: int) -> int:
    q_text = str(question or "").lower()
    match = re.search(r"\b(?:first|top|last)\s+(\d+)\s*(?:rows?|records?)?\b", q_text)
    if not match:
        return max(1, int(max_rows))
    return max(1, min(int(match.group(1)), int(max_rows)))


def _first_requested_family_phrase(question: str, families: dict[str, list[str]]) -> tuple[str | None, str | None]:
    q_text = str(question or "").lower()
    best_match: tuple[int, str, str] | None = None
    for family, aliases in families.items():
        for alias in aliases:
            alias_text = str(alias or "").strip().lower()
            if not alias_text:
                continue
            escaped = re.escape(alias_text).replace(r"\ ", r"\s+")
            match = re.search(rf"\b{escaped}\b", q_text)
            if match is None:
                continue
            current = (match.start(), family, alias_text)
            if best_match is None or current < best_match:
                best_match = current
    if best_match is None:
        return None, None
    return best_match[1], best_match[2]


def _available_dimension_columns(context: PlanningContext, target_sheet: str | None) -> list[str]:
    if not target_sheet:
        return []
    preferred = set(context.identifier_columns_by_sheet.get(target_sheet) or []) | set(context.categorical_columns_by_sheet.get(target_sheet) or [])
    ordered: list[str] = []
    for column_name in context.columns_by_sheet.get(target_sheet) or []:
        if column_name in preferred and column_name not in ordered:
            ordered.append(str(column_name))
    return ordered


def _available_metric_columns(context: PlanningContext, target_sheet: str | None) -> list[str]:
    if not target_sheet:
        return []
    available_metrics: list[str] = []
    for metric_name in (context.derived_metrics_by_sheet.get(target_sheet) or {}).keys():
        if str(metric_name) not in available_metrics:
            available_metrics.append(str(metric_name))
    for column_name in context.numeric_columns_by_sheet.get(target_sheet) or []:
        if str(column_name) not in available_metrics:
            available_metrics.append(str(column_name))
    return available_metrics


def _metric_phrase_match_position(question: str, phrase: str) -> int | None:
    phrase_text = str(phrase or "").strip().lower()
    if not phrase_text:
        return None
    match = re.search(rf"\b{re.escape(phrase_text).replace(r'\ ', r'\s+')}\b", str(question or "").lower())
    return match.start() if match is not None else None


def _canonical_metric_name(metric_name: str) -> str:
    normalized_metric = normalize_name(metric_name)
    return normalize_name(METRIC_ALIAS_TO_CANONICAL.get(normalized_metric, normalized_metric))


def _find_requested_metric_name(question: str, context: PlanningContext, target_sheet: str | None) -> str | None:
    candidates: list[tuple[int, int, int, str]] = []
    if target_sheet:
        numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
        for column_name in numeric_columns:
            for phrase in {str(column_name), str(column_name).replace("_", " "), normalize_name(column_name).replace("_", " ")}:
                position = _metric_phrase_match_position(question, phrase)
                if position is not None:
                    candidates.append((position, 0, -len(phrase), normalize_name(column_name)))
        for role_name, mapped_column in (context.semantic_roles.get(target_sheet) or {}).items():
            if str(mapped_column) not in numeric_columns:
                continue
            for phrase in semantic_aliases(role_name):
                if normalize_name(phrase) in GENERIC_METRIC_REQUEST_PHRASES:
                    continue
                position = _metric_phrase_match_position(question, phrase)
                if position is not None:
                    candidates.append((position, 1, -len(phrase), normalize_name(role_name)))

    for canonical_metric, phrases in CANONICAL_METRIC_PHRASES.items():
        for phrase in phrases:
            position = _metric_phrase_match_position(question, phrase)
            if position is not None:
                candidates.append((position, 2, -len(phrase), normalize_name(phrase)))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def _derived_metrics_for_sheet(context: PlanningContext, target_sheet: str | None) -> dict[str, str]:
    if not target_sheet:
        return {}
    return {
        str(metric_name): str(formula)
        for metric_name, formula in (context.derived_metrics_by_sheet.get(target_sheet) or {}).items()
        if str(metric_name).strip() and str(formula).strip()
    }


def _physical_metric_column_for_name(
    context: PlanningContext,
    target_sheet: str,
    metric_name: str,
    *,
    exact_only: bool,
) -> str | None:
    numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
    normalized_metric = normalize_name(metric_name)
    normalized_columns = {normalize_name(column_name): str(column_name) for column_name in numeric_columns}
    if normalized_metric in normalized_columns:
        return normalized_columns[normalized_metric]

    for role_name, mapped_column in (context.semantic_roles.get(target_sheet) or {}).items():
        if str(mapped_column) not in numeric_columns:
            continue
        normalized_role = normalize_name(role_name)
        if normalized_role == normalized_metric:
            return str(mapped_column)
        if any(normalize_name(alias) == normalized_metric for alias in semantic_aliases(role_name)):
            return str(mapped_column)

    if exact_only:
        return None

    for candidate_name in MEASURE_PRIORITY.get(normalized_metric, []):
        matched_column = normalized_columns.get(normalize_name(candidate_name))
        if matched_column:
            return matched_column

    return None


def _resolved_measure_payload(
    *,
    column_name: str,
    aggregation: str,
    requested_metric: str,
    canonical_metric: str,
    resolved_metric_type: str,
    formula: str | None = None,
) -> dict[str, Any]:
    measure: dict[str, Any] = {
        "column": column_name,
        "aggregation": aggregation,
        "alias": _build_measure_alias(aggregation, column_name),
        "requested_metric": requested_metric,
        "canonical_metric": canonical_metric,
        "resolved_metric_type": resolved_metric_type,
    }
    if formula:
        measure["formula"] = formula
        measure["derived"] = True
    return measure


def _resolve_requested_measure(
    question: str,
    context: PlanningContext,
    target_sheet: str,
    aggregation: str,
) -> tuple[dict[str, Any] | None, list[str], list[str], list[dict[str, Any]]]:
    requested_metric = _find_requested_metric_name(question, context, target_sheet)
    if not requested_metric:
        return None, [], [], []

    reasons: list[str] = []
    warnings: list[str] = []
    candidate_metrics: list[dict[str, Any]] = []
    canonical_metric = _canonical_metric_name(requested_metric)

    direct_physical = _physical_metric_column_for_name(context, target_sheet, requested_metric, exact_only=True)
    if direct_physical is not None:
        measure = _resolved_measure_payload(
            column_name=direct_physical,
            aggregation=aggregation,
            requested_metric=requested_metric,
            canonical_metric=canonical_metric,
            resolved_metric_type="physical",
        )
        reasons.append(f"Resolved requested metric '{requested_metric}' to physical column '{direct_physical}'.")
        candidate_metrics.append({**measure, "score": 0.98, "reasons": reasons.copy()})
        return measure, reasons, warnings, candidate_metrics

    canonical_physical = _physical_metric_column_for_name(context, target_sheet, canonical_metric, exact_only=False)
    if canonical_physical is not None:
        measure = _resolved_measure_payload(
            column_name=canonical_physical,
            aggregation=aggregation,
            requested_metric=requested_metric,
            canonical_metric=canonical_metric,
            resolved_metric_type="physical",
        )
        reasons.append(
            f"Resolved requested metric '{requested_metric}' via canonical metric '{canonical_metric}' to physical column '{canonical_physical}'."
        )
        candidate_metrics.append({**measure, "score": 0.95, "reasons": reasons.copy()})
        return measure, reasons, warnings, candidate_metrics

    derived_metrics = _derived_metrics_for_sheet(context, target_sheet)
    derived_formula = next(
        (
            str(formula)
            for metric_name, formula in derived_metrics.items()
            if normalize_name(metric_name) == canonical_metric
        ),
        None,
    )
    if derived_formula is not None:
        measure = _resolved_measure_payload(
            column_name=canonical_metric,
            aggregation=aggregation,
            requested_metric=requested_metric,
            canonical_metric=canonical_metric,
            resolved_metric_type="derived",
            formula=derived_formula,
        )
        reasons.append(
            f"Resolved requested metric '{requested_metric}' via canonical metric '{canonical_metric}' to derived formula '{derived_formula}'."
        )
        candidate_metrics.append({**measure, "score": 0.93, "reasons": reasons.copy()})
        return measure, reasons, warnings, candidate_metrics

    warnings.append(
        f"Requested metric '{requested_metric}' could not be resolved to a physical or derived metric on sheet '{target_sheet}'."
    )
    candidate_metrics.append(
        {
            "column": canonical_metric,
            "requested_metric": requested_metric,
            "canonical_metric": canonical_metric,
            "resolved_metric_type": "unsupported",
            "aggregation": aggregation,
            "score": 0.0,
            "reasons": warnings.copy(),
        }
    )
    return None, [], warnings, candidate_metrics


def _question_explicitly_requests_derived_metric(question: str, metric_name: str) -> bool:
    normalized_question = normalize_name(question).replace("_", " ")
    candidate_phrases = [normalize_name(metric_name).replace("_", " ")]
    candidate_phrases.extend(normalize_name(alias).replace("_", " ") for alias in DERIVED_METRIC_ALIASES.get(metric_name, []))
    return any(phrase and phrase in normalized_question for phrase in candidate_phrases)


def _parse_metric_formula(formula: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"\s*(.+?)\s*([+\-*/])\s*(.+?)\s*", str(formula or ""))
    if not match:
        return None
    return match.group(1).strip(), match.group(2), match.group(3).strip()


def _context_metric_reference_is_numeric(
    context: PlanningContext,
    target_sheet: str,
    reference_name: str,
    *,
    visited: set[str] | None = None,
) -> bool:
    visited = visited or set()
    normalized_reference = normalize_name(reference_name)
    normalized_numeric_columns = {
        normalize_name(column_name)
        for column_name in (context.numeric_columns_by_sheet.get(target_sheet) or [])
    }
    if normalized_reference in normalized_numeric_columns:
        return True

    derived_metrics = _derived_metrics_for_sheet(context, target_sheet)
    matching_metric_name = next(
        (metric_name for metric_name in derived_metrics if normalize_name(metric_name) == normalized_reference),
        None,
    )
    if not matching_metric_name or matching_metric_name in visited:
        return False

    parsed_formula = _parse_metric_formula(derived_metrics[matching_metric_name])
    if parsed_formula is None:
        return False
    left_operand, _operator, right_operand = parsed_formula
    next_visited = set(visited)
    next_visited.add(matching_metric_name)
    return _context_metric_reference_is_numeric(context, target_sheet, left_operand, visited=next_visited) and _context_metric_reference_is_numeric(context, target_sheet, right_operand, visited=next_visited)


def _resolved_dimension_matches_family(context: PlanningContext, target_sheet: str | None, dimensions: list[str], family: str | None) -> bool:
    if not family:
        return True
    if not target_sheet:
        return False
    for column_name in dimensions:
        role_name = _role_for_column(context, target_sheet, column_name)
        if family_matches_column_or_role(family, column_name, role_name):
            return True
    return False


def _resolved_metric_matches_family(context: PlanningContext, target_sheet: str | None, metric_column: str | None, family: str | None) -> bool:
    if not family:
        return True
    if not target_sheet or not metric_column:
        return False
    role_name = _role_for_column(context, target_sheet, metric_column)
    return family_matches_column_or_role(family, metric_column, role_name)


def _is_ranked_entity_question(canonical_intent: CanonicalIntent) -> bool:
    return canonical_intent.answer_type == "ranked_table" or canonical_intent.name in {"top_n_x_by_y", "highest_x", "lowest_x"}


def _suggest_supported_ranking_questions(
    *,
    available_dimensions: list[str],
    available_metrics: list[str],
    requested_top_n: int | None,
) -> list[str]:
    suggestions: list[str] = []
    top_prefix = f"Top {requested_top_n}" if requested_top_n and requested_top_n > 1 else "Top"
    for dimension_name in available_dimensions[:3]:
        for metric_name in available_metrics[:2]:
            suggestion = f"{top_prefix} {dimension_name} by {metric_name}"
            if suggestion not in suggestions:
                suggestions.append(suggestion)
            if len(suggestions) >= 3:
                return suggestions
    return suggestions


def _unsupported_ranking_guidance(
    *,
    target_sheet: str | None,
    requested_dimension: str | None,
    requested_metric: str | None,
    resolved_metric: str | None,
    missing_dimension_reason: str | None,
    missing_metric_reason: str | None,
    available_dimensions: list[str],
    available_metrics: list[str],
    requested_top_n: int | None,
) -> dict[str, Any]:
    problems = [item for item in [missing_dimension_reason, missing_metric_reason] if item]
    message = " ".join(problems).strip() or "The requested ranking could not be mapped safely to the dataset schema."
    if resolved_metric and requested_metric and normalize_name(resolved_metric) != normalize_name(requested_metric):
        message += f" Metric '{requested_metric}' was only approximated via '{resolved_metric}', so the ranking was blocked."
    suggestions = _suggest_supported_ranking_questions(
        available_dimensions=available_dimensions,
        available_metrics=available_metrics,
        requested_top_n=requested_top_n,
    )
    return {
        "message": message,
        "likely_sheet": target_sheet,
        "requested_dimension": requested_dimension,
        "requested_metric": requested_metric,
        "resolved_metric": resolved_metric,
        "available_dimensions": available_dimensions,
        "available_metrics": available_metrics,
        "suggested_questions": suggestions,
    }


def _blocked_ranking_result(
    *,
    question: str,
    context: PlanningContext,
    target_sheet: str | None,
    candidate_sheets: list[dict[str, Any]],
    extracted_filters: list[dict[str, Any]],
    inferred_value_mappings: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    canonical_intent: CanonicalIntent,
    measure: dict[str, Any] | None,
    requested_dimension: str | None,
    requested_metric: str | None,
    missing_dimension_reason: str | None,
    missing_metric_reason: str | None,
) -> PlannerResult:
    available_dimensions = _available_dimension_columns(context, target_sheet)
    available_metrics = _available_metric_columns(context, target_sheet)
    resolved_metric = str((measure or {}).get("column") or "").strip() or None
    warnings = [item for item in [missing_dimension_reason, missing_metric_reason] if item]
    reasons = list(dict.fromkeys([
        *canonical_intent.reasons,
        "Blocked ranking fallback because the requested entity or metric could not be resolved safely.",
    ]))
    fallback_blocked_reason = "missing_ranking_dimension" if missing_dimension_reason else "missing_ranking_metric"
    guidance = _unsupported_ranking_guidance(
        target_sheet=target_sheet,
        requested_dimension=requested_dimension,
        requested_metric=requested_metric,
        resolved_metric=resolved_metric,
        missing_dimension_reason=missing_dimension_reason,
        missing_metric_reason=missing_metric_reason,
        available_dimensions=available_dimensions,
        available_metrics=available_metrics,
        requested_top_n=canonical_intent.requested_top_n,
    )
    return PlannerResult(
        query_spec=None,
        query_shape="unsupported_query",
        confidence=0.96,
        warnings=warnings,
        reasons=reasons,
        target_sheet=target_sheet,
        extracted_filters=extracted_filters,
        inferred_value_mappings=inferred_value_mappings,
        classifier=None,
        guidance=guidance,
        degraded_mode=True,
        canonical_intent="unsupported_query",
        answer_type="guidance",
        candidate_sheets=candidate_sheets,
        selected_metric=measure,
        candidate_metrics=candidate_metrics,
        selected_dimensions=[],
        query_type="unsupported_query",
        pattern_id=None,
        matched_intent_keywords=matched_intent_keywords(question),
        auto_corrections=[],
        validation_result="blocked",
        fallback_reason=fallback_blocked_reason,
        requested_dimension=requested_dimension,
        requested_metric=requested_metric,
        resolved_dimension=None,
        resolved_metric=resolved_metric,
        missing_dimension_reason=missing_dimension_reason,
        missing_metric_reason=missing_metric_reason,
        fallback_blocked_reason=fallback_blocked_reason,
        available_dimensions=available_dimensions,
        available_metrics=available_metrics,
    )


def _question_requests_distinct_dimension_count(question: str, dimensions: list[str], group_by: list[str]) -> bool:
    q_text = str(question or "").lower()
    if not dimensions:
        return False
    if group_by and has_group_by_intent(question):
        return False
    if has_distinct_intent(q_text):
        return True
    if re.search(r"\bnumber of\b", q_text):
        return any(term in q_text for term in DISTINCT_DIMENSION_TERMS) and not any(term in q_text for term in ROW_COUNT_TERMS)
    return False


def _question_requests_row_count(question: str) -> bool:
    q_text = str(question or "").lower()
    return any(term in q_text for term in ["count records", "number of rows", "how many records", "count rows"])


def _mentions_distribution(question: str) -> bool:
    return bool(re.search(r"\bdistribution\b", str(question or "").lower()))


def _mentions_comparison(question: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|across)\b", str(question or "").lower()))


def _starts_with_projection_request(question: str) -> bool:
    return bool(re.search(r"^\s*(list|show|display|give(?:\s+me)?|return)\b", str(question or "").lower()))


def _is_summary_metric_like_column(column_name: str) -> bool:
    normalized = normalize_name(column_name)
    tokens = set(normalized.split("_"))
    if normalized in {"count", "row_count", "matched_rows", "total", "summary"}:
        return True
    return bool(tokens & {"count", "total", "sum", "avg", "average", "min", "max", "gap", "value", "summary"})


def _is_pre_aggregated_sheet(context: PlanningContext, target_sheet: str | None) -> bool:
    if not target_sheet:
        return False
    sheet_name = normalize_name(target_sheet)
    if "summary" in sheet_name or "aggregate" in sheet_name:
        return True
    numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
    return any(_is_summary_metric_like_column(column_name) for column_name in numeric_columns)


def _projection_match_position(question: str, column_name: str) -> int:
    question_text = str(question or "").lower()
    best = len(question_text) + 1000
    candidates = [str(column_name), str(column_name).replace("_", " "), normalize_name(column_name).replace("_", " ")]
    for candidate in candidates:
        candidate_text = str(candidate or "").strip().lower()
        if not candidate_text:
            continue
        match = re.search(rf"\b{re.escape(candidate_text)}\b", question_text)
        if match:
            best = min(best, match.start())
    return best


def _extract_requested_projection_phrases(question: str) -> list[str]:
    question_text = str(question or "").strip()
    if not question_text:
        return []

    match = re.search(
        r"^\s*(?:list|show|display|give(?:\s+me)?|return)\s+(.*?)(?:\s+(?:from|in|on)\s+.+)?$",
        question_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []

    requested_text = re.sub(r"\b(?:by|where)\b.*$", "", str(match.group(1) or ""), flags=re.IGNORECASE).strip(" ,")
    if not requested_text:
        return []

    phrases = [
        phrase.strip(" ,")
        for phrase in re.split(r"\s*(?:,|\band\b|&)\s*", requested_text, flags=re.IGNORECASE)
        if phrase.strip(" ,")
    ]
    return phrases


def _match_projection_phrase_to_column(phrase: str, sheet_columns: list[str]) -> str | None:
    normalized_phrase = normalize_name(phrase)
    if not normalized_phrase:
        return None

    for column_name in sheet_columns:
        if normalize_name(column_name) == normalized_phrase:
            return str(column_name)

    phrase_text = normalized_phrase.replace("_", " ")
    for column_name in sheet_columns:
        column_text = normalize_name(column_name).replace("_", " ")
        if phrase_text == column_text:
            return str(column_name)
    return None


def _detect_projection_columns(question: str, context: PlanningContext, target_sheet: str | None) -> tuple[list[str], list[str], float, bool, bool, list[str], list[str]]:
    if not target_sheet or not _starts_with_projection_request(question):
        return [], [], 0.0, False, False, [], []

    question_text = str(question or "")
    question_norm = normalize_name(question_text)
    sheet_columns = context.columns_by_sheet.get(target_sheet) or []
    pre_aggregated_sheet = _is_pre_aggregated_sheet(context, target_sheet)
    requested_projection_columns = _extract_requested_projection_phrases(question_text)
    explicit_matches: list[str] = []
    reasons: list[str] = []

    for phrase in requested_projection_columns:
        matched_column = _match_projection_phrase_to_column(phrase, sheet_columns)
        if matched_column and matched_column not in explicit_matches:
            explicit_matches.append(matched_column)
            reasons.append(f"Selected existing column '{matched_column}' from explicit projection request '{phrase}'.")

    selected: list[tuple[int, str, str]] = []
    for column_name in sheet_columns:
        normalized = normalize_name(column_name)
        if not normalized:
            continue
        matched = False
        for candidate in [str(column_name), str(column_name).replace("_", " "), normalized.replace("_", " ")]:
            candidate_text = str(candidate or "").strip().lower()
            if candidate_text and re.search(rf"\b{re.escape(candidate_text)}\b", question_text.lower()):
                matched = True
                break
        if matched or normalized in question_norm:
            selected.append((
                _projection_match_position(question_text, column_name),
                str(column_name),
                f"Selected existing column '{column_name}' for direct projection.",
            ))

    ordered_columns: list[str] = list(explicit_matches)
    for _position, column_name, reason in sorted(selected, key=lambda item: (item[0], item[1])):
        if column_name not in ordered_columns:
            ordered_columns.append(column_name)
            reasons.append(reason)

    if pre_aggregated_sheet and ordered_columns:
        reasons.append(f"Sheet '{target_sheet}' is treated as pre-aggregated.")

    confidence = 0.0
    if len(ordered_columns) >= 2:
        confidence = 0.96 if pre_aggregated_sheet else 0.91
    elif ordered_columns and pre_aggregated_sheet and any(_is_summary_metric_like_column(column) for column in ordered_columns):
        confidence = 0.88
    return ordered_columns, reasons, confidence, bool(ordered_columns), pre_aggregated_sheet, requested_projection_columns, list(ordered_columns)
    
def _question_without_explicit_sheet_context(question: str, explicit_sheet: str | None) -> str:
    question_text = str(question or "")
    sheet_name = str(explicit_sheet or "").strip()
    if not question_text or not sheet_name:
        return question_text
    variants = {
        sheet_name.lower(),
        sheet_name.lower().replace("_", " "),
        normalize_name(sheet_name).replace("_", " "),
    }
    rewritten = question_text
    for variant in sorted({value.strip() for value in variants if value.strip()}, key=len, reverse=True):
        escaped = re.escape(variant)
        rewritten = re.sub(rf"\b(?:from|in|on)\s+(?:the\s+)?{escaped}(?:\s+sheet)?\b", " ", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten or question_text


def _score_sheet(question: str, context: PlanningContext, sheet_name: str, canonical_intent: CanonicalIntent) -> tuple[float, list[str]]:
    q_norm = normalize_name(question)
    q_tokens = {token for token in q_norm.split("_") if token}
    token_variants = set(q_tokens)
    token_variants.update({token[:-1] for token in q_tokens if token.endswith("s") and len(token) > 3})
    billing_status_terms = {"under", "over", "unbilled", "billed", "lag", "lagged", "delay", "delayed", "exception", "variance"}
    score = 0.0
    reasons: list[str] = []
    if len(context.sheets) == 1:
        score += 0.25
        reasons.append("Single-sheet dataset preference.")
    if normalize_name(sheet_name) and normalize_name(sheet_name) in q_norm:
        score += 0.55
        reasons.append(f"Question mentions sheet '{sheet_name}'.")
    columns = context.columns_by_sheet.get(sheet_name) or []
    for column_name in columns:
        normalized = normalize_name(column_name)
        if normalized and normalized in q_norm:
            score += 0.06
        else:
            token_hits = len({token for token in normalized.split("_") if token} & token_variants)
            if token_hits:
                score += min(0.08, token_hits * 0.03)
    for role_name, mapped_column in (context.semantic_roles.get(sheet_name) or {}).items():
        if role_matches_question(role_name, question):
            score += 0.12
            reasons.append(f"Role '{role_name}' matched on '{sheet_name}'.")
        for metric_family in canonical_intent.metric_families:
            if family_matches_column_or_role(metric_family, str(mapped_column), role_name):
                score += 0.08
                reasons.append(f"Metric family '{metric_family}' fits sheet '{sheet_name}'.")
                break
        for dimension_family in canonical_intent.dimension_families:
            if family_matches_column_or_role(dimension_family, str(mapped_column), role_name):
                score += 0.06
                reasons.append(f"Dimension family '{dimension_family}' fits sheet '{sheet_name}'.")
                break
    status_like_role_or_column = any(
        role_name in {"status", "billing_status", "exception_type"}
        or {"status", "state", "stage", "exception"} & set(normalize_name(mapped_column).split("_"))
        for role_name, mapped_column in (context.semantic_roles.get(sheet_name) or {}).items()
    ) or any(
        {"status", "state", "stage", "exception"} & set(normalize_name(column_name).split("_"))
        for column_name in columns
    )
    if token_variants & billing_status_terms and status_like_role_or_column:
        score += 0.12
        reasons.append(f"Billing-status wording aligned with '{sheet_name}'.")
    sampled_values = context.sampled_categorical_values_by_sheet.get(sheet_name) or {}
    sample_hits = 0
    status_sample_hits = 0
    for values in sampled_values.values():
        for sample_value in values[:25]:
            normalized_value = normalize_name(sample_value)
            if normalized_value and normalized_value in q_norm:
                sample_hits += 1
    for column_name, values in sampled_values.items():
        normalized_column = normalize_name(column_name)
        role_name = _role_for_column(context, sheet_name, str(column_name))
        status_like_column = bool(
            {"status", "state", "stage", "exception", "billing", "variance", "gap"} & set(normalized_column.split("_"))
            or role_name in {"status", "billing_status", "exception_type"}
        )
        if not status_like_column:
            continue
        for sample_value in values[:25]:
            normalized_value = normalize_name(sample_value)
            if normalized_value and normalized_value in q_norm:
                status_sample_hits += 1
    if sample_hits:
        score += min(0.18, sample_hits * 0.04)
        reasons.append(f"Sample values matched on '{sheet_name}'.")
    if status_sample_hits:
        score += min(0.16, status_sample_hits * 0.08)
        reasons.append(f"Status-like sample values matched on '{sheet_name}'.")
    if "order" in token_variants and any("order" in normalize_name(column_name) for column_name in columns):
        score += 0.18
        reasons.append("Order-related wording aligned with sheet columns.")
    if "ticket" in token_variants and any("ticket" in normalize_name(column_name) for column_name in columns):
        score += 0.18
        reasons.append("Ticket-related wording aligned with sheet columns.")
    return score, reasons


def _pick_target_sheet(question: str, context: PlanningContext, explicit_sheet: str | None, canonical_intent: CanonicalIntent) -> tuple[str | None, float, list[str], list[str], list[dict[str, Any]]]:
    if explicit_sheet:
        return explicit_sheet, 1.0, [f"Explicit sheet '{explicit_sheet}' requested."], [], [{"sheet": explicit_sheet, "score": 1.0, "reasons": [f"Explicit sheet '{explicit_sheet}' requested."]}]
    if not context.sheets:
        return None, 0.0, [], ["No sheets available in planning context."], []
    scored = []
    for sheet_name in context.sheets:
        score, reasons = _score_sheet(question, context, sheet_name, canonical_intent)
        scored.append((score, sheet_name, reasons))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_sheet, best_reasons = scored[0]
    warnings: list[str] = []
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) <= 0.10:
        warnings.append(f"Target sheet ambiguous between {scored[0][1]} and {scored[1][1]}.")
        best_score = max(0.0, best_score - 0.15)
    candidates = [{"sheet": sheet_name, "score": round(score, 3), "reasons": reasons} for score, sheet_name, reasons in scored]
    return best_sheet, best_score, best_reasons, warnings, candidates


def _dimension_match_position(question: str, context: PlanningContext, target_sheet: str, column_name: str) -> int:
    question_text = str(question or "").lower()
    best = len(question_text) + 1000
    candidates = [str(column_name), column_name.replace("_", " "), normalize_name(column_name).replace("_", " ")]
    role_name = _role_for_column(context, target_sheet, column_name)
    if role_name:
        candidates.extend(semantic_aliases(role_name))
    for candidate in candidates:
        candidate_text = str(candidate or "").strip().lower()
        if not candidate_text:
            continue
        match = re.search(rf"\b{re.escape(candidate_text)}\b", question_text)
        if match:
            best = min(best, match.start())
    return best


def _detect_dimensions(question: str, context: PlanningContext, target_sheet: str | None, canonical_intent: CanonicalIntent) -> tuple[list[str], list[str], float]:
    if not target_sheet:
        return [], [], 0.0
    q_norm = normalize_name(question)
    reasons: list[str] = []
    columns = context.columns_by_sheet.get(target_sheet) or []
    allowed_dimensions = set(context.categorical_columns_by_sheet.get(target_sheet) or []) | set(context.date_columns_by_sheet.get(target_sheet) or [])
    if canonical_intent.query_shape == "list_rows":
        allowed_dimensions |= set(context.identifier_columns_by_sheet.get(target_sheet) or [])
    scored_dimensions: list[tuple[float, int, str, str]] = []
    for column_name in columns:
        if column_name not in allowed_dimensions:
            continue
        role_name = _role_for_column(context, target_sheet, column_name)
        score = 0.0
        column_norm = normalize_name(column_name)
        if column_norm and column_norm in q_norm:
            score += 0.40
        for family in canonical_intent.dimension_families:
            if family_matches_column_or_role(family, column_name, role_name):
                score += 0.38
        if canonical_intent.query_shape in {"aggregate_grouped", "comparison", "list_rows"} and re.search(rf"\bby\s+{re.escape(column_norm.replace('_', ' '))}\b", question.lower()):
            score += 0.35
        if canonical_intent.query_shape in {"aggregate_grouped", "comparison", "list_rows"} and re.search(rf"\bfor each\s+{re.escape(column_norm.replace('_', ' '))}\b", question.lower()):
            score += 0.35
        if canonical_intent.query_shape in {"aggregate_grouped", "comparison", "list_rows"} and re.search(rf"\b{re.escape(column_norm.replace('_', ' '))}\s+wise\b", question.lower()):
            score += 0.35
        if canonical_intent.query_shape == "trend_over_time" and (role_name in {"date", "service_date"} or column_name in (context.date_columns_by_sheet.get(target_sheet) or [])):
            score += 0.45
        if score >= 0.34:
            scored_dimensions.append((score, _dimension_match_position(question, context, target_sheet, column_name), column_name, f"Detected dimension '{column_name}' deterministically."))

    dimensions: list[str] = []
    for _score, _position, column_name, reason in sorted(scored_dimensions, key=lambda item: (-item[0], item[1], item[2])):
        if column_name not in dimensions:
            dimensions.append(column_name)
            reasons.append(reason)

    if "invoice status" in str(question or "").lower():
        filtered_dimensions = [column_name for column_name in dimensions if normalize_name(column_name) != "status"]
        if filtered_dimensions != dimensions:
            dimensions = filtered_dimensions
            reasons.append("Dropped generic 'status' dimension because the question explicitly requested invoice status.")

    if not dimensions and canonical_intent.query_shape in {"aggregate_grouped", "comparison", "list_rows"}:
        fallback_dimensions = []
        for role_name, mapped_column in (context.semantic_roles.get(target_sheet) or {}).items():
            if role_matches_question(role_name, question) and str(mapped_column) in allowed_dimensions:
                fallback_dimensions.append((_dimension_match_position(question, context, target_sheet, str(mapped_column)), str(mapped_column), f"Used semantic role '{role_name}' to select dimension '{mapped_column}'."))
        for _position, column_name, reason in sorted(fallback_dimensions, key=lambda item: (item[0], item[1])):
            if column_name not in dimensions:
                dimensions.append(column_name)
                reasons.append(reason)

    confidence = min(0.9, 0.45 + (0.18 * len(dimensions))) if dimensions else 0.0
    return dimensions, reasons, confidence


def _preferred_metric_from_sheet(question: str, context: PlanningContext, target_sheet: str | None) -> dict[str, Any] | None:
    if not target_sheet:
        return None
    derived_metrics = _derived_metrics_for_sheet(context, target_sheet)
    revenue_formula = derived_metrics.get("revenue")
    if revenue_formula and _question_explicitly_requests_derived_metric(question, "revenue"):
        return {"column": "revenue", "aggregation": "sum", "alias": _build_measure_alias("sum", "revenue"), "formula": revenue_formula, "derived": True}
    numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
    if not numeric_columns:
        return None
    normalized_map = {normalize_name(column): column for column in numeric_columns}
    if "amount" in normalized_map:
        return {"column": normalized_map["amount"], "aggregation": "sum", "alias": _build_measure_alias("sum", normalized_map["amount"])}
    role_map = context.semantic_roles.get(target_sheet) or {}
    for preferred_role in ["revenue", "amount", "expected_amount", "billed_amount", "value_gap", "profit", "quantity"]:
        mapped = str(role_map.get(preferred_role) or "")
        if mapped and mapped in numeric_columns:
            return {"column": mapped, "aggregation": "sum", "alias": _build_measure_alias("sum", mapped)}
    if len(numeric_columns) == 1:
        return {"column": numeric_columns[0], "aggregation": "sum", "alias": _build_measure_alias("sum", numeric_columns[0])}
    return None


def _pick_measure(question: str, context: PlanningContext, target_sheet: str | None, canonical_intent: CanonicalIntent) -> tuple[dict[str, Any] | None, list[str], float, list[str], list[dict[str, Any]]]:
    if not target_sheet:
        return None, [], 0.0, ["No target sheet available for measure detection."], []
    numeric_columns = context.numeric_columns_by_sheet.get(target_sheet) or []
    warnings: list[str] = []
    reasons: list[str] = []
    if canonical_intent.aggregation == "count":
        return {"column": "*", "aggregation": "count", "alias": "row_count"}, ["Detected count intent."], 0.95, warnings, [{"column": "*", "score": 0.95, "aggregation": "count", "reasons": ["Count intent"]}]

    requested_measure, requested_reasons, requested_warnings, requested_candidates = _resolve_requested_measure(
        question,
        context,
        target_sheet,
        canonical_intent.aggregation or "sum",
    )
    if requested_candidates:
        reasons.extend(requested_reasons)
        warnings.extend(requested_warnings)
        if requested_measure is not None:
            return requested_measure, reasons, 0.96, warnings, requested_candidates
        return None, reasons, 0.0, warnings, requested_candidates

    candidate_metrics: list[dict[str, Any]] = []
    for column_name in numeric_columns:
        role_name = _role_for_column(context, target_sheet, column_name)
        score = 0.0
        metric_reasons: list[str] = []
        if normalize_name(column_name) == "amount" and "amount" in normalize_name(question):
            score += 0.8
            metric_reasons.append("Exact 'Amount' column matched the user question.")
        for family in canonical_intent.metric_families:
            if family_matches_column_or_role(family, column_name, role_name):
                score += 0.48
                metric_reasons.append(f"Matched metric family '{family}'.")
            preferred_candidates = MEASURE_PRIORITY.get(family) or []
            for index, candidate_name in enumerate(preferred_candidates):
                candidate_norm = normalize_name(candidate_name)
                column_norm = normalize_name(column_name)
                if candidate_norm and (candidate_norm == column_norm or candidate_norm in column_norm or column_norm in candidate_norm):
                    score += max(0.0, 0.22 - (index * 0.03))
                    metric_reasons.append(f"Preferred deterministic metric candidate '{candidate_name}'.")
                    break
        if role_name and role_matches_question(role_name, question):
            score += 0.18
            metric_reasons.append(f"Question mentions role '{role_name}'.")
        normalized_column = normalize_name(column_name)
        if normalized_column and normalized_column in normalize_name(question):
            score += 0.18
            metric_reasons.append(f"Question mentions column '{column_name}'.")
        if len(numeric_columns) == 1:
            score += 0.22
            metric_reasons.append("Only numeric column available on sheet.")
        if score > 0:
            candidate_metrics.append({"column": column_name, "score": round(score, 3), "aggregation": canonical_intent.aggregation or "sum", "reasons": metric_reasons, "requested_metric": None, "canonical_metric": normalize_name(column_name), "resolved_metric_type": "physical"})
    candidate_metrics.sort(key=lambda item: item["score"], reverse=True)
    if candidate_metrics:
        best_metric = candidate_metrics[0]
        if len(candidate_metrics) > 1 and abs(candidate_metrics[0]["score"] - candidate_metrics[1]["score"]) <= 0.10:
            warnings.append("Measure ambiguous across numeric columns: " + ", ".join([str(item["column"]) for item in candidate_metrics[:4]]) + ".")
        resolved = str(best_metric["column"])
        aggregation = canonical_intent.aggregation or "sum"
        reasons.append(f"Selected measure '{resolved}' using deterministic business metric scoring.")
        selected_measure = _resolved_measure_payload(
            column_name=resolved,
            aggregation=aggregation,
            requested_metric=normalize_name(resolved),
            canonical_metric=normalize_name(resolved),
            resolved_metric_type="physical",
        )
        if best_metric.get("formula"):
            selected_measure["formula"] = str(best_metric["formula"])
            selected_measure["derived"] = True
        return selected_measure, reasons, min(0.96, float(best_metric["score"]) + 0.18), warnings, candidate_metrics
    fallback_metric = _preferred_metric_from_sheet(question, context, target_sheet)
    if fallback_metric is not None:
        reasons.append(f"Defaulted to primary business metric '{fallback_metric['column']}'.")
        return fallback_metric, reasons, 0.64, warnings, [{"column": fallback_metric["column"], "score": 0.64, "aggregation": fallback_metric["aggregation"], "reasons": reasons.copy()}]
    if len(numeric_columns) > 1:
        warnings.append(f"Measure ambiguous across numeric columns: {', '.join(numeric_columns[:6])}.")
    return None, reasons, 0.0, warnings, candidate_metrics


def _mapping_to_filter_dict(mapping: ValueMapping) -> dict[str, Any]:
    return {"column": mapping.column, "operator": mapping.operator, "value": mapping.value, "kind": mapping.kind, "confidence": mapping.confidence, "sheet": mapping.sheet, "source": mapping.source}


def _resolve_year_filter_column(context: PlanningContext, target_sheet: str | None) -> str | None:
    if not target_sheet:
        return None
    date_columns = context.date_columns_by_sheet.get(target_sheet) or []
    if date_columns:
        return str(date_columns[0])
    columns = context.columns_by_sheet.get(target_sheet) or []
    for column_name in columns:
        if normalize_name(column_name) in {"year", "fiscal_year", "calendar_year"}:
            return str(column_name)
    return None


def _extract_year_filters(filter_candidates: list[Any], context: PlanningContext, target_sheet: str | None) -> list[dict[str, Any]]:
    year_column = _resolve_year_filter_column(context, target_sheet)
    if not year_column:
        return []
    extracted: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in filter_candidates:
        if str(getattr(candidate, "kind", "") or "").lower() != "year":
            continue
        operator = str(getattr(candidate, "operator", "=") or "=")
        raw_value = str(getattr(candidate, "raw_value", "") or "").strip()
        if not raw_value:
            continue
        key = (year_column, operator, raw_value)
        if key in seen:
            continue
        seen.add(key)
        extracted.append({"column": year_column, "operator": operator, "value": raw_value, "kind": "year", "sheet": target_sheet, "source": str(getattr(candidate, "source", "year_phrase") or "year_phrase"), "confidence": float(getattr(candidate, "confidence", 0.9) or 0.9)})
    return extracted


def _dedupe_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for filter_spec in filters:
        if not isinstance(filter_spec, dict):
            continue
        value = filter_spec.get("value")
        normalized_value = str(tuple(value)) if isinstance(value, list) else str(value)
        key = (str(filter_spec.get("sheet") or ""), str(filter_spec.get("column") or ""), str(filter_spec.get("operator") or "="), normalized_value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(filter_spec)
    return deduped


def _collapse_mappings_to_filters(mappings: list[ValueMapping], *, query_shape: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    reasons: list[str] = []
    if query_shape != "comparison":
        return [_mapping_to_filter_dict(mapping) for mapping in mappings], warnings, reasons
    grouped: dict[tuple[str, str], list[ValueMapping]] = {}
    for mapping in mappings:
        grouped.setdefault((mapping.sheet, mapping.column), []).append(mapping)
    collapsed: list[dict[str, Any]] = []
    for (sheet_name, column_name), group in grouped.items():
        unique_values = []
        seen_values: set[str] = set()
        for mapping in group:
            normalized_value = str(mapping.value)
            if normalized_value in seen_values:
                continue
            seen_values.add(normalized_value)
            unique_values.append(mapping)
        if len(unique_values) >= 2:
            collapsed.append({"column": column_name, "operator": "in", "value": [mapping.value for mapping in unique_values], "kind": unique_values[0].kind, "confidence": max(mapping.confidence for mapping in unique_values), "sheet": sheet_name, "source": "comparison_value_merge"})
            reasons.append(f"Collapsed comparison values for '{column_name}' into an IN filter.")
        else:
            collapsed.append(_mapping_to_filter_dict(unique_values[0]))
    return collapsed, warnings, reasons


def _has_single_strong_execution_path(canonical_intent: CanonicalIntent, *, target_confidence: float, measure_confidence: float, dimension_confidence: float, candidate_sheets: list[dict[str, Any]], candidate_metrics: list[dict[str, Any]], dimensions: list[str]) -> bool:
    if canonical_intent.query_shape == "unknown" or not candidate_sheets:
        return False
    top_sheet_gap = 1.0 if len(candidate_sheets) == 1 else float(candidate_sheets[0]["score"]) - float(candidate_sheets[1]["score"])
    top_metric_gap = 1.0 if len(candidate_metrics) <= 1 else float(candidate_metrics[0]["score"]) - float(candidate_metrics[1]["score"])
    strong_sheet = target_confidence >= 0.4 and top_sheet_gap >= 0.08
    strong_metric = canonical_intent.aggregation == "count" or (measure_confidence >= 0.45 and (top_metric_gap >= 0.0 or len(candidate_metrics) <= 1))
    strong_dimension = canonical_intent.query_shape in {"aggregate_scalar", "unknown"} or bool(dimensions) or dimension_confidence >= 0.42
    return strong_sheet and strong_metric and strong_dimension


def _ordered_dimensions(question: str, context: PlanningContext, target_sheet: str | None, dimensions: list[str]) -> list[str]:
    if not target_sheet:
        return dimensions
    return list(dict.fromkeys(sorted(dimensions, key=lambda column: (_dimension_match_position(question, context, target_sheet, column), column))))


def _detect_partition_dimensions(question: str, context: PlanningContext, target_sheet: str | None, dimensions: list[str]) -> list[str]:
    if not target_sheet:
        return []
    q_text = str(question or "").lower()
    partition_hits: list[tuple[int, str]] = []
    for column_name in dimensions:
        role_name = _role_for_column(context, target_sheet, column_name)
        aliases = [column_name.replace("_", " "), normalize_name(column_name).replace("_", " ")]
        if role_name:
            aliases.extend(semantic_aliases(role_name))
        best_position = None
        for alias in aliases:
            alias_text = str(alias or "").strip().lower()
            if not alias_text:
                continue
            match = re.search(rf"\b(?:for each|per)\s+{re.escape(alias_text)}\b", q_text)
            if match:
                best_position = match.start() if best_position is None else min(best_position, match.start())
        if best_position is not None:
            partition_hits.append((best_position, column_name))
    return [column_name for _position, column_name in sorted(partition_hits, key=lambda item: (item[0], item[1]))]


def _detect_rank_dimension(question: str, context: PlanningContext, target_sheet: str | None, dimensions: list[str], partition_by: list[str], metric_column: str | None) -> str | None:
    if not target_sheet:
        return None
    q_text = str(question or "").lower()
    ranking_candidates = [dimension for dimension in dimensions if dimension not in partition_by]
    metric_phrase = normalize_name(metric_column or "").replace("_", " ")
    best_match: tuple[int, str] | None = None
    for column_name in ranking_candidates:
        role_name = _role_for_column(context, target_sheet, column_name)
        aliases = [column_name.replace("_", " "), normalize_name(column_name).replace("_", " ")]
        if role_name:
            aliases.extend(semantic_aliases(role_name))
        for alias in aliases:
            alias_text = str(alias or "").strip().lower()
            if not alias_text:
                continue
            patterns = [rf"\b(?:top|bottom|highest|lowest|best|worst|least)\s+{re.escape(alias_text)}\b", rf"\b{re.escape(alias_text)}\b\s+(?:for each|per)\b"]
            if metric_phrase:
                patterns.append(rf"\b(?:highest|lowest|top|bottom|best|worst)\s+{re.escape(metric_phrase)}\s+{re.escape(alias_text)}\b")
            for pattern in patterns:
                match = re.search(pattern, q_text)
                if match:
                    current = (match.start(), column_name)
                    if best_match is None or current < best_match:
                        best_match = current
    return best_match[1] if best_match is not None else None


def _determine_aggregation(question: str, keyword_map: dict[str, list[str]], default: str = "sum") -> str:
    if keyword_map.get("distinct"):
        return "count_distinct"
    if keyword_map.get("avg"):
        return "avg"
    if keyword_map.get("max"):
        return "max"
    if keyword_map.get("min"):
        return "min"
    if keyword_map.get("sum"):
        return "sum"
    if keyword_map.get("count"):
        return "count"
    return default


def _validate_query_plan(question: str, plan: QueryPlan, context: PlanningContext) -> QueryPlan:
    issues: list[str] = []
    corrections: list[str] = []
    target_sheet = str(plan.target_sheet or "")
    columns = set(context.columns_by_sheet.get(target_sheet) or [])
    numeric_columns = set(context.numeric_columns_by_sheet.get(target_sheet) or [])
    derived_metrics = set((context.derived_metrics_by_sheet.get(target_sheet) or {}).keys())
    keyword_map = plan.matched_keywords or matched_intent_keywords(question)
    for column_name in plan.columns:
        if column_name and column_name not in columns:
            issues.append(f"Projection column '{column_name}' does not exist on sheet '{target_sheet}'.")
    for column_name in plan.group_by + plan.partition_by + ([plan.rank_dimension] if plan.rank_dimension else []):
        if column_name and column_name not in columns:
            issues.append(f"Column '{column_name}' does not exist on sheet '{target_sheet}'.")
    if plan.metric_column and plan.metric_column != "*" and not plan.metric_formula and plan.metric_column not in columns:
        issues.append(f"Metric column '{plan.metric_column}' does not exist on sheet '{target_sheet}'.")
    if plan.metric_formula:
        parsed_formula = _parse_metric_formula(plan.metric_formula)
        if parsed_formula is None:
            issues.append(f"Metric formula '{plan.metric_formula}' is not a supported binary expression.")
        else:
            left_operand, _operator, right_operand = parsed_formula
            for operand in [left_operand, right_operand]:
                if not _context_metric_reference_is_numeric(context, target_sheet, operand):
                    issues.append(f"Metric formula operand '{operand}' is not a numeric column or derived metric on sheet '{target_sheet}'.")
    elif plan.metric_column in derived_metrics and plan.metric_column != "*":
        issues.append(f"Derived metric '{plan.metric_column}' is missing its formula metadata.")
    if plan.aggregation in {"sum", "avg", "max", "min"} and plan.metric_column not in numeric_columns and not plan.metric_formula:
        issues.append(f"Aggregation '{plan.aggregation}' requires a numeric metric column.")
    for filter_spec in plan.filters:
        column_name = str(filter_spec.get("column") or "")
        if column_name and column_name not in columns:
            issues.append(f"Filter column '{column_name}' does not exist on sheet '{target_sheet}'.")
        filter_value = filter_spec.get("value")
        if filter_value is None or filter_value == "" or filter_value == []:
            issues.append(f"Filter column '{column_name}' could not be mapped to a concrete value.")
    if plan.query_type == "describe_sheet":
        plan.aggregation = None
        plan.metric_column = None
        plan.metric_sheet = None
        plan.columns = []
        plan.group_by = []
        plan.partition_by = []
        plan.rank_dimension = None
        plan.filters = []
        plan.answer_type = "sheet_profile"
        plan.pre_aggregated_sheet = _is_pre_aggregated_sheet(context, plan.target_sheet)
        plan.aggregation_override_skipped_reason = "non_aggregate_intent_family"
        plan.skip_aggregate_pattern_matching = True
        plan.intent_family = plan.intent_family or "describe_sheet"
        plan.validation_issues = issues
        plan.validation_result = "invalid" if issues else "valid"
        return plan
    if plan.query_type == "row_preview":
        plan.aggregation = None
        plan.metric_column = None
        plan.metric_sheet = None
        plan.group_by = []
        plan.partition_by = []
        plan.rank_dimension = None
        plan.answer_type = "table"
        if not plan.columns:
            plan.columns = list(context.columns_by_sheet.get(target_sheet) or [])
        plan.selected_columns_exist = bool(plan.columns) and all(column_name in columns for column_name in plan.columns)
        plan.matched_existing_columns = [column_name for column_name in plan.columns if column_name in columns]
        plan.aggregation_override_skipped_reason = "non_aggregate_intent_family"
        plan.skip_aggregate_pattern_matching = True
        plan.intent_family = plan.intent_family or "row_preview"
        plan.validation_issues = issues
        plan.validation_result = "invalid" if issues else "valid"
        return plan
    if plan.query_type == "projection":
        plan.aggregation = None
        plan.metric_column = None
        plan.metric_sheet = None
        plan.group_by = []
        plan.partition_by = []
        plan.rank_dimension = None
        plan.answer_type = "table"
        plan.selected_columns_exist = bool(plan.columns) and all(column_name in columns for column_name in plan.columns)
        plan.matched_existing_columns = [column_name for column_name in plan.columns if column_name in columns]
        if not plan.requested_projection_columns:
            plan.requested_projection_columns = list(plan.columns)
        if not plan.matched_existing_columns:
            plan.matched_existing_columns = list(plan.columns)
        plan.aggregation_override_skipped_reason = "projection_query_selected_existing_columns"
        plan.skip_aggregate_pattern_matching = True
        plan.intent_family = plan.intent_family or "projection"
        plan.validation_issues = issues
        plan.validation_result = "invalid" if issues else "valid"
        return plan
    if plan.pattern_id in {"pattern_12_top_n_by_metric", "pattern_13_bottom_n_by_metric"}:
        expected_ranked_n_aggregation = top_n_aggregation_for_question(question) if plan.pattern_id == "pattern_12_top_n_by_metric" else bottom_n_aggregation_for_question(question)
        if plan.aggregation != expected_ranked_n_aggregation:
            plan.aggregation = expected_ranked_n_aggregation
            corrections.append(f"Aligned ranked-N aggregation to '{expected_ranked_n_aggregation}' from the explicit deterministic {plan.pattern_id} rule.")
    if plan.pattern_id == "pattern_17_top_member_for_each_parent":
        ranking_direction = plan.order_direction or best_or_worst_ranking_direction(question) or "desc"
        expected_ranking_aggregation = best_or_worst_ranking_aggregation_for_question(question, ranking_direction)
        if plan.aggregation != expected_ranking_aggregation and not ((keyword_map.get("max") or keyword_map.get("min")) and plan.aggregation == expected_ranking_aggregation):
            plan.aggregation = expected_ranking_aggregation
            corrections.append(f"Aligned per-parent ranking aggregation to '{expected_ranking_aggregation}' from the explicit deterministic pattern_17 rule.")
    top_only_max_intent = bool(keyword_map.get("max")) and all(keyword == "top" for keyword in keyword_map.get("max", []))
    if plan.pattern_id not in {"pattern_12_top_n_by_metric", "pattern_13_bottom_n_by_metric"} and keyword_map.get("max") and plan.aggregation == "sum" and not (plan.group_by and top_only_max_intent):
        plan.aggregation = "max"
        corrections.append("Overrode SUM to MAX because the question contains maximum intent.")
    if keyword_map.get("min") and plan.aggregation == "sum":
        plan.aggregation = "min"
        corrections.append("Overrode SUM to MIN because the question contains minimum intent.")
    if keyword_map.get("avg") and plan.aggregation != "avg":
        plan.aggregation = "avg"
        corrections.append("Overrode aggregation to AVG because the question contains average intent.")
    if keyword_map.get("count") and plan.aggregation not in {"count", "count_distinct"}:
        plan.aggregation = "count"
        corrections.append("Overrode aggregation to COUNT because the question contains count intent.")
    if keyword_map.get("distinct") and plan.aggregation != "count_distinct":
        plan.aggregation = "count_distinct"
        corrections.append("Overrode aggregation to COUNT DISTINCT because the question contains distinct intent.")
    plan.auto_corrections.extend(corrections)
    plan.validation_issues = issues
    plan.validation_result = "invalid" if issues else ("corrected" if corrections else "valid")
    return plan


def _measure_with_plan_aggregation(plan: QueryPlan) -> dict[str, Any] | None:
    if plan.aggregation is None:
        return None
    if plan.aggregation == "count" and plan.metric_column == "*":
        return {"column": "*", "aggregation": "count", "alias": "row_count", "sheet": plan.metric_sheet or plan.target_sheet}
    if not plan.metric_column:
        return None
    measure: dict[str, Any] = {"column": plan.metric_column, "aggregation": plan.aggregation, "alias": _build_measure_alias(plan.aggregation, plan.metric_column), "sheet": plan.metric_sheet or plan.target_sheet}
    if plan.metric_formula:
        measure["formula"] = plan.metric_formula
        measure["derived"] = True
    if plan.requested_metric_name:
        measure["requested_metric"] = plan.requested_metric_name
    if plan.canonical_metric_name:
        measure["canonical_metric"] = plan.canonical_metric_name
    if plan.resolved_metric_type:
        measure["resolved_metric_type"] = plan.resolved_metric_type
    return measure


def _query_spec_from_plan(plan: QueryPlan, confidence: float, warnings: list[str]) -> QuerySpec:
    measure = _measure_with_plan_aggregation(plan)
    order_by: list[dict[str, Any]] = []
    if plan.query_shape == "trend_over_time" and plan.time_grain:
        order_by = [{"field": plan.time_grain, "direction": "asc"}]
    elif plan.query_type == "aggregate" and measure is not None and (plan.group_by or plan.requested_top_n):
        order_by = [{"field": str(measure.get("alias") or "value"), "direction": plan.order_direction}]
    elif plan.query_type == "ranking" and measure is not None:
        order_by = [{"field": str(measure.get("alias") or "value"), "direction": plan.order_direction}]
    query_filters = []
    for item in plan.filters:
        next_filter = {
            "column": item["column"],
            "operator": item["operator"],
            "value": item["value"],
            **({"kind": item["kind"]} if item.get("kind") else {}),
        }
        if item.get("sheet") and item.get("sheet") != plan.target_sheet:
            next_filter["sheet"] = item["sheet"]
        query_filters.append(next_filter)
    query_dimensions = [] if plan.query_type in {"projection", "describe_sheet", "row_preview"} else list(plan.group_by)
    if plan.query_type == "list":
        filtered_dimension_names = {
            str(item.get("column") or "")
            for item in plan.filters
            if isinstance(item, dict) and not _is_status_like_column_name(str(item.get("column") or ""))
        }
        query_dimensions = [dimension for dimension in query_dimensions if str(dimension) not in filtered_dimension_names]
    return QuerySpec(
        intent=plan.canonical_intent,
        query_type=plan.query_type,
        pattern_id=plan.pattern_id,
        target_sheet=plan.target_sheet,
        columns=list(plan.columns),
        measures=[measure] if measure is not None else [],
        dimensions=query_dimensions,
        filters=query_filters,
        order_by=order_by,
        limit=None if plan.query_type == "describe_sheet" else plan.requested_top_n,
        time_grain=plan.time_grain,
        partition_by=list(plan.partition_by),
        rank_dimension=plan.rank_dimension,
        rank_limit=1 if plan.query_type == "ranking" else None,
        confidence=max(0.0, min(1.0, confidence)),
        warnings=warnings,
        debug={
            "pattern_id": plan.pattern_id,
            "matched_keywords": plan.matched_keywords,
            "auto_corrections": plan.auto_corrections,
            "validation_result": plan.validation_result,
            "selected_columns": list(plan.columns),
            "selected_columns_exist": plan.selected_columns_exist,
            "pre_aggregated_sheet": plan.pre_aggregated_sheet,
            "requested_projection_columns": list(plan.requested_projection_columns),
            "matched_existing_columns": list(plan.matched_existing_columns),
            "aggregation_override_skipped_reason": plan.aggregation_override_skipped_reason,
            "skip_aggregate_pattern_matching": plan.skip_aggregate_pattern_matching,
            "intent_family": plan.intent_family,
            "requested_metric": plan.requested_metric_name,
            "canonical_metric": plan.canonical_metric_name,
            "resolved_metric_type": plan.resolved_metric_type,
            "derived_formula": plan.metric_formula,
        },
        source="phase2_planner_deterministic",
    )


def _build_deterministic_plan(question: str, context: PlanningContext, canonical_intent: CanonicalIntent, target_sheet: str | None, dimensions: list[str], extracted_filters: list[dict[str, Any]], measure: dict[str, Any] | None, *, max_rows: int) -> QueryPlan | None:
    if not target_sheet or canonical_intent.name == "unsupported":
        return None
    keyword_map = matched_intent_keywords(question)
    top_n, explicit_direction = _requested_top_n(question)
    best_worst_direction = best_or_worst_ranking_direction(question)
    q_text = str(question or "").lower()
    ordered_dimensions = _ordered_dimensions(question, context, target_sheet, dimensions)
    filtered_columns = {item["column"] for item in extracted_filters if isinstance(item, dict)}
    group_by = [dimension for dimension in ordered_dimensions if dimension not in filtered_columns]
    comparison_filter_columns = [str(item.get("column") or "") for item in extracted_filters if isinstance(item, dict) and str(item.get("operator") or "").lower() == "in"]
    if canonical_intent.query_shape == "comparison" and not group_by:
        for column_name in comparison_filter_columns:
            if column_name and column_name not in group_by:
                group_by.append(column_name)
    projection_columns, _projection_reasons, _projection_confidence, selected_columns_exist, pre_aggregated_sheet, requested_projection_columns, matched_existing_columns = _detect_projection_columns(question, context, target_sheet)

    if canonical_intent.query_shape == "describe_sheet":
        return QueryPlan(
            "pattern_describe_sheet",
            "describe_sheet",
            "describe_sheet",
            "sheet_profile",
            "describe_sheet",
            target_sheet,
            matched_keywords=keyword_map,
            pre_aggregated_sheet=pre_aggregated_sheet,
            skip_aggregate_pattern_matching=True,
            intent_family="describe_sheet",
        )
    if canonical_intent.query_shape == "row_preview":
        return QueryPlan(
            "pattern_row_preview",
            "row_preview",
            "row_preview",
            "table",
            "row_preview",
            target_sheet,
            columns=list(context.columns_by_sheet.get(target_sheet) or []),
            filters=extracted_filters,
            requested_top_n=_requested_row_preview_limit(question, max_rows),
            matched_keywords=keyword_map,
            aggregation_override_skipped_reason="non_aggregate_intent_family",
            skip_aggregate_pattern_matching=True,
            intent_family="row_preview",
        )
    default_limit = min(max(1, int(max_rows)), top_n or canonical_intent.requested_top_n or max(1, int(max_rows)))
    explicit_metric = str((measure or {}).get("column") or "")
    explicit_formula = str((measure or {}).get("formula") or "").strip() or None
    aggregation = _determine_aggregation(question, keyword_map, default=str(canonical_intent.aggregation or (measure or {}).get("aggregation") or "sum"))
    explicit_max_keywords = [keyword for keyword in keyword_map.get("max", []) if keyword not in {"top"}]
    if canonical_intent.name == "top_n_x_by_y" and not any([
        keyword_map.get("avg"),
        explicit_max_keywords,
        keyword_map.get("min"),
        keyword_map.get("count"),
        keyword_map.get("distinct"),
        keyword_map.get("sum"),
    ]):
        aggregation = "sum"
    if aggregation == "count" and explicit_metric and explicit_metric != "*" and canonical_intent.name not in {"count_of_x", "count_of_x_where_y"} and not _question_requests_row_count(question):
        aggregation = "sum"
    partition_by = _detect_partition_dimensions(question, context, target_sheet, ordered_dimensions)
    rank_dimension = _detect_rank_dimension(question, context, target_sheet, ordered_dimensions, partition_by, str((measure or {}).get("column") or ""))
    if group_by and keyword_map.get("max") and not explicit_max_keywords and not any([
        keyword_map.get("avg"),
        keyword_map.get("min"),
        keyword_map.get("count"),
        keyword_map.get("distinct"),
        keyword_map.get("sum"),
    ]):
        aggregation = "sum"

    if canonical_intent.query_shape == "projection" and projection_columns:
        return QueryPlan(
            "pattern_projection_existing_columns",
            "projection",
            "projection",
            "table",
            "projection",
            target_sheet,
            columns=projection_columns,
            filters=extracted_filters,
            requested_top_n=default_limit,
            matched_keywords=keyword_map,
            selected_columns_exist=selected_columns_exist,
            pre_aggregated_sheet=pre_aggregated_sheet,
            requested_projection_columns=requested_projection_columns,
            matched_existing_columns=matched_existing_columns,
            skip_aggregate_pattern_matching=True,
            intent_family="projection",
        )

    if canonical_intent.query_shape == "list_rows":
        ordered_dimensions = _augment_list_dimensions(question, context, target_sheet, ordered_dimensions)
        return QueryPlan(
            "pattern_list_rows",
            "list",
            canonical_intent.name,
            canonical_intent.answer_type,
            "list_rows",
            target_sheet,
            None,
            None,
            None,
            group_by=ordered_dimensions,
            filters=extracted_filters,
            requested_top_n=default_limit,
            matched_keywords=keyword_map,
            intent_family="list_rows",
        )

    if canonical_intent.query_shape == "trend_over_time":
        return QueryPlan(
            "pattern_trend_over_time",
            "aggregate",
            canonical_intent.name,
            canonical_intent.answer_type,
            "trend_over_time",
            target_sheet,
            aggregation,
            explicit_metric if aggregation != "count" else "*",
            target_sheet,
            filters=extracted_filters,
            requested_top_n=default_limit,
            time_grain=canonical_intent.time_grain or "year",
            matched_keywords=keyword_map,
        )

    if _question_requests_distinct_dimension_count(question, ordered_dimensions, group_by):
        return QueryPlan("pattern_20_distinct_count", "aggregate", "count_distinct_x", "scalar_count", "aggregate_scalar", target_sheet, "count_distinct", ordered_dimensions[0], target_sheet, matched_keywords=keyword_map, requested_top_n=1, metric_formula=explicit_formula)

    if partition_by and rank_dimension and (keyword_map.get("max") or keyword_map.get("min") or top_n or best_worst_direction):
        ranking_direction = best_worst_direction or ("asc" if keyword_map.get("min") else "desc")
        ranking_agg = best_or_worst_ranking_aggregation_for_question(question, ranking_direction) if best_worst_direction else ("sum" if aggregation in {"max", "min"} else aggregation)
        return QueryPlan("pattern_17_top_member_for_each_parent", "ranking", "top_member_per_parent", "ranked_table", "aggregate_grouped", target_sheet, ranking_agg, explicit_metric, target_sheet, filters=extracted_filters, partition_by=partition_by, rank_dimension=rank_dimension, order_direction=ranking_direction, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if _question_requests_row_count(question):
        return QueryPlan("pattern_10_count_rows", "aggregate", "count_of_x", "scalar_count", "aggregate_scalar", target_sheet, "count", "*", target_sheet, matched_keywords=keyword_map, requested_top_n=1)

    if aggregation == "count" and group_by and has_group_by_intent(question):
        return QueryPlan("pattern_11_count_by_dimension", "aggregate", "count_of_x_where_y" if extracted_filters else "count_of_x", "grouped_table", "aggregate_grouped", target_sheet, "count", "*", target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map)

    if aggregation == "count" and not has_group_by_intent(question):
        return QueryPlan("pattern_10_count_rows", "aggregate", canonical_intent.name, canonical_intent.answer_type, "aggregate_scalar", target_sheet, "count", "*", target_sheet, filters=extracted_filters, matched_keywords=keyword_map, requested_top_n=1)

    ranked_limit = top_n or (canonical_intent.requested_top_n if best_worst_direction else None)
    ranked_direction = explicit_direction or best_worst_direction or canonical_intent.ranking or "desc"
    if ranked_limit and group_by:
        pattern_id = "pattern_12_top_n_by_metric" if ranked_direction == "desc" else "pattern_13_bottom_n_by_metric"
        ranked_n_aggregation = top_n_aggregation_for_question(question) if pattern_id == "pattern_12_top_n_by_metric" else bottom_n_aggregation_for_question(question)
        return QueryPlan(pattern_id, "aggregate", "top_n_x_by_y", "ranked_table", "aggregate_grouped", target_sheet, ranked_n_aggregation, explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, order_direction=ranked_direction, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "avg" and not group_by and not extracted_filters:
        return QueryPlan("pattern_04_average_metric", "aggregate", "total_x", "scalar_value", "aggregate_scalar", target_sheet, "avg", explicit_metric, target_sheet, matched_keywords=keyword_map, requested_top_n=1, metric_formula=explicit_formula)

    if aggregation == "avg" and group_by:
        return QueryPlan("pattern_05_average_metric_by_dimension", "aggregate", "average_x_by_y", "grouped_table", "aggregate_grouped", target_sheet, "avg", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "max" and not group_by and not extracted_filters:
        return QueryPlan("pattern_06_highest_metric_overall", "aggregate", "highest_x", "scalar_value", "aggregate_scalar", target_sheet, "max", explicit_metric, target_sheet, matched_keywords=keyword_map, requested_top_n=1, metric_formula=explicit_formula)

    if aggregation == "max" and len(group_by) >= 2 and has_group_by_intent(question) and bool(re.search(r"\b(for each|wise)\b", q_text)):
        return QueryPlan("pattern_18_highest_metric_for_each_dimension_pair", "aggregate", "max_by_y", "grouped_table", "aggregate_grouped", target_sheet, "max", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "max" and group_by:
        return QueryPlan("pattern_07_highest_metric_by_dimension", "aggregate", "highest_x", "ranked_table", "aggregate_grouped", target_sheet, "max", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "min" and not group_by and not extracted_filters:
        return QueryPlan("pattern_08_lowest_metric_overall", "aggregate", "lowest_x", "scalar_value", "aggregate_scalar", target_sheet, "min", explicit_metric, target_sheet, matched_keywords=keyword_map, requested_top_n=1, metric_formula=explicit_formula)

    if aggregation == "min" and group_by:
        return QueryPlan("pattern_09_lowest_metric_by_dimension", "aggregate", "lowest_x", "ranked_table", "aggregate_grouped", target_sheet, "min", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, order_direction="asc", requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if _mentions_comparison(question) and group_by:
        return QueryPlan("pattern_16_compare_metric_across_dimension", "aggregate", "compare_x_across_y", "comparison_table", "comparison", target_sheet, "sum", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if _mentions_distribution(question) and group_by:
        return QueryPlan("pattern_19_metric_distribution_by_dimension", "aggregate", "x_by_y", "grouped_table", "aggregate_grouped", target_sheet, "sum", explicit_metric, target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if extracted_filters and group_by:
        agg = "sum" if aggregation in {"count", "count_distinct"} else aggregation
        return QueryPlan("pattern_15_metric_by_dimension_with_filter", "aggregate", "x_by_y", "grouped_table", "aggregate_grouped", target_sheet, agg, explicit_metric if agg != "count" else "*", target_sheet, group_by=group_by, filters=extracted_filters, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula if agg != "count" else None)

    if extracted_filters and not group_by:
        agg = "sum" if aggregation in {"count", "count_distinct"} else aggregation
        return QueryPlan("pattern_14_metric_for_specific_filter", "aggregate", "total_x_for_y", "filtered_scalar", "aggregate_scalar", target_sheet, agg, explicit_metric if agg != "count" else "*", target_sheet, filters=extracted_filters, requested_top_n=1, matched_keywords=keyword_map, metric_formula=explicit_formula if agg != "count" else None)

    if aggregation == "sum" and not group_by and not extracted_filters:
        return QueryPlan("pattern_01_total_metric", "aggregate", "total_x", "scalar_value", "aggregate_scalar", target_sheet, "sum", explicit_metric, target_sheet, requested_top_n=1, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "sum" and len(group_by) == 1 and not extracted_filters:
        return QueryPlan("pattern_02_total_metric_by_one_dimension", "aggregate", "x_by_y", "grouped_table", "aggregate_grouped", target_sheet, "sum", explicit_metric, target_sheet, group_by=group_by, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    if aggregation == "sum" and len(group_by) >= 2 and not extracted_filters:
        return QueryPlan("pattern_03_total_metric_by_two_dimensions", "aggregate", "x_by_y", "grouped_table", "aggregate_grouped", target_sheet, "sum", explicit_metric, target_sheet, group_by=group_by, requested_top_n=default_limit, matched_keywords=keyword_map, metric_formula=explicit_formula)

    return None


def _classifier_guidance_result(question: str, context: PlanningContext, *, target_sheet: str | None, warnings: list[str], reasons: list[str], candidate_sheets: list[dict[str, Any]], target_confidence: float, dimensions: list[str], dimension_confidence: float, measure: dict[str, Any] | None, measure_confidence: float, extracted_filters: list[dict[str, Any]], inferred_value_mappings: list[dict[str, Any]], candidate_metrics: list[dict[str, Any]], classifier: QueryClassification | None, canonical_intent: CanonicalIntent, fallback_reason: str) -> PlannerResult:
    confidence_components = [max((classifier.confidence if classifier else 0.0), canonical_intent.confidence), target_confidence]
    if measure is not None:
        confidence_components.append(measure_confidence)
    if dimensions:
        confidence_components.append(dimension_confidence)
    confidence = min(1.0, round(sum(confidence_components) / max(len(confidence_components), 1), 3))
    guidance = build_planner_guidance(question, context, target_sheet=target_sheet, warnings=warnings, extracted_filters=extracted_filters)
    fallback_query_type = "describe_sheet" if canonical_intent.query_shape == "describe_sheet" else ("row_preview" if canonical_intent.query_shape == "row_preview" else "aggregate")
    return PlannerResult(query_spec=None, query_shape=canonical_intent.query_shape if canonical_intent.name != "unsupported" else (classifier.shape if classifier else "unknown"), confidence=confidence, warnings=warnings, reasons=reasons, target_sheet=target_sheet, extracted_filters=extracted_filters, inferred_value_mappings=inferred_value_mappings, classifier=classifier, guidance=guidance, degraded_mode=True, canonical_intent=canonical_intent.name, answer_type=canonical_intent.answer_type, candidate_sheets=candidate_sheets, selected_metric=measure, candidate_metrics=candidate_metrics, selected_dimensions=dimensions, query_type=fallback_query_type, matched_intent_keywords=matched_intent_keywords(question), validation_result="fallback", fallback_reason=fallback_reason)


def _llm_classifier_fallback_factory() -> Callable[[str, PlanningContext], dict[str, Any] | None] | None:
    return None


def _has_explicit_list_request(question: str) -> bool:
    return bool(re.search(r"\b(list|show|display|give(?:\s+me)?)\b", str(question or "").lower()))


def _promote_generic_list_intent(
    question: str,
    canonical_intent: CanonicalIntent,
    classifier: QueryClassification,
    explicit_sheet: str | None,
    target_sheet: str | None,
    dimensions: list[str],
) -> CanonicalIntent | None:
    if canonical_intent.query_shape == "projection":
        return None
    if not explicit_sheet:
        return None
    if canonical_intent.metric_families:
        return None
    if not target_sheet:
        return None
    if not _has_explicit_list_request(question):
        return None
    if classifier.shape != "list_rows" and not dimensions:
        return None
    reasons = list(dict.fromkeys([
        *canonical_intent.reasons,
        *classifier.reasons,
        "Promoted heuristic list-row classification into schema-backed deterministic planning.",
    ]))
    return CanonicalIntent(
        "list_rows",
        "list_rows",
        None,
        "row_list",
        max(0.86, float(classifier.confidence or 0.0)),
        reasons=reasons,
    )


def plan_query(question: str, context: PlanningContext, *, explicit_sheet: str | None, max_rows: int) -> PlannerResult:
    planning_question = _question_without_explicit_sheet_context(question, explicit_sheet)
    canonical_intent = detect_canonical_intent(planning_question)
    target_sheet, target_confidence, target_reasons, target_warnings, candidate_sheets = _pick_target_sheet(question, context, explicit_sheet, canonical_intent)
    pre_classifier = classify_query(planning_question, context, llm_fallback=_llm_classifier_fallback_factory())
    dimensions, dimension_reasons, dimension_confidence = _detect_dimensions(planning_question, context, target_sheet, canonical_intent)
    projection_columns, projection_reasons, projection_confidence, _selected_columns_exist, pre_aggregated_sheet, _requested_projection_columns, _matched_existing_columns = _detect_projection_columns(planning_question, context, target_sheet)
    projection_metric_selected = any(_is_summary_metric_like_column(column_name) for column_name in projection_columns)
    projection_allowed = bool(explicit_sheet) or pre_aggregated_sheet
    projection_requested = _starts_with_projection_request(planning_question) and not has_group_by_intent(planning_question)
    if projection_allowed and projection_requested and projection_columns and (len(projection_columns) >= 2 or (pre_aggregated_sheet and projection_metric_selected)):
        canonical_intent = CanonicalIntent(
            "projection",
            "projection",
            None,
            "table",
            max(0.9, projection_confidence),
            reasons=list(dict.fromkeys([
                *canonical_intent.reasons,
                *projection_reasons,
                "Promoted explicit column listing into deterministic projection planning.",
            ])),
        )
    promoted_intent = _promote_generic_list_intent(planning_question, canonical_intent, pre_classifier, explicit_sheet, target_sheet, dimensions)
    if promoted_intent is not None:
        promoted_dimensions, promoted_dimension_reasons, promoted_dimension_confidence = _detect_dimensions(planning_question, context, target_sheet, promoted_intent)
        if promoted_dimensions:
            dimensions = promoted_dimensions
            dimension_reasons = promoted_dimension_reasons
            dimension_confidence = promoted_dimension_confidence
            canonical_intent = promoted_intent
    filter_candidates = extract_filters(planning_question, context, target_sheet=target_sheet)
    mapping_candidates = [candidate for candidate in filter_candidates if str(getattr(candidate, "kind", "") or "").lower() != "year"]
    value_mappings, mapping_warnings = map_values_to_columns(planning_question, context, target_sheet=target_sheet, candidates=mapping_candidates)
    explicit_year_filters = _extract_year_filters(filter_candidates, context, target_sheet)
    if canonical_intent.query_shape in {"row_preview", "list_rows", "projection", "describe_sheet"}:
        measure, measure_reasons, measure_confidence, measure_warnings, candidate_metrics = None, [], 0.0, [], []
    else:
        measure, measure_reasons, measure_confidence, measure_warnings, candidate_metrics = _pick_measure(planning_question, context, target_sheet, canonical_intent)
    extracted_filters, collapsed_filter_warnings, collapsed_filter_reasons = _collapse_mappings_to_filters(value_mappings, query_shape=canonical_intent.query_shape)
    extracted_filters = _dedupe_filters(extracted_filters + explicit_year_filters)
    inferred_value_mappings = [{"sheet": mapping.sheet, "column": mapping.column, "value": mapping.value, "operator": mapping.operator, "confidence": mapping.confidence, "reasons": mapping.reasons, "ambiguous_columns": mapping.ambiguous_columns, "source": mapping.source} for mapping in value_mappings]
    base_warnings = list(dict.fromkeys(target_warnings + mapping_warnings + measure_warnings + collapsed_filter_warnings))
    base_reasons = list(dict.fromkeys(canonical_intent.reasons + target_reasons + dimension_reasons + measure_reasons + collapsed_filter_reasons))

    requested_dimension_family, requested_dimension_phrase = _first_requested_family_phrase(planning_question, BUSINESS_SYNONYMS["dimensions"])
    requested_metric_family, requested_metric_phrase = _first_requested_family_phrase(planning_question, BUSINESS_SYNONYMS["metrics"])
    if _is_ranked_entity_question(canonical_intent):
        missing_dimension_reason = None
        missing_metric_reason = None
        if requested_dimension_family and not _resolved_dimension_matches_family(context, target_sheet, dimensions, requested_dimension_family):
            missing_dimension_reason = (
                f"I can't rank {requested_dimension_phrase or requested_dimension_family} because sheet '{target_sheet or 'the selected sheet'}' does not contain a matching dimension."
            )
        if requested_metric_family and not _resolved_metric_matches_family(context, target_sheet, str((measure or {}).get("column") or "") or None, requested_metric_family):
            missing_metric_reason = (
                f"I can't rank by {requested_metric_phrase or requested_metric_family} because sheet '{target_sheet or 'the selected sheet'}' does not contain a matching metric."
            )
        if missing_dimension_reason or missing_metric_reason:
            return _blocked_ranking_result(
                question=question,
                context=context,
                target_sheet=target_sheet,
                candidate_sheets=candidate_sheets,
                extracted_filters=extracted_filters,
                inferred_value_mappings=inferred_value_mappings,
                candidate_metrics=candidate_metrics,
                canonical_intent=canonical_intent,
                measure=measure,
                requested_dimension=requested_dimension_phrase or requested_dimension_family,
                requested_metric=requested_metric_phrase or requested_metric_family,
                missing_dimension_reason=missing_dimension_reason,
                missing_metric_reason=missing_metric_reason,
            )

    deterministic_plan = _build_deterministic_plan(planning_question, context, canonical_intent, target_sheet, dimensions, extracted_filters, measure, max_rows=max_rows)
    if deterministic_plan is not None:
        if measure is not None:
            deterministic_plan.requested_metric_name = str(measure.get("requested_metric") or "").strip() or None
            deterministic_plan.canonical_metric_name = str(measure.get("canonical_metric") or "").strip() or None
            deterministic_plan.resolved_metric_type = str(measure.get("resolved_metric_type") or "").strip() or None
        if deterministic_plan.query_shape == "list_rows" and (not deterministic_plan.group_by or any("ambiguous" in warning.lower() for warning in base_warnings)):
            deterministic_plan = None
        else:
            deterministic_plan = _validate_query_plan(planning_question, deterministic_plan, context)
        if deterministic_plan is not None:
            warnings = list(dict.fromkeys(base_warnings + deterministic_plan.validation_issues))
            reasons = list(dict.fromkeys(base_reasons + [f"Matched deterministic pattern '{deterministic_plan.pattern_id}'.", *deterministic_plan.auto_corrections]))
            if deterministic_plan.validation_result in {"valid", "corrected"}:
                confidence_components = [max(canonical_intent.confidence, 0.86), target_confidence]
                if measure is not None:
                    confidence_components.append(max(measure_confidence, 0.62))
                if deterministic_plan.group_by or deterministic_plan.partition_by:
                    confidence_components.append(max(dimension_confidence, 0.62))
                if deterministic_plan.query_type == "projection" and deterministic_plan.columns:
                    confidence_components.append(max(projection_confidence, 0.9))
                if value_mappings:
                    confidence_components.append(sum(mapping.confidence for mapping in value_mappings) / max(len(value_mappings), 1))
                confidence = round(sum(confidence_components) / max(len(confidence_components), 1), 3)
                if warnings:
                    confidence = max(0.0, confidence - min(0.16, len(warnings) * 0.03))
                query_spec = _query_spec_from_plan(deterministic_plan, confidence, warnings)
                selected_dimensions = [] if deterministic_plan.query_type == "describe_sheet" else (deterministic_plan.columns if deterministic_plan.query_type in {"projection", "row_preview"} else (deterministic_plan.group_by or (deterministic_plan.partition_by + ([deterministic_plan.rank_dimension] if deterministic_plan.rank_dimension else []))))
                selected_metric = None if deterministic_plan.query_type in {"projection", "describe_sheet", "row_preview"} else _measure_with_plan_aggregation(deterministic_plan)
                return PlannerResult(query_spec=query_spec, query_shape=deterministic_plan.query_shape, confidence=confidence, warnings=warnings, reasons=reasons, target_sheet=target_sheet, extracted_filters=extracted_filters, inferred_value_mappings=inferred_value_mappings, classifier=None, guidance=None, degraded_mode=False, canonical_intent=deterministic_plan.canonical_intent, answer_type=deterministic_plan.answer_type, candidate_sheets=candidate_sheets, selected_metric=selected_metric, candidate_metrics=candidate_metrics, selected_dimensions=selected_dimensions, query_type=deterministic_plan.query_type, pattern_id=deterministic_plan.pattern_id, matched_intent_keywords=deterministic_plan.matched_keywords, auto_corrections=deterministic_plan.auto_corrections, validation_result=deterministic_plan.validation_result)

    classifier = pre_classifier
    warnings = list(dict.fromkeys(base_warnings + (["Deterministic planner did not produce a valid plan."] if deterministic_plan is None else deterministic_plan.validation_issues)))
    reasons = list(dict.fromkeys(base_reasons + ["Falling back to heuristic classifier path."]))
    fallback_reason = "no_deterministic_pattern_match" if deterministic_plan is None else f"deterministic_plan_{deterministic_plan.validation_result}"
    remaining_dimensions = [dimension for dimension in dimensions if dimension not in {item["column"] for item in extracted_filters}]
    if deterministic_plan is None and _has_single_strong_execution_path(canonical_intent, target_confidence=target_confidence, measure_confidence=measure_confidence, dimension_confidence=dimension_confidence, candidate_sheets=candidate_sheets, candidate_metrics=candidate_metrics, dimensions=remaining_dimensions):
        warnings.append("Deterministic pattern matching failed; heuristic fallback retained as a last-resort path.")
    return _classifier_guidance_result(question, context, target_sheet=target_sheet, warnings=warnings, reasons=reasons, candidate_sheets=candidate_sheets, target_confidence=target_confidence, dimensions=remaining_dimensions, dimension_confidence=dimension_confidence, measure=measure, measure_confidence=measure_confidence, extracted_filters=extracted_filters, inferred_value_mappings=inferred_value_mappings, candidate_metrics=candidate_metrics, classifier=classifier, canonical_intent=canonical_intent, fallback_reason=fallback_reason)