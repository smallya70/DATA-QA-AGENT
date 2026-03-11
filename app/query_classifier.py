from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import json
import re

from app.business_intents import detect_canonical_intent
from app.planning_context import PlanningContext, normalize_name


@dataclass(slots=True)
class QueryClassification:
    shape: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _question_token_set(question: str) -> set[str]:
    return {token for token in normalize_name(question).split("_") if token}


def _contains_metric_token(question: str, metric_tokens: list[str]) -> bool:
    q_norm = normalize_name(question)
    q_tokens = _question_token_set(question)
    for token in metric_tokens:
        normalized_token = normalize_name(token)
        token_parts = [part for part in normalized_token.split("_") if part]
        if not token_parts:
            continue
        if len(token_parts) == 1:
            if token_parts[0] in q_tokens:
                return True
            continue
        if normalized_token in q_norm:
            return True
    return False


def _has_metric_intent(question: str) -> bool:
    return _contains_metric_token(
        question,
        [
            "sum", "summary", "total", "avg", "average", "min", "max", "count",
            "revenue", "sales", "profit", "amount", "price", "value", "quantity", "qty",
        ],
    )


def _has_time_intent(question: str) -> bool:
    q_norm = normalize_name(question)
    q_text = str(question or "").lower()
    if any(token in q_norm for token in ["monthly", "quarterly", "weekly", "daily", "trend_over_time"]):
        return True
    if any(token in q_norm for token in ["month", "quarter", "week", "day"]) and any(
        marker in q_text for marker in [" by ", " per ", " trend", " over time", " across time"]
    ):
        return True
    return bool(re.search(r"\b(?:by|per|trend|over time|across time)\s+(?:year|month|quarter|week|day)\b", q_text))


def _has_grouping_intent(question: str) -> bool:
    q_text = str(question or "").lower()
    q_norm = normalize_name(question)
    return any(token in q_text or token in q_norm for token in [" by ", " per ", "group_by", "group by", "breakdown", "split by", "for each", "across"])


def _has_list_intent(question: str) -> bool:
    q_norm = normalize_name(question)
    return any(token in q_norm for token in ["list", "show", "display", "give", "give_me", "return"])


def _has_comparison_intent(question: str) -> bool:
    q_text = str(question or "").lower()
    q_norm = normalize_name(question)
    return any(token in q_text or token in q_norm for token in ["compare", "vs", "versus", "difference", "compared_to"])


def _has_join_summary_intent(question: str, context: PlanningContext) -> bool:
    q_text = str(question or "").lower()
    mentioned_sheets = [sheet for sheet in context.sheets if normalize_name(sheet) in normalize_name(question)]
    return len(mentioned_sheets) >= 2 or any(token in q_text for token in ["join ", "joined ", "across sheets", "between sheets"])


def classify_query(
    question: str,
    context: PlanningContext,
    *,
    llm_fallback: Callable[[str, PlanningContext], dict[str, Any] | None] | None = None,
) -> QueryClassification:
    q_text = str(question or "").strip()
    reasons: list[str] = []
    warnings: list[str] = []

    canonical_intent = detect_canonical_intent(q_text)
    if canonical_intent.name != "unsupported":
        reasons.extend(canonical_intent.reasons)
        return QueryClassification(canonical_intent.query_shape, canonical_intent.confidence, reasons, warnings)

    metric_intent = _has_metric_intent(q_text)
    grouping_intent = _has_grouping_intent(q_text)
    list_intent = _has_list_intent(q_text)
    time_intent = _has_time_intent(q_text)
    comparison_intent = _has_comparison_intent(q_text)
    join_summary_intent = _has_join_summary_intent(q_text, context)

    if time_intent and metric_intent:
        reasons.append("Detected time intent and metric intent.")
        return QueryClassification("trend_over_time", 0.92, reasons, warnings)
    if comparison_intent and metric_intent:
        reasons.append("Detected comparison language and metric intent.")
        return QueryClassification("comparison", 0.84, reasons, warnings)
    if join_summary_intent:
        reasons.append("Detected multi-sheet or join summary language.")
        return QueryClassification("join_summary", 0.74, reasons, warnings)
    if metric_intent and grouping_intent:
        reasons.append("Detected metric intent with grouping language.")
        return QueryClassification("aggregate_grouped", 0.88, reasons, warnings)
    if metric_intent:
        reasons.append("Detected metric intent without grouping language.")
        return QueryClassification("aggregate_scalar", 0.82, reasons, warnings)
    if list_intent:
        reasons.append("Detected list/display language without metric intent.")
        return QueryClassification("list_rows", 0.86, reasons, warnings)

    warnings.append("Question shape could not be classified confidently with deterministic rules.")
    if llm_fallback is not None:
        try:
            llm_result = llm_fallback(q_text, context) or {}
            shape = str(llm_result.get("shape") or "").strip()
            confidence = float(llm_result.get("confidence") or 0.0)
            if shape:
                reasons.extend([str(item) for item in list(llm_result.get("reasons") or []) if str(item).strip()])
                warnings.extend([str(item) for item in list(llm_result.get("warnings") or []) if str(item).strip()])
                return QueryClassification(shape, confidence, reasons, warnings)
        except Exception as exc:
            warnings.append(f"LLM classifier fallback failed: {exc}")

    return QueryClassification("unknown", 0.30, reasons, warnings)


def classifier_debug_json(classification: QueryClassification) -> str:
    return json.dumps(
        {
            "shape": classification.shape,
            "confidence": classification.confidence,
            "reasons": classification.reasons,
            "warnings": classification.warnings,
        },
        sort_keys=True,
    )