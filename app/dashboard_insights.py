from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import pandas as pd

from app.chart_semantics import ChartSemantics, ResultSemantics


SYNTHETIC_LABELS = {
    "category",
    "comparison_value",
    "metric_value",
    "row_count",
    "matched_rows",
    "__row_count",
}


@dataclass(slots=True)
class DashboardInsightFacts:
    dimension_label: str
    metric_label: str
    raw_dimension_name: str
    raw_metric_name: str
    split_label: str | None
    distinct_category_count: int
    top_category: str | None
    top_value: float | None
    bottom_category: str | None
    bottom_value: float | None
    concentration_top_n: int
    concentration_share: float | None
    active_filter_summary: str
    rows_after_filter: int
    rows_in_chart_data: int
    chart_grain_label: str
    aggregation_label: str
    confidence: float
    use_neutral_language: bool


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _clean_label(value: str | None) -> str:
    text = str(value or "").strip()
    return text or "value"


def _looks_generic_label(value: str | None) -> bool:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return not normalized or normalized in SYNTHETIC_LABELS


def _neutral_metric_label(raw_metric_name: str, preferred_metric_label: str) -> str:
    normalized = f"{raw_metric_name} {preferred_metric_label}".lower()
    if any(token in normalized for token in ["amount", "price", "sales", "revenue", "profit", "cost"]):
        return "amount"
    if "count" in normalized:
        return "value"
    return "value"


def _confidence_for_insight(
    *,
    raw_dimension_name: str,
    raw_metric_name: str,
    planner_warnings: list[str],
    distinct_category_count: int,
) -> float:
    confidence = 0.95
    if _looks_generic_label(raw_dimension_name):
        confidence -= 0.28
    if _looks_generic_label(raw_metric_name):
        confidence -= 0.22
    if planner_warnings:
        confidence -= min(0.2, 0.05 * len(planner_warnings))
    if distinct_category_count <= 1:
        confidence -= 0.12
    return max(0.0, min(1.0, confidence))


def compute_dashboard_insight_facts(
    aggregated_df: pd.DataFrame,
    *,
    chart_semantics: ChartSemantics,
    result_semantics: ResultSemantics,
    x_column: str,
    value_column: str,
    x_label: str,
    y_label: str,
    active_filter_summary: str,
    rows_after_filter: int,
) -> DashboardInsightFacts | None:
    if aggregated_df.empty or x_column not in aggregated_df.columns or value_column not in aggregated_df.columns:
        return None

    working_df = aggregated_df[[x_column, value_column]].copy()
    working_df[value_column] = pd.to_numeric(working_df[value_column], errors="coerce")
    working_df = working_df.dropna(subset=[x_column, value_column])
    if working_df.empty:
        return None

    grouped = working_df.groupby(x_column, dropna=False)[value_column].sum().reset_index(name="__insight_value")
    grouped = grouped.sort_values(["__insight_value", x_column], ascending=[False, True]).reset_index(drop=True)
    if grouped.empty:
        return None

    distinct_category_count = int(grouped[x_column].nunique(dropna=True))
    top_row = grouped.iloc[0]
    bottom_row = grouped.sort_values(["__insight_value", x_column], ascending=[True, True]).iloc[0]

    concentration_top_n = min(3, max(1, distinct_category_count))
    absolute_total = float(grouped["__insight_value"].abs().sum())
    absolute_top_n = float(grouped.head(concentration_top_n)["__insight_value"].abs().sum())
    concentration_share = None if absolute_total <= 0 else absolute_top_n / absolute_total

    raw_dimension_name = _clean_label(x_label or x_column)
    raw_metric_name = _clean_label(y_label or value_column)
    confidence = _confidence_for_insight(
        raw_dimension_name=raw_dimension_name,
        raw_metric_name=raw_metric_name,
        planner_warnings=result_semantics.planner_warnings,
        distinct_category_count=distinct_category_count,
    )
    use_neutral_language = confidence < 0.62

    return DashboardInsightFacts(
        dimension_label="category" if use_neutral_language else raw_dimension_name,
        metric_label=_neutral_metric_label(raw_metric_name, raw_metric_name) if use_neutral_language else raw_metric_name,
        raw_dimension_name=raw_dimension_name,
        raw_metric_name=raw_metric_name,
        split_label=_clean_label(chart_semantics.split_column) if chart_semantics.split_column else None,
        distinct_category_count=distinct_category_count,
        top_category=str(top_row[x_column]) if pd.notna(top_row[x_column]) else None,
        top_value=_coerce_float(top_row["__insight_value"]),
        bottom_category=str(bottom_row[x_column]) if pd.notna(bottom_row[x_column]) else None,
        bottom_value=_coerce_float(bottom_row["__insight_value"]),
        concentration_top_n=concentration_top_n,
        concentration_share=concentration_share,
        active_filter_summary=active_filter_summary or "None",
        rows_after_filter=int(rows_after_filter),
        rows_in_chart_data=int(aggregated_df.shape[0]),
        chart_grain_label=chart_semantics.chart_grain_label,
        aggregation_label=chart_semantics.aggregation_label,
        confidence=confidence,
        use_neutral_language=use_neutral_language,
    )


def _deterministic_insight_summary(facts: DashboardInsightFacts) -> str:
    parts: list[str] = []
    metric_label = facts.metric_label
    dimension_label = facts.dimension_label

    if facts.top_category is not None and facts.top_value is not None:
        parts.append(
            f"Top {dimension_label.lower()} is {facts.top_category} at {facts.top_value:,.2f} {metric_label.lower()}."
        )
    if facts.bottom_category is not None and facts.bottom_value is not None and facts.distinct_category_count > 1:
        parts.append(
            f"Lowest {dimension_label.lower()} is {facts.bottom_category} at {facts.bottom_value:,.2f} {metric_label.lower()}."
        )
    if facts.concentration_share is not None:
        parts.append(
            f"Top {facts.concentration_top_n} {dimension_label.lower()}{'' if facts.concentration_top_n == 1 else 'ies' if facts.dimension_label.lower().endswith('y') else 's'} account for {facts.concentration_share * 100:.1f}% of displayed {metric_label.lower()}."
        )
    parts.append(f"Distinct {dimension_label.lower()} count is {facts.distinct_category_count}.")
    if facts.active_filter_summary and facts.active_filter_summary != "None":
        parts.append(f"Current filters: {facts.active_filter_summary}.")
    return " ".join(parts)


def _llm_insight_summary_from_facts(facts: DashboardInsightFacts) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
        response = client.responses.create(
            model=model,
            instructions=(
                "Write a short dashboard insight summary in at most 3 sentences. "
                "Use only the provided structured facts. "
                "Do not invent columns, KPIs, causes, recommendations, or business meaning. "
                "If use_neutral_language is true, prefer the provided neutral metric_label and dimension_label exactly. "
                "Mention current filters only when active_filter_summary is not 'None'."
            ),
            input=json.dumps(asdict(facts), ensure_ascii=True),
            max_output_tokens=140,
        )
        summary = (getattr(response, "output_text", "") or "").strip()
        return summary or None
    except Exception:
        return None


def generate_dashboard_insight_summary(
    aggregated_df: pd.DataFrame,
    *,
    chart_semantics: ChartSemantics,
    result_semantics: ResultSemantics,
    x_column: str,
    value_column: str,
    x_label: str,
    y_label: str,
    active_filter_summary: str,
    rows_after_filter: int,
    prefer_llm: bool = True,
) -> tuple[str | None, DashboardInsightFacts | None]:
    facts = compute_dashboard_insight_facts(
        aggregated_df,
        chart_semantics=chart_semantics,
        result_semantics=result_semantics,
        x_column=x_column,
        value_column=value_column,
        x_label=x_label,
        y_label=y_label,
        active_filter_summary=active_filter_summary,
        rows_after_filter=rows_after_filter,
    )
    if facts is None:
        return None, None

    summary = _llm_insight_summary_from_facts(facts) if prefer_llm else None
    if not summary:
        summary = _deterministic_insight_summary(facts)
    return summary, facts