from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


IGNORED_DIMENSION_COLUMNS = {"row_index", "sheet_name"}
AUXILIARY_MEASURE_COLUMNS = {"matched_rows", "row_count", "__row_count"}


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _resolve_column_name(columns: list[str], candidate: str | None) -> str | None:
    if not candidate:
        return None
    if candidate in columns:
        return candidate
    normalized_candidate = _normalize_name(candidate)
    for column in columns:
        if _normalize_name(column) == normalized_candidate:
            return column
    return None


def _looks_like_time_dimension(column_name: str) -> bool:
    normalized = _normalize_name(column_name)
    return any(token in normalized for token in ["year", "month", "date", "quarter", "week", "day", "time"])


def grain_label(columns: list[str]) -> str:
    cleaned = [str(column).strip() for column in columns if str(column).strip()]
    if not cleaned:
        return "Scalar"
    return " × ".join(cleaned)


def _extract_query_spec_dimensions(result_payload: dict[str, Any] | None, columns: list[str]) -> list[str]:
    query_spec = result_payload.get("query_spec") if isinstance(result_payload, dict) else None
    planner_summary = result_payload.get("planner_summary") if isinstance(result_payload, dict) else None

    resolved: list[str] = []
    for source in [query_spec, planner_summary]:
        if not isinstance(source, dict):
            continue
        for item in list(source.get("dimensions") or []):
            resolved_name = _resolve_column_name(columns, str(item))
            if resolved_name and resolved_name not in resolved:
                resolved.append(resolved_name)
        time_grain = _resolve_column_name(columns, str(source.get("time_grain") or ""))
        if time_grain and time_grain not in resolved:
            resolved.append(time_grain)
    return resolved


def _extract_query_spec_measures(result_payload: dict[str, Any] | None, columns: list[str]) -> list[str]:
    query_spec = result_payload.get("query_spec") if isinstance(result_payload, dict) else None
    planner_summary = result_payload.get("planner_summary") if isinstance(result_payload, dict) else None

    resolved: list[str] = []
    for source in [query_spec, planner_summary]:
        if not isinstance(source, dict):
            continue
        for measure in list(source.get("measures") or []):
            if not isinstance(measure, dict):
                continue
            for field_name in ["alias", "column"]:
                resolved_name = _resolve_column_name(columns, str(measure.get(field_name) or ""))
                if resolved_name and resolved_name not in resolved:
                    resolved.append(resolved_name)
    return resolved


def _extract_query_type(result_payload: dict[str, Any] | None) -> str:
    for source in [
        result_payload.get("query_spec") if isinstance(result_payload, dict) else None,
        result_payload.get("planner_summary") if isinstance(result_payload, dict) else None,
    ]:
        if isinstance(source, dict):
            query_type = str(source.get("query_type") or "").strip().lower()
            if query_type:
                return query_type
    return ""


def _extract_query_spec_columns(result_payload: dict[str, Any] | None, columns: list[str]) -> list[str]:
    query_spec = result_payload.get("query_spec") if isinstance(result_payload, dict) else None
    planner_summary = result_payload.get("planner_summary") if isinstance(result_payload, dict) else None

    resolved: list[str] = []
    for source in [query_spec, planner_summary]:
        if not isinstance(source, dict):
            continue
        for item in list(source.get("columns") or []):
            resolved_name = _resolve_column_name(columns, str(item))
            if resolved_name and resolved_name not in resolved:
                resolved.append(resolved_name)
    return resolved


def valid_aggregations_for_y(df: pd.DataFrame, y_column: str | None, result_payload: dict[str, Any] | None = None) -> list[str]:
    if not y_column or y_column not in df.columns:
        return ["COUNT"]
    if _extract_query_type(result_payload) == "projection":
        if pd.api.types.is_numeric_dtype(df[y_column]):
            return ["RAW", "SUM", "AVG", "MIN", "MAX", "COUNT"]
        return ["RAW", "COUNT"]
    if pd.api.types.is_numeric_dtype(df[y_column]):
        return ["SUM", "AVG", "MIN", "MAX", "COUNT"]
    return ["COUNT"]


@dataclass(slots=True)
class ResultSemantics:
    base_grain_columns: list[str]
    dimension_columns: list[str]
    measure_columns: list[str]
    raw_row_count: int
    query_shape: str = ""
    query_type: str = ""
    planner_warnings: list[str] = field(default_factory=list)

    @property
    def base_grain_label(self) -> str:
        return grain_label(self.base_grain_columns)

    @property
    def grouped_result(self) -> bool:
        return len(self.base_grain_columns) >= 1 and bool(self.measure_columns)

    @property
    def time_dimension_columns(self) -> list[str]:
        return [column for column in self.dimension_columns if _looks_like_time_dimension(column)]


@dataclass(slots=True)
class ChartSemantics:
    base_grain_columns: list[str]
    chart_grain_columns: list[str]
    y_column: str
    aggregation: str
    chart_type: str
    split_column: str | None = None
    requires_secondary_aggregation: bool = False
    pie_requires_single_dimension: bool = False

    @property
    def base_grain_label(self) -> str:
        return grain_label(self.base_grain_columns)

    @property
    def chart_grain_label(self) -> str:
        return grain_label(self.chart_grain_columns)

    @property
    def aggregation_label(self) -> str:
        if self.aggregation == "RAW":
            return f"RAW({self.y_column}) by {self.chart_grain_label}"
        if self.aggregation == "COUNT":
            return f"COUNT(rows) by {self.chart_grain_label}"
        return f"{self.aggregation}({self.y_column}) by {self.chart_grain_label}"


def infer_result_semantics(df: pd.DataFrame, result_payload: dict[str, Any] | None = None) -> ResultSemantics:
    columns = [str(column) for column in df.columns]
    query_shape = str((result_payload or {}).get("query_shape") or "").strip()
    query_type = _extract_query_type(result_payload)
    planner_warnings = [
        str(item) for item in list((result_payload or {}).get("planner_warnings") or []) if str(item).strip()
    ]

    base_grain_columns = _extract_query_spec_dimensions(result_payload, columns)
    measure_columns = _extract_query_spec_measures(result_payload, columns)

    if query_type == "projection":
        projection_columns = _extract_query_spec_columns(result_payload, columns)
        if projection_columns:
            measure_columns = [column for column in projection_columns if pd.api.types.is_numeric_dtype(df[column])]
            base_grain_columns = [column for column in projection_columns if column not in measure_columns]

    if not measure_columns:
        measure_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(df[column]) and column not in {"row_index"}
        ]
        preferred = [column for column in measure_columns if column not in AUXILIARY_MEASURE_COLUMNS]
        if preferred:
            measure_columns = preferred + [column for column in measure_columns if column in AUXILIARY_MEASURE_COLUMNS]

    if not base_grain_columns:
        base_grain_columns = [
            column
            for column in columns
            if column not in measure_columns and column not in IGNORED_DIMENSION_COLUMNS
        ]

    dimension_columns = [
        column for column in columns if column not in measure_columns and column not in IGNORED_DIMENSION_COLUMNS
    ]

    return ResultSemantics(
        base_grain_columns=base_grain_columns,
        dimension_columns=dimension_columns,
        measure_columns=measure_columns,
        raw_row_count=int(df.shape[0]),
        query_shape=query_shape,
        query_type=query_type,
        planner_warnings=planner_warnings,
    )


def infer_chart_semantics(
    result_semantics: ResultSemantics,
    *,
    x_column: str,
    y_column: str,
    aggregation: str,
    chart_type: str,
    split_column: str | None = None,
) -> ChartSemantics:
    chart_grain_columns = [column for column in [x_column, split_column] if column]
    if chart_type == "Pie Chart":
        chart_grain_columns = [x_column]

    base_set = set(result_semantics.base_grain_columns)
    chart_set = set(chart_grain_columns)
    requires_secondary_aggregation = bool(base_set) and chart_set.issubset(base_set) and chart_set != base_set
    pie_requires_single_dimension = chart_type == "Pie Chart" and len(result_semantics.base_grain_columns) > 1

    return ChartSemantics(
        base_grain_columns=list(result_semantics.base_grain_columns),
        chart_grain_columns=chart_grain_columns,
        y_column=y_column,
        aggregation=aggregation,
        chart_type=chart_type,
        split_column=split_column if chart_type != "Pie Chart" else None,
        requires_secondary_aggregation=requires_secondary_aggregation,
        pie_requires_single_dimension=pie_requires_single_dimension,
    )


def use_grouped_split_bars(*, chart_type: str, aggregation: str) -> bool:
    return chart_type == "Grouped Bar" or (chart_type == "Bar Chart" and aggregation == "COUNT")