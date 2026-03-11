from __future__ import annotations

from typing import Any

from app.planning_context import build_planning_context
from app.query_planner import plan_query
from app.query_spec import QuerySpec


def classify_query_shape_compat(
    question: str,
    dimensions: list[str],
    filters: list[dict[str, Any]],
    measure: dict[str, Any] | None,
) -> str:
    filtered_columns = {str(item.get("column") or "") for item in filters if isinstance(item, dict)}
    unfiltered_dimensions = [dimension for dimension in dimensions if dimension not in filtered_columns]
    question_text = str(question or "").lower()
    listing_requested = any(token in question_text for token in ["list", "show", "display", "give me", "give"])
    metric_requested = any(token in question_text for token in ["sum", "total", "avg", "average", "max", "min", "count", "sales", "revenue", "price", "amount", "value", "quantity", "qty"])
    grouping_requested = any(token in question_text for token in [" by ", " per ", "group by", "grouped", "breakdown", "split by", "across", "for each"])

    if measure:
        if unfiltered_dimensions:
            return "aggregate_by_dimensions"
        return "aggregate_with_filters"
    if listing_requested and not metric_requested:
        return "list_rows"
    if unfiltered_dimensions and grouping_requested:
        return "aggregate_by_dimensions"
    if unfiltered_dimensions or filters:
        return "list_rows"
    return "unknown"


def build_query_spec_stub_compat(
    question: str,
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
    explicit_sheet: str | None,
    max_rows: int,
    value_samples_by_sheet: dict[str, dict[str, list[str]]] | None = None,
) -> QuerySpec | None:
    relationships_payload = {
        "tables": list(available_sheets or []),
        "relationships": list(relationships or []),
        "semantic_roles": {},
    }
    synthetic_rows: list[dict[str, Any]] = []
    for sheet_name, samples in (value_samples_by_sheet or {}).items():
        row_count = max((len(values or []) for values in (samples or {}).values()), default=0)
        for index in range(min(row_count, 5)):
            row_json: dict[str, Any] = {}
            for column_name, values in (samples or {}).items():
                if values:
                    row_json[column_name] = values[min(index, len(values) - 1)]
            if row_json:
                synthetic_rows.append({"sheet_name": sheet_name, "row_json": row_json})

    context = build_planning_context(
        dataset_id="phase2_stub",
        cols=sorted({str(column) for columns in (sheet_columns_by_sheet or {}).values() for column in columns}),
        col_types={},
        relationships_payload=relationships_payload,
        sheet_rows=synthetic_rows,
    )
    query_spec = plan_query(
        question,
        context,
        explicit_sheet=explicit_sheet,
        max_rows=max_rows,
    ).query_spec
    if query_spec is not None:
        filtered_columns = {
            str(filter_spec.get("column") or "")
            for filter_spec in list(query_spec.filters or [])
            if isinstance(filter_spec, dict)
        }
        query_spec.filters = [
            {
                key: value
                for key, value in dict(filter_spec).items()
                if key != "kind"
            }
            for filter_spec in list(query_spec.filters or [])
            if isinstance(filter_spec, dict)
        ]
        query_spec.measures = [
            {
                key: value
                for key, value in dict(measure_spec).items()
                if key not in {"sheet", "requested_metric", "canonical_metric", "resolved_metric_type", "formula", "derived"}
            }
            for measure_spec in list(query_spec.measures or [])
            if isinstance(measure_spec, dict)
        ]
        if query_spec.query_type == "list":
            query_spec.dimensions = [
                dimension
                for dimension in list(query_spec.dimensions or [])
                if str(dimension) not in filtered_columns
            ]
        elif query_spec.query_type == "projection":
            query_spec.dimensions = [
                column_name
                for column_name in list(query_spec.columns or [])
                if str(column_name) not in filtered_columns
            ]
    return query_spec