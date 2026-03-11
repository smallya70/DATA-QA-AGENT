from __future__ import annotations

from dataclasses import dataclass, field

from app.chart_semantics import ResultSemantics


@dataclass(slots=True)
class ChartRecommendation:
    default_chart_type: str
    chart_types: list[str] = field(default_factory=list)
    x_column: str | None = None
    y_column: str | None = None
    split_column: str | None = None
    chart_grain_columns: list[str] = field(default_factory=list)


def recommend_chart_config(result_semantics: ResultSemantics) -> ChartRecommendation:
    dimensions = list(result_semantics.base_grain_columns or result_semantics.dimension_columns)
    measures = list(result_semantics.measure_columns)
    y_column = measures[0] if measures else None
    time_dimensions = result_semantics.time_dimension_columns

    if result_semantics.query_type == "projection":
        if not dimensions or not y_column:
            return ChartRecommendation(
                default_chart_type="Table",
                chart_types=["Table"],
                y_column=y_column,
                chart_grain_columns=list(dimensions),
            )
        if len(dimensions) == 1:
            chart_types = ["Table", "Bar Chart", "Pie Chart"]
            if dimensions[0] in time_dimensions:
                chart_types.append("Trend Line")
            return ChartRecommendation(
                default_chart_type="Table",
                chart_types=list(dict.fromkeys(chart_types)),
                x_column=dimensions[0],
                y_column=y_column,
                chart_grain_columns=[dimensions[0]],
            )
        x_column = dimensions[1] if len(dimensions) > 1 else dimensions[0]
        split_column = dimensions[0] if len(dimensions) > 1 else None
        chart_types = ["Table", "Stacked Bar", "Grouped Bar", "Heatmap", "Treemap", "Bar Chart", "Pie Chart"]
        if time_dimensions:
            chart_types.append("Trend Line")
        return ChartRecommendation(
            default_chart_type="Table",
            chart_types=list(dict.fromkeys(chart_types)),
            x_column=x_column,
            y_column=y_column,
            split_column=split_column,
            chart_grain_columns=[column for column in [x_column, split_column] if column],
        )

    if not dimensions and y_column:
        return ChartRecommendation(
            default_chart_type="KPI Card",
            chart_types=["KPI Card"],
            y_column=y_column,
            chart_grain_columns=[],
        )

    if len(dimensions) == 1:
        default_chart_type = "Trend Line" if dimensions[0] in time_dimensions else "Bar Chart"
        chart_types = [default_chart_type, "Bar Chart", "Pie Chart"]
        if default_chart_type != "Trend Line":
            chart_types.append("Trend Line")
        return ChartRecommendation(
            default_chart_type=default_chart_type,
            chart_types=list(dict.fromkeys(chart_types)),
            x_column=dimensions[0],
            y_column=y_column,
            chart_grain_columns=[dimensions[0]],
        )

    if len(dimensions) >= 2:
        x_column = dimensions[1] if len(dimensions) > 1 else dimensions[0]
        split_column = dimensions[0] if len(dimensions) > 1 else None
        chart_types = ["Stacked Bar", "Grouped Bar", "Heatmap", "Treemap", "Bar Chart", "Pie Chart"]
        if time_dimensions:
            chart_types.insert(0, "Trend Line")
        return ChartRecommendation(
            default_chart_type="Stacked Bar" if not time_dimensions else "Trend Line",
            chart_types=list(dict.fromkeys(chart_types)),
            x_column=x_column,
            y_column=y_column,
            split_column=split_column,
            chart_grain_columns=[column for column in [x_column, split_column] if column],
        )

    return ChartRecommendation(default_chart_type="Table", chart_types=["Table"], y_column=y_column)