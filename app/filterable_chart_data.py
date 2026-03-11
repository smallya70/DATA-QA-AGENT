from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.chart_data_builder import ChartBuildResult, build_chart_dataframe
from app.dashboard_filters import DashboardFilterResult, apply_dashboard_filters


@dataclass(slots=True)
class FilterableChartDataResult:
    filter_result: DashboardFilterResult
    chart_result: ChartBuildResult


def build_filterable_chart_data(
    raw_df: pd.DataFrame,
    *,
    categorical_filters: dict[str, list[str]] | None,
    numeric_filters: dict[str, tuple[float, float]] | None,
    text_filters: dict[str, str] | None,
    x_column: str,
    y_column: str,
    aggregation: str,
    split_column: str | None = None,
    top_n: int | None = None,
    sort_by: str = "value_desc",
) -> FilterableChartDataResult:
    filter_result = apply_dashboard_filters(
        raw_df,
        categorical_filters=categorical_filters,
        numeric_filters=numeric_filters,
        text_filters=text_filters,
    )
    chart_result = build_chart_dataframe(
        filter_result.dataframe,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,
        split_column=split_column,
        top_n=top_n,
        sort_by=sort_by,
    )
    return FilterableChartDataResult(filter_result=filter_result, chart_result=chart_result)