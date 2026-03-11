from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ChartBuildResult:
    dataframe: pd.DataFrame
    preview_columns: list[str]
    rows_in_raw_result: int
    rows_in_chart_data: int
    value_column: str


def _normalize_chart_dimension_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, list):
        return json.dumps(value, default=str)
    if isinstance(value, set):
        return json.dumps(sorted(value, key=str), default=str)
    return value


def _normalize_chart_dimension_columns(chart_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized_df = chart_df.copy()
    for column in columns:
        if column not in normalized_df.columns:
            continue
        normalized_df[column] = normalized_df[column].map(_normalize_chart_dimension_value)
    return normalized_df


def _sort_chart_dataframe(
    chart_df: pd.DataFrame,
    *,
    x_column: str,
    value_column: str,
    split_column: str | None,
    sort_by: str,
) -> pd.DataFrame:
    if chart_df.empty:
        return chart_df

    if sort_by == "label_asc":
        return chart_df.sort_values(by=[x_column] + ([split_column] if split_column else []), ascending=True)
    if sort_by == "label_desc":
        return chart_df.sort_values(by=[x_column] + ([split_column] if split_column else []), ascending=False)

    ascending = sort_by == "value_asc"
    if split_column:
        totals = chart_df.groupby(x_column, dropna=False)[value_column].sum().reset_index(name="__sort_value")
        ordered = totals.sort_values(by="__sort_value", ascending=ascending)[x_column].tolist()
        rank = {value: index for index, value in enumerate(ordered)}
        sorted_df = chart_df.assign(__x_rank=chart_df[x_column].map(rank).fillna(len(rank)))
        sorted_df = sorted_df.sort_values(by=["__x_rank", value_column], ascending=[True, ascending])
        return sorted_df.drop(columns=["__x_rank"])

    return chart_df.sort_values(by=value_column, ascending=ascending)


def build_chart_dataframe(
    raw_df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    aggregation: str,
    split_column: str | None = None,
    top_n: int | None = None,
    sort_by: str = "value_desc",
) -> ChartBuildResult:
    group_columns = [column for column in [x_column, split_column] if column]
    working_columns = list(dict.fromkeys(group_columns + ([y_column] if y_column in raw_df.columns else [])))
    chart_df = raw_df[working_columns].copy()
    chart_df = _normalize_chart_dimension_columns(chart_df, group_columns)
    value_overlaps_grouping = bool(y_column) and y_column in group_columns

    if aggregation == "RAW":
        chart_df = chart_df.dropna(subset=[column for column in [x_column, y_column] if column in chart_df.columns])
        value_column = y_column
    elif aggregation == "COUNT" or value_overlaps_grouping:
        chart_df = chart_df.groupby(group_columns, dropna=False).size().reset_index(name="row_count")
        value_column = "row_count"
    else:
        chart_df = chart_df.dropna(subset=[column for column in [x_column, y_column] if column in chart_df.columns])
        pandas_agg = {
            "SUM": "sum",
            "AVG": "mean",
            "MIN": "min",
            "MAX": "max",
        }.get(aggregation, "sum")
        chart_df = chart_df.groupby(group_columns, dropna=False)[y_column].agg(pandas_agg).reset_index()
        value_column = y_column

    if top_n and top_n > 0 and not chart_df.empty:
        if split_column:
            totals = chart_df.groupby(x_column, dropna=False)[value_column].sum().reset_index(name="__total")
            totals = totals.sort_values(by="__total", ascending=False).head(int(top_n))
            allowed = set(totals[x_column].tolist())
            chart_df = chart_df[chart_df[x_column].isin(allowed)].copy()
        else:
            chart_df = chart_df.sort_values(by=value_column, ascending=False).head(int(top_n)).copy()

    chart_df = _sort_chart_dataframe(
        chart_df,
        x_column=x_column,
        value_column=value_column,
        split_column=split_column,
        sort_by=sort_by,
    ).reset_index(drop=True)

    preview_columns = list(dict.fromkeys(group_columns + [value_column]))
    return ChartBuildResult(
        dataframe=chart_df,
        preview_columns=preview_columns,
        rows_in_raw_result=int(raw_df.shape[0]),
        rows_in_chart_data=int(chart_df.shape[0]),
        value_column=value_column,
    )