from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class DashboardFilterResult:
    dataframe: pd.DataFrame
    rows_before_filter: int
    rows_after_filter: int
    active_filter_labels: list[str] = field(default_factory=list)

    @property
    def active_filter_summary(self) -> str:
        if not self.active_filter_labels:
            return "None"
        return "; ".join(self.active_filter_labels)


def summarize_active_filters(labels: list[str]) -> str:
    if not labels:
        return "Active filters: None"
    return "Active filters: " + "; ".join(labels)


def _format_categorical_filter_label(column_name: str, selected_values: list[str]) -> str:
    if len(selected_values) == 1:
        return f"{column_name} = {selected_values[0]}"
    return f"{column_name} in [{', '.join(selected_values)}]"


def apply_dashboard_filters(
    df: pd.DataFrame,
    *,
    categorical_filters: dict[str, list[str]] | None = None,
    numeric_filters: dict[str, tuple[float, float]] | None = None,
    text_filters: dict[str, str] | None = None,
) -> DashboardFilterResult:
    filtered_df = df.copy()
    rows_before_filter = int(filtered_df.shape[0])
    active_filter_labels: list[str] = []

    for column_name, values in (categorical_filters or {}).items():
        if column_name not in filtered_df.columns:
            continue
        selected_values = [str(value) for value in list(values or []) if str(value).strip()]
        if not selected_values:
            continue
        filtered_df = filtered_df[filtered_df[column_name].astype(str).isin(selected_values)]
        active_filter_labels.append(_format_categorical_filter_label(column_name, selected_values))

    for column_name, search_text in (text_filters or {}).items():
        if column_name not in filtered_df.columns:
            continue
        normalized_search = str(search_text or "").strip()
        if not normalized_search:
            continue
        mask = filtered_df[column_name].astype(str).str.contains(normalized_search, case=False, na=False)
        filtered_df = filtered_df[mask]
        active_filter_labels.append(f"{column_name} contains '{normalized_search}'")

    for column_name, range_values in (numeric_filters or {}).items():
        if column_name not in filtered_df.columns:
            continue
        if not isinstance(range_values, tuple) or len(range_values) != 2:
            continue
        minimum_value, maximum_value = float(range_values[0]), float(range_values[1])
        numeric_series = pd.to_numeric(filtered_df[column_name], errors="coerce")
        mask = numeric_series.between(minimum_value, maximum_value, inclusive="both")
        filtered_df = filtered_df[mask.fillna(False)]
        active_filter_labels.append(f"{column_name} between {minimum_value:,.2f} and {maximum_value:,.2f}")

    return DashboardFilterResult(
        dataframe=filtered_df.reset_index(drop=True),
        rows_before_filter=rows_before_filter,
        rows_after_filter=int(filtered_df.shape[0]),
        active_filter_labels=active_filter_labels,
    )