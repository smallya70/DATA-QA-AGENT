import pandas as pd

from app.dashboard_filters import apply_dashboard_filters, summarize_active_filters
from app.filterable_chart_data import build_filterable_chart_data


def _grouped_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Product": ["Chair", "Desk", "Chair", "Laptop", "Laptop"],
            "Region": ["West", "West", "South", "South", "East"],
            "sum_total_price": [100.0, 200.0, 150.0, 500.0, 700.0],
        }
    )


def test_categorical_multiselect_filtering():
    result = apply_dashboard_filters(
        _grouped_df(),
        categorical_filters={"Region": ["West", "South"]},
        numeric_filters={},
        text_filters={},
    )
    assert result.rows_after_filter == 4
    assert set(result.dataframe["Region"].tolist()) == {"West", "South"}


def test_numeric_range_filtering():
    result = apply_dashboard_filters(
        _grouped_df(),
        categorical_filters={},
        numeric_filters={"sum_total_price": (120.0, 550.0)},
        text_filters={},
    )
    assert result.rows_after_filter == 3
    assert result.dataframe["sum_total_price"].tolist() == [200.0, 150.0, 500.0]


def test_filtering_happens_before_secondary_aggregation():
    result = build_filterable_chart_data(
        _grouped_df(),
        categorical_filters={"Region": ["West", "South"]},
        numeric_filters={},
        text_filters={"Product": "Chair"},
        x_column="Region",
        y_column="sum_total_price",
        aggregation="SUM",
        split_column=None,
        top_n=None,
        sort_by="label_asc",
    )
    assert result.filter_result.rows_after_filter == 2
    assert result.chart_result.dataframe.to_dict("records") == [
        {"Region": "South", "sum_total_price": 150.0},
        {"Region": "West", "sum_total_price": 100.0},
    ]


def test_active_filter_summary_display():
    result = apply_dashboard_filters(
        _grouped_df(),
        categorical_filters={"Region": ["West", "South"]},
        numeric_filters={"sum_total_price": (100.0, 500.0)},
        text_filters={"Product": "Chair"},
    )
    summary = summarize_active_filters(result.active_filter_labels)
    assert "Active filters:" in summary
    assert "Region in [West, South]" in summary
    assert "Product contains 'Chair'" in summary
    assert "sum_total_price between 100.00 and 500.00" in summary


def test_singleton_categorical_filter_uses_equality_label():
    result = apply_dashboard_filters(
        _grouped_df(),
        categorical_filters={"Product": ["Chair"]},
        numeric_filters={},
        text_filters={},
    )
    summary = summarize_active_filters(result.active_filter_labels)
    assert result.rows_after_filter == 2
    assert summary == "Active filters: Product = Chair"