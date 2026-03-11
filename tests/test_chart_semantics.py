from app.chart_data_builder import build_chart_dataframe
from app.chart_recommender import recommend_chart_config
from app.chart_semantics import grain_label, infer_chart_semantics, infer_result_semantics, use_grouped_split_bars


def _sample_grouped_df():
    return {
        "Product": ["Chair", "Desk", "Chair", "Desk"],
        "Region": ["East", "East", "West", "West"],
        "sum_total_price": [100.0, 200.0, 50.0, 80.0],
    }


def test_detects_base_result_grain_from_query_spec():
    import pandas as pd

    df = pd.DataFrame(_sample_grouped_df())
    semantics = infer_result_semantics(
        df,
        {
            "query_spec": {
                "dimensions": ["Product", "Region"],
                "measures": [{"alias": "sum_total_price", "column": "TotalPrice", "aggregation": "sum"}],
            }
        },
    )
    assert semantics.base_grain_columns == ["Product", "Region"]
    assert semantics.base_grain_label == "Product × Region"


def test_secondary_aggregation_by_one_selected_dimension():
    import pandas as pd

    df = pd.DataFrame(_sample_grouped_df())
    chart = build_chart_dataframe(
        df,
        x_column="Region",
        y_column="sum_total_price",
        aggregation="SUM",
        split_column=None,
        top_n=None,
        sort_by="label_asc",
    )
    assert chart.dataframe.to_dict("records") == [
        {"Region": "East", "sum_total_price": 300.0},
        {"Region": "West", "sum_total_price": 130.0},
    ]


def test_pie_chart_is_marked_as_secondary_aggregation_for_multi_dimension_result():
    import pandas as pd

    df = pd.DataFrame(_sample_grouped_df())
    result_semantics = infer_result_semantics(
        df,
        {
            "query_spec": {
                "dimensions": ["Product", "Region"],
                "measures": [{"alias": "sum_total_price", "column": "TotalPrice", "aggregation": "sum"}],
            }
        },
    )
    chart_semantics = infer_chart_semantics(
        result_semantics,
        x_column="Region",
        y_column="sum_total_price",
        aggregation="SUM",
        chart_type="Pie Chart",
    )
    assert chart_semantics.requires_secondary_aggregation is True
    assert chart_semantics.pie_requires_single_dimension is True
    assert chart_semantics.aggregation_label == "SUM(sum_total_price) by Region"


def test_recommends_stacked_bar_for_two_dimension_grouped_result():
    import pandas as pd

    df = pd.DataFrame(_sample_grouped_df())
    semantics = infer_result_semantics(
        df,
        {
            "query_spec": {
                "dimensions": ["Product", "Region"],
                "measures": [{"alias": "sum_total_price", "column": "TotalPrice", "aggregation": "sum"}],
            }
        },
    )
    recommendation = recommend_chart_config(semantics)
    assert recommendation.default_chart_type == "Stacked Bar"
    assert recommendation.x_column == "Region"
    assert recommendation.split_column == "Product"


def test_chart_preview_data_generation_with_split_and_top_n():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Product": ["Chair", "Desk", "Lamp", "Chair", "Desk", "Lamp"],
            "Region": ["East", "East", "East", "West", "West", "West"],
            "sum_total_price": [100.0, 200.0, 50.0, 70.0, 120.0, 20.0],
        }
    )
    chart = build_chart_dataframe(
        df,
        x_column="Region",
        y_column="sum_total_price",
        aggregation="SUM",
        split_column="Product",
        top_n=1,
        sort_by="value_desc",
    )
    assert chart.preview_columns == ["Region", "Product", "sum_total_price"]
    assert set(chart.dataframe["Region"].tolist()) == {"East"}


def test_query_spec_aware_defaults_prefer_time_dimension_line_chart():
    import pandas as pd

    df = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03"],
            "sum_total_price": [100.0, 120.0, 90.0],
        }
    )
    semantics = infer_result_semantics(
        df,
        {
            "query_spec": {
                "dimensions": [],
                "time_grain": "month",
                "measures": [{"alias": "sum_total_price", "column": "TotalPrice", "aggregation": "sum"}],
            }
        },
    )
    recommendation = recommend_chart_config(semantics)
    assert grain_label(semantics.base_grain_columns) == "month"
    assert recommendation.default_chart_type == "Trend Line"


def test_projection_result_prefers_raw_aggregation_for_numeric_projected_column():
    import pandas as pd

    df = pd.DataFrame(
        {
            "exception_type": ["unbilled", "lagged"],
            "count": [3, 5],
        }
    )
    result_payload = {
        "query_spec": {
            "query_type": "projection",
            "columns": ["exception_type", "count"],
            "dimensions": [],
            "measures": [],
        }
    }

    semantics = infer_result_semantics(df, result_payload)
    assert semantics.base_grain_columns == ["exception_type"]
    assert semantics.measure_columns == ["count"]

    chart = build_chart_dataframe(
        df,
        x_column="exception_type",
        y_column="count",
        aggregation="RAW",
        split_column=None,
        top_n=None,
        sort_by="label_asc",
    )
    assert chart.dataframe.to_dict("records") == [
        {"exception_type": "lagged", "count": 5},
        {"exception_type": "unbilled", "count": 3},
    ]


def test_chart_builder_falls_back_to_row_count_when_value_column_matches_grouping_column():
    import pandas as pd

    df = pd.DataFrame(
        {
            "invoice_id": ["INV-1", "INV-1", "INV-2"],
            "invoice_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        }
    )

    chart = build_chart_dataframe(
        df,
        x_column="invoice_id",
        y_column="invoice_id",
        aggregation="SUM",
        split_column=None,
        top_n=None,
        sort_by="label_asc",
    )

    assert chart.value_column == "row_count"
    assert chart.preview_columns == ["invoice_id", "row_count"]
    assert chart.dataframe.to_dict("records") == [
        {"invoice_id": "INV-1", "row_count": 2},
        {"invoice_id": "INV-2", "row_count": 1},
    ]


def test_projection_recommendation_defaults_to_table_before_kpi():
    import pandas as pd

    df = pd.DataFrame(
        {
            "exception_type": ["unbilled", "lagged"],
            "count": [3, 5],
        }
    )
    semantics = infer_result_semantics(
        df,
        {
            "query_spec": {
                "query_type": "projection",
                "columns": ["exception_type", "count"],
                "dimensions": [],
                "measures": [],
            }
        },
    )

    recommendation = recommend_chart_config(semantics)
    assert recommendation.default_chart_type == "Table"
    assert "Table" in recommendation.chart_types
    assert recommendation.y_column == "count"


def test_count_split_bar_chart_prefers_grouped_rendering():
    assert use_grouped_split_bars(chart_type="Bar Chart", aggregation="COUNT") is True
    assert use_grouped_split_bars(chart_type="Grouped Bar", aggregation="COUNT") is True


def test_sum_split_bar_chart_keeps_stacked_bar_rendering():
    assert use_grouped_split_bars(chart_type="Bar Chart", aggregation="SUM") is False
    assert use_grouped_split_bars(chart_type="Stacked Bar", aggregation="SUM") is False


def test_chart_builder_aggregates_with_dict_dimension_values():
    import pandas as pd

    df = pd.DataFrame(
        {
            "event_meta": [{"event_id": "E1"}, {"event_id": "E1"}, {"event_id": "E2"}],
            "amount": [10.0, 15.0, 5.0],
        }
    )

    chart = build_chart_dataframe(
        df,
        x_column="event_meta",
        y_column="amount",
        aggregation="SUM",
        split_column=None,
        top_n=None,
        sort_by="label_asc",
    )

    assert chart.dataframe.to_dict("records") == [
        {"event_meta": '{"event_id": "E1"}', "amount": 25.0},
        {"event_meta": '{"event_id": "E2"}', "amount": 5.0},
    ]


def test_chart_builder_counts_with_list_split_values():
    import pandas as pd

    df = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E1"],
            "tags": [["a", "b"], ["a", "b"], ["c"]],
            "amount": [10.0, 15.0, 5.0],
        }
    )

    chart = build_chart_dataframe(
        df,
        x_column="event_id",
        y_column="amount",
        aggregation="COUNT",
        split_column="tags",
        top_n=None,
        sort_by="label_asc",
    )

    assert chart.value_column == "row_count"
    assert chart.dataframe.to_dict("records") == [
        {"event_id": "E1", "tags": '["a", "b"]', "row_count": 2},
        {"event_id": "E1", "tags": '["c"]', "row_count": 1},
    ]