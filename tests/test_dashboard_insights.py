from __future__ import annotations

import pandas as pd

from app.chart_semantics import ChartSemantics, ResultSemantics
from app.dashboard_insights import compute_dashboard_insight_facts, generate_dashboard_insight_summary


def test_compute_dashboard_insight_facts_for_grouped_chart():
    chart_df = pd.DataFrame(
        {
            "category": ["Desk", "Desk", "Chair", "Chair", "Table", "Table"],
            "comparison_value": ["East", "West", "East", "West", "East", "West"],
            "metric_value": [250.0, 175.0, 100.0, 150.0, 90.0, 60.0],
        }
    )
    result_semantics = ResultSemantics(
        base_grain_columns=["Product", "Region"],
        dimension_columns=["Product", "Region"],
        measure_columns=["Total Price"],
        raw_row_count=6,
        query_shape="aggregate_grouped",
        planner_warnings=[],
    )
    chart_semantics = ChartSemantics(
        base_grain_columns=["Product", "Region"],
        chart_grain_columns=["Product", "Region"],
        y_column="Total Price",
        aggregation="SUM",
        chart_type="Bar Chart",
        split_column="Region",
        requires_secondary_aggregation=False,
        pie_requires_single_dimension=False,
    )

    facts = compute_dashboard_insight_facts(
        chart_df,
        chart_semantics=chart_semantics,
        result_semantics=result_semantics,
        x_column="category",
        value_column="metric_value",
        x_label="Product",
        y_label="Total Price",
        active_filter_summary="Region in [East, West]",
        rows_after_filter=6,
    )

    assert facts is not None
    assert facts.top_category == "Desk"
    assert facts.top_value == 425.0
    assert facts.bottom_category == "Table"
    assert facts.bottom_value == 150.0
    assert facts.distinct_category_count == 3
    assert facts.concentration_top_n == 3
    assert facts.concentration_share == 1.0
    assert facts.use_neutral_language is False


def test_generate_dashboard_insight_summary_uses_neutral_language_when_confidence_low():
    chart_df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "metric_value": [10.0, 5.0, 1.0],
        }
    )
    result_semantics = ResultSemantics(
        base_grain_columns=["category"],
        dimension_columns=["category"],
        measure_columns=["metric_value"],
        raw_row_count=3,
        query_shape="aggregate_grouped",
        planner_warnings=["Synthetic labels used."],
    )
    chart_semantics = ChartSemantics(
        base_grain_columns=["category"],
        chart_grain_columns=["category"],
        y_column="metric_value",
        aggregation="SUM",
        chart_type="Bar Chart",
    )

    summary, facts = generate_dashboard_insight_summary(
        chart_df,
        chart_semantics=chart_semantics,
        result_semantics=result_semantics,
        x_column="category",
        value_column="metric_value",
        x_label="category",
        y_label="metric_value",
        active_filter_summary="None",
        rows_after_filter=3,
        prefer_llm=False,
    )

    assert facts is not None
    assert facts.use_neutral_language is True
    assert facts.dimension_label == "category"
    assert facts.metric_label == "value"
    assert summary is not None
    assert "Top category is A" in summary
    assert "displayed value" in summary


def test_generate_dashboard_insight_summary_mentions_current_filters():
    chart_df = pd.DataFrame(
        {
            "category": ["India", "UK"],
            "metric_value": [120.0, 80.0],
        }
    )
    result_semantics = ResultSemantics(
        base_grain_columns=["Country"],
        dimension_columns=["Country"],
        measure_columns=["Amount"],
        raw_row_count=2,
        query_shape="aggregate_grouped",
        planner_warnings=[],
    )
    chart_semantics = ChartSemantics(
        base_grain_columns=["Country"],
        chart_grain_columns=["Country"],
        y_column="Amount",
        aggregation="SUM",
        chart_type="Bar Chart",
    )

    summary, _facts = generate_dashboard_insight_summary(
        chart_df,
        chart_semantics=chart_semantics,
        result_semantics=result_semantics,
        x_column="category",
        value_column="metric_value",
        x_label="Country",
        y_label="Amount",
        active_filter_summary="Year = 2024; Region = East",
        rows_after_filter=2,
        prefer_llm=False,
    )

    assert summary is not None
    assert "Current filters: Year = 2024; Region = East." in summary