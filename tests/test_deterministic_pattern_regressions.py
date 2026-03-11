from __future__ import annotations

import pytest

from app.query_planner import plan_query
from app.sql_builder import build_sql_from_query_spec


def _metadata_from_context(context):
    return {
        "sheets": context.sheets,
        "columns_by_sheet": context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in context.sheets
        },
        "derived_metrics": {
            metric_name: formula
            for sheet_metrics in (context.derived_metrics_by_sheet or {}).values()
            for metric_name, formula in (sheet_metrics or {}).items()
        },
        "derived_metrics_by_sheet": context.derived_metrics_by_sheet,
        "relationships": context.joins,
    }


PATTERN_CASES = [
    {"question": "total sales", "fixture": "sample_planning_context", "pattern_id": "pattern_01_total_metric", "query_type": "aggregate", "intent": "total_x", "aggregation": "sum", "dimensions": [], "sql_contains": ["SUM("]},
    {"question": "sales by region", "fixture": "sample_planning_context", "pattern_id": "pattern_02_total_metric_by_one_dimension", "query_type": "aggregate", "intent": "x_by_y", "aggregation": "sum", "dimensions": ["Region"], "sql_contains": ["GROUP BY", "row_json->>'Region'", "SUM("]},
    {"question": "amount by product and country", "fixture": "chocolate_planning_context", "pattern_id": "pattern_03_total_metric_by_two_dimensions", "query_type": "aggregate", "intent": "x_by_y", "aggregation": "sum", "dimensions": ["Product", "Country"], "sql_contains": ["GROUP BY", "row_json->>'Product'", "row_json->>'Country'"]},
    {"question": "average amount", "fixture": "chocolate_planning_context", "pattern_id": "pattern_04_average_metric", "query_type": "aggregate", "intent": "total_x", "aggregation": "avg", "dimensions": [], "sql_contains": ["AVG("]},
    {"question": "average total price by product", "fixture": "sample_planning_context", "pattern_id": "pattern_05_average_metric_by_dimension", "query_type": "aggregate", "intent": "average_x_by_y", "aggregation": "avg", "dimensions": ["Product"], "sql_contains": ["AVG(", "GROUP BY"]},
    {"question": "highest amount", "fixture": "chocolate_planning_context", "pattern_id": "pattern_06_highest_metric_overall", "query_type": "aggregate", "intent": "highest_x", "aggregation": "max", "dimensions": [], "sql_contains": ["MAX("], "sql_not_contains": ["GROUP BY"]},
    {"question": "highest total sales by product", "fixture": "sample_planning_context", "pattern_id": "pattern_07_highest_metric_by_dimension", "query_type": "aggregate", "intent": "highest_x", "aggregation": "max", "dimensions": ["Product"], "sql_contains": ["MAX(", "GROUP BY", "ORDER BY max_total_price DESC"]},
    {"question": "lowest amount", "fixture": "chocolate_planning_context", "pattern_id": "pattern_08_lowest_metric_overall", "query_type": "aggregate", "intent": "lowest_x", "aggregation": "min", "dimensions": [], "sql_contains": ["MIN("], "sql_not_contains": ["GROUP BY"]},
    {"question": "lowest total sales by region", "fixture": "sample_planning_context", "pattern_id": "pattern_09_lowest_metric_by_dimension", "query_type": "aggregate", "intent": "lowest_x", "aggregation": "min", "dimensions": ["Region"], "sql_contains": ["MIN(", "GROUP BY", "ORDER BY min_total_price ASC"]},
    {"question": "count of products", "fixture": "sample_planning_context", "pattern_id": "pattern_10_count_rows", "query_type": "aggregate", "intent": "count_of_x", "aggregation": "count", "dimensions": [], "sql_contains": ["COUNT(*)"]},
    {"question": "count of tickets by stage", "fixture": "multi_sheet_business_context", "pattern_id": "pattern_11_count_by_dimension", "query_type": "aggregate", "intent": "count_of_x", "aggregation": "count", "dimensions": ["Stage"], "sql_contains": ["COUNT(*)", "GROUP BY", "row_json->>'Stage'"]},
    {"question": "top 5 amount by country", "fixture": "chocolate_planning_context", "pattern_id": "pattern_12_top_n_by_metric", "query_type": "aggregate", "intent": "top_n_x_by_y", "aggregation": "sum", "dimensions": ["Country"], "sql_contains": ["SUM(", "ORDER BY sum_amount DESC", "LIMIT %(limit)s"], "sql_not_contains": ["MAX("]},
    {"question": "bottom 2 amount by country", "fixture": "chocolate_planning_context", "pattern_id": "pattern_13_bottom_n_by_metric", "query_type": "aggregate", "intent": "top_n_x_by_y", "aggregation": "sum", "dimensions": ["Country"], "sql_contains": ["SUM(", "ORDER BY sum_amount ASC", "LIMIT %(limit)s"], "sql_not_contains": ["MIN("]},
    {"question": "total amount for india", "fixture": "chocolate_planning_context", "pattern_id": "pattern_14_metric_for_specific_filter", "query_type": "aggregate", "intent": "total_x_for_y", "aggregation": "sum", "dimensions": [], "sql_contains": ["SUM(", "WHERE", "Country"]},
    {"question": "total sales by product in east", "fixture": "sample_planning_context", "pattern_id": "pattern_15_metric_by_dimension_with_filter", "query_type": "aggregate", "intent": "x_by_y", "aggregation": "sum", "dimensions": ["Product"], "sql_contains": ["GROUP BY", "row_json->>'Product'", "Region"]},
    {"question": "compare total sales across region", "fixture": "sample_planning_context", "pattern_id": "pattern_16_compare_metric_across_dimension", "query_type": "aggregate", "intent": "compare_x_across_y", "aggregation": "sum", "dimensions": ["Region"], "sql_contains": ["GROUP BY", "row_json->>'Region'", "SUM("]},
    {"question": "highest amount product for each country", "fixture": "chocolate_planning_context", "pattern_id": "pattern_17_top_member_for_each_parent", "query_type": "ranking", "intent": "top_member_per_parent", "aggregation": "max", "dimensions": [], "sql_contains": ["ROW_NUMBER() OVER", "PARTITION BY Country", "WHERE rn <= %(rank_limit)s"]},
    {"question": "highest amount by product country wise", "fixture": "chocolate_planning_context", "pattern_id": "pattern_18_highest_metric_for_each_dimension_pair", "query_type": "aggregate", "intent": "max_by_y", "aggregation": "max", "dimensions": ["Product", "Country"], "sql_contains": ["MAX(", "GROUP BY", "row_json->>'Product'", "row_json->>'Country'"], "sql_not_contains": ["ROW_NUMBER() OVER"]},
    {"question": "distribution of amount by country", "fixture": "chocolate_planning_context", "pattern_id": "pattern_19_metric_distribution_by_dimension", "query_type": "aggregate", "intent": "x_by_y", "aggregation": "sum", "dimensions": ["Country"], "sql_contains": ["SUM(", "GROUP BY", "row_json->>'Country'"]},
    {"question": "count distinct products", "fixture": "chocolate_planning_context", "pattern_id": "pattern_20_distinct_count", "query_type": "aggregate", "intent": "count_distinct_x", "aggregation": "count_distinct", "dimensions": [], "sql_contains": ["COUNT(DISTINCT", "Product"]},
]


def test_pattern_12_top_n_with_explicit_max_wording_uses_max(chocolate_planning_context):
    result = plan_query("top 5 products by highest amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_12_top_n_by_metric"
    assert result.query_type == "aggregate"
    assert result.canonical_intent == "top_n_x_by_y"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["aggregation"] == "max"
    assert result.query_spec.dimensions == ["Product"]

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "MAX(" in sql
    assert "SUM(" not in sql
    assert "ORDER BY max_amount DESC" in sql
    assert "LIMIT %(limit)s" in sql


def test_pattern_13_bottom_n_with_explicit_min_wording_uses_min(chocolate_planning_context):
    result = plan_query("bottom 5 countries by lowest amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_13_bottom_n_by_metric"
    assert result.query_type == "aggregate"
    assert result.canonical_intent == "top_n_x_by_y"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["aggregation"] == "min"
    assert result.query_spec.dimensions == ["Country"]

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "MIN(" in sql
    assert "SUM(" not in sql
    assert "ORDER BY min_amount ASC" in sql
    assert "LIMIT %(limit)s" in sql


def test_best_grouped_question_defaults_to_top_1_sum(chocolate_planning_context):
    result = plan_query("best product by amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_12_top_n_by_metric"
    assert result.query_type == "aggregate"
    assert result.canonical_intent == "top_n_x_by_y"
    assert result.query_spec is not None
    assert result.query_spec.limit == 1
    assert result.query_spec.measures[0]["aggregation"] == "sum"
    assert result.query_spec.dimensions == ["Product"]

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "SUM(" in sql
    assert "MAX(" not in sql
    assert "ORDER BY sum_amount DESC" in sql
    assert "LIMIT %(limit)s" in sql


def test_worst_grouped_question_defaults_to_bottom_1_sum(chocolate_planning_context):
    result = plan_query("worst country by amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_13_bottom_n_by_metric"
    assert result.query_type == "aggregate"
    assert result.canonical_intent == "top_n_x_by_y"
    assert result.query_spec is not None
    assert result.query_spec.limit == 1
    assert result.query_spec.measures[0]["aggregation"] == "sum"
    assert result.query_spec.dimensions == ["Country"]

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "SUM(" in sql
    assert "MIN(" not in sql
    assert "ORDER BY sum_amount ASC" in sql
    assert "LIMIT %(limit)s" in sql


def test_best_member_for_each_parent_defaults_to_sum_ranking(chocolate_planning_context):
    result = plan_query("best product for each country by amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_17_top_member_for_each_parent"
    assert result.query_type == "ranking"
    assert result.canonical_intent == "top_member_per_parent"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["aggregation"] == "sum"
    assert result.query_spec.partition_by == ["Country"]
    assert result.query_spec.rank_dimension == "Product"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY Country" in sql
    assert "SUM(" in sql
    assert "MAX(" not in sql


def test_worst_member_for_each_parent_with_explicit_min_wording_uses_min(chocolate_planning_context):
    result = plan_query("worst product for each country by lowest amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == "pattern_17_top_member_for_each_parent"
    assert result.query_type == "ranking"
    assert result.canonical_intent == "top_member_per_parent"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["aggregation"] == "min"
    assert result.query_spec.partition_by == ["Country"]
    assert result.query_spec.rank_dimension == "Product"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(chocolate_planning_context))
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY Country" in sql
    assert "MIN(" in sql
    assert "SUM(" not in sql


@pytest.mark.parametrize("case", PATTERN_CASES, ids=[item["pattern_id"] for item in PATTERN_CASES])
def test_deterministic_patterns(case, request):
    context = request.getfixturevalue(case["fixture"])
    result = plan_query(case["question"], context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.pattern_id == case["pattern_id"]
    assert result.query_type == case["query_type"]
    assert result.canonical_intent == case["intent"]
    assert result.validation_result in {"valid", "corrected"}
    assert result.query_spec is not None
    assert result.query_spec.pattern_id == case["pattern_id"]
    assert result.query_spec.query_type == case["query_type"]
    assert result.query_spec.intent == case["intent"]
    assert result.query_spec.dimensions == case["dimensions"]

    if case["aggregation"] is not None:
        assert result.query_spec.measures
        assert result.query_spec.measures[0]["aggregation"] == case["aggregation"]

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(context))
    for expected_text in case.get("sql_contains", []):
        assert expected_text in sql
    for unexpected_text in case.get("sql_not_contains", []):
        assert unexpected_text not in sql