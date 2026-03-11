from __future__ import annotations

import pytest

from app.analytics_logging import log_analytics_failure
from app.business_intents import BUSINESS_SYNONYMS, detect_canonical_intent
from app.planning_context import build_planning_context
from app.query_planner import plan_query
from tests.gold.simple_query_gold_set import SIMPLE_QUERY_GOLD_SET


@pytest.mark.parametrize("case", SIMPLE_QUERY_GOLD_SET, ids=[item["question"] for item in SIMPLE_QUERY_GOLD_SET])
def test_simple_query_gold_set(case, request):
    context = request.getfixturevalue(case["context_fixture"])
    result = plan_query(
        case["question"],
        context,
        explicit_sheet=None,
        max_rows=100,
    )

    assert result.query_shape == case["expected_sql_shape"]
    assert result.canonical_intent == case["expected_query_intent"]
    assert result.answer_type == case["expected_answer_type"]
    assert result.target_sheet == case["expected_sheet"]
    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.intent == case["expected_query_intent"]

    if case["expected_metric_column"] == "*":
        assert result.query_spec.measures[0]["column"] == "*"
    elif case["expected_metric_column"]:
        assert result.query_spec.measures[0]["column"] == case["expected_metric_column"]

    if case["expected_dimension_column"]:
        assert case["expected_dimension_column"] in result.selected_dimensions or case["expected_dimension_column"] in result.query_spec.dimensions

    filters = {(item["column"], str(item["value"])) for item in result.query_spec.filters}
    for expected_column, expected_value in case["expected_filters"]:
        assert (expected_column, expected_value) in filters


def test_business_synonym_dictionary_covers_requested_vocab():
    metric_vocab = BUSINESS_SYNONYMS["metrics"]["amount"]
    assert {"amount", "value", "revenue", "sales", "total", "spend", "cost", "price"}.issubset(set(metric_vocab))
    assert {"status", "state", "stage"}.issubset(set(BUSINESS_SYNONYMS["dimensions"]["status"]))
    assert {"country", "region", "market", "location"}.issubset(set(BUSINESS_SYNONYMS["dimensions"]["country"] + BUSINESS_SYNONYMS["dimensions"]["region"]))
    assert {"product", "sku", "material", "service"}.issubset(set(BUSINESS_SYNONYMS["dimensions"]["product"]))
    assert {"customer", "client", "account", "buyer"}.issubset(set(BUSINESS_SYNONYMS["dimensions"]["customer"]))
    assert {"quantity", "qty", "units", "volume"}.issubset(set(BUSINESS_SYNONYMS["metrics"]["quantity"]))


def test_detect_canonical_intents_cover_requested_query_families():
    assert detect_canonical_intent("total sales").name == "total_x"
    assert detect_canonical_intent("total sales for chair").name == "total_x_for_y"
    assert detect_canonical_intent("sales by region").name == "x_by_y"
    assert detect_canonical_intent("top 5 sales by region").name == "top_n_x_by_y"
    assert detect_canonical_intent("highest amount by product").name == "highest_x"
    assert detect_canonical_intent("lowest amount by product").name == "lowest_x"
    assert detect_canonical_intent("count of orders").name == "count_of_x"
    assert detect_canonical_intent("count of orders where country india").name == "count_of_x_where_y"
    assert detect_canonical_intent("compare sales across region").name == "compare_x_across_y"
    assert detect_canonical_intent("trend of sales over time").name == "trend_of_x_over_time"
    assert detect_canonical_intent("average sales by region").name == "average_x_by_y"


def test_value_mapping_prefers_country_for_india(chocolate_planning_context):
    result = plan_query(
        "total amount for india",
        chocolate_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert any(item["column"] == "Country" and item["value"] == "india" for item in result.query_spec.filters)


def test_sheet_selection_prefers_sales_for_revenue_questions(multi_sheet_business_context):
    result = plan_query(
        "total revenue by country",
        multi_sheet_business_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.target_sheet == "Sales"
    assert result.degraded_mode is False


def test_sheet_selection_prefers_support_for_ticket_questions(multi_sheet_business_context):
    result = plan_query(
        "count of tickets by stage",
        multi_sheet_business_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.target_sheet == "Support"


def test_unsupported_question_degrades(sample_planning_context):
    result = plan_query(
        "tell me something interesting",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is True
    assert result.canonical_intent == "unsupported"


def test_ambiguous_question_degrades_when_metric_not_resolved():
    ambiguous_context = build_planning_context(
        dataset_id="ambiguous-business",
        cols=["Region", "Country", "Status"],
        col_types={"Region": "text", "Country": "text", "Status": "text"},
        relationships_payload={
            "tables": ["NorthSheet", "SouthSheet"],
            "relationships": [],
            "semantic_roles": {
                "NorthSheet": {"country": "Country", "region": "Region", "status": "Status"},
                "SouthSheet": {"country": "Country", "region": "Region", "status": "Status"},
            },
        },
        sheet_rows=[
            {"sheet_name": "NorthSheet", "row_json": {"Region": "North", "Country": "India", "Status": "Open"}},
            {"sheet_name": "SouthSheet", "row_json": {"Region": "South", "Country": "India", "Status": "Open"}},
        ],
    )
    result = plan_query(
        "show geography summary",
        ambiguous_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is True


def test_failure_logging_structure(tmp_path, monkeypatch):
    monkeypatch.setattr("app.analytics_logging.FAILURE_LOG_PATH", tmp_path / "analytics_failures.jsonl")
    ref = log_analytics_failure(
        {
            "raw_user_question": "show total by geography",
            "detected_intent": "phase2_planner_degraded",
            "planner_confidence": 0.41,
            "chosen_sheet": "Sales",
            "candidate_sheets": [{"sheet": "Sales", "score": 0.62}],
            "chosen_metric": {"column": "Amount"},
            "candidate_metrics": [{"column": "Amount", "score": 0.7}],
            "chosen_dimensions": ["Country"],
            "extracted_filters": [],
            "value_to_column_matches": [],
            "generated_sql": "",
            "validation_result": "phase2_planner_degraded",
            "execution_result": "degraded",
            "failure_reason": "planner_degraded",
        }
    )
    assert ref["entry_id"]
    assert str(tmp_path) in ref["path"]