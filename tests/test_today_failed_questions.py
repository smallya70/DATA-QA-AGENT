from __future__ import annotations

import pytest

from app.query_planner import plan_query
from main_upgraded import (
    _build_exception_detail_sql,
    _extract_recommended_action_filter,
    _is_customer_value_gap_summary_question,
    _is_exception_customer_list_question,
    _is_exception_detail_question,
    _should_apply_global_top_1_limit,
)
from tests.gold.today_failed_questions import TODAY_FAILED_QUESTION_SET


@pytest.mark.parametrize("case", TODAY_FAILED_QUESTION_SET, ids=[item["question"] for item in TODAY_FAILED_QUESTION_SET])
def test_today_failed_questions_regression(case, request):
    mode = case["mode"]
    question = case["question"]

    if mode == "planner":
        context = request.getfixturevalue(case["context_fixture"])
        result = plan_query(question, context, explicit_sheet=None, max_rows=100)
        assert result.degraded_mode is False
        assert result.query_shape == case["expected_shape"]
        assert result.canonical_intent == case["expected_intent"]
        assert result.target_sheet == case["expected_sheet"]
        assert result.query_spec is not None
        assert result.query_spec.measures[0]["column"] == case["expected_metric"]
        if case.get("expected_dimension"):
            assert case["expected_dimension"] in result.query_spec.dimensions or case["expected_dimension"] in result.selected_dimensions
        filters = {(item["column"], str(item["value"])) for item in result.query_spec.filters}
        for expected_filter in case.get("expected_filters", []):
            assert expected_filter in filters
        return

    if mode == "helper_recommended_action_filter":
        assert _extract_recommended_action_filter(question) == case["expected_value"]
        return

    if mode == "helper_exception_detail_sql":
        sql = _build_exception_detail_sql(
            case["available_sheets"],
            case["sheet_columns_by_sheet"],
            question,
            forced_sheet=case["forced_sheet"],
        )
        assert sql is not None
        for fragment in case["expected_sql_contains"]:
            assert fragment in sql
        return

    if mode == "helper_customer_value_gap_question":
        assert _is_customer_value_gap_summary_question(question) is case["expected_value"]
        return

    if mode == "helper_exception_customer_list_question":
        assert _is_exception_customer_list_question(question) is case["expected_value"]
        return

    if mode == "helper_exception_detail_question":
        assert _is_exception_detail_question(question) is case["expected_value"]
        return

    raise AssertionError(f"Unsupported today-failed regression mode: {mode}")


def test_per_parent_ranking_does_not_force_global_top_1_limit(chocolate_planning_context):
    result = plan_query("worst product for each country by lowest amount", chocolate_planning_context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "ranking"
    assert _should_apply_global_top_1_limit(True, result.query_spec) is False