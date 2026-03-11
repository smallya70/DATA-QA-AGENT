from app.planning_context import build_planning_context
from app.sheet_profile import build_sheet_profile


def test_build_sheet_profile_for_summary_sheet_marks_preaggregated_and_infers_fields():
    planning_context = build_planning_context(
        dataset_id="sheet-profile-summary-test",
        cols=["exception_type", "count", "total_value_gap"],
        col_types={
            "exception_type": "text",
            "count": "numeric",
            "total_value_gap": "numeric",
        },
        relationships_payload={
            "tables": ["Summary"],
            "relationships": [],
            "semantic_roles": {},
        },
        sheet_rows=[
            {"sheet_name": "Summary", "row_json": {"exception_type": "unbilled", "count": "3", "total_value_gap": "12"}},
            {"sheet_name": "Summary", "row_json": {"exception_type": "lagged", "count": "5", "total_value_gap": "7"}},
        ],
    )
    sheet_rows = [
        {"sheet_name": "Summary", "row_json": {"exception_type": "unbilled", "count": "3", "total_value_gap": "12"}},
        {"sheet_name": "Summary", "row_json": {"exception_type": "lagged", "count": "5", "total_value_gap": "7"}},
    ]

    profile = build_sheet_profile("Summary", planning_context, sheet_rows)

    assert profile["sheet_name"] == "Summary"
    assert profile["row_count"] == 2
    assert profile["column_count"] == 3
    assert profile["column_names"] == ["exception_type", "count", "total_value_gap"]
    assert profile["inferred_column_types"]["exception_type"] == "categorical"
    assert profile["inferred_column_types"]["count"] == "numeric"
    assert profile["inferred_column_types"]["total_value_gap"] == "numeric"
    assert profile["appears_pre_aggregated"] is True
    assert profile["numeric_field_summaries"] == [
        {"column": "count", "non_null_count": 2, "min": 3.0, "max": 5.0, "avg": 4.0},
        {"column": "total_value_gap", "non_null_count": 2, "min": 7.0, "max": 12.0, "avg": 9.5},
    ]
    assert profile["categorical_field_summaries"] == [
        {"column": "exception_type", "distinct_count": 2, "top_values": ["unbilled", "lagged"]},
    ]