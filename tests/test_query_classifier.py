from app.query_classifier import classify_query


def test_classifier_detects_list_rows(sample_planning_context):
    result = classify_query("List Product by Region", sample_planning_context)
    assert result.shape == "list_rows"
    assert result.confidence >= 0.8


def test_classifier_detects_row_preview(sample_planning_context):
    result = classify_query("Show first 20 rows from sheet Sheet1", sample_planning_context)
    assert result.shape == "row_preview"


def test_classifier_detects_describe_sheet(sample_planning_context):
    result = classify_query("Describe Sheet1", sample_planning_context)
    assert result.shape == "describe_sheet"


def test_classifier_detects_aggregate_scalar(sample_planning_context):
    result = classify_query("Show Total Sales value for Chair in the East Region", sample_planning_context)
    assert result.shape == "aggregate_scalar"


def test_classifier_detects_trend(sample_planning_context):
    result = classify_query("Show total sales by month", sample_planning_context)
    assert result.shape == "trend_over_time"


def test_classifier_detects_quarter_trend(sample_planning_context):
    result = classify_query("Show total sales by quarter", sample_planning_context)
    assert result.shape == "trend_over_time"


def test_classifier_detects_comparison_phrase(sample_planning_context):
    result = classify_query("Compare total sales East vs West", sample_planning_context)
    assert result.shape == "comparison"


def test_classifier_treats_year_constraint_as_scalar_filter(sample_planning_context):
    result = classify_query(
        "What is the Sum of Total Price on Chair in Region South and Year 2025",
        sample_planning_context,
    )
    assert result.shape == "aggregate_scalar"