from app.filter_extractor import extract_filters


def test_extracts_explicit_column_filter(sample_planning_context):
    filters = extract_filters("List Product in Region =East", sample_planning_context, target_sheet="Sheet1")
    assert any(item.column_hint == "Region" and item.raw_value == "east" for item in filters)


def test_extracts_region_phrase_and_generic_value(sample_planning_context):
    filters = extract_filters("Show Total Sales value for Chair in the East Region", sample_planning_context, target_sheet="Sheet1")
    assert any(item.column_hint == "region" and item.raw_value == "east" for item in filters)
    assert any(item.raw_value == "chair" for item in filters)


def test_extracts_comparison_phrase_values(sample_planning_context):
    filters = extract_filters("Compare total sales East vs West", sample_planning_context, target_sheet="Sheet1")
    raw_values = [item.raw_value for item in filters]
    assert "east" in raw_values
    assert "west" in raw_values


def test_extracts_explicit_year_equality_without_overcapturing_region(sample_planning_context):
    filters = extract_filters(
        "What is the Sum of Total Price on Chair in Region South and Year 2025",
        sample_planning_context,
        target_sheet="Sheet1",
    )
    region_values = [item.raw_value for item in filters if str(item.column_hint or "").strip().lower() == "region"]
    year_filters = [item for item in filters if item.kind == "year"]
    assert "south" in region_values
    assert all(value != "south and year 2025" for value in region_values)
    assert any(str(item.raw_value) == "2025" and item.operator == "=" for item in year_filters)