from app.filter_extractor import extract_filters
from app.value_mapper import map_values_to_columns


def test_maps_sample_values_to_expected_columns(sample_planning_context):
    candidates = extract_filters("Show Total Sales value for Chair in the East Region", sample_planning_context, target_sheet="Sheet1")
    mappings, warnings = map_values_to_columns(
        "Show Total Sales value for Chair in the East Region",
        sample_planning_context,
        target_sheet="Sheet1",
        candidates=candidates,
    )
    assert warnings == []
    resolved = {(mapping.column, mapping.value) for mapping in mappings}
    assert ("Product", "chair") in resolved
    assert ("Region", "east") in resolved


def test_maps_explicit_region_filter(sample_planning_context):
    candidates = extract_filters("List Product in Region =East", sample_planning_context, target_sheet="Sheet1")
    mappings, _ = map_values_to_columns(
        "List Product in Region =East",
        sample_planning_context,
        target_sheet="Sheet1",
        candidates=candidates,
    )
    assert any(mapping.column == "Region" and mapping.value == "east" for mapping in mappings)