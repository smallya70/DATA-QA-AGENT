import pytest

from main_upgraded import (
    _classify_query_shape,
    _build_exec_params,
    _build_exception_customer_list_sql,
    _build_exception_detail_sql,
    _build_query_spec_stub,
    _build_semantic_role_sheet_summary_sql,
    _build_billing_lag_value_gap_sql,
    _extract_recommended_action_filter,
    _enrich_aggregate_with_filters_rows,
    _infer_semantic_roles_for_sheet,
    _infer_semantic_roles_from_sheet_metadata,
    _is_customer_value_gap_summary_question,
    _is_exception_customer_list_question,
    _is_exception_detail_question,
    _is_semantic_role_sheet_summary_question,
    _merge_semantic_roles,
    _pick_customer_fallback_sheets,
    _pick_exception_detail_source_sheet,
    _question_requests_explicit_aggregation,
)
from app.query_spec import QuerySpec
from app.sql_builder import QueryBuildError, build_sql_from_query_spec
from app.sql_validator import validate_sql


METADATA = {
    "sheets": ["Orders", "Customers"],
    "columns_by_sheet": {
        "Orders": ["order_id", "customer_id", "amount", "status", "OrderDate"],
        "Customers": ["customer_id", "customer_name", "region"],
    },
    "column_types_by_sheet": {
        "Orders": {"order_id": "text", "customer_id": "text", "amount": "numeric", "status": "text", "OrderDate": "date"},
        "Customers": {"customer_id": "text", "customer_name": "text", "region": "text"},
    },
    "relationships": [
        {"left": "Orders", "left_col": "customer_id", "right": "Customers", "right_col": "customer_id"}
    ],
}


def test_query_spec_validation():
    spec = QuerySpec(
        intent="sales_by_status",
        target_sheet="Orders",
        measures=[{"column": "amount", "aggregation": "sum", "alias": "total_amount"}],
        dimensions=["status"],
        order_by=[{"field": "total_amount", "direction": "desc"}],
        limit=10,
        confidence=0.8,
    )
    sql, params = build_sql_from_query_spec(spec, METADATA)
    assert params["limit"] == 10
    assert "GROUP BY s.row_json->>'status'" in sql
    assert not validate_sql(sql, spec, METADATA)


def test_sql_builder_simple_aggregation():
    spec = QuerySpec(
        intent="sales_by_status",
        target_sheet="Orders",
        measures=[{"column": "amount", "aggregation": "sum", "alias": "total_amount"}],
        dimensions=["status"],
        order_by=[{"field": "total_amount", "direction": "desc"}],
        limit=25,
    )
    sql, _ = build_sql_from_query_spec(spec, METADATA)
    assert "SUM(" in sql
    assert "LIMIT %(limit)s" in sql
    assert "dataset_id" in sql


def test_build_exec_params_preserves_runtime_limit_over_deterministic_params():
    params = _build_exec_params(
        "SELECT 1 WHERE dataset_id = %(dataset_id)s LIMIT %(limit)s AND %(filter_0)s IS NOT NULL",
        dataset_id="dataset-123",
        limit=1,
        question="which Product has highest Amount in Country India",
        extra_params={"limit": 100, "filter_0": "india"},
    )
    assert params["dataset_id"] == "dataset-123"
    assert params["limit"] == 1
    assert params["filter_0"] == "india"


def test_sql_builder_join():
    spec = QuerySpec(
        intent="sales_by_region",
        target_sheet="Orders",
        measures=[{"column": "amount", "aggregation": "sum", "alias": "total_amount"}],
        dimensions=["region"],
        join_path=[{"left_sheet": "Orders", "left_col": "customer_id", "right_sheet": "Customers", "right_col": "customer_id"}],
        limit=20,
    )
    sql, _ = build_sql_from_query_spec(spec, METADATA)
    assert "LEFT JOIN dataset_rows" in sql
    assert "Customers" in sql


def test_invalid_column_rejection():
    spec = QuerySpec(
        intent="bad_column",
        target_sheet="Orders",
        measures=[{"column": "missing_amount", "aggregation": "sum"}],
        limit=10,
    )
    with pytest.raises(QueryBuildError):
        build_sql_from_query_spec(spec, METADATA)


def test_invalid_aggregation_rejection():
    spec = QuerySpec(
        intent="bad_agg",
        target_sheet="Orders",
        measures=[{"column": "status", "aggregation": "sum"}],
        limit=10,
    )
    with pytest.raises(QueryBuildError):
        build_sql_from_query_spec(spec, METADATA)


def test_sql_validator_catches_unsafe_sql():
    spec = QuerySpec(intent="unsafe", target_sheet="Orders", limit=10)
    issues = validate_sql("DELETE FROM dataset_rows", spec, METADATA)
    assert "SQL must be SELECT-only." in issues
    assert "SQL contains non-SELECT operations." in issues


def test_sql_builder_year_grouping():
    spec = QuerySpec(
        intent="sales_by_year",
        target_sheet="Orders",
        measures=[{"column": "amount", "aggregation": "sum", "alias": "total_amount"}],
        time_grain="year",
        limit=10,
    )
    sql, _ = build_sql_from_query_spec(spec, METADATA)
    assert "AS year" in sql
    assert "EXTRACT(YEAR FROM" in sql


def test_sql_builder_year_filter():
    spec = QuerySpec(
        intent="sales_2024",
        target_sheet="Orders",
        measures=[{"column": "amount", "aggregation": "sum", "alias": "total_amount"}],
        filters=[{"kind": "year", "column": "order_id", "operator": "=", "value": 2024}],
        limit=10,
    )
    with pytest.raises(QueryBuildError):
        build_sql_from_query_spec(spec, METADATA)


def test_list_product_by_region_uses_dimensions_without_sum_measure():
    available_sheets = ["Sheet1"]
    sheet_columns_by_sheet = {
        "Sheet1": {"Product", "Region", "TotalPrice"},
    }

    spec = _build_query_spec_stub(
        "List Product by Region",
        available_sheets,
        sheet_columns_by_sheet,
        relationships=[],
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert spec is not None
    assert spec.target_sheet == "Sheet1"
    assert spec.dimensions == ["Product", "Region"]
    assert spec.measures == []
    assert spec.filters == []


def test_query_shape_classifier_list_rows_for_dimension_listing():
    shape = _classify_query_shape(
        "List Product by Region",
        ["Product", "Region"],
        [],
        None,
    )

    assert shape == "list_rows"


def test_query_shape_classifier_aggregate_with_filters_for_sales_value_question():
    shape = _classify_query_shape(
        "what is the Sales Value of Product chair in East Region",
        ["Product", "Region"],
        [
            {"column": "Product", "operator": "=", "value": "chair"},
            {"column": "Region", "operator": "=", "value": "east"},
        ],
        {"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"},
    )

    assert shape == "aggregate_with_filters"


def test_query_shape_classifier_aggregate_by_dimensions_for_sales_by_region():
    shape = _classify_query_shape(
        "Show sales by region",
        ["Region"],
        [],
        {"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"},
    )

    assert shape == "aggregate_by_dimensions"


def test_query_shape_classifier_aggregate_by_dimensions_for_metric_with_unfiltered_product_dimension():
    shape = _classify_query_shape(
        "Can You give me List of the Sum of Total Price for Products in Region East",
        ["Product", "Region"],
        [{"column": "Region", "operator": "=", "value": "east"}],
        {"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"},
    )

    assert shape == "aggregate_by_dimensions"


def test_list_product_in_region_east_adds_text_filter():
    available_sheets = ["Sheet1"]
    sheet_columns_by_sheet = {
        "Sheet1": {"Product", "Region", "TotalPrice"},
    }

    spec = _build_query_spec_stub(
        "List Product in Region =East",
        available_sheets,
        sheet_columns_by_sheet,
        relationships=[],
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert spec is not None
    assert spec.dimensions == ["Product"]
    assert spec.measures == []
    assert spec.filters == [{"column": "Region", "operator": "=", "value": "east"}]


def test_sales_value_of_product_chair_in_east_region_uses_filters_not_grouping():
    available_sheets = ["Sheet1"]
    sheet_columns_by_sheet = {
        "Sheet1": {"Product", "Region", "TotalPrice"},
    }
    value_samples_by_sheet = {
        "Sheet1": {
            "Product": ["chair", "desk", "printer"],
            "Region": ["east", "west", "north", "south"],
        }
    }

    spec = _build_query_spec_stub(
        "what is the Sales Value of Product chair in East Region",
        available_sheets,
        sheet_columns_by_sheet,
        relationships=[],
        explicit_sheet="Sheet1",
        max_rows=100,
        value_samples_by_sheet=value_samples_by_sheet,
    )

    assert spec is not None
    assert spec.dimensions == []
    assert spec.measures == [{"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"}]
    assert {tuple(sorted(item.items())) for item in spec.filters} == {
        tuple(sorted({"column": "Product", "operator": "=", "value": "chair"}.items())),
        tuple(sorted({"column": "Region", "operator": "=", "value": "east"}.items())),
    }


def test_sales_value_for_chair_in_the_east_region_uses_sampled_value_filters():
    available_sheets = ["Sheet1"]
    sheet_columns_by_sheet = {
        "Sheet1": {"Product", "Region", "TotalPrice"},
    }
    value_samples_by_sheet = {
        "Sheet1": {
            "Product": ["chair", "desk", "printer"],
            "Region": ["east", "west", "north", "south"],
        }
    }

    spec = _build_query_spec_stub(
        "Show Total Sales value for Chair in the East Region",
        available_sheets,
        sheet_columns_by_sheet,
        relationships=[],
        explicit_sheet="Sheet1",
        max_rows=100,
        value_samples_by_sheet=value_samples_by_sheet,
    )

    assert spec is not None
    assert spec.dimensions == []
    assert spec.measures == [{"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"}]
    assert {tuple(sorted(item.items())) for item in spec.filters} == {
        tuple(sorted({"column": "Product", "operator": "=", "value": "chair"}.items())),
        tuple(sorted({"column": "Region", "operator": "=", "value": "east"}.items())),
    }


def test_sum_total_price_for_products_in_region_east_groups_by_product_and_filters_region():
    available_sheets = ["Sheet1"]
    sheet_columns_by_sheet = {
        "Sheet1": {"Product", "Region", "TotalPrice"},
    }
    value_samples_by_sheet = {
        "Sheet1": {
            "Product": ["chair", "desk", "printer"],
            "Region": ["east", "west", "north", "south"],
        }
    }

    spec = _build_query_spec_stub(
        "Can You give me List of the Sum of Total Price for Products in Region East",
        available_sheets,
        sheet_columns_by_sheet,
        relationships=[],
        explicit_sheet="Sheet1",
        max_rows=100,
        value_samples_by_sheet=value_samples_by_sheet,
    )

    assert spec is not None
    assert spec.dimensions == ["Product"]
    assert spec.measures == [{"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"}]
    assert spec.filters == [{"column": "Region", "operator": "=", "value": "east"}]


def test_sql_builder_uses_select_distinct_for_dimension_only_queries():
    metadata = {
        "sheets": ["Sheet1"],
        "columns_by_sheet": {
            "Sheet1": ["Product", "Region", "TotalPrice"],
        },
        "column_types_by_sheet": {
            "Sheet1": {"Product": "text", "Region": "text", "TotalPrice": "numeric"},
        },
        "relationships": [],
    }
    spec = QuerySpec(
        intent="list_product_by_region",
        target_sheet="Sheet1",
        dimensions=["Product", "Region"],
        measures=[],
        limit=100,
    )

    sql, _ = build_sql_from_query_spec(spec, metadata)

    assert "SELECT DISTINCT" in sql
    assert "AS Product" in sql
    assert "AS Region" in sql
    assert "SUM(" not in sql


def test_sql_builder_text_equality_filters_are_case_insensitive():
    metadata = {
        "sheets": ["Sheet1"],
        "columns_by_sheet": {
            "Sheet1": ["Product", "Region", "TotalPrice"],
        },
        "column_types_by_sheet": {
            "Sheet1": {"Product": "text", "Region": "text", "TotalPrice": "numeric"},
        },
        "relationships": [],
    }
    spec = QuerySpec(
        intent="list_product_in_region_east",
        target_sheet="Sheet1",
        dimensions=["Product", "Region"],
        measures=[],
        filters=[{"column": "Region", "operator": "=", "value": "east"}],
        limit=100,
    )

    sql, params = build_sql_from_query_spec(spec, metadata)

    assert "LOWER(COALESCE(s.row_json->>'Region', '')) = %(filter_0)s" in sql
    assert params["filter_0"] == "east"


def test_sql_builder_filtered_measure_query_has_no_group_by_when_no_dimensions():
    metadata = {
        "sheets": ["Sheet1"],
        "columns_by_sheet": {
            "Sheet1": ["Product", "Region", "TotalPrice"],
        },
        "column_types_by_sheet": {
            "Sheet1": {"Product": "text", "Region": "text", "TotalPrice": "numeric"},
        },
        "relationships": [],
    }
    spec = QuerySpec(
        intent="sales_value_of_product_chair_in_east_region",
        target_sheet="Sheet1",
        dimensions=[],
        measures=[{"column": "TotalPrice", "aggregation": "sum", "alias": "sum_total_price"}],
        filters=[
            {"column": "Product", "operator": "=", "value": "chair"},
            {"column": "Region", "operator": "=", "value": "east"},
        ],
        limit=100,
    )

    sql, params = build_sql_from_query_spec(spec, metadata)

    assert "SUM(" in sql
    assert "GROUP BY" not in sql
    assert params["filter_0"] == "chair"
    assert params["filter_1"] == "east"


def test_enrich_aggregate_with_filters_rows_adds_filter_context_and_match_count():
    rows = [{"sum_total_price": 107210.1945}]
    planner_summary = {
        "query_shape": "aggregate_with_filters",
        "filters": [
            {"column": "Product", "operator": "=", "value": "chair"},
            {"column": "Region", "operator": "=", "value": "east"},
        ],
    }

    enriched = _enrich_aggregate_with_filters_rows(rows, planner_summary, 40)

    assert enriched == [{
        "product": "chair",
        "region": "east",
        "matched_rows": 40,
        "sum_total_price": 107210.1945,
    }]


def test_exception_detail_question_prefers_row_level_shape():
    assert _is_exception_detail_question("Show value gap by customer") is True
    assert _is_exception_detail_question("Show customer and value gap") is True
    assert _is_exception_detail_question("Show customer value gap details") is True
    assert _is_exception_detail_question("List customer value gap exceptions") is True
    assert _is_exception_detail_question("Can You list of Unbilled customers and expected amount") is True


def test_extract_recommended_action_filter_normalizes_invoice_now():
    assert _extract_recommended_action_filter(
        "SHOW Exception type and Customer Id when Recommended action=Invoice Now"
    ) == "invoice now"


def test_extract_recommended_action_filter_supports_implicit_invoice_now_phrase():
    assert _extract_recommended_action_filter(
        "Show Customer with Invoice Now"
    ) == "invoice now"


def test_explicit_aggregation_keeps_customer_summary_routes():
    assert _question_requests_explicit_aggregation("Show total value gap by customer") is True
    assert _question_requests_explicit_aggregation("Summarize value gap by customer") is True
    assert _question_requests_explicit_aggregation("Show value gap grouped by customer") is True
    assert _question_requests_explicit_aggregation("Show customer totals for value gap") is True
    assert _question_requests_explicit_aggregation("Show customer breakdown for value gap") is True
    assert _question_requests_explicit_aggregation("Show value gap breakdown by customer") is True
    assert _is_exception_detail_question("Show total value gap by customer") is False
    assert _is_exception_detail_question("Show value gap grouped by customer") is False
    assert _is_exception_detail_question("Show customer totals for value gap") is False
    assert _is_exception_detail_question("Show customer breakdown for value gap") is False


def test_customer_value_gap_summary_question_detection():
    assert _is_customer_value_gap_summary_question("Show total value gap by customer") is True
    assert _is_customer_value_gap_summary_question("Show customer summary for value gap") is True
    assert _is_customer_value_gap_summary_question("Show customer level summary for value gap") is True
    assert _is_customer_value_gap_summary_question("Show customer breakdown for value gap") is True
    assert _is_customer_value_gap_summary_question("Show value gap split by customer") is True
    assert _is_customer_value_gap_summary_question("Show value gap by customer") is False


def test_exception_customer_list_question_detection():
    assert _is_exception_customer_list_question("can You give list of Unbilled Customers") is True
    assert _is_exception_customer_list_question("List unbilled customers") is True
    assert _is_exception_customer_list_question("Can You list of Unbilled customers and expected amount") is False
    assert _is_exception_customer_list_question("Show total unbilled value gap by customer") is False


def test_exception_routing_uses_semantic_roles_for_generic_sheet_names():
    available_sheets = ["Exceptions_2026", "EventLedger", "BillingFacts"]
    sheet_columns_by_sheet = {
        "Exceptions_2026": {"case_key", "party_code", "kind", "amount_due", "gap_value"},
        "EventLedger": {"case_key", "ship_to_party", "service_day"},
        "BillingFacts": {"case_key", "invoice_number", "amount_billed"},
    }
    semantic_roles = {
        "Exceptions_2026": {
            "event_id": "case_key",
            "customer_id": "party_code",
            "exception_type": "kind",
            "expected_amount": "amount_due",
            "value_gap": "gap_value",
        },
        "EventLedger": {
            "event_id": "case_key",
            "customer_id": "ship_to_party",
            "service_date": "service_day",
        },
        "BillingFacts": {
            "event_id": "case_key",
            "invoice_id": "invoice_number",
            "billed_amount": "amount_billed",
        },
    }
    relationships = [
        {"left": "Exceptions_2026", "left_col": "case_key", "right": "EventLedger", "right_col": "case_key"},
        {"left": "Exceptions_2026", "left_col": "case_key", "right": "BillingFacts", "right_col": "case_key"},
    ]

    assert _pick_exception_detail_source_sheet(
        available_sheets,
        sheet_columns_by_sheet,
        "List unbilled customers and amount due",
        relationships=relationships,
        semantic_roles=semantic_roles,
    ) == "Exceptions_2026"

    assert _pick_customer_fallback_sheets(
        available_sheets,
        sheet_columns_by_sheet,
        "Exceptions_2026",
        relationships=relationships,
        semantic_roles=semantic_roles,
    )[0] == "EventLedger"


def test_exception_detail_sql_filters_recommended_action_when_requested():
    available_sheets = ["Agent_Output"]
    sheet_columns_by_sheet = {
        "Agent_Output": {
            "event_id",
            "customer",
            "exception_type",
            "service_type",
            "service_date",
            "invoice_id",
            "invoice_date",
            "expected_amount",
            "billed_amount",
            "value_gap",
            "billing_lag_days",
            "recommended_action",
        }
    }

    sql = _build_exception_detail_sql(
        available_sheets,
        sheet_columns_by_sheet,
        "SHOW Exception type and Customer Id when Recommended action=Invoice Now",
        forced_sheet="Agent_Output",
    )

    assert sql is not None
    assert "recommended_action" in sql
    assert "regexp_replace(lower(COALESCE(s.row_json->>'recommended_action', '')), '[^a-z0-9]+', ' ', 'g') LIKE '%invoice now%'" in sql


def test_exception_detail_sql_filters_recommended_action_when_phrase_is_implicit():
    available_sheets = ["Agent_Output"]
    sheet_columns_by_sheet = {
        "Agent_Output": {
            "event_id",
            "customer",
            "exception_type",
            "service_type",
            "service_date",
            "invoice_id",
            "invoice_date",
            "expected_amount",
            "billed_amount",
            "value_gap",
            "billing_lag_days",
            "recommended_action",
        }
    }

    sql = _build_exception_detail_sql(
        available_sheets,
        sheet_columns_by_sheet,
        "Show Customer with Invoice Now",
        forced_sheet="Agent_Output",
    )

    assert sql is not None
    assert "recommended_action" in sql
    assert "regexp_replace(lower(COALESCE(s.row_json->>'recommended_action', '')), '[^a-z0-9]+', ' ', 'g') LIKE '%invoice now%'" in sql


def test_exception_customer_list_question_detects_implicit_recommended_action_phrase():
    assert _is_exception_customer_list_question("Show Customer with Invoice Now") is True
    assert _is_exception_customer_list_question("Show Customer with Recommended Action Invoice Now") is True


def test_exception_customer_list_sql_filters_recommended_action_without_empty_exception_in_clause():
    available_sheets = ["Agent_Output"]
    sheet_columns_by_sheet = {
        "Agent_Output": {
            "event_id",
            "customer",
            "exception_type",
            "service_type",
            "service_date",
            "invoice_id",
            "invoice_date",
            "expected_amount",
            "billed_amount",
            "value_gap",
            "billing_lag_days",
            "recommended_action",
        }
    }

    sql = _build_exception_customer_list_sql(
        available_sheets,
        sheet_columns_by_sheet,
        "Show Customer with Recommended action Invoice Now",
        forced_sheet="Agent_Output",
    )

    assert sql is not None
    assert "%invoice now%" in sql
    assert " IN ()" not in sql
    assert " AS event_id" in sql
    assert " AS recommended_action" in sql
    assert "SELECT DISTINCT ON" in sql


def test_exception_detail_sql_uses_recommended_action_semantic_role_mapping():
    available_sheets = ["Exceptions_2026"]
    sheet_columns_by_sheet = {
        "Exceptions_2026": {
            "case_key",
            "party_code",
            "kind",
            "next_action",
            "gap_value",
        }
    }
    semantic_roles = {
        "Exceptions_2026": {
            "event_id": "case_key",
            "customer_id": "party_code",
            "exception_type": "kind",
            "value_gap": "gap_value",
            "recommended_action": "next_action",
        }
    }

    sql = _build_exception_detail_sql(
        available_sheets,
        sheet_columns_by_sheet,
        "show customer id when recommended action = invoice now",
        forced_sheet="Exceptions_2026",
        semantic_roles=semantic_roles,
    )

    assert sql is not None
    assert "s.row_json->>'next_action' AS recommended_action" in sql
    assert "regexp_replace(lower(COALESCE(s.row_json->>'next_action', '')), '[^a-z0-9]+', ' ', 'g') LIKE '%invoice now%'" in sql


def test_semantic_role_inference_handles_generic_exception_sheet_names():
    inferred = _infer_semantic_roles_for_sheet(
        "Exceptions_2026",
        ["case_key", "party_code", "kind", "amount_due", "gap_value"],
        {
            "case_key": "text",
            "party_code": "text",
            "kind": "text",
            "amount_due": "numeric",
            "gap_value": "numeric",
        },
    )

    assert inferred["event_id"] == "case_key"
    assert inferred["customer_id"] == "party_code"
    assert inferred["exception_type"] == "kind"
    assert inferred["expected_amount"] == "amount_due"
    assert inferred["value_gap"] == "gap_value"


def test_semantic_role_inference_marks_invoice_status_on_invoice_sheet():
    inferred = _infer_semantic_roles_for_sheet(
        "Invoices_Synthetic",
        ["invoice_number", "status", "amount", "invoice_date"],
        {
            "invoice_number": "text",
            "status": "text",
            "amount": "numeric",
            "invoice_date": "date",
        },
    )

    assert inferred["invoice_id"] == "invoice_number"
    assert inferred["invoice_status"] == "status"
    assert inferred["billed_amount"] == "amount"
    assert inferred["invoice_date"] == "invoice_date"


def test_semantic_role_inference_detects_recommended_action_from_next_action_column():
    inferred = _infer_semantic_roles_for_sheet(
        "Exceptions_2026",
        ["case_key", "party_code", "kind", "next_action", "gap_value"],
        {
            "case_key": "text",
            "party_code": "text",
            "kind": "text",
            "next_action": "text",
            "gap_value": "numeric",
        },
    )

    assert inferred["recommended_action"] == "next_action"


def test_semantic_role_merge_preserves_existing_roles_when_not_overwriting():
    existing = {
        "Orders": {"customer_id": "customer_code"},
    }
    inferred = _infer_semantic_roles_from_sheet_metadata(
        {"Orders": {"customer_id", "sales_amount", "region"}},
        {"Orders": {"customer_id": "text", "sales_amount": "numeric", "region": "text"}},
    )

    merged = _merge_semantic_roles(existing, inferred, overwrite=False)

    assert merged["Orders"]["customer_id"] == "customer_code"
    assert merged["Orders"]["revenue"] == "sales_amount"


def test_semantic_role_sheet_summary_question_detection():
    assert _is_semantic_role_sheet_summary_question("Show a sheet-wise summary using detected semantic roles.") is True
    assert _is_semantic_role_sheet_summary_question("Show semantic roles by sheet") is True
    assert _is_semantic_role_sheet_summary_question("Show total value gap by customer") is False


def test_semantic_role_sheet_summary_sql_lists_sheets_and_roles():
    sql = _build_semantic_role_sheet_summary_sql(
        {
            "Summary": {"exception_type": "exception_type", "total_value_gap": "total_value_gap"},
            "Ops_Events": {"event_id": "event_id", "customer_id": "customer", "expected_amount": "expected_amount"},
        }
    )

    assert sql is not None
    assert "semantic_summary(sheet_name, role_count, detected_roles)" in sql
    assert "Summary" in sql
    assert "Ops_Events" in sql
    assert "detected_roles" in sql
    assert "COUNT(*) AS row_count" in sql


def test_semantic_role_inference_does_not_duplicate_total_value_gap_as_value_gap():
    inferred = _infer_semantic_roles_for_sheet(
        "Summary",
        ["exception_type", "count", "total_value_gap"],
        {
            "exception_type": "text",
            "count": "numeric",
            "total_value_gap": "numeric",
        },
    )

    assert inferred["exception_type"] == "exception_type"
    assert inferred["total_value_gap"] == "total_value_gap"
    assert "value_gap" not in inferred


def test_semantic_role_sheet_summary_sql_drops_duplicate_summary_value_gap_role():
    sql = _build_semantic_role_sheet_summary_sql(
        {
            "Summary": {
                "count": "count",
                "exception_type": "exception_type",
                "total_value_gap": "total_value_gap",
                "value_gap": "total_value_gap",
            }
        }
    )

    assert sql is not None
    assert "('Summary', 3, 'count, exception_type, total_value_gap')" in sql
    assert "count, exception_type, total_value_gap" in sql
    assert "count, exception_type, total_value_gap, value_gap" not in sql


def test_billing_lag_sql_prefers_fallback_alphanumeric_customer():
    available_sheets = ["Agent_Output", "Ops_Events"]
    sheet_columns_by_sheet = {
        "Agent_Output": {"event_id", "customer", "billing_lag_days", "value_gap"},
        "Ops_Events": {"event_id", "customer", "service_date"},
    }
    semantic_roles = {
        "Agent_Output": {
            "event_id": "event_id",
            "customer_id": "customer",
            "billing_lag_days": "billing_lag_days",
            "value_gap": "value_gap",
        },
        "Ops_Events": {
            "event_id": "event_id",
            "customer_id": "customer",
            "service_date": "service_date",
        },
    }
    relationships = [
        {"left": "Agent_Output", "left_col": "event_id", "right": "Ops_Events", "right_col": "event_id"},
    ]

    sql = _build_billing_lag_value_gap_sql(
        available_sheets,
        sheet_columns_by_sheet,
        "List event_id and customer_id from Agent_Output where billing_lag_days > 7 with value gap amount.",
        relationships=relationships,
        semantic_roles=semantic_roles,
        forced_sheet="Agent_Output",
    )

    assert sql is not None
    assert "LEFT JOIN dataset_rows cust1" in sql
    assert "COALESCE(NULLIF(TRIM(cust1.row_json->>'customer'), ''), NULLIF(TRIM(s.row_json->>'customer'), '')) AS customer_id" in sql
