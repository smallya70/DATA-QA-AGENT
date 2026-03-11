from app.guidance import build_planner_guidance
from app.planning_context import build_planning_context
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


def _ranking_sheet1_context():
    return build_planning_context(
        dataset_id="ranking-sheet1-test",
        cols=["order_id", "product", "country", "amount", "Cost", "Profit", "order_date"],
        col_types={
            "order_id": "text",
            "product": "text",
            "country": "text",
            "amount": "numeric",
            "Cost": "numeric",
            "Profit": "numeric",
            "order_date": "date",
        },
        relationships_payload={
            "tables": ["Sheet1"],
            "relationships": [],
            "semantic_roles": {
                "Sheet1": {
                    "product": "product",
                    "country": "country",
                    "amount": "amount",
                    "profit": "Profit",
                    "date": "order_date",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "Sheet1", "row_json": {"order_id": "O1", "product": "Chair", "country": "India", "amount": 1200, "Cost": 900, "Profit": 300, "order_date": "2024-01-01"}},
            {"sheet_name": "Sheet1", "row_json": {"order_id": "O2", "product": "Desk", "country": "USA", "amount": 1300, "Cost": 1000, "Profit": 300, "order_date": "2024-01-02"}},
            {"sheet_name": "Sheet1", "row_json": {"order_id": "O3", "product": "Chair", "country": "USA", "amount": 700, "Cost": 500, "Profit": 200, "order_date": "2024-01-03"}},
        ],
    )


def _derived_sales_context():
    return build_planning_context(
        dataset_id="derived-sales-test",
        cols=["customer", "product", "quantity", "unit_price", "cost", "order_date"],
        col_types={
            "customer": "text",
            "product": "text",
            "quantity": "numeric",
            "unit_price": "numeric",
            "cost": "numeric",
            "order_date": "date",
        },
        relationships_payload={
            "tables": ["Orders"],
            "relationships": [],
            "semantic_roles": {
                "Orders": {
                    "customer_id": "customer",
                    "product": "product",
                    "quantity": "quantity",
                    "unit_price": "unit_price",
                    "cost": "cost",
                    "date": "order_date",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "Orders", "row_json": {"customer": "A", "product": "Widget", "quantity": 2, "unit_price": 100, "cost": 120, "order_date": "2024-01-01"}},
            {"sheet_name": "Orders", "row_json": {"customer": "B", "product": "Widget", "quantity": 1, "unit_price": 150, "cost": 80, "order_date": "2024-01-02"}},
            {"sheet_name": "Orders", "row_json": {"customer": "A", "product": "Gadget", "quantity": 3, "unit_price": 50, "cost": 90, "order_date": "2024-01-03"}},
        ],
    )


def _inventory_context():
    return build_planning_context(
        dataset_id="inventory-value-test",
        cols=["warehouse", "stock_qty", "unit_price"],
        col_types={
            "warehouse": "text",
            "stock_qty": "numeric",
            "unit_price": "numeric",
        },
        relationships_payload={
            "tables": ["Inventory"],
            "relationships": [],
            "semantic_roles": {
                "Inventory": {
                    "warehouse": "warehouse",
                    "stock_qty": "stock_qty",
                    "unit_price": "unit_price",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "Inventory", "row_json": {"warehouse": "W1", "stock_qty": 10, "unit_price": 5}},
            {"sheet_name": "Inventory", "row_json": {"warehouse": "W2", "stock_qty": 3, "unit_price": 20}},
        ],
    )


def _value_gap_context():
    return build_planning_context(
        dataset_id="value-gap-derived-test",
        cols=["exception_type", "expected_amount", "actual_amount"],
        col_types={
            "exception_type": "text",
            "expected_amount": "numeric",
            "actual_amount": "numeric",
        },
        relationships_payload={
            "tables": ["Exceptions"],
            "relationships": [],
            "semantic_roles": {
                "Exceptions": {
                    "exception_type": "exception_type",
                    "expected_amount": "expected_amount",
                    "actual_amount": "actual_amount",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "Exceptions", "row_json": {"exception_type": "UNDER_BILLED", "expected_amount": 150, "actual_amount": 100}},
            {"sheet_name": "Exceptions", "row_json": {"exception_type": "OVER_BILLED", "expected_amount": 120, "actual_amount": 140}},
        ],
    )


def test_planner_builds_list_query(sample_planning_context):
    result = plan_query(
        "List Product by Region",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.measures == []
    assert set(result.query_spec.dimensions) == {"Product", "Region"}


def test_planner_builds_row_preview_query_without_aggregate_override(sample_planning_context):
    result = plan_query(
        "Show first 20 rows from sheet Sheet1",
        sample_planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "row_preview"
    assert result.answer_type == "table"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "row_preview"
    assert result.query_spec.limit == 20
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == []
    assert result.query_spec.columns == ["Date", "Region", "Product", "Quantity", "UnitPrice", "TotalPrice", "CustomerType"]
    assert result.query_spec.debug["skip_aggregate_pattern_matching"] is True
    assert result.query_spec.debug["intent_family"] == "row_preview"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(sample_planning_context))
    assert "SUM(" not in sql
    assert "COUNT(" not in sql
    assert "GROUP BY" not in sql
    assert "row_json->>'TotalPrice' AS TotalPrice" in sql


def test_planner_builds_describe_sheet_query_for_sheet_name_without_literal_sheet_keyword(sample_planning_context):
    result = plan_query("Describe Sheet1", sample_planning_context, explicit_sheet="Sheet1", max_rows=100)

    assert result.degraded_mode is False
    assert result.query_shape == "describe_sheet"
    assert result.answer_type == "sheet_profile"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "describe_sheet"
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == []
    assert result.query_spec.debug["skip_aggregate_pattern_matching"] is True
    assert result.query_spec.debug["intent_family"] == "describe_sheet"


def test_planner_keeps_explicit_total_value_gap_by_exception_type_as_aggregate():
    planning_context = build_planning_context(
        dataset_id="value-gap-aggregate-test",
        cols=["exception_type", "value_gap", "customer"],
        col_types={
            "exception_type": "text",
            "value_gap": "numeric",
            "customer": "text",
        },
        relationships_payload={
            "tables": ["Agent_Output"],
            "relationships": [],
            "semantic_roles": {
                "Agent_Output": {
                    "exception_type": "exception_type",
                    "value_gap": "value_gap",
                    "customer_id": "customer",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "Agent_Output", "row_json": {"exception_type": "UNDER_BILLED", "value_gap": 50, "customer": "A"}},
            {"sheet_name": "Agent_Output", "row_json": {"exception_type": "UNDER_BILLED", "value_gap": 60, "customer": "B"}},
            {"sheet_name": "Agent_Output", "row_json": {"exception_type": "OVER_BILLED", "value_gap": 10, "customer": "C"}},
        ],
    )

    result = plan_query(
        "Total value_gap by exception_type",
        planning_context,
        explicit_sheet=None,
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_grouped"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_spec.dimensions == ["exception_type"]
    assert result.query_spec.measures[0]["column"] == "value_gap"
    assert result.query_spec.measures[0]["aggregation"] == "sum"


def test_planner_blocks_unsupported_top_customers_by_revenue_query():
    planning_context = _ranking_sheet1_context()

    result = plan_query(
        "Who are the top 10 customers by revenue in Sheet1?",
        planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is True
    assert result.query_type == "unsupported_query"
    assert result.query_shape == "unsupported_query"
    assert result.query_spec is None
    assert result.fallback_reason == "missing_ranking_dimension"
    assert result.requested_dimension == "customers"
    assert result.requested_metric == "revenue"
    assert result.resolved_metric == "amount"
    assert "customer" in str(result.missing_dimension_reason or "").lower()
    assert result.available_dimensions == ["order_id", "product", "country"]
    assert "amount" in result.available_metrics
    assert "Cost" in result.available_metrics
    assert "Profit" in result.available_metrics
    assert result.guidance is not None
    assert any("amount" in suggestion.lower() for suggestion in (result.guidance.get("suggested_questions") or []))


def test_planner_builds_supported_top_products_by_amount_query():
    planning_context = _ranking_sheet1_context()

    result = plan_query(
        "Top 10 products by amount in Sheet1",
        planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_spec.dimensions == ["product"]
    assert result.query_spec.measures[0]["column"] == "amount"
    assert result.query_spec.measures[0]["aggregation"] == "sum"
    assert result.query_spec.limit == 10


def test_planner_builds_top_customers_by_revenue_from_derived_metric():
    planning_context = _derived_sales_context()

    result = plan_query(
        "Top customers by revenue",
        planning_context,
        explicit_sheet="Orders",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["customer"]
    measure = result.query_spec.measures[0]
    assert measure["column"] == "revenue"
    assert measure["aggregation"] == "sum"
    assert measure["resolved_metric_type"] == "derived"
    assert measure["requested_metric"] == "revenue"
    assert measure["canonical_metric"] == "revenue"
    assert measure["formula"] == "quantity * unit_price"
    assert result.query_spec.debug["requested_metric"] == "revenue"
    assert result.query_spec.debug["canonical_metric"] == "revenue"
    assert result.query_spec.debug["resolved_metric_type"] == "derived"


def test_planner_builds_top_customers_by_sales_via_revenue_alias():
    planning_context = _derived_sales_context()

    result = plan_query(
        "Top customers by sales",
        planning_context,
        explicit_sheet="Orders",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["customer"]
    measure = result.query_spec.measures[0]
    assert measure["column"] == "revenue"
    assert measure["resolved_metric_type"] == "derived"
    assert measure["requested_metric"] == "sales"
    assert measure["canonical_metric"] == "revenue"
    assert measure["formula"] == "quantity * unit_price"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "quantity" in sql
    assert "unit_price" in sql
    assert "sum_unit_price" not in sql


def test_planner_builds_supported_top_countries_by_cost_query():
    planning_context = _ranking_sheet1_context()

    result = plan_query(
        "Top 5 Country by Cost",
        planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_spec.dimensions == ["country"]
    assert result.query_spec.measures[0]["column"] == "Cost"
    assert result.query_spec.measures[0]["aggregation"] == "sum"
    assert result.query_spec.limit == 5

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "ORDER BY sum_cost DESC" in sql
    assert "sum_amount" not in sql


def test_planner_builds_total_inventory_value_by_warehouse_from_derived_metric():
    planning_context = _inventory_context()

    result = plan_query(
        "Total inventory value by warehouse",
        planning_context,
        explicit_sheet="Inventory",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["warehouse"]
    measure = result.query_spec.measures[0]
    assert measure["column"] == "inventory_value"
    assert measure["formula"] == "stock_qty * unit_price"
    assert measure["resolved_metric_type"] == "derived"


def test_planner_builds_total_value_gap_by_exception_type_from_alias_resolution():
    planning_context = _value_gap_context()

    result = plan_query(
        "Total variance by exception_type",
        planning_context,
        explicit_sheet="Exceptions",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["exception_type"]
    measure = result.query_spec.measures[0]
    assert measure["column"] == "value_gap"
    assert measure["formula"] == "expected_amount - actual_amount"
    assert measure["requested_metric"] == "variance"
    assert measure["canonical_metric"] == "value_gap"
    assert measure["resolved_metric_type"] == "derived"


def test_planner_builds_profit_by_product_from_derived_metric():
    planning_context = _derived_sales_context()

    result = plan_query(
        "Profit by product",
        planning_context,
        explicit_sheet="Orders",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["product"]
    measure = result.query_spec.measures[0]
    assert measure["column"] == "profit"
    assert measure["formula"] == "revenue - cost"
    assert measure["resolved_metric_type"] == "derived"


def test_planner_builds_supported_top_countries_by_profit_query():
    planning_context = _ranking_sheet1_context()

    result = plan_query(
        "Top countries by profit in Sheet1",
        planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_spec.dimensions == ["country"]
    assert result.query_spec.measures[0]["column"] == "Profit"
    assert result.query_spec.measures[0]["aggregation"] in {"sum", "max"}


def test_planner_keeps_scalar_highest_amount_query_valid_for_sheet1():
    planning_context = _ranking_sheet1_context()

    result = plan_query(
        "Highest amount in Sheet1",
        planning_context,
        explicit_sheet="Sheet1",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_shape == "aggregate_scalar"
    assert result.query_spec.measures[0]["column"] == "amount"
    assert result.query_spec.measures[0]["aggregation"] == "max"


def test_planner_builds_generic_list_query_from_explicit_sheet_column():
    planning_context = build_planning_context(
        dataset_id="summary-list-test",
        cols=["exception_type", "invoice_id", "total_value_gap"],
        col_types={
            "exception_type": "text",
            "invoice_id": "text",
            "total_value_gap": "numeric",
        },
        relationships_payload={
            "tables": ["Summary"],
            "relationships": [],
            "semantic_roles": {},
        },
        sheet_rows=[
            {"sheet_name": "Summary", "row_json": {"exception_type": "unbilled", "invoice_id": "INV-1", "total_value_gap": "12"}},
            {"sheet_name": "Summary", "row_json": {"exception_type": "lagged", "invoice_id": "INV-2", "total_value_gap": "7"}},
        ],
    )

    result = plan_query(
        "List exception_type from Summary",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "list"
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == ["exception_type"]


def test_planner_builds_explicit_sheet_list_query_without_measure_on_multi_numeric_sheet():
    planning_context = build_planning_context(
        dataset_id="summary-list-multi-metric-test",
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

    result = plan_query(
        "List exception_type in Summary Sheet",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == ["exception_type"]


def test_planner_builds_generic_grouped_list_query_from_explicit_sheet_columns():
    planning_context = build_planning_context(
        dataset_id="summary-grouped-list-test",
        cols=["exception_type", "invoice_id", "total_value_gap"],
        col_types={
            "exception_type": "text",
            "invoice_id": "text",
            "total_value_gap": "numeric",
        },
        relationships_payload={
            "tables": ["Summary"],
            "relationships": [],
            "semantic_roles": {},
        },
        sheet_rows=[
            {"sheet_name": "Summary", "row_json": {"exception_type": "unbilled", "invoice_id": "INV-1", "total_value_gap": "12"}},
            {"sheet_name": "Summary", "row_json": {"exception_type": "lagged", "invoice_id": "INV-2", "total_value_gap": "7"}},
        ],
    )

    result = plan_query(
        "Show invoice_id by exception_type from Summary",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "list"
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == ["invoice_id", "exception_type"]


def test_planner_builds_invoice_list_query_without_sheet_ambiguity_for_under_billed_invoices():
    planning_context = build_planning_context(
        dataset_id="billing-under-billed-invoices-test",
        cols=["invoice_id", "exception_type", "expected_amount", "billed_amount", "value_gap", "customer"],
        col_types={
            "invoice_id": "text",
            "exception_type": "text",
            "expected_amount": "numeric",
            "billed_amount": "numeric",
            "value_gap": "numeric",
            "customer": "text",
        },
        relationships_payload={
            "tables": ["Agent_Output", "Invoices_Synthetic"],
            "relationships": [],
            "semantic_roles": {
                "Agent_Output": {
                    "invoice_id": "invoice_id",
                    "status": "exception_type",
                    "expected_amount": "expected_amount",
                    "billed_amount": "billed_amount",
                    "value_gap": "value_gap",
                    "customer_id": "customer",
                },
                "Invoices_Synthetic": {
                    "invoice_id": "invoice_id",
                    "billed_amount": "billed_amount",
                    "customer_id": "customer",
                },
            },
        },
        sheet_rows=[
            {
                "sheet_name": "Agent_Output",
                "row_json": {
                    "invoice_id": "INV-1",
                    "exception_type": "UNDER_BILLED",
                    "expected_amount": "150",
                    "billed_amount": "100",
                    "value_gap": "50",
                    "customer": "A",
                },
            },
            {
                "sheet_name": "Agent_Output",
                "row_json": {
                    "invoice_id": "INV-2",
                    "exception_type": "UNDER_BILLED",
                    "expected_amount": "180",
                    "billed_amount": "120",
                    "value_gap": "60",
                    "customer": "B",
                },
            },
            {
                "sheet_name": "Invoices_Synthetic",
                "row_json": {
                    "invoice_id": "INV-1",
                    "billed_amount": "100",
                    "customer": "A",
                },
            },
            {
                "sheet_name": "Invoices_Synthetic",
                "row_json": {
                    "invoice_id": "INV-2",
                    "billed_amount": "120",
                    "customer": "B",
                },
            },
        ],
    )

    result = plan_query(
        "Show Under Billed Invoices",
        planning_context,
        explicit_sheet=None,
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.target_sheet == "Agent_Output"
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "list"
    assert result.query_spec.dimensions == ["invoice_id", "customer", "exception_type", "expected_amount", "billed_amount", "value_gap"]
    assert result.query_spec.measures == []
    assert result.query_spec.filters == [
        {"column": "exception_type", "operator": "=", "value": "under billed", "kind": "equality"}
    ]
    assert not any("ambiguous" in warning.lower() for warning in result.warnings)

    sql, params = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "regexp_replace(LOWER(COALESCE(s.row_json->>'exception_type', '')), '[^a-z0-9]+', '', 'g') = %(filter_0)s" in sql
    assert params["filter_0"] == "underbilled"
    assert "s.row_json->>'customer' AS customer" in sql
    assert "s.row_json->>'expected_amount' AS expected_amount" in sql
    assert "s.row_json->>'billed_amount' AS billed_amount" in sql


def test_planner_treats_invoice_status_under_billed_as_filtered_invoice_list():
    planning_context = build_planning_context(
        dataset_id="invoice-status-under-billed-test",
        cols=["invoice_id", "invoice_status", "status", "customer", "invoice_date", "expected_amount"],
        col_types={
            "invoice_id": "text",
            "invoice_status": "text",
            "status": "text",
            "customer": "text",
            "invoice_date": "date",
            "expected_amount": "numeric",
        },
        relationships_payload={
            "tables": ["Invoices_Synthetic"],
            "relationships": [],
            "semantic_roles": {
                "Invoices_Synthetic": {
                    "invoice_id": "invoice_id",
                    "invoice_status": "invoice_status",
                    "status": "status",
                    "customer_id": "customer",
                    "invoice_date": "invoice_date",
                    "expected_amount": "expected_amount",
                }
            },
        },
        sheet_rows=[
            {
                "sheet_name": "Invoices_Synthetic",
                "row_json": {
                    "invoice_id": "INV-1",
                    "invoice_status": "SENT_UNDERBILLED",
                    "status": "COMPLETED",
                    "customer": "A",
                    "invoice_date": "2026-01-01",
                    "expected_amount": "100",
                },
            },
            {
                "sheet_name": "Invoices_Synthetic",
                "row_json": {
                    "invoice_id": "INV-2",
                    "invoice_status": "SENT",
                    "status": "COMPLETED",
                    "customer": "B",
                    "invoice_date": "2026-01-02",
                    "expected_amount": "120",
                },
            },
        ],
    )

    result = plan_query(
        "Show Invoice with Invoice Status Under Billed",
        planning_context,
        explicit_sheet=None,
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.target_sheet == "Invoices_Synthetic"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "list"
    assert result.query_spec.dimensions == ["invoice_id", "invoice_date", "customer", "invoice_status", "expected_amount"]
    assert result.query_spec.filters == [
        {"column": "invoice_status", "operator": "contains", "value": "under billed", "kind": "equality"}
    ]

    sql, params = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "regexp_replace(LOWER(COALESCE(s.row_json->>'invoice_status', '')), '[^a-z0-9]+', '', 'g') LIKE %(filter_0)s" in sql
    assert params["filter_0"] == "%underbilled%"
    assert "s.row_json->>'customer' AS customer" in sql
    assert "s.row_json->>'invoice_status' AS invoice_status" in sql
    assert "s.row_json->>'expected_amount' AS expected_amount" in sql


def test_planner_builds_projection_query_for_preaggregated_summary_sheet_columns():
    planning_context = build_planning_context(
        dataset_id="summary-projection-test",
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

    result = plan_query(
        "List Exception Type and Count in Summary Sheet",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "projection"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "projection"
    assert result.query_spec.columns == ["exception_type", "count"]
    assert result.query_spec.dimensions == []
    assert result.query_spec.measures == []
    assert result.query_spec.debug["selected_columns_exist"] is True
    assert result.query_spec.debug["pre_aggregated_sheet"] is True
    assert result.query_spec.debug["requested_projection_columns"] == ["Exception Type", "Count"]
    assert result.query_spec.debug["matched_existing_columns"] == ["exception_type", "count"]
    assert result.query_spec.debug["aggregation_override_skipped_reason"] == "projection_query_selected_existing_columns"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "row_json->>'exception_type' AS exception_type" in sql
    assert "row_json->>'count' AS count" in sql
    assert "GROUP BY" not in sql
    assert "COUNT(" not in sql
    assert "SUM(" not in sql


def test_planner_builds_projection_query_for_multiple_existing_summary_columns():
    planning_context = build_planning_context(
        dataset_id="summary-projection-gap-test",
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

    result = plan_query(
        "Show exception_type and total_value_gap from Summary Sheet",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "projection"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "projection"
    assert result.query_spec.columns == ["exception_type", "total_value_gap"]
    assert result.query_spec.measures == []
    assert result.query_spec.debug["requested_projection_columns"] == ["exception_type", "total_value_gap"]
    assert result.query_spec.debug["matched_existing_columns"] == ["exception_type", "total_value_gap"]
    assert result.query_spec.debug["aggregation_override_skipped_reason"] == "projection_query_selected_existing_columns"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "row_json->>'exception_type' AS exception_type" in sql
    assert "row_json->>'total_value_gap' AS total_value_gap" in sql
    assert "GROUP BY" not in sql


def test_planner_builds_describe_sheet_query_for_describe_summary_sheet():
    planning_context = build_planning_context(
        dataset_id="describe-summary-sheet-test",
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

    result = plan_query("describe summary sheet", planning_context, explicit_sheet="Summary", max_rows=100)

    assert result.degraded_mode is False
    assert result.query_shape == "describe_sheet"
    assert result.answer_type == "sheet_profile"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "describe_sheet"
    assert result.query_spec.target_sheet == "Summary"
    assert result.query_spec.measures == []
    assert result.query_spec.dimensions == []
    assert result.query_spec.debug["pre_aggregated_sheet"] is True


def test_planner_builds_describe_sheet_query_for_summarize_summary_data_sheet():
    planning_context = build_planning_context(
        dataset_id="summarize-summary-sheet-test",
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

    result = plan_query("summarize summary data sheet", planning_context, explicit_sheet="Summary", max_rows=100)

    assert result.degraded_mode is False
    assert result.query_shape == "describe_sheet"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "describe_sheet"
    assert result.query_spec.target_sheet == "Summary"


def test_planner_builds_describe_sheet_query_for_tell_me_about_summary_sheet():
    planning_context = build_planning_context(
        dataset_id="tell-me-about-summary-sheet-test",
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

    result = plan_query("tell me about summary sheet", planning_context, explicit_sheet="Summary", max_rows=100)

    assert result.degraded_mode is False
    assert result.query_shape == "describe_sheet"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "describe_sheet"
    assert result.query_spec.target_sheet == "Summary"


def test_planner_keeps_count_column_as_projection_in_summary_sheet():
    planning_context = build_planning_context(
        dataset_id="summary-projection-count-column-test",
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

    result = plan_query(
        "Show Exception Type and count from Summary Sheet",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "projection"
    assert result.query_spec is not None
    assert result.query_spec.query_type == "projection"
    assert result.query_spec.columns == ["exception_type", "count"]
    assert result.query_spec.dimensions == []
    assert result.query_spec.measures == []
    assert result.query_spec.debug["requested_projection_columns"] == ["Exception Type", "count"]
    assert result.query_spec.debug["matched_existing_columns"] == ["exception_type", "count"]
    assert result.query_spec.debug["aggregation_override_skipped_reason"] == "projection_query_selected_existing_columns"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "row_json->>'exception_type' AS exception_type" in sql
    assert "row_json->>'count' AS count" in sql
    assert "COUNT(" not in sql
    assert "GROUP BY" not in sql


def test_planner_counts_rows_for_count_rows_query_on_summary_sheet():
    planning_context = build_planning_context(
        dataset_id="summary-row-count-test",
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

    result = plan_query(
        "Count rows in Summary Sheet",
        planning_context,
        explicit_sheet="Summary",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "aggregate"
    assert result.query_spec.columns == []
    assert result.query_spec.measures[0]["column"] == "*"
    assert result.query_spec.measures[0]["aggregation"] == "count"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "COUNT(*)" in sql or "COUNT(1)" in sql



def test_planner_ignores_explicit_sheet_name_false_positive_for_status_sheet():
    planning_context = build_planning_context(
        dataset_id="status-sheet-test",
        cols=["invoice_id", "stage_code", "owner_name"],
        col_types={
            "invoice_id": "text",
            "stage_code": "text",
            "owner_name": "text",
        },
        relationships_payload={
            "tables": ["Status"],
            "relationships": [],
            "semantic_roles": {},
        },
        sheet_rows=[
            {"sheet_name": "Status", "row_json": {"invoice_id": "INV-1", "stage_code": "A", "owner_name": "Ann"}},
            {"sheet_name": "Status", "row_json": {"invoice_id": "INV-2", "stage_code": "B", "owner_name": "Ben"}},
        ],
    )

    result = plan_query(
        "List invoice_id from Status",
        planning_context,
        explicit_sheet="Status",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "list_rows"
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["invoice_id"]


def test_planner_builds_scalar_aggregate_with_filters(sample_planning_context):
    result = plan_query(
        "Show Total Sales value for Chair in the East Region",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_scalar"
    assert result.query_spec is not None
    assert result.query_spec.dimensions == []
    assert result.query_spec.measures[0]["column"] == "TotalPrice"
    filters = {(item["column"], item["value"]) for item in result.query_spec.filters}
    assert ("Product", "chair") in filters
    assert ("Region", "east") in filters


def test_planner_builds_grouped_aggregate_with_remaining_dimension(sample_planning_context):
    result = plan_query(
        "Can You give me List of the Sum of Total Price for Products in Region East",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_grouped"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["column"] == "TotalPrice"
    assert result.query_spec.dimensions == ["Product"]
    assert any(item["column"] == "Region" and item["value"] == "east" for item in result.query_spec.filters)


def test_planner_returns_guidance_when_measure_is_ambiguous(sample_planning_context):
    result = plan_query(
        "Show summary",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is True
    assert result.guidance is not None


def test_planner_guidance_uses_exception_suggestions_for_ocean_style_context():
    relationships = {
        "tables": ["Agent_Output"],
        "relationships": [],
        "semantic_roles": {
            "Agent_Output": {
                "event_id": "event_id",
                "customer_id": "customer",
                "exception_type": "exception_type",
                "expected_amount": "expected_amount",
                "billed_amount": "billed_amount",
                "value_gap": "value_gap",
                "billing_lag_days": "billing_lag_days",
                "recommended_action": "recommended_action",
            }
        },
    }
    sheet_rows = [
        {
            "sheet_name": "Agent_Output",
            "row_json": {
                "event_id": "126010003",
                "customer": "EDIINT1I001",
                "exception_type": "UNBILLED",
                "expected_amount": 1251.5,
                "billed_amount": 0,
                "value_gap": 1251.5,
                "billing_lag_days": 0,
                "recommended_action": "Invoice now (create invoice)",
            },
        }
    ]
    context = build_planning_context(
        dataset_id="ocean-test",
        cols=["event_id", "customer", "exception_type", "expected_amount", "billed_amount", "value_gap", "billing_lag_days", "recommended_action"],
        col_types={
            "event_id": "text",
            "customer": "text",
            "exception_type": "text",
            "expected_amount": "numeric",
            "billed_amount": "numeric",
            "value_gap": "numeric",
            "billing_lag_days": "numeric",
            "recommended_action": "text",
        },
        relationships_payload=relationships,
        sheet_rows=sheet_rows,
    )

    guidance = build_planner_guidance(
        "Show summary",
        context,
        target_sheet="Agent_Output",
        warnings=["Ambiguous metric"],
        extracted_filters=[],
    )

    suggestions = guidance["suggested_questions"]
    assert any("recommended action Invoice Now" in suggestion for suggestion in suggestions)
    assert any("exception summary by exception_type" in suggestion for suggestion in suggestions)
    assert all("Region = East" not in suggestion for suggestion in suggestions)


def test_planner_builds_monthly_trend_query(sample_planning_context):
    result = plan_query(
        "Show total sales by month",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "trend_over_time"
    assert result.query_spec is not None
    assert result.query_spec.time_grain == "month"
    assert result.query_spec.dimensions == []
    assert result.query_spec.measures[0]["column"] == "TotalPrice"
    assert result.query_spec.order_by == [{"field": "month", "direction": "asc"}]


def test_planner_builds_comparison_query(sample_planning_context):
    result = plan_query(
        "Compare total sales by region",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "comparison"
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["Region"]
    assert result.query_spec.measures[0]["column"] == "TotalPrice"
    assert result.query_spec.order_by == [{"field": "sum_total_price", "direction": "desc"}]


def test_planner_builds_value_comparison_query(sample_planning_context):
    result = plan_query(
        "Compare total sales East vs West",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "comparison"
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["Region"]
    assert result.query_spec.filters == [{"column": "Region", "operator": "in", "value": ["east", "west"], "kind": "equality"}]


def test_planner_builds_between_value_comparison_query(sample_planning_context):
    result = plan_query(
        "Compare total sales between East and West",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "comparison"
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["Region"]
    assert result.query_spec.filters == [{"column": "Region", "operator": "in", "value": ["east", "west"], "kind": "equality"}]


def test_month_trend_sql_builder_supports_month_time_grain(sample_planning_context):
    result = plan_query(
        "Show total sales by month",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    metadata = {
        "sheets": sample_planning_context.sheets,
        "columns_by_sheet": sample_planning_context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (sample_planning_context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (sample_planning_context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (sample_planning_context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in sample_planning_context.sheets
        },
        "relationships": sample_planning_context.joins,
    }
    sql, _ = build_sql_from_query_spec(result.query_spec, metadata)
    assert "AS month" in sql
    assert "TO_CHAR(" in sql
    assert "ORDER BY month ASC" in sql


def test_planner_builds_quarterly_trend_query(sample_planning_context):
    result = plan_query(
        "Show total sales by quarter",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "trend_over_time"
    assert result.query_spec is not None
    assert result.query_spec.time_grain == "quarter"
    assert result.query_spec.dimensions == []
    assert result.query_spec.order_by == [{"field": "quarter", "direction": "asc"}]


def test_quarter_trend_sql_builder_supports_quarter_time_grain(sample_planning_context):
    result = plan_query(
        "Show total sales by quarter",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    metadata = {
        "sheets": sample_planning_context.sheets,
        "columns_by_sheet": sample_planning_context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (sample_planning_context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (sample_planning_context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (sample_planning_context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in sample_planning_context.sheets
        },
        "relationships": sample_planning_context.joins,
    }
    sql, _ = build_sql_from_query_spec(result.query_spec, metadata)
    assert "AS quarter" in sql
    assert "ORDER BY quarter ASC" in sql


def test_comparison_sql_builder_uses_case_insensitive_in_filter(sample_planning_context):
    result = plan_query(
        "Compare total sales East vs West",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    metadata = {
        "sheets": sample_planning_context.sheets,
        "columns_by_sheet": sample_planning_context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (sample_planning_context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (sample_planning_context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (sample_planning_context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in sample_planning_context.sheets
        },
        "relationships": sample_planning_context.joins,
    }
    sql, params = build_sql_from_query_spec(result.query_spec, metadata)
    assert "LOWER(COALESCE(s.row_json->>'Region', '')) IN (%(filter_0_0)s, %(filter_0_1)s)" in sql
    assert params["filter_0_0"] == "east"
    assert params["filter_0_1"] == "west"


def test_planner_builds_scalar_aggregate_with_region_product_and_year_filters(sample_planning_context):
    result = plan_query(
        "What is the Sum of Total Price on Chair in Region South and Year 2025",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_scalar"
    assert result.query_spec is not None
    assert result.query_spec.time_grain is None
    assert result.query_spec.dimensions == []
    filters = result.query_spec.filters
    assert any(item["column"] == "Product" and item["value"] == "chair" for item in filters)
    assert any(item["column"] == "Region" and item["value"] == "south" for item in filters)
    assert any(item["column"] == "Date" and item["value"] == "2025" and item.get("kind") == "year" for item in filters)
    assert sum(1 for item in filters if item["column"] == "Region") == 1


def test_planner_does_not_confuse_country_with_count_intent():
    planning_context = build_planning_context(
        dataset_id="chocolate-sales-test",
        cols=["Sales Person", "Country", "Product", "Date", "Amount", "Boxes Shipped"],
        col_types={
            "Sales Person": "text",
            "Country": "text",
            "Product": "text",
            "Date": "date",
            "Amount": "numeric",
            "Boxes Shipped": "numeric",
        },
        relationships_payload={
            "tables": ["ChoclateSales"],
            "relationships": [],
            "semantic_roles": {
                "ChoclateSales": {
                    "product": "Product",
                    "country": "Country",
                    "date": "Date",
                    "revenue": "Amount",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "A", "Country": "India", "Product": "Dark Bars", "Date": "2024-01-01", "Amount": "$50.00", "Boxes Shipped": "2"}},
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "B", "Country": "India", "Product": "Mint Chip", "Date": "2024-01-02", "Amount": "$80.00", "Boxes Shipped": "3"}},
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "C", "Country": "UK", "Product": "Dark Bars", "Date": "2024-01-03", "Amount": "$90.00", "Boxes Shipped": "4"}},
        ],
    )

    result = plan_query(
        "which Product has highest Amount in Country India",
        planning_context,
        explicit_sheet="ChoclateSales",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_grouped"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["column"] == "Amount"
    assert result.query_spec.measures[0]["aggregation"] == "max"
    assert result.query_spec.measures[0]["alias"] == "max_amount"
    assert result.query_spec.dimensions == ["Product"]
    assert any(item["column"] == "Country" and item["value"] == "india" for item in result.query_spec.filters)
    assert result.query_spec.order_by == [{"field": "max_amount", "direction": "desc"}]


def test_planner_uses_max_for_grouped_country_wise_highest_query():
    planning_context = build_planning_context(
        dataset_id="chocolate-sales-test",
        cols=["Sales Person", "Country", "Product", "Date", "Amount", "Boxes Shipped"],
        col_types={
            "Sales Person": "text",
            "Country": "text",
            "Product": "text",
            "Date": "date",
            "Amount": "numeric",
            "Boxes Shipped": "numeric",
        },
        relationships_payload={
            "tables": ["ChoclateSales"],
            "relationships": [],
            "semantic_roles": {
                "ChoclateSales": {
                    "product": "Product",
                    "country": "Country",
                    "date": "Date",
                    "revenue": "Amount",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "A", "Country": "India", "Product": "Dark Bars", "Date": "2024-01-01", "Amount": "$50.00", "Boxes Shipped": "2"}},
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "B", "Country": "India", "Product": "Mint Chip", "Date": "2024-01-02", "Amount": "$80.00", "Boxes Shipped": "3"}},
            {"sheet_name": "ChoclateSales", "row_json": {"Sales Person": "C", "Country": "UK", "Product": "Dark Bars", "Date": "2024-01-03", "Amount": "$90.00", "Boxes Shipped": "4"}},
        ],
    )

    result = plan_query(
        "Highest Amount for each product country wise",
        planning_context,
        explicit_sheet="ChoclateSales",
        max_rows=100,
    )

    assert result.degraded_mode is False
    assert result.query_shape == "aggregate_grouped"
    assert result.query_spec is not None
    assert result.query_spec.measures[0]["column"] == "Amount"
    assert result.query_spec.measures[0]["aggregation"] == "max"
    assert set(result.query_spec.dimensions) == {"Product", "Country"}

    metadata = {
        "sheets": planning_context.sheets,
        "columns_by_sheet": planning_context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (planning_context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (planning_context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (planning_context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in planning_context.sheets
        },
        "relationships": planning_context.joins,
    }
    sql, _ = build_sql_from_query_spec(result.query_spec, metadata)
    assert "MAX(" in sql
    assert "SUM(" not in sql
    assert "GROUP BY" in sql
    assert "row_json->>'Product'" in sql
    assert "row_json->>'Country'" in sql


def test_sql_builder_uses_year_filter_without_grouping_for_scalar_query(sample_planning_context):
    result = plan_query(
        "What is the Sum of Total Price on Chair in Region South and Year 2025",
        sample_planning_context,
        explicit_sheet=None,
        max_rows=100,
    )
    metadata = {
        "sheets": sample_planning_context.sheets,
        "columns_by_sheet": sample_planning_context.columns_by_sheet,
        "column_types_by_sheet": {
            sheet: {
                column: (
                    "numeric" if column in (sample_planning_context.numeric_columns_by_sheet.get(sheet) or []) else
                    "date" if column in (sample_planning_context.date_columns_by_sheet.get(sheet) or []) else
                    "text"
                )
                for column in (sample_planning_context.columns_by_sheet.get(sheet) or [])
            }
            for sheet in sample_planning_context.sheets
        },
        "relationships": sample_planning_context.joins,
    }
    sql, params = build_sql_from_query_spec(result.query_spec, metadata)
    assert "EXTRACT(YEAR FROM" in sql
    assert "GROUP BY" not in sql
    assert "LOWER(COALESCE(s.row_json->>'Region', '')) = %(filter_" in sql
    assert params["filter_2"] == 2025 or params["filter_1"] == 2025 or params["filter_0"] == 2025


def test_planner_prefers_formula_backed_revenue_metric_for_top_customers_query():
    planning_context = build_planning_context(
        dataset_id="derived-revenue-test",
        cols=["customer", "quantity", "unit_price"],
        col_types={
            "customer": "text",
            "quantity": "numeric",
            "unit_price": "numeric",
        },
        relationships_payload={
            "tables": ["sales"],
            "relationships": [],
            "semantic_roles": {
                "sales": {
                    "customer_id": "customer",
                    "quantity": "quantity",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "sales", "row_json": {"customer": "Acme", "quantity": 2, "unit_price": 10}},
            {"sheet_name": "sales", "row_json": {"customer": "Globex", "quantity": 5, "unit_price": 7}},
        ],
    )

    assert planning_context.derived_metrics_by_sheet["sales"]["revenue"] == "quantity * unit_price"
    assert planning_context.derived_metrics_by_sheet["sales"]["sales"] == "quantity * unit_price"
    assert planning_context.derived_metrics_by_sheet["sales"]["order_value"] == "quantity * unit_price"
    assert planning_context.derived_metrics_by_sheet["sales"]["avg_price"] == "revenue / quantity"

    result = plan_query(
        "top 10 customers by revenue",
        planning_context,
        explicit_sheet="sales",
        max_rows=10,
    )

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.dimensions == ["customer"]
    assert result.query_spec.measures[0]["column"] == "revenue"
    assert result.query_spec.measures[0]["formula"] == "quantity * unit_price"
    assert result.query_spec.measures[0]["aggregation"] == "sum"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(planning_context))
    assert "SUM((NULLIF(regexp_replace((s.row_json->>'quantity')" in sql
    assert "(s.row_json->>'unit_price')" in sql
    assert "AS sum_revenue" in sql
    assert "GROUP BY s.row_json->>'customer'" in sql
    assert "ORDER BY sum_revenue DESC" in sql