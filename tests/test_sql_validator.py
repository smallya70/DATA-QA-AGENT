from app.planning_context import build_planning_context
from app.query_planner import plan_query
from app.sql_builder import build_sql_from_query_spec
from app.sql_validator import validate_sql


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


def test_validate_sql_allows_cte_ranking_query():
    context = build_planning_context(
        dataset_id="runtime-check",
        cols=["Date", "Country", "Product", "Amount"],
        col_types={"Date": "date", "Country": "text", "Product": "text", "Amount": "numeric"},
        relationships_payload={
            "tables": ["sheet1"],
            "relationships": [],
            "semantic_roles": {
                "sheet1": {
                    "date": "Date",
                    "country": "Country",
                    "product": "Product",
                    "revenue": "Amount",
                }
            },
        },
        sheet_rows=[
            {"sheet_name": "sheet1", "row_json": {"Date": "2024-01-01", "Country": "India", "Product": "Eclairs", "Amount": "120"}},
            {"sheet_name": "sheet1", "row_json": {"Date": "2024-01-02", "Country": "India", "Product": "Mint Chip", "Amount": "80"}},
            {"sheet_name": "sheet1", "row_json": {"Date": "2024-01-03", "Country": "UK", "Product": "Eclairs", "Amount": "210"}},
        ],
    )

    result = plan_query("worst product for each country by lowest amount", context, explicit_sheet=None, max_rows=100)

    assert result.degraded_mode is False
    assert result.query_spec is not None
    assert result.query_spec.query_type == "ranking"

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(context))
    issues = validate_sql(sql, result.query_spec, _metadata_from_context(context))

    assert sql.startswith("WITH aggregated AS (")
    assert issues == []


def test_validate_sql_allows_formula_backed_derived_metric_query():
    context = build_planning_context(
        dataset_id="derived-validator-check",
        cols=["customer", "quantity", "unit_price"],
        col_types={"customer": "text", "quantity": "numeric", "unit_price": "numeric"},
        relationships_payload={
            "tables": ["sales"],
            "relationships": [],
            "semantic_roles": {"sales": {"customer_id": "customer", "quantity": "quantity"}},
        },
        sheet_rows=[
            {"sheet_name": "sales", "row_json": {"customer": "Acme", "quantity": 2, "unit_price": 10}},
            {"sheet_name": "sales", "row_json": {"customer": "Globex", "quantity": 5, "unit_price": 7}},
        ],
    )

    result = plan_query("top 10 customers by revenue", context, explicit_sheet="sales", max_rows=10)

    assert result.degraded_mode is False
    assert result.query_spec is not None

    sql, _ = build_sql_from_query_spec(result.query_spec, _metadata_from_context(context))
    issues = validate_sql(sql, result.query_spec, _metadata_from_context(context))

    assert "quantity" in sql and "unit_price" in sql
    assert issues == []