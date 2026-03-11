from __future__ import annotations


def session_case(mode: str, question: str, **expected) -> dict:
    return {
        "mode": mode,
        "question": question,
        **expected,
    }


TODAY_FAILED_QUESTION_SET = [
    session_case(
        "planner",
        "Show Total Sales value for Chair in the East Region",
        context_fixture="sample_planning_context",
        expected_shape="aggregate_scalar",
        expected_intent="total_x_for_y",
        expected_sheet="Sheet1",
        expected_metric="TotalPrice",
        expected_filters=[("Product", "chair"), ("Region", "east")],
    ),
    session_case(
        "planner",
        "Can You give me List of the Sum of Total Price for Products in Region East",
        context_fixture="sample_planning_context",
        expected_shape="aggregate_grouped",
        expected_intent="x_by_y",
        expected_sheet="Sheet1",
        expected_metric="TotalPrice",
        expected_dimension="Product",
        expected_filters=[("Region", "east")],
    ),
    session_case(
        "planner",
        "which Product has highest Amount in Country India",
        context_fixture="chocolate_planning_context",
        expected_shape="aggregate_grouped",
        expected_intent="highest_x",
        expected_sheet="ChoclateSales",
        expected_metric="Amount",
        expected_dimension="Product",
        expected_filters=[("Country", "india")],
    ),
    session_case(
        "helper_recommended_action_filter",
        "SHOW Exception type and Customer Id when Recommended action=Invoice Now",
        expected_value="invoice now",
    ),
    session_case(
        "helper_recommended_action_filter",
        "Show Customer with Invoice Now",
        expected_value="invoice now",
    ),
    session_case(
        "helper_exception_detail_sql",
        "SHOW Exception type and Customer Id when Recommended action=Invoice Now",
        available_sheets=["Agent_Output"],
        sheet_columns_by_sheet={
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
        },
        forced_sheet="Agent_Output",
        expected_sql_contains=["recommended_action", "%invoice now%"],
    ),
    session_case(
        "helper_exception_detail_sql",
        "Show Customer with Invoice Now",
        available_sheets=["Agent_Output"],
        sheet_columns_by_sheet={
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
        },
        forced_sheet="Agent_Output",
        expected_sql_contains=["recommended_action", "%invoice now%"],
    ),
    session_case(
        "helper_customer_value_gap_question",
        "Show total value gap by customer",
        expected_value=True,
    ),
    session_case(
        "helper_exception_customer_list_question",
        "can You give list of Unbilled Customers",
        expected_value=True,
    ),
    session_case(
        "helper_exception_detail_question",
        "Can You list of Unbilled customers and expected amount",
        expected_value=True,
    ),
]