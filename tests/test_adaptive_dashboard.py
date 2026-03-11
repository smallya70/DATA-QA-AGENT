from __future__ import annotations

import pandas as pd

from app.adaptive_dashboard import (
    DATASET_BILLING,
    DATASET_SALES,
    _dashboard_filter_signature,
    _build_manual_dashboard_chart_spec,
    _enrich_selected_sheet_frame,
    _prepare_chart_dataframe_for_render,
    _prepare_display_frame,
    build_ai_summary,
    build_billing_drilldown_frame,
    build_chart_specs,
    build_joined_frame,
    build_metric_cards,
    build_prompt_drilldown_frame,
    build_prompt_metric_cards,
    build_prompt_chart_spec,
    clean_column_names,
    column_metadata_frame,
    prepare_dataset,
)


def test_clean_column_names_snake_case_and_uniqueness():
    df = pd.DataFrame(columns=["Total Price", "Total-Price", "Customer ID", ""])
    cleaned = clean_column_names(df)
    assert cleaned.columns.tolist() == ["total_price", "total_price_2", "customer_id", "column_4"]


def test_prepare_dataset_detects_sales_profile():
    df = pd.DataFrame(
        {
            "Sales Person": ["Rep A", "Rep B", "Rep A"],
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "Region": ["East", "West", "East"],
            "Product": ["Chair", "Desk", "Chair"],
            "Customer": ["A", "B", "A"],
            "Total Price": [100.0, 250.0, 125.0],
            "Quantity": [1, 2, 1],
        }
    )
    prepared_df, profile = prepare_dataset(df, "sales_sample")
    assert profile.dataset_type == DATASET_SALES
    assert profile.amount_columns == ["total_price"]
    assert profile.date_columns == ["order_date"]
    assert "sales_person" not in profile.amount_columns
    assert "region" in profile.region_columns
    assert "product" in profile.product_columns
    assert "customer" in profile.customer_columns
    assert prepared_df["order_date"].notna().all()


def test_prepare_dataset_detects_billing_profile():
    df = pd.DataFrame(
        {
            "Invoice Number": ["INV-1", "INV-2", "INV-3"],
            "Invoice Date": ["2025-02-01", "2025-02-03", "2025-02-05"],
            "Invoice Status": ["Paid", "Unpaid", "Billed"],
            "Customer": ["A", "B", "C"],
            "Expected Amount": [100.0, 150.0, 175.0],
        }
    )
    _prepared_df, profile = prepare_dataset(df, "billing_sample")
    assert profile.dataset_type == DATASET_BILLING
    assert "invoice_status" in profile.status_columns
    assert "expected_amount" in profile.amount_columns


def test_prepare_dataset_detects_exception_type_as_billing_status():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CFS/CFS", "CY/CY", "CY/CY"],
            "Customer": ["A", "B", "C"],
            "Expected Amount": [100.0, 150.0, 175.0],
        }
    )
    _prepared_df, profile = prepare_dataset(df, "billing_exception_sample")
    assert profile.dataset_type == DATASET_BILLING
    assert "exception_type" in profile.status_columns


def test_prepare_dataset_does_not_classify_service_date_as_product_dimension():
    df = pd.DataFrame(
        {
            "Service Date": ["2026-01-01", "2026-01-02"],
            "Customer": ["A", "B"],
            "Value Gap": [10.0, 20.0],
            "Exception Type": ["UNBILLED", "UNDER_BILLED"],
        }
    )
    _prepared_df, profile = prepare_dataset(df, "service_date_role_sample")
    assert "service_date" in profile.date_columns
    assert "service_date" not in profile.product_columns


def test_dashboard_filter_signature_changes_with_sheet_state_inputs():
    signature_a = _dashboard_filter_signature(
        "dataset-a",
        "Sheet1",
        {"customer": ["A", "B"]},
        {"invoice_date": ("2025-02-01", "2025-02-05")},
    )
    signature_b = _dashboard_filter_signature(
        "dataset-a",
        "Sheet1",
        {"customer": ["A", "B", "C"]},
        {"invoice_date": ("2025-02-01", "2025-02-05")},
    )

    assert signature_a != signature_b


def test_metric_cards_and_summary_use_duckdb_aggregations():
    df = pd.DataFrame(
        {
            "invoice_number": ["INV-1", "INV-2", "INV-3"],
            "invoice_status": ["paid", "unpaid", "billed"],
            "customer": ["A", "B", "C"],
            "expected_amount": [100.0, 150.0, 175.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_summary")
    metrics = build_metric_cards(prepared_df, profile)
    summary = build_ai_summary(prepared_df, profile, metrics, [])
    metric_titles = {metric.title for metric in metrics}
    assert "Total amount" in metric_titles
    assert "Average amount" in metric_titles
    assert "Unpaid amount" in metric_titles
    assert "Unbilled invoices" in metric_titles
    assert "This billing dataset contains 3 records" in summary


def test_build_ai_summary_mentions_active_filters_and_top_category_without_llm():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "Region": ["East", "West", "East"],
            "Product": ["Chair", "Desk", "Chair"],
            "Customer": ["A", "B", "C"],
            "Total Price": [100.0, 250.0, 125.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "sales_dashboard_summary_sample")
    filtered_df = prepared_df[prepared_df["region"] == "East"].copy()
    metrics = build_metric_cards(filtered_df, profile)
    charts = build_chart_specs(filtered_df, profile)

    summary = build_ai_summary(
        filtered_df,
        profile,
        metrics,
        charts,
        active_filter_summary="Region: East",
        prefer_llm=False,
    )

    assert "This sales dataset contains 2 records across 5 columns." in summary
    assert "Key metrics:" in summary
    assert "Top customer is C with $125.00 total price." in summary
    assert "Current filters: Region: East." in summary


def test_metric_cards_distinguish_sheet_records_from_filtered_records():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "Region": ["East", "West", "East"],
            "Product": ["Chair", "Desk", "Chair"],
            "Customer": ["A", "B", "C"],
            "Total Price": [100.0, 250.0, 125.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "sales_metric_count_sample")
    filtered_df = prepared_df[prepared_df["region"] == "East"].copy()

    metrics = build_metric_cards(filtered_df, profile)
    values = {metric.title: metric.value for metric in metrics}

    assert values["Sheet records"] == "3"
    assert values["Filtered records"] == "2"


def test_default_full_date_range_should_not_drop_rows_with_missing_dates():
    df = pd.DataFrame(
        {
            "Invoice Date": ["2025-02-01", "2025-02-03", None, ""],
            "Invoice Status": ["Paid", "Unpaid", "Billed", "Paid"],
            "Customer": ["A", "B", "C", "D"],
            "Expected Amount": [100.0, 150.0, 175.0, 125.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_missing_dates_sample")

    assert profile.row_count == 4
    assert len(prepared_df) == 4

    metrics = build_metric_cards(prepared_df, profile)
    values = {metric.title: metric.value for metric in metrics}

    assert values["Filtered records"] == "4"


def test_prompt_metric_cards_ignore_dashboard_customer_subset_and_use_full_sheet():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "BILLING_DELAY", "UNBILLED"],
            "Customer": ["A", "B", "C", "D"],
            "Expected Amount": [10.0, 20.0, 30.0, 40.0],
            "Invoice ID": ["INV-1", "", "INV-3", ""],
        }
    )
    prepared_df, profile = prepare_dataset(df, "prompt_metric_scope_sample")
    dashboard_subset_df = prepared_df[prepared_df["customer"].isin(["A", "B"])].copy()

    dashboard_values = {metric.title: metric.value for metric in build_metric_cards(dashboard_subset_df, profile)}
    prompt_values = {metric.title: metric.value for metric in build_prompt_metric_cards(prepared_df, profile, "show under billed as pie chart")}

    assert dashboard_values["Filtered records"] == "2"
    assert prompt_values["Filtered records"] == "1"
    assert prompt_values["Sheet records"] == "4"


def test_billing_metrics_count_blank_invoice_rows_and_use_gap_amount_for_unpaid():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNBILLED", "UNBILLED", "UNDER_BILLED"],
            "Invoice ID": ["INV-1", None, "", "INV-4"],
            "Customer": ["A", "B", "C", "D"],
            "Billed Amount": [0.0, 0.0, 0.0, 50.0],
            "Value Gap": [120.0, 1251.5, 2274.32, 20.0],
            "Expected Amount": [120.0, 1251.5, 2274.32, 70.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_gap_sample")
    metrics = build_metric_cards(prepared_df, profile)
    values = {metric.title: metric.value for metric in metrics}
    assert values["Unbilled invoices"] == "3"
    assert values["Under billed records"] == "1"
    assert values["Unpaid amount"] == "$3,645.82"


def test_exception_billing_metrics_prioritize_sheet_totals_over_average_amount():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "BILLING_DELAY"],
            "Event ID": ["126010097", "126010003", "126010004"],
            "Customer": ["DIPPAR1I001", "EDIINT1I001", "BEAINTKY0001"],
            "Service Type": ["CY/CY", "CY/CY", "CY/CY"],
            "Expected Amount": [682.04, 1251.5, 3789.7],
            "Billed Amount": [545.63, 0.0, 3789.7],
            "Value Gap": [136.41, 1251.5, 0.0],
            "Invoice ID": ["INV-6000", "", "INV-6015"],
        }
    )
    prepared_df, profile = prepare_dataset(df, "agent_output_metric_sample")
    metrics = build_metric_cards(prepared_df, profile)
    metric_titles = {metric.title for metric in metrics}
    assert "Average amount" not in metric_titles
    assert "Total Expected Amount" in metric_titles
    assert "Total Billed Amount" in metric_titles
    assert "Total Value Gap" in metric_titles


def test_manual_dashboard_chart_spec_can_build_count_chart():
    df = pd.DataFrame(
        {
            "Invoice Date": ["2025-02-01", "2025-02-01", "2025-02-02", "2025-02-03"],
            "Customer": ["A", "B", "A", "C"],
            "Invoice Status": ["UNBILLED", "UNBILLED", "UNDER_BILLED", "BILLED"],
            "Expected Amount": [100.0, 150.0, 175.0, 120.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "manual_dashboard_count_sample")

    chart = _build_manual_dashboard_chart_spec(
        prepared_df,
        profile,
        x_column="customer",
        y_column="__row_count",
        aggregation="count",
        chart_type="bar",
        top_n=2,
    )

    assert chart is not None
    assert chart.title == "Count by Customer"
    assert chart.y_label == "Count"
    assert chart.dataframe["customer"].tolist() == ["A", "B"]
    assert chart.dataframe["metric_value"].tolist() == [2, 1]


def test_manual_dashboard_chart_spec_can_build_split_sum_chart():
    df = pd.DataFrame(
        {
            "Customer": ["A", "A", "B", "B"],
            "Invoice Status": ["UNBILLED", "UNDER_BILLED", "UNBILLED", "BILLED"],
            "Expected Amount": [100.0, 60.0, 70.0, 30.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "manual_dashboard_split_sample")

    chart = _build_manual_dashboard_chart_spec(
        prepared_df,
        profile,
        x_column="customer",
        y_column="expected_amount",
        aggregation="sum",
        chart_type="stacked_bar",
        split_column="invoice_status",
        top_n=5,
    )

    assert chart is not None
    assert chart.color_column == "invoice_status"
    assert chart.chart_type == "stacked_bar"
    assert chart.y_label == "Expected Amount"
    assert set(chart.dataframe.columns) >= {"customer", "invoice_status", "metric_value"}
    assert float(chart.dataframe[chart.dataframe["customer"] == "A"]["metric_value"].sum()) == 160.0


def test_exception_billing_metrics_infer_relevant_totals_from_nonstandard_column_names():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "BILLING_DELAY"],
            "Event ID": ["1", "2", "3"],
            "Customer": ["A", "B", "C"],
            "Service Type": ["CY/CY", "CY/CY", "CFS/CFS"],
            "Detected Amount": [120.0, 80.0, 60.0],
            "Invoice Amount": [100.0, 0.0, 60.0],
            "Variance": [20.0, 80.0, 0.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "nonstandard_exception_metric_sample")
    metrics = build_metric_cards(prepared_df, profile)
    metric_titles = {metric.title for metric in metrics}
    assert "Average amount" not in metric_titles
    assert "Total Detected Amount" in metric_titles
    assert "Total Invoice Amount" in metric_titles
    assert "Total Variance" in metric_titles


def test_billing_drilldown_prioritizes_agent_output_columns():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNBILLED"],
            "Event ID": ["126010097", "126010003"],
            "Customer": ["DIPPAR1I001", "EDIINT1I001"],
            "Service Type": ["CY/CY", "CY/CY"],
            "Service Date": ["2026-01-08", "2026-01-01"],
            "Expected Amount": [682.04, 1251.5],
            "Billed Amount": [545.63, 0.0],
            "Value Gap": [136.41, 1251.5],
            "Invoice ID": ["INV-6000", ""],
        }
    )
    prepared_df, profile = prepare_dataset(df, "agent_output_drilldown_sample")
    drilldown_df = build_billing_drilldown_frame(prepared_df, profile)
    assert drilldown_df.columns.tolist()[:8] == [
        "exception_type",
        "event_id",
        "customer",
        "service_type",
        "service_date",
        "expected_amount",
        "billed_amount",
        "value_gap",
    ]


def test_chart_specs_and_metadata_frame_generated():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-15", "2025-02-01", "2025-02-10"],
            "Region": ["East", "East", "West", "West"],
            "Status": ["Open", "Closed", "Open", "Closed"],
            "Total Price": [100.0, 200.0, 150.0, 175.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "chart_sample")
    charts = build_chart_specs(prepared_df, profile)
    metadata_df = column_metadata_frame(profile)
    assert any(chart.chart_type == "line" for chart in charts)
    assert any(chart.chart_type in {"bar", "stacked_bar"} for chart in charts)
    assert set(metadata_df.columns.tolist()) == {
        "column_name",
        "detected_type",
        "distinct_count",
        "null_percentage",
        "semantic_role",
    }


def test_billing_chart_specs_support_switching_between_bar_and_pie():
    df = pd.DataFrame(
        {
            "Invoice Number": ["INV-1", "INV-2", "INV-3", "INV-4"],
            "Invoice Date": ["2025-02-01", "2025-02-03", "2025-02-05", "2025-02-10"],
            "Invoice Status": ["Paid", "Unpaid", "Billed", "Overdue"],
            "Customer": ["A", "B", "C", "B"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_chart_sample")
    charts = build_chart_specs(prepared_df, profile)
    assert any(chart.title.lower().startswith("invoice status breakdown") for chart in charts)
    assert any("pie" in chart.supported_chart_types for chart in charts)
    assert any("line" in chart.supported_chart_types for chart in charts if "breakdown" in chart.title.lower())


def test_prompt_chart_uses_total_price_measure_for_sales_prompt():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "Region": ["East", "East", "West", "West"],
            "Product": ["Chair", "Desk", "Chair", "Desk"],
            "Total Price": [100.0, 250.0, 150.0, 175.0],
            "Discount": [5.0, 10.0, 8.0, 12.0],
        }
    )

    prepared_df, profile = prepare_dataset(df, "prompt_sales_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "Show total price by product grouped by region")

    assert chart is not None
    assert chart.x_label == "Product"
    assert chart.y_label == "Total Price"
    assert chart.color_column == "comparison_value"
    assert chart.chart_type == "grouped_bar"
    assert set(chart.dataframe["comparison_value"].tolist()) == {"East", "West"}
    assert chart.dataframe["metric_value"].sum() == 675.0


def test_billing_chart_specs_infer_relevant_measure_groups_from_schema():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "BILLING_DELAY", "UNDER_BILLED"],
            "Customer": ["A", "A", "B", "C"],
            "Service Type": ["CY/CY", "CFS/CFS", "CY/CY", "DOOR/DOOR"],
            "Service Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "Detected Amount": [120.0, 80.0, 200.0, 60.0],
            "Invoice Amount": [100.0, 0.0, 200.0, 50.0],
            "Variance": [20.0, 80.0, 0.0, 10.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "nonstandard_billing_chart_sample")
    charts = build_chart_specs(prepared_df, profile)
    titles = [chart.title for chart in charts]
    assert any(title == "Detected Amount over time" for title in titles)
    assert any(title == "Detected Amount by service type" for title in titles)
    assert any(title == "Variance by service type" for title in titles)


def test_billing_dashboard_adds_aging_and_drilldown_columns():
    anchor_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df = pd.DataFrame(
        {
            "Invoice Number": ["INV-1", "INV-2", "INV-3"],
            "Due Date": [
                (anchor_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                (anchor_date - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                (anchor_date - pd.Timedelta(days=95)).strftime("%Y-%m-%d"),
            ],
            "Invoice Status": ["Unpaid", "Overdue", "Billed"],
            "Customer": ["A", "B", "C"],
            "Expected Amount": [100.0, 150.0, 175.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_aging_sample")
    charts = build_chart_specs(prepared_df, profile)
    drilldown_df = build_billing_drilldown_frame(prepared_df, profile)
    assert any(chart.title == "Aging bucket breakdown" for chart in charts)
    assert "aging_days" in drilldown_df.columns
    assert "aging_bucket" in drilldown_df.columns
    assert set(drilldown_df["aging_bucket"].dropna().astype(str).tolist()) == {"0-30", "31-60", "90+"}


def test_prompt_chart_spec_supports_unbilled_customer_pie_prompt():
    df = pd.DataFrame(
        {
            "Invoice Number": ["INV-1", "INV-2", "INV-3", "INV-4"],
            "Invoice Status": ["Paid", "Unpaid", "Unpaid", "Overdue"],
            "Customer": ["A", "B", "B", "C"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show unbilled invoices by customer as pie chart")
    assert chart is not None
    assert chart.chart_type == "pie"
    assert chart.x_column == "category"
    assert chart.y_column == "metric_value"
    assert set(chart.supported_chart_types) >= {"bar", "pie", "line"}
    assert chart.dataframe["category"].tolist() == ["B", "C"]


def test_prompt_chart_spec_supports_under_billed_exception_prompt():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CFS/CFS", "CY/CY", "CY/CY", "DOOR/DOOR"],
            "Customer": ["A", "B", "B", "C"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_exception_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show under billed by service type as pie chart")
    assert chart is not None
    assert chart.chart_type == "pie"
    assert chart.dataframe["category"].tolist() == ["CY/CY"]
    assert chart.dataframe["metric_value"].tolist() == [2]


def test_prompt_invoice_listing_returns_drilldown_rows_instead_of_count_chart():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Invoice ID": ["", "INV-1", "INV-2", "INV-3"],
            "Service Type": ["CFS/CFS", "CY/CY", "CY/CY", "DOOR/DOOR"],
            "Customer": ["A", "B", "C", "D"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_invoice_listing_prompt_sample")

    chart = build_prompt_chart_spec(prepared_df, profile, "Show Under Billed Invoices")
    drilldown_df = build_prompt_drilldown_frame(prepared_df, profile, "Show Under Billed Invoices")

    assert chart is None
    assert drilldown_df is not None
    assert drilldown_df["invoice_id"].tolist() == ["INV-1", "INV-2"]
    assert set(drilldown_df["exception_type"].tolist()) == {"UNDER_BILLED"}
    assert "service_type" in drilldown_df.columns
    assert "expected_amount" in drilldown_df.columns


def test_prompt_customer_listing_filters_recommended_action_to_matching_customers():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY", "UNBILLED"],
            "Customer": ["A", "A", "A", "D", "E"],
            "Recommended Action": ["Invoice Now", "Invoice Now", "Invoice Now", "Review", "Monitor"],
            "Service Type": ["CY/CY", "CY/CY", "DOOR/DOOR", "CY/CY", "CFS/CFS"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0, 50.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "recommended_action_prompt_sample")

    chart = build_prompt_chart_spec(prepared_df, profile, "Show Customer Recommended action = Invoice Now")
    drilldown_df = build_prompt_drilldown_frame(prepared_df, profile, "Show Customer Recommended action = Invoice Now")
    metrics = build_prompt_metric_cards(prepared_df, profile, "Show Customer Recommended action = Invoice Now")
    metric_values = {metric.title: metric.value for metric in metrics}

    assert chart is None
    assert drilldown_df is not None
    assert drilldown_df["customer"].tolist() == ["A"]
    assert set(drilldown_df["recommended_action"].tolist()) == {"Invoice Now"}
    assert "service_type" in drilldown_df.columns
    assert "expected_amount" in drilldown_df.columns
    assert metric_values["Filtered records"] == "3"


def test_prompt_customer_listing_supports_implicit_invoice_now_phrase():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY", "UNBILLED"],
            "Customer": ["A", "A", "A", "D", "E"],
            "Recommended Action": ["Invoice Now", "Invoice Now", "Invoice Now", "Review", "Monitor"],
            "Service Type": ["CY/CY", "CY/CY", "DOOR/DOOR", "CY/CY", "CFS/CFS"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0, 50.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "recommended_action_prompt_sample_implicit")

    chart = build_prompt_chart_spec(prepared_df, profile, "Show Customer with Invoice Now")
    drilldown_df = build_prompt_drilldown_frame(prepared_df, profile, "Show Customer with Invoice Now")
    metrics = build_prompt_metric_cards(prepared_df, profile, "Show Customer with Invoice Now")
    metric_values = {metric.title: metric.value for metric in metrics}

    assert chart is None
    assert drilldown_df is not None
    assert drilldown_df["customer"].tolist() == ["A"]
    assert set(drilldown_df["recommended_action"].tolist()) == {"Invoice Now"}
    assert metric_values["Filtered records"] == "3"


def test_prompt_chart_spec_uses_value_gap_when_explicitly_requested():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CY/CY", "DOOR/DOOR", "CY/CY"],
            "Expected Amount": [702.72, 150.0, 999.0],
            "Value Gap": [140.54, 20.0, 0.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_value_gap_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show under billed vs value gap")
    assert chart is not None
    assert chart.chart_type == "bar"
    assert chart.color_column is None
    assert chart.y_label == "Value Gap"
    assert chart.x_label == "Service Type"
    assert chart.dataframe["metric_value"].tolist() == [140.54, 20.0]


def test_prompt_chart_spec_prefers_explicit_invoice_date_dimension():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNBILLED", "UNDER_BILLED"],
            "Service Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "Invoice Date": ["2026-01-03", "2026-01-03", "2026-01-04"],
            "Service Type": ["CY/CY", "DOOR/DOOR", "CY/CY"],
            "Value Gap": [100.0, 50.0, 25.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_invoice_date_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show unbilled value gap by invoice date")
    assert chart is not None
    assert chart.x_label == "Invoice Date"
    assert chart.y_label == "Value Gap"
    assert chart.dataframe["metric_value"].tolist() == [150.0]


def test_prompt_parser_does_not_treat_unbilled_prompt_as_billed_comparison():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CY/CY", "DOOR/DOOR", "CY/CY"],
            "Value Gap": [100.0, 50.0, 10.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_focus_parse_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "display event type = unbilled vs value gap")
    assert chart is not None
    assert chart.chart_type == "bar"
    assert chart.color_column is None
    assert chart.x_label == "Service Type"
    assert chart.y_label == "Value Gap"


def test_prompt_chart_spec_supports_event_type_alias_as_dimension():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CY/CY", "DOOR/DOOR", "CFS/CFS"],
            "Value Gap": [100.0, 50.0, 10.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_event_type_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show dashboard event_type vs value gap")
    assert chart is not None
    assert chart.chart_type == "bar"
    assert chart.x_label == "Exception Type"
    assert chart.y_label == "Value Gap"
    assert chart.dataframe["category"].tolist() == ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"]


def test_prompt_chart_spec_prefers_business_dimension_over_date_fallback():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED"] * 15,
            "Customer": [f"C{i:02d}" for i in range(15)],
            "Service Date": pd.date_range("2026-01-01", periods=15).astype(str),
            "Value Gap": [float(i + 1) for i in range(15)],
        }
    )
    prepared_df, profile = prepare_dataset(df, "prompt_business_dimension_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "unbilled vs value gap")
    assert chart is not None
    assert chart.x_label == "Customer"
    assert chart.y_label == "Value Gap"
    assert chart.x_column == "category"


def test_prompt_chart_spec_supports_measure_comparison_by_customer():
    df = pd.DataFrame(
        {
            "Customer": ["A", "A", "B", "B"],
            "Expected Amount": [100.0, 50.0, 80.0, 20.0],
            "Value Gap": [10.0, 5.0, 8.0, 2.0],
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_compare_measure_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "compare expected amount vs value gap by customer")
    assert chart is not None
    assert chart.chart_type == "grouped_bar"
    assert chart.color_column == "comparison_value"
    assert set(chart.dataframe["comparison_value"].tolist()) == {"Expected Amount", "Value Gap"}
    assert chart.x_label == "Customer"


def test_prompt_chart_spec_supports_focus_comparison_by_customer():
    df = pd.DataFrame(
        {
            "Customer": ["A", "A", "B", "B", "C"],
            "Exception Type": ["UNDER_BILLED", "UNBILLED", "UNDER_BILLED", "UNBILLED", "BILLING_DELAY"],
            "Expected Amount": [100.0, 90.0, 80.0, 70.0, 60.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_compare_focus_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show under billed and unbilled side by side by customer")
    assert chart is not None
    assert chart.chart_type == "grouped_bar"
    assert chart.color_column == "comparison_value"
    assert set(chart.dataframe["comparison_value"].tolist()) == {"Under Billed", "Unbilled"}
    assert chart.y_label == "Count"


def test_prompt_chart_spec_supports_exception_type_dimension_prompt():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CFS/CFS", "CY/CY", "CY/CY", "DOOR/DOOR"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_exception_dimension_prompt_sample")
    chart = build_prompt_chart_spec(prepared_df, profile, "show exception type as pie chart")
    assert chart is not None
    assert chart.chart_type == "pie"
    assert set(chart.dataframe["category"].tolist()) == {"UNBILLED", "UNDER_BILLED", "BILLING_DELAY"}


def test_billing_chart_specs_prioritize_meaningful_dimension_over_identifier_columns():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "UNDER_BILLED", "BILLING_DELAY", "UNBILLED"],
            "Event ID": ["E1", "E2", "E3", "E4", "E5"],
            "Invoice ID": ["INV1", "INV2", "INV3", "INV4", "INV5"],
            "Service Type": ["CFS/CFS", "CY/CY", "CY/CY", "DOOR/DOOR", "CFS/CFS"],
            "Expected Amount": [100.0, 150.0, 175.0, 90.0, 110.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "billing_chart_dimension_sample")
    charts = build_chart_specs(prepared_df, profile)
    titles = [chart.title.lower() for chart in charts]
    assert any("exception type breakdown" in title for title in titles)
    assert any("service type" in title for title in titles)
    assert not any("invoice id" in title or "event id" in title for title in titles)


def test_prepare_display_frame_preserves_alphanumeric_identifier_text():
    df = pd.DataFrame(
        {
            "Invoice ID": ["INV-6000", "INV-6015", None],
            "Customer": ["DIPPAR11001", "BEAINTKY0001", "KYJHB"],
            "Exception Type": ["UNDER_BILLED", "BILLING_DELAY", "UNBILLED"],
            "Expected Amount": [682.04, 3789.7, 165.01],
        }
    )
    prepared_df, profile = prepare_dataset(df, "display_text_sample")
    display_df = _prepare_display_frame(prepared_df, profile)
    assert display_df["invoice_id"].tolist() == ["INV-6000", "INV-6015", ""]
    assert display_df["customer"].tolist() == ["DIPPAR11001", "BEAINTKY0001", "KYJHB"]


def test_prepare_dataset_keeps_identifier_like_numeric_text_as_categorical():
    df = pd.DataFrame(
        {
            "Customer": ["1001", "27001", "45001"],
            "Invoice ID": ["-6000", "-6022", ""],
            "Event ID": ["126010097", "3126010001", "126010041"],
            "Expected Amount": [682.04, 20243.48, 2274.32],
        }
    )
    prepared_df, profile = prepare_dataset(df, "identifier_preservation_sample")
    assert "customer" in profile.categorical_columns
    assert "invoice_id" in profile.categorical_columns
    assert prepared_df["customer"].tolist() == ["1001", "27001", "45001"]
    assert prepared_df["invoice_id"].tolist() == ["-6000", "-6022", ""]


def test_enrich_selected_sheet_frame_recovers_customer_text_and_invoice_prefix():
    frames = {
        "Agent_Output": pd.DataFrame(
            {
                "event_id": ["126010097", "126010041"],
                "customer": ["1001", "27001"],
                "invoice_id": ["-6000", ""],
            }
        ),
        "Ops_Events": pd.DataFrame(
            {
                "event_id": ["126010097", "126010041"],
                "customer": ["DIPPAR1I001", "WESDIS27001"],
            }
        ),
    }
    enriched = _enrich_selected_sheet_frame(frames["Agent_Output"], "Agent_Output", frames)
    assert enriched["customer"].tolist() == ["DIPPAR1I001", "WESDIS27001"]
    assert enriched["invoice_id"].tolist() == ["INV-6000", ""]


def test_trend_chart_uses_source_date_label_not_period():
    df = pd.DataFrame(
        {
            "Service Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "Expected Amount": [100.0, 150.0, 125.0],
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
        }
    )
    prepared_df, profile = prepare_dataset(df, "trend_label_sample")
    charts = build_chart_specs(prepared_df, profile)
    trend_chart = next(chart for chart in charts if chart.chart_type == "line")
    assert trend_chart.x_label == "Service Date"
    assert set(trend_chart.supported_chart_types) >= {"line", "area", "bar"}


def test_temporal_bar_render_frame_uses_readable_date_labels():
    from app.adaptive_dashboard import ChartSpec

    df = pd.DataFrame(
        {
            "period": pd.to_datetime(["2026-01-03", "2026-01-06", "2026-01-08"]),
            "metric_value": [10.0, 20.0, 30.0],
        }
    )
    chart = ChartSpec(
        title="Prompt view",
        chart_type="bar",
        dataframe=df,
        x_column="period",
        y_column="metric_value",
        x_label="Service Date",
        y_label="Value Gap",
    )
    render_df, render_x_column, x_is_temporal = _prepare_chart_dataframe_for_render(chart, "bar")
    assert x_is_temporal is False
    assert render_x_column == "period_label"
    assert render_df[render_x_column].tolist() == ["2026-01-03", "2026-01-06", "2026-01-08"]


def test_chart_specs_use_human_friendly_metric_labels():
    df = pd.DataFrame(
        {
            "Exception Type": ["UNBILLED", "UNDER_BILLED", "BILLING_DELAY"],
            "Service Type": ["CFS/CFS", "CY/CY", "CFS/CFS"],
            "Expected Amount": [120.0, 250.0, 300.0],
        }
    )
    prepared_df, profile = prepare_dataset(df, "chart_label_sample")
    charts = build_chart_specs(prepared_df, profile)
    assert any(chart.y_label in {"Count", "Expected Amount"} for chart in charts)


def test_build_joined_frame_merges_detected_relationship_sheets():
    sheet_frames = {
        "Orders": pd.DataFrame(
            {
                "Customer ID": ["C-1", "C-2", "C-3"],
                "Order Amount": [100.0, 150.0, 200.0],
            }
        ),
        "Customers": pd.DataFrame(
            {
                "Customer ID": ["c-1", "c-2", "c-4"],
                "Region": ["East", "West", "North"],
            }
        ),
    }
    joined_df = build_joined_frame(
        sheet_frames,
        {"left": "Orders", "left_col": "Customer ID", "right": "Customers", "right_col": "Customer ID"},
        join_how="left",
    )
    assert "order_amount" in joined_df.columns
    assert any(column.startswith("region") for column in joined_df.columns)
    assert joined_df.shape[0] == 3