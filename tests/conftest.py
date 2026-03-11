from __future__ import annotations

import pytest

from app.planning_context import build_planning_context


@pytest.fixture()
def sample_planning_context():
    relationships = {
        "tables": ["Sheet1"],
        "relationships": [],
        "semantic_roles": {
            "Sheet1": {
                "date": "Date",
                "region": "Region",
                "product": "Product",
                "quantity": "Quantity",
            }
        },
    }
    sheet_rows = [
        {
            "sheet_name": "Sheet1",
            "row_json": {
                "Date": "2024-01-01",
                "Region": "East",
                "Product": "Chair",
                "Quantity": 2,
                "UnitPrice": 12.5,
                "TotalPrice": 25.0,
                "CustomerType": "Retail",
            },
        },
        {
            "sheet_name": "Sheet1",
            "row_json": {
                "Date": "2024-01-02",
                "Region": "West",
                "Product": "Table",
                "Quantity": 5,
                "UnitPrice": 30.0,
                "TotalPrice": 150.0,
                "CustomerType": "Wholesale",
            },
        },
        {
            "sheet_name": "Sheet1",
            "row_json": {
                "Date": "2024-02-01",
                "Region": "East",
                "Product": "Desk",
                "Quantity": 1,
                "UnitPrice": 80.0,
                "TotalPrice": 80.0,
                "CustomerType": "Retail",
            },
        },
    ]
    return build_planning_context(
        dataset_id="test-dataset",
        cols=["Date", "Region", "Product", "Quantity", "UnitPrice", "TotalPrice", "CustomerType"],
        col_types={
            "Date": "date",
            "Region": "text",
            "Product": "text",
            "Quantity": "numeric",
            "UnitPrice": "numeric",
            "TotalPrice": "numeric",
            "CustomerType": "text",
        },
        relationships_payload=relationships,
        sheet_rows=sheet_rows,
    )


@pytest.fixture()
def chocolate_planning_context():
    relationships = {
        "tables": ["ChoclateSales"],
        "relationships": [],
        "semantic_roles": {
            "ChoclateSales": {
                "date": "Date",
                "country": "Country",
                "region": "Region",
                "product": "Product",
                "customer_id": "Sales Person",
                "revenue": "Amount",
                "quantity": "Boxes Shipped",
                "status": "Status",
            }
        },
    }
    sheet_rows = [
        {"sheet_name": "ChoclateSales", "row_json": {"Date": "2024-01-01", "Country": "India", "Region": "South Asia", "Product": "Eclairs", "Sales Person": "Asha", "Amount": "$120.00", "Boxes Shipped": 10, "Status": "Closed"}},
        {"sheet_name": "ChoclateSales", "row_json": {"Date": "2024-01-02", "Country": "India", "Region": "South Asia", "Product": "Mint Chip", "Sales Person": "Vikram", "Amount": "$80.00", "Boxes Shipped": 7, "Status": "Open"}},
        {"sheet_name": "ChoclateSales", "row_json": {"Date": "2024-01-03", "Country": "UK", "Region": "Europe", "Product": "Eclairs", "Sales Person": "John", "Amount": "$210.00", "Boxes Shipped": 12, "Status": "Closed"}},
    ]
    return build_planning_context(
        dataset_id="chocolate-sales",
        cols=["Date", "Country", "Region", "Product", "Sales Person", "Amount", "Boxes Shipped", "Status"],
        col_types={
            "Date": "date",
            "Country": "text",
            "Region": "text",
            "Product": "text",
            "Sales Person": "text",
            "Amount": "numeric",
            "Boxes Shipped": "numeric",
            "Status": "text",
        },
        relationships_payload=relationships,
        sheet_rows=sheet_rows,
    )


@pytest.fixture()
def multi_sheet_business_context():
    relationships = {
        "tables": ["Sales", "Support"],
        "relationships": [],
        "semantic_roles": {
            "Sales": {
                "date": "Order Date",
                "country": "Country",
                "region": "Region",
                "product": "Product",
                "customer_id": "Customer",
                "revenue": "Amount",
                "quantity": "Units",
                "status": "Order Status",
            },
            "Support": {
                "date": "Created Date",
                "country": "Market",
                "region": "Territory",
                "customer_id": "Client",
                "status": "Stage",
                "quantity": "Ticket Count",
            },
        },
    }
    sheet_rows = [
        {"sheet_name": "Sales", "row_json": {"Order Date": "2024-02-01", "Country": "India", "Region": "West", "Product": "Chair", "Customer": "Acme", "Amount": "1000", "Units": "5", "Order Status": "Closed"}},
        {"sheet_name": "Sales", "row_json": {"Order Date": "2024-02-02", "Country": "USA", "Region": "East", "Product": "Desk", "Customer": "Globex", "Amount": "500", "Units": "2", "Order Status": "Open"}},
        {"sheet_name": "Support", "row_json": {"Created Date": "2024-02-01", "Market": "India", "Territory": "West", "Client": "Acme", "Stage": "Open", "Ticket Count": "4"}},
        {"sheet_name": "Support", "row_json": {"Created Date": "2024-02-03", "Market": "USA", "Territory": "East", "Client": "Globex", "Stage": "Closed", "Ticket Count": "1"}},
    ]
    return build_planning_context(
        dataset_id="multi-sheet-business",
        cols=["Order Date", "Country", "Region", "Product", "Customer", "Amount", "Units", "Order Status", "Created Date", "Market", "Territory", "Client", "Stage", "Ticket Count"],
        col_types={
            "Order Date": "date",
            "Country": "text",
            "Region": "text",
            "Product": "text",
            "Customer": "text",
            "Amount": "numeric",
            "Units": "numeric",
            "Order Status": "text",
            "Created Date": "date",
            "Market": "text",
            "Territory": "text",
            "Client": "text",
            "Stage": "text",
            "Ticket Count": "numeric",
        },
        relationships_payload=relationships,
        sheet_rows=sheet_rows,
    )