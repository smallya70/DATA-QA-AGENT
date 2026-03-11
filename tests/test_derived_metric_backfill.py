from __future__ import annotations

from main_upgraded import _backfill_derived_metrics


def test_backfill_derived_metrics_persists_inferred_formulas(monkeypatch):
    updates: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "main_upgraded._load_datasets_for_derived_metric_backfill",
        lambda dataset_id=None: [
            {
                "id": "dataset-1",
                "name": "Legacy sales",
                "columns": ["Customer", "Quantity", "UnitPrice"],
                "column_types": {"Customer": "text", "Quantity": "numeric", "UnitPrice": "numeric"},
                "relationships": {"tables": ["Sales"], "relationships": []},
            }
        ],
    )
    monkeypatch.setattr(
        "main_upgraded._load_dataset_sheet_columns",
        lambda dataset_id: {"Sales": {"Customer", "Quantity", "UnitPrice"}},
    )
    monkeypatch.setattr(
        "main_upgraded._update_dataset_relationships_payload",
        lambda dataset_id, relationships_payload: updates.append((dataset_id, relationships_payload)),
    )

    results = _backfill_derived_metrics()

    assert results == [
        {
            "dataset_id": "dataset-1",
            "name": "Legacy sales",
            "updated": True,
            "sheet_count": 1,
            "metric_count": 4,
        }
    ]
    assert len(updates) == 1
    dataset_id, relationships_payload = updates[0]
    assert dataset_id == "dataset-1"
    assert relationships_payload["derived_metrics_by_sheet"] == {
        "Sales": {
            "revenue": "quantity * unit_price",
            "sales": "quantity * unit_price",
            "order_value": "quantity * unit_price",
            "avg_price": "revenue / quantity",
        }
    }
    assert relationships_payload["derived_metrics"] == {
        "revenue": "quantity * unit_price",
        "sales": "quantity * unit_price",
        "order_value": "quantity * unit_price",
        "avg_price": "revenue / quantity",
    }


def test_backfill_derived_metrics_skips_when_payload_is_already_current(monkeypatch):
    monkeypatch.setattr(
        "main_upgraded._load_datasets_for_derived_metric_backfill",
        lambda dataset_id=None: [
            {
                "id": "dataset-1",
                "name": "Legacy sales",
                "columns": ["Customer", "Quantity", "UnitPrice"],
                "column_types": {"Customer": "text", "Quantity": "numeric", "UnitPrice": "numeric"},
                "relationships": {
                    "tables": ["Sales"],
                    "relationships": [],
                    "derived_metrics": {
                        "revenue": "quantity * unit_price",
                        "sales": "quantity * unit_price",
                        "order_value": "quantity * unit_price",
                        "avg_price": "revenue / quantity",
                    },
                    "derived_metrics_by_sheet": {
                        "Sales": {
                            "revenue": "quantity * unit_price",
                            "sales": "quantity * unit_price",
                            "order_value": "quantity * unit_price",
                            "avg_price": "revenue / quantity",
                        }
                    },
                },
            }
        ],
    )
    monkeypatch.setattr(
        "main_upgraded._load_dataset_sheet_columns",
        lambda dataset_id: {"Sales": {"Customer", "Quantity", "UnitPrice"}},
    )

    def _fail_update(dataset_id, relationships_payload):
        raise AssertionError("backfill should not update an unchanged dataset")

    monkeypatch.setattr("main_upgraded._update_dataset_relationships_payload", _fail_update)

    results = _backfill_derived_metrics()

    assert results == [
        {
            "dataset_id": "dataset-1",
            "name": "Legacy sales",
            "updated": False,
            "sheet_count": 1,
            "metric_count": 4,
        }
    ]