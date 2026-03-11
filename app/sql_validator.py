import re
from typing import Any

from app.query_spec import QuerySpec


WRITE_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"}
NUMERIC_HINTS = {
    "amount",
    "revenue",
    "profit",
    "sales",
    "total",
    "price",
    "cost",
    "qty",
    "quantity",
    "tax",
    "discount",
    "value",
    "gap",
}


def _normalize_name(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or "").strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")


def _columns_by_sheet(dataset_metadata: dict[str, Any]) -> dict[str, list[str]]:
    columns = dataset_metadata.get("columns_by_sheet") or dataset_metadata.get("columns") or {}
    return {str(sheet): [str(col) for col in cols] for sheet, cols in columns.items()}


def _column_types_by_sheet(dataset_metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    value = dataset_metadata.get("column_types_by_sheet") or {}
    return {
        str(sheet): {str(col): str(col_type) for col, col_type in cols.items()}
        for sheet, cols in value.items()
    }


def _derived_metrics_by_sheet(dataset_metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    explicit_value = dataset_metadata.get("derived_metrics_by_sheet") or {}
    if isinstance(explicit_value, dict) and explicit_value:
        return {
            str(sheet): {
                str(metric_name): str(formula)
                for metric_name, formula in (sheet_metrics or {}).items()
                if str(metric_name).strip() and str(formula).strip()
            }
            for sheet, sheet_metrics in explicit_value.items()
            if isinstance(sheet_metrics, dict)
        }

    flattened = dataset_metadata.get("derived_metrics") or {}
    sheets = [str(sheet) for sheet in (dataset_metadata.get("sheets") or []) if str(sheet).strip()]
    if len(sheets) == 1 and isinstance(flattened, dict):
        return {
            sheets[0]: {
                str(metric_name): str(formula)
                for metric_name, formula in flattened.items()
                if str(metric_name).strip() and str(formula).strip()
            }
        }
    return {}


def _sheet_columns(sheet: str, dataset_metadata: dict[str, Any]) -> set[str]:
    return set(_columns_by_sheet(dataset_metadata).get(sheet) or [])


def _reachable_sheets(query_spec: QuerySpec) -> set[str]:
    sheets: set[str] = set()
    if query_spec.target_sheet:
        sheets.add(query_spec.target_sheet)
    for join_step in query_spec.join_path or []:
        if not isinstance(join_step, dict):
            continue
        if join_step.get("left_sheet"):
            sheets.add(str(join_step.get("left_sheet")))
        if join_step.get("right_sheet"):
            sheets.add(str(join_step.get("right_sheet")))
    return sheets


def _column_is_numeric(sheet: str, column_name: str, dataset_metadata: dict[str, Any]) -> bool:
    types = _column_types_by_sheet(dataset_metadata).get(sheet) or {}
    if str(types.get(column_name) or "").lower() == "numeric":
        return True
    return any(token in _normalize_name(column_name) for token in NUMERIC_HINTS)


def _parse_formula(formula: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"\s*(.+?)\s*([+\-*/])\s*(.+?)\s*", str(formula or ""))
    if not match:
        return None
    return match.group(1).strip(), match.group(2), match.group(3).strip()


def _formula_reference_is_numeric(
    sheet: str,
    reference_name: str,
    dataset_metadata: dict[str, Any],
    *,
    visited: set[str] | None = None,
) -> bool:
    visited = visited or set()
    normalized_reference = _normalize_name(reference_name)

    for column_name in _sheet_columns(sheet, dataset_metadata):
        if _normalize_name(column_name) == normalized_reference:
            return _column_is_numeric(sheet, column_name, dataset_metadata)

    derived_metrics = _derived_metrics_by_sheet(dataset_metadata).get(sheet) or {}
    for metric_name, formula in derived_metrics.items():
        if _normalize_name(metric_name) != normalized_reference or metric_name in visited:
            continue
        parsed_formula = _parse_formula(formula)
        if parsed_formula is None:
            return False
        left_operand, _operator, right_operand = parsed_formula
        next_visited = set(visited)
        next_visited.add(metric_name)
        return _formula_reference_is_numeric(sheet, left_operand, dataset_metadata, visited=next_visited) and _formula_reference_is_numeric(sheet, right_operand, dataset_metadata, visited=next_visited)

    return False


def _extract_alias_sheet_map(sql: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+dataset_rows\s+([A-Za-z_][A-Za-z0-9_]*)", sql or "", flags=re.IGNORECASE):
        alias = match.group(1)
        sheet_match = re.search(
            rf"\b{re.escape(alias)}\.sheet_name\s*=\s*'([^']+)'",
            sql or "",
            flags=re.IGNORECASE,
        )
        if sheet_match:
            alias_map[alias] = sheet_match.group(1)
    return alias_map


def _relationship_exists(left_sheet: str, left_col: str, right_sheet: str, right_col: str, dataset_metadata: dict[str, Any]) -> bool:
    for relationship in dataset_metadata.get("relationships") or []:
        if not isinstance(relationship, dict):
            continue
        direct = (
            str(relationship.get("left") or "") == left_sheet
            and str(relationship.get("left_col") or "") == left_col
            and str(relationship.get("right") or "") == right_sheet
            and str(relationship.get("right_col") or "") == right_col
        )
        reverse = (
            str(relationship.get("left") or "") == right_sheet
            and str(relationship.get("left_col") or "") == right_col
            and str(relationship.get("right") or "") == left_sheet
            and str(relationship.get("right_col") or "") == left_col
        )
        if direct or reverse:
            return True
    return False


def validate_sql(sql: str, query_spec: QuerySpec, dataset_metadata: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    text = str(sql or "").strip()
    lower_sql = text.lower()
    sheets = {str(sheet) for sheet in (dataset_metadata.get("sheets") or [])}

    if not (lower_sql.startswith("select") or lower_sql.startswith("with ") or lower_sql.startswith("with\n")):
        issues.append("SQL must be SELECT-only.")
    if ";" in text:
        issues.append("SQL must not contain semicolons.")
    if any(keyword in lower_sql for keyword in WRITE_KEYWORDS):
        issues.append("SQL contains non-SELECT operations.")
    if "dataset_id" not in lower_sql:
        issues.append("SQL must filter by dataset_id.")
    if not re.search(r"\blimit\b", lower_sql):
        issues.append("SQL must have a LIMIT.")

    if query_spec.target_sheet and query_spec.target_sheet not in sheets:
        issues.append(f"Sheet '{query_spec.target_sheet}' does not exist.")

    alias_map = _extract_alias_sheet_map(text)
    for alias, sheet in alias_map.items():
        if sheet not in sheets:
            issues.append(f"Referenced sheet '{sheet}' for alias '{alias}' does not exist.")

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'", text, flags=re.IGNORECASE):
        alias = match.group(1)
        column_name = match.group(2)
        sheet = alias_map.get(alias)
        if not sheet:
            continue
        if column_name not in _sheet_columns(sheet, dataset_metadata):
            issues.append(f"Column '{column_name}' does not exist in sheet '{sheet}'.")

    for join_step in query_spec.join_path or []:
        if not isinstance(join_step, dict):
            issues.append("Each join_path item must be an object.")
            continue
        left_sheet = str(join_step.get("left_sheet") or "")
        left_col = str(join_step.get("left_col") or "")
        right_sheet = str(join_step.get("right_sheet") or "")
        right_col = str(join_step.get("right_col") or "")
        if not _relationship_exists(left_sheet, left_col, right_sheet, right_col, dataset_metadata):
            issues.append(
                f"Join path {left_sheet}.{left_col} -> {right_sheet}.{right_col} is not a known relationship."
            )

    reachable_sheets = _reachable_sheets(query_spec) or ({query_spec.target_sheet} if query_spec.target_sheet else set())
    for dimension in query_spec.dimensions:
        if not any(dimension in _sheet_columns(sheet, dataset_metadata) for sheet in reachable_sheets if sheet):
            target_sheet = query_spec.target_sheet or next(iter(sheets), "")
            issues.append(f"Column '{dimension}' does not exist in sheet '{target_sheet}'.")

    for measure in query_spec.measures:
        if not isinstance(measure, dict):
            issues.append("Each measure must be an object.")
            continue
        sheet = str(measure.get("sheet") or query_spec.target_sheet or next(iter(sheets), ""))
        column_name = str(measure.get("column") or "*")
        formula = str(measure.get("formula") or "").strip() or None
        aggregation = str(measure.get("aggregation") or measure.get("agg") or "count").lower()
        if formula:
            parsed_formula = _parse_formula(formula)
            if parsed_formula is None:
                issues.append(f"Measure formula '{formula}' is not a supported binary expression.")
                continue
            left_operand, _operator, right_operand = parsed_formula
            for operand in [left_operand, right_operand]:
                if not _formula_reference_is_numeric(sheet, operand, dataset_metadata):
                    issues.append(f"Measure formula operand '{operand}' is not numeric in sheet '{sheet}'.")
        elif column_name != "*" and column_name not in _sheet_columns(sheet, dataset_metadata):
            issues.append(f"Measure column '{column_name}' does not exist in sheet '{sheet}'.")
        if aggregation in {"sum", "avg", "min", "max"} and column_name != "*" and not formula and not _column_is_numeric(sheet, column_name, dataset_metadata):
            issues.append(
                f"Aggregation '{aggregation}' is not valid for non-numeric column '{column_name}' in sheet '{sheet}'."
            )

    return issues
