import re
from typing import Any

from app.query_spec import QuerySpec


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
NUMERIC_AGGREGATIONS = {"sum", "avg", "min", "max"}
SPECIAL_AGGREGATIONS = {"count_distinct"}
SAFE_FILTER_OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "contains", "in"}
DATE_HINTS = {"date", "orderdate", "invoicedate", "deliverydate", "servicedate", "transactiondate"}


class QueryBuildError(ValueError):
    pass


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


def _sheet_exists(sheet: str, dataset_metadata: dict[str, Any]) -> bool:
    return str(sheet) in set(dataset_metadata.get("sheets") or [])


def _resolve_column(sheet: str, column_name: str, dataset_metadata: dict[str, Any]) -> str:
    columns = _columns_by_sheet(dataset_metadata).get(sheet) or []
    if column_name in columns:
        return column_name

    normalized = _normalize_name(column_name)
    for candidate in columns:
        if _normalize_name(candidate) == normalized:
            return candidate

    raise QueryBuildError(f"Column '{column_name}' does not exist in sheet '{sheet}'.")


def _resolve_column_in_alias_scope(
    column_name: str,
    alias_map: dict[str, str],
    dataset_metadata: dict[str, Any],
    preferred_sheet: str,
) -> tuple[str, str]:
    candidate_sheets = [preferred_sheet] + [sheet for sheet in alias_map if sheet != preferred_sheet]
    for sheet in candidate_sheets:
        try:
            resolved = _resolve_column(sheet, column_name, dataset_metadata)
            return sheet, resolved
        except QueryBuildError:
            continue
    raise QueryBuildError(f"Column '{column_name}' is not available in the reachable sheets for this query.")


def _column_is_numeric(sheet: str, column_name: str, dataset_metadata: dict[str, Any]) -> bool:
    col_types = _column_types_by_sheet(dataset_metadata).get(sheet) or {}
    col_type = str(col_types.get(column_name) or "").lower()
    if col_type == "numeric":
        return True
    normalized = _normalize_name(column_name)
    return any(token in normalized for token in NUMERIC_HINTS)


def _column_is_date(sheet: str, column_name: str, dataset_metadata: dict[str, Any]) -> bool:
    col_types = _column_types_by_sheet(dataset_metadata).get(sheet) or {}
    col_type = str(col_types.get(column_name) or "").lower()
    if col_type == "date":
        return True
    normalized = _normalize_name(column_name).replace("_", "")
    return normalized in DATE_HINTS or normalized.endswith("date")


def _pick_time_column(alias_map: dict[str, str], dataset_metadata: dict[str, Any], preferred_sheet: str) -> tuple[str, str]:
    candidate_sheets = [preferred_sheet] + [sheet for sheet in alias_map if sheet != preferred_sheet]
    for sheet in candidate_sheets:
        for column_name in _columns_by_sheet(dataset_metadata).get(sheet) or []:
            if _column_is_date(sheet, column_name, dataset_metadata):
                return sheet, column_name
    raise QueryBuildError("No date-like column is available for time-grain grouping.")


def _date_parse_expr(alias: str, column_name: str) -> str:
    text_expr = _text_expr(alias, column_name)
    return (
        "(CASE "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{1,2}}-\\d{{1,2}}$' THEN TO_DATE({text_expr}, 'YYYY-MM-DD') "
        f"WHEN {text_expr} ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$' THEN TO_DATE({text_expr}, 'MM/DD/YYYY') "
        f"WHEN {text_expr} ~ '^\\d{{1,2}}-\\d{{1,2}}-\\d{{4}}$' THEN TO_DATE({text_expr}, 'MM-DD-YYYY') "
        "ELSE NULL END)"
    )


def _relationship_exists(join_step: dict[str, Any], dataset_metadata: dict[str, Any]) -> bool:
    relationships = dataset_metadata.get("relationships") or []
    left_sheet = str(join_step.get("left_sheet") or "")
    left_col = str(join_step.get("left_col") or "")
    right_sheet = str(join_step.get("right_sheet") or "")
    right_col = str(join_step.get("right_col") or "")
    for relationship in relationships:
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


def _text_expr(alias: str, column_name: str) -> str:
    return f"{alias}.row_json->>'{column_name}'"


def _status_like_column(column_name: str) -> bool:
    return bool({"status", "state", "stage", "exception"} & set(_normalize_name(column_name).split("_")))


def _normalized_text_filter_expr(alias: str, column_name: str) -> str:
    return f"regexp_replace(LOWER(COALESCE({_text_expr(alias, column_name)}, '')), '[^a-z0-9]+', '', 'g')"


def _normalized_text_filter_value(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _numeric_expr(alias: str, column_name: str) -> str:
    return (
        f"NULLIF(regexp_replace(({alias}.row_json->>'{column_name}'), '[^0-9.\\-]', '', 'g'), '')::numeric"
    )


def _parse_formula(formula: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"\s*(.+?)\s*([+\-*/])\s*(.+?)\s*", str(formula or ""))
    if not match:
        raise QueryBuildError(f"Unsupported derived metric formula '{formula}'.")
    return match.group(1).strip(), match.group(2), match.group(3).strip()


def _formula_operand_expr(
    source_alias: str,
    source_sheet: str,
    reference_name: str,
    dataset_metadata: dict[str, Any],
    *,
    visited: set[str] | None = None,
) -> str:
    visited = visited or set()
    normalized_reference = _normalize_name(reference_name)
    derived_metrics = _derived_metrics_by_sheet(dataset_metadata).get(source_sheet) or {}

    for metric_name, formula in derived_metrics.items():
        if _normalize_name(metric_name) != normalized_reference:
            continue
        if metric_name in visited:
            raise QueryBuildError(f"Derived metric '{metric_name}' has a cyclic formula definition.")
        left_operand, operator, right_operand = _parse_formula(formula)
        next_visited = set(visited)
        next_visited.add(metric_name)
        left_expr = _formula_operand_expr(source_alias, source_sheet, left_operand, dataset_metadata, visited=next_visited)
        right_expr = _formula_operand_expr(source_alias, source_sheet, right_operand, dataset_metadata, visited=next_visited)
        return f"({left_expr} {operator} {right_expr})"

    resolved_column = _resolve_column(source_sheet, reference_name, dataset_metadata)
    if not _column_is_numeric(source_sheet, resolved_column, dataset_metadata):
        raise QueryBuildError(
            f"Derived metric reference '{resolved_column}' is not numeric in sheet '{source_sheet}'."
        )
    return _numeric_expr(source_alias, resolved_column)


def _formula_sql_expression(
    formula: str,
    source_alias: str,
    source_sheet: str,
    dataset_metadata: dict[str, Any],
) -> str:
    left_operand, operator, right_operand = _parse_formula(formula)
    left_expr = _formula_operand_expr(source_alias, source_sheet, left_operand, dataset_metadata, visited=set())
    right_expr = _formula_operand_expr(source_alias, source_sheet, right_operand, dataset_metadata, visited=set())
    return f"({left_expr} {operator} {right_expr})"


def _base_from_and_filters(
    query_spec: QuerySpec,
    dataset_metadata: dict[str, Any],
) -> tuple[str, dict[str, str], list[str], list[str], dict[str, Any], str]:
    sheets = [str(sheet) for sheet in (dataset_metadata.get("sheets") or [])]
    if not sheets:
        raise QueryBuildError("No dataset sheets available for SQL building.")

    join_path = [step for step in (query_spec.join_path or []) if isinstance(step, dict)]
    base_sheet = query_spec.target_sheet or (str(join_path[0].get("left_sheet")) if join_path else sheets[0])
    if not _sheet_exists(base_sheet, dataset_metadata):
        raise QueryBuildError(f"Sheet '{base_sheet}' does not exist.")

    alias_map: dict[str, str] = {base_sheet: "s"}
    join_clauses: list[str] = []
    where_clauses = ["s.dataset_id = %(dataset_id)s", f"s.sheet_name = '{base_sheet.replace("'", "''")}'"]
    params: dict[str, Any] = {"limit": int(query_spec.limit or 100)}

    for index, join_step in enumerate(join_path, start=1):
        left_sheet = str(join_step.get("left_sheet") or base_sheet)
        right_sheet = str(join_step.get("right_sheet") or "")
        left_col = _resolve_column(left_sheet, str(join_step.get("left_col") or ""), dataset_metadata)
        right_col = _resolve_column(right_sheet, str(join_step.get("right_col") or ""), dataset_metadata)
        if not right_sheet or not _sheet_exists(right_sheet, dataset_metadata):
            raise QueryBuildError(f"Sheet '{right_sheet}' does not exist.")
        if not _relationship_exists({
            "left_sheet": left_sheet,
            "left_col": left_col,
            "right_sheet": right_sheet,
            "right_col": right_col,
        }, dataset_metadata):
            raise QueryBuildError(
                f"Join from {left_sheet}.{left_col} to {right_sheet}.{right_col} is not a known relationship."
            )
        left_alias = alias_map.setdefault(left_sheet, "s" if left_sheet == base_sheet else f"j{len(alias_map)}")
        right_alias = alias_map.setdefault(right_sheet, f"j{index}")
        join_clauses.append(
            f"LEFT JOIN dataset_rows {right_alias} ON {right_alias}.dataset_id = {left_alias}.dataset_id "
            f"AND {right_alias}.sheet_name = '{right_sheet.replace("'", "''")}' "
            f"AND NULLIF(LOWER(TRIM({_text_expr(left_alias, left_col)})), '') = "
            f"NULLIF(LOWER(TRIM({_text_expr(right_alias, right_col)})), '')"
        )

    for filter_index, filter_spec in enumerate(query_spec.filters):
        if not isinstance(filter_spec, dict):
            raise QueryBuildError("Each filter must be an object.")
        sheet = str(filter_spec.get("sheet") or base_sheet)
        if not _sheet_exists(sheet, dataset_metadata):
            raise QueryBuildError(f"Sheet '{sheet}' does not exist.")
        alias = alias_map.get(sheet)
        if alias is None:
            raise QueryBuildError(f"Sheet '{sheet}' is not reachable in the join path.")
        operator = str(filter_spec.get("operator") or filter_spec.get("op") or "=").strip().lower()
        if operator not in SAFE_FILTER_OPERATORS:
            raise QueryBuildError(f"Unsupported filter operator '{operator}'.")

        column_name = _resolve_column(sheet, str(filter_spec.get("column") or ""), dataset_metadata)
        param_name = f"filter_{filter_index}"
        value = filter_spec.get("value")
        if operator == "contains":
            if _status_like_column(column_name):
                where_clauses.append(f"{_normalized_text_filter_expr(alias, column_name)} LIKE %({param_name})s")
                params[param_name] = f"%{_normalized_text_filter_value(value)}%"
            else:
                where_clauses.append(f"LOWER(COALESCE({_text_expr(alias, column_name)}, '')) LIKE %({param_name})s")
                params[param_name] = f"%{str(value or '').lower()}%"
        elif str(filter_spec.get("kind") or "").lower() == "year":
            if not _column_is_date(sheet, column_name, dataset_metadata):
                raise QueryBuildError(
                    f"Year filter requires a date-like column, but '{column_name}' in sheet '{sheet}' is not date-like."
                )
            where_clauses.append(
                f"EXTRACT(YEAR FROM {_date_parse_expr(alias, column_name)})::int {operator} %({param_name})s"
            )
            params[param_name] = int(value)
        elif operator == "in":
            values = list(value or [])
            if not values:
                raise QueryBuildError(f"Filter '{column_name}' with operator 'in' requires values.")
            placeholders = []
            use_normalized_text_filter = _status_like_column(column_name)
            for value_index, item in enumerate(values):
                item_param = f"{param_name}_{value_index}"
                placeholders.append(f"%({item_param})s")
                if use_normalized_text_filter:
                    params[item_param] = _normalized_text_filter_value(item)
                else:
                    params[item_param] = "" if item is None else str(item).lower()
            filter_expr = _normalized_text_filter_expr(alias, column_name) if use_normalized_text_filter else f"LOWER(COALESCE({_text_expr(alias, column_name)}, ''))"
            where_clauses.append(
                f"{filter_expr} IN ({', '.join(placeholders)})"
            )
        elif operator in {">", ">=", "<", "<="}:
            if not _column_is_numeric(sheet, column_name, dataset_metadata):
                raise QueryBuildError(
                    f"Filter operator '{operator}' requires numeric column '{column_name}' in sheet '{sheet}'."
                )
            where_clauses.append(f"{_numeric_expr(alias, column_name)} {operator} %({param_name})s")
            params[param_name] = value
        else:
            if operator in {"=", "!="}:
                if _status_like_column(column_name):
                    where_clauses.append(f"{_normalized_text_filter_expr(alias, column_name)} {operator} %({param_name})s")
                    params[param_name] = _normalized_text_filter_value(value)
                else:
                    where_clauses.append(f"LOWER(COALESCE({_text_expr(alias, column_name)}, '')) {operator} %({param_name})s")
                    params[param_name] = "" if value is None else str(value).lower()
            else:
                where_clauses.append(f"COALESCE({_text_expr(alias, column_name)}, '') {operator} %({param_name})s")
                params[param_name] = "" if value is None else str(value)

    return base_sheet, alias_map, join_clauses, where_clauses, params, "FROM dataset_rows s"


def _measure_sql_expression(
    aggregation: str,
    source_alias: str,
    source_sheet: str,
    resolved_column: str,
    dataset_metadata: dict[str, Any],
    *,
    formula: str | None = None,
) -> str:
    value_expression = _formula_sql_expression(formula, source_alias, source_sheet, dataset_metadata) if formula else _numeric_expr(source_alias, resolved_column)
    if aggregation == "count":
        return f"COUNT({value_expression if formula else _text_expr(source_alias, resolved_column)})"
    if aggregation == "count_distinct":
        return f"COUNT(DISTINCT {value_expression if formula else _text_expr(source_alias, resolved_column)})"
    if aggregation in NUMERIC_AGGREGATIONS:
        if not formula and not _column_is_numeric(source_sheet, resolved_column, dataset_metadata):
            raise QueryBuildError(
                f"Aggregation '{aggregation}' is not valid for non-numeric column '{resolved_column}' in sheet '{source_sheet}'."
            )
        return f"{aggregation.upper()}({value_expression})"
    raise QueryBuildError(f"Unsupported aggregation '{aggregation}'.")


def _build_ranking_sql(query_spec: QuerySpec, dataset_metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not query_spec.partition_by:
        raise QueryBuildError("Ranking query requires partition_by columns.")
    if not query_spec.rank_dimension:
        raise QueryBuildError("Ranking query requires rank_dimension.")
    if not query_spec.measures:
        raise QueryBuildError("Ranking query requires a metric measure.")

    base_sheet, alias_map, join_clauses, where_clauses, params, from_clause = _base_from_and_filters(query_spec, dataset_metadata)
    params["rank_limit"] = int(query_spec.rank_limit or 1)

    partition_selects: list[str] = []
    partition_group_by: list[str] = []
    partition_aliases: list[str] = []
    for column_name in query_spec.partition_by:
        partition_sheet, partition_column = _resolve_column_in_alias_scope(column_name, alias_map, dataset_metadata, base_sheet)
        expr = _text_expr(alias_map[partition_sheet], partition_column)
        partition_selects.append(f"{expr} AS {partition_column}")
        partition_group_by.append(expr)
        partition_aliases.append(partition_column)

    rank_sheet, rank_column = _resolve_column_in_alias_scope(query_spec.rank_dimension, alias_map, dataset_metadata, base_sheet)
    rank_expr = _text_expr(alias_map[rank_sheet], rank_column)

    measure = query_spec.measures[0]
    aggregation = str(measure.get("aggregation") or measure.get("agg") or "sum").strip().lower()
    source_sheet = str(measure.get("sheet") or base_sheet)
    if not _sheet_exists(source_sheet, dataset_metadata):
        raise QueryBuildError(f"Sheet '{source_sheet}' does not exist.")
    source_alias = alias_map.get(source_sheet)
    if source_alias is None:
        raise QueryBuildError(f"Sheet '{source_sheet}' is not reachable in the join path.")
    column_name = str(measure.get("column") or "*").strip()
    alias_name = str(measure.get("alias") or "value").strip() or "value"
    formula = str(measure.get("formula") or "").strip() or None
    if column_name == "*":
        metric_expr = "COUNT(*)"
    else:
        resolved_column = _resolve_column(source_sheet, column_name, dataset_metadata) if not formula else column_name
        metric_expr = _measure_sql_expression(aggregation, source_alias, source_sheet, resolved_column, dataset_metadata, formula=formula)

    base_select_parts = partition_selects + [f"{rank_expr} AS {rank_column}", f"{metric_expr} AS {alias_name}"]
    base_group_by = partition_group_by + [rank_expr]
    base_sql = (
        "SELECT\n    "
        + ",\n    ".join(base_select_parts)
        + f"\n{from_clause}"
    )
    if join_clauses:
        base_sql += "\n" + "\n".join(join_clauses)
    base_sql += "\nWHERE " + " AND ".join(where_clauses)
    base_sql += "\nGROUP BY " + ", ".join(base_group_by)

    order_direction = "DESC"
    if query_spec.order_by:
        order_direction = str(query_spec.order_by[0].get("direction") or "DESC").upper()
        if order_direction not in {"ASC", "DESC"}:
            order_direction = "DESC"

    partition_clause = ", ".join(partition_aliases)
    select_columns = partition_aliases + [rank_column, alias_name]
    order_columns = ", ".join(partition_aliases + [rank_column]) if partition_aliases else rank_column

    sql = (
        "WITH aggregated AS (\n"
        f"{base_sql}\n"
        "), ranked AS (\n"
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + ",\n    ROW_NUMBER() OVER (\n"
        f"        PARTITION BY {partition_clause}\n"
        f"        ORDER BY {alias_name} {order_direction}, {rank_column} ASC\n"
        "    ) AS rn\n"
        "FROM aggregated\n"
        ")\n"
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + "\nFROM ranked\n"
        "WHERE rn <= %(rank_limit)s\n"
        f"ORDER BY {order_columns}\n"
        "LIMIT %(limit)s"
    )
    return sql, params


def build_sql_from_query_spec(query_spec: QuerySpec, dataset_metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if str(query_spec.query_type or "").strip().lower() == "ranking":
        return _build_ranking_sql(query_spec, dataset_metadata)

    base_sheet, alias_map, join_clauses, where_clauses, params, _from_clause = _base_from_and_filters(query_spec, dataset_metadata)

    if str(query_spec.query_type or "").strip().lower() == "row_preview":
        preview_columns = list(query_spec.columns or (_columns_by_sheet(dataset_metadata).get(base_sheet) or []))
        if not preview_columns:
            raise QueryBuildError("Row preview query requires at least one selected column.")

        select_parts: list[str] = []
        available_select_names: set[str] = set()
        for column_name in preview_columns:
            source_sheet, resolved_column = _resolve_column_in_alias_scope(
                column_name,
                alias_map,
                dataset_metadata,
                base_sheet,
            )
            select_parts.append(f"{_text_expr(alias_map[source_sheet], resolved_column)} AS {resolved_column}")
            available_select_names.add(resolved_column)

        order_by_parts: list[str] = []
        for order_spec in query_spec.order_by:
            if not isinstance(order_spec, dict):
                continue
            field = str(order_spec.get("field") or order_spec.get("column") or "").strip()
            direction = str(order_spec.get("direction") or "asc").strip().upper()
            if field not in available_select_names:
                continue
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            order_by_parts.append(f"{field} {direction}")

        sql_parts = ["SELECT", ", ".join(select_parts), "FROM dataset_rows s"]
        sql_parts.extend(join_clauses)
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
        if order_by_parts:
            sql_parts.append("ORDER BY " + ", ".join(order_by_parts))
        sql_parts.append("LIMIT %(limit)s")

        sql = "\n".join(sql_parts).strip()
        if ";" in sql:
            raise QueryBuildError("Generated SQL must not contain semicolons.")
        return sql, params

    if str(query_spec.query_type or "").strip().lower() == "projection":
        if not query_spec.columns:
            raise QueryBuildError("Projection query requires at least one selected column.")

        select_parts: list[str] = []
        available_select_names: set[str] = set()
        for column_name in query_spec.columns:
            source_sheet, resolved_column = _resolve_column_in_alias_scope(
                column_name,
                alias_map,
                dataset_metadata,
                base_sheet,
            )
            select_parts.append(f"{_text_expr(alias_map[source_sheet], resolved_column)} AS {resolved_column}")
            available_select_names.add(resolved_column)

        order_by_parts: list[str] = []
        for order_spec in query_spec.order_by:
            if not isinstance(order_spec, dict):
                continue
            field = str(order_spec.get("field") or order_spec.get("column") or "").strip()
            direction = str(order_spec.get("direction") or "asc").strip().upper()
            if field not in available_select_names:
                continue
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            order_by_parts.append(f"{field} {direction}")

        sql_parts = ["SELECT", ", ".join(select_parts), "FROM dataset_rows s"]
        sql_parts.extend(join_clauses)
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
        if order_by_parts:
            sql_parts.append("ORDER BY " + ", ".join(order_by_parts))
        sql_parts.append("LIMIT %(limit)s")

        sql = "\n".join(sql_parts).strip()
        if ";" in sql:
            raise QueryBuildError("Generated SQL must not contain semicolons.")
        return sql, params

    select_parts: list[str] = []
    group_by_parts: list[str] = []
    available_select_names: set[str] = set()

    if query_spec.time_grain:
        time_grain = str(query_spec.time_grain).strip().lower()
        if time_grain == "year":
            time_sheet, time_column = _pick_time_column(alias_map, dataset_metadata, base_sheet)
            year_expr = f"EXTRACT(YEAR FROM {_date_parse_expr(alias_map[time_sheet], time_column)})::int"
            select_parts.append(f"{year_expr} AS year")
            group_by_parts.append(year_expr)
            available_select_names.add("year")
        elif time_grain == "month":
            time_sheet, time_column = _pick_time_column(alias_map, dataset_metadata, base_sheet)
            month_expr = f"TO_CHAR({_date_parse_expr(alias_map[time_sheet], time_column)}, 'YYYY-MM')"
            select_parts.append(f"{month_expr} AS month")
            group_by_parts.append(month_expr)
            available_select_names.add("month")
        elif time_grain == "quarter":
            time_sheet, time_column = _pick_time_column(alias_map, dataset_metadata, base_sheet)
            quarter_expr = f"TO_CHAR({_date_parse_expr(alias_map[time_sheet], time_column)}, 'YYYY-\"Q\"Q')"
            select_parts.append(f"{quarter_expr} AS quarter")
            group_by_parts.append(quarter_expr)
            available_select_names.add("quarter")
        else:
            raise QueryBuildError(f"Unsupported time_grain '{query_spec.time_grain}'.")

    for dimension in query_spec.dimensions:
        dimension_sheet, dimension_col = _resolve_column_in_alias_scope(
            dimension,
            alias_map,
            dataset_metadata,
            base_sheet,
        )
        expr = _text_expr(alias_map[dimension_sheet], dimension_col)
        select_parts.append(f"{expr} AS {dimension_col}")
        group_by_parts.append(expr)
        available_select_names.add(dimension_col)

    aggregate_present = False
    for measure_index, measure in enumerate(query_spec.measures):
        if not isinstance(measure, dict):
            raise QueryBuildError("Each measure must be an object.")

        aggregation = str(measure.get("aggregation") or measure.get("agg") or "count").strip().lower()
        source_sheet = str(measure.get("sheet") or base_sheet)
        if not _sheet_exists(source_sheet, dataset_metadata):
            raise QueryBuildError(f"Sheet '{source_sheet}' does not exist.")
        source_alias = alias_map.get(source_sheet)
        if source_alias is None:
            raise QueryBuildError(f"Sheet '{source_sheet}' is not reachable in the join path.")

        column_name = str(measure.get("column") or "*").strip()
        formula = str(measure.get("formula") or "").strip() or None
        alias_name = str(measure.get("alias") or "").strip()
        if not alias_name:
            alias_name = "row_count" if column_name == "*" and aggregation == "count" else f"{aggregation}_{_normalize_name(column_name or 'value')}"

        if aggregation == "count" and column_name == "*":
            select_parts.append(f"COUNT(*) AS {alias_name}")
            available_select_names.add(alias_name)
            aggregate_present = True
            continue

        resolved_column = _resolve_column(source_sheet, column_name, dataset_metadata) if not formula else column_name
        if aggregation in NUMERIC_AGGREGATIONS and not formula and not _column_is_numeric(source_sheet, resolved_column, dataset_metadata):
            raise QueryBuildError(
                f"Aggregation '{aggregation}' is not valid for non-numeric column '{resolved_column}' in sheet '{source_sheet}'."
            )

        measure_expr = _measure_sql_expression(aggregation, source_alias, source_sheet, resolved_column, dataset_metadata, formula=formula)

        select_parts.append(f"{measure_expr} AS {alias_name}")
        available_select_names.add(alias_name)
        aggregate_present = True

    if not select_parts:
        raise QueryBuildError("QuerySpec did not produce any selectable fields.")

    order_by_parts: list[str] = []
    for order_spec in query_spec.order_by:
        if not isinstance(order_spec, dict):
            continue
        field = str(order_spec.get("field") or order_spec.get("column") or "").strip()
        direction = str(order_spec.get("direction") or "desc").strip().upper()
        if not field:
            continue
        if field not in available_select_names:
            _resolved_sheet, resolved_field = _resolve_column_in_alias_scope(
                field,
                alias_map,
                dataset_metadata,
                base_sheet,
            )
            if resolved_field not in available_select_names:
                raise QueryBuildError(f"Order-by field '{field}' is not part of the built query.")
            field = resolved_field
        if direction not in {"ASC", "DESC"}:
            direction = "DESC"
        order_by_parts.append(f"{field} {direction}")

    select_keyword = "SELECT DISTINCT" if not aggregate_present and bool(query_spec.dimensions) else "SELECT"
    sql_parts = [select_keyword, ", ".join(select_parts), "FROM dataset_rows s"]
    sql_parts.extend(join_clauses)
    sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if aggregate_present and group_by_parts:
        sql_parts.append("GROUP BY " + ", ".join(group_by_parts))
    if order_by_parts:
        sql_parts.append("ORDER BY " + ", ".join(order_by_parts))
    sql_parts.append("LIMIT %(limit)s")

    sql = "\n".join(sql_parts).strip()
    if ";" in sql:
        raise QueryBuildError("Generated SQL must not contain semicolons.")
    return sql, params
