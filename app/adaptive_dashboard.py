from __future__ import annotations

from dataclasses import asdict, dataclass, field
import io
import json
import math
import os
import re
from typing import Any, Literal

try:
    import altair as alt
except Exception:
    alt = None

try:
    import duckdb
except Exception:
    duckdb = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import pandas as pd


DATASET_SALES = "sales"
DATASET_BILLING = "billing"
DATASET_GENERIC = "generic"

AMOUNT_TOKENS = {
    "amount",
    "price",
    "revenue",
    "cost",
    "total",
    "sales",
    "profit",
    "balance",
    "billed",
    "expected",
    "invoice_amount",
}
STATUS_TOKENS = {
    "status",
    "state",
    "payment_status",
    "invoice_status",
    "billing_status",
    "exception",
    "exception_type",
    "billing_exception",
}
REGION_TOKENS = {"region", "state", "country", "city", "territory", "market"}
PRODUCT_TOKENS = {"product", "item", "sku", "service", "category"}
CUSTOMER_TOKENS = {"customer", "client", "account", "buyer", "vendor"}
QUANTITY_TOKENS = {"quantity", "qty", "units", "unit", "volume"}

STATUS_POSITIVE_VALUES = {"paid", "billed", "invoiced", "closed", "complete", "completed", "settled"}
STATUS_NEGATIVE_VALUES = {"unpaid", "unbilled", "open", "pending", "outstanding", "overdue", "due"}

BILLING_FOCUS_LABELS = {
    "all": "All",
    "unbilled": "Unbilled",
    "under_billed": "Under Billed",
    "billing_delay": "Billing Delay",
    "overdue": "Overdue",
    "billed": "Billed",
}

KNOWN_RECOMMENDED_ACTION_PHRASES = [
    "invoice now",
    "create invoice",
    "fix rate qty",
    "debit note",
    "correction",
    "reduce billing lag",
    "process automation",
]


@dataclass(slots=True)
class ColumnMetadata:
    column_name: str
    detected_type: str
    distinct_count: int
    null_percentage: float
    semantic_role: str


@dataclass(slots=True)
class DatasetProfile:
    dataset_name: str
    row_count: int
    column_count: int
    dataset_type: str
    numeric_columns: list[str]
    date_columns: list[str]
    categorical_columns: list[str]
    amount_columns: list[str]
    status_columns: list[str]
    region_columns: list[str]
    product_columns: list[str]
    customer_columns: list[str]
    quantity_columns: list[str]
    column_metadata: list[ColumnMetadata]


@dataclass(slots=True)
class MetricCard:
    title: str
    value: str
    help_text: str = ""


@dataclass(slots=True)
class ChartSpec:
    title: str
    chart_type: str
    dataframe: pd.DataFrame
    x_column: str
    y_column: str
    color_column: str | None = None
    description: str = ""
    supported_chart_types: list[str] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""


@dataclass(slots=True)
class AdaptiveDashboardSummaryFacts:
    dataset_type: str
    row_count: int
    column_count: int
    active_filter_summary: str
    metric_highlights: list[str]
    top_dimension_label: str | None
    top_category: str | None
    top_metric_label: str | None
    top_metric_value: float | None
    chart_count: int
    confidence: float
    use_neutral_language: bool


CHART_TYPE_LABELS = {
    "bar": "Bar",
    "pie": "Pie",
    "line": "Line",
    "area": "Area",
    "grouped_bar": "Grouped Bar",
    "stacked_bar": "Stacked Bar",
}

SIMPLE_CHART_TYPES = ["bar", "pie", "line"]
TREND_CHART_TYPES = ["line", "area", "bar"]
COMPARISON_CHART_TYPES = ["grouped_bar", "stacked_bar", "bar"]


def _normalize_join_key_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized


def load_tabular_frames(file_name: str, file_bytes: bytes) -> dict[str, pd.DataFrame]:
    suffix = str(file_name or "").lower().rsplit(".", 1)[-1] if "." in str(file_name or "") else ""
    buffer = io.BytesIO(file_bytes)

    if suffix == "csv":
        df = pd.read_csv(buffer)
        return {"data": df}

    if suffix not in {"xlsx", "xls"}:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    workbook = pd.ExcelFile(buffer)
    frames: dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        sheet_df = workbook.parse(sheet_name=sheet_name)
        if isinstance(sheet_df, pd.DataFrame) and not sheet_df.empty:
            frames[str(sheet_name)] = sheet_df
    if not frames:
        raise ValueError("The workbook does not contain any non-empty sheets.")
    return frames


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for index, original_name in enumerate(cleaned_df.columns, start=1):
        text = str(original_name or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_") or f"column_{index}"
        count = seen.get(text, 0)
        seen[text] = count + 1
        new_columns.append(text if count == 0 else f"{text}_{count + 1}")

    cleaned_df.columns = new_columns
    return cleaned_df


def build_joined_frame(
    sheet_frames: dict[str, pd.DataFrame],
    relationship: dict[str, Any],
    *,
    join_how: Literal["left", "inner", "right", "outer"] = "left",
) -> pd.DataFrame:
    left_sheet = str(relationship.get("left") or "").strip()
    right_sheet = str(relationship.get("right") or "").strip()
    left_col = str(relationship.get("left_col") or "").strip()
    right_col = str(relationship.get("right_col") or "").strip()
    if not left_sheet or not right_sheet or not left_col or not right_col:
        raise ValueError("Relationship must include left, right, left_col, and right_col.")
    if left_sheet not in sheet_frames or right_sheet not in sheet_frames:
        raise ValueError("Selected relationship references sheets that are not loaded.")

    left_df = clean_column_names(sheet_frames[left_sheet]).copy()
    right_df = clean_column_names(sheet_frames[right_sheet]).copy()
    left_join_col = re.sub(r"[^a-z0-9]+", "_", left_col.strip().lower()).strip("_")
    right_join_col = re.sub(r"[^a-z0-9]+", "_", right_col.strip().lower()).strip("_")
    if left_join_col not in left_df.columns or right_join_col not in right_df.columns:
        raise ValueError("Join columns are not present in the selected sheets.")

    left_df["__join_key"] = _normalize_join_key_series(left_df[left_join_col])
    right_df["__join_key"] = _normalize_join_key_series(right_df[right_join_col])
    left_df = left_df[left_df["__join_key"] != ""].copy()
    right_df = right_df[right_df["__join_key"] != ""].copy()

    merged = left_df.merge(
        right_df,
        on="__join_key",
        how=join_how,
        suffixes=(f"_{left_sheet.lower()}", f"_{right_sheet.lower()}"),
    )
    merged = merged.drop(columns=["__join_key"], errors="ignore")
    return merged.reset_index(drop=True)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    normalized = _normalize_text_series(series)
    normalized = normalized.str.replace(r"[$€£¥₹,%]", "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


def _looks_date_like_series(series: pd.Series) -> bool:
    non_empty = _normalize_text_series(series)
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        return False

    sample = non_empty.head(25)
    date_pattern = (
        r"^(?:"
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"|\d{4}-\d{1,2}"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,-]+\d{1,2}(?:[\s,/-]+\d{2,4})?"
        r"|\d{1,2}[\s,-]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:[\s,/-]+\d{2,4})?"
        r")$"
    )
    match_ratio = float(sample.str.match(date_pattern, case=False).mean())
    return match_ratio >= 0.6


def _coerce_date_series(series: pd.Series) -> pd.Series:
    normalized = _normalize_text_series(series)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if normalized.eq("").all():
        return parsed

    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y-%m"]:
        remaining_mask = normalized.ne("") & parsed.isna()
        if not remaining_mask.any():
            break
        parsed.loc[remaining_mask] = pd.to_datetime(normalized.loc[remaining_mask], format=fmt, errors="coerce")

    remaining_mask = normalized.ne("") & parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(normalized.loc[remaining_mask], format="mixed", errors="coerce")

    return parsed


def _is_forced_text_column(column_name: str) -> bool:
    normalized = str(column_name or "").strip().lower()
    if _is_identifier_like_column(normalized):
        return True
    return normalized in {"customer", "event_id", "invoice_id", "invoice_number", "invoice_no", "invoice_num"}


def _detect_series_type(series: pd.Series) -> tuple[str, pd.Series]:
    non_null = series.dropna()
    if non_null.empty:
        return "categorical", _normalize_text_series(series)

    if pd.api.types.is_numeric_dtype(series):
        return "numeric", pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date", pd.to_datetime(series, errors="coerce")

    numeric_series = _coerce_numeric_series(series)
    numeric_ratio = float(numeric_series.notna().sum()) / float(max(len(non_null), 1))
    if numeric_ratio >= 0.85:
        return "numeric", numeric_series

    normalized = _normalize_text_series(series)
    if not _looks_date_like_series(series):
        return "categorical", normalized

    parsed_dates = _coerce_date_series(series)
    date_ratio = float(parsed_dates.notna().sum()) / float(max(len(non_null), 1))
    if date_ratio >= 0.85:
        return "date", parsed_dates

    return "categorical", normalized


def _guess_semantic_role(column_name: str, detected_type: str) -> str:
    tokens = set(str(column_name or "").split("_"))
    normalized = str(column_name or "")

    if detected_type == "numeric" and (tokens & AMOUNT_TOKENS or normalized in AMOUNT_TOKENS):
        return "amount"
    if detected_type == "date":
        return "date"
    if tokens & STATUS_TOKENS or any(token in normalized for token in STATUS_TOKENS):
        return "status"
    if tokens & REGION_TOKENS or any(token in normalized for token in REGION_TOKENS):
        return "region"
    if tokens & PRODUCT_TOKENS or any(token in normalized for token in PRODUCT_TOKENS):
        return "product"
    if tokens & CUSTOMER_TOKENS or any(token in normalized for token in CUSTOMER_TOKENS):
        return "customer"
    if detected_type == "numeric" and (tokens & QUANTITY_TOKENS or normalized in QUANTITY_TOKENS):
        return "quantity"
    if detected_type == "numeric":
        return "metric"
    return "category"


def _infer_dataset_type(
    amount_columns: list[str],
    status_columns: list[str],
    region_columns: list[str],
    product_columns: list[str],
    customer_columns: list[str],
    date_columns: list[str],
) -> str:
    joined_amounts = " ".join(amount_columns)
    joined_statuses = " ".join(status_columns)
    if any(token in joined_amounts for token in ["invoice", "billed", "unpaid", "expected"]) or any(
        token in joined_statuses for token in ["invoice", "payment", "billing"]
    ):
        return DATASET_BILLING
    if amount_columns and (region_columns or product_columns or customer_columns) and date_columns:
        return DATASET_SALES
    return DATASET_GENERIC


def prepare_dataset(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, DatasetProfile]:
    working_df = clean_column_names(df)
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    categorical_columns: list[str] = []
    amount_columns: list[str] = []
    status_columns: list[str] = []
    region_columns: list[str] = []
    product_columns: list[str] = []
    customer_columns: list[str] = []
    quantity_columns: list[str] = []
    column_metadata: list[ColumnMetadata] = []

    for column_name in list(working_df.columns):
        if _is_forced_text_column(column_name):
            detected_type, coerced_series = "categorical", _normalize_text_series(working_df[column_name])
        else:
            detected_type, coerced_series = _detect_series_type(working_df[column_name])
        working_df[column_name] = coerced_series
        role = _guess_semantic_role(column_name, detected_type)

        if detected_type == "numeric":
            numeric_columns.append(column_name)
        elif detected_type == "date":
            date_columns.append(column_name)
        else:
            categorical_columns.append(column_name)

        if role == "amount":
            amount_columns.append(column_name)
        elif role == "status":
            status_columns.append(column_name)
        elif role == "region":
            region_columns.append(column_name)
        elif role == "product":
            product_columns.append(column_name)
        elif role == "customer":
            customer_columns.append(column_name)
        elif role == "quantity":
            quantity_columns.append(column_name)

        distinct_count = int(working_df[column_name].nunique(dropna=True))
        null_percentage = float(working_df[column_name].isna().mean()) * 100.0
        column_metadata.append(
            ColumnMetadata(
                column_name=column_name,
                detected_type=detected_type,
                distinct_count=distinct_count,
                null_percentage=round(null_percentage, 2),
                semantic_role=role,
            )
        )

    dataset_type = _infer_dataset_type(
        amount_columns,
        status_columns,
        region_columns,
        product_columns,
        customer_columns,
        date_columns,
    )

    profile = DatasetProfile(
        dataset_name=dataset_name,
        row_count=int(working_df.shape[0]),
        column_count=int(working_df.shape[1]),
        dataset_type=dataset_type,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        categorical_columns=categorical_columns,
        amount_columns=amount_columns,
        status_columns=status_columns,
        region_columns=region_columns,
        product_columns=product_columns,
        customer_columns=customer_columns,
        quantity_columns=quantity_columns,
        column_metadata=column_metadata,
    )
    return working_df, profile


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _run_query(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed in the active environment.")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("analytics_df", df)
        return connection.execute(sql).df()
    finally:
        connection.close()


def _format_metric_value(value: Any, *, currency: bool = False, integer: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if integer:
        return f"{int(value):,}"
    if currency:
        return f"${float(value):,.2f}"
    if isinstance(value, (int, float)):
        return f"{float(value):,.2f}"
    return str(value)


def _format_label(name: str) -> str:
    text = str(name or "").strip().replace("_", " ")
    return text.title() if text else "Value"


def _metric_value_label(measure_column: str | None, *, count_fallback: bool = False) -> str:
    if count_fallback or not measure_column:
        return "Count"
    return _format_label(measure_column)


def _classify_billing_measure_columns(profile: DatasetProfile) -> dict[str, list[str]]:
    numeric_candidates = list(dict.fromkeys([*profile.amount_columns, *profile.numeric_columns]))
    grouped: dict[str, list[str]] = {"expected": [], "billed": [], "gap": [], "other": []}

    for column_name in numeric_candidates:
        normalized = str(column_name or "").strip().lower()
        if any(token in normalized for token in ["gap", "variance", "difference", "delta", "shortfall"]):
            grouped["gap"].append(column_name)
        elif any(token in normalized for token in ["billed", "invoice", "invoiced", "paid"]):
            grouped["billed"].append(column_name)
        elif any(token in normalized for token in ["expected", "detected", "due", "charge", "target"]):
            grouped["expected"].append(column_name)
        else:
            grouped["other"].append(column_name)

    return grouped


def _preferred_billing_measure_columns(profile: DatasetProfile) -> list[str]:
    grouped = _classify_billing_measure_columns(profile)
    ordered: list[str] = []
    for group_name in ["expected", "billed", "gap", "other"]:
        for column_name in grouped[group_name]:
            if column_name not in ordered:
                ordered.append(column_name)
    return ordered


def _preferred_billing_chart_measures(filtered_df: pd.DataFrame, profile: DatasetProfile, *, max_count: int = 2) -> list[str]:
    grouped = _classify_billing_measure_columns(profile)
    ordered: list[str] = []
    for group_name in ["expected", "gap", "billed", "other"]:
        for column_name in grouped[group_name]:
            if column_name in filtered_df.columns and column_name not in ordered:
                ordered.append(column_name)
                break
        if len(ordered) >= max_count:
            break
    if not ordered:
        primary = _pick_primary_amount_column(profile)
        if primary and primary in filtered_df.columns:
            ordered.append(primary)
    return ordered[:max_count]


def _append_chart_if_missing(charts: list[ChartSpec], chart: ChartSpec | None) -> None:
    if chart is None:
        return
    existing_titles = {item.title for item in charts}
    if chart.title in existing_titles:
        return
    charts.append(chart)


def _pick_prompt_measure_column(prompt_text: str, profile: DatasetProfile) -> str | None:
    normalized = str(prompt_text or "").strip().lower()
    numeric_candidates = list(dict.fromkeys([*profile.amount_columns, *profile.numeric_columns]))
    explicit_candidates = [
        ("value gap", ["value_gap", "value_gap_amount", "gap_amount", "gap_value", "variance", "difference"]),
        ("gap", ["value_gap", "value_gap_amount", "gap_amount", "gap_value", "variance", "difference"]),
        ("expected amount", ["expected_amount", "service_detected_amount", "detected_amount", "amount_due", "charge_amount"]),
        ("billed amount", ["billed_amount", "invoice_amount", "invoiced_amount"]),
        ("invoice amount", ["invoice_amount", "billed_amount", "invoiced_amount"]),
        ("total price", ["total_price", "price", "amount", "sales_amount", "revenue"]),
        ("price", ["total_price", "unit_price", "price", "amount", "sales_amount", "revenue"]),
        ("revenue", ["revenue", "sales_amount", "total_price", "amount", "price"]),
        ("sales", ["sales_amount", "revenue", "total_price", "amount", "price"]),
        ("total", ["total_price", "sales_amount", "revenue", "amount", "price"]),
        ("amount", numeric_candidates),
    ]

    for token, candidates in explicit_candidates:
        if token not in normalized:
            continue
        for candidate in candidates:
            if candidate in numeric_candidates:
                return candidate
    return None


def _extract_prompt_measure_columns(prompt_text: str, profile: DatasetProfile) -> list[str]:
    normalized = str(prompt_text or "").strip().lower()
    numeric_candidates = list(dict.fromkeys([*profile.amount_columns, *profile.numeric_columns]))
    token_mappings = [
        ("value gap", ["value_gap", "value_gap_amount", "gap_amount", "gap_value", "variance", "difference"]),
        ("expected amount", ["expected_amount", "service_detected_amount", "detected_amount", "amount_due", "charge_amount"]),
        ("billed amount", ["billed_amount", "invoice_amount", "invoiced_amount"]),
        ("invoice amount", ["invoice_amount", "billed_amount", "invoiced_amount"]),
        ("total price", ["total_price", "price", "amount", "sales_amount", "revenue"]),
        ("price", ["total_price", "unit_price", "price", "amount", "sales_amount", "revenue"]),
        ("revenue", ["revenue", "sales_amount", "total_price", "amount", "price"]),
        ("sales", ["sales_amount", "revenue", "total_price", "amount", "price"]),
        ("total", ["total_price", "sales_amount", "revenue", "amount", "price"]),
        ("amount", numeric_candidates),
    ]
    matches: list[str] = []
    for token, candidates in token_mappings:
        if token not in normalized:
            continue
        for candidate in candidates:
            if candidate in numeric_candidates and candidate not in matches:
                matches.append(candidate)
                break
    return matches


def _prompt_mentions_phrase(prompt_text: str, phrase: str) -> bool:
    normalized_prompt = re.sub(r"[_\-=]+", " ", str(prompt_text or "").strip().lower())
    normalized_prompt = re.sub(r"\s+", " ", normalized_prompt)
    normalized_phrase = re.sub(r"[_\-=]+", " ", str(phrase or "").strip().lower())
    normalized_phrase = re.sub(r"\s+", " ", normalized_phrase)
    if not normalized_phrase:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_prompt))


def _extract_prompt_focuses(prompt_text: str) -> list[str]:
    mappings = [
        ("under_billed", ["under billed", "underbilled"]),
        ("billing_delay", ["billing delay", "delay", "lag"]),
        ("unbilled", ["unbilled", "unpaid", "open", "pending"]),
        ("overdue", ["overdue"]),
        ("billed", ["paid", "billed", "invoiced"]),
    ]
    matches: list[str] = []
    for focus, tokens in mappings:
        if focus == "billed":
            matched = any(
                _prompt_mentions_phrase(prompt_text, token) for token in ["paid", "invoiced"]
            ) or bool(re.search(r"(?<!under\s)\bbilled\b", re.sub(r"[_\-=]+", " ", str(prompt_text or "").strip().lower())))
        else:
            matched = any(_prompt_mentions_phrase(prompt_text, token) for token in tokens)
        if matched and focus not in matches:
            matches.append(focus)
    return matches


def _matched_prompt_dimension_columns(prompt_text: str, profile: DatasetProfile) -> list[str]:
    matches: list[tuple[int, str]] = []
    seen: set[str] = set()

    alias_candidates: list[tuple[list[str], str | None]] = [
        (["event type", "event_type", "exception type", "exception_type"], _pick_billing_focus_column(profile) or (profile.status_columns[0] if profile.status_columns else None)),
        (["invoice date", "invoice_date"], _pick_primary_billing_date_column(profile) if profile.date_columns else None),
    ]
    for aliases, column_name in alias_candidates:
        if not column_name or column_name in seen:
            continue
        positions = [match.start() for alias in aliases for match in [re.search(rf"(?<![a-z0-9]){re.escape(re.sub(r'[_\-=]+', ' ', alias.lower()).strip())}(?![a-z0-9])", re.sub(r"\s+", " ", re.sub(r"[_\-=]+", " ", str(prompt_text or "").strip().lower())))] if match]
        if positions:
            matches.append((min(positions), column_name))
            seen.add(column_name)

    candidate_columns = list(
        dict.fromkeys(
            [
                *profile.status_columns,
                *profile.product_columns,
                *profile.customer_columns,
                *profile.region_columns,
                *profile.date_columns,
                *profile.categorical_columns,
            ]
        )
    )
    for column_name in candidate_columns:
        if column_name in seen:
            continue
        label = _format_label(column_name).lower()
        normalized_name = str(column_name or "").replace("_", " ").lower()
        positions = []
        for phrase in [label, normalized_name]:
            if not phrase:
                continue
            match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", re.sub(r"\s+", " ", re.sub(r"[_\-=]+", " ", str(prompt_text or "").strip().lower())))
            if match:
                positions.append(match.start())
        if positions:
            matches.append((min(positions), column_name))
            seen.add(column_name)

    return [column_name for _position, column_name in sorted(matches, key=lambda item: (item[0], item[1]))]


def _pick_prompt_dimension_column(prompt_text: str, profile: DatasetProfile) -> str | None:
    matched_columns = _matched_prompt_dimension_columns(prompt_text, profile)
    return matched_columns[0] if matched_columns else None


def _prompt_uses_status_as_filter(prompt_text: str, profile: DatasetProfile, focus: str) -> bool:
    if not profile.status_columns or focus == "all":
        return False
    normalized_prompt = re.sub(r"[_-]+", " ", str(prompt_text or "").strip().lower())
    normalized_prompt = re.sub(r"\s+", " ", normalized_prompt)
    return any(
        re.search(pattern, normalized_prompt)
        for pattern in [
            r"\bstatus\s*=",
            r"\bstatus\s+is\b",
            r"\bexception type\s*=",
            r"\bexception type\s+is\b",
            r"\bevent type\s*=",
            r"\bevent type\s+is\b",
        ]
    )


def _pick_recommended_action_column(profile: DatasetProfile) -> str | None:
    preferred_names = [
        "recommended_action",
        "next_action",
        "recommended_next_action",
        "action_recommended",
        "action",
    ]
    for candidate in preferred_names:
        if candidate in profile.categorical_columns:
            return candidate

    for metadata in profile.column_metadata:
        normalized = str(metadata.column_name or "").strip().lower()
        if normalized == "next_action":
            return metadata.column_name
        if "recommend" in normalized and "action" in normalized:
            return metadata.column_name
        if normalized.endswith("_action"):
            return metadata.column_name
    return None


def _normalize_prompt_filter_value(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", normalized)


def _extract_prompt_value_filters(prompt_text: str, profile: DatasetProfile) -> list[dict[str, str]]:
    normalized_prompt = _normalize_prompt_filter_value(prompt_text)
    filters: list[dict[str, str]] = []

    recommended_action_column = _pick_recommended_action_column(profile)
    if recommended_action_column:
        pattern = r"(?:recommended\s+action|next\s+action|action)\s*(?:=|is|:)\s*([a-z0-9][a-z0-9\s/_-]*?)(?=\s+(?:by|per|grouped|group|split|across|compare|versus|vs|as|chart|graph)\b|$)"
        match = re.search(pattern, normalized_prompt, flags=re.IGNORECASE)
        if match:
            filter_value = _normalize_prompt_filter_value(match.group(1))
            if filter_value:
                filters.append({"column": recommended_action_column, "value": filter_value, "mode": "contains"})
        if not filters:
            for phrase in KNOWN_RECOMMENDED_ACTION_PHRASES:
                if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized_prompt):
                    filters.append({"column": recommended_action_column, "value": phrase, "mode": "contains"})
                    break

    return filters


def _apply_prompt_value_filters(filtered_df: pd.DataFrame, prompt_filters: list[dict[str, str]]) -> pd.DataFrame:
    working_df = filtered_df.copy()
    for prompt_filter in prompt_filters:
        column_name = str(prompt_filter.get("column") or "")
        filter_value = _normalize_prompt_filter_value(prompt_filter.get("value") or "")
        mode = str(prompt_filter.get("mode") or "contains").lower()
        if not column_name or column_name not in working_df.columns or not filter_value:
            continue
        normalized_series = working_df[column_name].astype("string").fillna("").map(_normalize_prompt_filter_value)
        if mode == "equals":
            mask = normalized_series.eq(filter_value)
        else:
            mask = normalized_series.str.contains(re.escape(filter_value), na=False)
        working_df = working_df.loc[mask].copy()
        if working_df.empty:
            break
    return working_df


def _build_prompt_entity_listing_frame(filtered_df: pd.DataFrame, profile: DatasetProfile, prompt_config: dict[str, Any]) -> pd.DataFrame | None:
    dimension = str(prompt_config.get("dimension") or "")
    if not dimension or dimension not in filtered_df.columns:
        return None

    ordered_columns: list[str] = []
    for column_name in [
        dimension,
        _pick_recommended_action_column(profile),
        _pick_billing_focus_column(profile),
        *profile.product_columns[:1],
        _pick_primary_billing_date_column(profile),
        _pick_primary_amount_column(profile),
        _pick_primary_invoice_column(profile),
    ]:
        if column_name and column_name in filtered_df.columns and column_name not in ordered_columns:
            ordered_columns.append(column_name)

    ordered_columns.extend([column_name for column_name in filtered_df.columns if column_name not in ordered_columns])
    listing_df = filtered_df[ordered_columns].copy() if ordered_columns else filtered_df.copy()
    listing_df = listing_df[listing_df[dimension].notna()].copy()
    if listing_df.empty:
        return None
    if dimension in profile.customer_columns or "customer" in dimension:
        listing_df = listing_df.drop_duplicates(subset=[dimension], keep="first")
    return listing_df.reset_index(drop=True) if not listing_df.empty else None


def _build_prompt_invoice_listing_frame(filtered_df: pd.DataFrame, profile: DatasetProfile, invoice_column: str) -> pd.DataFrame | None:
    ordered_columns = _pick_billing_relevant_columns(filtered_df, profile)
    listing_df = filtered_df[ordered_columns].copy() if ordered_columns else filtered_df.copy()
    if invoice_column and invoice_column in listing_df.columns:
        non_empty_mask = listing_df[invoice_column].astype("string").str.strip().ne("")
        if non_empty_mask.any():
            listing_df = listing_df.loc[non_empty_mask].copy()
    return listing_df.reset_index(drop=True) if not listing_df.empty else None


def _pick_primary_amount_column(profile: DatasetProfile) -> str | None:
    if profile.dataset_type == DATASET_BILLING:
        billing_priority = _preferred_billing_measure_columns(profile)
        if billing_priority:
            return billing_priority[0]
    if profile.amount_columns:
        return profile.amount_columns[0]
    return profile.numeric_columns[0] if profile.numeric_columns else None


def _pick_billing_gap_amount_column(profile: DatasetProfile) -> str | None:
    grouped = _classify_billing_measure_columns(profile)
    for group_name in ["gap", "expected", "billed", "other"]:
        if grouped[group_name]:
            return grouped[group_name][0]
    return _pick_primary_amount_column(profile)


def _pick_primary_dimension_column(profile: DatasetProfile, *, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    preferred = [
        *profile.region_columns,
        *profile.product_columns,
        *profile.customer_columns,
        *profile.categorical_columns,
    ]
    for column_name in preferred:
        if column_name not in exclude:
            return column_name
    return None


def _pick_billing_focus_column(profile: DatasetProfile) -> str | None:
    preferred_names = [
        "exception_type",
        "billing_exception",
        "invoice_status",
        "billing_status",
        "payment_status",
        "status",
    ]
    for candidate in preferred_names:
        if candidate in profile.status_columns:
            return candidate
    return profile.status_columns[0] if profile.status_columns else None


def _canonical_billing_focus_value(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    if normalized == "overdue":
        return "overdue"
    if "under" in normalized and "bill" in normalized:
        return "under_billed"
    if ("delay" in normalized and "bill" in normalized) or ("bill" in normalized and "lag" in normalized):
        return "billing_delay"
    if normalized in STATUS_POSITIVE_VALUES:
        return "billed"
    if normalized in STATUS_NEGATIVE_VALUES or "unbill" in normalized or "unpaid" in normalized:
        return "unbilled"
    return None


def _is_identifier_like_column(column_name: str) -> bool:
    normalized = str(column_name or "").strip().lower()
    identifier_tokens = {"id", "number", "no", "num", "code", "key", "ref", "reference", "uuid"}
    tokens = set(normalized.split("_"))
    if any(token in tokens for token in identifier_tokens):
        return True
    return any(part in normalized for part in ["invoice_id", "event_id", "customer_id", "record_id", "account_id", "bill_no"])


def _prepare_display_frame(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    display_df = df.copy()
    text_like_columns = set(
        profile.categorical_columns
        + profile.customer_columns
        + profile.product_columns
        + profile.region_columns
        + profile.status_columns
    )
    for column_name in list(display_df.columns):
        if column_name in text_like_columns or _is_identifier_like_column(column_name):
            series = display_df[column_name]
            display_df[column_name] = series.astype("string").replace({pd.NA: ""}).fillna("")
    return display_df


def _normalize_invoice_identifier(value: Any) -> str:
    text = _clean_text_value(value)
    if not text:
        return ""
    if re.fullmatch(r"-\d+", text):
        return f"INV{text}"
    if re.fullmatch(r"\d+", text):
        return f"INV-{text}"
    return text


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _has_alpha_content(value: Any) -> bool:
    return bool(re.search(r"[A-Za-z]", _clean_text_value(value)))


def _event_customer_lookup(sheet_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for frame in sheet_frames.values():
        cleaned = clean_column_names(frame)
        if "event_id" not in cleaned.columns or "customer" not in cleaned.columns:
            continue
        for _, row in cleaned[["event_id", "customer"]].dropna(how="all").iterrows():
            event_id = _clean_text_value(row.get("event_id"))
            customer = _clean_text_value(row.get("customer"))
            if not event_id or not customer or not _has_alpha_content(customer):
                continue
            lookup.setdefault(event_id, customer)
    return lookup


def _enrich_selected_sheet_frame(
    source_df: pd.DataFrame,
    selected_sheet: str,
    sheet_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    enriched_df = clean_column_names(source_df).copy()

    if "invoice_id" in enriched_df.columns:
        enriched_df["invoice_id"] = enriched_df["invoice_id"].map(_normalize_invoice_identifier)

    if selected_sheet and str(selected_sheet).strip().lower() in {"agent_output", "invoices_synthetic"}:
        if "customer" in enriched_df.columns and "event_id" in enriched_df.columns:
            current_customers = enriched_df["customer"].map(_clean_text_value)
            rich_ratio = float(current_customers.map(_has_alpha_content).mean()) if len(current_customers) else 0.0
            if rich_ratio < 0.3:
                lookup = _event_customer_lookup(sheet_frames)
                if lookup:
                    event_ids = enriched_df["event_id"].map(_clean_text_value)
                    richer_customers = event_ids.map(lookup)
                    replace_mask = current_customers.ne("") & ~current_customers.map(_has_alpha_content) & richer_customers.notna()
                    enriched_df.loc[replace_mask, "customer"] = richer_customers.loc[replace_mask]

    return enriched_df


def _is_temporal_chart_x(chart: ChartSpec) -> bool:
    if chart.x_column not in chart.dataframe.columns:
        return False
    return pd.api.types.is_datetime64_any_dtype(chart.dataframe[chart.x_column])


def _prepare_chart_dataframe_for_render(chart: ChartSpec, selected_chart_type: str) -> tuple[pd.DataFrame, str, bool]:
    render_df = chart.dataframe.copy()
    x_column = chart.x_column
    x_is_temporal = _is_temporal_chart_x(chart)
    if not x_is_temporal:
        return render_df, x_column, False

    if selected_chart_type == "line":
        return render_df, x_column, True

    render_x_column = f"{x_column}_label"
    series = pd.to_datetime(render_df[x_column], errors="coerce")
    if series.notna().all() and series.dt.day.eq(1).all() and series.dt.hour.eq(0).all() and series.dt.minute.eq(0).all():
        render_df[render_x_column] = series.dt.strftime("%Y-%m")
    else:
        render_df[render_x_column] = series.dt.strftime("%Y-%m-%d")
    return render_df, render_x_column, False


def _manual_dashboard_dimension_options(filtered_df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    options: list[str] = []
    candidates = list(
        dict.fromkeys(
            [
                *profile.date_columns,
                *profile.product_columns,
                *profile.customer_columns,
                *profile.region_columns,
                *profile.status_columns,
                *profile.categorical_columns,
            ]
        )
    )
    for column_name in candidates:
        if column_name not in filtered_df.columns:
            continue
        series = filtered_df[column_name]
        distinct_count = int(series.dropna().nunique())
        if distinct_count < 2:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            options.append(column_name)
            continue
        if _is_identifier_like_column(column_name):
            continue
        if distinct_count > 30:
            continue
        options.append(column_name)
    return options


def _manual_dashboard_measure_options(filtered_df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    options: list[str] = []
    candidates = list(
        dict.fromkeys(
            [
                *profile.amount_columns,
                *profile.quantity_columns,
                *profile.numeric_columns,
            ]
        )
    )
    for column_name in candidates:
        if column_name not in filtered_df.columns:
            continue
        numeric_series = pd.to_numeric(filtered_df[column_name], errors="coerce")
        if numeric_series.notna().any() and column_name not in options:
            options.append(column_name)
    options.append("__row_count")
    return options


def _manual_dashboard_aggregation_options(filtered_df: pd.DataFrame, y_column: str) -> list[str]:
    if y_column == "__row_count":
        return ["count"]
    if y_column not in filtered_df.columns:
        return ["count"]
    numeric_series = pd.to_numeric(filtered_df[y_column], errors="coerce")
    if numeric_series.notna().any():
        return ["sum", "avg", "count"]
    return ["count"]


def _build_manual_dashboard_chart_spec(
    filtered_df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    x_column: str,
    y_column: str,
    aggregation: str,
    chart_type: str,
    split_column: str | None = None,
    top_n: int | None = 12,
) -> ChartSpec | None:
    if filtered_df.empty or x_column not in filtered_df.columns:
        return None

    working_df = filtered_df.copy()
    value_column = y_column
    if y_column == "__row_count":
        working_df[value_column] = 1
    elif y_column in working_df.columns:
        working_df[value_column] = pd.to_numeric(working_df[value_column], errors="coerce")
    else:
        return None

    group_columns = [x_column]
    if split_column and split_column in working_df.columns and split_column != x_column:
        group_columns.append(split_column)
    else:
        split_column = None

    working_df = working_df.loc[working_df[x_column].notna()].copy()
    if split_column:
        working_df = working_df.loc[working_df[split_column].notna()].copy()
    if working_df.empty:
        return None

    count_mode = aggregation == "count" or value_column == "__row_count"
    if count_mode:
        grouped = working_df.groupby(group_columns, dropna=False).size().reset_index(name="metric_value")
        y_label = "Count"
    else:
        aggregate_name = "mean" if aggregation == "avg" else "sum"
        grouped = (
            working_df.groupby(group_columns, dropna=False)[value_column]
            .agg(aggregate_name)
            .reset_index(name="metric_value")
        )
        grouped = grouped[grouped["metric_value"].notna()].copy()
        y_label = _metric_value_label(value_column)
    if grouped.empty:
        return None

    x_is_temporal = pd.api.types.is_datetime64_any_dtype(grouped[x_column])
    if split_column:
        totals = grouped.groupby(x_column, dropna=False)["metric_value"].sum().reset_index(name="category_total")
        grouped = grouped.merge(totals, on=x_column, how="left")
        if x_is_temporal:
            grouped = grouped.sort_values([x_column, split_column], ascending=[True, True]).reset_index(drop=True)
        else:
            grouped = grouped.sort_values(["category_total", "metric_value", x_column], ascending=[False, False, True])
            if top_n is not None and top_n > 0:
                allowed = totals.sort_values(["category_total", x_column], ascending=[False, True]).head(top_n)[x_column]
                grouped = grouped[grouped[x_column].isin(allowed)].copy()
            grouped = grouped.reset_index(drop=True)
        grouped = grouped.drop(columns=["category_total"], errors="ignore")
    elif x_is_temporal:
        grouped = grouped.sort_values(x_column, ascending=True).reset_index(drop=True)
    else:
        grouped = grouped.sort_values(["metric_value", x_column], ascending=[False, True]).reset_index(drop=True)
        if top_n is not None and top_n > 0:
            grouped = grouped.head(top_n).reset_index(drop=True)

    if count_mode:
        title = f"Count by {_format_label(x_column)}"
    elif aggregation == "avg":
        title = f"Average {_format_label(y_column)} by {_format_label(x_column)}"
    else:
        title = f"{_format_label(y_column)} by {_format_label(x_column)}"
    if split_column:
        title = f"{title} and {_format_label(split_column)}"

    if split_column:
        supported_chart_types = COMPARISON_CHART_TYPES
    elif x_is_temporal:
        supported_chart_types = TREND_CHART_TYPES
    else:
        supported_chart_types = SIMPLE_CHART_TYPES

    selected_chart_type = chart_type if chart_type in supported_chart_types else supported_chart_types[0]
    description = f"Manual chart using {y_label.lower()} by {_format_label(x_column).lower()}"
    if split_column:
        description = f"{description} split by {_format_label(split_column).lower()}"

    return ChartSpec(
        title=title,
        chart_type=selected_chart_type,
        dataframe=grouped,
        x_column=x_column,
        y_column="metric_value",
        color_column=split_column,
        description=description,
        supported_chart_types=supported_chart_types,
        x_label=_format_label(x_column),
        y_label=y_label,
    )


def _render_chart_spec(chart: ChartSpec, *, control_key: str) -> None:
    import streamlit as st

    st.caption(chart.description or chart.title)
    chart_options = chart.supported_chart_types or [chart.chart_type]
    selected_chart_type = st.selectbox(
        f"Display {chart.title} as",
        options=chart_options,
        index=chart_options.index(chart.chart_type) if chart.chart_type in chart_options else 0,
        key=f"{control_key}_mode",
        format_func=lambda value: str(CHART_TYPE_LABELS.get(str(value), str(value).replace("_", " ").title())),
    )
    if alt is not None:
        render_df, render_x_column, x_is_temporal = _prepare_chart_dataframe_for_render(chart, selected_chart_type)
        color_tooltip = [alt.Tooltip(f"{chart.color_column}:N", title=_format_label(chart.color_column))] if chart.color_column else []
        if selected_chart_type == "line":
            chart_obj = (
                alt.Chart(render_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{render_x_column}:T", title=chart.x_label or _format_label(chart.x_column)),
                    y=alt.Y(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:T", title=chart.x_label or _format_label(chart.x_column)),
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        elif selected_chart_type == "area":
            chart_obj = (
                alt.Chart(render_df)
                .mark_area(opacity=0.65)
                .encode(
                    x=alt.X(f"{render_x_column}:T", title=chart.x_label or _format_label(chart.x_column)),
                    y=alt.Y(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:T", title=chart.x_label or _format_label(chart.x_column)),
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        elif selected_chart_type == "pie":
            chart_obj = (
                alt.Chart(render_df)
                .mark_arc()
                .encode(
                    theta=alt.Theta(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    color=alt.Color(f"{render_x_column}:N", title=chart.x_label or _format_label(chart.x_column)),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:N", title=chart.x_label or _format_label(chart.x_column)),
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        elif selected_chart_type == "grouped_bar" and chart.color_column:
            chart_obj = (
                alt.Chart(render_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{render_x_column}:{'T' if x_is_temporal else 'N'}",
                        title=chart.x_label or _format_label(chart.x_column),
                        sort=None if x_is_temporal else "-y",
                    ),
                    xOffset=alt.XOffset(f"{chart.color_column}:N"),
                    y=alt.Y(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    color=alt.Color(f"{chart.color_column}:N", title=_format_label(chart.color_column)),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:{'T' if x_is_temporal else 'N'}", title=chart.x_label or _format_label(chart.x_column)),
                        alt.Tooltip(f"{chart.color_column}:N", title=_format_label(chart.color_column)),
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        elif selected_chart_type == "stacked_bar" and chart.color_column:
            chart_obj = (
                alt.Chart(render_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{render_x_column}:{'T' if x_is_temporal else 'N'}",
                        title=chart.x_label or _format_label(chart.x_column),
                        sort=None if x_is_temporal else "-y",
                    ),
                    y=alt.Y(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    color=alt.Color(f"{chart.color_column}:N", title=_format_label(chart.color_column)),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:{'T' if x_is_temporal else 'N'}", title=chart.x_label or _format_label(chart.x_column)),
                        alt.Tooltip(f"{chart.color_column}:N", title=_format_label(chart.color_column)),
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        else:
            chart_obj = (
                alt.Chart(render_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{render_x_column}:{'T' if x_is_temporal else 'N'}",
                        title=chart.x_label or _format_label(chart.x_column),
                        sort=None if x_is_temporal else "-y",
                    ),
                    y=alt.Y(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column)),
                    color=alt.Color(f"{chart.color_column}:N", title=_format_label(chart.color_column)) if chart.color_column else alt.value("#1f77b4"),
                    tooltip=[
                        alt.Tooltip(f"{render_x_column}:{'T' if x_is_temporal else 'N'}", title=chart.x_label or _format_label(chart.x_column)),
                        *color_tooltip,
                        alt.Tooltip(f"{chart.y_column}:Q", title=chart.y_label or _format_label(chart.y_column), format=",.2f"),
                    ],
                )
                .properties(title=chart.title, height=320)
            )
        st.altair_chart(chart_obj, use_container_width=True)
    elif selected_chart_type in {"line", "area"}:
        st.line_chart(chart.dataframe.set_index(chart.x_column)[chart.y_column])
    else:
        st.bar_chart(chart.dataframe.set_index(chart.x_column)[chart.y_column])


def _has_exception_billing_shape(profile: DatasetProfile) -> bool:
    if profile.dataset_type != DATASET_BILLING:
        return False
    grouped = _classify_billing_measure_columns(profile)
    present_groups = [group_name for group_name in ["expected", "billed", "gap"] if grouped[group_name]]
    return "exception_type" in profile.status_columns and len(present_groups) >= 2


def _pick_billing_relevant_columns(working_df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    invoice_column = _pick_primary_invoice_column(profile)
    date_column = _pick_primary_billing_date_column(profile)
    status_column = _pick_billing_focus_column(profile) or (profile.status_columns[0] if profile.status_columns else None)
    grouped = _classify_billing_measure_columns(profile)
    key_measure_columns: list[str] = []
    for group_name in ["expected", "billed", "gap"]:
        if grouped[group_name]:
            key_measure_columns.append(grouped[group_name][0])

    if _has_exception_billing_shape(profile):
        preferred = [
            status_column,
            "event_id",
            *profile.customer_columns[:1],
            *profile.product_columns[:1],
            date_column,
            *key_measure_columns,
            invoice_column,
            "aging_days",
            "aging_bucket",
        ]
    else:
        preferred = [
            invoice_column,
            date_column,
            *profile.customer_columns[:1],
            status_column,
            _pick_primary_amount_column(profile),
            "aging_days",
            "aging_bucket",
        ]

    preferred_columns = []
    for column_name in preferred:
        if column_name and column_name in working_df.columns and column_name not in preferred_columns:
            preferred_columns.append(column_name)
    remaining_columns = [column_name for column_name in working_df.columns if column_name not in preferred_columns]
    return preferred_columns + remaining_columns


def _build_dataframe_column_config(df: pd.DataFrame, profile: DatasetProfile) -> dict[str, Any]:
    import streamlit as st

    column_config: dict[str, Any] = {}
    for column_name in df.columns:
        label = _format_label(column_name)
        if _is_identifier_like_column(column_name) or column_name in profile.customer_columns:
            column_config[column_name] = st.column_config.TextColumn(label, width="large")
        elif column_name in profile.status_columns or column_name in profile.product_columns or column_name in profile.region_columns:
            column_config[column_name] = st.column_config.TextColumn(label, width="medium")
        elif column_name in profile.date_columns:
            column_config[column_name] = st.column_config.DatetimeColumn(label, width="medium", format="YYYY-MM-DD")
        elif column_name in profile.amount_columns or column_name == "metric_value":
            column_config[column_name] = st.column_config.NumberColumn(label, width="medium", format="%.2f")
    return column_config


def _is_chartable_dimension(filtered_df: pd.DataFrame, column_name: str) -> bool:
    if column_name not in filtered_df.columns or _is_identifier_like_column(column_name):
        return False

    distinct_count = int(filtered_df[column_name].dropna().nunique())
    if distinct_count < 2:
        return False

    row_count = max(len(filtered_df), 1)
    unique_ratio = distinct_count / float(row_count)
    if distinct_count > 20:
        return False
    if distinct_count > 12 and unique_ratio > 0.4:
        return False
    return True


def _pick_meaningful_dimension_column(
    filtered_df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    if profile.dataset_type == DATASET_BILLING:
        preferred = [
            *profile.product_columns,
            *profile.customer_columns,
            *profile.region_columns,
            *profile.categorical_columns,
        ]
    else:
        preferred = [
            *profile.region_columns,
            *profile.product_columns,
            *profile.customer_columns,
            *profile.categorical_columns,
        ]

    for column_name in preferred:
        if column_name in exclude:
            continue
        if _is_chartable_dimension(filtered_df, column_name):
            return column_name
    return None


def _pick_prompt_fallback_dimension_column(
    filtered_df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    preferred = [
        *profile.product_columns,
        *profile.customer_columns,
        *profile.region_columns,
        *profile.categorical_columns,
    ]
    for column_name in preferred:
        if column_name in exclude or column_name not in filtered_df.columns:
            continue
        if column_name in profile.status_columns or _is_identifier_like_column(column_name):
            continue
        if int(filtered_df[column_name].dropna().nunique()) >= 2:
            return column_name
    return None


def _pick_primary_invoice_column(profile: DatasetProfile) -> str | None:
    preferred_names = [
        "invoice_id",
        "invoice_number",
        "invoice_no",
        "invoice_num",
        "bill_number",
        "bill_no",
    ]
    for candidate in preferred_names:
        for column_name in profile.categorical_columns:
            if column_name == candidate:
                return column_name

    for metadata in profile.column_metadata:
        normalized = str(metadata.column_name or "")
        if "invoice" not in normalized and "bill" not in normalized:
            continue
        if metadata.detected_type == "date" or metadata.semantic_role in {"amount", "status", "date"}:
            continue
        return metadata.column_name
    return None


def _count_entities(filtered_df: pd.DataFrame, column_name: str | None, mask: pd.Series) -> int:
    scoped_df = filtered_df.loc[mask]
    if column_name and column_name in scoped_df.columns:
        entity_series = scoped_df[column_name]
        non_empty_mask = entity_series.notna() & entity_series.astype("string").str.strip().ne("")
        non_empty_count = int(entity_series.loc[non_empty_mask].nunique())
        blank_count = int((~non_empty_mask).sum())
        return non_empty_count + blank_count
    return int(mask.sum())


def _pick_primary_billing_date_column(profile: DatasetProfile) -> str | None:
    preferred_names = ["due_date", "payment_due_date", "invoice_date", "billing_date", "date"]
    for candidate in preferred_names:
        if candidate in profile.date_columns:
            return candidate
    for column_name in profile.date_columns:
        if "due" in column_name or "invoice" in column_name or "bill" in column_name:
            return column_name
    return profile.date_columns[0] if profile.date_columns else None


def _billing_status_masks(filtered_df: pd.DataFrame, profile: DatasetProfile) -> dict[str, pd.Series]:
    empty_mask = pd.Series(True, index=filtered_df.index)
    false_mask = pd.Series(False, index=filtered_df.index)
    if not profile.status_columns:
        return {"all": empty_mask}

    status_column = _pick_billing_focus_column(profile)
    if not status_column or status_column not in filtered_df.columns:
        return {"all": empty_mask}

    canonical_values = filtered_df[status_column].map(_canonical_billing_focus_value)
    overdue_mask = canonical_values.eq("overdue")
    masks = {
        "all": empty_mask,
        "unbilled": canonical_values.eq("unbilled") | overdue_mask,
        "under_billed": canonical_values.eq("under_billed"),
        "billing_delay": canonical_values.eq("billing_delay"),
        "overdue": overdue_mask,
        "billed": canonical_values.eq("billed"),
    }

    if not any(mask.any() for focus, mask in masks.items() if focus != "all"):
        normalized_status = filtered_df[status_column].astype("string").str.strip().str.lower()
        masks["unbilled"] = normalized_status.isin(STATUS_NEGATIVE_VALUES)
        masks["billed"] = normalized_status.isin(STATUS_POSITIVE_VALUES)
        masks["overdue"] = normalized_status.eq("overdue")
        masks["under_billed"] = false_mask
        masks["billing_delay"] = false_mask
    return masks


def _available_billing_focuses(filtered_df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    masks = _billing_status_masks(filtered_df, profile)
    ordered = ["all", "unbilled", "under_billed", "billing_delay", "overdue", "billed"]
    options = [focus for focus in ordered if focus == "all" or masks.get(focus, pd.Series(False, index=filtered_df.index)).any()]
    return options or ["all"]


def _attach_billing_aging_columns(filtered_df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    working_df = filtered_df.copy()
    if profile.dataset_type != DATASET_BILLING:
        return working_df

    date_column = _pick_primary_billing_date_column(profile)
    if not date_column or date_column not in working_df.columns:
        return working_df

    anchor_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    aging_dates = pd.to_datetime(working_df[date_column], errors="coerce")
    aging_days = (anchor_date - aging_dates).dt.days
    working_df["aging_days"] = aging_days
    working_df["aging_bucket"] = pd.cut(
        aging_days.clip(lower=0),
        bins=[-1, 30, 60, 90, float("inf")],
        labels=["0-30", "31-60", "61-90", "90+"],
    ).astype("string")
    return working_df


def _filter_billing_focus(filtered_df: pd.DataFrame, profile: DatasetProfile, focus: str) -> pd.DataFrame:
    masks = _billing_status_masks(filtered_df, profile)
    selected_mask = masks.get(str(focus or "all").lower(), masks["all"])
    return filtered_df.loc[selected_mask].copy()


def _build_billing_aging_chart_spec(filtered_df: pd.DataFrame, profile: DatasetProfile) -> ChartSpec | None:
    if profile.dataset_type != DATASET_BILLING:
        return None

    aging_df = _attach_billing_aging_columns(filtered_df, profile)
    if "aging_bucket" not in aging_df.columns:
        return None

    scoped_df = aging_df[aging_df["aging_bucket"].notna()].copy()
    if scoped_df.empty:
        return None

    amount_column = _pick_primary_amount_column(profile)
    exposure_amount_column = _pick_billing_gap_amount_column(profile) if profile.dataset_type == DATASET_BILLING else amount_column
    if amount_column and amount_column in scoped_df.columns:
        grouped = (
            scoped_df.groupby("aging_bucket", dropna=False)[amount_column]
            .sum()
            .reset_index(name="metric_value")
        )
    else:
        grouped = scoped_df.groupby("aging_bucket", dropna=False).size().reset_index(name="metric_value")

    grouped["aging_bucket"] = pd.Categorical(grouped["aging_bucket"], categories=["0-30", "31-60", "61-90", "90+"], ordered=True)
    grouped = grouped.sort_values("aging_bucket").rename(columns={"aging_bucket": "category"}).reset_index(drop=True)
    return ChartSpec(
        title="Aging bucket breakdown",
        chart_type="bar",
        dataframe=grouped,
        x_column="category",
        y_column="metric_value",
        description="Invoice aging distribution",
        supported_chart_types=SIMPLE_CHART_TYPES,
        x_label="Aging Bucket",
        y_label=_metric_value_label(amount_column, count_fallback=not bool(amount_column and amount_column in scoped_df.columns)),
    )


def _parse_dashboard_prompt(prompt: str, profile: DatasetProfile) -> dict[str, Any]:
    prompt_text = str(prompt or "").strip().lower()
    explicit_time_intent = any(token in prompt_text for token in ["month", "date", "trend", "time", "daily", "weekly", "monthly"])
    comparison_intent = any(token in prompt_text for token in ["compare", " versus ", " vs ", "side by side"])
    preferred_chart = "bar"
    if "pie" in prompt_text:
        preferred_chart = "pie"
    elif "line" in prompt_text or "trend" in prompt_text:
        preferred_chart = "line"
    elif "grouped" in prompt_text:
        preferred_chart = "grouped_bar"
    elif "stacked" in prompt_text:
        preferred_chart = "stacked_bar"

    prompt_measure_columns = _extract_prompt_measure_columns(prompt_text, profile)
    explicit_measure_column = prompt_measure_columns[0] if prompt_measure_columns else _pick_prompt_measure_column(prompt_text, profile)
    measure = "amount" if explicit_measure_column or any(token in prompt_text for token in ["amount", "value", "revenue", "total", "gap"]) else "count"
    prompt_focuses = _extract_prompt_focuses(prompt_text)
    matched_dimensions = _matched_prompt_dimension_columns(prompt_text, profile)
    focus = prompt_focuses[0] if prompt_focuses else "all"
    invoice_column = _pick_primary_invoice_column(profile)
    prompt_filters = _extract_prompt_value_filters(prompt_text, profile)
    invoice_listing = bool(
        re.search(r"\binvoices?\b", prompt_text)
        and not any(token in prompt_text for token in [" by ", " per ", " grouped", " group by", " split by", " across", " compare", " versus ", " vs ", " pie", " bar", " line", " trend"])
    )
    customer_listing = bool(
        re.search(r"\bcustomers?\b", prompt_text)
        and not any(token in prompt_text for token in [" by ", " per ", " grouped", " group by", " split by", " across", " compare", " versus ", " vs ", " pie", " bar", " line", " trend"])
    )

    comparison_kind = "none"
    if comparison_intent and len(prompt_measure_columns) >= 2:
        comparison_kind = "measure"
        if preferred_chart == "bar":
            preferred_chart = "grouped_bar"
    elif comparison_intent and len(prompt_focuses) >= 2:
        comparison_kind = "focus"
        if preferred_chart == "bar":
            preferred_chart = "grouped_bar"
    elif any(token in prompt_text for token in ["grouped by", "split by", "across"]) and len(matched_dimensions) >= 2:
        comparison_kind = "dimension"
        if preferred_chart == "bar":
            preferred_chart = "grouped_bar"

    dimension = matched_dimensions[0] if matched_dimensions else _pick_prompt_dimension_column(prompt_text, profile)
    comparison_dimension = matched_dimensions[1] if len(matched_dimensions) >= 2 else None
    status_filter_request = _prompt_uses_status_as_filter(prompt_text, profile, focus)
    if status_filter_request:
        dimension = None
    if "aging" in prompt_text or "bucket" in prompt_text:
        dimension = "aging_bucket"
    elif dimension:
        pass
    elif "customer" in prompt_text and profile.customer_columns:
        dimension = profile.customer_columns[0]
    elif not status_filter_request and any(token in prompt_text for token in ["status", "exception", "exception type", "event type"]) and profile.status_columns:
        dimension = profile.status_columns[0]
    elif "region" in prompt_text and profile.region_columns:
        dimension = profile.region_columns[0]
    elif any(token in prompt_text for token in ["product", "service", "service type", "category"]) and profile.product_columns:
        dimension = profile.product_columns[0]
    elif explicit_time_intent and profile.date_columns:
        dimension = _pick_primary_billing_date_column(profile) if profile.dataset_type == DATASET_BILLING else profile.date_columns[0]
    else:
        if profile.dataset_type == DATASET_BILLING and profile.product_columns:
            dimension = profile.product_columns[0]
        else:
            dimension = _pick_primary_dimension_column(profile) or (_pick_primary_billing_date_column(profile) if profile.date_columns else None)

    return {
        "focus": focus,
        "focuses": prompt_focuses,
        "filters": prompt_filters,
        "dimension": dimension,
        "invoice_column": invoice_column,
        "invoice_listing": invoice_listing,
        "customer_listing": customer_listing,
        "measure": measure,
        "measure_column": explicit_measure_column,
        "measure_columns": prompt_measure_columns,
        "chart_type": preferred_chart,
        "time_intent": explicit_time_intent,
        "comparison_kind": comparison_kind,
        "comparison_dimension": comparison_dimension,
    }


def build_prompt_chart_spec(filtered_df: pd.DataFrame, profile: DatasetProfile, prompt: str) -> ChartSpec | None:
    prompt_config = _parse_dashboard_prompt(prompt, profile)
    if prompt_config.get("invoice_listing") or prompt_config.get("customer_listing"):
        return None
    scoped_df = _filter_billing_focus(filtered_df, profile, prompt_config["focus"])
    scoped_df = _apply_prompt_value_filters(scoped_df, prompt_config.get("filters", []))
    scoped_df = _attach_billing_aging_columns(scoped_df, profile)
    if scoped_df.empty:
        return None

    dimension = prompt_config["dimension"]
    if dimension in profile.date_columns and not prompt_config.get("time_intent"):
        dimension = (
            _pick_meaningful_dimension_column(scoped_df, profile, exclude=set(profile.status_columns))
            or _pick_prompt_fallback_dimension_column(scoped_df, profile, exclude=set(profile.status_columns))
            or dimension
        )
    if not dimension or dimension not in scoped_df.columns:
        return None

    if prompt_config.get("comparison_kind") == "measure":
        working_df = scoped_df[scoped_df[dimension].notna()].copy()
        if working_df.empty:
            return None
        frames: list[pd.DataFrame] = []
        for measure_column in prompt_config.get("measure_columns", []):
            if measure_column not in working_df.columns:
                continue
            grouped = working_df.groupby(dimension, dropna=False)[measure_column].sum().reset_index(name="metric_value")
            grouped["comparison_value"] = _format_label(measure_column)
            frames.append(grouped)
        if not frames:
            return None
        comparison_df = pd.concat(frames, ignore_index=True)
        comparison_df = comparison_df.rename(columns={dimension: "category"})
        totals = comparison_df.groupby("category", dropna=False)["metric_value"].sum().reset_index(name="category_total")
        comparison_df = comparison_df.merge(totals, on="category", how="left")
        comparison_df = comparison_df.sort_values(["category_total", "metric_value"], ascending=[False, False]).drop(columns=["category_total"]).reset_index(drop=True)
        top_categories = comparison_df["category"].drop_duplicates().head(12).tolist()
        comparison_df = comparison_df[comparison_df["category"].isin(top_categories)].reset_index(drop=True)
        return ChartSpec(
            title=f"Prompt view: {prompt.strip()}",
            chart_type=prompt_config["chart_type"],
            dataframe=comparison_df,
            x_column="category",
            y_column="metric_value",
            color_column="comparison_value",
            description="Prompt-driven comparison view",
            supported_chart_types=COMPARISON_CHART_TYPES,
            x_label=_format_label(dimension),
            y_label="Amount",
        )

    if prompt_config.get("comparison_kind") == "focus":
        frames = []
        base_df = _attach_billing_aging_columns(filtered_df, profile)
        amount_column = prompt_config.get("measure_column") or _pick_primary_amount_column(profile)
        use_amount = prompt_config["measure"] == "amount" and bool(amount_column and amount_column in base_df.columns)
        for focus_value in prompt_config.get("focuses", []):
            focus_df = _filter_billing_focus(base_df, profile, focus_value)
            focus_df = focus_df[focus_df[dimension].notna()].copy()
            if focus_df.empty:
                continue
            if use_amount and amount_column:
                grouped = focus_df.groupby(dimension, dropna=False)[amount_column].sum().reset_index(name="metric_value")
            else:
                grouped = focus_df.groupby(dimension, dropna=False).size().reset_index(name="metric_value")
            grouped["comparison_value"] = BILLING_FOCUS_LABELS.get(focus_value, focus_value.replace("_", " ").title())
            frames.append(grouped)
        if not frames:
            return None
        comparison_df = pd.concat(frames, ignore_index=True)
        comparison_df = comparison_df.rename(columns={dimension: "category"})
        totals = comparison_df.groupby("category", dropna=False)["metric_value"].sum().reset_index(name="category_total")
        comparison_df = comparison_df.merge(totals, on="category", how="left")
        comparison_df = comparison_df.sort_values(["category_total", "metric_value"], ascending=[False, False]).drop(columns=["category_total"]).reset_index(drop=True)
        top_categories = comparison_df["category"].drop_duplicates().head(12).tolist()
        comparison_df = comparison_df[comparison_df["category"].isin(top_categories)].reset_index(drop=True)
        return ChartSpec(
            title=f"Prompt view: {prompt.strip()}",
            chart_type=prompt_config["chart_type"],
            dataframe=comparison_df,
            x_column="category",
            y_column="metric_value",
            color_column="comparison_value",
            description="Prompt-driven focus comparison view",
            supported_chart_types=COMPARISON_CHART_TYPES,
            x_label=_format_label(dimension),
            y_label=_metric_value_label(amount_column, count_fallback=not use_amount),
        )

    if prompt_config.get("comparison_kind") == "dimension":
        comparison_dimension = prompt_config.get("comparison_dimension")
        amount_column = prompt_config.get("measure_column") or _pick_primary_amount_column(profile)
        working_df = scoped_df.copy()
        if not comparison_dimension or comparison_dimension not in working_df.columns:
            return None
        working_df = working_df[working_df[dimension].notna() & working_df[comparison_dimension].notna()].copy()
        if working_df.empty:
            return None
        if prompt_config["measure"] == "amount" and amount_column and amount_column in working_df.columns:
            grouped = working_df.groupby([dimension, comparison_dimension], dropna=False)[amount_column].sum().reset_index(name="metric_value")
        else:
            grouped = working_df.groupby([dimension, comparison_dimension], dropna=False).size().reset_index(name="metric_value")
        grouped = grouped.rename(columns={dimension: "category", comparison_dimension: "comparison_value"})
        totals = grouped.groupby("category", dropna=False)["metric_value"].sum().reset_index(name="category_total")
        grouped = grouped.merge(totals, on="category", how="left")
        grouped = grouped.sort_values(["category_total", "metric_value"], ascending=[False, False]).drop(columns=["category_total"]).reset_index(drop=True)
        top_categories = grouped["category"].drop_duplicates().head(12).tolist()
        grouped = grouped[grouped["category"].isin(top_categories)].reset_index(drop=True)
        return ChartSpec(
            title=f"Prompt view: {prompt.strip()}",
            chart_type=prompt_config["chart_type"],
            dataframe=grouped,
            x_column="category",
            y_column="metric_value",
            color_column="comparison_value",
            description="Prompt-driven grouped comparison view",
            supported_chart_types=COMPARISON_CHART_TYPES,
            x_label=_format_label(dimension),
            y_label=_metric_value_label(amount_column, count_fallback=prompt_config["measure"] != "amount" or not (amount_column and amount_column in working_df.columns)),
        )

    amount_column = prompt_config.get("measure_column") or _pick_primary_amount_column(profile)
    if pd.api.types.is_datetime64_any_dtype(scoped_df[dimension]):
        granularity = "month" if scoped_df[dimension].nunique(dropna=True) > 45 else "day"
        period = scoped_df[dimension].dt.to_period("M").dt.to_timestamp() if granularity == "month" else scoped_df[dimension].dt.normalize()
        metric_series = scoped_df[amount_column] if prompt_config["measure"] == "amount" and amount_column in scoped_df.columns else pd.Series(1, index=scoped_df.index)
        grouped = pd.DataFrame({"period": period, "metric_value": metric_series}).groupby("period", dropna=False)["metric_value"].sum().reset_index()
        if grouped.empty:
            return None
        return ChartSpec(
            title=f"Prompt view: {prompt.strip()}",
            chart_type="line" if prompt_config["chart_type"] == "line" else "bar",
            dataframe=grouped,
            x_column="period",
            y_column="metric_value",
            description="Prompt-driven trend view",
            supported_chart_types=TREND_CHART_TYPES,
            x_label=_format_label(dimension),
            y_label=_metric_value_label(amount_column, count_fallback=prompt_config["measure"] != "amount" or not (amount_column and amount_column in scoped_df.columns)),
        )

    working_df = scoped_df[scoped_df[dimension].notna()].copy()
    if working_df.empty:
        return None
    if prompt_config["measure"] == "amount" and amount_column and amount_column in working_df.columns:
        grouped = working_df.groupby(dimension, dropna=False)[amount_column].sum().reset_index(name="metric_value")
    else:
        grouped = working_df.groupby(dimension, dropna=False).size().reset_index(name="metric_value")
    grouped = grouped.rename(columns={dimension: "category"}).sort_values("metric_value", ascending=False).head(12).reset_index(drop=True)
    return ChartSpec(
        title=f"Prompt view: {prompt.strip()}",
        chart_type=prompt_config["chart_type"],
        dataframe=grouped,
        x_column="category",
        y_column="metric_value",
        description="Prompt-driven metric view",
        supported_chart_types=SIMPLE_CHART_TYPES,
        x_label=_format_label(dimension),
        y_label=_metric_value_label(amount_column, count_fallback=prompt_config["measure"] != "amount" or not (amount_column and amount_column in working_df.columns)),
    )


def build_billing_drilldown_frame(filtered_df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    working_df = _attach_billing_aging_columns(filtered_df, profile)
    ordered_columns = _pick_billing_relevant_columns(working_df, profile)
    preview_columns = ordered_columns[:10]
    return working_df[preview_columns].copy() if preview_columns else working_df.copy()


def build_metric_cards(filtered_df: pd.DataFrame, profile: DatasetProfile) -> list[MetricCard]:
    base_metrics: list[MetricCard] = [
        MetricCard("Filtered records", _format_metric_value(len(filtered_df), integer=True), "Rows after filters"),
    ]
    if int(profile.row_count) != int(len(filtered_df)):
        base_metrics.insert(
            0,
            MetricCard("Sheet records", _format_metric_value(profile.row_count, integer=True), "Rows in the selected sheet before dashboard filters"),
        )
    business_metrics: list[MetricCard] = []
    supporting_metrics: list[MetricCard] = []
    amount_column = _pick_primary_amount_column(profile)
    exposure_amount_column = _pick_billing_gap_amount_column(profile) if profile.dataset_type == DATASET_BILLING else amount_column
    exception_billing_shape = _has_exception_billing_shape(profile)
    grouped_measures = _classify_billing_measure_columns(profile)

    if amount_column and amount_column in filtered_df.columns:
        quoted_amount = _quote_identifier(amount_column)
        amount_df = _run_query(
            filtered_df,
            f"SELECT SUM(COALESCE({quoted_amount}, 0)) AS total_amount, AVG({quoted_amount}) AS average_amount FROM analytics_df",
        )
        amount_total = amount_df.iloc[0]["total_amount"] if not amount_df.empty else None
        amount_avg = amount_df.iloc[0]["average_amount"] if not amount_df.empty else None
        if not exception_billing_shape:
            business_metrics.append(MetricCard("Total amount", _format_metric_value(amount_total, currency=True), amount_column))
        if not exception_billing_shape:
            business_metrics.append(MetricCard("Average amount", _format_metric_value(amount_avg, currency=True), amount_column))

    if exception_billing_shape:
        for group_name in ["expected", "billed", "gap"]:
            if not grouped_measures[group_name]:
                continue
            column_name = grouped_measures[group_name][0]
            if column_name in filtered_df.columns:
                total_value = float(filtered_df[column_name].fillna(0).sum())
                business_metrics.append(
                    MetricCard(
                        f"Total {_format_label(column_name)}",
                        _format_metric_value(total_value, currency=True),
                        column_name,
                    )
                )

    if profile.quantity_columns:
        quantity_column = profile.quantity_columns[0]
        quantity_df = _run_query(
            filtered_df,
            f"SELECT SUM(COALESCE({_quote_identifier(quantity_column)}, 0)) AS total_quantity FROM analytics_df",
        )
        business_metrics.append(
            MetricCard(
                "Units",
                _format_metric_value(quantity_df.iloc[0]["total_quantity"] if not quantity_df.empty else None, integer=True),
                quantity_column,
            )
        )

    unique_entity_column = next(iter(profile.customer_columns or profile.product_columns), None)
    if unique_entity_column and unique_entity_column in filtered_df.columns:
        unique_df = _run_query(
            filtered_df,
            f"SELECT COUNT(DISTINCT {_quote_identifier(unique_entity_column)}) AS unique_count FROM analytics_df",
        )
        supporting_metrics.append(
            MetricCard(
                f"Unique {unique_entity_column.replace('_', ' ')}",
                _format_metric_value(unique_df.iloc[0]["unique_count"] if not unique_df.empty else None, integer=True),
                unique_entity_column,
            )
        )

    if profile.status_columns:
        status_column = _pick_billing_focus_column(profile) or profile.status_columns[0]
        invoice_column = _pick_primary_invoice_column(profile)
        distinct_status_df = _run_query(
            filtered_df,
            f"SELECT COUNT(DISTINCT {_quote_identifier(status_column)}) AS distinct_statuses FROM analytics_df",
        )
        supporting_metrics.append(
            MetricCard(
                "Status categories",
                _format_metric_value(
                    distinct_status_df.iloc[0]["distinct_statuses"] if not distinct_status_df.empty else None,
                    integer=True,
                ),
                status_column,
            )
        )

        if amount_column and amount_column in filtered_df.columns:
            focus_masks = _billing_status_masks(filtered_df, profile)
            positive_mask = focus_masks.get("billed", pd.Series(False, index=filtered_df.index))
            negative_mask = focus_masks.get("unbilled", pd.Series(False, index=filtered_df.index))
            overdue_mask = focus_masks.get("overdue", pd.Series(False, index=filtered_df.index))
            under_billed_mask = focus_masks.get("under_billed", pd.Series(False, index=filtered_df.index))
            billing_delay_mask = focus_masks.get("billing_delay", pd.Series(False, index=filtered_df.index))
            if profile.dataset_type == DATASET_BILLING and negative_mask.any():
                business_metrics.append(
                    MetricCard(
                        "Unbilled invoices",
                        _format_metric_value(_count_entities(filtered_df, invoice_column, negative_mask), integer=True),
                        invoice_column or status_column,
                    )
                )
            if profile.dataset_type == DATASET_BILLING and under_billed_mask.any():
                business_metrics.append(
                    MetricCard(
                        "Under billed records",
                        _format_metric_value(_count_entities(filtered_df, invoice_column, under_billed_mask), integer=True),
                        invoice_column or status_column,
                    )
                )
            if profile.dataset_type == DATASET_BILLING and billing_delay_mask.any():
                business_metrics.append(
                    MetricCard(
                        "Billing delay records",
                        _format_metric_value(_count_entities(filtered_df, invoice_column, billing_delay_mask), integer=True),
                        invoice_column or status_column,
                    )
                )
            if profile.dataset_type == DATASET_BILLING and positive_mask.any():
                business_metrics.append(
                    MetricCard(
                        "Billed invoices",
                        _format_metric_value(_count_entities(filtered_df, invoice_column, positive_mask), integer=True),
                        invoice_column or status_column,
                    )
                )
            if profile.dataset_type == DATASET_BILLING and overdue_mask.any():
                business_metrics.append(
                    MetricCard(
                        "Overdue invoices",
                        _format_metric_value(_count_entities(filtered_df, invoice_column, overdue_mask), integer=True),
                        invoice_column or status_column,
                    )
                )
            if positive_mask.any():
                invoiced_amount = float(filtered_df.loc[positive_mask, amount_column].fillna(0).sum())
                business_metrics.append(MetricCard("Invoiced amount", _format_metric_value(invoiced_amount, currency=True), status_column))
            if negative_mask.any() and exposure_amount_column and exposure_amount_column in filtered_df.columns:
                unpaid_amount = float(filtered_df.loc[negative_mask, exposure_amount_column].fillna(0).sum())
                business_metrics.append(MetricCard("Unpaid amount", _format_metric_value(unpaid_amount, currency=True), status_column))

    metrics = base_metrics + business_metrics + supporting_metrics
    deduped_metrics: list[MetricCard] = []
    seen_titles: set[str] = set()
    for metric in metrics:
        if metric.title in seen_titles:
            continue
        seen_titles.add(metric.title)
        deduped_metrics.append(metric)
    return deduped_metrics[:8]


def build_chart_specs(filtered_df: pd.DataFrame, profile: DatasetProfile) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    amount_column = _pick_primary_amount_column(profile)
    billing_chart_measures = _preferred_billing_chart_measures(filtered_df, profile, max_count=2) if profile.dataset_type == DATASET_BILLING else []

    aging_chart = _build_billing_aging_chart_spec(filtered_df, profile)
    _append_chart_if_missing(charts, aging_chart)

    trend_measure_column = billing_chart_measures[0] if billing_chart_measures else amount_column
    if profile.date_columns and trend_measure_column and trend_measure_column in filtered_df.columns:
        date_column = profile.date_columns[0]
        granularity = "month" if filtered_df[date_column].nunique(dropna=True) > 45 else "day"
        trend_df = _run_query(
            filtered_df,
            (
                f"SELECT CAST(date_trunc('{granularity}', {_quote_identifier(date_column)}) AS DATE) AS period, "
                f"SUM(COALESCE({_quote_identifier(trend_measure_column)}, 0)) AS metric_value "
                f"FROM analytics_df WHERE {_quote_identifier(date_column)} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 1"
            ),
        )
        if not trend_df.empty:
            _append_chart_if_missing(
                charts,
                ChartSpec(
                    title=f"{trend_measure_column.replace('_', ' ').title()} over time",
                    chart_type="line",
                    dataframe=trend_df,
                    x_column="period",
                    y_column="metric_value",
                    description=f"Trend by {granularity}",
                    supported_chart_types=TREND_CHART_TYPES,
                    x_label=_format_label(date_column),
                    y_label=_format_label(trend_measure_column),
                ),
            )

    primary_dimension = _pick_meaningful_dimension_column(filtered_df, profile, exclude=set(profile.status_columns))
    if primary_dimension and primary_dimension in filtered_df.columns:
        chart_measure_candidates = billing_chart_measures if billing_chart_measures else ([amount_column] if amount_column and amount_column in filtered_df.columns else [])
        if chart_measure_candidates:
            for measure_column in chart_measure_candidates:
                category_df = _run_query(
                    filtered_df,
                    (
                        f"SELECT {_quote_identifier(primary_dimension)} AS category, "
                        f"SUM(COALESCE({_quote_identifier(measure_column)}, 0)) AS metric_value "
                        f"FROM analytics_df WHERE {_quote_identifier(primary_dimension)} IS NOT NULL "
                        f"GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
                    ),
                )
                if not category_df.empty:
                    _append_chart_if_missing(
                        charts,
                        ChartSpec(
                            title=f"{measure_column.replace('_', ' ').title()} by {primary_dimension.replace('_', ' ')}",
                            chart_type="bar",
                            dataframe=category_df,
                            x_column="category",
                            y_column="metric_value",
                            supported_chart_types=SIMPLE_CHART_TYPES,
                            x_label=_format_label(primary_dimension),
                            y_label=_format_label(measure_column),
                        ),
                    )
        else:
            count_df = _run_query(
                filtered_df,
                (
                    f"SELECT {_quote_identifier(primary_dimension)} AS category, COUNT(*) AS metric_value "
                    f"FROM analytics_df WHERE {_quote_identifier(primary_dimension)} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
                ),
            )
            if not count_df.empty:
                _append_chart_if_missing(
                    charts,
                    ChartSpec(
                        title=f"Record count by {primary_dimension.replace('_', ' ')}",
                        chart_type="bar",
                        dataframe=count_df,
                        x_column="category",
                        y_column="metric_value",
                        supported_chart_types=SIMPLE_CHART_TYPES,
                        x_label=_format_label(primary_dimension),
                        y_label="Count",
                    ),
                )

    if profile.status_columns:
        status_column = _pick_billing_focus_column(profile) or profile.status_columns[0]
        status_measure_column = trend_measure_column if trend_measure_column and trend_measure_column in filtered_df.columns else None
        use_amount_for_status = bool(status_measure_column and "exception" not in status_column)
        status_value_sql = (
            f"SUM(COALESCE({_quote_identifier(status_measure_column)}, 0))"
            if use_amount_for_status and status_measure_column
            else "COUNT(*)"
        )
        status_df = _run_query(
            filtered_df,
            (
                f"SELECT {_quote_identifier(status_column)} AS category, {status_value_sql} AS metric_value "
                f"FROM analytics_df WHERE {_quote_identifier(status_column)} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
            ),
        )
        if not status_df.empty:
            _append_chart_if_missing(
                charts,
                ChartSpec(
                    title=f"{status_column.replace('_', ' ').title()} breakdown",
                    chart_type="bar",
                    dataframe=status_df,
                    x_column="category",
                    y_column="metric_value",
                    description="Status distribution" if use_amount_for_status else "Issue count distribution",
                    supported_chart_types=SIMPLE_CHART_TYPES,
                    x_label=_format_label(status_column),
                    y_label=_metric_value_label(status_measure_column, count_fallback=not use_amount_for_status),
                ),
            )

    if profile.status_columns:
        status_column = _pick_billing_focus_column(profile) or profile.status_columns[0]
        stack_dimension = _pick_meaningful_dimension_column(filtered_df, profile, exclude={status_column})
        if stack_dimension and stack_dimension in filtered_df.columns:
            stack_measure_column = trend_measure_column
            value_sql = (
                f"SUM(COALESCE({_quote_identifier(stack_measure_column)}, 0))"
                if stack_measure_column and stack_measure_column in filtered_df.columns and "exception" not in status_column
                else "COUNT(*)"
            )
            stacked_df = _run_query(
                filtered_df,
                (
                    f"SELECT {_quote_identifier(stack_dimension)} AS category, {_quote_identifier(status_column)} AS status_value, "
                    f"{value_sql} AS metric_value FROM analytics_df "
                    f"WHERE {_quote_identifier(stack_dimension)} IS NOT NULL AND {_quote_identifier(status_column)} IS NOT NULL "
                    f"GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 60"
                ),
            )
            if not stacked_df.empty:
                _append_chart_if_missing(
                    charts,
                    ChartSpec(
                        title=f"{status_column.replace('_', ' ').title()} by {stack_dimension.replace('_', ' ')}",
                        chart_type="stacked_bar",
                        dataframe=stacked_df,
                        x_column="category",
                        y_column="metric_value",
                        color_column="status_value",
                        supported_chart_types=COMPARISON_CHART_TYPES,
                        x_label=_format_label(stack_dimension),
                        y_label=_metric_value_label(stack_measure_column, count_fallback=not (stack_measure_column and stack_measure_column in filtered_df.columns and "exception" not in status_column)),
                    ),
                )

    return charts[:4]


def build_prompt_metric_cards(base_df: pd.DataFrame, profile: DatasetProfile, prompt: str) -> list[MetricCard]:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return []
    prompt_config = _parse_dashboard_prompt(prompt_text, profile)
    prompt_scoped_df = _filter_billing_focus(base_df, profile, prompt_config["focus"])
    prompt_scoped_df = _apply_prompt_value_filters(prompt_scoped_df, prompt_config.get("filters", []))
    if prompt_scoped_df.empty:
        return []
    return build_metric_cards(prompt_scoped_df, profile)


def build_prompt_drilldown_frame(base_df: pd.DataFrame, profile: DatasetProfile, prompt: str) -> pd.DataFrame | None:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return None
    prompt_config = _parse_dashboard_prompt(prompt_text, profile)
    if not prompt_config.get("invoice_listing") and not prompt_config.get("customer_listing"):
        return None

    scoped_df = _filter_billing_focus(base_df, profile, prompt_config["focus"])
    scoped_df = _apply_prompt_value_filters(scoped_df, prompt_config.get("filters", []))
    if scoped_df.empty:
        return None

    if prompt_config.get("customer_listing"):
        return _build_prompt_entity_listing_frame(scoped_df, profile, prompt_config)

    invoice_column = str(prompt_config.get("invoice_column") or "")
    drilldown_df = _build_prompt_invoice_listing_frame(scoped_df, profile, invoice_column)
    if drilldown_df is None or drilldown_df.empty:
        return None
    return drilldown_df


def _compute_adaptive_dashboard_summary_facts(
    filtered_df: pd.DataFrame,
    profile: DatasetProfile,
    metrics: list[MetricCard],
    charts: list[ChartSpec],
    *,
    active_filter_summary: str,
) -> AdaptiveDashboardSummaryFacts:
    preferred_metric_titles = [
        "Total amount",
        "Total expected amount",
        "Total billed amount",
        "Total value gap",
        "Unpaid amount",
        "Average amount",
        "Units",
        "Unbilled invoices",
        "Under billed records",
        "Billing delay records",
        "Overdue invoices",
        "Billed invoices",
        "Unique customer",
        "Unique product",
        "Status categories",
    ]
    metric_lookup = {metric.title: metric for metric in metrics if metric.value != "N/A"}
    metric_highlights: list[str] = []
    for title in preferred_metric_titles:
        metric = metric_lookup.get(title)
        if metric is None:
            continue
        metric_highlights.append(f"{metric.title}: {metric.value}")
        if len(metric_highlights) >= 3:
            break

    top_chart = next(
        (
            chart
            for chart in charts
            if not chart.dataframe.empty and chart.chart_type in {"bar", "stacked_bar", "grouped_bar", "pie"}
        ),
        None,
    )
    top_dimension_label: str | None = None
    top_category: str | None = None
    top_metric_label: str | None = None
    top_metric_value: float | None = None
    confidence = 0.94

    if top_chart is not None and top_chart.x_column in top_chart.dataframe.columns and top_chart.y_column in top_chart.dataframe.columns:
        grouped = (
            top_chart.dataframe[[top_chart.x_column, top_chart.y_column]]
            .copy()
            .assign(**{top_chart.y_column: pd.to_numeric(top_chart.dataframe[top_chart.y_column], errors="coerce")})
            .dropna(subset=[top_chart.x_column, top_chart.y_column])
            .groupby(top_chart.x_column, dropna=False)[top_chart.y_column]
            .sum()
            .reset_index(name="metric_total")
            .sort_values(["metric_total", top_chart.x_column], ascending=[False, True])
            .reset_index(drop=True)
        )
        if not grouped.empty:
            top_row = grouped.iloc[0]
            top_dimension_label = str(top_chart.x_label or _format_label(top_chart.x_column) or "category")
            top_category = str(top_row[top_chart.x_column])
            top_metric_label = str(top_chart.y_label or _format_label(top_chart.y_column) or "value")
            top_metric_value = float(top_row["metric_total"])
            if top_dimension_label.strip().lower() in {"category", "value"}:
                confidence -= 0.24
            if top_metric_label.strip().lower() in {"value", "metric value"}:
                confidence -= 0.18
    else:
        confidence -= 0.2

    if not metric_highlights:
        confidence -= 0.08

    return AdaptiveDashboardSummaryFacts(
        dataset_type=profile.dataset_type,
        row_count=int(len(filtered_df)),
        column_count=int(profile.column_count),
        active_filter_summary=active_filter_summary or "None",
        metric_highlights=metric_highlights,
        top_dimension_label=top_dimension_label,
        top_category=top_category,
        top_metric_label=top_metric_label,
        top_metric_value=top_metric_value,
        chart_count=len(charts),
        confidence=max(0.0, min(1.0, confidence)),
        use_neutral_language=confidence < 0.62,
    )


def _deterministic_adaptive_dashboard_summary(facts: AdaptiveDashboardSummaryFacts) -> str:
    parts = [
        f"This {facts.dataset_type} dataset contains {facts.row_count:,} records across {facts.column_count} columns."
    ]

    if facts.metric_highlights:
        parts.append(f"Key metrics: {'; '.join(facts.metric_highlights[:3])}.")

    if facts.top_category is not None and facts.top_metric_value is not None:
        dimension_label = "category" if facts.use_neutral_language else str(facts.top_dimension_label or "category").lower()
        metric_label = "value" if facts.use_neutral_language else str(facts.top_metric_label or "value").lower()
        parts.append(
            f"Top {dimension_label} is {facts.top_category} with {_format_metric_value(facts.top_metric_value, currency='amount' in metric_label or 'price' in metric_label or 'sales' in metric_label)} {metric_label}."
        )

    if facts.active_filter_summary and facts.active_filter_summary != "None":
        parts.append(f"Current filters: {facts.active_filter_summary}.")

    return " ".join(parts)


def _llm_adaptive_dashboard_summary(facts: AdaptiveDashboardSummaryFacts) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
        response = client.responses.create(
            model=model,
            instructions=(
                "Write a short dashboard overview in at most 3 sentences. "
                "Use only the provided structured facts. "
                "Do not invent columns, KPIs, causes, recommendations, or business meaning. "
                "Mention current filters only when active_filter_summary is not 'None'. "
                "If use_neutral_language is true, prefer neutral words like category and value."
            ),
            input=json.dumps(asdict(facts), ensure_ascii=True),
            max_output_tokens=140,
        )
        summary = (getattr(response, "output_text", "") or "").strip()
        return summary or None
    except Exception:
        return None


def build_ai_summary(
    filtered_df: pd.DataFrame,
    profile: DatasetProfile,
    metrics: list[MetricCard],
    charts: list[ChartSpec],
    *,
    active_filter_summary: str = "None",
    prefer_llm: bool = True,
) -> str:
    facts = _compute_adaptive_dashboard_summary_facts(
        filtered_df,
        profile,
        metrics,
        charts,
        active_filter_summary=active_filter_summary,
    )
    summary = _llm_adaptive_dashboard_summary(facts) if prefer_llm else None
    return summary or _deterministic_adaptive_dashboard_summary(facts)


def _summarize_active_filters(active_filters: list[str]) -> str:
    cleaned = [str(item).strip() for item in active_filters if str(item).strip()]
    return "; ".join(cleaned) if cleaned else "None"


def _clear_dashboard_filter_state(scope_key: str, filter_columns: list[str], date_columns: list[str]) -> None:
    import streamlit as st

    keys_to_clear = [f"{scope_key}_billing_focus"]
    for column_name in filter_columns:
        keys_to_clear.append(f"{scope_key}_filter_{column_name}")
    for column_name in date_columns:
        keys_to_clear.append(f"{scope_key}_date_filter_{column_name}")
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _dashboard_filter_signature(
    source_name: str,
    selected_sheet: str,
    filter_options: dict[str, list[str]],
    date_ranges: dict[str, tuple[str, str]],
) -> tuple[str, str, tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, str]], ...]]:
    return (
        str(source_name or ""),
        str(selected_sheet or ""),
        tuple(sorted((column_name, tuple(values)) for column_name, values in filter_options.items())),
        tuple(sorted((column_name, (start_value, end_value)) for column_name, (start_value, end_value) in date_ranges.items())),
    )


def _initialize_dashboard_filter_state(
    scope_key: str,
    signature: tuple[str, str, tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, str]], ...]],
    filter_options: dict[str, list[str]],
    date_ranges: dict[str, tuple[str, str]],
) -> None:
    import streamlit as st

    signature_key = f"{scope_key}_dashboard_filter_signature"
    if st.session_state.get(signature_key) == signature:
        return

    for column_name, options in filter_options.items():
        st.session_state[f"{scope_key}_filter_{column_name}"] = list(options)
    for column_name, (start_value, end_value) in date_ranges.items():
        st.session_state[f"{scope_key}_date_filter_{column_name}"] = (
            pd.Timestamp(start_value).date(),
            pd.Timestamp(end_value).date(),
        )
    st.session_state[f"{scope_key}_billing_focus"] = "all"
    st.session_state[signature_key] = signature


def column_metadata_frame(profile: DatasetProfile) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in profile.column_metadata])


def render_adaptive_dashboard(sheet_frames: dict[str, pd.DataFrame], source_name: str, scope_key: str) -> None:
    import streamlit as st

    if duckdb is None:
        st.warning("DuckDB is not installed, so the instant adaptive dashboard is unavailable.")
        return

    if not sheet_frames:
        st.info("No sheet data is available for dashboard generation.")
        return

    st.subheader("Instant analytics dashboard")
    sheet_names = list(sheet_frames.keys())
    selected_sheet = sheet_names[0]
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox(
            "Sheet for instant analytics",
            options=sheet_names,
            key=f"{scope_key}_sheet",
        )

    source_df = _enrich_selected_sheet_frame(sheet_frames[selected_sheet], selected_sheet, sheet_frames)
    prepared_df, profile = prepare_dataset(source_df, dataset_name=f"{source_name} / {selected_sheet}")

    st.caption(f"Dataset type guess: {profile.dataset_type}")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Rows", f"{profile.row_count:,}")
    summary_cols[1].metric("Columns", f"{profile.column_count:,}")
    summary_cols[2].metric("Sheet", selected_sheet)

    filterable_categories = [
        column_name
        for column_name in profile.categorical_columns
        if prepared_df[column_name].nunique(dropna=True) <= 30
    ]
    prioritized_filters = list(dict.fromkeys([
        *profile.region_columns,
        *profile.product_columns,
        *profile.customer_columns,
        *filterable_categories,
    ]))
    filter_columns = prioritized_filters[:8]
    filter_options = {
        column_name: sorted([item for item in prepared_df[column_name].dropna().astype(str).unique().tolist() if item])
        for column_name in filter_columns
    }
    date_ranges = {
        column_name: (
            pd.to_datetime(prepared_df[column_name], errors="coerce").dropna().min().date().isoformat(),
            pd.to_datetime(prepared_df[column_name], errors="coerce").dropna().max().date().isoformat(),
        )
        for column_name in profile.date_columns[:2]
        if not pd.to_datetime(prepared_df[column_name], errors="coerce").dropna().empty
    }
    _initialize_dashboard_filter_state(
        scope_key,
        _dashboard_filter_signature(source_name, selected_sheet, filter_options, date_ranges),
        filter_options,
        date_ranges,
    )

    filtered_df = prepared_df.copy()
    active_filters: list[str] = []
    with st.container(border=True):
        st.markdown("**Filters**")
        if st.button("Reset dashboard filters", key=f"{scope_key}_reset_dashboard_filters"):
            _clear_dashboard_filter_state(scope_key, filter_columns, profile.date_columns[:2])
            st.rerun()
        for column_name in filter_columns:
            options = filter_options.get(column_name) or []
            if not options:
                continue
            selected_values = st.multiselect(
                f"Filter {column_name}",
                options=options,
                default=options,
                key=f"{scope_key}_filter_{column_name}",
            )
            if selected_values and len(selected_values) < len(options):
                filtered_df = filtered_df[filtered_df[column_name].astype(str).isin(selected_values)]
                visible_values = selected_values[:3]
                suffix = " ..." if len(selected_values) > 3 else ""
                active_filters.append(f"{_format_label(column_name)}: {', '.join(visible_values)}{suffix}")

        for column_name in profile.date_columns[:2]:
            stored_range = date_ranges.get(column_name)
            if not stored_range:
                continue
            start_date = pd.Timestamp(stored_range[0]).date()
            end_date = pd.Timestamp(stored_range[1]).date()
            selected_range = st.date_input(
                f"Filter {column_name}",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
                key=f"{scope_key}_date_filter_{column_name}",
            )
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                start_value, end_value = selected_range
                if start_value != start_date or end_value != end_date:
                    filtered_df = filtered_df[
                        pd.to_datetime(filtered_df[column_name], errors="coerce").between(
                            pd.Timestamp(start_value),
                            pd.Timestamp(end_value),
                            inclusive="both",
                        )
                    ]
                    active_filters.append(f"{_format_label(column_name)}: {start_value.isoformat()} to {end_value.isoformat()}")

    dashboard_df = filtered_df.copy()
    billing_focus = "all"
    if profile.dataset_type == DATASET_BILLING:
        billing_focus_options = _available_billing_focuses(filtered_df, profile)
        billing_focus = st.selectbox(
            "Billing focus",
            options=billing_focus_options,
            format_func=lambda value: str(BILLING_FOCUS_LABELS.get(str(value), str(value).replace("_", " ").title())),
            key=f"{scope_key}_billing_focus",
        )
        dashboard_df = _filter_billing_focus(dashboard_df, profile, billing_focus)
        if billing_focus != "all":
            active_filters.append(f"Billing focus: {BILLING_FOCUS_LABELS.get(billing_focus, billing_focus.replace('_', ' ').title())}")

    prompt_text = st.text_input(
        "Dashboard prompt",
        value="",
        key=f"{scope_key}_dashboard_prompt",
        placeholder="Example: show under billed by service type as pie chart",
    )
    metric_cards = build_metric_cards(dashboard_df, profile)
    charts = build_chart_specs(dashboard_df, profile) if not dashboard_df.empty else []
    summary_text = build_ai_summary(
        dashboard_df,
        profile,
        metric_cards,
        charts,
        active_filter_summary=_summarize_active_filters(active_filters),
    )
    prompt_config = _parse_dashboard_prompt(prompt_text, profile) if prompt_text.strip() else None
    prompt_chart = build_prompt_chart_spec(prepared_df, profile, prompt_text) if prompt_text.strip() else None
    prompt_metrics = build_prompt_metric_cards(prepared_df, profile, prompt_text) if prompt_text.strip() else []
    prompt_drilldown_df = build_prompt_drilldown_frame(prepared_df, profile, prompt_text) if prompt_text.strip() else None
    manual_chart: ChartSpec | None = None
    dashboard_view_mode = "Auto"

    st.info(summary_text)

    if metric_cards:
        metric_columns = st.columns(min(4, len(metric_cards)))
        for index, metric in enumerate(metric_cards):
            metric_columns[index % len(metric_columns)].metric(metric.title, metric.value, help=metric.help_text)

    if prompt_text.strip():
        st.markdown("**Prompt-driven view**")
        st.caption("Prompt-driven view uses the full selected sheet by default and does not inherit dashboard customer/date filters.")
        if prompt_metrics:
            prompt_metric_columns = st.columns(min(4, len(prompt_metrics)))
            for index, metric in enumerate(prompt_metrics[:4]):
                prompt_metric_columns[index % len(prompt_metric_columns)].metric(metric.title, metric.value, help=metric.help_text)
        if prompt_drilldown_df is not None:
            prompt_listing_label = "Prompt-driven invoices" if prompt_config and prompt_config.get("invoice_listing") else "Prompt-driven results"
            st.markdown(f"**{prompt_listing_label}**")
            prompt_display_df = _prepare_display_frame(prompt_drilldown_df.head(200), profile)
            st.dataframe(
                prompt_display_df,
                use_container_width=True,
                hide_index=True,
                column_config=_build_dataframe_column_config(prompt_display_df, profile),
            )
        if prompt_chart is None:
            if prompt_drilldown_df is None:
                st.info("The prompt could not be mapped to a chart from the current dataset shape.")
        else:
            charts = [prompt_chart] + charts

    dimension_options = _manual_dashboard_dimension_options(dashboard_df, profile)
    measure_options = _manual_dashboard_measure_options(dashboard_df, profile)
    if dimension_options:
        with st.expander("Customize initial dashboard", expanded=False):
            st.caption("Override the auto-generated dashboard with your own chart configuration for this sheet and current filters.")
            dashboard_view_mode = st.radio(
                "Dashboard layout",
                options=["Auto", "Auto + custom", "Custom only"],
                index=0,
                horizontal=True,
                key=f"{scope_key}_manual_dashboard_mode",
            )
            manual_col1, manual_col2, manual_col3 = st.columns(3)
            with manual_col1:
                manual_x = st.selectbox(
                    "Custom X column",
                    options=dimension_options,
                    index=0,
                    key=f"{scope_key}_manual_dashboard_x",
                )
            with manual_col2:
                default_measure = measure_options[0] if measure_options else "__row_count"
                manual_y = st.selectbox(
                    "Custom Y column",
                    options=measure_options,
                    index=measure_options.index(default_measure),
                    key=f"{scope_key}_manual_dashboard_y",
                )
            with manual_col3:
                aggregation_options = _manual_dashboard_aggregation_options(dashboard_df, manual_y)
                manual_aggregation = st.selectbox(
                    "Aggregation",
                    options=aggregation_options,
                    index=0,
                    key=f"{scope_key}_manual_dashboard_aggregation",
                    format_func=lambda value: value.upper(),
                )

            split_options = [None] + [column for column in dimension_options if column != manual_x]
            chart_type_options = COMPARISON_CHART_TYPES if len(split_options) > 1 else SIMPLE_CHART_TYPES
            manual_col4, manual_col5, manual_col6 = st.columns(3)
            with manual_col4:
                manual_split = st.selectbox(
                    "Split by",
                    options=split_options,
                    index=0,
                    key=f"{scope_key}_manual_dashboard_split",
                    format_func=lambda value: "None" if value is None else _format_label(value),
                )
            with manual_col5:
                if manual_split:
                    chart_type_options = COMPARISON_CHART_TYPES
                elif manual_x in profile.date_columns:
                    chart_type_options = TREND_CHART_TYPES
                else:
                    chart_type_options = SIMPLE_CHART_TYPES
                manual_chart_type = st.selectbox(
                    "Custom chart type",
                    options=chart_type_options,
                    index=0,
                    key=f"{scope_key}_manual_dashboard_chart_type",
                    format_func=lambda value: str(CHART_TYPE_LABELS.get(str(value), str(value).replace("_", " ").title())),
                )
            with manual_col6:
                top_n_cap = max(1, int(dashboard_df[manual_x].dropna().nunique()))
                manual_top_n = st.slider(
                    "Top N",
                    min_value=1,
                    max_value=top_n_cap,
                    value=min(10, top_n_cap),
                    key=f"{scope_key}_manual_dashboard_top_n",
                    disabled=manual_x in profile.date_columns,
                )

            manual_chart = _build_manual_dashboard_chart_spec(
                dashboard_df,
                profile,
                x_column=manual_x,
                y_column=manual_y,
                aggregation=manual_aggregation,
                chart_type=manual_chart_type,
                split_column=manual_split,
                top_n=None if manual_x in profile.date_columns else manual_top_n,
            )
            if manual_chart is None:
                st.info("The selected custom dashboard settings did not produce any chartable rows.")

    charts_to_render = list(charts)
    if dashboard_view_mode == "Custom only":
        charts_to_render = [manual_chart] if manual_chart is not None else []
    elif dashboard_view_mode == "Auto + custom" and manual_chart is not None:
        charts_to_render = [manual_chart, *charts]

    if charts_to_render:
        st.markdown("**Charts**")
        for index, chart in enumerate(charts_to_render):
            _render_chart_spec(chart, control_key=f"{scope_key}_chart_{index}")

    if profile.dataset_type == DATASET_BILLING:
        st.markdown("**Invoice drilldown**")
        drilldown_df = _prepare_display_frame(build_billing_drilldown_frame(dashboard_df, profile), profile)
        st.dataframe(
            drilldown_df.head(200),
            use_container_width=True,
            hide_index=True,
            column_config=_build_dataframe_column_config(drilldown_df, profile),
        )

    st.markdown("**Data preview**")
    preview_df = _prepare_display_frame(dashboard_df.head(200), profile)
    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
        column_config=_build_dataframe_column_config(preview_df, profile),
    )

    with st.expander("Column metadata", expanded=False):
        st.dataframe(column_metadata_frame(profile), use_container_width=True, hide_index=True)