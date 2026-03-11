from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from io import BytesIO
import os, re, json, requests, importlib
from typing import Any, Dict
from difflib import SequenceMatcher
from uuid import uuid4
import pandas as pd
from openai import OpenAI

from app.analytics_logging import log_analytics_failure
from app.planning_context import build_planning_context, detect_derived_metrics_by_sheet, flatten_derived_metrics_by_sheet
from app.planner_compat import build_query_spec_stub_compat, classify_query_shape_compat
from app.query_planner import plan_query
from app.query_spec import QuerySpec
from app.sheet_profile import build_sheet_profile
from app.sql_builder import QueryBuildError, build_sql_from_query_spec
from app.sql_validator import validate_sql

sqlparse = None
psycopg = None
dict_row = None
try:
    sqlparse = importlib.import_module("sqlparse")
except Exception:
    sqlparse = None

try:
    psycopg = importlib.import_module("psycopg")
    psycopg_rows = importlib.import_module("psycopg.rows")
    dict_row = getattr(psycopg_rows, "dict_row", None)
except Exception:
    psycopg = None
    dict_row = None

create_client = None
try:
    supabase_module = importlib.import_module("supabase")
    create_client = getattr(supabase_module, "create_client", None)
except Exception:
    create_client = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
SQL_QUERY_MAX_ROWS = int(os.getenv("SQL_QUERY_MAX_ROWS", "5000"))
ALLOW_DIRECT_LLM_SQL = os.getenv("ALLOW_DIRECT_LLM_SQL", "false").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_LLM_SQL_REPAIR = os.getenv("ALLOW_LLM_SQL_REPAIR", "false").strip().lower() in {"1", "true", "yes", "on"}
AUTO_UPLOAD_REGISTRY_ENABLED = os.getenv("AUTO_UPLOAD_REGISTRY_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
AUTO_UPLOAD_REGISTRY_MAX_RULES = max(0, int(os.getenv("AUTO_UPLOAD_REGISTRY_MAX_RULES", "4")))
AUTO_UPLOAD_QA_ENABLED = os.getenv("AUTO_UPLOAD_QA_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
AUTO_UPLOAD_QA_MAX_CASES = max(0, int(os.getenv("AUTO_UPLOAD_QA_MAX_CASES", "3")))
AUTO_UPLOAD_QA_BLOCK_ON_FAILURE = os.getenv("AUTO_UPLOAD_QA_BLOCK_ON_FAILURE", "true").strip().lower() not in {"0", "false", "no"}
oa = OpenAI(api_key=OPENAI_API_KEY)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL in .env (Postgres connection string)")

rest_base_url = SUPABASE_URL.rstrip("/") + "/rest/v1"

sb = None
supabase_init_error = None
if create_client is None:
    supabase_init_error = "Supabase SDK not installed. Will use REST."
else:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        supabase_init_error = f"Supabase SDK init failed: {e}"

app = FastAPI()
from fastapi.responses import JSONResponse
import traceback

@app.middleware("http")
async def catch_exceptions(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()[:4000],
                "path": str(request.url),
            },
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FastAPI backend running"}

# -------------------------
# HELPERS
# -------------------------
def _clean(x) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NULL_LIKE_TOKENS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nan",
    "null",
    "none",
    "nil",
    "blank",
    "missing",
    "not available",
    "not applicable",
    "-",
    "--",
}


def _normalize_sheet_name(sheet_name: str) -> str:
    return normalize_col_name(sheet_name or "")


def _is_null_like_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = _clean(value).lower()
    return text in NULL_LIKE_TOKENS


def _normalize_column_names(columns: list[Any]) -> tuple[list[str], dict[str, str]]:
    next_columns: list[str] = []
    rename_map: dict[str, str] = {}
    seen: dict[str, int] = {}

    for index, original in enumerate(columns, start=1):
        original_text = _clean(original)
        base = original_text or f"column_{index}"
        base = re.sub(r"\s+", " ", base).strip()
        if not base:
            base = f"column_{index}"

        count = seen.get(base, 0) + 1
        seen[base] = count
        cleaned = base if count == 1 else f"{base}_{count}"

        next_columns.append(cleaned)
        if original_text != cleaned:
            rename_map[str(original)] = cleaned

    return next_columns, rename_map


def _normalize_numeric_text(value: str) -> str | None:
    text = _clean(value)
    if not text:
        return ""
    if re.search(r"[A-Za-z]", text):
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    text = re.sub(r"[$€£¥₹,%]", "", text)
    text = text.replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None

    try:
        normalized = format(Decimal(text), "f")
    except (InvalidOperation, ValueError):
        return None

    normalized = normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
    if normalized == "-0":
        normalized = "0"
    if negative and normalized != "0":
        normalized = f"-{normalized.lstrip('-')}"
    return normalized


def _standardize_text_series(series: pd.Series, *, force_text: bool = False) -> tuple[pd.Series, dict[str, int]]:
    stats = {
        "nulls_normalized": 0,
        "numeric_cells_normalized": 0,
        "date_cells_normalized": 0,
    }

    cleaned_values: list[str] = []
    for value in series.tolist():
        if _is_null_like_value(value):
            cleaned_values.append("")
            if value not in {"", None}:
                stats["nulls_normalized"] += 1
            continue
        cleaned_values.append(_clean(value))

    cleaned = pd.Series(cleaned_values, index=series.index, dtype="object")
    non_empty = cleaned[cleaned != ""]
    if non_empty.empty:
        return cleaned, stats

    if force_text:
        return cleaned, stats

    numeric_candidates = non_empty.map(_normalize_numeric_text)
    numeric_success = int(numeric_candidates.notna().sum())
    if numeric_success >= max(1, int(len(non_empty) * 0.8)):
        normalized_values: list[str] = []
        for original, maybe_number in zip(cleaned.tolist(), cleaned.map(_normalize_numeric_text).tolist()):
            if original == "":
                normalized_values.append("")
                continue
            if maybe_number is None:
                normalized_values.append(original)
                continue
            normalized_values.append(maybe_number)
            if maybe_number != original:
                stats["numeric_cells_normalized"] += 1
        return pd.Series(normalized_values, index=series.index, dtype="object"), stats

    parsed_dates = pd.to_datetime(non_empty, errors="coerce")
    date_success = int(parsed_dates.notna().sum())
    if date_success >= max(1, int(len(non_empty) * 0.8)):
        normalized_values = []
        for original in cleaned.tolist():
            if original == "":
                normalized_values.append("")
                continue
            parsed = pd.to_datetime(original, errors="coerce")
            if pd.isna(parsed):
                normalized_values.append(original)
                continue
            iso_text = parsed.strftime("%Y-%m-%d")
            normalized_values.append(iso_text)
            if iso_text != original:
                stats["date_cells_normalized"] += 1
        return pd.Series(normalized_values, index=series.index, dtype="object"), stats

    return cleaned, stats


def clean_and_standardize_tables(
    tables: dict[str, pd.DataFrame],
    *,
    include_assumptions_sheet: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cleaned_tables: dict[str, pd.DataFrame] = {}
    sheet_reports: list[dict[str, Any]] = []
    excluded_sheets: list[str] = []
    totals = {
        "input_sheet_count": len(tables or {}),
        "output_sheet_count": 0,
        "dropped_empty_row_count": 0,
        "dropped_empty_column_count": 0,
        "nulls_normalized": 0,
        "numeric_cells_normalized": 0,
        "date_cells_normalized": 0,
        "renamed_column_count": 0,
    }

    for original_sheet_name, df in (tables or {}).items():
        sheet_name = str(original_sheet_name)
        if not include_assumptions_sheet and _normalize_sheet_name(sheet_name) == "assumptions":
            excluded_sheets.append(sheet_name)
            continue

        working_df = df.copy()
        original_row_count = int(working_df.shape[0])
        original_col_count = int(working_df.shape[1])

        new_columns, rename_map = _normalize_column_names(list(working_df.columns))
        working_df.columns = new_columns

        per_sheet_stats = {
            "nulls_normalized": 0,
            "numeric_cells_normalized": 0,
            "date_cells_normalized": 0,
        }
        for column in list(working_df.columns):
            normalized_series, series_stats = _standardize_text_series(
                working_df[column],
                force_text=looks_like_id_col(column),
            )
            working_df[column] = normalized_series
            for key, value in series_stats.items():
                per_sheet_stats[key] += int(value)

        empty_row_mask = (working_df == "").all(axis=1)
        dropped_empty_rows = int(empty_row_mask.sum())
        if dropped_empty_rows:
            working_df = working_df.loc[~empty_row_mask].reset_index(drop=True)

        empty_columns = [column for column in working_df.columns if (working_df[column] == "").all()]
        dropped_empty_columns = list(empty_columns)
        if dropped_empty_columns:
            working_df = working_df.drop(columns=dropped_empty_columns)

        if working_df.empty or working_df.shape[1] == 0:
            excluded_sheets.append(sheet_name)
            continue

        cleaned_tables[sheet_name] = working_df
        totals["output_sheet_count"] += 1
        totals["dropped_empty_row_count"] += dropped_empty_rows
        totals["dropped_empty_column_count"] += len(dropped_empty_columns)
        totals["nulls_normalized"] += per_sheet_stats["nulls_normalized"]
        totals["numeric_cells_normalized"] += per_sheet_stats["numeric_cells_normalized"]
        totals["date_cells_normalized"] += per_sheet_stats["date_cells_normalized"]
        totals["renamed_column_count"] += len(rename_map)

        sheet_reports.append(
            {
                "sheet": sheet_name,
                "input_row_count": original_row_count,
                "output_row_count": int(working_df.shape[0]),
                "input_column_count": original_col_count,
                "output_column_count": int(working_df.shape[1]),
                "renamed_columns": rename_map,
                "dropped_empty_rows": dropped_empty_rows,
                "dropped_empty_columns": dropped_empty_columns,
                **per_sheet_stats,
            }
        )

    report = {
        "enabled": True,
        "include_assumptions_sheet": bool(include_assumptions_sheet),
        "excluded_sheets": excluded_sheets,
        "sheet_reports": sheet_reports,
        "totals": totals,
    }
    return cleaned_tables, report

def _row_to_text(row: dict) -> str:
    return " | ".join([f"{k}: {_clean(v)}" for k, v in row.items()])

def _rest_headers(prefer: str | None = None) -> dict:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers

def _rest_request(method: str, path: str, *, params: dict | None = None, payload: Any = None, prefer: str | None = None) -> Any:
    url = f"{rest_base_url}/{path.lstrip('/')}"
    try:
        res = requests.request(
            method=method,
            url=url,
            headers=_rest_headers(prefer),
            params=params,
            json=payload,
            timeout=60,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase request failed: {e}")

    if res.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase error {res.status_code}: {res.text}")

    if not res.text.strip():
        return None

    try:
        return res.json()
    except Exception:
        return res.text


def _load_dataset_frame_payload(dataset_id: str, *, page_size: int = 1000) -> dict[str, Any]:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    meta: dict[str, Any] = {}
    if sb is not None:
        meta = (
            sb.table("datasets")
            .select("id,name")
            .eq("id", dataset_id)
            .single()
            .execute()
            .data
        ) or {}
    else:
        rows = _rest_request(
            "GET",
            "datasets",
            params={"select": "id,name", "id": f"eq.{dataset_id}", "limit": 1},
        ) or []
        meta = rows[0] if isinstance(rows, list) and rows else {}

    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    collected_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        if sb is not None:
            batch = (
                sb.table("dataset_rows")
                .select("sheet_name,row_index,row_json")
                .eq("dataset_id", dataset_id)
                .order("sheet_name")
                .order("row_index")
                .range(offset, offset + page_size - 1)
                .execute()
                .data
            ) or []
        else:
            batch = _rest_request(
                "GET",
                "dataset_rows",
                params={
                    "select": "sheet_name,row_index,row_json",
                    "dataset_id": f"eq.{dataset_id}",
                    "order": "sheet_name.asc,row_index.asc",
                    "limit": page_size,
                    "offset": offset,
                },
            ) or []

        if not batch:
            break
        collected_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    frames: dict[str, list[dict[str, Any]]] = {}
    for row in collected_rows:
        if not isinstance(row, dict):
            continue
        sheet_name = str(row.get("sheet_name") or "data").strip() or "data"
        row_json = row.get("row_json")
        if not isinstance(row_json, dict):
            continue
        frames.setdefault(sheet_name, []).append(dict(row_json))

    return {
        "dataset_id": dataset_id,
        "name": str(meta.get("name") or dataset_id),
        "sheets": list(frames.keys()),
        "frames": frames,
    }


def _load_rules_registry() -> list[dict[str, Any]]:
    select_cols = "id,intent,enabled,priority,patterns,builder,sql_template,description"

    if sb is not None:
        try:
            rules = (
                sb.table("rules_registry")
                .select(select_cols)
                .order("priority")
                .execute()
                .data
            ) or []
            if rules:
                return rules
        except Exception:
            pass

    try:
        rules = _rest_request(
            "GET",
            "rules_registry",
            params={
                "select": select_cols,
                "order": "priority.asc",
            },
        ) or []
        if rules:
            return rules
    except Exception:
        pass

    local_registry_path = os.path.join(os.path.dirname(__file__), "app", "rules_registry.json")
    try:
        with open(local_registry_path, "r", encoding="utf-8") as handle:
            rules = json.load(handle)
            return rules if isinstance(rules, list) else []
    except Exception:
        return []


def _local_rules_registry_path() -> str:
    return os.path.join(os.path.dirname(__file__), "app", "rules_registry.json")


def _write_local_rules_registry(rules: list[dict[str, Any]]) -> None:
    path = _local_rules_registry_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rules, handle, indent=2)


def _normalize_registry_rule_record(rule: dict[str, Any]) -> dict[str, Any]:
    patterns = rule.get("patterns") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    return {
        "id": str(rule.get("id") or uuid4()),
        "intent": str(rule.get("intent") or "").strip(),
        "enabled": bool(rule.get("enabled", True)),
        "priority": int(rule.get("priority", 100) or 100),
        "patterns": [str(p).strip() for p in patterns if str(p).strip()],
        "builder": (str(rule.get("builder")).strip() if rule.get("builder") is not None else None) or None,
        "sql_template": (str(rule.get("sql_template")).strip() if rule.get("sql_template") is not None else None) or None,
        "description": (str(rule.get("description")).strip() if rule.get("description") is not None else None) or None,
    }


def _validate_registry_rule_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_registry_rule_record(payload)
    if not normalized["intent"]:
        raise HTTPException(status_code=400, detail="intent is required")
    if not normalized["patterns"]:
        raise HTTPException(status_code=400, detail="patterns must contain at least one regex")
    if not normalized["builder"] and not normalized["sql_template"]:
        raise HTTPException(status_code=400, detail="builder or sql_template is required")
    return normalized


def _create_registry_rule(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _validate_registry_rule_payload(payload)

    if sb is not None:
        try:
            created = sb.table("rules_registry").insert(normalized).execute().data or []
            if created:
                return dict(created[0])
        except Exception:
            pass

    try:
        created = _rest_request("POST", "rules_registry", payload=normalized, prefer="return=representation") or []
        if isinstance(created, list) and created:
            return dict(created[0])
    except Exception:
        pass

    rules = _load_rules_registry()
    rules = [r for r in rules if str(r.get("id")) != normalized["id"]]
    rules.append(normalized)
    rules.sort(key=lambda item: (int(item.get("priority", 100) or 100), str(item.get("intent", ""))))
    _write_local_rules_registry(rules)
    return normalized


def _update_registry_rule(rule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload or {})
    merged["id"] = rule_id
    normalized = _validate_registry_rule_payload(merged)

    if sb is not None:
        try:
            updated = sb.table("rules_registry").update(normalized).eq("id", rule_id).execute().data or []
            if updated:
                return dict(updated[0])
        except Exception:
            pass

    try:
        updated = _rest_request("PATCH", f"rules_registry?id=eq.{rule_id}", payload=normalized, prefer="return=representation") or []
        if isinstance(updated, list) and updated:
            return dict(updated[0])
    except Exception:
        pass

    rules = _load_rules_registry()
    updated_rule = None
    next_rules = []
    for rule in rules:
        if str(rule.get("id")) == str(rule_id):
            updated_rule = normalized
            next_rules.append(normalized)
        else:
            next_rules.append(rule)
    if updated_rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    next_rules.sort(key=lambda item: (int(item.get("priority", 100) or 100), str(item.get("intent", ""))))
    _write_local_rules_registry(next_rules)
    return updated_rule


def _delete_registry_rule(rule_id: str) -> dict[str, Any]:
    if sb is not None:
        try:
            deleted = sb.table("rules_registry").delete().eq("id", rule_id).execute().data or []
            if deleted:
                return {"deleted": True, "id": rule_id}
        except Exception:
            pass

    try:
        _rest_request("DELETE", f"rules_registry?id=eq.{rule_id}", prefer="return=representation")
        return {"deleted": True, "id": rule_id}
    except Exception:
        pass

    rules = _load_rules_registry()
    next_rules = [rule for rule in rules if str(rule.get("id")) != str(rule_id)]
    if len(next_rules) == len(rules):
        raise HTTPException(status_code=404, detail="Rule not found")
    _write_local_rules_registry(next_rules)
    return {"deleted": True, "id": rule_id}


class RuleRegistryPayload(BaseModel):
    intent: str
    enabled: bool = True
    priority: int = 100
    patterns: list[str]
    builder: str | None = None
    sql_template: str | None = None
    description: str | None = None


@app.get("/rules_registry")
def list_rules_registry():
    return _load_rules_registry()


@app.post("/rules_registry")
def create_rules_registry(body: RuleRegistryPayload):
    return _create_registry_rule(body.model_dump())


@app.put("/rules_registry/{rule_id}")
def update_rules_registry(rule_id: str, body: RuleRegistryPayload):
    return _update_registry_rule(rule_id, body.model_dump())


@app.delete("/rules_registry/{rule_id}")
def delete_rules_registry(rule_id: str):
    return _delete_registry_rule(rule_id)


def _match_rules_registry_sql_rule(question: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    q = (question or "").strip()
    if not q:
        return None

    sorted_rules = sorted(
        [rule for rule in (rules or []) if rule and rule.get("enabled", True)],
        key=lambda rule: (int(rule.get("priority", 100) or 100), str(rule.get("intent", ""))),
    )

    for rule in sorted_rules:
        sql_template = (rule.get("sql_template") or "").strip()
        if not sql_template:
            continue
        patterns = rule.get("patterns") or []
        for pattern in patterns:
            try:
                if re.search(str(pattern), q, flags=re.IGNORECASE):
                    return rule
            except re.error:
                continue

    return None


def _intent_slug(value: str) -> str:
    txt = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "generated_rule"


def _build_question_exact_regex(question: str) -> str:
    escaped = re.escape((question or "").strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return f"(?i)^{escaped}$"


UPLOAD_CURATED_REGISTRY_QA_CASES: list[dict[str, Any]] = [
    {
        "intent": "ops_events_contracts_ratecard_customer_summary",
        "question": "Join Ops_Events and Contracts_RateCard on contract_id = contract_id and summarize key metrics by customer.",
        "required_sheets": ["Ops_Events", "Contracts_RateCard"],
    },
    {
        "intent": "ops_events_invoices_customer_summary",
        "question": "Join Ops_Events and Invoices_Synthetic on event_id = event_id and summarize key metrics by customer.",
        "required_sheets": ["Ops_Events", "Invoices_Synthetic"],
    },
    {
        "intent": "ops_events_invoices_customer_variance_summary",
        "question": "Show expected vs billed variance by customer for Ops_Events and Invoices_Synthetic.",
        "required_sheets": ["Ops_Events", "Invoices_Synthetic"],
    },
    {
        "intent": "ops_events_invoices_uninvoiced_by_customer",
        "question": "Show uninvoiced events by customer.",
        "required_sheets": ["Ops_Events", "Invoices_Synthetic"],
    },
    {
        "intent": "ops_events_invoices_invoice_coverage_by_customer",
        "question": "Show invoice coverage summary by customer.",
        "required_sheets": ["Ops_Events", "Invoices_Synthetic"],
    },
    {
        "intent": "ops_events_invoices_billed_unbilled_status_by_customer",
        "question": "Show billed vs unbilled summary by customer.",
        "required_sheets": ["Ops_Events", "Invoices_Synthetic"],
    },
]


def _extract_dataset_generated_registry_rules(relationships_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(relationships_payload, dict):
        return []
    rules = relationships_payload.get("generated_registry_rules") or []
    if not isinstance(rules, list):
        return []
    normalized_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        try:
            normalized_rules.append(_normalize_registry_rule_record(rule))
        except Exception:
            continue
    return normalized_rules


def _extract_rule_sql_requirements(sql: str) -> dict[str, set[str]]:
    requirements: dict[str, set[str]] = {}
    alias_to_sheet = _extract_alias_sheet_map(sql)

    for sheet_name in alias_to_sheet.values():
        requirements.setdefault(str(sheet_name), set())

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'", sql or "", flags=re.IGNORECASE):
        alias = match.group(1)
        column_name = match.group(2)
        sheet_name = alias_to_sheet.get(alias)
        if not sheet_name:
            continue
        requirements.setdefault(str(sheet_name), set()).add(str(column_name))

    return requirements


def _rule_sql_matches_dataset_schema(
    sql: str,
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
) -> bool:
    requirements = _extract_rule_sql_requirements(sql)
    if not requirements:
        return False

    available_sheet_set = {str(sheet).strip() for sheet in (available_sheets or []) if str(sheet).strip()}
    for sheet_name, columns in requirements.items():
        if sheet_name not in available_sheet_set:
            return False
        sampled_columns = {str(col) for col in (sheet_columns_by_sheet.get(sheet_name) or set())}
        if not columns.issubset(sampled_columns):
            return False

    return True


def _run_upload_curated_registry_qa(
    dataset_id: str,
    shared_rules: list[dict[str, Any]],
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": AUTO_UPLOAD_QA_ENABLED,
        "scope": "curated_registry",
        "status": "disabled",
        "passed": True,
        "relevant_rule_count": 0,
        "checked_rule_count": 0,
        "checks": [],
        "failures": [],
    }

    if not AUTO_UPLOAD_QA_ENABLED:
        return summary

    shared_rules = [rule for rule in (shared_rules or []) if isinstance(rule, dict) and rule.get("enabled", True)]
    rules_by_intent = {
        str(rule.get("intent") or "").strip(): rule
        for rule in shared_rules
        if str(rule.get("intent") or "").strip()
    }
    available_sheet_set = {str(sheet).strip() for sheet in (available_sheets or []) if str(sheet).strip()}

    relevant_cases: list[dict[str, Any]] = []
    for case in UPLOAD_CURATED_REGISTRY_QA_CASES:
        required_sheets = {
            str(sheet).strip()
            for sheet in (case.get("required_sheets") or [])
            if str(sheet).strip()
        }
        if required_sheets and not required_sheets.issubset(available_sheet_set):
            continue

        expected_intent = str(case.get("intent") or "").strip()
        if not expected_intent:
            continue

        expected_rule = rules_by_intent.get(expected_intent)
        if expected_rule is None:
            summary["failures"].append(
                {
                    "scope": "curated_registry",
                    "question": str(case.get("question") or "").strip(),
                    "intent": expected_intent,
                    "failure": "registry_rule_missing",
                }
            )
            summary["checks"].append(
                {
                    "scope": "curated_registry",
                    "question": str(case.get("question") or "").strip(),
                    "intent": expected_intent,
                    "status": "failed",
                    "failure": "registry_rule_missing",
                }
            )
            continue

        sql = str(expected_rule.get("sql_template") or "").strip()
        if not sql or not _rule_sql_matches_dataset_schema(sql, available_sheets, sheet_columns_by_sheet):
            continue

        relevant_cases.append(case)

    if not relevant_cases and not summary["checks"]:
        summary["status"] = "no_relevant_rules"
        return summary

    case_count = len(relevant_cases)
    if AUTO_UPLOAD_QA_MAX_CASES > 0:
        case_count = min(case_count, AUTO_UPLOAD_QA_MAX_CASES)
    cases_to_run = relevant_cases[:case_count]
    summary["relevant_rule_count"] = len(cases_to_run)

    for case in cases_to_run:
        question = str(case.get("question") or "").strip()
        expected_intent = str(case.get("intent") or "").strip()
        check = {
            "scope": "curated_registry",
            "question": question,
            "intent": expected_intent,
            "status": "passed",
        }

        failure_reason = None
        matched_rule = _match_rules_registry_sql_rule(question, shared_rules) if question else None
        if not question:
            failure_reason = "missing_sample_question"
        elif matched_rule is None:
            failure_reason = "sample_question_did_not_match_curated_registry"
        elif str(matched_rule.get("intent") or "").strip() != expected_intent:
            failure_reason = (
                f"sample_question_matched_unexpected_intent:{str(matched_rule.get('intent') or '').strip()}"
            )
        else:
            sql = str(matched_rule.get("sql_template") or "").strip()
            if not sql:
                failure_reason = "missing_sql_template"
            elif not _rule_sql_matches_dataset_schema(sql, available_sheets, sheet_columns_by_sheet):
                failure_reason = "sql_template_not_compatible_with_dataset_schema"
            elif not _is_safe_select(sql):
                failure_reason = "unsafe_sql_template"
            elif "dataset_id" not in sql.lower():
                failure_reason = "sql_template_missing_dataset_id"
            elif not _maybe_execute_generated_sql(sql, dataset_id, question):
                failure_reason = "sql_template_execution_failed"

        if failure_reason:
            check["status"] = "failed"
            check["failure"] = failure_reason
            summary["failures"].append(
                {
                    "scope": "curated_registry",
                    "question": question,
                    "intent": expected_intent,
                    "failure": failure_reason,
                }
            )

        summary["checks"].append(check)

    summary["checked_rule_count"] = len(summary["checks"])
    summary["passed"] = len(summary["failures"]) == 0
    summary["status"] = "passed" if summary["passed"] else "failed"
    return summary


def _combine_upload_qa_summaries(
    generated_summary: dict[str, Any],
    curated_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = list(generated_summary.get("checks") or []) + list(curated_summary.get("checks") or [])
    failures = list(generated_summary.get("failures") or []) + list(curated_summary.get("failures") or [])
    passed = len(failures) == 0

    if not AUTO_UPLOAD_QA_ENABLED:
        status = "disabled"
    elif not checks:
        status = "no_checks"
    else:
        status = "passed" if passed else "failed"

    return {
        "enabled": AUTO_UPLOAD_QA_ENABLED,
        "block_on_failure": AUTO_UPLOAD_QA_BLOCK_ON_FAILURE,
        "scope": "registry",
        "status": status,
        "passed": passed,
        "generated_rule_count": int(generated_summary.get("generated_rule_count") or 0),
        "relevant_curated_rule_count": int(curated_summary.get("relevant_rule_count") or 0),
        "checked_rule_count": len(checks),
        "checks": checks,
        "failures": failures,
        "generated_registry": generated_summary,
        "curated_registry": curated_summary,
    }


def _delete_uploaded_dataset(dataset_id: str) -> None:
    errors: list[str] = []
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return

    if sb is not None:
        try:
            sb.table("dataset_rows").delete().eq("dataset_id", dataset_id).execute()
        except Exception as exc:
            errors.append(f"dataset_rows_delete_failed:{exc}")
        try:
            sb.table("datasets").delete().eq("id", dataset_id).execute()
        except Exception as exc:
            errors.append(f"dataset_delete_failed:{exc}")
    else:
        try:
            _rest_request(
                "DELETE",
                "dataset_rows",
                params={"dataset_id": f"eq.{dataset_id}"},
                prefer="return=minimal",
            )
        except Exception as exc:
            errors.append(f"dataset_rows_delete_failed:{exc}")
        try:
            _rest_request(
                "DELETE",
                "datasets",
                params={"id": f"eq.{dataset_id}"},
                prefer="return=minimal",
            )
        except Exception as exc:
            errors.append(f"dataset_delete_failed:{exc}")

    if errors:
        raise RuntimeError("; ".join(errors))


def _run_upload_generated_registry_qa(dataset_id: str, generated_rules: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": AUTO_UPLOAD_QA_ENABLED,
        "scope": "generated_registry",
        "status": "disabled",
        "passed": True,
        "generated_rule_count": len(generated_rules or []),
        "checked_rule_count": 0,
        "checks": [],
        "failures": [],
    }

    if not AUTO_UPLOAD_QA_ENABLED:
        return summary

    if not generated_rules:
        summary["status"] = "no_generated_rules"
        return summary

    normalized_rules = []
    for rule in generated_rules:
        if not isinstance(rule, dict):
            continue
        try:
            normalized_rule = _normalize_registry_rule_record(rule)
            normalized_rule["sample_question"] = str(rule.get("sample_question") or "").strip()
            normalized_rules.append(normalized_rule)
        except Exception:
            continue

    if not normalized_rules:
        summary["status"] = "no_generated_rules"
        return summary

    case_count = len(normalized_rules)
    if AUTO_UPLOAD_QA_MAX_CASES > 0:
        case_count = min(case_count, AUTO_UPLOAD_QA_MAX_CASES)

    for rule in normalized_rules[:case_count]:
        question = str(rule.get("sample_question") or "").strip()
        expected_intent = str(rule.get("intent") or "").strip()
        check = {
            "question": question,
            "intent": expected_intent,
            "status": "passed",
        }

        failure_reason = None
        matched_rule = _match_rules_registry_sql_rule(question, normalized_rules) if question else None
        if not question:
            failure_reason = "missing_sample_question"
        elif matched_rule is None:
            failure_reason = "sample_question_did_not_match_generated_registry"
        elif str(matched_rule.get("intent") or "").strip() != expected_intent:
            failure_reason = (
                f"sample_question_matched_unexpected_intent:{str(matched_rule.get('intent') or '').strip()}"
            )
        else:
            sql = str(matched_rule.get("sql_template") or "").strip()
            if not sql:
                failure_reason = "missing_sql_template"
            elif not _is_safe_select(sql):
                failure_reason = "unsafe_sql_template"
            elif "dataset_id" not in sql.lower():
                failure_reason = "sql_template_missing_dataset_id"
            elif not _maybe_execute_generated_sql(sql, dataset_id, question):
                failure_reason = "sql_template_execution_failed"

        if failure_reason:
            check["status"] = "failed"
            check["failure"] = failure_reason
            summary["failures"].append(
                {
                    "scope": "generated_registry",
                    "question": question,
                    "intent": expected_intent,
                    "failure": failure_reason,
                }
            )

        summary["checks"].append(check)

    summary["checked_rule_count"] = len(summary["checks"])
    summary["passed"] = len(summary["failures"]) == 0
    summary["status"] = "passed" if summary["passed"] else "failed"
    return summary

def _is_missing_column_types_error(err: Any) -> bool:
    text = str(err).lower()
    return "column_types" in text and "does not exist" in text

def _is_missing_relationships_error(err: Any) -> bool:
    text = str(err).lower()
    return "relationships" in text and "does not exist" in text

def detect_column_types(df: pd.DataFrame):
    types = {}
    for col in df.columns:
        series = df[col].astype(str).str.strip()
        numeric_test = pd.to_numeric(series, errors="coerce")
        if numeric_test.notna().sum() > len(series) * 0.8:
            types[col] = "numeric"
            continue
        date_test = pd.to_datetime(series, errors="coerce")
        if date_test.notna().sum() > len(series) * 0.8:
            types[col] = "date"
            continue
        types[col] = "text"
    return types

def normalize_col_name(c: str) -> str:
    c = (c or "").lower().strip()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _sanitize_semantic_roles(semantic_roles: dict[str, dict[str, str]] | None) -> dict[str, dict[str, str]]:
    semantic_roles = semantic_roles or {}
    sanitized: dict[str, dict[str, str]] = {}
    total_role_pairs = [
        ("value_gap", "total_value_gap"),
        ("expected_amount", "total_expected_amount"),
        ("billed_amount", "total_billed_amount"),
    ]

    for sheet_name, role_map in semantic_roles.items():
        next_role_map = {
            str(role_name): str(column_name)
            for role_name, column_name in (role_map or {}).items()
            if str(role_name).strip() and str(column_name).strip()
        }

        for base_role, total_role in total_role_pairs:
            base_column = normalize_col_name(next_role_map.get(base_role, ""))
            total_column = normalize_col_name(next_role_map.get(total_role, ""))
            if base_column and total_column and base_column == total_column and total_column.startswith("total_"):
                next_role_map.pop(base_role, None)

        if next_role_map:
            sanitized[str(sheet_name)] = next_role_map

    return sanitized


SEMANTIC_ROLE_RULES: dict[str, dict[str, Any]] = {
    "event_id": {
        "exact": ["event_id", "eventid", "ops_event_id", "operation_event_id", "case_key", "case_id", "job_id", "shipment_id", "service_id", "transaction_id"],
        "tokens": [("event", "id"), ("case", "key"), ("case", "id"), ("shipment", "id"), ("job", "id"), ("service", "id")],
    },
    "customer_id": {
        "exact": ["customer_id", "customerid", "customer", "cust_id", "customer_code", "cust_code", "party_code", "ship_to_party", "bill_to_party", "client_id", "client_code", "account_id", "account_number"],
        "tokens": [("customer", "id"), ("customer", "code"), ("cust", "id"), ("party", "code"), ("client", "id"), ("account", "id")],
    },
    "invoice_id": {
        "exact": ["invoice_id", "invoiceid", "invoice_number", "invoice_no", "invoice_num", "bill_number", "bill_no"],
        "tokens": [("invoice", "id"), ("invoice", "number"), ("bill", "number")],
    },
    "invoice_date": {
        "exact": ["invoice_date", "invoiced_date", "bill_date", "billing_date"],
        "tokens": [("invoice", "date"), ("billing", "date"), ("bill", "date")],
        "preferred_types": {"date"},
    },
    "invoice_status": {
        "exact": ["invoice_status", "invoice_state", "bill_status", "billing_status"],
        "tokens": [("invoice", "status"), ("billing", "status"), ("bill", "status")],
    },
    "exception_type": {
        "exact": ["exception_type", "exception", "exception_kind", "exception_category", "kind"],
        "tokens": [("exception", "type"), ("exception", "kind")],
    },
    "expected_amount": {
        "exact": ["expected_amount", "expected_amt", "amount_due", "expected_value", "charge_amount"],
        "tokens": [("expected", "amount"), ("amount", "due"), ("expected", "value")],
        "preferred_types": {"numeric"},
    },
    "billed_amount": {
        "exact": ["billed_amount", "amount_billed", "invoice_amount", "billed_amt", "invoiced_amount"],
        "tokens": [("billed", "amount"), ("invoice", "amount"), ("amount", "billed")],
        "preferred_types": {"numeric"},
    },
    "value_gap": {
        "exact": ["value_gap", "value_gap_amount", "gap_value", "gap_amount", "variance", "difference", "amount_gap"],
        "tokens": [("value", "gap"), ("gap", "value"), ("gap", "amount")],
        "preferred_types": {"numeric"},
    },
    "billing_lag_days": {
        "exact": ["billing_lag_days", "lag_days", "invoice_lag_days", "billing_delay_days", "days_to_invoice"],
        "tokens": [("billing", "lag", "days"), ("lag", "days"), ("days", "invoice")],
        "preferred_types": {"numeric"},
    },
    "recommended_action": {
        "exact": ["recommended_action", "action", "next_action", "recommended_next_action", "action_recommended"],
        "tokens": [("recommended", "action"), ("next", "action")],
    },
    "service_type": {
        "exact": ["service_type", "service_category", "service_name"],
        "tokens": [("service", "type"), ("service", "category")],
    },
    "service_date": {
        "exact": ["service_date", "service_day", "event_date", "completed_date"],
        "tokens": [("service", "date"), ("service", "day"), ("event", "date")],
        "preferred_types": {"date"},
    },
    "status": {
        "exact": ["status", "state", "event_status"],
        "tokens": [("event", "status")],
    },
    "revenue": {
        "exact": ["revenue", "sales_amount", "net_sales", "total_sales", "total_price", "sales", "net_revenue"],
        "tokens": [("sales", "amount"), ("total", "sales"), ("total", "price"), ("net", "sales")],
        "preferred_types": {"numeric"},
    },
    "profit": {
        "exact": ["profit", "gross_profit", "net_profit", "profit_amount"],
        "tokens": [("gross", "profit"), ("net", "profit")],
        "preferred_types": {"numeric"},
    },
    "tax": {
        "exact": ["tax", "tax_amount", "gst", "vat"],
        "tokens": [("tax", "amount")],
        "preferred_types": {"numeric"},
    },
    "discount": {
        "exact": ["discount", "discount_amount"],
        "tokens": [("discount", "amount")],
        "preferred_types": {"numeric"},
    },
    "quantity": {
        "exact": ["quantity", "qty", "units"],
        "tokens": [("order", "qty")],
        "preferred_types": {"numeric"},
    },
    "product": {
        "exact": ["product", "product_name", "item", "item_name", "service_name"],
        "tokens": [("product", "name"), ("item", "name")],
    },
    "sku": {
        "exact": ["sku", "sku_id", "item_code", "product_code", "sku_code"],
        "tokens": [("sku", "id"), ("item", "code"), ("product", "code")],
    },
    "region": {"exact": ["region"]},
    "country": {"exact": ["country"]},
    "city": {"exact": ["city"]},
    "year": {
        "exact": ["year", "fiscal_year", "calendar_year"],
        "tokens": [("fiscal", "year"), ("calendar", "year")],
        "preferred_types": {"numeric", "date"},
    },
    "month": {
        "exact": ["month", "month_name", "fiscal_month", "calendar_month"],
        "tokens": [("month", "name"), ("fiscal", "month"), ("calendar", "month")],
    },
    "date": {
        "exact": ["date", "order_date", "transaction_date", "posting_date"],
        "tokens": [("order", "date"), ("transaction", "date"), ("posting", "date")],
        "preferred_types": {"date"},
    },
    "total_value_gap": {
        "exact": ["total_value_gap"],
        "tokens": [("total", "value", "gap")],
        "preferred_types": {"numeric"},
    },
    "total_expected_amount": {
        "exact": ["total_expected_amount"],
        "tokens": [("total", "expected", "amount")],
        "preferred_types": {"numeric"},
    },
    "total_billed_amount": {
        "exact": ["total_billed_amount"],
        "tokens": [("total", "billed", "amount")],
        "preferred_types": {"numeric"},
    },
}

SEMANTIC_ROLE_ORDER = [
    "event_id",
    "customer_id",
    "invoice_id",
    "invoice_date",
    "invoice_status",
    "exception_type",
    "expected_amount",
    "billed_amount",
    "value_gap",
    "billing_lag_days",
    "recommended_action",
    "service_type",
    "service_date",
    "total_value_gap",
    "total_expected_amount",
    "total_billed_amount",
    "revenue",
    "profit",
    "tax",
    "discount",
    "quantity",
    "product",
    "sku",
    "region",
    "country",
    "city",
    "year",
    "month",
    "date",
    "status",
]


def _normalized_name_tokens(value: str) -> set[str]:
    return {token for token in normalize_col_name(value).split("_") if token}


def _semantic_sheet_flags(sheet_name: str, normalized_columns: list[str]) -> dict[str, bool]:
    sheet_tokens = _normalized_name_tokens(sheet_name)
    column_tokens: set[str] = set()
    for column_name in normalized_columns:
        column_tokens.update(_normalized_name_tokens(column_name))

    invoice_like = bool(sheet_tokens & {"invoice", "billing", "billed", "bill"}) or any(
        column_name in {"invoice_id", "invoiceid", "invoice_number", "invoice_status", "invoice_date", "billing_status", "bill_status"}
        for column_name in normalized_columns
    )
    exception_like = bool(sheet_tokens & {"exception", "variance", "gap", "unbilled", "agent", "recon", "output"}) or any(
        column_name in {"exception_type", "exception", "value_gap", "value_gap_amount", "gap_value", "billing_lag_days", "lag_days"}
        for column_name in normalized_columns
    )
    sales_like = bool(sheet_tokens & {"sales", "sale", "revenue", "profit", "orders", "order"}) or any(
        column_name in {"revenue", "sales_amount", "net_sales", "total_sales", "profit", "gross_profit"}
        for column_name in normalized_columns
    )
    event_like = bool(sheet_tokens & {"event", "ops", "service", "operation", "ledger", "master", "raw"}) or any(
        column_name in {"event_id", "eventid", "ops_event_id", "operation_event_id", "service_date", "service_day"}
        for column_name in normalized_columns
    )

    return {
        "invoice_like": invoice_like,
        "exception_like": exception_like,
        "sales_like": sales_like,
        "event_like": event_like,
        "summary_like": any(column_name.startswith("total_") for column_name in normalized_columns),
    }


def _semantic_role_score(
    role_name: str,
    column_name: str,
    column_type: str,
    sheet_name: str,
    normalized_columns: list[str],
) -> float:
    rules = SEMANTIC_ROLE_RULES.get(role_name) or {}
    normalized = normalize_col_name(column_name)
    tokens = _normalized_name_tokens(column_name)
    score = 0.0

    exact_matches = rules.get("exact") or []
    if normalized in exact_matches:
        score = max(score, 100.0 - exact_matches.index(normalized))

    for index, token_set in enumerate(rules.get("tokens") or []):
        if all(token in tokens for token in token_set):
            score = max(score, 84.0 - index)

    flags = _semantic_sheet_flags(sheet_name, normalized_columns)

    if role_name == "invoice_status" and normalized in {"status", "state"} and flags["invoice_like"]:
        score = max(score, 80.0)
    if role_name == "status" and normalized in {"status", "state", "event_status"} and not flags["invoice_like"]:
        score = max(score, 74.0)
    if role_name == "service_date" and normalized == "date" and flags["event_like"] and not flags["invoice_like"]:
        score = max(score, 72.0)
    if role_name == "date" and normalized.endswith("date"):
        score = max(score, 70.0)
    if role_name == "customer_id" and {"party", "code"}.issubset(tokens):
        score = max(score, 82.0)
    if role_name == "event_id" and ({"case", "key"}.issubset(tokens) or {"case", "id"}.issubset(tokens)):
        score = max(score, 82.0)

    preferred_types = set(rules.get("preferred_types") or set())
    if preferred_types:
        if column_type in preferred_types:
            score += 6.0
        elif column_type and column_type != "text":
            score -= 6.0

    if role_name in {"invoice_id", "invoice_date", "invoice_status", "billed_amount"} and flags["invoice_like"]:
        score += 4.0
    if role_name in {"exception_type", "expected_amount", "value_gap", "billing_lag_days"} and flags["exception_like"]:
        score += 4.0
    if role_name in {"revenue", "profit", "tax", "discount", "quantity", "product", "sku"} and flags["sales_like"]:
        score += 4.0
    if role_name in {"event_id", "customer_id", "service_date", "service_type", "status"} and flags["event_like"]:
        score += 3.0
    if role_name.startswith("total_") and flags["summary_like"]:
        score += 3.0

    return score


def _first_matching_normalized(columns: list[str], candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(column): column for column in columns}
    for candidate in candidates:
        match = normalized_to_original.get(candidate)
        if match:
            return match
    return None


def _infer_semantic_roles_for_sheet(
    sheet_name: str,
    columns: list[str],
    column_types: dict[str, str] | None = None,
) -> dict[str, str]:
    roles: dict[str, str] = {}
    column_types = column_types or {}
    normalized_columns = [normalize_col_name(column) for column in columns]
    flags = _semantic_sheet_flags(sheet_name, normalized_columns)

    for role_name in SEMANTIC_ROLE_ORDER:
        best_column = None
        best_score = 0.0
        for column_name in columns:
            score = _semantic_role_score(
                role_name,
                column_name,
                str(column_types.get(column_name) or "text"),
                sheet_name,
                normalized_columns,
            )
            if score > best_score:
                best_score = score
                best_column = column_name

        min_score = 60.0 if role_name in {"status", "date", "month", "year"} else 70.0
        if best_column and best_score >= min_score:
            roles[role_name] = best_column

    used_columns = set(roles.values())
    if "expected_amount" not in roles and flags["exception_like"]:
        fallback = _first_matching_normalized(columns, ["amount_due", "expected_value", "amount", "total_amount", "total_value"])
        if fallback and fallback not in used_columns:
            roles["expected_amount"] = fallback
            used_columns.add(fallback)
    if "billed_amount" not in roles and flags["invoice_like"]:
        fallback = _first_matching_normalized(columns, ["amount", "total_amount", "total_value"])
        if fallback and fallback not in used_columns:
            roles["billed_amount"] = fallback
            used_columns.add(fallback)
    if "revenue" not in roles and flags["sales_like"]:
        fallback = _first_matching_normalized(columns, ["sales_amount", "net_sales", "total_sales", "total_price", "total_amount", "amount"])
        if fallback and fallback not in used_columns:
            roles["revenue"] = fallback
            used_columns.add(fallback)
    if "service_date" not in roles and flags["event_like"] and not flags["invoice_like"]:
        fallback = _first_matching_normalized(columns, ["date"])
        if fallback:
            roles["service_date"] = fallback
    if "invoice_status" not in roles and flags["invoice_like"]:
        fallback = _first_matching_normalized(columns, ["status", "state"])
        if fallback:
            roles["invoice_status"] = fallback

    return (_sanitize_semantic_roles({str(sheet_name): roles}).get(str(sheet_name)) or {})


def _infer_semantic_roles_from_sheet_metadata(
    sheet_columns_by_sheet: dict[str, set[str]],
    column_types_by_sheet: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    semantic_roles: dict[str, dict[str, str]] = {}
    column_types_by_sheet = column_types_by_sheet or {}

    for sheet_name, columns in (sheet_columns_by_sheet or {}).items():
        ordered_columns = sorted([str(column) for column in (columns or set())], key=lambda value: value.lower())
        inferred = _infer_semantic_roles_for_sheet(
            sheet_name,
            ordered_columns,
            column_types_by_sheet.get(sheet_name) or {},
        )
        if inferred:
            semantic_roles[str(sheet_name)] = inferred

    return _sanitize_semantic_roles(semantic_roles)


def _infer_semantic_roles_from_tables(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    column_types_by_sheet: dict[str, dict[str, str]] = {}
    sheet_columns_by_sheet: dict[str, set[str]] = {}
    for sheet_name, df in (tables or {}).items():
        column_types_by_sheet[str(sheet_name)] = detect_column_types(df)
        sheet_columns_by_sheet[str(sheet_name)] = {str(column) for column in df.columns}
    return _infer_semantic_roles_from_sheet_metadata(sheet_columns_by_sheet, column_types_by_sheet)


def _merge_semantic_roles(
    existing_roles: dict[str, dict[str, str]] | None,
    inferred_roles: dict[str, dict[str, str]] | None,
    *,
    overwrite: bool = False,
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    existing_roles = existing_roles or {}
    inferred_roles = inferred_roles or {}

    for sheet_name in sorted(set(existing_roles) | set(inferred_roles)):
        existing_sheet = dict(existing_roles.get(sheet_name) or {})
        inferred_sheet = dict(inferred_roles.get(sheet_name) or {})
        merged_sheet = dict(existing_sheet)
        if overwrite:
            merged_sheet.update(inferred_sheet)
        else:
            for role_name, column_name in inferred_sheet.items():
                merged_sheet.setdefault(role_name, column_name)
        if merged_sheet:
            merged[str(sheet_name)] = merged_sheet

    return _sanitize_semantic_roles(merged)


def _load_dataset_sheet_columns(dataset_id: str) -> dict[str, set[str]]:
    if psycopg is None or dict_row is None:
        raise HTTPException(status_code=500, detail="psycopg is required to backfill semantic roles")

    sql = (
        "SELECT sheet_name, array_agg(DISTINCT key ORDER BY key) AS columns "
        "FROM ("
        "  SELECT sheet_name, jsonb_object_keys(row_json) AS key "
        "  FROM dataset_rows "
        "  WHERE dataset_id = %(dataset_id)s"
        ") s "
        "GROUP BY sheet_name "
        "ORDER BY sheet_name"
    )

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"dataset_id": dataset_id})
            rows = cur.fetchall() or []

    return {
        str(row.get("sheet_name") or ""): {str(column) for column in (row.get("columns") or []) if str(column).strip()}
        for row in rows
        if str(row.get("sheet_name") or "").strip()
    }


def _update_dataset_relationships_payload(dataset_id: str, relationships_payload: dict[str, Any]) -> None:
    if sb is not None:
        try:
            sb.table("datasets").update({"relationships": relationships_payload}).eq("id", dataset_id).execute()
            return
        except Exception as exc:
            if _is_missing_relationships_error(exc):
                raise HTTPException(status_code=500, detail="datasets.relationships column is required to store semantic roles")
            raise HTTPException(status_code=502, detail=f"Supabase dataset update failed: {exc}")

    try:
        _rest_request(
            "PATCH",
            "datasets",
            params={"id": f"eq.{dataset_id}"},
            payload={"relationships": relationships_payload},
            prefer="return=minimal",
        )
    except HTTPException as exc:
        if _is_missing_relationships_error(exc.detail):
            raise HTTPException(status_code=500, detail="datasets.relationships column is required to store semantic roles")
        raise


def _load_datasets_for_semantic_backfill(dataset_id: str | None = None) -> list[dict[str, Any]]:
    select_cols = "id,name,relationships"
    if sb is not None:
        query = sb.table("datasets").select(select_cols)
        if dataset_id:
            query = query.eq("id", dataset_id)
        return query.order("created_at", desc=True).execute().data or []

    params = {"select": select_cols, "order": "created_at.desc"}
    if dataset_id:
        params["id"] = f"eq.{dataset_id}"
    rows = _rest_request("GET", "datasets", params=params) or []
    return rows if isinstance(rows, list) else []


def _backfill_semantic_roles(dataset_id: str | None = None, *, overwrite: bool = False) -> list[dict[str, Any]]:
    datasets = _load_datasets_for_semantic_backfill(dataset_id)
    results: list[dict[str, Any]] = []

    for dataset in datasets:
        current_payload = dataset.get("relationships") if isinstance(dataset.get("relationships"), dict) else {}
        existing_roles = current_payload.get("semantic_roles") if isinstance(current_payload, dict) else {}
        sheet_columns_by_sheet = _load_dataset_sheet_columns(str(dataset.get("id") or ""))
        inferred_roles = _infer_semantic_roles_from_sheet_metadata(sheet_columns_by_sheet)
        merged_roles = _merge_semantic_roles(existing_roles, inferred_roles, overwrite=overwrite)

        if merged_roles == (existing_roles or {}):
            results.append(
                {
                    "dataset_id": str(dataset.get("id") or ""),
                    "name": str(dataset.get("name") or ""),
                    "updated": False,
                    "sheet_count": len(merged_roles),
                    "role_count": sum(len(roles) for roles in merged_roles.values()),
                }
            )
            continue

        next_payload = dict(current_payload or {})
        next_payload["tables"] = list(next_payload.get("tables") or sheet_columns_by_sheet.keys())
        next_payload["semantic_roles"] = merged_roles
        _update_dataset_relationships_payload(str(dataset.get("id") or ""), next_payload)
        results.append(
            {
                "dataset_id": str(dataset.get("id") or ""),
                "name": str(dataset.get("name") or ""),
                "updated": True,
                "sheet_count": len(merged_roles),
                "role_count": sum(len(roles) for roles in merged_roles.values()),
            }
        )

    return results


def _sanitize_derived_metrics_by_sheet(derived_metrics_by_sheet: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    sanitized: dict[str, dict[str, str]] = {}
    for sheet_name, sheet_metrics in (derived_metrics_by_sheet or {}).items():
        if not isinstance(sheet_metrics, dict):
            continue
        next_sheet = {
            str(metric_name): str(formula)
            for metric_name, formula in sheet_metrics.items()
            if str(metric_name).strip() and str(formula).strip()
        }
        if next_sheet:
            sanitized[str(sheet_name)] = next_sheet
    return sanitized


def _existing_derived_metrics_by_sheet(
    relationships_payload: dict[str, Any] | None,
    available_sheets: list[str],
) -> dict[str, dict[str, str]]:
    relationships_payload = relationships_payload if isinstance(relationships_payload, dict) else {}
    by_sheet = _sanitize_derived_metrics_by_sheet(relationships_payload.get("derived_metrics_by_sheet") or {})
    if by_sheet:
        return by_sheet

    flattened = relationships_payload.get("derived_metrics") or {}
    if isinstance(flattened, dict) and len(available_sheets) == 1:
        single_sheet = str(available_sheets[0])
        inferred_flattened = {
            str(metric_name): str(formula)
            for metric_name, formula in flattened.items()
            if str(metric_name).strip() and str(formula).strip()
        }
        if inferred_flattened:
            return {single_sheet: inferred_flattened}
    return {}


def _merge_derived_metrics_by_sheet(
    existing_metrics: dict[str, dict[str, str]] | None,
    inferred_metrics: dict[str, dict[str, str]] | None,
    *,
    overwrite: bool = False,
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    existing_metrics = _sanitize_derived_metrics_by_sheet(existing_metrics)
    inferred_metrics = _sanitize_derived_metrics_by_sheet(inferred_metrics)

    for sheet_name in sorted(set(existing_metrics) | set(inferred_metrics)):
        existing_sheet = dict(existing_metrics.get(sheet_name) or {})
        inferred_sheet = dict(inferred_metrics.get(sheet_name) or {})
        merged_sheet = dict(existing_sheet)
        if overwrite:
            merged_sheet.update(inferred_sheet)
        else:
            for metric_name, formula in inferred_sheet.items():
                merged_sheet.setdefault(metric_name, formula)
        if merged_sheet:
            merged[str(sheet_name)] = merged_sheet

    return merged


def _load_datasets_for_derived_metric_backfill(dataset_id: str | None = None) -> list[dict[str, Any]]:
    select_cols = "id,name,columns,column_types,relationships"
    if sb is not None:
        query = sb.table("datasets").select(select_cols)
        if dataset_id:
            query = query.eq("id", dataset_id)
        return query.order("created_at", desc=True).execute().data or []

    params = {"select": select_cols, "order": "created_at.desc"}
    if dataset_id:
        params["id"] = f"eq.{dataset_id}"
    rows = _rest_request("GET", "datasets", params=params) or []
    return rows if isinstance(rows, list) else []


def _backfill_derived_metrics(dataset_id: str | None = None, *, overwrite: bool = False) -> list[dict[str, Any]]:
    datasets = _load_datasets_for_derived_metric_backfill(dataset_id)
    results: list[dict[str, Any]] = []

    for dataset in datasets:
        dataset_id_text = str(dataset.get("id") or "")
        current_payload = dataset.get("relationships") if isinstance(dataset.get("relationships"), dict) else {}
        sheet_columns_by_sheet = _load_dataset_sheet_columns(dataset_id_text)
        available_sheets = [
            str(sheet_name)
            for sheet_name in list((current_payload or {}).get("tables") or sheet_columns_by_sheet.keys())
            if str(sheet_name).strip()
        ]
        existing_metrics = _existing_derived_metrics_by_sheet(current_payload, available_sheets)

        inferred_metadata = _build_sql_engine_metadata(
            [str(column_name) for column_name in list(dataset.get("columns") or []) if str(column_name).strip()],
            dict(dataset.get("column_types") or {}),
            available_sheets,
            sheet_columns_by_sheet,
            list((current_payload or {}).get("relationships") or []),
            relationships_payload=None,
        )
        inferred_metrics = _sanitize_derived_metrics_by_sheet(inferred_metadata.get("derived_metrics_by_sheet") or {})
        merged_metrics = _merge_derived_metrics_by_sheet(existing_metrics, inferred_metrics, overwrite=overwrite)
        flattened_metrics = flatten_derived_metrics_by_sheet(merged_metrics)

        changed = merged_metrics != existing_metrics
        if changed:
            next_payload = dict(current_payload or {})
            next_payload["tables"] = available_sheets
            next_payload["derived_metrics_by_sheet"] = merged_metrics
            next_payload["derived_metrics"] = flattened_metrics
            _update_dataset_relationships_payload(dataset_id_text, next_payload)

        results.append(
            {
                "dataset_id": dataset_id_text,
                "name": str(dataset.get("name") or ""),
                "updated": changed,
                "sheet_count": len(merged_metrics),
                "metric_count": sum(len(sheet_metrics) for sheet_metrics in merged_metrics.values()),
            }
        )

    return results

def looks_like_id_col(col: str) -> bool:
    coln = normalize_col_name(col)
    return (
        coln.endswith("_id") or coln in {"id", "key"} or
        "code" in coln or "customer" in coln or "cust" in coln or
        "invoice" in coln or "order" in coln or "product" in coln or
        "vendor" in coln or "supplier" in coln or "sku" in coln or
        coln.endswith("_no") or coln.endswith("_number")
    )

def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_col_name(a), normalize_col_name(b)).ratio()

def _norm_value_series(s: pd.Series) -> pd.Series:
    s2 = s.dropna().astype(str).str.strip()
    s2 = s2.str.replace(r"\s+", " ", regex=True).str.lower()
    return s2

def value_overlap(a: pd.Series, b: pd.Series, max_set: int = 5000) -> float:
    a_vals = _norm_value_series(a).unique()[:max_set]
    b_vals = _norm_value_series(b).unique()[:max_set]
    if len(a_vals) == 0 or len(b_vals) == 0:
        return 0.0
    aset = set(a_vals)
    bset = set(b_vals)
    inter = len(aset & bset)
    denom = min(len(aset), len(bset))
    return inter / denom if denom else 0.0

def profile_table(df: pd.DataFrame) -> dict:
    prof = {"columns": {}}
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        n_non_null = len(non_null)
        n_unique = int(non_null.astype(str).nunique()) if n_non_null else 0
        unique_ratio = float(n_unique / max(n_non_null, 1))
        prof["columns"][c] = {
            "norm": normalize_col_name(c),
            "null_pct": float(s.isna().mean()),
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "is_id_like": looks_like_id_col(c),
        }
    return prof

def join_type(lu: float, ru: float) -> str:
    if ru > 0.9 and lu < 0.9:
        return "many-to-one"
    if lu > 0.9 and ru < 0.9:
        return "one-to-many"
    if lu > 0.9 and ru > 0.9:
        return "one-to-one"
    return "many-to-many"

def discover_joins(tables: Dict[str, pd.DataFrame], min_score: float = 0.70, top_k_per_pair: int = 2) -> dict:
    profiles = {t: profile_table(df) for t, df in tables.items()}
    rels = []
    names = list(tables.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t1, t2 = names[i], names[j]
            df1, df2 = tables[t1], tables[t2]
            p1, p2 = profiles[t1], profiles[t2]

            candidates = []
            for c1 in df1.columns:
                for c2 in df2.columns:
                    ns = name_similarity(c1, c2)
                    if not (p1["columns"][c1]["is_id_like"] or p2["columns"][c2]["is_id_like"] or ns > 0.75):
                        continue
                    ov = value_overlap(df1[c1], df2[c2])
                    id_bonus = 0.10 if (p1["columns"][c1]["is_id_like"] and p2["columns"][c2]["is_id_like"]) else 0.0
                    score = max(0.0, min(1.0, 0.45 * ov + 0.45 * ns + id_bonus))
                    if score >= min_score:
                        candidates.append((score, c1, c2, ov, ns))

            candidates.sort(reverse=True, key=lambda x: x[0])
            for score, c1, c2, ov, ns in candidates[:top_k_per_pair]:
                rels.append({
                    "left": t1, "left_col": c1,
                    "right": t2, "right_col": c2,
                    "type": join_type(p1["columns"][c1]["unique_ratio"], p2["columns"][c2]["unique_ratio"]),
                    "score": round(score, 3),
                    "overlap": round(ov, 3),
                    "name_sim": round(ns, 3),
                })

    rels.sort(key=lambda r: r["score"], reverse=True)
    return {"tables": names, "relationships": rels}

# -------------------------
# UPLOAD ENDPOINT
# -------------------------
@app.post("/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_name: str = Form(default=""),
    include_assumptions_sheet: bool = Form(default=True),
):
    content = await file.read()
    filename = file.filename or "uploaded.csv"

    # Read CSV as single virtual sheet, Excel as multi-sheet
    try:
        if filename.lower().endswith(".csv"):
            tables = {"sheet1": pd.read_csv(BytesIO(content)).fillna("")}
        else:
            raw_sheets = pd.read_excel(BytesIO(content), sheet_name=None)
            tables = {str(sheet): df.fillna("") for sheet, df in raw_sheets.items() if df is not None and not df.empty}
            if not tables:
                raise HTTPException(status_code=400, detail="Excel file has no non-empty sheets")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    tables, cleaning_report = clean_and_standardize_tables(
        tables,
        include_assumptions_sheet=include_assumptions_sheet,
    )
    if not tables:
        raise HTTPException(status_code=400, detail="Upload produced no usable sheets after cleaning")

    # Use first sheet for legacy columns metadata; keep all sheets in relationships
    first_sheet_name = next(iter(tables.keys()))
    first_df = tables[first_sheet_name]
    cols = list(first_df.columns)
    column_types_by_sheet = {str(sheet_name): detect_column_types(df) for sheet_name, df in tables.items()}
    column_types = column_types_by_sheet.get(first_sheet_name) or detect_column_types(first_df)
    available_sheets = list(tables.keys())
    sheet_columns_by_sheet = {sheet_name: set(map(str, df.columns)) for sheet_name, df in tables.items()}
    derived_metrics_by_sheet = detect_derived_metrics_by_sheet(
        {str(sheet_name): [str(column) for column in df.columns] for sheet_name, df in tables.items()},
        column_types_by_sheet=column_types_by_sheet,
    )
    row_count = int(sum(df.shape[0] for df in tables.values()))
    col_count = int(first_df.shape[1])
    join_graph = discover_joins(tables, min_score=0.70, top_k_per_pair=2)
    relationships_payload = dict(join_graph)
    relationships_payload["semantic_roles"] = _infer_semantic_roles_from_tables(tables)
    relationships_payload["derived_metrics"] = flatten_derived_metrics_by_sheet(derived_metrics_by_sheet)
    relationships_payload["derived_metrics_by_sheet"] = derived_metrics_by_sheet
    relationships_payload["cleaning_report"] = cleaning_report

    name = dataset_name.strip() or os.path.splitext(filename)[0]
    dataset_insert = {
        "name": name,
        "original_filename": filename,
        "row_count": row_count,
        "col_count": col_count,
        "columns": cols,
        "column_types": column_types,
        "relationships": relationships_payload,
    }

    # 1) Insert dataset metadata
    if sb is not None:
        try:
            ds_res = sb.table("datasets").insert(dataset_insert).execute()
        except Exception as e:
            # fallback if schema not upgraded yet
            if _is_missing_column_types_error(e) or _is_missing_relationships_error(e):
                fallback_insert = {k: v for k, v in dataset_insert.items() if k not in {"column_types", "relationships"}}
                ds_res = sb.table("datasets").insert(fallback_insert).execute()
                dataset_id = ds_res.data[0]["id"]
                # best-effort update for whichever columns exist
                try:
                    sb.table("datasets").update({"column_types": column_types, "relationships": relationships_payload}).eq("id", dataset_id).execute()
                except Exception:
                    try:
                        sb.table("datasets").update({"column_types": column_types}).eq("id", dataset_id).execute()
                    except Exception:
                        pass
                    try:
                        sb.table("datasets").update({"relationships": relationships_payload}).eq("id", dataset_id).execute()
                    except Exception:
                        pass
            else:
                raise HTTPException(status_code=502, detail=f"Supabase dataset insert failed: {e}")
        if 'dataset_id' not in locals():
            dataset_id = ds_res.data[0]["id"]
    else:
        try:
            ds_rows = _rest_request("POST", "datasets", payload=dataset_insert, prefer="return=representation")
        except HTTPException as e:
            if _is_missing_column_types_error(e.detail) or _is_missing_relationships_error(e.detail):
                fallback_insert = {k: v for k, v in dataset_insert.items() if k not in {"column_types", "relationships"}}
                ds_rows = _rest_request("POST", "datasets", payload=fallback_insert, prefer="return=representation")
                dataset_id = ds_rows[0]["id"]
                for upd in ({"column_types": column_types}, {"relationships": relationships_payload}):
                    try:
                        _rest_request("PATCH", "datasets", params={"id": f"eq.{dataset_id}"}, payload=upd, prefer="return=minimal")
                    except Exception:
                        pass
            else:
                raise
        if not isinstance(ds_rows, list) or not ds_rows:
            raise HTTPException(status_code=502, detail="Supabase returned no dataset row after insert")
        if 'dataset_id' not in locals():
            dataset_id = ds_rows[0]["id"]

    # 2) Insert rows in chunks with sheet_name
    payload = []
    for sheet_name, df in tables.items():
        for i in range(int(df.shape[0])):
            row_dict = {c: _clean(df.iloc[i][c]) for c in df.columns}
            payload.append({
                "dataset_id": dataset_id,
                "sheet_name": sheet_name,
                "row_index": i,
                "row_json": row_dict,
                "row_text": _row_to_text(row_dict)
            })

    CHUNK = 500
    for start in range(0, len(payload), CHUNK):
        chunk = payload[start:start+CHUNK]
        if sb is not None:
            sb.table("dataset_rows").insert(chunk).execute()
        else:
            _rest_request("POST", "dataset_rows", payload=chunk, prefer="return=minimal")

    generated_registry_rules: list[dict[str, Any]] = []
    upload_qa: dict[str, Any] = {
        "enabled": AUTO_UPLOAD_QA_ENABLED,
        "block_on_failure": AUTO_UPLOAD_QA_BLOCK_ON_FAILURE,
        "scope": "registry",
        "status": "not_run",
        "passed": True,
        "generated_rule_count": 0,
        "relevant_curated_rule_count": 0,
        "checked_rule_count": 0,
        "checks": [],
        "failures": [],
    }
    try:
        generated_registry_rules = _generate_dataset_scoped_registry_rules(
            str(dataset_id),
            cols,
            column_types,
            available_sheets,
            sheet_columns_by_sheet,
            (relationships_payload.get("relationships") or []),
        )
    except Exception:
        generated_registry_rules = []

    if generated_registry_rules:
        relationships_payload["generated_registry_rules"] = generated_registry_rules
        if sb is not None:
            try:
                sb.table("datasets").update({"relationships": relationships_payload}).eq("id", dataset_id).execute()
            except Exception:
                pass
        else:
            try:
                _rest_request(
                    "PATCH",
                    "datasets",
                    params={"id": f"eq.{dataset_id}"},
                    payload={"relationships": relationships_payload},
                    prefer="return=minimal",
                )
            except Exception:
                pass

    try:
        generated_upload_qa = _run_upload_generated_registry_qa(str(dataset_id), generated_registry_rules)
        curated_upload_qa = _run_upload_curated_registry_qa(
            str(dataset_id),
            _load_rules_registry(),
            available_sheets,
            sheet_columns_by_sheet,
        )
        upload_qa = _combine_upload_qa_summaries(generated_upload_qa, curated_upload_qa)
    except Exception as exc:
        upload_qa = {
            "enabled": AUTO_UPLOAD_QA_ENABLED,
            "block_on_failure": AUTO_UPLOAD_QA_BLOCK_ON_FAILURE,
            "scope": "registry",
            "status": "error",
            "passed": False,
            "generated_rule_count": len(generated_registry_rules),
            "relevant_curated_rule_count": 0,
            "checked_rule_count": 0,
            "checks": [],
            "failures": [{"scope": "registry", "failure": f"upload_qa_error:{exc}"}],
        }

    if AUTO_UPLOAD_QA_BLOCK_ON_FAILURE and upload_qa.get("status") in {"failed", "error"}:
        cleanup_error = None
        try:
            _delete_uploaded_dataset(str(dataset_id))
        except Exception as exc:
            cleanup_error = str(exc)

        detail: dict[str, Any] = {
            "message": "Upload rejected because registry QA failed.",
            "dataset_id": dataset_id,
            "name": name,
            "filename": filename,
            "upload_qa": upload_qa,
        }
        if cleanup_error:
            detail["cleanup_error"] = cleanup_error
        raise HTTPException(status_code=400, detail=detail)

    return {
        "dataset_id": dataset_id,
        "name": name,
        "filename": filename,
        "rows": row_count,
        "sheets": list(tables.keys()),
        "columns": cols,
        "relationships": relationships_payload,
        "cleaning_report": cleaning_report,
        "generated_registry_rule_count": len(generated_registry_rules),
        "generated_registry_questions": [rule.get("sample_question") for rule in generated_registry_rules if rule.get("sample_question")],
        "upload_qa": upload_qa,
    }

# -------------------------
# LIST DATASETS
# -------------------------
@app.get("/datasets")
def list_datasets():
    select_cols = "id,name,row_count,col_count,columns,relationships,created_at"
    fallback_cols = "id,name,row_count,col_count,columns,created_at"
    if sb is not None:
        try:
            return sb.table("datasets").select(select_cols).order("created_at", desc=True).execute().data
        except Exception as e:
            if _is_missing_relationships_error(e):
                return sb.table("datasets").select(fallback_cols).order("created_at", desc=True).execute().data
            raise HTTPException(status_code=502, detail=f"Supabase datasets list failed: {e}")
    try:
        return _rest_request("GET", "datasets", params={"select": select_cols, "order": "created_at.desc"})
    except HTTPException as e:
        if _is_missing_relationships_error(e.detail):
            return _rest_request("GET", "datasets", params={"select": fallback_cols, "order": "created_at.desc"})
        raise


@app.get("/datasets/{dataset_id}/frames")
def get_dataset_frames(dataset_id: str):
    return _load_dataset_frame_payload(dataset_id)


class SemanticRoleBackfillBody(BaseModel):
    dataset_id: str | None = None
    overwrite: bool = False


@app.post("/datasets/backfill_semantic_roles")
def backfill_semantic_roles(body: SemanticRoleBackfillBody):
    results = _backfill_semantic_roles(body.dataset_id, overwrite=bool(body.overwrite))
    return {
        "dataset_count": len(results),
        "updated_dataset_count": sum(1 for item in results if item.get("updated")),
        "skipped_dataset_count": sum(1 for item in results if not item.get("updated")),
        "results": results,
    }


class DerivedMetricBackfillBody(BaseModel):
    dataset_id: str | None = None
    overwrite: bool = False


@app.post("/datasets/backfill_derived_metrics")
def backfill_derived_metrics(body: DerivedMetricBackfillBody):
    results = _backfill_derived_metrics(body.dataset_id, overwrite=bool(body.overwrite))
    return {
        "dataset_count": len(results),
        "updated_dataset_count": sum(1 for item in results if item.get("updated")),
        "skipped_dataset_count": sum(1 for item in results if not item.get("updated")),
        "results": results,
    }

# -------------------------
# OPTION 1: ASK (ROW SAMPLING)
# -------------------------
class AskBody(BaseModel):
    dataset_id: str
    question: str
    max_rows: int = 500

@app.post("/ask")
def ask(body: AskBody):
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    if sb is not None:
        meta = sb.table("datasets").select("name,row_count,col_count,columns").eq("id", body.dataset_id).single().execute().data
        limit = min(max(int(body.max_rows), 10), 500)
        rows = (
            sb.table("dataset_rows")
            .select("row_index,row_text")
            .eq("dataset_id", body.dataset_id)
            .order("row_index")
            .range(0, limit - 1)
            .execute()
            .data
        ) or []
    else:
        meta = _rest_request("GET", "datasets", params={"select": "name,row_count,col_count,columns", "id": f"eq.{body.dataset_id}", "limit": 1})
        meta = meta[0] if meta else {}
        rows = _rest_request(
            "GET",
            "dataset_rows",
            params={
                "select": "row_index,row_text",
                "dataset_id": f"eq.{body.dataset_id}",
                "order": "row_index.asc",
                "limit": min(max(body.max_rows, 10), 500),
            },
        ) or []

    context = "\n".join([f"Row {r['row_index']}: {r['row_text']}" for r in rows])
    instructions = (
        "You are a data analyst. Answer using ONLY the provided rows. "
        "If you can't answer from these rows, say what extra rows/filters are needed."
    )
    input_text = (
        f"Dataset: {meta.get('name')} | rows={meta.get('row_count')} | cols={meta.get('col_count')}\n"
        f"Columns: {meta.get('columns')}\n\n"
        f"Question: {q}\n\nRows:\n{context}"
    )

    try:
        resp = oa.responses.create(model=TEXT_MODEL, instructions=instructions, input=input_text, max_output_tokens=700)
        answer = (getattr(resp, "output_text", "") or "").strip() or "No answer text returned by the model. Please try again or reduce sampled rows."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "rows_used": len(rows)}

# -------------------------
# OPTION 2: ASK SQL
# -------------------------
def _is_safe_select(sql: str) -> bool:
    if sqlparse is None:
        return False
    s = sql.strip()
    if not s:
        return False
    if ";" in s:
        return False
    parsed = sqlparse.parse(s)
    if len(parsed) != 1:
        return False
    first = parsed[0].token_first(skip_cm=True)
    if not first:
        return False
    if first.value.upper() not in ("SELECT", "WITH"):
        return False
    upper = s.upper()
    for bad in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE", "CREATE"]:
        if bad in upper:
            return False
    return True


def _sanitize_numeric_casts(sql: str) -> str:
    # Rewrite fragile numeric casts so currency/commas in text values do not fail execution.
    pattern = re.compile(r"\(([^()]+)\)::numeric", re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        expr = (match.group(1) or "").strip()
        if not expr or "regexp_replace" in expr.lower():
            return match.group(0)
        return f"(NULLIF(regexp_replace(({expr}), '[^0-9.\\-]', '', 'g'), '')::numeric)"

    return pattern.sub(_repl, sql)


def _sanitize_date_casts(sql: str) -> str:
    # Rewrite fragile date casts from text into guarded CASE/TO_DATE expressions.
    pattern = re.compile(r"\(([^()]+)\)::date", re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        expr = (match.group(1) or "").strip()
        if not expr:
            return match.group(0)
        lower_expr = expr.lower()
        if "to_date(" in lower_expr or "::date" in lower_expr:
            return match.group(0)
        return (
            "(CASE "
            f"WHEN ({expr}) ~ '^\\d{{4}}-\\d{{1,2}}-\\d{{1,2}}$' THEN to_date(({expr}), 'YYYY-MM-DD') "
            f"WHEN ({expr}) ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$' AND split_part(({expr}), '/', 1)::int > 12 THEN to_date(({expr}), 'DD/MM/YYYY') "
            f"WHEN ({expr}) ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$' THEN to_date(({expr}), 'MM/DD/YYYY') "
            f"WHEN ({expr}) ~ '^\\d{{1,2}}-\\d{{1,2}}-\\d{{4}}$' AND split_part(({expr}), '-', 1)::int > 12 THEN to_date(({expr}), 'DD-MM-YYYY') "
            f"WHEN ({expr}) ~ '^\\d{{1,2}}-\\d{{1,2}}-\\d{{4}}$' THEN to_date(({expr}), 'MM-DD-YYYY') "
            "ELSE NULL END)"
        )

    return pattern.sub(_repl, sql)


def _sanitize_to_date_calls(sql: str) -> str:
    # Replace TO_DATE(text, '...') with robust parser that supports mixed source formats.
    pattern = re.compile(r"TO_DATE\(\s*([^,]+?)\s*,\s*'[^']+'\s*\)", re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        expr = (match.group(1) or "").strip()
        if not expr:
            return match.group(0)
        return _date_parse_expr(expr)

    return pattern.sub(_repl, sql)


def _normalize_date_literals_in_sql(sql: str) -> str:
    # Normalize quoted date literals to ISO to avoid Postgres DateStyle ambiguity.
    s = sql or ""

    # Fix malformed literals such as '4/1/2022"' produced by model quoting glitches.
    s = re.sub(r"'(\d{1,2}[/-]\d{1,2}[/-]\d{4})\"+'", r"'\1'", s)

    pattern = re.compile(r"'\s*(\d{1,2})([/-])(\d{1,2})\2(\d{4})\s*'")

    def _repl(match: re.Match) -> str:
        a = int(match.group(1))
        b = int(match.group(3))
        y = int(match.group(4))

        # Heuristic: if first number > 12 it's day-first; otherwise prefer month-first.
        if a > 12:
            d, m = a, b
        else:
            m, d = a, b

        try:
            iso = date(y, m, d).isoformat()
        except Exception:
            return match.group(0)
        return f"'{iso}'"

    return pattern.sub(_repl, s)


def _rewrite_sql_for_date_errors(sql: str) -> str:
    rewritten = _sanitize_to_date_calls(sql)
    rewritten = _sanitize_date_casts(rewritten)
    rewritten = _normalize_date_literals_in_sql(rewritten)
    return rewritten


def _is_aggregate_like_sql(sql: str) -> bool:
    upper = (sql or "").upper()
    has_agg = any(fn in upper for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])
    has_group = " GROUP BY " in upper
    has_distinct = "SELECT DISTINCT" in upper
    return (has_agg or has_group or has_distinct) and "SELECT *" not in upper


def _strip_limit_clause(sql: str) -> str:
    stripped = re.sub(r"\s+LIMIT\s+(?:%\(\w+\)s|\d+)\b", "", sql, flags=re.IGNORECASE).strip()
    return stripped


def _prepare_sql_for_psycopg(sql: str) -> str:
    # psycopg treats % as placeholder syntax; escape literal % while keeping named placeholders.
    prepared = (sql or "").replace("%", "%%")
    prepared = re.sub(r"%%\((\w+)\)s", r"%(\1)s", prepared)
    return prepared



def _extract_year_from_question(question: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", question or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _build_exec_params(
    sql: str,
    dataset_id: str,
    limit: int,
    question: str,
    extra_params: dict[str, Any] | None = None,
) -> dict:
    params = {"dataset_id": dataset_id, "limit": limit}
    placeholders = set(re.findall(r"%\((\w+)\)s", sql or ""))
    if "year" in placeholders:
        year = _extract_year_from_question(question)
        params["year"] = year
    if extra_params:
        for key, value in extra_params.items():
            if key in {"dataset_id", "limit"}:
                continue
            params[key] = value
    return params


def _should_apply_global_top_1_limit(force_top_1: bool, query_spec: QuerySpec | None) -> bool:
    if not force_top_1:
        return False
    if query_spec is None:
        return True
    return str(getattr(query_spec, "query_type", "") or "").strip().lower() != "ranking"


def _sanitize_unbound_year_placeholder(sql: str, question: str) -> str:
    # If model emits %(year)s but user didn't ask for a year, broaden filter instead of failing.
    if _extract_year_from_question(question) is not None:
        return sql
    return re.sub(r"=\s*%\(year\)s", " IS NOT NULL", sql or "", flags=re.IGNORECASE)


def _build_planner_debug_payload(
    *,
    question: str,
    route_intent: str,
    route_source: str,
    planner_confidence: Any,
    planner_summary: dict[str, Any] | None,
    planner_query_spec: dict[str, Any] | None,
    planner_query_shape: str | None,
    planner_warnings: list[str],
    planner_reasons: list[str],
    extracted_filters: list[dict[str, Any]],
    inferred_value_mappings: list[dict[str, Any]],
    candidate_sheets: list[dict[str, Any]],
    selected_metric: dict[str, Any] | None,
    candidate_metrics: list[dict[str, Any]],
    selected_dimensions: list[str],
    answer_type: str | None,
    sql: str,
    failure_reason: str | None = None,
    requested_dimension: str | None = None,
    requested_metric: str | None = None,
    resolved_dimension: str | None = None,
    resolved_metric: str | None = None,
    missing_dimension_reason: str | None = None,
    missing_metric_reason: str | None = None,
    fallback_blocked_reason: str | None = None,
    available_dimensions: list[str] | None = None,
    available_metrics: list[str] | None = None,
) -> dict[str, Any]:
    target_sheet = ""
    query_type = ""
    pattern_id = ""
    selected_columns: list[str] = []
    selected_columns_exist = False
    pre_aggregated_sheet = False
    requested_projection_columns: list[str] = []
    matched_existing_columns: list[str] = []
    aggregation_override_skipped_reason = ""
    matched_keywords: dict[str, Any] = {}
    auto_corrections: list[str] = []
    validation_result = ""
    if isinstance(planner_summary, dict):
        target_sheet = str(planner_summary.get("target_sheet") or "").strip()
        if not query_type:
            query_type = str(planner_summary.get("query_type") or "").strip()
        selected_columns = [str(item) for item in list(planner_summary.get("columns") or []) if str(item).strip()]
    if not target_sheet and isinstance(planner_query_spec, dict):
        target_sheet = str(planner_query_spec.get("target_sheet") or "").strip()
    if isinstance(planner_query_spec, dict):
        query_type = str(planner_query_spec.get("query_type") or "").strip()
        pattern_id = str(planner_query_spec.get("pattern_id") or "").strip()
        selected_columns = [str(item) for item in list(planner_query_spec.get("columns") or []) if str(item).strip()]
        debug_payload = planner_query_spec.get("debug") or {}
        if isinstance(debug_payload, dict):
            matched_keywords = dict(debug_payload.get("matched_keywords") or {})
            auto_corrections = [str(item) for item in list(debug_payload.get("auto_corrections") or []) if str(item).strip()]
            validation_result = str(debug_payload.get("validation_result") or "").strip()
            selected_columns_exist = bool(debug_payload.get("selected_columns_exist"))
            pre_aggregated_sheet = bool(debug_payload.get("pre_aggregated_sheet"))
            requested_projection_columns = [str(item) for item in list(debug_payload.get("requested_projection_columns") or []) if str(item).strip()]
            matched_existing_columns = [str(item) for item in list(debug_payload.get("matched_existing_columns") or []) if str(item).strip()]
            aggregation_override_skipped_reason = str(debug_payload.get("aggregation_override_skipped_reason") or "").strip()
    return {
        "question": question,
        "interpreted_intent": route_intent,
        "route_source": route_source,
        "query_shape": planner_query_shape,
        "query_type": query_type,
        "query_classification": query_type or planner_query_shape,
        "pattern_id": pattern_id,
        "answer_type": answer_type,
        "planner_confidence": planner_confidence,
        "selected_sheet": target_sheet,
        "selected_columns": selected_columns,
        "selected_columns_exist": selected_columns_exist,
        "pre_aggregated_sheet": pre_aggregated_sheet,
        "requested_projection_columns": requested_projection_columns,
        "matched_existing_columns": matched_existing_columns,
        "aggregation_override_skipped_reason": aggregation_override_skipped_reason,
        "candidate_sheets": candidate_sheets,
        "selected_metric": selected_metric,
        "candidate_metrics": candidate_metrics,
        "selected_dimensions": selected_dimensions,
        "requested_dimension": requested_dimension,
        "requested_metric": requested_metric,
        "resolved_dimension": resolved_dimension,
        "resolved_metric": resolved_metric,
        "missing_dimension_reason": missing_dimension_reason,
        "missing_metric_reason": missing_metric_reason,
        "fallback_blocked_reason": fallback_blocked_reason,
        "available_dimensions": list(available_dimensions or []),
        "available_metrics": list(available_metrics or []),
        "filters": extracted_filters,
        "value_matches": inferred_value_mappings,
        "matched_intent_keywords": matched_keywords,
        "auto_corrections": auto_corrections,
        "validation_result": validation_result,
        "planner_warnings": planner_warnings,
        "planner_reasons": planner_reasons,
        "final_sql": sql,
        "failure_reason": failure_reason,
    }


def _repair_sql_after_execution_error(sql: str, question: str, error_text: str) -> str:
    if not ALLOW_LLM_SQL_REPAIR:
        return ""
    instructions = (
        "You are fixing a PostgreSQL SELECT query that failed at execution time. "
        "Return ONE corrected SQL query only, no markdown, no explanation, no semicolon. "
        "Keep it SELECT-only and keep parameter placeholders like %(dataset_id)s and %(limit)s when present. "
        "If aggregates are used, ensure all non-aggregated selected expressions are included in GROUP BY. "
        "Use dataset_rows and row_json->>'ColumnName' patterns."
    )
    prompt = (
        f"User question: {question}\n"
        f"Execution error: {error_text}\n"
        f"Failed SQL:\n{sql}\n\n"
        "Return corrected SQL only."
    )

    resp = oa.responses.create(
        model=TEXT_MODEL,
        instructions=instructions,
        input=prompt,
        max_output_tokens=500,
    )

    fixed_sql = (getattr(resp, "output_text", "") or "").strip()
    if not fixed_sql:
        output_items = getattr(resp, "output", None) or []
        parts = []
        for item in output_items:
            contents = getattr(item, "content", None) or []
            for content in contents:
                text_value = getattr(content, "text", None)
                if text_value:
                    parts.append(str(text_value))
                else:
                    maybe_text = getattr(content, "output_text", None)
                    if maybe_text:
                        parts.append(str(maybe_text))
        fixed_sql = "\n".join([p.strip() for p in parts if p and p.strip()]).strip()

    if fixed_sql.startswith("```"):
        fixed_sql = re.sub(r"^```[a-zA-Z]*\n?", "", fixed_sql).strip()
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3].strip()

    return fixed_sql


def _extract_alias_sheet_map(sql: str) -> dict[str, str]:
    alias_to_sheet: dict[str, str] = {}
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+dataset_rows\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b", sql or "", flags=re.IGNORECASE):
        alias = match.group(1)
        sheet_match = re.search(rf"\b{re.escape(alias)}\.sheet_name\s*=\s*'([^']+)'", sql or "", flags=re.IGNORECASE)
        if sheet_match:
            alias_to_sheet[alias] = sheet_match.group(1)
    return alias_to_sheet


def _find_sql_schema_issues(sql: str, question: str, sheet_columns_by_sheet: dict[str, set[str]]) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    alias_to_sheet = _extract_alias_sheet_map(sql)
    normalized_columns = {
        sheet: {normalize_col_name(col): col for col in (cols or set())}
        for sheet, cols in (sheet_columns_by_sheet or {}).items()
    }

    if "dataset_rows" in (sql or "").lower() and not alias_to_sheet:
        issues.append("SQL does not clearly constrain dataset_rows aliases to concrete sheet_name values.")

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'", sql or "", flags=re.IGNORECASE):
        alias = match.group(1)
        column_name = match.group(2)
        sheet_name = alias_to_sheet.get(alias)
        if not sheet_name:
            continue
        sampled_columns = sheet_columns_by_sheet.get(sheet_name) or set()
        if column_name in sampled_columns:
            continue
        normalized_match = normalized_columns.get(sheet_name, {}).get(normalize_col_name(column_name))
        if normalized_match:
            issue = f"Alias {alias} on sheet '{sheet_name}' uses column '{column_name}', but sampled schema shows '{normalized_match}'."
        else:
            available_text = ", ".join(sorted(sampled_columns)[:12]) if sampled_columns else "no sampled columns"
            issue = f"Alias {alias} on sheet '{sheet_name}' uses missing column '{column_name}'. Sampled columns include: {available_text}."
        if issue not in seen:
            seen.add(issue)
            issues.append(issue)

    for match in re.finditer(
        r"\b(SUM|AVG)\s*\([^)]*?\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'[^)]*\)",
        sql or "",
        flags=re.IGNORECASE | re.DOTALL,
    ):
        agg_fn = (match.group(1) or "").upper()
        alias = match.group(2)
        column_name = match.group(3)
        sheet_name = alias_to_sheet.get(alias)
        if not sheet_name:
            continue

        normalized = normalize_col_name(column_name)
        if looks_like_id_col(column_name) or normalized in {"event", "eventid"}:
            sampled_columns = sheet_columns_by_sheet.get(sheet_name) or set()
            numeric_candidates = [
                col for col in sorted(sampled_columns)
                if normalize_col_name(col) in {
                    "expected_amount",
                    "expected_amt",
                    "billed_amount",
                    "amount",
                    "total_value",
                    "total_amount",
                    "total_price",
                    "totalprice",
                    "sales_amount",
                    "salesamount",
                    "net_sales",
                    "netsales",
                    "unit_price",
                    "unitprice",
                    "revenue",
                    "gross_profit",
                    "value_gap",
                    "qty",
                    "quantity",
                }
            ]
            candidate_text = ", ".join(numeric_candidates[:8]) if numeric_candidates else "no obvious amount columns"
            issue = (
                f"{agg_fn} is using identifier-like column '{column_name}' on sheet '{sheet_name}'. "
                f"Use a real numeric business metric instead, such as: {candidate_text}."
            )
            if issue not in seen:
                seen.add(issue)
                issues.append(issue)

    if "completed" in (question or "").lower():
        for match in re.finditer(r"row_json->>'([^']+)'.{0,120}?=\s*'completed'", sql or "", flags=re.IGNORECASE | re.DOTALL):
            column_name = match.group(1)
            normalized = normalize_col_name(column_name)
            if normalized not in {"status", "event_status", "state", "invoice_status"}:
                issue = f"The literal 'completed' is being compared against column '{column_name}' instead of a status-like column."
                if issue not in seen:
                    seen.add(issue)
                    issues.append(issue)

    if "sent" in (question or "").lower():
        for match in re.finditer(r"row_json->>'([^']+)'.{0,120}?=\s*'sent'", sql or "", flags=re.IGNORECASE | re.DOTALL):
            column_name = match.group(1)
            normalized = normalize_col_name(column_name)
            if normalized not in {"status", "invoice_status", "invoice_state", "state"}:
                issue = f"The literal 'sent' is being compared against column '{column_name}' instead of an invoice-status-like column."
                if issue not in seen:
                    seen.add(issue)
                    issues.append(issue)

    return issues


def _repair_sql_with_schema_context(
    sql: str,
    question: str,
    issues: list[str],
    sheet_schema_context: str,
    rel_desc: str,
) -> str:
    if not ALLOW_LLM_SQL_REPAIR:
        return ""
    instructions = (
        "You are fixing a PostgreSQL SELECT query before execution. "
        "Return ONE corrected SQL query only, no markdown, no explanation, no semicolon. "
        "Keep it SELECT-only and keep parameter placeholders like %(dataset_id)s and %(limit)s when present. "
        "Use only real sheet_name values and only row_json keys that exist in the provided schema. "
        "Preserve the user's intent and prefer the detected relationships."
    )
    prompt = (
        f"User question: {question}\n"
        f"Schema issues found:\n- " + "\n- ".join(issues) + "\n\n"
        f"Per-sheet schema:\n{sheet_schema_context}\n\n"
        f"Detected relationships:\n{rel_desc}\n\n"
        f"Failed SQL candidate:\n{sql}\n\n"
        "Return corrected SQL only."
    )

    resp = oa.responses.create(
        model=TEXT_MODEL,
        instructions=instructions,
        input=prompt,
        max_output_tokens=600,
    )

    fixed_sql = (getattr(resp, "output_text", "") or "").strip()
    if not fixed_sql:
        output_items = getattr(resp, "output", None) or []
        parts = []
        for item in output_items:
            contents = getattr(item, "content", None) or []
            for content in contents:
                text_value = getattr(content, "text", None)
                if text_value:
                    parts.append(str(text_value))
                else:
                    maybe_text = getattr(content, "output_text", None)
                    if maybe_text:
                        parts.append(str(maybe_text))
        fixed_sql = "\n".join([p.strip() for p in parts if p and p.strip()]).strip()

    if fixed_sql.startswith("```"):
        fixed_sql = re.sub(r"^```[a-zA-Z]*\n?", "", fixed_sql).strip()
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3].strip()

    return fixed_sql


def _pick_schema_replacement_column(sheet_name: str, missing_column: str, sampled_columns: set[str], question: str) -> str | None:
    normalized_to_original = {normalize_col_name(col): col for col in (sampled_columns or set())}
    normalized_missing = normalize_col_name(missing_column)
    if normalized_missing in normalized_to_original:
        return normalized_to_original[normalized_missing]

    q_norm = normalize_col_name(question)
    sheet_norm = normalize_col_name(sheet_name)

    amount_priority = [
        "expected_amount",
        "billed_amount",
        "total_amount",
        "total_value",
        "total_price",
        "sales_amount",
        "net_sales",
        "amount",
        "unit_price",
    ]
    if "invoice" in sheet_norm or "billed" in q_norm or "invoice" in q_norm:
        amount_priority = [
            "billed_amount",
            "expected_amount",
            "total_amount",
            "total_value",
            "total_price",
            "sales_amount",
            "net_sales",
            "amount",
            "unit_price",
        ]

    replacement_groups = {
        "amount": amount_priority,
        "value": amount_priority + ["value_gap"],
        "revenue": ["total_price", "sales_amount", "net_sales", "total_value", "total_amount", "amount", "billed_amount", "expected_amount", "unit_price"],
        "sales": ["total_price", "sales_amount", "net_sales", "total_value", "total_amount", "amount", "billed_amount", "expected_amount", "unit_price"],
        "profit": ["gross_profit", "value_gap", "expected_amount", "billed_amount"],
        "customer": ["customer", "customer_id", "customerid"],
        "invoice_amount": ["billed_amount", "expected_amount", "total_amount", "total_price", "amount"],
    }

    for prefix, candidates in replacement_groups.items():
        if normalized_missing == prefix or normalized_missing.startswith(prefix + "_"):
            for candidate in candidates:
                if candidate in normalized_to_original:
                    return normalized_to_original[candidate]

    return None


def _find_closing_paren(sql: str, open_paren_index: int) -> int | None:
    depth = 0
    in_single = False
    i = open_paren_index
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            in_single = not in_single
        elif not in_single:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _rewrite_identifier_metric_aggregates(
    sql: str,
    question: str,
    sheet_columns_by_sheet: dict[str, set[str]],
) -> tuple[str, bool]:
    alias_to_sheet = _extract_alias_sheet_map(sql)
    changed = False
    pieces: list[str] = []
    cursor = 0

    for match in re.finditer(r"\b(SUM|AVG)\s*\(", sql or "", flags=re.IGNORECASE):
        open_paren_index = match.end() - 1
        close_paren_index = _find_closing_paren(sql, open_paren_index)
        if close_paren_index is None:
            continue

        expr = sql[open_paren_index + 1:close_paren_index]
        updated_expr = expr

        for col_match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'", expr, flags=re.IGNORECASE):
            alias = col_match.group(1)
            column_name = col_match.group(2)
            if not looks_like_id_col(column_name) and normalize_col_name(column_name) not in {"event", "eventid"}:
                continue

            sheet_name = alias_to_sheet.get(alias)
            if not sheet_name:
                continue

            sampled_columns = sheet_columns_by_sheet.get(sheet_name) or set()
            replacement = _pick_schema_replacement_column(sheet_name, "amount", sampled_columns, question)
            if not replacement or replacement == column_name:
                continue

            updated_expr = updated_expr.replace(
                f"{alias}.row_json->>'{column_name}'",
                f"{alias}.row_json->>'{replacement}'",
            )

        if updated_expr != expr:
            pieces.append(sql[cursor:open_paren_index + 1])
            pieces.append(updated_expr)
            cursor = close_paren_index
            changed = True

    if not changed:
        return sql, False

    pieces.append(sql[cursor:])
    return "".join(pieces), True


def _locally_rewrite_sql_columns_to_sampled_schema(
    sql: str,
    question: str,
    sheet_columns_by_sheet: dict[str, set[str]],
) -> tuple[str, bool]:
    alias_to_sheet = _extract_alias_sheet_map(sql)
    changed = False

    def _repl(match: re.Match) -> str:
        nonlocal changed
        alias = match.group(1)
        column_name = match.group(2)
        sheet_name = alias_to_sheet.get(alias)
        if not sheet_name:
            return match.group(0)

        sampled_columns = sheet_columns_by_sheet.get(sheet_name) or set()
        if column_name in sampled_columns:
            return match.group(0)

        replacement = _pick_schema_replacement_column(sheet_name, column_name, sampled_columns, question)
        if not replacement or replacement == column_name:
            return match.group(0)

        changed = True
        return f"{alias}.row_json->>'{replacement}'"

    rewritten = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.row_json->>'([^']+)'",
        _repl,
        sql or "",
        flags=re.IGNORECASE,
    )
    rewritten, agg_changed = _rewrite_identifier_metric_aggregates(rewritten, question, sheet_columns_by_sheet)
    return rewritten, (changed or agg_changed)


def _repair_sql_until_schema_valid(
    sql: str,
    question: str,
    sheet_columns_by_sheet: dict[str, set[str]],
    sheet_schema_context: str,
    rel_desc: str,
    max_rounds: int = 2,
) -> tuple[str, list[str], bool]:
    candidate = sql or ""
    changed = False

    for _ in range(max_rounds):
        issues = _find_sql_schema_issues(candidate, question, sheet_columns_by_sheet)
        if not issues:
            return candidate, [], changed

        locally_rewritten, local_changed = _locally_rewrite_sql_columns_to_sampled_schema(
            candidate,
            question,
            sheet_columns_by_sheet,
        )
        if local_changed:
            candidate = locally_rewritten
            changed = True
            continue

        repaired_sql = _repair_sql_with_schema_context(candidate, question, issues, sheet_schema_context, rel_desc)
        if not repaired_sql:
            return candidate, issues, changed

        repaired_sql = repaired_sql.strip()
        if not repaired_sql:
            return candidate, issues, changed

        if repaired_sql != candidate.strip():
            changed = True
        candidate = repaired_sql

    final_issues = _find_sql_schema_issues(candidate, question, sheet_columns_by_sheet)
    return candidate, final_issues, changed


def _generate_sql_with_llm(system: str, user: str) -> str:
    if not ALLOW_DIRECT_LLM_SQL:
        raise RuntimeError("Direct LLM SQL generation is disabled.")
    resp = oa.responses.create(model=TEXT_MODEL, instructions=system, input=user, max_output_tokens=500)
    sql = (getattr(resp, "output_text", "") or "").strip()
    if not sql:
        output_items = getattr(resp, "output", None) or []
        parts = []
        for item in output_items:
            contents = getattr(item, "content", None) or []
            for content in contents:
                text_value = getattr(content, "text", None)
                if text_value:
                    parts.append(str(text_value))
                else:
                    maybe_text = getattr(content, "output_text", None)
                    if maybe_text:
                        parts.append(str(maybe_text))
        sql = "\n".join([p.strip() for p in parts if p and p.strip()]).strip()

    if not sql and TEXT_MODEL != "gpt-4o-mini":
        fallback_resp = oa.responses.create(
            model="gpt-4o-mini",
            instructions=system,
            input=user,
            max_output_tokens=500,
        )
        sql = (getattr(fallback_resp, "output_text", "") or "").strip()

    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql).strip()
        if sql.endswith("```"):
            sql = sql[:-3].strip()

    return sql


def _is_effectively_empty_aggregate_rows(rows: list[dict[str, Any]]) -> bool:
    if not rows or len(rows) != 1 or not isinstance(rows[0], dict):
        return False
    row = rows[0]
    if not row:
        return True

    saw_nonblank = False
    for value in row.values():
        if value is None:
            continue
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            saw_nonblank = True
            if value != 0:
                return False
            continue

        text = str(value).strip().lower()
        if not text or text in {"none", "null"}:
            continue
        saw_nonblank = True
        if text not in {"0", "0.0", "0.00", "0.000"}:
            return False

    return True if saw_nonblank else True


def _has_unbilled_total_value_shape(rows: list[dict[str, Any]]) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return False
    row = rows[0]
    return any(key in row for key in {"total_unbilled_value", "total_value"})


def _sql_uses_value_metric(sql: str) -> bool:
    metric_columns = {
        "expected_amount",
        "expected_amt",
        "billed_amount",
        "invoice_amount",
        "amount",
        "total_amount",
        "total_value",
        "price_per_unit",
        "unit_price",
        "unit_rate",
        "rate",
        "price",
        "value_gap",
    }
    for match in re.finditer(r"row_json->>'([^']+)'", sql or "", flags=re.IGNORECASE):
        if normalize_col_name(match.group(1)) in metric_columns:
            return True
    return False


def _has_invoiced_amount_shape(rows: list[dict[str, Any]]) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return False
    return "invoice_amount" in rows[0]


def _has_positive_invoiced_amount_rows(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("invoice_amount")
        if value is None:
            continue
        try:
            if float(str(value).strip().replace(",", "")) > 0:
                return True
        except Exception:
            continue
    return False


def _coerce_row_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)

    normalized = _normalize_numeric_text(str(value))
    if normalized is None or normalized == "":
        return None
    try:
        return float(normalized)
    except Exception:
        return None


def _is_metric_like_key(key_name: str) -> bool:
    normalized = normalize_col_name(key_name)
    if not normalized:
        return False
    metric_prefixes = ("sum_", "total_", "avg_", "min_", "max_", "count_")
    metric_suffixes = (
        "_count",
        "_amount",
        "_revenue",
        "_sales",
        "_profit",
        "_gap",
        "_value",
        "_qty",
        "_quantity",
    )
    metric_names = {
        "count",
        "row_count",
        "value_gap",
        "value_gap_amount",
        "variance",
        "difference",
        "amount",
        "revenue",
        "sales",
        "profit",
        "quantity",
        "qty",
    }
    return (
        normalized.startswith(metric_prefixes)
        or normalized.endswith(metric_suffixes)
        or normalized in metric_names
    )


def _focused_metric_keys_for_question(row: dict[str, Any], question: str) -> list[str]:
    q = normalize_col_name(question or "")
    key_map = {str(key): normalize_col_name(str(key)) for key in row.keys()}

    focus_tokens: list[str] = []
    if any(token in q for token in ["gap", "variance", "difference"]):
        focus_tokens.extend(["gap", "variance", "difference"])
    elif any(token in q for token in ["revenue", "sales"]):
        focus_tokens.extend(["revenue", "sales", "price", "amount"])
    elif "profit" in q:
        focus_tokens.extend(["profit", "margin"])
    elif any(token in q for token in ["amount", "billed", "expected", "value"]):
        focus_tokens.extend(["amount", "billed", "expected", "value"])
    elif any(token in q for token in ["quantity", "qty", "count"]):
        focus_tokens.extend(["quantity", "qty", "count"])

    if not focus_tokens:
        return []

    return [key for key, normalized in key_map.items() if any(token in normalized for token in focus_tokens)]


def _drop_zero_only_grouped_rows(rows: list[dict[str, Any]], question: str, sql: str) -> list[dict[str, Any]]:
    if "group by" not in str(sql or "").lower():
        return rows

    filtered_rows: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            filtered_rows.append(row)
            continue

        metric_values: dict[str, float] = {}
        has_dimension_value = False

        for key, value in row.items():
            key_name = str(key)
            normalized_key_name = normalize_col_name(key_name)
            if normalized_key_name in {"row_index", "rank"}:
                continue

            if _is_metric_like_key(key_name):
                numeric_value = _coerce_row_number(value)
                if numeric_value is not None:
                    metric_values[key_name] = numeric_value
                continue

            if value is None:
                continue

            text_value = str(value).strip()
            if text_value:
                has_dimension_value = True

        if not has_dimension_value:
            continue

        if not metric_values:
            filtered_rows.append(row)
            continue

        focused_metric_keys = _focused_metric_keys_for_question(row, question)
        focused_metric_values = [metric_values[key] for key in focused_metric_keys if key in metric_values]
        if focused_metric_values and all(abs(number) < 1e-9 for number in focused_metric_values):
            continue

        if all(abs(number) < 1e-9 for number in metric_values.values()):
            continue

        filtered_rows.append(row)

    return filtered_rows


def _split_sql_csv(expr: str) -> list[str]:
    parts = []
    buf = []
    depth = 0
    in_single = False
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "'":
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if not in_single:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_sql_sections(sql: str) -> tuple[str, str, str] | None:
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    select_expr = m.group(1).strip()
    from_idx = m.end() - len("FROM")
    head = sql[:from_idx]
    tail = sql[from_idx:]
    return head, select_expr, tail


def _is_aggregate_expr(expr: str) -> bool:
    e = expr.upper()
    return any(fn in e for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])


def _strip_alias(expr: str) -> str:
    e = re.sub(r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", expr, flags=re.IGNORECASE).strip()
    if e != expr.strip():
        return e
    m = re.match(r"^(.*?)(?:\s+[A-Za-z_][A-Za-z0-9_]*)\s*$", e)
    if m and ")" in m.group(1):
        return m.group(1).strip()
    return e


def _repair_group_by_sql_locally(sql: str) -> str:
    sections = _extract_sql_sections(sql)
    if not sections:
        return sql
    _head, select_expr, tail = sections
    select_items = _split_sql_csv(select_expr)
    if not select_items:
        return sql

    non_agg_items = []
    has_agg = False
    for item in select_items:
        core = _strip_alias(item)
        if _is_aggregate_expr(core):
            has_agg = True
        else:
            non_agg_items.append(core)

    if not has_agg or not non_agg_items:
        return sql

    gb_match = re.search(r"\bGROUP\s+BY\b", tail, flags=re.IGNORECASE)
    order_match = re.search(r"\bORDER\s+BY\b", tail, flags=re.IGNORECASE)
    limit_match = re.search(r"\bLIMIT\b", tail, flags=re.IGNORECASE)

    if gb_match:
        gb_start = gb_match.end()
        gb_end = len(tail)
        for m in [order_match, limit_match]:
            if m and m.start() > gb_start:
                gb_end = min(gb_end, m.start())
        current_gb = tail[gb_start:gb_end].strip()
        gb_items = _split_sql_csv(current_gb)
        gb_norm = {re.sub(r"\s+", " ", g).strip().lower() for g in gb_items}
        changed = False
        for item in non_agg_items:
            norm = re.sub(r"\s+", " ", item).strip().lower()
            if norm not in gb_norm:
                gb_items.append(item)
                gb_norm.add(norm)
                changed = True
        if not changed:
            return sql
        new_group = " GROUP BY " + ", ".join(gb_items) + " "
        repaired = tail[:gb_match.start()] + new_group + tail[gb_end:]
        return sql[:len(sql) - len(tail)] + repaired

    # No GROUP BY but aggregate + non-aggregate select exists.
    insert_pos = len(tail)
    for m in [order_match, limit_match]:
        if m:
            insert_pos = min(insert_pos, m.start())
    group_clause = " GROUP BY " + ", ".join(non_agg_items) + " "
    repaired_tail = tail[:insert_pos] + group_clause + tail[insert_pos:]
    return sql[:len(sql) - len(tail)] + repaired_tail


def _extract_requested_top_n(question: str) -> int | None:
    q = (question or "").lower()
    word_to_num = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    # Handles: "top 3", "topp 3", "bottom 5"
    m = re.search(r"\b(?:top+|bottom)\s*(\d+)\b", q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    # Handles: "top3", "topp10", "bottom2"
    m = re.search(r"\b(?:top+|bottom)(\d+)\b", q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    # Handles: "top three", "bottom five"
    m = re.search(r"\b(?:top+|bottom)\s*(one|two|three|four|five|six|seven|eight|nine|ten)\b", q)
    if m:
        return word_to_num.get(m.group(1))

    return None


def _extract_lag_days_threshold(question: str) -> int | None:
    q = question or ""
    patterns = [
        r"billing[_\s-]*lag[_\s-]*days\s*(?:>|>=|over|above|more than|greater than)\s*(\d+)",
        r"lag[_\s-]*days\s*(?:>|>=|over|above|more than|greater than)\s*(\d+)",
        r"(?:over|above|more than|greater than)\s*(\d+)\s*days",
    ]
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _is_billing_lag_value_gap_question(question: str) -> bool:
    q = (question or "").lower()
    mentions_lag = any(token in q for token in ["billing_lag_days", "billing lag days", "lag days", "billing lag"])
    mentions_gap = any(token in q for token in ["value gap", "value_gap", "gap amount", "value gap amount"])
    return mentions_lag and mentions_gap


def _is_billing_lag_question(question: str) -> bool:
    q = (question or "").lower()
    mentions_lag = any(token in q for token in ["billing_lag_days", "billing lag days", "lag days", "billing lag"])
    mentions_gap = any(token in q for token in ["value gap", "value_gap", "gap amount", "value gap amount"])
    return mentions_lag and not mentions_gap


def _question_requires_positive_gap(question: str) -> bool:
    q = (question or "").lower()
    if _extract_value_gap_condition(question) is not None:
        return False
    positive_gap_phrases = [
        "positive value gap",
        "positive gap",
        "non-zero value gap",
        "non zero value gap",
        "non-zero gap",
        "non zero gap",
        "value gap > 0",
        "value_gap > 0",
        "gap amount > 0",
        "value gap greater than 0",
        "value gap more than 0",
        "gap amount greater than 0",
        "gap amount more than 0",
    ]
    return any(token in q for token in positive_gap_phrases)


def _extract_value_gap_condition(question: str) -> tuple[str, str] | None:
    q = (question or "").lower()
    patterns = [
        r"value[_\s-]*gap(?:[_\s-]*amount)?\s*(=|==|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)",
        r"gap[_\s-]*amount\s*(=|==|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)",
        r"value[_\s-]*gap(?:[_\s-]*amount)?\s+is\s+(-?\d+(?:\.\d+)?)",
    ]

    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        if idx < 2:
            op = "=" if match.group(1) == "==" else match.group(1)
            return op, match.group(2)
        return "=", match.group(1)

    if any(token in q for token in ["value_gap = 0", "value gap = 0", "gap amount = 0", "zero gap", "no gap"]):
        return "=", "0"

    return None


def _sheet_has_any_column(sheet_columns_by_sheet: dict[str, set[str]], sheet_name: str, candidates: list[str]) -> bool:
    cols = {normalize_col_name(c) for c in (sheet_columns_by_sheet.get(sheet_name) or set())}
    return any(candidate in cols for candidate in candidates)


def _pick_sheet_column(sheet_columns_by_sheet: dict[str, set[str]], sheet_name: str, candidates: list[str]) -> str | None:
    original_cols = list(sheet_columns_by_sheet.get(sheet_name) or set())
    normalized_to_original = {normalize_col_name(c): c for c in original_cols}
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


QUESTION_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "get", "give", "i", "id", "in", "is", "it",
    "list", "me", "of", "on", "or", "please", "show", "that", "the", "this", "to", "use", "what",
    "where", "which", "with",
}


def _question_terms(text: str) -> set[str]:
    normalized = normalize_col_name(text or "")
    if not normalized:
        return set()
    return {part for part in normalized.split("_") if part and part not in QUESTION_STOPWORDS and len(part) > 1}


def _score_sheet_for_question(sheet_name: str, columns: set[str], question: str) -> tuple[float, list[str]]:
    q_terms = _question_terms(question)
    sheet_terms = _question_terms(sheet_name)
    score = 0.0
    relevant_columns: list[str] = []

    sheet_overlap = q_terms & sheet_terms
    score += 4.0 * len(sheet_overlap)

    for col in sorted(columns):
        col_terms = _question_terms(col)
        overlap = q_terms & col_terms
        if overlap:
            score += 2.0 * len(overlap)
            relevant_columns.append(col)

        normalized_col = normalize_col_name(col)
        normalized_question = normalize_col_name(question)
        if normalized_col and normalized_col in normalized_question:
            score += 3.0
            if col not in relevant_columns:
                relevant_columns.append(col)

    if any(term in q_terms for term in {"billing", "lag"}) and any(normalize_col_name(c) in {"billing_lag_days", "lag_days", "invoice_lag_days"} for c in columns):
        score += 5.0
    if any(term in q_terms for term in {"gap", "amount", "value"}) and any(normalize_col_name(c) in {"value_gap", "value_gap_amount", "variance", "difference", "billed_amount", "expected_amount"} for c in columns):
        score += 4.0

    if not relevant_columns:
        relevant_columns = sorted(list(columns))[:6]

    return score, relevant_columns[:8]


def _rank_sheets_for_question(available_sheets: list[str], sheet_columns_by_sheet: dict[str, set[str]], question: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for sheet in available_sheets or []:
        cols = sheet_columns_by_sheet.get(sheet) or set()
        score, relevant_columns = _score_sheet_for_question(sheet, cols, question)
        ranked.append({
            "sheet": sheet,
            "score": round(score, 3),
            "relevant_columns": relevant_columns,
        })
    ranked.sort(key=lambda item: (-item["score"], item["sheet"].lower()))
    return ranked


def _format_sheet_schema_context(available_sheets: list[str], sheet_columns_by_sheet: dict[str, set[str]], question: str) -> str:
    ranked = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)
    if not ranked:
        return "No sampled sheet schema is available."

    lines = []
    for item in ranked[:6]:
        cols = ", ".join(item["relevant_columns"]) if item["relevant_columns"] else "no sampled columns"
        lines.append(f"- {item['sheet']} (score={item['score']}): {cols}")
    return "Question-ranked sheet candidates and sampled columns:\n" + "\n".join(lines)


def _build_sql_generation_messages(
    question: str,
    cols: list[str],
    col_types: dict[str, str],
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
    *,
    analyze_all_rows: bool,
    requested_top_n: int | None,
    force_top_1: bool,
    explicit_sheet: str | None,
) -> tuple[str, str, str, str]:
    if len(available_sheets) == 1:
        sheet_hint = f"Only one valid sheet_name exists: '{available_sheets[0]}'."
    elif explicit_sheet:
        sheet_hint = f"User explicitly requested sheet_name '{explicit_sheet}'. Restrict to that sheet only."
    elif available_sheets:
        sheet_hint = "Valid sheet_name values are: " + ", ".join([f"'{s}'" for s in available_sheets])
    else:
        sheet_hint = "No explicit sheet_name guidance available; avoid hardcoded placeholders."

    col_desc = _format_legacy_metadata_context(cols, col_types, available_sheets, question)
    sheet_schema_context = _format_sheet_schema_context(available_sheets, sheet_columns_by_sheet, question)
    rel_desc = "\n".join([
        f"- sheet '{r['left']}' column '{r['left_col']}' joins to sheet '{r['right']}' column '{r['right_col']}' (type={r['type']}, score={r['score']})"
        for r in relationships[:15]
    ]) or "No reliable relationships detected."

    system = (
        "You are a PostgreSQL expert. Generate ONE safe SQL query (SELECT-only) for Supabase Postgres.\n"
        "All uploaded data lives in one table: dataset_rows(dataset_id uuid, sheet_name text, row_index int, row_json jsonb, row_text text).\n"
        "For single-sheet queries, query dataset_rows once and filter by dataset_id.\n"
        "For multi-sheet joins, self-join dataset_rows with aliases, and filter each alias by both dataset_id and sheet_name.\n"
        "Use row_json->>'ColumnName' to access values.\n"
        "For text join keys, normalize both sides with NULLIF(LOWER(TRIM(...)), '') before comparison.\n"
        "Example normalized join predicate: NULLIF(LOWER(TRIM(a.row_json->>'event_id')), '') = NULLIF(LOWER(TRIM(b.row_json->>'event_id')), '').\n"
        "For anti-joins (missing matches), use LEFT JOIN plus joined_alias.row_index IS NULL in WHERE.\n"
        "Do not assume nonexistent keys (for example billing_status); for invoice state use invoice-sheet keys like invoice_status or status.\n"
        "Business terms may differ from physical column names: revenue or sales often maps to columns like TotalPrice, SalesAmount, NetSales, TotalAmount, or Amount depending on the dataset.\n"
        "Numeric values may include symbols like '$' or commas.\n"
        "When casting to numeric, sanitize first, e.g. SUM(NULLIF(regexp_replace((row_json->>'Profit'), '[^0-9.\\-]', '', 'g'), '')::numeric).\n"
        "When filtering by year, parse date text and use EXTRACT(YEAR FROM parsed_date) = <year>; do not use LIKE on date strings.\n"
        f"{sheet_hint}\n"
        f"{sheet_schema_context}\n"
        "If the dataset has multiple sheets, treat per-sheet sampled schema as authoritative and ignore conflicting first-sheet metadata.\n"
        "Never use placeholder values like 'YourSheetName'.\n"
        "Do NOT invent joins. Prefer the detected relationships below.\n"
        f"Detected relationships:\n{rel_desc}\n"
        "Always include a dataset_id filter for every alias used.\n"
        "Return only the SQL text, nothing else.\n"
    )

    user = (
        f"Legacy first-sheet metadata context:\n{col_desc}\n\n"
        f"Per-sheet sampled schema for this question:\n{sheet_schema_context}\n\n"
        f"User question: {question}\n"
        "Rules:\n"
        "- Output ONE SQL query only (no semicolon)\n"
        "- Base table is dataset_rows\n"
        "- For joins, use aliases like s, c, p on dataset_rows\n"
        "- Every alias must include dataset_id = %(dataset_id)s and its own sheet_name filter\n"
        "- Join key comparisons should be null-safe and normalized: NULLIF(LOWER(TRIM(lhs)), '') = NULLIF(LOWER(TRIM(rhs)), '')\n"
        "- For missing/unmatched rows, use LEFT JOIN and test joined alias with row_index IS NULL\n"
        "- Use only real sheet_name values provided above; do not invent values\n"
        "- If the user asks about revenue or sales contribution, prefer real metric columns such as TotalPrice, SalesAmount, NetSales, TotalAmount, Amount, or UnitPrice before inventing a Revenue column\n"
        "- If question mentions a specific year, filter using parsed date year (EXTRACT), not string LIKE\n"
        f"- {'Add LIMIT %(limit)s and ensure ordering matches the user\'s top/bottom intent' if requested_top_n else ('Do not add LIMIT unless user explicitly asks for one' if analyze_all_rows and not force_top_1 else 'Add LIMIT %(limit)s')}\n"
        "- If question asks for highest/lowest/top/bottom one result, return only one row\n"
    )

    return system, user, sheet_schema_context, rel_desc


def _pick_registry_group_dimension(
    left_sheet: str,
    right_sheet: str,
    sheet_columns_by_sheet: dict[str, set[str]],
) -> tuple[str, str]:
    dimension_candidates = [
        ("customer", ["customer", "customer_id", "customerid", "cust_id"]),
        ("service type", ["service_type", "service", "type"]),
        ("status", ["status", "state"]),
    ]
    for label, candidates in dimension_candidates:
        if _sheet_has_any_column(sheet_columns_by_sheet, left_sheet, candidates) or _sheet_has_any_column(sheet_columns_by_sheet, right_sheet, candidates):
            return label, candidates[0]
    return "join key", "join_key"


def _build_auto_registry_candidate_questions(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for rel in relationships:
        left = str(rel.get("left") or "").strip()
        right = str(rel.get("right") or "").strip()
        left_col = str(rel.get("left_col") or "").strip()
        right_col = str(rel.get("right_col") or "").strip()
        if not left or not right or not left_col or not right_col:
            continue
        pair_key = (left.lower(), left_col.lower(), right.lower(), right_col.lower())
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        group_label, group_slug = _pick_registry_group_dimension(left, right, sheet_columns_by_sheet)
        question = f"Join {left} and {right} on {left_col} = {right_col} and summarize key metrics by {group_label}."
        exact_pattern = _build_question_exact_regex(question)
        loose_pattern = f"(?i){re.escape(left.replace('_', ' '))}.*{re.escape(right.replace('_', ' '))}.*{group_slug.replace('_', r'[_ ]')}"
        candidates.append(
            {
                "intent": f"auto_{_intent_slug(left)}_{_intent_slug(right)}_{_intent_slug(group_slug)}_summary",
                "question": question,
                "patterns": [exact_pattern, loose_pattern],
                "priority": 200 + len(candidates),
                "description": f"Auto-generated dataset-scoped summary for {left} joined to {right}.",
                "left": left,
                "right": right,
                "left_col": left_col,
                "right_col": right_col,
                "group_label": group_label,
                "group_slug": group_slug,
            }
        )

        if len(candidates) >= AUTO_UPLOAD_REGISTRY_MAX_RULES:
            break

    if not candidates:
        ranked_sheets = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, "summary total amount")
        for item in ranked_sheets[:AUTO_UPLOAD_REGISTRY_MAX_RULES]:
            sheet = item["sheet"]
            question = f"Summarize key metrics for {sheet}."
            candidates.append(
                {
                    "intent": f"auto_{_intent_slug(sheet)}_summary",
                    "question": question,
                    "patterns": [_build_question_exact_regex(question)],
                    "priority": 200 + len(candidates),
                    "description": f"Auto-generated dataset-scoped summary for {sheet}.",
                }
            )
    return candidates[:AUTO_UPLOAD_REGISTRY_MAX_RULES]


def _pick_first_matching_column(columns: set[str], candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(col): col for col in (columns or set())}
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


def _build_join_summary_group_expr(
    left_sheet: str,
    right_sheet: str,
    group_slug: str,
    sheet_columns_by_sheet: dict[str, set[str]],
) -> tuple[str, str]:
    group_candidates = {
        "customer": ["customer", "customer_id", "customerid", "cust_id"],
        "service_type": ["service_type", "service", "type"],
        "status": ["status", "state"],
        "join_key": [],
    }
    candidates = group_candidates.get(group_slug, [])
    left_group_col = _pick_first_matching_column(sheet_columns_by_sheet.get(left_sheet) or set(), candidates)
    right_group_col = _pick_first_matching_column(sheet_columns_by_sheet.get(right_sheet) or set(), candidates)
    if left_group_col or right_group_col:
        parts = []
        if left_group_col:
            parts.append(f"NULLIF(TRIM(s.row_json->>'{left_group_col}'), '')")
        if right_group_col:
            parts.append(f"NULLIF(TRIM(j.row_json->>'{right_group_col}'), '')")
        return f"COALESCE({', '.join(parts)}, 'Unknown')", group_slug
    return "COALESCE(NULLIF(TRIM(s.row_json->>'" + "" + "'), ''), 'Unknown')", "group_value"


def _build_auto_summary_sql(candidate: dict[str, Any], sheet_columns_by_sheet: dict[str, set[str]]) -> str | None:
    left_sheet = str(candidate.get("left") or "").strip()
    right_sheet = str(candidate.get("right") or "").strip()
    left_col = str(candidate.get("left_col") or "").strip()
    right_col = str(candidate.get("right_col") or "").strip()
    group_slug = normalize_col_name(str(candidate.get("group_slug") or "")) or "group_value"
    if not left_sheet or not right_sheet or not left_col or not right_col:
        return None

    left_columns = sheet_columns_by_sheet.get(left_sheet) or set()
    right_columns = sheet_columns_by_sheet.get(right_sheet) or set()
    left_id_col = _pick_first_matching_column(left_columns, ["event_id", "invoice_id", "contract_id", normalize_col_name(left_col)])
    if not left_id_col:
        left_id_col = next((col for col in sorted(left_columns) if looks_like_id_col(col)), None)
    right_id_col = _pick_first_matching_column(right_columns, ["invoice_id", "contract_id", "event_id", normalize_col_name(right_col)])
    if not right_id_col:
        right_id_col = next((col for col in sorted(right_columns) if looks_like_id_col(col)), None)

    group_expr, group_alias = _build_join_summary_group_expr(left_sheet, right_sheet, group_slug, sheet_columns_by_sheet)
    if "row_json->>''" in group_expr:
        group_expr = f"COALESCE(NULLIF(TRIM(s.row_json->>'{left_col}'), ''), 'Unknown')"
        group_alias = normalize_col_name(left_col) or "group_value"

    select_items = [f"{group_expr} AS {group_alias}"]
    if left_id_col:
        select_items.append(f"COUNT(DISTINCT s.row_json->>'{left_id_col}') AS left_record_count")
    if right_id_col:
        select_items.append(f"COUNT(DISTINCT j.row_json->>'{right_id_col}') AS right_record_count")

    left_amount_col = _pick_first_matching_column(left_columns, ["expected_amount", "billed_amount", "amount", "total_amount", "total_value"])
    right_amount_col = _pick_first_matching_column(right_columns, ["billed_amount", "expected_amount", "amount", "total_amount", "total_value"])
    left_qty_col = _pick_first_matching_column(left_columns, ["quantity", "qty"])
    right_rate_col = _pick_first_matching_column(right_columns, ["price_per_unit", "unit_price", "unit_rate", "rate", "price"])

    if left_amount_col:
        select_items.append(
            f"COALESCE(SUM(COALESCE(NULLIF(regexp_replace((s.row_json->>'{left_amount_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)), 0) AS total_{normalize_col_name(left_amount_col)}"
        )
    if right_amount_col:
        select_items.append(
            f"COALESCE(SUM(COALESCE(NULLIF(regexp_replace((j.row_json->>'{right_amount_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)), 0) AS total_{normalize_col_name(right_amount_col)}"
        )
    if left_qty_col:
        select_items.append(
            f"COALESCE(SUM(COALESCE(NULLIF(regexp_replace((s.row_json->>'{left_qty_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)), 0) AS total_{normalize_col_name(left_qty_col)}"
        )
    if left_qty_col and right_rate_col:
        select_items.append(
            f"COALESCE(SUM(COALESCE(NULLIF(regexp_replace((s.row_json->>'{left_qty_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0) * COALESCE(NULLIF(regexp_replace((j.row_json->>'{right_rate_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)), 0) AS total_join_value"
        )

    if len(select_items) < 2:
        return None

    left_sheet_sql = left_sheet.replace("'", "''")
    right_sheet_sql = right_sheet.replace("'", "''")
    return (
        f"SELECT {', '.join(select_items)} "
        "FROM dataset_rows s "
        "LEFT JOIN dataset_rows j "
        "ON j.dataset_id = s.dataset_id "
        f"AND j.sheet_name = '{right_sheet_sql}' "
        f"AND NULLIF(LOWER(TRIM(s.row_json->>'{left_col}')), '') = NULLIF(LOWER(TRIM(j.row_json->>'{right_col}')), '') "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{left_sheet_sql}' "
        "GROUP BY 1 "
        "ORDER BY 2 DESC "
        "LIMIT %(limit)s"
    )


def _maybe_execute_generated_sql(sql: str, dataset_id: str, question: str) -> bool:
    if psycopg is None or dict_row is None:
        return True
    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(_prepare_sql_for_psycopg(sql), _build_exec_params(sql, dataset_id, 10, question))
                cur.fetchall()
        return True
    except Exception:
        return False


def _generate_dataset_scoped_registry_rules(
    dataset_id: str,
    cols: list[str],
    col_types: dict[str, str],
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not AUTO_UPLOAD_REGISTRY_ENABLED or AUTO_UPLOAD_REGISTRY_MAX_RULES < 1:
        return []
    if sqlparse is None:
        return []

    generated_rules: list[dict[str, Any]] = []
    candidates = _build_auto_registry_candidate_questions(available_sheets, sheet_columns_by_sheet, relationships)

    for candidate in candidates:
        question = str(candidate.get("question") or "").strip()
        if not question:
            continue
        if _match_rules_registry_sql_rule(question, _load_rules_registry()):
            continue
        try:
            sql = _build_auto_summary_sql(candidate, sheet_columns_by_sheet)
            if sql and _is_safe_select(sql) and "dataset_id" in sql.lower() and _maybe_execute_generated_sql(sql, dataset_id, question):
                generated_rules.append(
                    {
                        "id": str(uuid4()),
                        "intent": candidate["intent"],
                        "enabled": True,
                        "priority": int(candidate.get("priority", 200) or 200),
                        "patterns": list(candidate.get("patterns") or []),
                        "builder": None,
                        "sql_template": sql,
                        "description": candidate.get("description"),
                        "sample_question": question,
                    }
                )
                continue

            system, user, sheet_schema_context, rel_desc = _build_sql_generation_messages(
                question,
                cols,
                col_types,
                available_sheets,
                sheet_columns_by_sheet,
                relationships,
                analyze_all_rows=False,
                requested_top_n=None,
                force_top_1=False,
                explicit_sheet=_extract_explicit_sheet_reference(question, available_sheets),
            )
            sql = _generate_sql_with_llm(system, user)
            if not sql:
                continue
            sql = _sanitize_unbound_year_placeholder(sql, question)
            sql, schema_issues, _schema_changed = _repair_sql_until_schema_valid(
                sql,
                question,
                sheet_columns_by_sheet,
                sheet_schema_context,
                rel_desc,
            )
            if schema_issues:
                continue
            if not re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE):
                sql = f"{sql}\nLIMIT %(limit)s"
            if not _is_safe_select(sql) or "dataset_id" not in sql.lower():
                continue
            if not _maybe_execute_generated_sql(sql, dataset_id, question):
                continue
            generated_rules.append(
                {
                    "id": str(uuid4()),
                    "intent": candidate["intent"],
                    "enabled": True,
                    "priority": int(candidate.get("priority", 200) or 200),
                    "patterns": list(candidate.get("patterns") or []),
                    "builder": None,
                    "sql_template": sql,
                    "description": candidate.get("description"),
                    "sample_question": question,
                }
            )
        except Exception:
            continue

    return generated_rules


def _format_legacy_metadata_context(
    cols: list[str],
    col_types: dict[str, str],
    available_sheets: list[str],
    question: str,
) -> str:
    if len(available_sheets or []) > 1:
        return (
            "Legacy first-sheet metadata is intentionally omitted because this dataset has multiple sheets. "
            "Use only the per-sheet sampled schema above when choosing columns."
        )

    if not cols:
        return "No legacy first-sheet metadata is available."

    q_terms = _question_terms(question)
    scored_cols: list[tuple[float, str]] = []
    for col in cols:
        norm_col = normalize_col_name(col)
        col_terms = _question_terms(col)
        overlap = len(q_terms & col_terms)
        score = float(overlap * 2)
        if norm_col and norm_col in normalize_col_name(question):
            score += 3.0
        scored_cols.append((score, col))

    scored_cols.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    selected = [col for score, col in scored_cols if score > 0][:20]
    if not selected:
        selected = list(cols)[:20]

    return "\n".join([f"{c} -> {col_types.get(c, 'text')}" for c in selected])


def _extract_explicit_sheet_reference(question: str, available_sheets: list[str]) -> str | None:
    q = (question or "").lower()
    for sheet in available_sheets or []:
        variants = {
            str(sheet).lower(),
            str(sheet).lower().replace("_", " "),
            normalize_col_name(str(sheet)).replace("_", " "),
        }
        for variant in variants:
            if variant and (f"from {variant}" in q or f"in {variant}" in q or f"on {variant}" in q):
                return sheet
    return None


def _build_empty_sheet_sql(sheet_name: str) -> str:
    sheet_name_sql = (sheet_name or "").replace("'", "''")
    return (
        "SELECT NULL::text AS event_id, NULL::text AS customer_id, NULL::numeric AS value_gap_amount "
        "FROM dataset_rows s "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{sheet_name_sql}' "
        "AND 1 = 0 "
        "LIMIT %(limit)s"
    )


def _build_explicit_sheet_schema_note(sheet_name: str, sheet_columns_by_sheet: dict[str, set[str]], required_columns: list[str]) -> str:
    available = sorted(list(sheet_columns_by_sheet.get(sheet_name) or set()))
    available_text = ", ".join(available) if available else "no sampled columns"
    missing_text = ", ".join(required_columns)
    return (
        f"The question explicitly requested sheet '{sheet_name}', so the query was restricted to that sheet only. "
        f"But '{sheet_name}' does not contain the columns needed for this request: {missing_text}. "
        f"Available columns in '{sheet_name}' are: {available_text}."
    )


def _pick_billing_lag_source_sheet(available_sheets: list[str], sheet_columns_by_sheet: dict[str, set[str]], question: str) -> str | None:
    lag_candidates = ["billing_lag_days", "lag_days", "invoice_lag_days"]
    gap_candidates = ["value_gap", "value_gap_amount", "variance", "difference"]
    ranked = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)

    for item in ranked:
        sheet = item["sheet"]
        if _sheet_has_any_column(sheet_columns_by_sheet, sheet, lag_candidates) and _sheet_has_any_column(sheet_columns_by_sheet, sheet, gap_candidates):
            return sheet

    return None


def _pick_billing_lag_only_source_sheet(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
) -> str | None:
    lag_candidates = ["billing_lag_days", "lag_days", "invoice_lag_days"]
    event_candidates = ["event_id", "eventid", "ops_event_id", "operation_event_id"]
    ranked = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)

    for item in ranked:
        sheet = item["sheet"]
        if _sheet_has_any_column(sheet_columns_by_sheet, sheet, lag_candidates) and _sheet_has_any_column(sheet_columns_by_sheet, sheet, event_candidates):
            return sheet

    return None


def _semantic_role_column(semantic_roles: dict[str, dict[str, str]], sheet_name: str, role_names: list[str]) -> str | None:
    roles = semantic_roles.get(sheet_name) or {}
    for role_name in role_names:
        value = roles.get(role_name)
        if value:
            return str(value)
    return None


def _relationship_connects_sheets(relationships: list[dict[str, Any]], left_sheet: str, right_sheet: str) -> bool:
    for rel in relationships or []:
        left = str(rel.get("left") or "")
        right = str(rel.get("right") or "")
        if {left, right} == {left_sheet, right_sheet}:
            return True
    return False


def _pick_customer_fallback_sheets(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    source_sheet: str,
    *,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    fallback_sheets: list[str] = []
    event_candidates = ["event_id", "eventid", "ops_event_id", "operation_event_id"]
    customer_candidates = ["customer", "customer_id", "customerid", "cust_id"]
    semantic_roles = semantic_roles or {}
    scored: list[tuple[float, str]] = []

    for sheet in available_sheets:
        if sheet == source_sheet:
            continue

        has_event = _semantic_role_column(semantic_roles, sheet, ["event_id"]) is not None or _sheet_has_any_column(sheet_columns_by_sheet, sheet, event_candidates)
        has_customer = _semantic_role_column(semantic_roles, sheet, ["customer_id"]) is not None or _sheet_has_any_column(sheet_columns_by_sheet, sheet, customer_candidates)
        if not (has_event and has_customer):
            continue

        score = 0.0
        if _relationship_connects_sheets(relationships or [], source_sheet, sheet):
            score += 5.0
        if _semantic_role_column(semantic_roles, sheet, ["event_id"]):
            score += 3.0
        if _semantic_role_column(semantic_roles, sheet, ["customer_id"]):
            score += 3.0
        if _semantic_role_column(semantic_roles, sheet, ["status", "expected_amount", "quantity"]):
            score += 1.0
        if _semantic_role_column(semantic_roles, sheet, ["invoice_id", "invoice_date", "invoice_status"]):
            score -= 0.75
        scored.append((score, sheet))

    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    return [sheet for _, sheet in scored[:2]]


def _build_preferred_customer_expr(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    source_sheet: str,
    source_alias: str,
    source_event_col: str,
    source_customer_col: str,
    *,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    source_customer_expr = f"NULLIF(TRIM({source_alias}.row_json->>'{source_customer_col}'), '')"
    fallback_exprs: list[str] = []
    join_clauses: list[str] = []

    for idx, fallback_sheet in enumerate(
        _pick_customer_fallback_sheets(
            available_sheets,
            sheet_columns_by_sheet,
            source_sheet,
            relationships=relationships,
            semantic_roles=semantic_roles,
        ),
        start=1,
    ):
        alias = f"cust{idx}"
        fallback_sheet_sql = fallback_sheet.replace("'", "''")
        fallback_event_col = _semantic_role_column(semantic_roles or {}, fallback_sheet, ["event_id"]) or _pick_sheet_column(
            sheet_columns_by_sheet,
            fallback_sheet,
            ["event_id", "eventid", "ops_event_id", "operation_event_id"],
        )
        fallback_customer_col = _semantic_role_column(semantic_roles or {}, fallback_sheet, ["customer_id"]) or _pick_sheet_column(
            sheet_columns_by_sheet,
            fallback_sheet,
            ["customer", "customer_id", "customerid", "cust_id"],
        )
        if not fallback_event_col or not fallback_customer_col:
            continue

        join_clauses.append(
            f"LEFT JOIN dataset_rows {alias} "
            f"ON {alias}.dataset_id = {source_alias}.dataset_id "
            f"AND {alias}.sheet_name = '{fallback_sheet_sql}' "
            f"AND NULLIF(LOWER(TRIM({alias}.row_json->>'{fallback_event_col}')), '') = NULLIF(LOWER(TRIM({source_alias}.row_json->>'{source_event_col}')), '') "
        )
        fallback_exprs.append(f"NULLIF(TRIM({alias}.row_json->>'{fallback_customer_col}'), '')")

    customer_expr = f"COALESCE({', '.join(fallback_exprs + [source_customer_expr])})" if fallback_exprs else source_customer_expr
    return "".join(join_clauses), customer_expr


def _question_requests_explicit_aggregation(question: str) -> bool:
    raw_q = (question or "").lower()
    q = normalize_col_name(question or "")
    aggregate_tokens = {
        "sum",
        "totals",
        "total",
        "count",
        "average",
        "avg",
        "minimum",
        "maximum",
        "min",
        "max",
        "aggregate",
        "aggregated",
        "summary",
        "summarize",
        "breakdown",
        "breakdowns",
        "rollup",
    }
    aggregate_phrases = [
        "group by",
        "grouped by",
        "grouping by",
        "broken down by",
        "break down by",
        "split by",
    ]
    return any(token in q.split("_") for token in aggregate_tokens) or any(phrase in raw_q for phrase in aggregate_phrases)


def _is_exception_detail_question(question: str) -> bool:
    q = (question or "").lower()
    mentions_customer = any(token in q for token in ["customer", "customer id", "customer_id"])
    mentions_exception_metric = any(
        token in q
        for token in [
            "value gap",
            "value_gap",
            "recommended_action",
            "gap amount",
            "expected amount",
            "billed amount",
            "billing lag",
            "billing_lag_days",
            "lag days",
            "recommended action",
            "invoice id",
            "invoice_id",
            "event id",
            "event_id",
        ]
    )
    if not mentions_exception_metric and _extract_recommended_action_filter(question):
        mentions_exception_metric = True
    return mentions_customer and mentions_exception_metric and not _question_requests_explicit_aggregation(question)


def _is_customer_value_gap_summary_question(question: str) -> bool:
    q = (question or "").lower()
    mentions_customer = any(token in q for token in ["customer", "customer id", "customer_id"])
    mentions_gap = any(token in q for token in ["value gap", "value_gap", "gap amount", "variance", "difference"])
    return mentions_customer and mentions_gap and _question_requests_explicit_aggregation(question)


def _is_semantic_role_sheet_summary_question(question: str) -> bool:
    q = normalize_col_name(question or "")
    return (
        "semantic_roles" in q
        or "semantic_role" in q
        or ("sheet_wise_summary" in q and "detected" in q and "roles" in q)
        or ("sheetwise_summary" in q and "semantic" in q)
    )


def _build_semantic_role_sheet_summary_sql(
    semantic_roles: dict[str, dict[str, str]] | None,
) -> str | None:
    semantic_roles = _sanitize_semantic_roles(semantic_roles)
    if not semantic_roles:
        return None

    value_rows: list[str] = []
    for sheet_name in sorted(semantic_roles):
        role_map = semantic_roles.get(sheet_name) or {}
        role_names = sorted(str(role_name) for role_name in role_map.keys() if str(role_name).strip())
        detected_roles = ", ".join(role_names)
        role_count = len(role_names)
        sheet_sql = str(sheet_name).replace("'", "''")
        roles_sql = detected_roles.replace("'", "''")
        value_rows.append(f"('{sheet_sql}', {role_count}, '{roles_sql}')")

    if not value_rows:
        return None

    return (
        "WITH semantic_summary(sheet_name, role_count, detected_roles) AS ("
        f"VALUES {', '.join(value_rows)}"
        ") "
        "SELECT s.sheet_name, "
        "COALESCE(r.row_count, 0) AS row_count, "
        "s.role_count, "
        "s.detected_roles "
        "FROM semantic_summary s "
        "LEFT JOIN ("
        "  SELECT sheet_name, COUNT(*) AS row_count "
        "  FROM dataset_rows "
        "  WHERE dataset_id = %(dataset_id)s "
        "  GROUP BY sheet_name"
        ") r ON r.sheet_name = s.sheet_name "
        "ORDER BY s.sheet_name "
        "LIMIT %(limit)s"
    )


def _is_exception_customer_list_question(question: str) -> bool:
    q = (question or "").lower()
    exception_type_filters = _extract_exception_type_filters(question)
    recommended_action_filter = _extract_recommended_action_filter(question)
    mentions_customer = any(token in q for token in ["customer", "customers", "customer id", "customer_id"])
    list_intent = any(token in q for token in ["list", "show", "give", "which", "who"])
    detail_tokens = [
        "event id",
        "event_id",
        "invoice id",
        "invoice_id",
        "expected amount",
        "billed amount",
        "value gap",
        "value_gap",
        "gap amount",
        "billing lag",
        "lag days",
        "service type",
        "service date",
        "invoice date",
    ]
    asks_detail_columns = any(token in q for token in detail_tokens)
    return bool(exception_type_filters or recommended_action_filter) and mentions_customer and list_intent and not asks_detail_columns and not _question_requests_explicit_aggregation(question)


def _extract_exception_type_filters(question: str) -> list[str]:
    q = (question or "").lower()
    filters: list[str] = []

    if any(token in q for token in ["unbilled", "not billed", "not invoiced", "without invoice", "invoice not sent"]):
        filters.append("UNBILLED")
    if any(token in q for token in ["under billed", "underbilled", "short billed", "shortbilled"]):
        filters.append("UNDER_BILLED")
    if any(token in q for token in ["over billed", "overbilled"]):
        filters.append("OVER_BILLED")
    if any(token in q for token in ["billing delay", "billing lag", "lag days", "delayed invoice"]):
        filters.append("BILLING_DELAY")

    seen: set[str] = set()
    ordered: list[str] = []
    for value in filters:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _normalize_text_filter_phrase(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_known_recommended_action_phrase(question: str) -> str | None:
    normalized_question = _normalize_text_filter_phrase(question)
    known_action_phrases = [
        "invoice now",
        "create invoice",
        "fix rate qty",
        "debit note",
        "correction",
        "reduce billing lag",
        "process automation",
    ]
    for phrase in known_action_phrases:
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized_question):
            return phrase
    return None


def _extract_recommended_action_filter(question: str) -> str | None:
    q = str(question or "")
    explicit_patterns = [
        r"(?:recommended[_\s-]*action|action)\s*(?:=|is|equals?|like)\s*([^,;]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.split(
            r"\b(?:and|or|with|where|when|for|by|having|limit|order)\b",
            match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        normalized = _normalize_text_filter_phrase(candidate)
        if normalized:
            return normalized
    return _extract_known_recommended_action_phrase(q)


def _pick_exception_detail_source_sheet(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    *,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
) -> str | None:
    ranked = [item["sheet"] for item in _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)]
    required_event = ["event_id", "eventid", "ops_event_id", "operation_event_id"]
    required_customer = ["customer", "customer_id", "customerid", "cust_id"]
    detail_markers = [
        "value_gap",
        "value_gap_amount",
        "variance",
        "difference",
        "billing_lag_days",
        "lag_days",
        "invoice_lag_days",
        "expected_amount",
        "billed_amount",
        "exception_type",
        "recommended_action",
    ]
    semantic_roles = semantic_roles or {}
    scored: list[tuple[float, str]] = []

    for idx, sheet in enumerate(ranked):
        has_event = _semantic_role_column(semantic_roles, sheet, ["event_id"]) is not None or _sheet_has_any_column(sheet_columns_by_sheet, sheet, required_event)
        has_customer = _semantic_role_column(semantic_roles, sheet, ["customer_id"]) is not None or _sheet_has_any_column(sheet_columns_by_sheet, sheet, required_customer)
        has_detail = any(
            _semantic_role_column(semantic_roles, sheet, [role]) is not None
            for role in ["value_gap", "billing_lag_days", "expected_amount", "billed_amount", "exception_type", "recommended_action"]
        ) or _sheet_has_any_column(sheet_columns_by_sheet, sheet, detail_markers)

        if not (has_event and has_customer):
            continue
        if not has_detail:
            continue

        score = float(max(0, len(ranked) - idx))
        if _semantic_role_column(semantic_roles, sheet, ["exception_type"]):
            score += 6.0
        if _semantic_role_column(semantic_roles, sheet, ["event_id"]):
            score += 3.0
        if _semantic_role_column(semantic_roles, sheet, ["customer_id"]):
            score += 3.0
        if _semantic_role_column(semantic_roles, sheet, ["value_gap", "billing_lag_days", "expected_amount", "billed_amount"]):
            score += 2.0
        if _semantic_role_column(semantic_roles, sheet, ["recommended_action"]):
            score += 1.0
        if any(
            _relationship_connects_sheets(relationships or [], sheet, other)
            and _semantic_role_column(semantic_roles, other, ["customer_id", "event_id"])
            for other in available_sheets
            if other != sheet
        ):
            score += 1.5
        scored.append((score, sheet))

    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    return scored[0][1] if scored else None


def _build_exception_customer_list_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
    forced_sheet: str | None = None,
) -> str | None:
    source_sheet = forced_sheet or _pick_exception_detail_source_sheet(
        available_sheets,
        sheet_columns_by_sheet,
        question,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )
    if not source_sheet:
        return None

    customer_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["customer_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["customer", "customer_id", "customerid", "cust_id"],
    )
    event_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["event_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["event_id", "eventid", "ops_event_id", "operation_event_id"],
    )
    exception_type_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["exception_type"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["exception_type", "exception", "status"])
    expected_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["expected_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["expected_amount", "expected_amt", "amount", "total_amount"])
    billed_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["billed_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["billed_amount", "invoice_amount", "amount_billed"])
    gap_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["value_gap"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["value_gap", "value_gap_amount", "variance", "difference"])
    lag_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["billing_lag_days"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["billing_lag_days", "lag_days", "invoice_lag_days"])
    invoice_id_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["invoice_id"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["invoice_id", "invoiceid"])
    invoice_date_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["invoice_date"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["invoice_date", "invoiced_date"])
    service_type_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["service_type"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["service_type", "service", "service_category"])
    service_date_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["service_date"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["service_date", "date"])
    recommended_action_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["recommended_action"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["recommended_action", "action", "next_action"])
    exception_type_filters = _extract_exception_type_filters(question)
    recommended_action_filter = _extract_recommended_action_filter(question)
    if not customer_col or not event_col or not (exception_type_filters or recommended_action_filter):
        return None

    customer_joins_sql, customer_expr = _build_preferred_customer_expr(
        available_sheets,
        sheet_columns_by_sheet,
        source_sheet,
        "s",
        event_col,
        customer_col,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )

    source_sheet_sql = source_sheet.replace("'", "''")
    filters = [
        "s.dataset_id = %(dataset_id)s",
        f"s.sheet_name = '{source_sheet_sql}'",
        f"NULLIF(TRIM(s.row_json->>'{customer_col}'), '') IS NOT NULL",
    ]
    if exception_type_col and exception_type_filters:
        allowed = ", ".join([f"'{value}'" for value in exception_type_filters])
        filters.append(f"UPPER(TRIM(COALESCE(s.row_json->>'{exception_type_col}', ''))) IN ({allowed})")
    elif "UNBILLED" in exception_type_filters and invoice_id_col:
        filters.append(f"NULLIF(TRIM(COALESCE(s.row_json->>'{invoice_id_col}', '')), '') IS NULL")
    if recommended_action_col and recommended_action_filter:
        action_expr = f"regexp_replace(lower(COALESCE(s.row_json->>'{recommended_action_col}', '')), '[^a-z0-9]+', ' ', 'g')"
        action_value_sql = recommended_action_filter.replace("'", "''")
        filters.append(f"{action_expr} LIKE '%{action_value_sql}%'")

    selected_cols = [
        f"{customer_expr} AS customer",
        f"s.row_json->>'{event_col}' AS event_id",
    ]
    if exception_type_col:
        selected_cols.append(f"s.row_json->>'{exception_type_col}' AS exception_type")
    if service_type_col:
        selected_cols.append(f"s.row_json->>'{service_type_col}' AS service_type")
    if service_date_col:
        selected_cols.append(f"s.row_json->>'{service_date_col}' AS service_date")
    if invoice_id_col:
        selected_cols.append(f"s.row_json->>'{invoice_id_col}' AS invoice_id")
    if invoice_date_col:
        selected_cols.append(f"s.row_json->>'{invoice_date_col}' AS invoice_date")
    if expected_amount_col:
        selected_cols.append(f"s.row_json->>'{expected_amount_col}' AS expected_amount")
    if billed_amount_col:
        selected_cols.append(f"s.row_json->>'{billed_amount_col}' AS billed_amount")
    if gap_col:
        selected_cols.append(f"s.row_json->>'{gap_col}' AS value_gap")
    if lag_col:
        selected_cols.append(f"s.row_json->>'{lag_col}' AS billing_lag_days")
    if recommended_action_col:
        selected_cols.append(f"s.row_json->>'{recommended_action_col}' AS recommended_action")

    gap_expr = _numeric_json_expr("s", gap_col, zero_default=False)
    lag_expr = _numeric_json_expr("s", lag_col, zero_default=False)
    order_by_parts = [f"{customer_expr}"]
    if gap_expr:
        order_by_parts.append(f"ABS(COALESCE({gap_expr}, 0)) DESC")
    if lag_expr:
        order_by_parts.append(f"COALESCE({lag_expr}, 0) DESC")
    order_by_parts.append(f"s.row_json->>'{event_col}'")

    return (
        f"SELECT DISTINCT ON ({customer_expr}) {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        f"{customer_joins_sql}"
        f"WHERE {' AND '.join(filters)} "
        f"ORDER BY {', '.join(order_by_parts)} "
        "LIMIT %(limit)s"
    )


def _build_exception_detail_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
    forced_sheet: str | None = None,
) -> str | None:
    source_sheet = forced_sheet or _pick_exception_detail_source_sheet(
        available_sheets,
        sheet_columns_by_sheet,
        question,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )
    if not source_sheet:
        return None
    source_sheet_sql = source_sheet.replace("'", "''")

    event_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["event_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["event_id", "eventid", "ops_event_id", "operation_event_id"],
    )
    customer_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["customer_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["customer", "customer_id", "customerid", "cust_id"],
    )
    if not event_col or not customer_col:
        return None

    customer_joins_sql, customer_expr = _build_preferred_customer_expr(
        available_sheets,
        sheet_columns_by_sheet,
        source_sheet,
        "s",
        event_col,
        customer_col,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )

    exception_type_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["exception_type"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["exception_type", "exception", "status"])
    expected_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["expected_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["expected_amount", "expected_amt", "amount", "total_amount"])
    billed_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["billed_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["billed_amount", "invoice_amount", "amount_billed"])
    gap_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["value_gap"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["value_gap", "value_gap_amount", "variance", "difference"])
    lag_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["billing_lag_days"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["billing_lag_days", "lag_days", "invoice_lag_days"])
    invoice_id_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["invoice_id"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["invoice_id", "invoiceid"])
    invoice_date_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["invoice_date"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["invoice_date", "invoiced_date"])
    service_type_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["service_type"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["service_type", "service", "service_category"])
    service_date_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["service_date"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["service_date", "date"])
    recommended_action_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["recommended_action"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["recommended_action", "action", "next_action"])
    exception_type_filters = _extract_exception_type_filters(question)
    recommended_action_filter = _extract_recommended_action_filter(question)

    selected_cols = [
        f"s.row_json->>'{event_col}' AS event_id",
        f"{customer_expr} AS customer",
    ]
    if exception_type_col:
        selected_cols.append(f"s.row_json->>'{exception_type_col}' AS exception_type")
    if expected_amount_col:
        selected_cols.append(f"s.row_json->>'{expected_amount_col}' AS expected_amount")
    if billed_amount_col:
        selected_cols.append(f"s.row_json->>'{billed_amount_col}' AS billed_amount")
    if gap_col:
        selected_cols.append(f"s.row_json->>'{gap_col}' AS value_gap")
    if lag_col:
        selected_cols.append(f"s.row_json->>'{lag_col}' AS billing_lag_days")
    if invoice_id_col:
        selected_cols.append(f"s.row_json->>'{invoice_id_col}' AS invoice_id")
    if invoice_date_col:
        selected_cols.append(f"s.row_json->>'{invoice_date_col}' AS invoice_date")
    if service_type_col:
        selected_cols.append(f"s.row_json->>'{service_type_col}' AS service_type")
    if service_date_col:
        selected_cols.append(f"s.row_json->>'{service_date_col}' AS service_date")
    if recommended_action_col:
        selected_cols.append(f"s.row_json->>'{recommended_action_col}' AS recommended_action")

    filters = [
        "s.dataset_id = %(dataset_id)s",
        f"s.sheet_name = '{source_sheet_sql}'",
        f"NULLIF(TRIM(s.row_json->>'{event_col}'), '') IS NOT NULL",
        f"NULLIF(TRIM(s.row_json->>'{customer_col}'), '') IS NOT NULL",
    ]

    if exception_type_filters and exception_type_col:
        allowed = ", ".join([f"'{value}'" for value in exception_type_filters])
        filters.append(f"UPPER(TRIM(COALESCE(s.row_json->>'{exception_type_col}', ''))) IN ({allowed})")
    elif "UNBILLED" in exception_type_filters and invoice_id_col:
        filters.append(f"NULLIF(TRIM(COALESCE(s.row_json->>'{invoice_id_col}', '')), '') IS NULL")

    gap_expr = _numeric_json_expr("s", gap_col, zero_default=False)
    lag_expr = _numeric_json_expr("s", lag_col, zero_default=False)
    gap_condition = _extract_value_gap_condition(question)
    if gap_expr:
        if gap_condition is not None:
            filters.append(f"COALESCE({gap_expr}, 0) {gap_condition[0]} {gap_condition[1]}")
        elif any(token in (question or "").lower() for token in ["value gap", "value_gap", "gap amount"]):
            filters.append(f"COALESCE({gap_expr}, 0) <> 0")

    lag_threshold = _extract_lag_days_threshold(question)
    if lag_expr and lag_threshold is not None:
        filters.append(f"COALESCE({lag_expr}, 0) > {lag_threshold}")

    if recommended_action_col and recommended_action_filter:
        action_expr = f"regexp_replace(lower(COALESCE(s.row_json->>'{recommended_action_col}', '')), '[^a-z0-9]+', ' ', 'g')"
        action_value_sql = recommended_action_filter.replace("'", "''")
        filters.append(f"{action_expr} LIKE '%{action_value_sql}%'")

    order_by_parts: list[str] = []
    if gap_expr:
        order_by_parts.append(f"ABS(COALESCE({gap_expr}, 0)) DESC")
    if lag_expr:
        order_by_parts.append(f"COALESCE({lag_expr}, 0) DESC")
    order_by_parts.append(f"s.row_json->>'{event_col}'")

    return (
        f"SELECT {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        f"{customer_joins_sql}"
        f"WHERE {' AND '.join(filters)} "
        f"ORDER BY {', '.join(order_by_parts)} "
        "LIMIT %(limit)s"
    )


def _build_customer_value_gap_summary_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
    forced_sheet: str | None = None,
) -> str | None:
    source_sheet = forced_sheet or _pick_exception_detail_source_sheet(
        available_sheets,
        sheet_columns_by_sheet,
        question,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )
    if not source_sheet:
        return None

    customer_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["customer_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["customer", "customer_id", "customerid", "cust_id"],
    )
    event_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["event_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["event_id", "eventid", "ops_event_id", "operation_event_id"],
    )
    expected_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["expected_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["expected_amount", "expected_amt", "amount", "total_amount"])
    billed_amount_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["billed_amount"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["billed_amount", "invoice_amount", "amount_billed"])
    gap_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["value_gap"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["value_gap", "value_gap_amount", "variance", "difference"])
    exception_type_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["exception_type"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["exception_type", "exception", "status"])
    invoice_id_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["invoice_id"]) or _pick_sheet_column(sheet_columns_by_sheet, source_sheet, ["invoice_id", "invoiceid"])
    if not customer_col or not event_col or not gap_col:
        return None

    customer_joins_sql, customer_expr = _build_preferred_customer_expr(
        available_sheets,
        sheet_columns_by_sheet,
        source_sheet,
        "s",
        event_col,
        customer_col,
        relationships=relationships,
        semantic_roles=semantic_roles,
    )

    source_sheet_sql = source_sheet.replace("'", "''")
    expected_expr = _numeric_json_expr("s", expected_amount_col)
    billed_expr = _numeric_json_expr("s", billed_amount_col)
    gap_expr = _numeric_json_expr("s", gap_col)
    exception_type_filters = _extract_exception_type_filters(question)
    if not gap_expr:
        return None

    selected_cols = [
        f"{customer_expr} AS customer_id",
        f"COUNT(DISTINCT s.row_json->>'{event_col}') AS event_count",
    ]
    if expected_expr:
        selected_cols.append(f"COALESCE(SUM({expected_expr}), 0) AS total_expected_amount")
    if billed_expr:
        selected_cols.append(f"COALESCE(SUM({billed_expr}), 0) AS total_billed_amount")
    selected_cols.append(f"COALESCE(SUM({gap_expr}), 0) AS total_value_gap")

    exception_filter_sql = ""
    if exception_type_filters and exception_type_col:
        allowed = ", ".join([f"'{value}'" for value in exception_type_filters])
        exception_filter_sql = f"AND UPPER(TRIM(COALESCE(s.row_json->>'{exception_type_col}', ''))) IN ({allowed}) "
    elif "UNBILLED" in exception_type_filters and invoice_id_col:
        exception_filter_sql = f"AND NULLIF(TRIM(COALESCE(s.row_json->>'{invoice_id_col}', '')), '') IS NULL "

    return (
        f"SELECT {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        f"{customer_joins_sql}"
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{source_sheet_sql}' "
        f"AND NULLIF(TRIM(s.row_json->>'{customer_col}'), '') IS NOT NULL "
        f"AND NULLIF(TRIM(s.row_json->>'{event_col}'), '') IS NOT NULL "
        f"{exception_filter_sql}"
        f"GROUP BY {customer_expr} "
        "ORDER BY total_value_gap DESC, customer_id "
        "LIMIT %(limit)s"
    )


def _build_billing_lag_value_gap_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
    forced_sheet: str | None = None,
) -> str | None:
    source_sheet = forced_sheet or _pick_billing_lag_source_sheet(available_sheets, sheet_columns_by_sheet, question)

    if not source_sheet:
        return None

    lag_threshold = _extract_lag_days_threshold(question) or 7
    source_sheet_sql = source_sheet.replace("'", "''")

    event_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["event_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["event_id", "eventid", "ops_event_id", "operation_event_id"],
    )
    customer_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["customer_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["customer", "customer_id", "customerid", "cust_id"],
    )
    lag_col = _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["billing_lag_days", "lag_days", "invoice_lag_days"],
    )
    gap_col = _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["value_gap", "value_gap_amount", "variance", "difference"],
    )

    if not event_col or not lag_col or not gap_col:
        return None

    event_expr = f"s.row_json->>'{event_col}'"
    if customer_col:
        joins_sql, customer_expr = _build_preferred_customer_expr(
            available_sheets,
            sheet_columns_by_sheet,
            source_sheet,
            "s",
            event_col,
            customer_col,
            relationships=relationships,
            semantic_roles=semantic_roles,
        )
    else:
        joins_sql, customer_expr = "", "NULL"
    lag_expr = f"COALESCE(NULLIF(regexp_replace((s.row_json->>'{lag_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)"
    gap_expr = f"COALESCE(NULLIF(regexp_replace((s.row_json->>'{gap_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)"
    gap_condition = _extract_value_gap_condition(question)
    if gap_condition is not None:
        gap_filter = f" AND {gap_expr} {gap_condition[0]} {gap_condition[1]} "
    else:
        require_positive_gap = _question_requires_positive_gap(question)
        gap_filter = f" AND {gap_expr} > 0 " if require_positive_gap else ""

    return (
        f"SELECT {event_expr} AS event_id, {customer_expr} AS customer_id, {gap_expr} AS value_gap_amount "
        "FROM dataset_rows s "
        f"{joins_sql}"
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{source_sheet_sql}' "
        f"AND {lag_expr} > {lag_threshold} "
        f"{gap_filter}"
        f"ORDER BY {gap_expr} DESC, {lag_expr} DESC "
        "LIMIT %(limit)s"
    )


def _build_billing_lag_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    relationships: list[dict[str, Any]] | None = None,
    semantic_roles: dict[str, dict[str, str]] | None = None,
    forced_sheet: str | None = None,
) -> str | None:
    source_sheet = forced_sheet or _pick_billing_lag_only_source_sheet(available_sheets, sheet_columns_by_sheet, question)

    if not source_sheet:
        return None

    lag_threshold = _extract_lag_days_threshold(question) or 7
    source_sheet_sql = source_sheet.replace("'", "''")

    event_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["event_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["event_id", "eventid", "ops_event_id", "operation_event_id"],
    )
    customer_col = _semantic_role_column(semantic_roles or {}, source_sheet, ["customer_id"]) or _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["customer", "customer_id", "customerid", "cust_id"],
    )
    lag_col = _pick_sheet_column(
        sheet_columns_by_sheet,
        source_sheet,
        ["billing_lag_days", "lag_days", "invoice_lag_days"],
    )

    if not event_col or not lag_col:
        return None

    event_expr = f"s.row_json->>'{event_col}'"
    if customer_col:
        joins_sql, customer_expr = _build_preferred_customer_expr(
            available_sheets,
            sheet_columns_by_sheet,
            source_sheet,
            "s",
            event_col,
            customer_col,
            relationships=relationships,
            semantic_roles=semantic_roles,
        )
    else:
        joins_sql, customer_expr = "", "NULL"
    lag_expr = f"COALESCE(NULLIF(regexp_replace((s.row_json->>'{lag_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)"

    return (
        f"SELECT {event_expr} AS event_id, {customer_expr} AS customer_id "
        "FROM dataset_rows s "
        f"{joins_sql}"
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{source_sheet_sql}' "
        f"AND {lag_expr} > {lag_threshold} "
        f"ORDER BY {lag_expr} DESC "
        "LIMIT %(limit)s"
    )


def _build_billing_lag_value_gap_note(
    cur,
    dataset_id: str,
    source_sheet: str,
    lag_threshold: int,
    gap_condition_text: str | None,
) -> str | None:
    lag_sql = "COALESCE(NULLIF(regexp_replace((row_json->>'billing_lag_days'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)"
    gap_sql = "COALESCE(NULLIF(regexp_replace((row_json->>'value_gap'), '[^0-9.\\-]', '', 'g'), '')::numeric, 0)"

    cur.execute(
        f"""
        select
            count(*) as matched_rows,
            count(*) filter (where {gap_sql} > 0) as matched_nonzero_gap_rows
        from dataset_rows
        where dataset_id = %s
          and sheet_name = %s
          and {lag_sql} > %s
        """,
        (dataset_id, source_sheet, lag_threshold),
    )
    counts = cur.fetchone() or {}
    matched_rows = int(counts.get("matched_rows") or 0)
    matched_nonzero_gap_rows = int(counts.get("matched_nonzero_gap_rows") or 0)
    if matched_rows < 1:
        return None

    if gap_condition_text == "value_gap > 0" and matched_nonzero_gap_rows < 1:
        cur.execute(
            f"""
            select
                row_json->>'event_id' as event_id,
                row_json->>'exception_type' as exception_type,
                row_json->>'value_gap' as value_gap,
                row_json->>'billing_lag_days' as billing_lag_days
            from dataset_rows
            where dataset_id = %s
              and sheet_name = %s
              and {gap_sql} > 0
            order by {gap_sql} desc, row_json->>'event_id'
            limit 3
            """,
            (dataset_id, source_sheet),
        )
        examples = cur.fetchall() or []
        if examples:
            example_text = "; ".join(
                [
                    f"event_id {r.get('event_id')} ({r.get('exception_type')}, gap={r.get('value_gap')}, lag={r.get('billing_lag_days') or 'blank'})"
                    for r in examples
                ]
            )
            return (
                f"No rows in {source_sheet} satisfy both billing_lag_days > {lag_threshold} and value_gap > 0. "
                f"The {matched_rows} billing-delay matches all have gap 0. Non-zero gap rows in the same sheet are: {example_text}."
            )
        return (
            f"No rows in {source_sheet} satisfy both billing_lag_days > {lag_threshold} and value_gap > 0. "
            f"The {matched_rows} billing-delay matches all have gap 0."
        )

    if gap_condition_text == "value_gap = 0":
        return None

    if matched_nonzero_gap_rows > 0:
        return None

    cur.execute(
        f"""
        select
            row_json->>'event_id' as event_id,
            row_json->>'exception_type' as exception_type,
            row_json->>'value_gap' as value_gap,
            row_json->>'billing_lag_days' as billing_lag_days
        from dataset_rows
        where dataset_id = %s
          and sheet_name = %s
          and {gap_sql} > 0
        order by {gap_sql} desc, row_json->>'event_id'
        limit 3
        """,
        (dataset_id, source_sheet),
    )
    examples = cur.fetchall() or []
    if examples:
        example_text = "; ".join(
            [
                f"event_id {r.get('event_id')} ({r.get('exception_type')}, gap={r.get('value_gap')}, lag={r.get('billing_lag_days') or 'blank'})"
                for r in examples
            ]
        )
        return (
            f"Matched {matched_rows} rows in {source_sheet} with billing_lag_days > {lag_threshold}, "
            "but all of them have stored value_gap = 0. "
            f"Non-zero value_gap exists only on other exception rows in {source_sheet}: {example_text}."
        )

    return (
        f"Matched {matched_rows} rows in {source_sheet} with billing_lag_days > {lag_threshold}, "
        "but all of them have stored value_gap = 0. No non-zero value_gap rows were found in that sheet."
    )


def _is_top_seller_year_question(question: str) -> bool:
    q = (question or "").lower()
    return ("top seller" in q or "best seller" in q or "top sales" in q) and "year" in q


def _question_mentions_revenue_metric(question: str) -> bool:
    q = (question or "").lower()
    if any(token in q for token in ["revenue", "sales", "sale", "totalprice", "total price", "net sales"]):
        return True
    return re.search(r"\bre+v+e+n+u+e+\b", q) is not None


def _is_top_revenue_by_year_question(question: str) -> bool:
    q = (question or "").lower()
    has_year = _extract_year_from_question(question) is not None
    asks_top = any(token in q for token in ["highest", "top", "maximum", "max", "most"])
    return has_year and asks_top and _question_mentions_revenue_metric(question)


def _pick_col_by_candidates(cols: list[str], candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(c): c for c in (cols or [])}
    for candidate in candidates:
        normalized_candidate = normalize_col_name(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    for c in cols or []:
        normalized_col = normalize_col_name(c)
        compact_col = normalized_col.replace("_", "")
        if any(
            normalize_col_name(candidate) in normalized_col
            or normalize_col_name(candidate).replace("_", "") == compact_col
            for candidate in candidates
        ):
            return c
    return None


def _pick_revenue_dimension_columns(question: str, cols: list[str]) -> list[str]:
    q = normalize_col_name(question)
    selected: list[str] = []

    dimension_groups = [
        (["product", "products", "service", "services", "sku", "item"], ["product", "service_type", "service", "sku", "item"]),
        (["region", "regions", "location", "locations", "state", "country", "city"], ["region", "store_location", "location", "state", "country", "city"]),
        (["salesperson", "sales person", "seller", "rep"], ["salesperson", "sales_person", "seller", "sales_rep", "rep"]),
        (["customer", "customers"], ["customer_name", "customer", "customer_type"]),
    ]

    for mention_terms, candidate_cols in dimension_groups:
        if any(normalize_col_name(term) in q for term in mention_terms):
            picked = _pick_col_by_candidates(cols, candidate_cols)
            if picked and picked not in selected:
                selected.append(picked)

    if selected:
        return selected[:3]

    for fallback_candidates in [
        ["product", "service_type", "service"],
        ["region", "store_location", "location"],
        ["customer_name", "customer", "customer_type"],
    ]:
        picked = _pick_col_by_candidates(cols, fallback_candidates)
        if picked and picked not in selected:
            selected.append(picked)
        if selected:
            break

    return selected[:3]


def _build_top_revenue_by_year_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    question: str,
    forced_sheet: str | None = None,
) -> str | None:
    year = _extract_year_from_question(question)
    if year is None:
        return None

    source_sheet = forced_sheet
    if not source_sheet:
        ranked = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)
        for item in ranked:
            candidate_sheet = str(item.get("sheet") or "").strip()
            if not candidate_sheet:
                continue
            candidate_cols = sorted(list(sheet_columns_by_sheet.get(candidate_sheet) or set()))
            amount_col = _pick_col_by_candidates(candidate_cols, [
                "total_price", "totalprice", "sales_amount", "salesamount", "net_sales", "netsales", "total_amount", "totalamount", "total_value", "totalvalue", "amount", "unit_price", "unitprice"
            ])
            date_col = _pick_col_by_candidates(candidate_cols, [
                "order_date", "orderdate", "date", "invoice_date", "invoicedate", "delivery_date", "deliverydate", "service_date", "servicedate"
            ])
            if amount_col and date_col:
                source_sheet = str(candidate_sheet)
                break

    if not source_sheet and len(available_sheets) == 1:
        source_sheet = available_sheets[0]
    if not source_sheet:
        return None

    cols = sorted(list(sheet_columns_by_sheet.get(source_sheet) or set()))
    amount_col = _pick_col_by_candidates(cols, [
        "total_price", "totalprice", "sales_amount", "salesamount", "net_sales", "netsales", "total_amount", "totalamount", "total_value", "totalvalue", "amount", "unit_price", "unitprice"
    ])
    date_col = _pick_col_by_candidates(cols, [
        "order_date", "orderdate", "date", "invoice_date", "invoicedate", "delivery_date", "deliverydate", "service_date", "servicedate"
    ])
    dimension_cols = _pick_revenue_dimension_columns(question, cols)
    if not amount_col or not date_col or not dimension_cols:
        return None

    source_sheet_sql = source_sheet.replace("'", "''")
    dimension_select = [f"s.row_json->>'{col}' AS {normalize_col_name(col)}" for col in dimension_cols]
    dimension_group = ", ".join([f"s.row_json->>'{col}'" for col in dimension_cols])
    parsed_date = _date_parse_expr(f"s.row_json->>'{date_col}'")
    amount_expr = f"NULLIF(regexp_replace((s.row_json->>'{amount_col}'), '[^0-9.\\-]', '', 'g'), '')::numeric"

    return (
        f"SELECT {', '.join(dimension_select)}, "
        f"SUM({amount_expr}) AS total_revenue "
        "FROM dataset_rows s "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{source_sheet_sql}' "
        f"AND EXTRACT(YEAR FROM {parsed_date})::int = {year} "
        f"GROUP BY {dimension_group} "
        "ORDER BY total_revenue DESC "
        "LIMIT %(limit)s"
    )


def _is_uninvoiced_ops_question(question: str) -> bool:
    q = (question or "").lower()
    invoicing_intent = any(
        token in q
        for token in [
            "not yet been invoiced",
            "not invoiced",
            "uninvoiced",
            "without invoice",
            "missing invoice",
        ]
    )
    ops_intent = any(token in q for token in ["ops", "operational", "event", "events"])
    return invoicing_intent and ops_intent


def _is_invoiced_sent_ops_question(question: str) -> bool:
    q = (question or "").lower()
    sent_intent = any(token in q for token in ["invoice status sent", "invoiced sent", "invoice sent", "status sent"])
    ops_intent = any(token in q for token in ["ops", "operational", "event", "events"])
    return sent_intent and ops_intent


def _is_invoiced_ops_with_amount_question(question: str) -> bool:
    q = (question or "").lower()
    invoiced_intent = any(token in q for token in ["with invoice amount", "invoice amount", "billed amount"])
    completed_intent = "completed" in q
    ops_intent = any(token in q for token in ["ops", "operational", "event", "events"])
    return invoiced_intent and completed_intent and ops_intent


def _is_unbilled_total_value_question(question: str) -> bool:
    q = (question or "").lower()
    total_value_intent = (
        any(token in q for token in ["total value", "sum", "total amount", "total revenue", "total sales"])
        or ("total" in q and any(token in q for token in ["value", "amount", "revenue", "sales"]))
    )
    unbilled_intent = any(token in q for token in ["unbilled", "not billed", "not invoiced", "without invoice", "invoice not sent"])
    completed_intent = "completed" in q
    ops_intent = any(token in q for token in ["ops", "operational", "event", "events"])
    return total_value_intent and unbilled_intent and completed_intent and ops_intent


def _pick_sheet_by_keywords(available_sheets: list[str], keywords: list[str]) -> str | None:
    for sheet in available_sheets:
        norm = normalize_col_name(sheet)
        if all(k in norm for k in keywords):
            return sheet
    return None


def _numeric_json_expr(alias: str, column_name: str | None, *, zero_default: bool = True) -> str | None:
    if not column_name:
        return None
    base_expr = f"NULLIF(regexp_replace(({alias}.row_json->>'{column_name}'), '[^0-9.\\-]', '', 'g'), '')::numeric"
    if zero_default:
        return f"COALESCE({base_expr}, 0)"
    return base_expr


def _build_uninvoiced_ops_fallback_sql(available_sheets: list[str], cols: list[str]) -> str | None:
    ops_sheet = _pick_sheet_by_keywords(available_sheets, ["ops", "event"]) or _pick_sheet_by_keywords(available_sheets, ["operation", "event"])
    invoice_sheet = _pick_sheet_by_keywords(available_sheets, ["invoice"])
    if not ops_sheet or not invoice_sheet:
        return None

    event_col = _pick_col_by_keywords(cols, ["event_id", "eventid", "event"])
    status_col = _pick_col_by_keywords(cols, ["status", "event_status", "state"])
    customer_col = _pick_col_by_keywords(cols, ["customer_id", "customer"])
    contract_col = _pick_col_by_keywords(cols, ["contract_id", "contract"])
    service_type_col = _pick_col_by_keywords(cols, ["service_type", "service", "type"])
    service_date_col = _pick_col_by_keywords(cols, ["service_date", "date"])

    if not event_col:
        return None

    ops_sheet_sql = ops_sheet.replace("'", "''")
    invoice_sheet_sql = invoice_sheet.replace("'", "''")

    selected_cols = [
        f"s.row_json->>'{event_col}' AS event_id",
    ]
    if customer_col:
        selected_cols.append(f"s.row_json->>'{customer_col}' AS customer_id")
    if contract_col:
        selected_cols.append(f"s.row_json->>'{contract_col}' AS contract_id")
    if service_type_col:
        selected_cols.append(f"s.row_json->>'{service_type_col}' AS service_type")
    if service_date_col:
        selected_cols.append(f"s.row_json->>'{service_date_col}' AS service_date")

    status_filter = ""
    if status_col:
        status_filter = f" AND LOWER(TRIM(COALESCE(s.row_json->>'{status_col}', ''))) = 'completed'"

    return (
        f"SELECT {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        "LEFT JOIN dataset_rows i "
        "ON s.dataset_id = i.dataset_id "
        f"AND i.sheet_name = '{invoice_sheet_sql}' "
        f"AND NULLIF(LOWER(TRIM(s.row_json->>'{event_col}')), '') = NULLIF(LOWER(TRIM(i.row_json->>'{event_col}')), '') "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{ops_sheet_sql}'"
        f"{status_filter} "
        "AND i.row_index IS NULL "
        "LIMIT %(limit)s"
    )


def _build_invoiced_sent_ops_fallback_sql(available_sheets: list[str], cols: list[str]) -> str | None:
    ops_sheet = _pick_sheet_by_keywords(available_sheets, ["ops", "event"]) or _pick_sheet_by_keywords(available_sheets, ["operation", "event"])
    invoice_sheet = _pick_sheet_by_keywords(available_sheets, ["invoice"])
    if not ops_sheet or not invoice_sheet:
        return None

    event_col = _pick_col_by_keywords(cols, ["event_id", "eventid", "event"])
    status_col = _pick_col_by_keywords(cols, ["status", "event_status", "state"])
    customer_col = _pick_col_by_keywords(cols, ["customer_id", "customer"])
    contract_col = _pick_col_by_keywords(cols, ["contract_id", "contract"])
    service_type_col = _pick_col_by_keywords(cols, ["service_type", "service", "type"])
    service_date_col = _pick_col_by_keywords(cols, ["service_date", "date"])

    if not event_col:
        return None

    ops_sheet_sql = ops_sheet.replace("'", "''")
    invoice_sheet_sql = invoice_sheet.replace("'", "''")

    selected_cols = [
        f"s.row_json->>'{event_col}' AS event_id",
    ]
    if customer_col:
        selected_cols.append(f"s.row_json->>'{customer_col}' AS customer_id")
    if contract_col:
        selected_cols.append(f"s.row_json->>'{contract_col}' AS contract_id")
    if service_type_col:
        selected_cols.append(f"s.row_json->>'{service_type_col}' AS service_type")
    if service_date_col:
        selected_cols.append(f"s.row_json->>'{service_date_col}' AS service_date")

    status_filter = ""
    if status_col:
        status_filter = f" AND LOWER(TRIM(COALESCE(s.row_json->>'{status_col}', ''))) = 'completed'"

    invoice_status_expr = (
        "LOWER(TRIM(COALESCE("
        "i.row_json->>'invoice_status', "
        "i.row_json->>'Invoice Status', "
        "i.row_json->>'invoice status', "
        "i.row_json->>'status', "
        "'')))"
    )

    return (
        f"SELECT {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        "JOIN dataset_rows i "
        "ON s.dataset_id = i.dataset_id "
        f"AND i.sheet_name = '{invoice_sheet_sql}' "
        f"AND NULLIF(LOWER(TRIM(s.row_json->>'{event_col}')), '') = NULLIF(LOWER(TRIM(i.row_json->>'{event_col}')), '') "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{ops_sheet_sql}'"
        f"{status_filter} "
        f"AND {invoice_status_expr} = 'sent' "
        "LIMIT %(limit)s"
    )


def _build_invoiced_ops_with_amount_fallback_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
) -> str | None:
    ops_sheet = _pick_sheet_by_keywords(available_sheets, ["ops", "event"]) or _pick_sheet_by_keywords(available_sheets, ["operation", "event"])
    invoice_sheet = _pick_sheet_by_keywords(available_sheets, ["invoice"])
    if not ops_sheet or not invoice_sheet:
        return None

    ops_event_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["event_id", "eventid", "event"])
    ops_status_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["status", "event_status", "state"])
    ops_customer_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["customer_id", "customer", "customerid"])
    ops_contract_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["contract_id", "contract"])
    ops_service_type_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["service_type", "service", "type"])
    ops_service_date_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["service_date", "date"])

    invoice_event_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["event_id", "eventid", "event"])
    invoice_amount_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["billed_amount", "amount", "total_amount", "invoice_amount"])
    invoice_quantity_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["quantity", "qty"])
    invoice_unit_price_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["unit_price", "price_per_unit", "unit_rate", "rate", "price"])

    if not ops_event_col or not invoice_event_col:
        return None

    invoice_amount_expr = _numeric_json_expr("i", invoice_amount_col)
    if not invoice_amount_expr and invoice_quantity_col and invoice_unit_price_col:
        invoice_quantity_expr = _numeric_json_expr("i", invoice_quantity_col)
        invoice_unit_price_expr = _numeric_json_expr("i", invoice_unit_price_col)
        if invoice_quantity_expr and invoice_unit_price_expr:
            invoice_amount_expr = f"({invoice_quantity_expr} * {invoice_unit_price_expr})"

    if not invoice_amount_expr:
        return None

    selected_cols = [f"s.row_json->>'{ops_event_col}' AS event_id"]
    if ops_customer_col:
        selected_cols.append(f"s.row_json->>'{ops_customer_col}' AS customer_id")
    if ops_contract_col:
        selected_cols.append(f"s.row_json->>'{ops_contract_col}' AS contract_id")
    if ops_service_type_col:
        selected_cols.append(f"s.row_json->>'{ops_service_type_col}' AS service_type")
    if ops_service_date_col:
        selected_cols.append(f"s.row_json->>'{ops_service_date_col}' AS service_date")
    selected_cols.append(f"{invoice_amount_expr} AS invoice_amount")

    status_filter = ""
    if ops_status_col:
        status_filter = f" AND LOWER(TRIM(COALESCE(s.row_json->>'{ops_status_col}', ''))) = 'completed'"

    invoice_sheet_sql = invoice_sheet.replace("'", "''")
    ops_sheet_sql = ops_sheet.replace("'", "''")

    return (
        f"SELECT {', '.join(selected_cols)} "
        "FROM dataset_rows s "
        "JOIN dataset_rows i "
        "ON s.dataset_id = i.dataset_id "
        f"AND i.sheet_name = '{invoice_sheet_sql}' "
        f"AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_event_col}')), '') = NULLIF(LOWER(TRIM(i.row_json->>'{invoice_event_col}')), '') "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{ops_sheet_sql}'"
        f"{status_filter} "
        "ORDER BY invoice_amount DESC, event_id "
        "LIMIT %(limit)s"
    )


def _build_unbilled_total_value_fallback_sql(
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
) -> str | None:
    ops_sheet = _pick_sheet_by_keywords(available_sheets, ["ops", "event"]) or _pick_sheet_by_keywords(available_sheets, ["operation", "event"])
    if not ops_sheet:
        return None

    ops_event_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["event_id", "eventid", "event"])
    ops_customer_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["customer_id", "customer", "customerid"])
    ops_contract_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["contract_id", "contract"])
    ops_service_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["service_type", "service"])
    ops_quantity_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["quantity", "qty"])
    ops_status_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["status", "event_status", "state"])
    ops_expected_col = _pick_sheet_column(sheet_columns_by_sheet, ops_sheet, ["expected_amount", "expected_amt", "amount", "total_value"])
    if not ops_event_col:
        return None

    ops_sheet_sql = ops_sheet.replace("'", "''")

    status_filter = ""
    if ops_status_col:
        status_filter = f" AND LOWER(TRIM(COALESCE(s.row_json->>'{ops_status_col}', ''))) = 'completed'"

    exception_sheet = None
    preferred_exception_sheets = ["Agent_Output"] + [sheet for sheet in available_sheets if sheet != "Agent_Output"]
    for sheet in preferred_exception_sheets:
        if _sheet_has_any_column(sheet_columns_by_sheet, sheet, ["exception_type"]) and _sheet_has_any_column(
            sheet_columns_by_sheet,
            sheet,
            ["event_id", "eventid", "event"],
        ):
            exception_sheet = sheet
            break

    if exception_sheet:
        exception_sheet_sql = exception_sheet.replace("'", "''")
        exception_event_col = _pick_sheet_column(sheet_columns_by_sheet, exception_sheet, ["event_id", "eventid", "event"])
        exception_customer_col = _pick_sheet_column(sheet_columns_by_sheet, exception_sheet, ["customer_id", "customer", "customerid"])
        exception_type_col = _pick_sheet_column(sheet_columns_by_sheet, exception_sheet, ["exception_type", "exception", "status"])
        exception_expected_col = _pick_sheet_column(
            sheet_columns_by_sheet,
            exception_sheet,
            ["expected_amount", "expected_amt", "amount", "total_value"],
        )
        exception_gap_col = _pick_sheet_column(
            sheet_columns_by_sheet,
            exception_sheet,
            ["value_gap", "value_gap_amount", "variance", "difference"],
        )

        if exception_event_col and exception_type_col and (exception_expected_col or exception_gap_col):
            exception_value_expr = _numeric_json_expr("a", exception_expected_col)
            if not exception_value_expr and exception_gap_col:
                exception_value_expr = _numeric_json_expr("a", exception_gap_col)

            if exception_value_expr:
                return (
                    f"SELECT COUNT(*) AS unbilled_events, COALESCE(SUM({exception_value_expr}), 0) AS total_unbilled_value "
                    "FROM dataset_rows a "
                    "JOIN dataset_rows s "
                    "ON s.dataset_id = a.dataset_id "
                    f"AND s.sheet_name = '{ops_sheet_sql}' "
                    f"AND NULLIF(LOWER(TRIM(a.row_json->>'{exception_event_col}')), '') = NULLIF(LOWER(TRIM(s.row_json->>'{ops_event_col}')), '')"
                    " "
                    "WHERE a.dataset_id = %(dataset_id)s "
                    f"AND a.sheet_name = '{exception_sheet_sql}' "
                    f"AND LOWER(TRIM(COALESCE(a.row_json->>'{exception_type_col}', ''))) = 'unbilled'"
                    f"{status_filter}"
                )

    invoice_sheet = _pick_sheet_by_keywords(available_sheets, ["invoice"])
    if not invoice_sheet:
        return None

    invoice_sheet_sql = invoice_sheet.replace("'", "''")
    invoice_event_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["event_id", "eventid", "event"])
    invoice_customer_col = _pick_sheet_column(sheet_columns_by_sheet, invoice_sheet, ["customer_id", "customer", "customerid"])

    join_invoice_customer = ""
    if ops_customer_col and invoice_customer_col:
        join_invoice_customer = (
            f" AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_customer_col}')), '') = "
            f"NULLIF(LOWER(TRIM(i.row_json->>'{invoice_customer_col}')), '')"
        )

    total_value_expr = _numeric_json_expr("s", ops_expected_col)
    contract_join_sql = ""

    contract_sheet = _pick_sheet_by_keywords(available_sheets, ["contract", "rate"]) or _pick_sheet_by_keywords(available_sheets, ["contract"])
    if not total_value_expr and contract_sheet and ops_quantity_col:
        contract_sheet_sql = contract_sheet.replace("'", "''")
        contract_customer_col = _pick_sheet_column(sheet_columns_by_sheet, contract_sheet, ["customer_id", "customer", "customerid"])
        contract_contract_col = _pick_sheet_column(sheet_columns_by_sheet, contract_sheet, ["contract_id", "contract"])
        contract_service_col = _pick_sheet_column(sheet_columns_by_sheet, contract_sheet, ["service_type", "service"])
        contract_rate_col = _pick_sheet_column(
            sheet_columns_by_sheet,
            contract_sheet,
            ["unit_rate", "price_per_unit", "unit_price", "rate", "price"],
        )

        join_contract_customer = ""
        if ops_customer_col and contract_customer_col:
            join_contract_customer = (
                f" AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_customer_col}')), '') = "
                f"NULLIF(LOWER(TRIM(c.row_json->>'{contract_customer_col}')), '')"
            )

        join_contract_id = ""
        if ops_contract_col and contract_contract_col:
            join_contract_id = (
                f" AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_contract_col}')), '') = "
                f"NULLIF(LOWER(TRIM(c.row_json->>'{contract_contract_col}')), '')"
            )

        join_contract_service = ""
        if ops_service_col and contract_service_col:
            join_contract_service = (
                f" AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_service_col}')), '') = "
                f"NULLIF(LOWER(TRIM(c.row_json->>'{contract_service_col}')), '')"
            )

        if contract_rate_col:
            quantity_expr = _numeric_json_expr("s", ops_quantity_col)
            rate_expr = _numeric_json_expr("c", contract_rate_col)
            if quantity_expr and rate_expr:
                total_value_expr = f"({quantity_expr} * {rate_expr})"
                contract_join_sql = (
                    " LEFT JOIN dataset_rows c "
                    "ON c.dataset_id = s.dataset_id "
                    f"AND c.sheet_name = '{contract_sheet_sql}'"
                    f"{join_contract_id}"
                    f"{join_contract_customer}"
                    f"{join_contract_service}"
                )

    if not total_value_expr or not invoice_event_col:
        return None

    return (
        f"SELECT COUNT(*) AS unbilled_events, COALESCE(SUM({total_value_expr}), 0) AS total_unbilled_value "
        "FROM dataset_rows s "
        "LEFT JOIN dataset_rows i "
        "ON s.dataset_id = i.dataset_id "
        f"AND i.sheet_name = '{invoice_sheet_sql}' "
        f"AND NULLIF(LOWER(TRIM(s.row_json->>'{ops_event_col}')), '') = NULLIF(LOWER(TRIM(i.row_json->>'{invoice_event_col}')), '')"
        f"{join_invoice_customer}"
        f"{contract_join_sql} "
        "WHERE s.dataset_id = %(dataset_id)s "
        f"AND s.sheet_name = '{ops_sheet_sql}'"
        f"{status_filter} "
        "AND i.row_index IS NULL"
    )


def _pick_col_by_keywords(cols: list[str], keywords: list[str]) -> str | None:
    for c in cols:
        norm = normalize_col_name(c)
        if any(k in norm for k in keywords):
            return c
    return None


def _date_parse_expr(value_expr: str) -> str:
    return (
        "(CASE "
        f"WHEN ({value_expr}) ~ '^\\d{{4}}-\\d{{1,2}}-\\d{{1,2}}$' THEN to_date(({value_expr}), 'YYYY-MM-DD') "
        f"WHEN ({value_expr}) ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$' AND split_part(({value_expr}), '/', 1)::int > 12 THEN to_date(({value_expr}), 'DD/MM/YYYY') "
        f"WHEN ({value_expr}) ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$' THEN to_date(({value_expr}), 'MM/DD/YYYY') "
        f"WHEN ({value_expr}) ~ '^\\d{{1,2}}-\\d{{1,2}}-\\d{{4}}$' AND split_part(({value_expr}), '-', 1)::int > 12 THEN to_date(({value_expr}), 'DD-MM-YYYY') "
        f"WHEN ({value_expr}) ~ '^\\d{{1,2}}-\\d{{1,2}}-\\d{{4}}$' THEN to_date(({value_expr}), 'MM-DD-YYYY') "
        "ELSE NULL END)"
    )

class AskSqlBody(BaseModel):
    dataset_id: str
    question: str
    max_rows: int = 1000
    analyze_all_rows: bool = False


def _build_sql_engine_metadata(
    cols: list[str],
    col_types: dict[str, str],
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
    relationships_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    numeric_hints = {
        "amount", "revenue", "profit", "sales", "total", "price", "cost", "qty", "quantity", "tax", "discount", "value", "gap"
    }
    column_types_by_sheet: dict[str, dict[str, str]] = {}
    single_sheet = len(available_sheets or []) == 1
    for sheet in available_sheets or []:
        per_sheet: dict[str, str] = {}
        for column in sorted(sheet_columns_by_sheet.get(sheet) or set()):
            normalized = normalize_col_name(column)
            inferred_type = "numeric" if any(token in normalized for token in numeric_hints) else "text"
            if single_sheet and column in (col_types or {}):
                inferred_type = str(col_types.get(column) or inferred_type)
            per_sheet[str(column)] = inferred_type
        column_types_by_sheet[str(sheet)] = per_sheet

    if single_sheet and available_sheets:
        base_sheet = available_sheets[0]
        for column in cols or []:
            column_types_by_sheet.setdefault(base_sheet, {})[str(column)] = str(col_types.get(column) or column_types_by_sheet.get(base_sheet, {}).get(str(column), "text"))

    derived_metrics_by_sheet = detect_derived_metrics_by_sheet(
        {
            str(sheet): [str(column) for column in sorted(sheet_columns_by_sheet.get(sheet) or set())]
            for sheet in (available_sheets or [])
        },
        column_types_by_sheet=column_types_by_sheet,
    )
    if isinstance(relationships_payload, dict):
        payload_derived_metrics = relationships_payload.get("derived_metrics_by_sheet") or {}
        if isinstance(payload_derived_metrics, dict):
            for sheet_name, sheet_metrics in payload_derived_metrics.items():
                if not isinstance(sheet_metrics, dict):
                    continue
                derived_metrics_by_sheet[str(sheet_name)] = {
                    str(metric_name): str(formula)
                    for metric_name, formula in sheet_metrics.items()
                    if str(metric_name).strip() and str(formula).strip()
                }

    return {
        "sheets": [str(sheet) for sheet in (available_sheets or [])],
        "columns_by_sheet": {
            str(sheet): sorted([str(column) for column in (sheet_columns_by_sheet.get(sheet) or set())])
            for sheet in (available_sheets or [])
        },
        "column_types_by_sheet": column_types_by_sheet,
        "derived_metrics": flatten_derived_metrics_by_sheet(derived_metrics_by_sheet),
        "derived_metrics_by_sheet": derived_metrics_by_sheet,
        "relationships": relationships or [],
    }


def _pick_dimension_from_question(question: str, columns: set[str]) -> str | None:
    dimensions = _pick_dimensions_from_question(question, columns)
    return dimensions[0] if dimensions else None


def _pick_dimensions_from_question(question: str, columns: set[str]) -> list[str]:
    normalized_map = {normalize_col_name(column): column for column in (columns or set())}
    keyword_groups = [
        ("customer", ["customer", "customer_id", "customer_name"]),
        ("region", ["region", "state", "country", "city"]),
        ("product", ["product", "product_name", "item", "sku"]),
        ("status", ["status", "invoice_status", "state"]),
        ("year", ["year"]),
        ("month", ["month"]),
    ]
    q_norm = normalize_col_name(question)
    matches: list[tuple[int, str]] = []
    for trigger, candidates in keyword_groups:
        if trigger not in q_norm:
            continue
        trigger_pos = q_norm.find(trigger)
        for candidate in candidates:
            if candidate in normalized_map:
                resolved = normalized_map[candidate]
                if resolved not in [value for _, value in matches]:
                    matches.append((trigger_pos if trigger_pos >= 0 else 10_000, resolved))
                break
    return [value for _, value in sorted(matches, key=lambda item: item[0])]


def _question_requests_dimension_listing(question: str) -> bool:
    q_norm = normalize_col_name(question)
    if not any(token in q_norm for token in ["list", "show", "display", "give_me"]):
        return False
    metric_tokens = [
        "sum",
        "total",
        "avg",
        "average",
        "max",
        "min",
        "count",
        "number_of",
        "revenue",
        "sales",
        "profit",
        "amount",
        "price",
        "value",
        "quantity",
        "qty",
        "top",
        "bottom",
        "highest",
        "lowest",
        "most",
        "least",
    ]
    return not any(token in q_norm for token in metric_tokens)


def _question_requests_metric(question: str) -> bool:
    q_norm = normalize_col_name(question)
    metric_tokens = [
        "sum",
        "total",
        "avg",
        "average",
        "max",
        "min",
        "count",
        "number_of",
        "revenue",
        "sales",
        "profit",
        "amount",
        "price",
        "value",
        "quantity",
        "qty",
    ]
    return any(token in q_norm for token in metric_tokens)


def _question_requests_grouping(question: str) -> bool:
    q_text = str(question or "").lower()
    q_norm = normalize_col_name(question)
    return any(
        token in q_text or token in q_norm
        for token in [" by ", " per ", "group by", "grouped", "breakdown", "split by", "across", "for each"]
    )


def _normalize_text_phrase(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _classify_query_shape(
    question: str,
    dimensions: list[str],
    filters: list[dict[str, Any]],
    measure: dict[str, Any] | None,
) -> str:
    return classify_query_shape_compat(question, dimensions, filters, measure)


def _extract_text_filters_from_question(question: str, columns: set[str]) -> list[dict[str, Any]]:
    normalized_map = {normalize_col_name(column): column for column in (columns or set())}
    q_text = str(question or "").strip().lower()
    filters: list[dict[str, Any]] = []
    seen_columns: set[str] = set()

    for normalized_column, original_column in normalized_map.items():
        if normalized_column in {"year", "month"}:
            continue
        matchers = [
            rf"\b{re.escape(normalized_column.replace('_', ' '))}\b\s*(?:=|is)\s*([a-z0-9][a-z0-9 _-]*)",
            rf"\bin\s+{re.escape(normalized_column.replace('_', ' '))}\b\s*(?:=|is)?\s*([a-z0-9][a-z0-9 _-]*)",
            rf"\bfor\s+{re.escape(normalized_column.replace('_', ' '))}\b\s*(?:=|is)?\s*([a-z0-9][a-z0-9 _-]*)",
        ]
        for pattern in matchers:
            match = re.search(pattern, q_text)
            if not match:
                continue
            if original_column in seen_columns:
                break
            raw_value = re.split(r"\b(?:and|or|by|with|where)\b", str(match.group(1) or ""), maxsplit=1)[0].strip(" _-")
            if not raw_value:
                continue
            filters.append(
                {
                    "column": original_column,
                    "operator": "=",
                    "value": raw_value.replace("_", " "),
                }
            )
            seen_columns.add(original_column)
            break

    implied_patterns = {
        "product": [
            r"\bproduct\s+([a-z0-9][a-z0-9 _-]*)",
            r"\bof\s+product\s+([a-z0-9][a-z0-9 _-]*)",
        ],
        "region": [
            r"\bin\s+([a-z0-9][a-z0-9 _-]*)\s+region\b",
        ],
    }
    stop_words = r"(?:and|or|by|with|where|for|of|what|is|the|sales|value|revenue|price|amount|in)"
    for normalized_column, original_column in normalized_map.items():
        if original_column in seen_columns:
            continue
        for pattern in implied_patterns.get(normalized_column, []):
            match = re.search(pattern, q_text)
            if not match:
                continue
            raw_value = re.split(stop_words, str(match.group(1) or ""), maxsplit=1, flags=re.IGNORECASE)[0].strip(" _-")
            if not raw_value:
                continue
            filters.append(
                {
                    "column": original_column,
                    "operator": "=",
                    "value": raw_value.replace("_", " "),
                }
            )
            seen_columns.add(original_column)
            break

    return filters


def _extract_sample_value_filters_from_question(
    question: str,
    columns: set[str],
    value_samples_for_sheet: dict[str, list[str]] | None,
    existing_filters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    value_samples_for_sheet = value_samples_for_sheet or {}
    existing_filters = existing_filters or []
    seen_columns = {str(item.get("column") or "") for item in existing_filters if isinstance(item, dict)}
    question_text = _normalize_text_phrase(question)
    if not question_text:
        return []

    candidates: list[tuple[int, str, str]] = []
    for column_name in sorted([str(column) for column in (columns or set())]):
        if column_name in seen_columns:
            continue
        sample_values = list(value_samples_for_sheet.get(column_name) or [])
        if not sample_values:
            continue
        column_text = _normalize_text_phrase(column_name)
        column_mentioned = bool(column_text and re.search(rf"\b{re.escape(column_text)}\b", question_text))
        for raw_value in sample_values:
            normalized_value = _normalize_text_phrase(raw_value)
            if not normalized_value or len(normalized_value) < 2:
                continue
            if normalized_value in {"true", "false", "yes", "no", "null", "none", "unknown"}:
                continue
            if not re.search(rf"\b{re.escape(normalized_value)}\b", question_text):
                continue
            score = len(normalized_value)
            if column_mentioned:
                score += 50
            score += min(len(normalized_value.split()), 3) * 10
            candidates.append((score, column_name, normalized_value))

    selected: list[dict[str, Any]] = []
    used_columns: set[str] = set()
    for _score, column_name, normalized_value in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if column_name in used_columns:
            continue
        selected.append({"column": column_name, "operator": "=", "value": normalized_value})
        used_columns.add(column_name)

    return selected


def _build_value_samples_by_sheet(sheet_rows: list[dict[str, Any]] | None) -> dict[str, dict[str, list[str]]]:
    sheet_rows = sheet_rows or []
    collected: dict[str, dict[str, set[str]]] = {}
    for row in sheet_rows:
        if not isinstance(row, dict):
            continue
        sheet_name = str(row.get("sheet_name") or "").strip()
        row_json = row.get("row_json")
        if not sheet_name or not isinstance(row_json, dict):
            continue
        per_sheet = collected.setdefault(sheet_name, {})
        for column_name, value in row_json.items():
            if _is_null_like_value(value):
                continue
            normalized_value = _normalize_text_phrase(value)
            if not normalized_value or len(normalized_value) < 2:
                continue
            if _normalize_numeric_text(str(value)) is not None:
                continue
            if re.fullmatch(r"\d{4} \d{1,2} \d{1,2}", normalized_value):
                continue
            values = per_sheet.setdefault(str(column_name), set())
            if len(values) < 100:
                values.add(normalized_value)

    finalized: dict[str, dict[str, list[str]]] = {}
    for sheet_name, per_sheet in collected.items():
        next_sheet: dict[str, list[str]] = {}
        for column_name, values in per_sheet.items():
            if 1 < len(values) <= 50:
                next_sheet[column_name] = sorted(values)
        if next_sheet:
            finalized[sheet_name] = next_sheet
    return finalized


def _pick_date_column(columns: set[str]) -> str | None:
    preferred = ["date", "order_date", "invoice_date", "delivery_date", "service_date", "transaction_date"]
    normalized_map = {normalize_col_name(column): column for column in (columns or set())}
    for candidate in preferred:
        if candidate in normalized_map:
            return normalized_map[candidate]
    for normalized, original in normalized_map.items():
        if normalized.endswith("date"):
            return original
    return None


def _pick_time_grain_from_question(question: str) -> str | None:
    q_norm = normalize_col_name(question)
    if any(token in q_norm for token in ["by_year", "yearly", "per_year", "sales_trend", "trend_by_year", "year_over_year"]):
        return "year"
    if " by year" in (question or "").lower():
        return "year"
    return None


def _pick_measure_from_question(question: str, columns: set[str]) -> dict[str, Any] | None:
    normalized_map = {normalize_col_name(column): column for column in (columns or set())}
    canonical_map = {normalize_col_name(column).replace("_", ""): column for column in (columns or set())}
    q_norm = normalize_col_name(question)
    q_canonical = q_norm.replace("_", "")
    agg = "count" if any(token in q_norm for token in ["how_many", "count", "number_of"]) else "sum"
    if any(token in q_norm for token in ["revenue", "sales", "total_sales", "total_revenue"]):
        metric_candidates = [
            "total_price", "sales_amount", "net_sales", "total_amount", "amount", "revenue", "billed_amount", "expected_amount", "profit", "quantity"
        ]
    elif any(token in q_norm for token in ["profit", "margin"]):
        metric_candidates = [
            "profit", "gross_profit", "net_profit", "total_price", "sales_amount", "amount", "quantity"
        ]
    elif any(token in q_norm for token in ["quantity", "qty", "units"]):
        metric_candidates = ["quantity", "qty", "units", "amount"]
    else:
        metric_candidates = [
            "amount", "total_amount", "total_price", "revenue", "sales_amount", "net_sales", "expected_amount", "billed_amount", "profit", "value_gap", "quantity"
        ]
    normalized_candidates = [normalize_col_name(candidate) for candidate in metric_candidates]
    canonical_candidates = [candidate.replace("_", "") for candidate in normalized_candidates]
    for candidate, canonical_candidate in zip(normalized_candidates, canonical_candidates):
        if (candidate in q_norm or canonical_candidate in q_canonical) and canonical_candidate in canonical_map:
            return {"column": canonical_map[canonical_candidate], "aggregation": agg if canonical_candidate == "quantity" or agg != "count" else "count", "alias": f"{agg}_{candidate}"}
    if agg == "count":
        return {"column": "*", "aggregation": "count", "alias": "row_count"}
    for candidate, canonical_candidate in zip(normalized_candidates, canonical_candidates):
        if canonical_candidate in canonical_map:
            return {"column": canonical_map[canonical_candidate], "aggregation": agg, "alias": f"{agg}_{candidate}"}
    return {"column": "*", "aggregation": "count", "alias": "row_count"}


def _pick_join_path_for_question(question: str, relationships: list[dict[str, Any]], available_sheets: list[str]) -> list[dict[str, Any]]:
    q_norm = normalize_col_name(question)
    mentioned_sheets = [sheet for sheet in (available_sheets or []) if normalize_col_name(sheet) in q_norm]
    if len(mentioned_sheets) >= 2:
        for relationship in relationships or []:
            left = str(relationship.get("left") or "")
            right = str(relationship.get("right") or "")
            if left in mentioned_sheets and right in mentioned_sheets:
                return [{
                    "left_sheet": left,
                    "left_col": str(relationship.get("left_col") or ""),
                    "right_sheet": right,
                    "right_col": str(relationship.get("right_col") or ""),
                }]
    return []


def _build_query_spec_stub(
    question: str,
    available_sheets: list[str],
    sheet_columns_by_sheet: dict[str, set[str]],
    relationships: list[dict[str, Any]],
    explicit_sheet: str | None,
    max_rows: int,
    value_samples_by_sheet: dict[str, dict[str, list[str]]] | None = None,
) -> QuerySpec | None:
    return build_query_spec_stub_compat(
        question,
        available_sheets,
        sheet_columns_by_sheet,
        relationships,
        explicit_sheet,
        max_rows,
        value_samples_by_sheet,
    )


def _build_degraded_guidance(question: str, issues: list[str], available_sheets: list[str], sheet_columns_by_sheet: dict[str, set[str]]) -> dict[str, Any]:
    ranked = _rank_sheets_for_question(available_sheets, sheet_columns_by_sheet, question)
    sheets = [
        {
            "sheet_name": item["sheet"],
            "sample_columns": item.get("relevant_columns") or sorted(list(sheet_columns_by_sheet.get(item["sheet"]) or set()))[:12],
        }
        for item in ranked[:6]
    ]
    suggested_questions: list[str] = []
    if ranked:
        top_sheet = ranked[0]["sheet"]
        suggested_questions.append(f"Show first 20 rows from {top_sheet}.")
        relevant_columns = ranked[0].get("relevant_columns") or []
        if relevant_columns:
            suggested_questions.append(
                f"Summarize {relevant_columns[0]} from {top_sheet}."
            )
    return {
        "message": "The question could not be mapped to a schema-safe SQL query.",
        "issues": issues,
        "suggested_questions": suggested_questions[:4],
        "sheets": sheets,
    }


def _summarize_query_spec(query_spec: QuerySpec | None, query_shape: str | None = None) -> dict[str, Any] | None:
    if query_spec is None:
        return None

    measures: list[dict[str, Any]] = []
    for measure in list(query_spec.measures or []):
        if not isinstance(measure, dict):
            continue
        measures.append(
            {
                "column": str(measure.get("column") or ""),
                "aggregation": str(measure.get("aggregation") or measure.get("agg") or ""),
                "alias": str(measure.get("alias") or ""),
            }
        )

    filters: list[dict[str, Any]] = []
    for filter_spec in list(query_spec.filters or []):
        if not isinstance(filter_spec, dict):
            continue
        filters.append(
            {
                "column": str(filter_spec.get("column") or ""),
                "operator": str(filter_spec.get("operator") or filter_spec.get("op") or "="),
                "value": filter_spec.get("value"),
                "kind": str(filter_spec.get("kind") or ""),
            }
        )

    return {
        "query_shape": str(query_shape or ""),
        "query_type": str(query_spec.query_type or ""),
        "target_sheet": str(query_spec.target_sheet or ""),
        "columns": [str(item) for item in list(query_spec.columns or []) if str(item).strip()],
        "dimensions": [str(item) for item in list(query_spec.dimensions or []) if str(item).strip()],
        "measures": measures,
        "filters": filters,
        "time_grain": str(query_spec.time_grain or ""),
        "warnings": [str(item) for item in list(query_spec.warnings or []) if str(item).strip()],
        "debug": dict(query_spec.debug or {}),
        "source": str(query_spec.source or ""),
    }


def _build_describe_sheet_response(
    *,
    question: str,
    body,
    query_spec: QuerySpec,
    planner_query_shape: str | None,
    planner_confidence: Any,
    planner_warnings: list[str],
    planner_reasons: list[str],
    extracted_filters: list[dict[str, Any]],
    inferred_value_mappings: list[dict[str, Any]],
    planner_candidate_sheets: list[dict[str, Any]],
    planner_selected_metric: dict[str, Any] | None,
    planner_candidate_metrics: list[dict[str, Any]],
    planner_selected_dimensions: list[str],
    planner_answer_type: str | None,
    planner_query_spec: dict[str, Any] | None,
    planner_summary: dict[str, Any] | None,
    sheet_rows: list[dict[str, Any]],
    planning_context,
    registry_rule_id: str | None,
    result_note: str | None,
    failure_log,
):
    sheet_profile = build_sheet_profile(str(query_spec.target_sheet or ""), planning_context, sheet_rows)
    planner_debug = _build_planner_debug_payload(
        question=question,
        route_intent=query_spec.intent,
        route_source="phase2_describe_sheet_profile",
        planner_confidence=planner_confidence,
        planner_summary=planner_summary,
        planner_query_spec=planner_query_spec,
        planner_query_shape=planner_query_shape,
        planner_warnings=planner_warnings,
        planner_reasons=planner_reasons,
        extracted_filters=extracted_filters,
        inferred_value_mappings=inferred_value_mappings,
        candidate_sheets=planner_candidate_sheets,
        selected_metric=planner_selected_metric,
        candidate_metrics=planner_candidate_metrics,
        selected_dimensions=planner_selected_dimensions,
        answer_type=planner_answer_type,
        sql="",
    )
    return {
        "sql": "",
        "route_intent": query_spec.intent,
        "route_source": "phase2_describe_sheet_profile",
        "registry_rule_id": registry_rule_id,
        "response_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied_limit": None,
        "analyze_all_requested": body.analyze_all_rows,
        "analyze_all_applied": False,
        "result_note": result_note,
        "rows_returned": 1,
        "degraded_mode": False,
        "guidance": None,
        "query_shape": planner_query_shape,
        "planner_confidence": planner_confidence,
        "planner_warnings": planner_warnings,
        "planner_reasons": planner_reasons,
        "planner_summary": planner_summary,
        "extracted_filters": extracted_filters,
        "inferred_value_mappings": inferred_value_mappings,
        "query_spec": planner_query_spec,
        "planner_debug": planner_debug,
        "failure_log": failure_log,
        "rows": [sheet_profile],
        "sheet_profile": sheet_profile,
    }


def _count_rows_for_planner_summary(
    cur,
    dataset_id: str,
    question: str,
    planner_summary: dict[str, Any] | None,
    sql_engine_metadata: dict[str, Any],
) -> int | None:
    if not isinstance(planner_summary, dict):
        return None

    target_sheet = str(planner_summary.get("target_sheet") or "").strip()
    filters = [item for item in list(planner_summary.get("filters") or []) if isinstance(item, dict)]
    if not target_sheet or not filters:
        return None

    count_spec = QuerySpec(
        intent="matched_rows",
        target_sheet=target_sheet,
        measures=[{"column": "*", "aggregation": "count", "alias": "matched_rows"}],
        dimensions=[],
        filters=filters,
        limit=1,
        source="planner",
    )
    count_sql, count_params = build_sql_from_query_spec(count_spec, sql_engine_metadata)
    cur.execute(
        _prepare_sql_for_psycopg(count_sql),
        _build_exec_params(count_sql, dataset_id, 1, question, count_params),
    )
    count_rows = cur.fetchall() or []
    if not count_rows:
        return 0
    try:
        return int((count_rows[0] or {}).get("matched_rows") or 0)
    except Exception:
        return None


def _enrich_aggregate_with_filters_rows(
    rows: list[dict[str, Any]] | None,
    planner_summary: dict[str, Any] | None,
    matched_rows: int | None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return rows or []
    if not isinstance(planner_summary, dict):
        return rows

    filters = [item for item in list(planner_summary.get("filters") or []) if isinstance(item, dict)]
    if not filters:
        return rows

    enriched_row: dict[str, Any] = {}
    for filter_spec in filters:
        column_name = normalize_col_name(str(filter_spec.get("column") or ""))
        if not column_name:
            continue
        enriched_row[column_name] = filter_spec.get("value")
    if matched_rows is not None:
        enriched_row["matched_rows"] = matched_rows
    enriched_row.update(rows[0])
    return [enriched_row]


def _is_top1_question(question: str) -> bool:
    q = (question or "").lower()
    patterns = [
        r"\bhighest\b",
        r"\blowest\b",
        r"\btop\s*1\b",
        r"\btop\s+one\b",
        r"\bbottom\s*1\b",
        r"\bbottom\s+one\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\btop\s+seller\b",
        r"\bbest\s+seller\b",
        r"\btop\s+sales\s*person\b",
        r"\btop\s+salesperson\b",
        r"\bwho\s+is\s+(the\s+)?top\s+\w+\b",
    ]
    return any(re.search(p, q) for p in patterns)

@app.post("/ask_sql")
def ask_sql(body: AskSqlBody):
    if sqlparse is None or psycopg is None or dict_row is None:
        raise HTTPException(status_code=500, detail="SQL dependencies missing: install sqlparse and psycopg[binary] in the active Python environment")

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    requested_top_n = _extract_requested_top_n(q)
    # If user explicitly asked top/bottom N (N>1), do not collapse to top-1 on words like "highest".
    force_top_1 = (requested_top_n == 1) or (_is_top1_question(q) and requested_top_n is None)
    route_intent = "unknown"
    route_source = "unknown"
    result_note = None
    registry_rule_id = None
    degraded_mode = False
    guidance = None
    planner_query_shape = None
    planner_summary = None
    planner_confidence = None
    planner_warnings: list[str] = []
    planner_reasons: list[str] = []
    extracted_filters: list[dict[str, Any]] = []
    inferred_value_mappings: list[dict[str, Any]] = []
    query_spec = None
    planner_query_spec = None
    planner_candidate_sheets: list[dict[str, Any]] = []
    planner_selected_metric: dict[str, Any] | None = None
    planner_candidate_metrics: list[dict[str, Any]] = []
    planner_selected_dimensions: list[str] = []
    planner_answer_type: str | None = None
    planner_requested_dimension: str | None = None
    planner_requested_metric: str | None = None
    planner_resolved_dimension: str | None = None
    planner_resolved_metric: str | None = None
    planner_missing_dimension_reason: str | None = None
    planner_missing_metric_reason: str | None = None
    planner_fallback_blocked_reason: str | None = None
    planner_available_dimensions: list[str] = []
    planner_available_metrics: list[str] = []
    planner_debug: dict[str, Any] | None = None
    failure_log: dict[str, Any] | None = None

    if sb is not None:
        try:
            meta = sb.table("datasets").select("name,row_count,col_count,columns,column_types,relationships").eq("id", body.dataset_id).single().execute().data
        except Exception as e:
            meta = sb.table("datasets").select("name,row_count,col_count,columns").eq("id", body.dataset_id).single().execute().data
    else:
        meta_list = _rest_request("GET", "datasets", params={"select": "name,row_count,col_count,columns,column_types,relationships", "id": f"eq.{body.dataset_id}", "limit": 1})
        meta = meta_list[0] if meta_list else {}

    cols = meta.get("columns") or []
    col_types = meta.get("column_types") or {}
    relationships_payload = meta.get("relationships") or {}
    relationships = relationships_payload.get("relationships", []) if isinstance(relationships_payload, dict) else []
    semantic_roles = relationships_payload.get("semantic_roles", {}) if isinstance(relationships_payload, dict) else {}
    semantic_roles = _sanitize_semantic_roles(semantic_roles)
    sheet_columns_by_sheet: dict[str, set[str]] = {}
    if sb is not None:
        try:
            sheet_rows = (
                sb.table("dataset_rows")
                .select("sheet_name,row_json")
                .eq("dataset_id", body.dataset_id)
                .limit(2000)
                .execute()
                .data
            ) or []
            available_sheets = sorted({str(r.get("sheet_name", "")).strip() for r in sheet_rows if r.get("sheet_name")})
            for r in sheet_rows:
                s_name = str(r.get("sheet_name", "")).strip()
                rj = r.get("row_json")
                if s_name and isinstance(rj, dict):
                    sheet_columns_by_sheet.setdefault(s_name, set()).update(rj.keys())
        except Exception:
            available_sheets = []
    else:
        try:
            sheet_rows = _rest_request(
                "GET",
                "dataset_rows",
                params={
                    "select": "sheet_name,row_json",
                    "dataset_id": f"eq.{body.dataset_id}",
                    "limit": 2000,
                },
            ) or []
            available_sheets = sorted({str(r.get("sheet_name", "")).strip() for r in sheet_rows if r.get("sheet_name")})
            for r in sheet_rows:
                s_name = str(r.get("sheet_name", "")).strip()
                rj = r.get("row_json")
                if s_name and isinstance(rj, dict):
                    sheet_columns_by_sheet.setdefault(s_name, set()).update(rj.keys())
        except Exception:
            available_sheets = []

    value_samples_by_sheet = _build_value_samples_by_sheet(sheet_rows if 'sheet_rows' in locals() else [])

    summary_lag_sql = None
    semantic_role_sheet_summary_sql = None
    customer_value_gap_summary_sql = None
    exception_customer_list_sql = None
    exception_detail_sql = None
    billing_lag_sql = None
    invoiced_amount_sql = None
    top_revenue_year_sql = None
    explicit_sheet_sql = None
    registry_sql = None
    allow_protected_fallback = False
    summary_lag_threshold = None
    summary_lag_source_sheet = None
    summary_lag_gap_condition_text = None
    explicit_sheet = _extract_explicit_sheet_reference(q, available_sheets)
    if _is_customer_value_gap_summary_question(q):
        customer_value_gap_summary_sql = _build_customer_value_gap_summary_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            relationships=relationships,
            semantic_roles=semantic_roles,
            forced_sheet=explicit_sheet,
        )
    if _is_exception_customer_list_question(q):
        exception_customer_list_sql = _build_exception_customer_list_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            relationships=relationships,
            semantic_roles=semantic_roles,
            forced_sheet=explicit_sheet,
        )
    if _is_exception_detail_question(q):
        exception_detail_sql = _build_exception_detail_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            relationships=relationships,
            semantic_roles=semantic_roles,
            forced_sheet=explicit_sheet,
        )
    if _is_semantic_role_sheet_summary_question(q):
        semantic_role_sheet_summary_sql = _build_semantic_role_sheet_summary_sql(semantic_roles)
    if _is_billing_lag_value_gap_question(q):
        summary_lag_sql = _build_billing_lag_value_gap_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            relationships=relationships,
            semantic_roles=semantic_roles,
            forced_sheet=explicit_sheet,
        )
        if summary_lag_sql:
            summary_lag_threshold = _extract_lag_days_threshold(q) or 7
            summary_lag_source_sheet = explicit_sheet or _pick_billing_lag_source_sheet(available_sheets, sheet_columns_by_sheet, q)
            gap_condition = _extract_value_gap_condition(q)
            if gap_condition is not None:
                summary_lag_gap_condition_text = f"value_gap {gap_condition[0]} {gap_condition[1]}"
            elif _question_requires_positive_gap(q):
                summary_lag_gap_condition_text = "value_gap > 0"
        elif explicit_sheet:
            route_intent = "billing_lag_value_gap"
            route_source = "explicit_sheet_unanswerable"
            explicit_sheet_sql = _build_empty_sheet_sql(explicit_sheet)
            result_note = _build_explicit_sheet_schema_note(
                explicit_sheet,
                sheet_columns_by_sheet,
                ["event_id", "customer/customer_id", "billing_lag_days", "value_gap"],
            )
    elif _is_billing_lag_question(q):
        billing_lag_sql = _build_billing_lag_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            relationships=relationships,
            semantic_roles=semantic_roles,
            forced_sheet=explicit_sheet,
        )
        if not billing_lag_sql and explicit_sheet:
            route_intent = "billing_lag"
            route_source = "explicit_sheet_unanswerable"
            explicit_sheet_sql = _build_empty_sheet_sql(explicit_sheet)
            result_note = _build_explicit_sheet_schema_note(
                explicit_sheet,
                sheet_columns_by_sheet,
                ["event_id", "customer/customer_id", "billing_lag_days"],
            )

    if _is_invoiced_ops_with_amount_question(q) and not explicit_sheet_sql:
        invoiced_amount_sql = _build_invoiced_ops_with_amount_fallback_sql(available_sheets, sheet_columns_by_sheet)

    if _is_top_revenue_by_year_question(q) and not explicit_sheet_sql:
        top_revenue_year_sql = _build_top_revenue_by_year_sql(
            available_sheets,
            sheet_columns_by_sheet,
            q,
            forced_sheet=explicit_sheet,
        )

    allow_protected_fallback = bool(
        semantic_role_sheet_summary_sql
        or
        customer_value_gap_summary_sql
        or exception_customer_list_sql
        or exception_detail_sql
        or summary_lag_sql
        or billing_lag_sql
        or invoiced_amount_sql
        or top_revenue_year_sql
        or explicit_sheet_sql
        or _is_unbilled_total_value_question(q)
    )

    dataset_generated_rules = _extract_dataset_generated_registry_rules(meta.get("relationships"))
    matched_registry_rule = None
    if dataset_generated_rules and not explicit_sheet_sql and not semantic_role_sheet_summary_sql and not customer_value_gap_summary_sql and not exception_customer_list_sql and not exception_detail_sql:
        matched_registry_rule = _match_rules_registry_sql_rule(q, dataset_generated_rules)
        if matched_registry_rule:
            registry_sql = (matched_registry_rule.get("sql_template") or "").strip()
            if registry_sql.startswith("```"):
                registry_sql = re.sub(r"^```[a-zA-Z]*\n?", "", registry_sql).strip()
                if registry_sql.endswith("```"):
                    registry_sql = registry_sql[:-3].strip()
            if _rule_sql_matches_dataset_schema(registry_sql, available_sheets, sheet_columns_by_sheet):
                route_intent = str(matched_registry_rule.get("intent") or "dataset_generated_registry_sql")
                route_source = "dataset_generated_registry_sql_template"
                registry_rule_id = matched_registry_rule.get("id")
            else:
                registry_sql = ""
                matched_registry_rule = None

    if not registry_sql:
        matched_registry_rule = _match_rules_registry_sql_rule(q, _load_rules_registry())

    if matched_registry_rule and not explicit_sheet_sql and not customer_value_gap_summary_sql and not exception_customer_list_sql and not exception_detail_sql and not registry_sql:
        registry_sql = (matched_registry_rule.get("sql_template") or "").strip()
        if registry_sql.startswith("```"):
            registry_sql = re.sub(r"^```[a-zA-Z]*\n?", "", registry_sql).strip()
            if registry_sql.endswith("```"):
                registry_sql = registry_sql[:-3].strip()
        if _rule_sql_matches_dataset_schema(registry_sql, available_sheets, sheet_columns_by_sheet):
            route_intent = str(matched_registry_rule.get("intent") or "registry_sql")
            route_source = "rules_registry_sql_template"
            registry_rule_id = matched_registry_rule.get("id")
        else:
            registry_sql = ""
            matched_registry_rule = None

    system, user, sheet_schema_context, rel_desc = _build_sql_generation_messages(
        q,
        cols,
        col_types,
        available_sheets,
        sheet_columns_by_sheet,
        relationships,
        analyze_all_rows=body.analyze_all_rows,
        requested_top_n=requested_top_n,
        force_top_1=force_top_1,
        explicit_sheet=explicit_sheet,
    )

    sql_engine_metadata = _build_sql_engine_metadata(
        cols,
        col_types,
        available_sheets,
        sheet_columns_by_sheet,
        relationships,
        relationships_payload=relationships_payload if isinstance(relationships_payload, dict) else None,
    )
    planning_context = build_planning_context(
        dataset_id=body.dataset_id,
        cols=cols,
        col_types=col_types,
        relationships_payload={
            **(relationships_payload if isinstance(relationships_payload, dict) else {}),
            "semantic_roles": semantic_roles,
        },
        sheet_rows=sheet_rows if 'sheet_rows' in locals() else [],
    )
    deterministic_params = None

    if semantic_role_sheet_summary_sql:
        sql = semantic_role_sheet_summary_sql
        route_intent = "semantic_role_sheet_summary"
        route_source = "deterministic_semantic_role_sheet_summary"
    elif customer_value_gap_summary_sql:
        sql = customer_value_gap_summary_sql
        route_intent = "customer_value_gap_summary"
        route_source = "deterministic_customer_value_gap_summary"
    elif exception_customer_list_sql:
        sql = exception_customer_list_sql
        route_intent = "exception_customer_list"
        route_source = "deterministic_exception_customer_list"
    elif exception_detail_sql:
        sql = exception_detail_sql
        route_intent = "exception_detail"
        route_source = "deterministic_exception_detail"
    elif registry_sql:
        sql = registry_sql
    elif explicit_sheet_sql:
        sql = explicit_sheet_sql
    elif top_revenue_year_sql:
        sql = top_revenue_year_sql
        route_intent = "top_revenue_by_year"
        route_source = "deterministic_year_revenue_fallback"
    else:
        planner_result = plan_query(
            q,
            planning_context,
            explicit_sheet=explicit_sheet,
            max_rows=body.max_rows,
        )
        query_spec = planner_result.query_spec
        planner_query_shape = planner_result.query_shape
        planner_confidence = planner_result.confidence
        planner_warnings = list(planner_result.warnings or [])
        planner_reasons = list(planner_result.reasons or [])
        extracted_filters = list(planner_result.extracted_filters or [])
        inferred_value_mappings = list(planner_result.inferred_value_mappings or [])
        planner_candidate_sheets = list(planner_result.candidate_sheets or [])
        planner_selected_metric = planner_result.selected_metric
        planner_candidate_metrics = list(planner_result.candidate_metrics or [])
        planner_selected_dimensions = list(planner_result.selected_dimensions or [])
        planner_answer_type = planner_result.answer_type
        planner_requested_dimension = planner_result.requested_dimension
        planner_requested_metric = planner_result.requested_metric
        planner_resolved_dimension = planner_result.resolved_dimension
        planner_resolved_metric = planner_result.resolved_metric
        planner_missing_dimension_reason = planner_result.missing_dimension_reason
        planner_missing_metric_reason = planner_result.missing_metric_reason
        planner_fallback_blocked_reason = planner_result.fallback_blocked_reason
        planner_available_dimensions = list(planner_result.available_dimensions or [])
        planner_available_metrics = list(planner_result.available_metrics or [])
        planner_query_spec = query_spec.model_dump() if query_spec is not None else None
        if query_spec is not None:
            planner_summary = _summarize_query_spec(query_spec, planner_query_shape)
        else:
            planner_summary = {
                "query_shape": planner_query_shape,
                "target_sheet": str(planner_result.target_sheet or ""),
                "dimensions": [],
                "measures": [],
                "filters": extracted_filters,
                "time_grain": "",
                "warnings": planner_warnings,
                "source": "phase2_planner",
            }

        if planner_result.degraded_mode:
            degraded_mode = True
            guidance = planner_result.guidance or _build_degraded_guidance(
                q,
                planner_warnings or ["Phase 2 planner confidence was below the execution threshold."],
                available_sheets,
                sheet_columns_by_sheet,
            )
            route_intent = getattr(query_spec, "intent", None) or planner_result.canonical_intent or "phase2_planner_degraded"
            route_source = "phase2_planner_degraded"
            sql = ""
            planner_debug = _build_planner_debug_payload(
                question=q,
                route_intent=route_intent,
                route_source=route_source,
                planner_confidence=planner_confidence,
                planner_summary=planner_summary,
                planner_query_spec=planner_query_spec,
                planner_query_shape=planner_query_shape,
                planner_warnings=planner_warnings,
                planner_reasons=planner_reasons,
                extracted_filters=extracted_filters,
                inferred_value_mappings=inferred_value_mappings,
                candidate_sheets=planner_candidate_sheets,
                selected_metric=planner_selected_metric,
                candidate_metrics=planner_candidate_metrics,
                selected_dimensions=planner_selected_dimensions,
                answer_type=planner_answer_type,
                sql="",
                failure_reason="planner_degraded",
                requested_dimension=planner_requested_dimension,
                requested_metric=planner_requested_metric,
                resolved_dimension=planner_resolved_dimension,
                resolved_metric=planner_resolved_metric,
                missing_dimension_reason=planner_missing_dimension_reason,
                missing_metric_reason=planner_missing_metric_reason,
                fallback_blocked_reason=planner_fallback_blocked_reason,
                available_dimensions=planner_available_dimensions,
                available_metrics=planner_available_metrics,
            )
        elif query_spec is not None:
            if query_spec.query_type == "describe_sheet":
                return _build_describe_sheet_response(
                    question=q,
                    body=body,
                    query_spec=query_spec,
                    planner_query_shape=planner_query_shape,
                    planner_confidence=planner_confidence,
                    planner_warnings=planner_warnings,
                    planner_reasons=planner_reasons,
                    extracted_filters=extracted_filters,
                    inferred_value_mappings=inferred_value_mappings,
                    planner_candidate_sheets=planner_candidate_sheets,
                    planner_selected_metric=planner_selected_metric,
                    planner_candidate_metrics=planner_candidate_metrics,
                    planner_selected_dimensions=planner_selected_dimensions,
                    planner_answer_type=planner_answer_type,
                    planner_query_spec=planner_query_spec,
                    planner_summary=planner_summary,
                    sheet_rows=sheet_rows if 'sheet_rows' in locals() else [],
                    planning_context=planning_context,
                    registry_rule_id=registry_rule_id,
                    result_note=result_note,
                    failure_log=failure_log,
                )
            try:
                sql, deterministic_params = build_sql_from_query_spec(query_spec, sql_engine_metadata)
                validation_issues = validate_sql(sql, query_spec, sql_engine_metadata)
                if validation_issues:
                    degraded_mode = True
                    guidance = _build_degraded_guidance(q, validation_issues, available_sheets, sheet_columns_by_sheet)
                    route_intent = query_spec.intent
                    route_source = "phase2_query_spec_validation_failed"
                    sql = ""
                    planner_debug = _build_planner_debug_payload(
                        question=q,
                        route_intent=route_intent,
                        route_source=route_source,
                        planner_confidence=planner_confidence,
                        planner_summary=planner_summary,
                        planner_query_spec=planner_query_spec,
                        planner_query_shape=planner_query_shape,
                        planner_warnings=planner_warnings,
                        planner_reasons=planner_reasons,
                        extracted_filters=extracted_filters,
                        inferred_value_mappings=inferred_value_mappings,
                        candidate_sheets=planner_candidate_sheets,
                        selected_metric=planner_selected_metric,
                        candidate_metrics=planner_candidate_metrics,
                        selected_dimensions=planner_selected_dimensions,
                        answer_type=planner_answer_type,
                        sql="",
                        failure_reason="sql_validation_failed",
                    )
                else:
                    route_intent = query_spec.intent
                    route_source = "phase2_query_spec_deterministic"
            except QueryBuildError as exc:
                degraded_mode = True
                guidance = _build_degraded_guidance(q, [str(exc)], available_sheets, sheet_columns_by_sheet)
                route_intent = getattr(query_spec, "intent", "query_spec")
                route_source = "phase2_query_spec_build_failed"
                sql = ""
                planner_debug = _build_planner_debug_payload(
                    question=q,
                    route_intent=route_intent,
                    route_source=route_source,
                    planner_confidence=planner_confidence,
                    planner_summary=planner_summary,
                    planner_query_spec=planner_query_spec,
                    planner_query_shape=planner_query_shape,
                    planner_warnings=planner_warnings,
                    planner_reasons=planner_reasons,
                    extracted_filters=extracted_filters,
                    inferred_value_mappings=inferred_value_mappings,
                    candidate_sheets=planner_candidate_sheets,
                    selected_metric=planner_selected_metric,
                    candidate_metrics=planner_candidate_metrics,
                    selected_dimensions=planner_selected_dimensions,
                    answer_type=planner_answer_type,
                    sql="",
                    failure_reason=str(exc),
                )
        else:
            degraded_mode = True
            guidance = _build_degraded_guidance(q, ["No safe target sheet could be inferred for this question."], available_sheets, sheet_columns_by_sheet)
            route_intent = "phase2_query_spec_unresolved"
            route_source = "phase2_query_spec_unresolved"
            sql = ""
            planner_debug = _build_planner_debug_payload(
                question=q,
                route_intent=route_intent,
                route_source=route_source,
                planner_confidence=planner_confidence,
                planner_summary=planner_summary,
                planner_query_spec=planner_query_spec,
                planner_query_shape=planner_query_shape,
                planner_warnings=planner_warnings,
                planner_reasons=planner_reasons,
                extracted_filters=extracted_filters,
                inferred_value_mappings=inferred_value_mappings,
                candidate_sheets=planner_candidate_sheets,
                selected_metric=planner_selected_metric,
                candidate_metrics=planner_candidate_metrics,
                selected_dimensions=planner_selected_dimensions,
                answer_type=planner_answer_type,
                sql="",
                failure_reason="target_sheet_unresolved",
            )

        if degraded_mode and not ALLOW_DIRECT_LLM_SQL:
            failure_log = log_analytics_failure(
                {
                    "raw_user_question": q,
                    "detected_intent": route_intent,
                    "planner_confidence": planner_confidence,
                    "chosen_sheet": (planner_summary or {}).get("target_sheet") if isinstance(planner_summary, dict) else None,
                    "candidate_sheets": planner_candidate_sheets,
                    "chosen_metric": planner_selected_metric,
                    "candidate_metrics": planner_candidate_metrics,
                    "chosen_dimensions": planner_selected_dimensions,
                    "extracted_filters": extracted_filters,
                    "value_to_column_matches": inferred_value_mappings,
                    "generated_sql": "",
                    "validation_result": route_source,
                    "execution_result": "degraded",
                    "failure_reason": (planner_debug or {}).get("failure_reason") or route_source,
                }
            )
            return {
                "sql": "",
                "route_intent": route_intent,
                "route_source": route_source,
                "registry_rule_id": registry_rule_id,
                "response_generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "applied_limit": None,
                "analyze_all_requested": body.analyze_all_rows,
                "analyze_all_applied": False,
                "result_note": result_note,
                "rows_returned": 0,
                "rows": [],
                "degraded_mode": True,
                "degraded_reason": planner_fallback_blocked_reason or "planner_degraded",
                "guidance": guidance,
                "query_shape": planner_query_shape,
                "planner_confidence": planner_confidence,
                "planner_warnings": planner_warnings,
                "planner_reasons": planner_reasons,
                "planner_summary": planner_summary,
                "extracted_filters": extracted_filters,
                "inferred_value_mappings": inferred_value_mappings,
                "query_spec": planner_query_spec,
                "planner_debug": planner_debug,
                "failure_log": failure_log,
            }

        if degraded_mode and ALLOW_DIRECT_LLM_SQL:
            degraded_mode = False
            guidance = None
            route_intent = "llm_generated"
            route_source = "llm"
            try:
                sql = _generate_sql_with_llm(system, user)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM failed: {e}")

    if not sql:
        failure_log = log_analytics_failure(
            {
                "raw_user_question": q,
                "detected_intent": route_intent,
                "planner_confidence": planner_confidence,
                "chosen_sheet": (planner_summary or {}).get("target_sheet") if isinstance(planner_summary, dict) else None,
                "candidate_sheets": planner_candidate_sheets,
                "chosen_metric": planner_selected_metric,
                "candidate_metrics": planner_candidate_metrics,
                "chosen_dimensions": planner_selected_dimensions,
                "extracted_filters": extracted_filters,
                "value_to_column_matches": inferred_value_mappings,
                "generated_sql": "",
                "validation_result": "empty_sql",
                "execution_result": "failed",
                "failure_reason": "empty_sql",
            }
        )
        raise HTTPException(status_code=400, detail="LLM returned empty SQL text")

    sql = _sanitize_unbound_year_placeholder(sql, q)
    apply_global_top_1_limit = _should_apply_global_top_1_limit(force_top_1, query_spec)

    if route_source == "llm" and ALLOW_LLM_SQL_REPAIR:
        sql, schema_issues, schema_changed = _repair_sql_until_schema_valid(
            sql,
            q,
            sheet_columns_by_sheet,
            sheet_schema_context,
            rel_desc,
        )
        if schema_changed:
            route_source = "llm_schema_repaired"
        if schema_issues:
            if allow_protected_fallback:
                pass
            else:
                raise HTTPException(status_code=400, detail=f"LLM SQL still references invalid schema after repair: {' | '.join(schema_issues)}")

    if apply_global_top_1_limit:
        sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", sql, flags=re.IGNORECASE)
        if not re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE):
            sql = f"{sql}\nLIMIT %(limit)s"
    elif requested_top_n:
        sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", sql, flags=re.IGNORECASE)
        if not re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE):
            sql = f"{sql}\nLIMIT %(limit)s"

    analyze_all_applied = False
    if body.analyze_all_rows and not apply_global_top_1_limit and requested_top_n is None and _is_aggregate_like_sql(sql):
        sql = _strip_limit_clause(sql)
        analyze_all_applied = True

    if not _is_safe_select(sql):
        failure_log = log_analytics_failure(
            {
                "raw_user_question": q,
                "detected_intent": route_intent,
                "planner_confidence": planner_confidence,
                "chosen_sheet": (planner_summary or {}).get("target_sheet") if isinstance(planner_summary, dict) else None,
                "candidate_sheets": planner_candidate_sheets,
                "chosen_metric": planner_selected_metric,
                "candidate_metrics": planner_candidate_metrics,
                "chosen_dimensions": planner_selected_dimensions,
                "extracted_filters": extracted_filters,
                "value_to_column_matches": inferred_value_mappings,
                "generated_sql": sql,
                "validation_result": "unsafe_sql",
                "execution_result": "failed",
                "failure_reason": "unsafe_sql",
            }
        )
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL generated: {sql or '<empty>'}")

    if "dataset_id" not in sql.lower():
        failure_log = log_analytics_failure(
            {
                "raw_user_question": q,
                "detected_intent": route_intent,
                "planner_confidence": planner_confidence,
                "chosen_sheet": (planner_summary or {}).get("target_sheet") if isinstance(planner_summary, dict) else None,
                "candidate_sheets": planner_candidate_sheets,
                "chosen_metric": planner_selected_metric,
                "candidate_metrics": planner_candidate_metrics,
                "chosen_dimensions": planner_selected_dimensions,
                "extracted_filters": extracted_filters,
                "value_to_column_matches": inferred_value_mappings,
                "generated_sql": sql,
                "validation_result": "missing_dataset_id_filter",
                "execution_result": "failed",
                "failure_reason": "missing_dataset_id_filter",
            }
        )
        raise HTTPException(status_code=400, detail=f"SQL missing dataset_id filter: {sql}")

    if not analyze_all_applied and not re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE):
        sql = f"{sql}\nLIMIT %(limit)s"

    if apply_global_top_1_limit:
        limit = 1
    elif requested_top_n is not None:
        limit = min(max(int(requested_top_n), 1), max(1, SQL_QUERY_MAX_ROWS))
    else:
        limit = min(max(int(body.max_rows), 1), max(1, SQL_QUERY_MAX_ROWS))

    executed_sql = sql
    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        _prepare_sql_for_psycopg(executed_sql),
                        _build_exec_params(executed_sql, body.dataset_id, limit, q, deterministic_params),
                    )
                    rows = cur.fetchall()
                except Exception as e:
                    err_text = str(e)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                    if err_text and summary_lag_sql and route_source != "explicit_sheet_unanswerable":
                        try:
                            route_intent = "billing_lag_value_gap"
                            route_source = "deterministic_fallback_after_llm"
                            executed_sql = summary_lag_sql
                            cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                            rows = cur.fetchall()
                            err_text = ""
                        except Exception:
                            try:
                                conn.rollback()
                            except Exception:
                                pass

                    if err_text and billing_lag_sql and route_source != "explicit_sheet_unanswerable":
                        try:
                            route_intent = "billing_lag"
                            route_source = "deterministic_fallback_after_llm"
                            executed_sql = billing_lag_sql
                            cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                            rows = cur.fetchall()
                            err_text = ""
                        except Exception:
                            try:
                                conn.rollback()
                            except Exception:
                                pass

                    if err_text and _is_unbilled_total_value_question(q):
                        fallback_sql = _build_unbilled_total_value_fallback_sql(available_sheets, sheet_columns_by_sheet)
                        if fallback_sql and _is_safe_select(fallback_sql) and "dataset_id" in fallback_sql.lower():
                            try:
                                route_intent = "unbilled_total_value"
                                route_source = "deterministic_fallback_after_llm"
                                executed_sql = fallback_sql
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                                err_text = ""
                            except Exception:
                                try:
                                    conn.rollback()
                                except Exception:
                                    pass

                    if err_text and invoiced_amount_sql and route_source != "explicit_sheet_unanswerable":
                        try:
                            route_intent = "invoiced_ops_with_amount"
                            route_source = "deterministic_fallback_after_llm"
                            executed_sql = invoiced_amount_sql
                            cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                            rows = cur.fetchall()
                            err_text = ""
                        except Exception:
                            try:
                                conn.rollback()
                            except Exception:
                                pass

                    if route_source in {"rules_registry_sql_template", "dataset_generated_registry_sql_template"}:
                        try:
                            executed_sql = _generate_sql_with_llm(system, user)
                            if not executed_sql:
                                raise
                            executed_sql = _sanitize_unbound_year_placeholder(executed_sql, q)
                            executed_sql, schema_issues, schema_changed = _repair_sql_until_schema_valid(
                                executed_sql,
                                q,
                                sheet_columns_by_sheet,
                                sheet_schema_context,
                                rel_desc,
                            )
                            if schema_issues:
                                if not allow_protected_fallback:
                                    raise HTTPException(status_code=400, detail=f"LLM SQL after registry fallback still references invalid schema: {' | '.join(schema_issues)}")
                            if apply_global_top_1_limit:
                                executed_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", executed_sql, flags=re.IGNORECASE)
                                if not re.search(r"\bLIMIT\b", executed_sql, flags=re.IGNORECASE):
                                    executed_sql = f"{executed_sql}\nLIMIT %(limit)s"
                            elif requested_top_n:
                                executed_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", executed_sql, flags=re.IGNORECASE)
                                if not re.search(r"\bLIMIT\b", executed_sql, flags=re.IGNORECASE):
                                    executed_sql = f"{executed_sql}\nLIMIT %(limit)s"
                            if analyze_all_applied and _is_aggregate_like_sql(executed_sql):
                                executed_sql = _strip_limit_clause(executed_sql)
                            elif not re.search(r"\bLIMIT\b", executed_sql, flags=re.IGNORECASE):
                                executed_sql = f"{executed_sql}\nLIMIT %(limit)s"
                            if not _is_safe_select(executed_sql) or "dataset_id" not in executed_sql.lower():
                                raise
                            route_intent = "llm_generated"
                            route_source = "llm_schema_repaired_after_registry_failure" if schema_changed else "llm_after_registry_failure"
                            registry_rule_id = None
                            cur.execute(_prepare_sql_for_psycopg(executed_sql), deterministic_params or _build_exec_params(executed_sql, body.dataset_id, limit, q))
                            rows = cur.fetchall()
                        except Exception:
                            err_text = str(e)
                        else:
                            err_text = ""

                    if not err_text:
                        pass
                    elif "invalid input syntax for type numeric" in err_text.lower():
                        rewritten_sql = _sanitize_numeric_casts(sql)
                        if rewritten_sql != sql and _is_safe_select(rewritten_sql):
                            executed_sql = rewritten_sql
                            cur.execute(_prepare_sql_for_psycopg(executed_sql), deterministic_params or _build_exec_params(executed_sql, body.dataset_id, limit, q))
                            rows = cur.fetchall()
                        else:
                            raise
                    elif "date/time field value out of range" in err_text.lower() or "invalid input syntax for type date" in err_text.lower() or "invalid input syntax for type timestamp" in err_text.lower():
                        date_repaired = False
                        if _is_top_seller_year_question(q):
                            seller_col = _pick_col_by_keywords(cols, ["sales_person", "seller", "salesperson", "sales_rep", "rep"])
                            amount_col = _pick_col_by_keywords(cols, ["amount", "revenue", "profit", "total", "sales_amount", "net_sales"])
                            date_col = _pick_col_by_keywords(cols, ["date", "order_date", "invoice_date"])
                            if seller_col and amount_col and date_col:
                                seller_expr = f"s.row_json->>'{seller_col}'"
                                amount_expr = f"s.row_json->>'{amount_col}'"
                                date_expr = f"s.row_json->>'{date_col}'"
                                parsed_date = _date_parse_expr(date_expr)
                                year_expr = f"EXTRACT(YEAR FROM {parsed_date})::int"
                                sheet_filter = ""
                                if len(available_sheets) == 1:
                                    sheet_filter = f" AND s.sheet_name = '{available_sheets[0]}'"
                                executed_sql = (
                                    f"SELECT {seller_expr} AS top_seller, {year_expr} AS year, "
                                    f"SUM(NULLIF(regexp_replace(({amount_expr}), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_sales "
                                    f"FROM dataset_rows s "
                                    f"WHERE s.dataset_id = %(dataset_id)s{sheet_filter} "
                                    f"GROUP BY {seller_expr}, {year_expr} "
                                    f"ORDER BY total_sales DESC "
                                    f"LIMIT %(limit)s"
                                )
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                                date_repaired = True

                        if not date_repaired:
                            candidate_sql = _rewrite_sql_for_date_errors(executed_sql)
                            if candidate_sql == executed_sql:
                                candidate_sql = _repair_sql_after_execution_error(executed_sql, q, err_text)
                                candidate_sql = _rewrite_sql_for_date_errors(candidate_sql)

                            if not candidate_sql:
                                raise

                            if apply_global_top_1_limit or requested_top_n:
                                candidate_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", candidate_sql, flags=re.IGNORECASE)
                                if not re.search(r"\bLIMIT\b", candidate_sql, flags=re.IGNORECASE):
                                    candidate_sql = f"{candidate_sql}\nLIMIT %(limit)s"

                            if analyze_all_applied:
                                candidate_sql = _strip_limit_clause(candidate_sql)
                            elif not re.search(r"\bLIMIT\b", candidate_sql, flags=re.IGNORECASE):
                                candidate_sql = f"{candidate_sql}\nLIMIT %(limit)s"

                            if not _is_safe_select(candidate_sql) or "dataset_id" not in candidate_sql.lower():
                                raise

                            executed_sql = candidate_sql
                            try:
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                            except Exception as second_err:
                                try:
                                    conn.rollback()
                                except Exception:
                                    pass
                                final_sql = _repair_sql_after_execution_error(executed_sql, q, str(second_err))
                                final_sql = _rewrite_sql_for_date_errors(final_sql)
                                if not final_sql:
                                    raise
                                if apply_global_top_1_limit or requested_top_n:
                                    final_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", final_sql, flags=re.IGNORECASE)
                                    if not re.search(r"\bLIMIT\b", final_sql, flags=re.IGNORECASE):
                                        final_sql = f"{final_sql}\nLIMIT %(limit)s"
                                if analyze_all_applied:
                                    final_sql = _strip_limit_clause(final_sql)
                                elif not re.search(r"\bLIMIT\b", final_sql, flags=re.IGNORECASE):
                                    final_sql = f"{final_sql}\nLIMIT %(limit)s"
                                if not _is_safe_select(final_sql) or "dataset_id" not in final_sql.lower():
                                    raise
                                executed_sql = final_sql
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                    elif "must appear in the group by clause" in err_text.lower():
                        fallback_done = False
                        if _is_top_seller_year_question(q):
                            seller_col = _pick_col_by_keywords(cols, ["sales_person", "seller", "salesperson", "sales_rep", "rep"])
                            amount_col = _pick_col_by_keywords(cols, ["amount", "revenue", "profit", "total", "sales_amount", "net_sales"])
                            date_col = _pick_col_by_keywords(cols, ["date", "order_date", "invoice_date"])
                            if seller_col and amount_col and date_col:
                                seller_expr = f"s.row_json->>'{seller_col}'"
                                amount_expr = f"s.row_json->>'{amount_col}'"
                                date_expr = f"s.row_json->>'{date_col}'"
                                parsed_date = _date_parse_expr(date_expr)
                                year_expr = f"EXTRACT(YEAR FROM {parsed_date})::int"
                                sheet_filter = ""
                                if len(available_sheets) == 1:
                                    sheet_filter = f" AND s.sheet_name = '{available_sheets[0]}'"
                                executed_sql = (
                                    f"SELECT {seller_expr} AS top_seller, {year_expr} AS year, "
                                    f"SUM(NULLIF(regexp_replace(({amount_expr}), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_sales "
                                    f"FROM dataset_rows s "
                                    f"WHERE s.dataset_id = %(dataset_id)s{sheet_filter} "
                                    f"GROUP BY {seller_expr}, {year_expr} "
                                    f"ORDER BY total_sales DESC "
                                    f"LIMIT %(limit)s"
                                )
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                                fallback_done = True

                        if not fallback_done:
                            repaired_sql = _repair_group_by_sql_locally(executed_sql)
                            if repaired_sql == executed_sql:
                                repaired_sql = _repair_sql_after_execution_error(executed_sql, q, err_text)
                            if not repaired_sql:
                                raise
                            if apply_global_top_1_limit or requested_top_n:
                                repaired_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", repaired_sql, flags=re.IGNORECASE)
                                if not re.search(r"\bLIMIT\b", repaired_sql, flags=re.IGNORECASE):
                                    repaired_sql = f"{repaired_sql}\nLIMIT %(limit)s"

                            if analyze_all_applied:
                                repaired_sql = _strip_limit_clause(repaired_sql)
                            elif not re.search(r"\bLIMIT\b", repaired_sql, flags=re.IGNORECASE):
                                repaired_sql = f"{repaired_sql}\nLIMIT %(limit)s"

                            if not _is_safe_select(repaired_sql) or "dataset_id" not in repaired_sql.lower():
                                raise

                            executed_sql = repaired_sql
                            try:
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                            except Exception as second_err:
                                # Last-chance repair by LLM if deterministic repair still fails.
                                repaired_sql = _repair_sql_after_execution_error(executed_sql, q, str(second_err))
                                if not repaired_sql:
                                    raise
                                if apply_global_top_1_limit or requested_top_n:
                                    repaired_sql = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT %(limit)s", repaired_sql, flags=re.IGNORECASE)
                                    if not re.search(r"\bLIMIT\b", repaired_sql, flags=re.IGNORECASE):
                                        repaired_sql = f"{repaired_sql}\nLIMIT %(limit)s"
                                if analyze_all_applied:
                                    repaired_sql = _strip_limit_clause(repaired_sql)
                                elif not re.search(r"\bLIMIT\b", repaired_sql, flags=re.IGNORECASE):
                                    repaired_sql = f"{repaired_sql}\nLIMIT %(limit)s"
                                if not _is_safe_select(repaired_sql) or "dataset_id" not in repaired_sql.lower():
                                    raise
                                executed_sql = repaired_sql
                                cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                                rows = cur.fetchall()
                    else:
                        raise

                # Deterministic retry for uninvoiced operational-event intent when LLM SQL is too strict.
                if not rows and _is_uninvoiced_ops_question(q):
                    fallback_sql = _build_uninvoiced_ops_fallback_sql(available_sheets, cols)
                    if fallback_sql and _is_safe_select(fallback_sql) and "dataset_id" in fallback_sql.lower():
                        executed_sql = fallback_sql
                        cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                        rows = cur.fetchall()

                # Deterministic retry for completed ops events with invoice status sent.
                if not rows and _is_invoiced_sent_ops_question(q):
                    fallback_sql = _build_invoiced_sent_ops_fallback_sql(available_sheets, cols)
                    if fallback_sql and _is_safe_select(fallback_sql) and "dataset_id" in fallback_sql.lower():
                        executed_sql = fallback_sql
                        cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                        rows = cur.fetchall()

                if invoiced_amount_sql and (
                    not rows or not _has_invoiced_amount_shape(rows) or not _has_positive_invoiced_amount_rows(rows)
                ):
                    route_intent = "invoiced_ops_with_amount"
                    route_source = "deterministic_fallback_after_llm"
                    executed_sql = invoiced_amount_sql
                    cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                    rows = cur.fetchall()

                if not rows and summary_lag_sql:
                    route_intent = "billing_lag_value_gap"
                    route_source = "deterministic_fallback_after_llm"
                    executed_sql = summary_lag_sql
                    cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                    rows = cur.fetchall()

                if not rows and billing_lag_sql:
                    route_intent = "billing_lag"
                    route_source = "deterministic_fallback_after_llm"
                    executed_sql = billing_lag_sql
                    cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                    rows = cur.fetchall()

                # Minimal deterministic fallback for aggregate unbilled value when LLM returns an empty aggregate.
                if _is_unbilled_total_value_question(q) and (
                    _is_effectively_empty_aggregate_rows(rows)
                    or not _has_unbilled_total_value_shape(rows)
                    or not _sql_uses_value_metric(executed_sql)
                ):
                    fallback_sql = _build_unbilled_total_value_fallback_sql(available_sheets, sheet_columns_by_sheet)
                    if fallback_sql and _is_safe_select(fallback_sql) and "dataset_id" in fallback_sql.lower():
                        route_intent = "unbilled_total_value"
                        route_source = "deterministic_fallback_after_llm"
                        executed_sql = fallback_sql
                        cur.execute(_prepare_sql_for_psycopg(executed_sql), _build_exec_params(executed_sql, body.dataset_id, limit, q))
                        rows = cur.fetchall()

                if route_intent == "billing_lag_value_gap" and summary_lag_source_sheet and summary_lag_threshold is not None:
                    result_note = _build_billing_lag_value_gap_note(
                        cur,
                        body.dataset_id,
                        summary_lag_source_sheet,
                        summary_lag_threshold,
                        summary_lag_gap_condition_text,
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {e}")

    if isinstance(rows, list):
        rows = _drop_zero_only_grouped_rows(rows, q, executed_sql)
        if str(planner_query_shape or "") in {"aggregate_with_filters", "aggregate_scalar"}:
            matched_rows = None
            try:
                with psycopg.connect(DATABASE_URL, row_factory=dict_row) as count_conn:
                    with count_conn.cursor() as count_cur:
                        matched_rows = _count_rows_for_planner_summary(
                            count_cur,
                            body.dataset_id,
                            q,
                            planner_summary,
                            sql_engine_metadata,
                        )
            except Exception:
                matched_rows = None
            rows = _enrich_aggregate_with_filters_rows(rows, planner_summary, matched_rows)

    planner_debug = planner_debug or _build_planner_debug_payload(
        question=q,
        route_intent=route_intent,
        route_source=route_source,
        planner_confidence=planner_confidence,
        planner_summary=planner_summary,
        planner_query_spec=planner_query_spec,
        planner_query_shape=planner_query_shape,
        planner_warnings=planner_warnings,
        planner_reasons=planner_reasons,
        extracted_filters=extracted_filters,
        inferred_value_mappings=inferred_value_mappings,
        candidate_sheets=planner_candidate_sheets,
        selected_metric=planner_selected_metric,
        candidate_metrics=planner_candidate_metrics,
        selected_dimensions=planner_selected_dimensions,
        answer_type=planner_answer_type,
        sql=executed_sql,
        requested_dimension=planner_requested_dimension,
        requested_metric=planner_requested_metric,
        resolved_dimension=planner_resolved_dimension,
        resolved_metric=planner_resolved_metric,
        missing_dimension_reason=planner_missing_dimension_reason,
        missing_metric_reason=planner_missing_metric_reason,
        fallback_blocked_reason=planner_fallback_blocked_reason,
        available_dimensions=planner_available_dimensions,
        available_metrics=planner_available_metrics,
    )

    return {
        "sql": executed_sql,
        "route_intent": route_intent,
        "route_source": route_source,
        "registry_rule_id": registry_rule_id,
        "response_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied_limit": None if analyze_all_applied else limit,
        "analyze_all_requested": body.analyze_all_rows,
        "analyze_all_applied": analyze_all_applied,
        "result_note": result_note,
        "rows_returned": len(rows),
        "degraded_mode": degraded_mode,
        "degraded_reason": planner_fallback_blocked_reason or "",
        "guidance": guidance,
        "query_shape": planner_query_shape,
        "planner_confidence": planner_confidence,
        "planner_warnings": planner_warnings,
        "planner_reasons": planner_reasons,
        "planner_summary": planner_summary,
        "extracted_filters": extracted_filters,
        "inferred_value_mappings": inferred_value_mappings,
        "query_spec": planner_query_spec,
        "planner_debug": planner_debug,
        "failure_log": failure_log,
        "rows": rows,
    }


