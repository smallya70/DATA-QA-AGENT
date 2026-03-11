import io
import importlib
import json
import os
from pathlib import Path
import re
import textwrap
from dataclasses import asdict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException

from app.adaptive_dashboard import build_joined_frame, load_tabular_frames, render_adaptive_dashboard
from app.chart_data_builder import build_chart_dataframe
from app.chart_recommender import recommend_chart_config
from app.chart_semantics import (
    grain_label,
    infer_chart_semantics,
    infer_result_semantics,
    use_grouped_split_bars,
    valid_aggregations_for_y,
)
from app.dashboard_insights import generate_dashboard_insight_summary
from app.dashboard_filters import apply_dashboard_filters, summarize_active_filters
from app.filterable_chart_data import build_filterable_chart_data

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

create_client = None
supabase_import_error = ""
try:
    supabase_module = importlib.import_module("supabase")
    create_client = getattr(supabase_module, "create_client", None)
except Exception as exc:
    supabase_import_error = str(exc)
    create_client = None

if load_dotenv is not None:
    try:
        # Ensure Streamlit resolves the workspace .env regardless of launch directory.
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path)
    except Exception:
        load_dotenv()

try:
    import altair as alt
except Exception:
    alt = None

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8099")
DEFAULT_SQL_MAX_ROWS = int(os.getenv("DEFAULT_SQL_MAX_ROWS", "1000"))
UI_SQL_MAX_ROWS_CAP = int(os.getenv("UI_SQL_MAX_ROWS_CAP", "5000"))
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

sb = None
supabase_init_status = "ok"
if create_client is None:
    supabase_init_status = (
        "Supabase SDK not importable. Install package `supabase` in the active environment."
    )
elif not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    supabase_init_status = (
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in process environment."
    )
else:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as exc:
        supabase_init_status = f"Supabase client init failed: {exc}"
        sb = None

if create_client is None and supabase_import_error:
    supabase_init_status = f"{supabase_init_status} (import error: {supabase_import_error})"


def _intent_slug(value: str) -> str:
    txt = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "custom_rule"


def _upload_to_backend(
    filename: str,
    file_bytes: bytes,
    dataset_name: str,
    include_assumptions_sheet: bool = True,
) -> dict:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {
        "dataset_name": dataset_name or "",
        "include_assumptions_sheet": str(bool(include_assumptions_sheet)).lower(),
    }

    response = requests.post(
        f"{BACKEND}/upload_csv",
        files=files,
        data=data,
        timeout=180,
    )

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            raise RuntimeError(response.text)

        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict):
            message = str(detail.get("message") or "Upload failed").strip() or "Upload failed"
            upload_qa = detail.get("upload_qa") or {}
            failures = upload_qa.get("failures") or []
            failure_text = "; ".join(
                [
                    f"{str(item.get('scope') or 'registry')}: {str(item.get('failure') or 'unknown_failure')}"
                    for item in failures[:3]
                    if isinstance(item, dict)
                ]
            )
            if failure_text:
                raise RuntimeError(f"{message} {failure_text}")
            raise RuntimeError(message)

        raise RuntimeError(str(body))

    return response.json()


def _fetch_datasets() -> list[dict]:
    response = requests.get(f"{BACKEND}/datasets", timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(response.text)
    data = response.json()
    return data if isinstance(data, list) else []


def _fetch_dataset_frames(dataset_id: str) -> dict:
    response = requests.get(f"{BACKEND}/datasets/{dataset_id}/frames", timeout=180)
    try:
        body = response.json()
    except Exception:
        body = {"error": response.text}
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


def _ask_sql(
    dataset_id: str,
    question: str,
    max_rows: int = DEFAULT_SQL_MAX_ROWS,
    analyze_all_rows: bool = False,
) -> dict:
    payload = {
        "dataset_id": dataset_id,
        "question": question,
        "max_rows": max_rows,
        "analyze_all_rows": analyze_all_rows,
    }
    response = requests.post(f"{BACKEND}/ask_sql", json=payload, timeout=120)
    try:
        body = response.json()
    except Exception:
        body = {"error": response.text}
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body


def _fetch_rules_registry() -> list[dict]:
    response = requests.get(f"{BACKEND}/rules_registry", timeout=60)
    try:
        body = response.json()
    except Exception:
        body = response.text
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body if isinstance(body, list) else []


def _create_rule_registry(payload: dict) -> dict:
    response = requests.post(f"{BACKEND}/rules_registry", json=payload, timeout=60)
    try:
        body = response.json()
    except Exception:
        body = response.text
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


def _update_rule_registry(rule_id: str, payload: dict) -> dict:
    response = requests.put(f"{BACKEND}/rules_registry/{rule_id}", json=payload, timeout=60)
    try:
        body = response.json()
    except Exception:
        body = response.text
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


def _delete_rule_registry(rule_id: str) -> dict:
    response = requests.delete(f"{BACKEND}/rules_registry/{rule_id}", timeout=60)
    try:
        body = response.json()
    except Exception:
        body = response.text
    if response.status_code >= 400:
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


def _run_and_store_sql_question(
    dataset_id: str,
    question: str,
    *,
    max_rows: int = DEFAULT_SQL_MAX_ROWS,
    analyze_all_rows: bool = False,
) -> bool:
    try:
        # Clear stale result first so failed runs do not keep showing outdated output.
        st.session_state.get("sql_results_by_dataset", {}).pop(dataset_id, None)
        with st.spinner("Thinking... analyzing your data"):
            out = _ask_sql(
                dataset_id=dataset_id,
                question=question,
                max_rows=max_rows,
                analyze_all_rows=analyze_all_rows,
            )
        st.session_state["sql_results_by_dataset"][dataset_id] = {
            "question": question,
            "result": out,
        }
        st.session_state["ask_sql_question"] = question
        return True
    except Exception as exc:
        st.error(f"SQL analysis failed: {exc}")
        return False


def _is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


@st.cache_data(show_spinner=False)
def _load_local_dashboard_frames(file_name: str, file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return load_tabular_frames(file_name, file_bytes)


@st.cache_data(show_spinner=False)
def _load_uploaded_dataset_frames(dataset_id: str) -> dict[str, pd.DataFrame]:
    payload = _fetch_dataset_frames(dataset_id)
    frames_payload = payload.get("frames") or {}
    frames: dict[str, pd.DataFrame] = {}
    for sheet_name, rows in frames_payload.items():
        if not isinstance(rows, list):
            continue
        frames[str(sheet_name)] = pd.DataFrame(rows)
    return frames


def _render_instant_file_dashboard(file_name: str, file_bytes: bytes, scope_key: str) -> None:
    try:
        frames = _load_local_dashboard_frames(file_name, file_bytes)
    except Exception as exc:
        st.warning(f"Instant dashboard preview unavailable: {exc}")
        return
    render_adaptive_dashboard(frames, source_name=file_name, scope_key=scope_key)


def _render_uploaded_dataset_dashboard(dataset_id: str, dataset_name: str, scope_key: str) -> None:
    try:
        frames = _load_uploaded_dataset_frames(dataset_id)
    except Exception as exc:
        st.warning(f"Selected dataset dashboard unavailable: {exc}")
        return
    if not frames:
        st.info("No stored rows were found for the selected dataset.")
        return
    render_adaptive_dashboard(frames, source_name=dataset_name or dataset_id, scope_key=scope_key)


def _render_joined_uploaded_dataset_dashboard(
    dataset_id: str,
    dataset_name: str,
    relationship: dict,
    join_how: str,
    scope_key: str,
) -> None:
    try:
        frames = _load_uploaded_dataset_frames(dataset_id)
        joined_df = build_joined_frame(frames, relationship, join_how=join_how)
    except Exception as exc:
        st.warning(f"Joined dataset dashboard unavailable: {exc}")
        return
    if joined_df.empty:
        st.info("The selected join produced no matching rows.")
        return
    join_label = f"{relationship.get('left')} + {relationship.get('right')}"
    render_adaptive_dashboard({join_label: joined_df}, source_name=dataset_name or dataset_id, scope_key=scope_key)


def _show_relationships(payload: dict) -> None:
    rels = (payload.get("relationships") or {}).get("relationships", [])
    if not rels:
        return
    st.subheader("Detected joins")
    for rel in rels[:10]:
        st.write(
            f"{rel['left']}.{rel['left_col']} <-> {rel['right']}.{rel['right_col']} | score={rel['score']}"
        )


def _show_cleaning_report(payload: dict) -> None:
    report = payload.get("cleaning_report") or (payload.get("relationships") or {}).get("cleaning_report") or {}
    if not isinstance(report, dict) or not report:
        return

    totals = report.get("totals") or {}
    excluded_sheets = report.get("excluded_sheets") or []
    sheet_reports = report.get("sheet_reports") or []

    st.subheader("Cleaning summary")
    st.write(
        "Rows/values normalized: "
        f"nulls={int(totals.get('nulls_normalized') or 0)}, "
        f"numeric={int(totals.get('numeric_cells_normalized') or 0)}, "
        f"dates={int(totals.get('date_cells_normalized') or 0)}, "
        f"renamed_columns={int(totals.get('renamed_column_count') or 0)}"
    )
    if excluded_sheets:
        st.write("Excluded sheets:", ", ".join([str(sheet) for sheet in excluded_sheets]))

    if sheet_reports:
        with st.expander("Cleaning details", expanded=False):
            for item in sheet_reports:
                st.write(
                    f"{item.get('sheet')}: rows {item.get('input_row_count')} -> {item.get('output_row_count')}, "
                    f"cols {item.get('input_column_count')} -> {item.get('output_column_count')}, "
                    f"renamed={len(item.get('renamed_columns') or {})}, "
                    f"empty_rows_dropped={int(item.get('dropped_empty_rows') or 0)}, "
                    f"empty_cols_dropped={len(item.get('dropped_empty_columns') or [])}"
                )


def _show_upload_qa(payload: dict) -> None:
    generated_count = int(payload.get("generated_registry_rule_count") or 0)
    generated_questions = payload.get("generated_registry_questions") or []
    qa = payload.get("upload_qa") or {}
    if generated_count > 0:
        st.write(f"Generated registry rules: {generated_count}")
    if generated_questions:
        with st.expander("Generated registry questions", expanded=False):
            for question in generated_questions:
                st.write(f"- {question}")

    if not isinstance(qa, dict) or not qa:
        return

    status = str(qa.get("status") or "").strip().lower()
    checked = int(qa.get("checked_rule_count") or 0)
    failures = qa.get("failures") or []
    checks = qa.get("checks") or []

    if status == "disabled":
        st.caption("Upload QA hook is disabled.")
        return
    if status == "not_run":
        return
    if status == "no_checks":
        st.info("Upload QA found no relevant registry checks for this dataset.")
        return
    if status == "no_generated_rules":
        st.info("Upload QA found no generated registry rules to validate.")
        return

    if qa.get("passed", False):
        curated_checked = int(((qa.get("curated_registry") or {}).get("checked_rule_count") or 0))
        st.success(
            f"Upload QA passed: validated {checked} registry check(s) "
            f"({generated_count} generated rule(s), {curated_checked} curated route(s))."
        )
    elif status == "error":
        st.error("Upload QA failed to run.")
    else:
        st.error(f"Upload QA failed: {len(failures)} issue(s) found across {checked} registry check(s).")

    with st.expander("Upload QA details", expanded=not qa.get("passed", False)):
        if checks:
            for check in checks:
                question = str(check.get("question") or "").strip()
                intent = str(check.get("intent") or "").strip()
                scope = str(check.get("scope") or "registry").strip()
                if check.get("status") == "passed":
                    st.write(f"PASS | {scope} | {intent} | {question}")
                else:
                    failure = str(check.get("failure") or "unknown_failure")
                    st.write(f"FAIL | {scope} | {intent} | {question} | {failure}")
        elif failures:
            for failure in failures:
                st.write(str(failure))


def _describe_column(col_name: str, role: str) -> str:
    c = (col_name or "").lower()
    if "year" in c:
        return f"{role}: {col_name} (time period in years)."
    if "date" in c:
        return f"{role}: {col_name} (calendar date or date-derived value)."
    if "amount" in c or "revenue" in c or "sales" in c or "profit" in c or "total" in c:
        return f"{role}: {col_name} (numeric business metric used for comparison)."
    return f"{role}: {col_name}."


def _extract_top_n_from_question(question: str) -> int | None:
    q = (question or "").lower()
    m = re.search(r"\b(?:top+|bottom)\s*(\d+)\b", q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b(?:top+|bottom)(\d+)\b", q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _set_question_year(question: str, year: int) -> str:
    q = (question or "").strip()
    if not q:
        return f"for year {year}"
    if re.search(r"\b(19|20)\d{2}\b", q):
        return re.sub(r"\b(19|20)\d{2}\b", str(year), q, count=1)
    return f"{q} for year {year}"


def _extract_year_from_text(question: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", question or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _download_figure(fig: Figure, label: str, file_name: str, key: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=file_name,
        mime="image/png",
        key=key,
    )


def _compact_label(value: object, max_len: int = 24) -> str:
    label = str(value)
    return label if len(label) <= max_len else f"{label[:max_len - 1]}..."


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        # Dict/list values (for example row_json cells) are unhashable for nunique.
        coerced = series.map(
            lambda v: json.dumps(v, sort_keys=True)
            if isinstance(v, (dict, list, tuple, set))
            else v
        )
        return int(coerced.nunique(dropna=True))


def _numeric_range_filter_widget(column_name: str, valid_series: pd.Series, scope_key: str) -> tuple[float, float] | None:
    cleaned = pd.to_numeric(valid_series, errors="coerce").dropna()
    if cleaned.empty:
        return None

    min_value = float(cleaned.min())
    max_value = float(cleaned.max())
    if not (pd.notna(min_value) and pd.notna(max_value)):
        return None

    if min_value == max_value:
        st.caption(f"{column_name} is constant at {min_value:,.2f}; range filter is not needed.")
        return (min_value, max_value)

    key = f"{scope_key}_chart_range_filter_{column_name}"
    span = max_value - min_value
    step = max(span / 100.0, 0.01)
    if span >= 100:
        step = max(span / 100.0, 1.0)

    try:
        selected_range = st.slider(
            f"Range {column_name}",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            step=float(step),
            key=key,
        )
        return (float(selected_range[0]), float(selected_range[1]))
    except StreamlitAPIException:
        st.caption(f"Using manual range inputs for {column_name} because slider rendering is unsupported for this value range.")
        col1, col2 = st.columns(2)
        with col1:
            lower = st.number_input(
                f"Min {column_name}",
                min_value=min_value,
                max_value=max_value,
                value=min_value,
                step=float(step),
                key=f"{key}_min",
            )
        with col2:
            upper = st.number_input(
                f"Max {column_name}",
                min_value=min_value,
                max_value=max_value,
                value=max_value,
                step=float(step),
                key=f"{key}_max",
            )
        return (float(min(lower, upper)), float(max(lower, upper)))


def _looks_like_year_series(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return False
    text = series.astype(str).str.strip()
    year_values = text.str.extract(r"^((?:19|20)\d{2})$", expand=False)
    non_empty = text[text != ""]
    if non_empty.empty:
        return False
    match_ratio = float(year_values.notna().sum()) / float(len(non_empty))
    return match_ratio >= 0.8


def _is_year_axis_column(df: pd.DataFrame, column_name: str) -> bool:
    if column_name not in df.columns:
        return False
    if "year" not in str(column_name).lower():
        return False
    return _looks_like_year_series(df[column_name])


def _is_identifier_col_name(col_name: str) -> bool:
    c = str(col_name or "").lower()
    return (
        c == "row_index"
        or c.endswith("_id")
        or c in {"id", "event_id", "invoice_id", "order_id", "transaction_id"}
        or "_id_" in c
    )


def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.85) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for c in list(out.columns):
        c_name = str(c).lower()
        # Prevent customer/customer_id and event_id columns from being coerced to numeric
        if c_name in ["row_index", "customer", "customer_id", "cust_id", "client_id", "event_id"]:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue

        raw = out[c].astype(str).str.strip()
        non_empty_mask = (
            (raw != "")
            & (~raw.str.lower().isin({"nan", "none", "null", "na", "n/a"}))
        )
        non_empty = raw[non_empty_mask]
        if non_empty.empty:
            continue

        cleaned_non_empty = non_empty.str.replace(r"[^0-9.\-]", "", regex=True)
        parsed_non_empty = pd.to_numeric(cleaned_non_empty, errors="coerce")
        numeric_ratio = float(parsed_non_empty.notna().mean()) if len(parsed_non_empty) else 0.0
        if numeric_ratio < min_ratio:
            continue

        cleaned_all = raw.str.replace(r"[^0-9.\-]", "", regex=True)
        out[c] = pd.to_numeric(cleaned_all, errors="coerce")

    return out


def _to_display_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "row_json" not in df.columns:
        return df

    has_dict_payload = df["row_json"].map(lambda v: isinstance(v, dict)).any()
    if not has_dict_payload:
        return df

    base_cols = [c for c in ["sheet_name", "row_index"] if c in df.columns]
    passthrough_cols = [c for c in df.columns if c not in base_cols + ["row_json"]]
    row_payload = df["row_json"].map(lambda v: v if isinstance(v, dict) else {})
    expanded = pd.json_normalize(row_payload.tolist())

    merged = pd.concat(
        [df[base_cols + passthrough_cols].reset_index(drop=True), expanded.reset_index(drop=True)],
        axis=1,
    )
    return merged


def _default_visible_columns(df: pd.DataFrame, hide_sheet_name: bool = False) -> list[str]:
    if df.empty:
        return list(df.columns)

    preferred_first = [
        "sheet_name",
        "row_index",
        "event_id",
        "customer_id",
        "contract_id",
        "event_type",
        "service_type",
        "service_date",
        "status",
        "invoice_status",
        "invoice_id",
        "invoice_date",
        "order_id",
        "transaction_date",
        "quantity",
        "price_per_unit",
        "billing_amount",
        "value_gap_amount",
        "amount",
        "revenue",
        "profit",
        "tax",
        "discount",
        "line_value",
    ]

    # Keep critical financial output columns visible even when sparse/null-heavy.
    always_show = {
        "event_id",
        "customer_id",
        "event_type",
        "service_type",
        "service_date",
        "quantity",
        "price_per_unit",
        "billing_amount",
        "value_gap_amount",
    }

    cols = list(df.columns)
    non_null_ratio = {
        c: float(df[c].notna().mean()) if len(df) else 0.0
        for c in cols
    }

    default_cols = []
    for c in cols:
        if hide_sheet_name and c == "sheet_name":
            continue
        if c in always_show:
            default_cols.append(c)
            continue
        # Hide mostly-empty columns by default; user can still enable them.
        if non_null_ratio.get(c, 0.0) < 0.10:
            continue
        default_cols.append(c)

    if not default_cols:
        default_cols = [c for c in cols if not (hide_sheet_name and c == "sheet_name")]

    # Keep useful business columns toward the front.
    preferred_rank = {name: i for i, name in enumerate(preferred_first)}
    default_cols.sort(key=lambda c: (preferred_rank.get(c, 10_000), cols.index(c)))
    return default_cols


def _render_table_with_column_controls(
    df: pd.DataFrame,
    scope_key: str,
    *,
    hide_sheet_name_default: bool = False,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    all_cols = list(df.columns)
    default_cols = _default_visible_columns(df, hide_sheet_name=hide_sheet_name_default)

    with st.expander("Visible columns", expanded=False):
        selected_cols = st.multiselect(
            "Choose columns to display",
            options=all_cols,
            default=default_cols,
            key=f"{scope_key}_visible_cols",
        )

    # Streamlit preserves multiselect state per key across reruns. If users selected
    # columns before new backend fields existed, important financial columns can stay
    # hidden. Ensure they are included whenever present in result rows.
    required_if_present = [
        "quantity",
        "price_per_unit",
        "billing_amount",
        "value_gap_amount",
    ]
    required_if_present.extend([str(col) for col in (required_columns or []) if str(col).strip()])
    for col in required_if_present:
        if col in all_cols and col not in selected_cols:
            selected_cols.append(col)

    if selected_cols:
        view_df = df[selected_cols].copy()
    else:
        view_df = df.copy()

    st.dataframe(view_df, use_container_width=True, hide_index=True)
    return view_df


def _prepare_result_dataframe_for_display(df: pd.DataFrame, route_intent: str) -> pd.DataFrame:
    if df.empty:
        return df

    if route_intent == "semantic_role_sheet_summary":
        rename_map = {
            "row_count": "sheet_row_count",
            "role_count": "semantic_role_count",
            "detected_roles": "semantic_roles",
        }
        applicable = {old: new for old, new in rename_map.items() if old in df.columns}
        if applicable:
            return df.rename(columns=applicable)

    return df


def _render_guidance_panel(result: dict, dataset_id: str) -> None:
    guidance = result.get("guidance") if isinstance(result, dict) else None
    degraded_mode = bool(result.get("degraded_mode")) if isinstance(result, dict) else False
    degraded_reason = str(result.get("degraded_reason") or "") if isinstance(result, dict) else ""

    if not degraded_mode and not guidance:
        return

    if degraded_mode:
        reason_text = f" Reason: {degraded_reason}." if degraded_reason else ""
        st.warning(
            "Query was handled in safe preview mode because the dataset schema did not fully match the question."
            f"{reason_text}"
        )
    else:
        st.info("Schema guidance is available for this result.")

    if not isinstance(guidance, dict):
        return

    msg = str(guidance.get("message") or "").strip()
    if msg:
        st.caption(msg)

    suggested_questions = guidance.get("suggested_questions") or []
    if suggested_questions:
        st.markdown("**Suggested follow-up questions**")
        for idx, suggestion in enumerate(suggested_questions[:6]):
            suggestion_text = str(suggestion).strip()
            if not suggestion_text:
                continue
            if st.button(
                suggestion_text,
                key=f"guidance_q_{dataset_id}_{idx}",
                use_container_width=True,
            ):
                ok = _run_and_store_sql_question(
                    dataset_id=dataset_id,
                    question=suggestion_text,
                    max_rows=DEFAULT_SQL_MAX_ROWS,
                    analyze_all_rows=False,
                )
                if ok:
                    st.rerun()

    sheets = guidance.get("sheets") or []
    if sheets:
        with st.expander("Schema preview used for guidance", expanded=False):
            for sheet in sheets[:8]:
                if not isinstance(sheet, dict):
                    continue
                sheet_name = str(sheet.get("sheet_name") or "")
                cols = sheet.get("sample_columns") or []
                st.markdown(f"`{sheet_name}`")
                if cols:
                    st.caption(", ".join([str(c) for c in cols[:20]]))
                else:
                    st.caption("No sampled columns available.")


def _render_planner_summary(result: dict) -> None:
    if not isinstance(result, dict):
        return

    planner_summary = result.get("planner_summary")
    planner_debug_raw = result.get("planner_debug")
    planner_debug: dict = planner_debug_raw if isinstance(planner_debug_raw, dict) else {}
    failure_log_raw = result.get("failure_log")
    failure_log: dict = failure_log_raw if isinstance(failure_log_raw, dict) else {}
    query_shape = str(result.get("query_shape") or "").strip()
    planner_confidence = result.get("planner_confidence")
    planner_warnings = [str(item) for item in list(result.get("planner_warnings") or []) if str(item).strip()]
    planner_reasons = [str(item) for item in list(result.get("planner_reasons") or []) if str(item).strip()]
    extracted_filters = [item for item in list(result.get("extracted_filters") or []) if isinstance(item, dict)]
    inferred_value_mappings = [item for item in list(result.get("inferred_value_mappings") or []) if isinstance(item, dict)]
    query_spec = result.get("query_spec")
    if not isinstance(planner_summary, dict) and not query_shape and planner_confidence in (None, ""):
        return

    effective_shape = query_shape or str((planner_summary or {}).get("query_shape") or "").strip()
    target_sheet = str((planner_summary or {}).get("target_sheet") or "").strip()
    dimensions = [str(item) for item in list((planner_summary or {}).get("dimensions") or []) if str(item).strip()]
    measures = [item for item in list((planner_summary or {}).get("measures") or []) if isinstance(item, dict)]
    filters = [item for item in list((planner_summary or {}).get("filters") or []) if isinstance(item, dict)]
    warnings = [str(item) for item in list((planner_summary or {}).get("warnings") or []) if str(item).strip()]

    st.caption(f"Interpretation: {effective_shape or 'unknown'}")
    if planner_confidence not in (None, ""):
        try:
            st.caption(f"Planner confidence: {float(planner_confidence):.2f}")
        except Exception:
            st.caption(f"Planner confidence: {planner_confidence}")
    if target_sheet:
        st.caption(f"Target sheet: {target_sheet}")
    if planner_debug.get("answer_type"):
        st.caption(f"Answer type: {planner_debug.get('answer_type')}")
    if dimensions:
        st.caption("Dimensions: " + ", ".join(dimensions))
    if measures:
        measure_labels = []
        for measure in measures:
            aggregation = str(measure.get("aggregation") or "").strip()
            column = str(measure.get("column") or "").strip()
            alias = str(measure.get("alias") or "").strip()
            label = f"{aggregation}({column})" if aggregation and column and column != "*" else (alias or column or aggregation)
            if alias and alias not in label:
                label = f"{label} as {alias}"
            measure_labels.append(label)
        if measure_labels:
            st.caption("Measures: " + ", ".join(measure_labels))
    if filters:
        filter_labels = []
        for filter_spec in filters:
            column = str(filter_spec.get("column") or "").strip()
            operator = str(filter_spec.get("operator") or filter_spec.get("op") or "=").strip()
            value = filter_spec.get("value")
            if column:
                filter_labels.append(f"{column} {operator} {value}")
        if filter_labels:
            st.caption("Filters: " + "; ".join(filter_labels))
    for warning in list(dict.fromkeys(warnings + planner_warnings)):
        st.caption(f"Planner warning: {warning}")
    if failure_log.get("path"):
        st.caption(f"Failure log: {failure_log.get('path')}")

    show_debug = bool(st.session_state.get("show_planner_debug"))
    if show_debug and (planner_reasons or extracted_filters or inferred_value_mappings or isinstance(query_spec, dict) or planner_debug):
        with st.expander("Planner debug", expanded=False):
            if planner_debug:
                st.markdown("**Planner debug payload**")
                st.json(planner_debug)
            if planner_reasons:
                st.markdown("**Reasons**")
                for reason in planner_reasons:
                    st.write(f"- {reason}")
            if extracted_filters:
                st.markdown("**Extracted filters**")
                st.json(extracted_filters)
            if inferred_value_mappings:
                st.markdown("**Inferred value mappings**")
                st.json(inferred_value_mappings)
            if isinstance(query_spec, dict):
                st.markdown("**QuerySpec**")
                st.json(query_spec)


def _render_chart_semantic_summary(
    *,
    applied_filter_summary: str,
    result_grain_label: str,
    chart_grain_label: str,
    chart_aggregation_label: str,
    raw_rows: int,
    chart_rows: int,
) -> None:
    with st.container(border=True):
        st.markdown("**Chart semantics**")
        st.write(f"Applied filters: {applied_filter_summary}")
        st.write(f"Base result grain: {result_grain_label}")
        st.write(f"Chart grain: {chart_grain_label}")
        st.write(f"Chart aggregation: {chart_aggregation_label}")
        st.write(f"Rows in chart input result: {raw_rows}")
        st.write(f"Rows in chart data: {chart_rows}")


def _clear_result_filter_state(scope_key: str, dimension_columns: list[str], measure_columns: list[str]) -> None:
    keys_to_clear = [
        f"{scope_key}_chart_filters_reset",
    ]
    for column in dimension_columns:
        keys_to_clear.append(f"{scope_key}_chart_filter_{column}")
    for column in measure_columns:
        keys_to_clear.append(f"{scope_key}_chart_range_filter_{column}")
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _singleton_categorical_filters(categorical_filters: dict[str, list[str]]) -> dict[str, str]:
    singleton_filters: dict[str, str] = {}
    for column_name, selected_values in categorical_filters.items():
        normalized_values = [str(value) for value in list(selected_values or []) if str(value).strip()]
        if len(normalized_values) == 1:
            singleton_filters[column_name] = normalized_values[0]
    return singleton_filters


def _extract_result_filter_specs(result_payload: dict | None) -> list[dict]:
    if not isinstance(result_payload, dict):
        return []
    collected: list[dict] = []
    for source in [
        result_payload.get("query_spec"),
        result_payload.get("planner_summary"),
        {"filters": result_payload.get("extracted_filters")},
    ]:
        if not isinstance(source, dict):
            continue
        for filter_spec in list(source.get("filters") or []):
            if isinstance(filter_spec, dict):
                collected.append(filter_spec)
    return collected


def _result_singleton_filters(result_payload: dict | None) -> dict[str, str]:
    singleton_filters: dict[str, str] = {}
    for filter_spec in _extract_result_filter_specs(result_payload):
        column_name = str(filter_spec.get("column") or "").strip()
        operator = str(filter_spec.get("operator") or filter_spec.get("op") or "=").strip().lower()
        value = filter_spec.get("value")
        if not column_name:
            continue
        if operator in {"=", "=="} and value not in (None, ""):
            singleton_filters[column_name] = str(value)
            continue
        if operator == "in" and isinstance(value, list):
            normalized_values = [str(item) for item in value if str(item).strip()]
            if len(normalized_values) == 1:
                singleton_filters[column_name] = normalized_values[0]
    return singleton_filters


def _result_filter_labels(result_payload: dict | None) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for filter_spec in _extract_result_filter_specs(result_payload):
        column_name = str(filter_spec.get("column") or "").strip()
        operator = str(filter_spec.get("operator") or filter_spec.get("op") or "=").strip().lower()
        value = filter_spec.get("value")
        if not column_name:
            continue
        if operator in {"=", "=="} and value not in (None, ""):
            label = f"{column_name} = {value}"
        elif operator == "in" and isinstance(value, list):
            normalized_values = [str(item) for item in value if str(item).strip()]
            if not normalized_values:
                continue
            label = f"{column_name} = {normalized_values[0]}" if len(normalized_values) == 1 else f"{column_name} in [{', '.join(normalized_values)}]"
        elif operator in {">", ">=", "<", "<="} and value not in (None, ""):
            label = f"{column_name} {operator} {value}"
        else:
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _build_chart_preview_dataframe(
    chart_df: pd.DataFrame,
    preview_columns: list[str],
    singleton_filtered_dimensions: dict[str, str],
    *,
    retain_singleton_dimensions: bool,
) -> pd.DataFrame:
    preview_df = chart_df[preview_columns].copy()
    if not retain_singleton_dimensions:
        return preview_df

    additional_columns = [
        column_name for column_name in singleton_filtered_dimensions if column_name not in preview_df.columns
    ]
    for column_name in additional_columns:
        preview_df[column_name] = singleton_filtered_dimensions[column_name]

    if additional_columns and preview_columns:
        return preview_df[preview_columns[:-1] + additional_columns + preview_columns[-1:]]
    return preview_df


def _append_context_columns(df: pd.DataFrame, context_values: dict[str, str]) -> pd.DataFrame:
    context_df = df.copy()
    for column_name, value in context_values.items():
        if column_name not in context_df.columns:
            context_df[column_name] = value
    return context_df


def _describe_hidden_singleton_filters(hidden_singleton_filters: dict[str, str], chart_grain_label: str) -> str:
    hidden_columns = list(hidden_singleton_filters.keys())
    if not hidden_columns:
        return ""
    if len(hidden_columns) == 1:
        return (
            f"{hidden_columns[0]} = {hidden_singleton_filters[hidden_columns[0]]} is acting as a filter, not a display dimension, "
            f"because chart grain is {chart_grain_label}."
        )
    formatted_filters = ", ".join(
        f"{column_name} = {value}" for column_name, value in hidden_singleton_filters.items()
    )
    return (
        f"{formatted_filters} are acting as filters, not display dimensions, "
        f"because chart grain is {chart_grain_label}."
    )


def _singleton_filter_context_caption(hidden_singleton_filters: dict[str, str]) -> str:
    if not hidden_singleton_filters:
        return ""
    formatted_filters = ", ".join(
        f"{column_name} = {value}" for column_name, value in hidden_singleton_filters.items()
    )
    return f"Chart context: {formatted_filters}"


def _suggest_questions_for_dataset(dataset_meta: dict) -> list[str]:
    cols = [str(c) for c in (dataset_meta.get("columns") or [])]
    col_norm = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in cols]
    relationships = dataset_meta.get("relationships") or {}
    rel_edges = (relationships.get("relationships") or []) if isinstance(relationships, dict) else []
    semantic_roles = (relationships.get("semantic_roles") or {}) if isinstance(relationships, dict) else {}
    sheet_names = (relationships.get("tables") or []) if isinstance(relationships, dict) else []

    def has_any(tokens: list[str]) -> bool:
        return any(any(tok in c for tok in tokens) for c in col_norm)

    def has_role(role: str) -> bool:
        for _sheet, roles in semantic_roles.items():
            if role in (roles or {}):
                return True
        return False

    def role_sheet(role: str) -> str | None:
        for sheet, roles in semantic_roles.items():
            if role in (roles or {}):
                return str(sheet)
        return None

    def sheet_has_role(sheet: str, role: str) -> bool:
        roles = semantic_roles.get(sheet) or {}
        return role in roles

    def sheet_with_roles(required_roles: list[str]) -> str | None:
        for sheet, roles in semantic_roles.items():
            role_map = roles or {}
            if all(role in role_map for role in required_roles):
                return str(sheet)
        return None

    billing_or_exception_focused = any(
        has_role(role)
        for role in ["exception_type", "value_gap", "billing_lag_days", "invoice_status", "expected_amount", "billed_amount"]
    )

    suggestions: list[str] = []

    if sheet_names:
        top_sheets = ", ".join([str(s) for s in sheet_names[:4]])
        suggestions.append(f"Show first 20 rows from sheets {top_sheets}.")
    else:
        suggestions.append("Can you show a data frame of the sheets uploaded here?")

    if rel_edges:
        for rel in rel_edges[:2]:
            left = rel.get("left")
            left_col = rel.get("left_col")
            right = rel.get("right")
            right_col = rel.get("right_col")
            if left and left_col and right and right_col:
                suggestions.append(
                    f"Join {left} and {right} on {left_col} = {right_col} and summarize key metrics by customer."
                )

    billing_gap_sheet = sheet_with_roles(["event_id", "billing_lag_days", "value_gap"])
    if billing_gap_sheet:
        suggestions.append(
            f"List event_id and customer_id from {billing_gap_sheet} where billing_lag_days > 7 with value gap amount."
        )

    summary_sheet = sheet_with_roles(["exception_type", "total_value_gap"])
    if summary_sheet:
        suggestions.append(f"Show exception summary by exception_type from {summary_sheet}.")

    ops_sheet = sheet_with_roles(["event_id"])
    if ops_sheet and (has_role("invoice_status") or has_any(["invoice", "billing"])):
        suggestions.append(
            f"List completed events from {ops_sheet} that are not yet invoiced with value gap amount."
        )
        suggestions.append(
            f"What is the total unbilled value for completed events in {ops_sheet}?"
        )

    sales_sheet = None if billing_or_exception_focused else (role_sheet("revenue") or role_sheet("profit"))
    if sales_sheet and has_role("revenue"):
        suggestions.append(f"Show revenue trend by year from {sales_sheet}.")
        suggestions.append(f"Who are the top 10 customers by revenue in {sales_sheet}?")
    if sales_sheet and has_role("tax"):
        suggestions.append(f"Show revenue, tax, and discount by month from {sales_sheet}.")
    if sales_sheet and has_role("profit") and has_any(["state", "region", "country", "city"]):
        suggestions.append(f"Which region has the highest profit in {sales_sheet}?")

    if has_any(["date", "year", "month"]) and not has_role("revenue") and not billing_or_exception_focused:
        suggestions.append("Show trend by year for key numeric metrics.")

    if has_any(["product", "sku", "item", "service_type"]) and not billing_or_exception_focused:
        suggestions.append("Which products or services contribute most to revenue?")

    if semantic_roles:
        suggestions.append("Show a sheet-wise summary using detected semantic roles.")

    # Preserve order while removing duplicates.
    seen = set()
    uniq: list[str] = []
    for q in suggestions:
        key = q.strip().lower()
        if key and key not in seen:
            uniq.append(q)
            seen.add(key)
    return uniq[:10]


def _render_dashboard(
    df: pd.DataFrame,
    scope_key: str = "default",
    *,
    result_payload: dict | None = None,
) -> None:
    st.subheader("Dashboard")

    raw_result_df = df.copy()
    # Convert numeric-looking text values (for example "$12,345") into numeric dtype
    # so dashboard axis selection and metric detection are more intuitive.
    raw_result_df = _coerce_numeric_like_columns(raw_result_df)
    original_rows = int(raw_result_df.shape[0])
    result_semantics = infer_result_semantics(raw_result_df, result_payload)
    recommendation = recommend_chart_config(result_semantics)
    projection_query = result_semantics.query_type == "projection"

    if recommendation.default_chart_type == "KPI Card" and recommendation.y_column and not projection_query:
        value_series = pd.to_numeric(raw_result_df[recommendation.y_column], errors="coerce")
        if value_series.notna().sum() > 0:
            kpi_value = float(value_series.iloc[0]) if len(value_series) == 1 else float(value_series.sum())
            st.metric(recommendation.y_column, f"{kpi_value:,.2f}")
            return

    metric_cols = [column for column in result_semantics.measure_columns if column in raw_result_df.columns]
    if not metric_cols:
        metric_cols = [
            column
            for column in raw_result_df.select_dtypes(include="number").columns.tolist()
            if column in raw_result_df.columns and str(column).lower() != "row_index"
        ]
    if not metric_cols:
        raw_result_df = raw_result_df.copy()
        raw_result_df["__row_count"] = 1
        metric_cols = ["__row_count"]

    grain_options = list(result_semantics.base_grain_columns or result_semantics.dimension_columns)
    if not grain_options:
        raw_num_cols = [
            column
            for column in raw_result_df.select_dtypes(include="number").columns.tolist()
            if str(column).lower() != "row_index"
        ]
        raw_cat_cols = [column for column in raw_result_df.columns if column not in raw_num_cols]
        grain_options = [column for column in raw_cat_cols if column in raw_result_df.columns]
    if not grain_options:
        if projection_query:
            st.caption("Projection results default to a table because there is no chart grain to preserve.")
            st.dataframe(raw_result_df, use_container_width=True, hide_index=True)
            return
        st.info("No categorical dimensions are available for charting this result.")
        return

    default_y = recommendation.y_column if recommendation.y_column in metric_cols else metric_cols[0]
    default_chart_grain = [column for column in recommendation.chart_grain_columns if column in grain_options] or [grain_options[0]]
    default_sort = "label_asc" if any(_is_year_axis_column(raw_result_df, column) for column in default_chart_grain) else "value_desc"
    chart_type_options = list(dict.fromkeys(recommendation.chart_types + ["Table", "Bar Chart", "Pie Chart", "Trend Line"]))
    default_chart_type_index = chart_type_options.index(recommendation.default_chart_type) if recommendation.default_chart_type in chart_type_options else 0

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        st.text_input(
            "Base result grain",
            value=result_semantics.base_grain_label,
            key=f"{scope_key}_dash_base_grain",
            disabled=True,
        )
    with control_col2:
        chart_grain = st.multiselect(
            "Chart grain",
            options=grain_options,
            default=default_chart_grain,
            max_selections=2,
            key=f"{scope_key}_dash_chart_grain",
            help="Choose the dimensions to preserve in chart data before plotting.",
        )

    if not chart_grain:
        chart_grain = [grain_options[0]]

    x_options = list(chart_grain)
    x_default = 0
    if recommendation.x_column in x_options:
        x_default = x_options.index(recommendation.x_column)

    control_col3, control_col4, control_col5 = st.columns(3)
    with control_col3:
        x_col = st.selectbox(
            "X column",
            options=x_options,
            index=x_default,
            key=f"{scope_key}_dash_x_col",
            help="Primary grouping field shown on the X-axis or pie labels.",
        )
    with control_col4:
        y_default_index = metric_cols.index(default_y) if isinstance(default_y, str) and default_y in metric_cols else 0
        y_col = st.selectbox(
            "Y column",
            options=metric_cols,
            index=y_default_index,
            key=f"{scope_key}_dash_y_col",
            help="Measure column used for chart aggregation.",
        )
    with control_col5:
        chart_type = st.selectbox(
            "Chart type",
            options=chart_type_options,
            index=default_chart_type_index,
            key=f"{scope_key}_dash_chart_type",
        )

    aggregation_options = valid_aggregations_for_y(raw_result_df, y_col, result_payload)
    default_aggregation = "RAW" if "RAW" in aggregation_options else ("SUM" if "SUM" in aggregation_options else aggregation_options[0])
    split_options = [None] + [column for column in chart_grain if column != x_col]
    default_split = recommendation.split_column if recommendation.split_column in split_options else split_options[0]
    if chart_type == "Pie Chart":
        default_split = None

    control_col6, control_col7, control_col8, control_col9 = st.columns(4)
    with control_col6:
        aggregation = st.selectbox(
            "Chart aggregation",
            options=aggregation_options,
            index=aggregation_options.index(default_aggregation),
            key=f"{scope_key}_dash_aggregation",
        )
    with control_col7:
        split_col = st.selectbox(
            "Split by",
            options=split_options,
            index=split_options.index(default_split) if default_split in split_options else 0,
            key=f"{scope_key}_dash_split_by",
            disabled=chart_type == "Pie Chart",
        )
    with control_col8:
        sort_by = st.selectbox(
            "Sort by",
            options=["value_desc", "value_asc", "label_asc", "label_desc"],
            index=["value_desc", "value_asc", "label_asc", "label_desc"].index(default_sort),
            key=f"{scope_key}_dash_sort_by",
            format_func=lambda value: {
                "value_desc": "value desc",
                "value_asc": "value asc",
                "label_asc": "label asc",
                "label_desc": "label desc",
            }[value],
        )
    with control_col9:
        top_n_cap = max(1, _safe_nunique(raw_result_df[x_col]))
        top_n_default = min(10, top_n_cap)
        if top_n_cap <= 1:
            top_n = 1
            st.caption("Only one chart category is available, so Top N is fixed at 1.")
        else:
            try:
                top_n = st.slider(
                    "Top N",
                    min_value=1,
                    max_value=top_n_cap,
                    value=top_n_default,
                    key=f"{scope_key}_dash_top_n",
                )
            except StreamlitAPIException:
                top_n = top_n_default
                st.caption(f"Top N is fixed at {top_n_default} because the slider range is not valid for this chart.")

    chart_filter_dimension_cols = [
        column for column in result_semantics.dimension_columns if column in raw_result_df.columns
    ]
    chart_filter_measure_cols = [
        column for column in result_semantics.measure_columns if column in raw_result_df.columns
    ]
    if not chart_filter_measure_cols and y_col in raw_result_df.columns:
        chart_filter_measure_cols = [y_col]

    st.markdown("**Chart filters**")
    reset_requested = st.button("Reset filters", key=f"{scope_key}_chart_filters_reset")
    if reset_requested:
        _clear_result_filter_state(scope_key, chart_filter_dimension_cols, chart_filter_measure_cols)
        st.rerun()

    categorical_filter_values: dict[str, list[str]] = {}
    numeric_filter_values: dict[str, tuple[float, float]] = {}

    for column_name in chart_filter_dimension_cols:
        options = sorted(raw_result_df[column_name].dropna().astype(str).unique().tolist())
        if not options:
            continue
        selected_values = st.multiselect(
            f"Filter {column_name}",
            options=options,
            default=options,
            key=f"{scope_key}_chart_filter_{column_name}",
        )
        if len(selected_values) < len(options):
            categorical_filter_values[column_name] = selected_values

    for column_name in chart_filter_measure_cols:
        numeric_series = pd.to_numeric(raw_result_df[column_name], errors="coerce")
        valid_series = numeric_series.dropna()
        if valid_series.empty:
            continue
        min_value = float(valid_series.min())
        max_value = float(valid_series.max())
        selected_range = _numeric_range_filter_widget(column_name, valid_series, scope_key)
        if selected_range is not None and selected_range != (min_value, max_value):
            numeric_filter_values[column_name] = (float(selected_range[0]), float(selected_range[1]))

    filter_result = apply_dashboard_filters(
        raw_result_df,
        categorical_filters=categorical_filter_values,
        numeric_filters=numeric_filter_values,
        text_filters={},
    )
    query_filter_labels = _result_filter_labels(result_payload)
    query_singleton_filters = _result_singleton_filters(result_payload)
    singleton_dimension_filters = {
        **query_singleton_filters,
        **_singleton_categorical_filters(categorical_filter_values),
    }
    combined_filter_labels = [
        *query_filter_labels,
        *[label for label in filter_result.active_filter_labels if label not in query_filter_labels],
    ]
    combined_filter_summary = "; ".join(combined_filter_labels) if combined_filter_labels else "None"
    filtered_chart_input_df = filter_result.dataframe
    total_rows = int(filtered_chart_input_df.shape[0])
    total_cols = int(filtered_chart_input_df.shape[1])
    filtered_num_cols = [
        column
        for column in filtered_chart_input_df.select_dtypes(include="number").columns.tolist()
        if str(column).lower() != "row_index"
    ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", total_rows)
    m2.metric("Columns", total_cols)
    m3.metric("Numeric columns", len(filtered_num_cols))
    m4.metric("Filtered out", max(0, original_rows - total_rows))

    if total_rows == 0:
        st.info("No rows match the selected chart filters.")
        return

    st.caption(summarize_active_filters(combined_filter_labels))
    st.caption(f"Rows after filter: {total_rows} of {original_rows}")

    with st.expander("Filtered chart input preview", expanded=False):
        st.dataframe(filtered_chart_input_df, use_container_width=True, hide_index=True)

    if chart_type == "Table":
        st.markdown("**Table**")
        st.dataframe(filtered_chart_input_df, use_container_width=True, hide_index=True)
        return

    if x_col is not None:
        st.caption(_describe_column(x_col, "X-axis"))
    if y_col is not None:
        st.caption(_describe_column(y_col, "Y-axis"))

    chart_semantics = infer_chart_semantics(
        result_semantics,
        x_column=x_col,
        y_column=y_col,
        aggregation=aggregation,
        chart_type=chart_type,
        split_column=split_col,
    )
    filtered_chart_data = build_filterable_chart_data(
        raw_result_df,
        categorical_filters=categorical_filter_values,
        numeric_filters=numeric_filter_values,
        text_filters={},
        x_column=x_col,
        y_column=y_col,
        aggregation=aggregation,
        split_column=chart_semantics.split_column,
        top_n=top_n,
        sort_by=sort_by,
    )
    chart_result = filtered_chart_data.chart_result
    chart_df = chart_result.dataframe
    insight_chart_result = build_chart_dataframe(
        filtered_chart_input_df,
        x_column=x_col,
        y_column=y_col,
        aggregation=aggregation,
        split_column=chart_semantics.split_column,
        top_n=None,
        sort_by=sort_by,
    )
    if chart_df.empty:
        st.info("No chartable rows remain after applying chart aggregation.")
        return

    insight_summary, insight_facts = generate_dashboard_insight_summary(
        insight_chart_result.dataframe,
        chart_semantics=chart_semantics,
        result_semantics=result_semantics,
        x_column=x_col,
        value_column=insight_chart_result.value_column,
        x_label=x_col,
        y_label=y_col,
        active_filter_summary=combined_filter_summary,
        rows_after_filter=total_rows,
        prefer_llm=True,
    )

    if insight_summary:
        st.markdown("**Insights**")
        st.info(insight_summary)
        show_debug = bool(st.session_state.get("show_planner_debug"))
        if show_debug and insight_facts is not None:
            with st.expander("Insight facts", expanded=False):
                st.json(asdict(insight_facts))

    if result_semantics.grouped_result:
        _render_chart_semantic_summary(
            applied_filter_summary=combined_filter_summary,
            result_grain_label=chart_semantics.base_grain_label,
            chart_grain_label=chart_semantics.chart_grain_label,
            chart_aggregation_label=chart_semantics.aggregation_label,
            raw_rows=chart_result.rows_in_raw_result,
            chart_rows=chart_result.rows_in_chart_data,
        )

    if chart_semantics.requires_secondary_aggregation:
        st.info(
            f"Chart is using secondary aggregation over the SQL result: {chart_semantics.aggregation_label}"
        )
    if chart_semantics.pie_requires_single_dimension:
        st.info(
            f"Pie chart is using secondary aggregation: {chart_semantics.aggregation_label}"
        )
    if result_semantics.planner_warnings:
        for warning in result_semantics.planner_warnings[:3]:
            st.caption(f"Planner warning carried into dashboard: {warning}")

    hidden_singleton_filters = {
        column_name: value
        for column_name, value in singleton_dimension_filters.items()
        if column_name not in chart_result.preview_columns
    }
    retain_singleton_dimensions = st.checkbox(
        "Retain singleton filtered dimensions in chart preview",
        value=bool(hidden_singleton_filters),
        key=f"{scope_key}_chart_preview_retain_singletons",
        help="Show one-value filtered dimensions in the aggregated chart preview as contextual columns.",
    )
    chart_preview_df = _build_chart_preview_dataframe(
        chart_df,
        chart_result.preview_columns,
        hidden_singleton_filters,
        retain_singleton_dimensions=retain_singleton_dimensions,
    )

    singleton_context_caption = _singleton_filter_context_caption(hidden_singleton_filters)
    if singleton_context_caption:
        st.caption(singleton_context_caption)

    with st.expander("Chart data after aggregation", expanded=False):
        st.dataframe(
            chart_preview_df,
            use_container_width=True,
            hide_index=True,
        )

    if hidden_singleton_filters and not retain_singleton_dimensions:
        st.caption(_describe_hidden_singleton_filters(hidden_singleton_filters, chart_semantics.chart_grain_label))

    value_col = chart_result.value_column
    x_unique = _safe_nunique(filtered_chart_input_df[x_col]) if x_col is not None else 0
    x_is_id_like = _is_identifier_col_name(x_col) if x_col is not None else False
    if x_col not in (None, "__row_order") and (x_is_id_like or x_unique > max(25, int(total_rows * 0.7))):
        st.info(
            f"`{x_col}` is high-cardinality/ID-like ({x_unique} unique values). "
            "Charts may be less meaningful. Consider choosing a broader chart grain."
        )

    st.markdown(f"**{chart_type}**")

    if chart_type == "Pie Chart":
        pie_df = _append_context_columns(chart_df[[x_col, value_col]].copy(), hidden_singleton_filters)
        if alt is not None:
            plot_df = pie_df.copy()
            total_val = float(plot_df[value_col].sum() or 0)
            plot_df["_pct"] = (plot_df[value_col] / total_val) * 100.0 if total_val > 0 else 0.0
            pie_tooltips = [alt.Tooltip(f"{x_col}:N", title=x_col)]
            pie_tooltips.extend(
                [alt.Tooltip(f"{column_name}:N", title=column_name) for column_name in hidden_singleton_filters]
            )
            pie_tooltips.extend(
                [
                    alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.2f"),
                    alt.Tooltip("_pct:Q", title="Percent", format=".2f"),
                ]
            )
            show_pie_values = st.checkbox(
                "Show values on pie slices",
                value=True,
                key=f"{scope_key}_pie_show_values",
            )
            plot_df["_label"] = plot_df.apply(
                lambda row: f"{float(row[value_col]):,.2f} ({float(row['_pct']):.1f}%)",
                axis=1,
            )
            base_pie = (
                alt.Chart(plot_df)
                .mark_arc()
                .encode(
                    theta=alt.Theta(f"{value_col}:Q", stack=True),
                    color=alt.Color(f"{x_col}:N", legend=alt.Legend(title=x_col)),
                    tooltip=pie_tooltips,
                )
            )
            if show_pie_values:
                text_layer = (
                    alt.Chart(plot_df)
                    .mark_text(radius=120, size=11)
                    .encode(theta=alt.Theta(f"{value_col}:Q", stack=True), text=alt.Text("_label:N"))
                )
                st.altair_chart((base_pie + text_layer).properties(height=420), use_container_width=True)
            else:
                st.altair_chart(base_pie.properties(height=360), use_container_width=True)
        else:
            total_for_pie = float(pie_df[value_col].sum() or 0)
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))

            def _autopct_with_value(pct: float) -> str:
                val = (pct / 100.0) * total_for_pie
                return f"{pct:.1f}%\n({val:,.2f})"

            ax_pie.pie(
                pie_df[value_col].tolist(),
                labels=pie_df[x_col].astype(str).tolist(),
                autopct=_autopct_with_value,
                startangle=90,
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
            _download_figure(fig_pie, "Download pie chart", "pie_chart.png", f"{scope_key}_download_pie_chart")
            plt.close(fig_pie)
        return

    if chart_type == "Trend Line":
        if chart_semantics.split_column and alt is not None:
            trend_tooltips = [
                alt.Tooltip(f"{x_col}:N", title=x_col),
                alt.Tooltip(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
                alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.2f"),
            ]
            trend_tooltips.extend(
                [alt.Tooltip(f"{column_name}:N", title=column_name) for column_name in hidden_singleton_filters]
            )
            trend_df = _append_context_columns(chart_df.copy(), hidden_singleton_filters)
            trend_chart = (
                alt.Chart(trend_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{x_col}:N", title=x_col, sort=None),
                    y=alt.Y(f"{value_col}:Q", title=value_col),
                    color=alt.Color(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
                    tooltip=trend_tooltips,
                )
                .properties(height=360)
            )
            st.altair_chart(trend_chart, use_container_width=True)
        else:
            trend_df = chart_df[[x_col, value_col]].copy().set_index(x_col)
            st.line_chart(trend_df[value_col])
        return

    if chart_type == "Heatmap":
        if not chart_semantics.split_column:
            st.info("Heatmap needs a split-by dimension to create a two-dimensional grid.")
            return
        if alt is not None:
            heatmap_df = _append_context_columns(chart_df.copy(), hidden_singleton_filters)
            heatmap_tooltips = [
                alt.Tooltip(f"{x_col}:N", title=x_col),
                alt.Tooltip(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
                alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.2f"),
            ]
            heatmap_tooltips.extend(
                [alt.Tooltip(f"{column_name}:N", title=column_name) for column_name in hidden_singleton_filters]
            )
            heatmap = (
                alt.Chart(heatmap_df)
                .mark_rect()
                .encode(
                    x=alt.X(f"{x_col}:N", title=x_col),
                    y=alt.Y(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
                    color=alt.Color(f"{value_col}:Q", title=value_col),
                    tooltip=heatmap_tooltips,
                )
                .properties(height=360)
            )
            st.altair_chart(heatmap, use_container_width=True)
        else:
            heatmap_df = chart_df.pivot(index=chart_semantics.split_column, columns=x_col, values=value_col).fillna(0)
            fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
            im = ax_heat.imshow(heatmap_df.values, aspect="auto")
            ax_heat.set_xticks(range(len(heatmap_df.columns)))
            ax_heat.set_xticklabels([str(item) for item in heatmap_df.columns], rotation=30, ha="right")
            ax_heat.set_yticks(range(len(heatmap_df.index)))
            ax_heat.set_yticklabels([str(item) for item in heatmap_df.index])
            fig_heat.colorbar(im, ax=ax_heat)
            st.pyplot(fig_heat)
            _download_figure(fig_heat, "Download heatmap", "heatmap.png", f"{scope_key}_download_heatmap")
            plt.close(fig_heat)
        return

    if chart_type == "Treemap":
        try:
            squarify = importlib.import_module("squarify")
        except Exception:
            squarify = None
        if squarify is None:
            st.info("Treemap rendering is unavailable because the optional `squarify` package is not installed.")
            return
        labels = chart_df[x_col].astype(str).tolist()
        if chart_semantics.split_column:
            labels = [f"{label} | {split}" for label, split in zip(chart_df[x_col], chart_df[chart_semantics.split_column])]
        fig_tree, ax_tree = plt.subplots(figsize=(10, 5))
        squarify.plot(
            sizes=chart_df[value_col].clip(lower=0.000001).tolist(),
            label=[_compact_label(label, max_len=30) for label in labels],
            ax=ax_tree,
        )
        ax_tree.axis("off")
        st.pyplot(fig_tree)
        _download_figure(fig_tree, "Download treemap", "treemap.png", f"{scope_key}_download_treemap")
        plt.close(fig_tree)
        return

    if chart_semantics.split_column and chart_type in {"Stacked Bar", "Grouped Bar", "Bar Chart"} and alt is not None:
        bar_df = _append_context_columns(chart_df.copy(), hidden_singleton_filters)
        value_format = ",.0f" if aggregation == "COUNT" else ",.2f"
        bar_tooltips = [
            alt.Tooltip(f"{x_col}:N", title=x_col),
            alt.Tooltip(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
            alt.Tooltip(f"{value_col}:Q", title=value_col, format=value_format),
        ]
        bar_tooltips.extend(
            [alt.Tooltip(f"{column_name}:N", title=column_name) for column_name in hidden_singleton_filters]
        )
        encode = {
            "x": alt.X(f"{x_col}:N", title=x_col, sort=None),
            "y": alt.Y(f"{value_col}:Q", title=value_col),
            "color": alt.Color(f"{chart_semantics.split_column}:N", title=chart_semantics.split_column),
            "tooltip": bar_tooltips,
        }
        if use_grouped_split_bars(chart_type=chart_type, aggregation=aggregation):
            encode["xOffset"] = alt.XOffset(f"{chart_semantics.split_column}:N")
        bar_chart = alt.Chart(bar_df).mark_bar().encode(**encode).properties(height=360)
        st.altair_chart(bar_chart, use_container_width=True)
        return

    if alt is not None:
        bar_plot_df = _append_context_columns(chart_df[[x_col, value_col]].copy(), hidden_singleton_filters)
        bar_tooltips = [
            alt.Tooltip(f"{x_col}:N", title=x_col),
            alt.Tooltip(f"{value_col}:Q", title=value_col, format=",.2f"),
        ]
        bar_tooltips.extend(
            [alt.Tooltip(f"{column_name}:N", title=column_name) for column_name in hidden_singleton_filters]
        )
        chart = (
            alt.Chart(bar_plot_df)
            .mark_bar()
            .encode(
                y=alt.Y(f"{x_col}:N", title=x_col, sort="-x", axis=alt.Axis(labelLimit=180)),
                x=alt.X(f"{value_col}:Q", title=value_col),
                tooltip=bar_tooltips,
            )
            .properties(height=max(280, min(700, 28 * len(bar_plot_df))))
        )
        st.altair_chart(chart, use_container_width=True)
        return

    labels = [_compact_label(value) for value in chart_df[x_col].tolist()]
    many_categories = len(labels) > 10
    long_labels = any(len(str(value)) > 14 for value in chart_df[x_col].tolist())
    if many_categories or long_labels:
        fig_h = max(4.5, min(12.0, 0.42 * len(labels) + 1.8))
        fig_bar, ax_bar = plt.subplots(figsize=(9, fig_h))
        y_pos = list(range(len(labels)))
        ax_bar.barh(y_pos, chart_df[value_col].tolist())
        wrapped = ["\n".join(textwrap.wrap(label, width=18)) for label in labels]
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(wrapped)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel(value_col)
        ax_bar.set_ylabel(x_col)
    else:
        fig_bar, ax_bar = plt.subplots(figsize=(9, 4.8))
        ax_bar.bar(labels, chart_df[value_col])
        ax_bar.set_xlabel(x_col)
        ax_bar.set_ylabel(value_col)
        ax_bar.tick_params(axis="x", labelrotation=30)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    _download_figure(fig_bar, "Download bar chart", "bar_chart.png", f"{scope_key}_download_bar_chart")
    plt.close(fig_bar)


st.set_page_config(page_title="Data QA Agent", layout="wide")
st.title("Data QA Agent")
st.caption(f"Backend: {BACKEND}")

if st.sidebar.checkbox("Manage Business Rules"):
    st.subheader("Rules Registry")
    # Pre-seed the Registry with the Value Gap Summary
    if st.button("Add Default Value Gap Rule"):
        _create_rule_registry({
            "intent": "customer_value_gap_summary",
            "patterns": ["Customer value gap", "Show billing gap"],
            "sql_template": """
SELECT 
    o.row_json->>'customer' AS customer_id,
    COUNT(DISTINCT o.row_json->>'event_id') AS event_count,
    SUM(NULLIF(regexp_replace((o.row_json->>'expected_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_expected,
    SUM(NULLIF(regexp_replace((c.row_json->>'billed_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_billed,
    (SUM(NULLIF(regexp_replace((o.row_json->>'expected_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) - 
     SUM(NULLIF(regexp_replace((c.row_json->>'billed_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric)) AS value_gap
FROM dataset_rows o
LEFT JOIN dataset_rows c ON o.dataset_id = c.dataset_id 
    AND c.sheet_name = 'Agent_Output'
    AND NULLIF(LOWER(TRIM(o.row_json->>'event_id')), '') = NULLIF(LOWER(TRIM(c.row_json->>'event_id')), '')
WHERE o.dataset_id = %(dataset_id)s AND o.sheet_name = 'Ops_Events'
GROUP BY o.row_json->>'customer'
ORDER BY value_gap DESC
            """.strip(),
            "enabled": True
        })
        st.success("Value Gap rule added to registry!")

    # Fetch existing rules from Supabase (or backend fallback)
    try:
        if sb is not None:
            # Add customer value gap summary rule to registry
            try:
                _create_rule_registry({
                    "intent": "customer_value_gap_summary",
                    "patterns": [
                        "Show customer billing gap",
                        "Summarize expected vs billed by customer",
                        "Customer value gap",
                        "Customer billing gap analysis"
                    ],
                    "sql_template": "SELECT \n    o.row_json->>'customer' AS customer_id,\n    COUNT(DISTINCT o.row_json->>'event_id') AS event_count,\n    SUM(NULLIF(regexp_replace((o.row_json->>'expected_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_expected_amount,\n    SUM(NULLIF(regexp_replace((c.row_json->>'billed_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) AS total_billed_amount,\n    (SUM(NULLIF(regexp_replace((o.row_json->>'expected_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric) - \n     SUM(NULLIF(regexp_replace((c.row_json->>'billed_amount'), '[^0-9.\\-]', '', 'g'), '')::numeric)) AS value_gap\nFROM dataset_rows o\nLEFT JOIN dataset_rows c ON o.dataset_id = c.dataset_id \n    AND c.sheet_name = 'Agent_Output'\n    AND NULLIF(LOWER(TRIM(o.row_json->>'event_id')), '') = NULLIF(LOWER(TRIM(c.row_json->>'event_id')), '')\nWHERE o.dataset_id = %(dataset_id)s\n  AND o.sheet_name = 'Ops_Events'\nGROUP BY o.row_json->>'customer'\nORDER BY value_gap DESC",
                    "enabled": True,
                    "priority": 100,
                    "description": "Summarize expected vs billed amounts and value gap by customer."
                })
            except Exception as exc:
                st.error(f"Failed to add customer value gap rule: {exc}")
            rules = sb.table("rules_registry").select("*").execute().data or []
        else:
            rules = _fetch_rules_registry()
    except Exception as exc:
        rules = []
        st.error(f"Failed to fetch rules: {exc}")

    if rules:
        st.dataframe(pd.DataFrame(rules), use_container_width=True)

    # UI for adding a new Rule (Intent -> SQL Template)
    with st.form("new_rule"):
        pattern = st.text_input("Question Pattern (e.g., 'total profit')")
        template = st.text_area("SQL Template (Use %(dataset_id)s)")
        if st.form_submit_button("Save Rule"):
            if not (pattern or "").strip() or not (template or "").strip():
                st.error("Pattern and SQL template are required.")
            else:
                try:
                    if sb is not None:
                        # Try legacy fields first (as requested snippet), then current schema fallback.
                        try:
                            sb.table("rules_registry").insert({"pattern": pattern, "sql": template}).execute()
                        except Exception:
                            sb.table("rules_registry").insert(
                                {
                                    "intent": f"custom_{_intent_slug(pattern)[:40]}",
                                    "patterns": [pattern],
                                    "sql_template": template,
                                    "enabled": True,
                                    "priority": 100,
                                }
                            ).execute()
                    else:
                        _create_rule_registry(
                            {
                                "intent": f"custom_{_intent_slug(pattern)[:40]}",
                                "patterns": [pattern],
                                "sql_template": template,
                                "enabled": True,
                                "priority": 100,
                            }
                        )
                    st.success("Rule saved!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

if "selected_dataset_id" not in st.session_state:
    st.session_state["selected_dataset_id"] = ""
if "selected_dataset_name" not in st.session_state:
    st.session_state["selected_dataset_name"] = ""
if "sql_results_by_dataset" not in st.session_state:
    st.session_state["sql_results_by_dataset"] = {}
if "ask_sql_question" not in st.session_state:
    st.session_state["ask_sql_question"] = ""
if "ask_sql_question_by_dataset" not in st.session_state:
    st.session_state["ask_sql_question_by_dataset"] = {}
if "previous_selected_dataset_id" not in st.session_state:
    st.session_state["previous_selected_dataset_id"] = ""
if "show_planner_debug" not in st.session_state:
    st.session_state["show_planner_debug"] = False

with st.expander("How to upload from Downloads", expanded=True):
    st.markdown(
        """
1. Use **Upload via file picker** and choose a file from your Downloads folder.
2. Or use **Upload by Downloads path** and paste the full path.

Supported types: `.csv`, `.xlsx`, `.xls`
"""
    )

st.subheader("Upload via file picker")
uploaded = st.file_uploader("Select a CSV or Excel file", type=["csv", "xlsx", "xls"])
dataset_name_picker = st.text_input("Dataset name (optional)", key="dataset_name_picker")
include_assumptions_sheet = st.checkbox(
    "Include 'Assumptions' sheet (Excel only)",
    value=True,
    key="include_assumptions_sheet",
)

if uploaded is not None:
    _render_instant_file_dashboard(
        uploaded.name,
        uploaded.getvalue(),
        scope_key="uploaded_file_instant_dashboard",
    )

if st.button("Upload selected file", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Please choose a file first.")
    else:
        try:
            with st.spinner("Thinking... uploading and profiling your file"):
                out = _upload_to_backend(
                    uploaded.name,
                    uploaded.getvalue(),
                    dataset_name_picker,
                    include_assumptions_sheet=include_assumptions_sheet,
                )
            st.session_state["selected_dataset_id"] = out.get("dataset_id", "")
            st.session_state["selected_dataset_name"] = out.get("name", "")
            st.success(f"Uploaded: {out.get('name', uploaded.name)} ({out.get('rows', 0)} rows)")
            if out.get("sheets"):
                st.write("Sheets:", ", ".join(out["sheets"]))
            _show_cleaning_report(out)
            _show_relationships(out)
            _show_upload_qa(out)
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

st.divider()
st.subheader("Upload by Downloads path")
default_downloads = str(Path.home() / "Downloads")
path_input = st.text_input(
    "File path",
    value=default_downloads,
    help="Example: C:/Users/YourName/Downloads/sales.csv",
)
dataset_name_path = st.text_input("Dataset name (optional)", key="dataset_name_path")

path_preview = Path(path_input).expanduser()
if path_input and path_preview.exists() and path_preview.is_file() and _is_supported_file(path_preview):
    _render_instant_file_dashboard(
        path_preview.name,
        path_preview.read_bytes(),
        scope_key="path_file_instant_dashboard",
    )

if st.button("Upload file from path", use_container_width=True):
    try:
        file_path = Path(path_input).expanduser()

        if not file_path.exists() or not file_path.is_file():
            st.error("File not found. Check the path and try again.")
        elif not _is_supported_file(file_path):
            st.error("Unsupported file type. Use CSV or Excel.")
        else:
            data = file_path.read_bytes()
            with st.spinner("Thinking... uploading and profiling your file"):
                out = _upload_to_backend(
                    file_path.name,
                    data,
                    dataset_name_path,
                    include_assumptions_sheet=include_assumptions_sheet,
                )
            st.session_state["selected_dataset_id"] = out.get("dataset_id", "")
            st.session_state["selected_dataset_name"] = out.get("name", "")
            st.success(f"Uploaded: {out.get('name', file_path.name)} ({out.get('rows', 0)} rows)")
            if out.get("sheets"):
                st.write("Sheets:", ", ".join(out["sheets"]))
            _show_cleaning_report(out)
            _show_relationships(out)
            _show_upload_qa(out)
    except Exception as exc:
        st.error(f"Upload failed: {exc}")

st.divider()
st.subheader("Select dataset")

if st.button("Refresh dataset list", use_container_width=True):
    st.rerun()

try:
    datasets = _fetch_datasets()
except Exception as exc:
    datasets = []
    st.error(f"Failed to load datasets: {exc}")

active_dataset_ids = {str(d.get("id", "")) for d in datasets if d.get("id")}

# Keep session state aligned with backend state so deleted datasets disappear cleanly.
if st.session_state.get("selected_dataset_id") and st.session_state["selected_dataset_id"] not in active_dataset_ids:
    st.session_state["selected_dataset_id"] = ""
    st.session_state["selected_dataset_name"] = ""

if st.session_state.get("sql_results_by_dataset"):
    stale_result_ids = [
        ds_id
        for ds_id in list(st.session_state["sql_results_by_dataset"].keys())
        if ds_id not in active_dataset_ids
    ]
    for ds_id in stale_result_ids:
        st.session_state["sql_results_by_dataset"].pop(ds_id, None)

dataset_options = {
    f"{d.get('name', 'unnamed')} ({d.get('id', '')[:8]})": d.get("id", "") for d in datasets
}

selected_dataset_meta: dict = {}

if dataset_options:
    labels = list(dataset_options.keys())
    default_index = 0
    if st.session_state["selected_dataset_id"]:
        for i, label in enumerate(labels):
            if dataset_options[label] == st.session_state["selected_dataset_id"]:
                default_index = i
                break

    selected_label = st.selectbox("Dataset", labels, index=default_index)
    st.session_state["selected_dataset_id"] = dataset_options[selected_label]
    st.session_state["selected_dataset_name"] = selected_label
    selected_dataset_meta = next(
        (d for d in datasets if d.get("id") == st.session_state["selected_dataset_id"]),
        {},
    )
else:
    st.info("No datasets found yet. Upload one to continue.")

current_selected_dataset_id = str(st.session_state.get("selected_dataset_id") or "")
previous_selected_dataset_id = str(st.session_state.get("previous_selected_dataset_id") or "")
if current_selected_dataset_id != previous_selected_dataset_id:
    if previous_selected_dataset_id:
        st.session_state["ask_sql_question_by_dataset"][previous_selected_dataset_id] = st.session_state.get("ask_sql_question", "")
    st.session_state["ask_sql_question"] = st.session_state["ask_sql_question_by_dataset"].get(current_selected_dataset_id, "")
    st.session_state["previous_selected_dataset_id"] = current_selected_dataset_id

st.caption(
    f"Selected dataset: {st.session_state.get('selected_dataset_name') or st.session_state.get('selected_dataset_id') or 'none'}"
)

if st.session_state.get("selected_dataset_id"):
    with st.expander("Adaptive dashboard for selected dataset", expanded=False):
        st.caption("Load the instant adaptive dashboard from rows already stored in the backend for this dataset.")
        if st.button("Load dashboard for selected dataset", key="load_selected_dataset_dashboard", use_container_width=True):
            st.session_state["show_selected_dataset_dashboard"] = True

        if st.session_state.get("show_selected_dataset_dashboard"):
            _render_uploaded_dataset_dashboard(
                st.session_state["selected_dataset_id"],
                str(selected_dataset_meta.get("name") or st.session_state.get("selected_dataset_name") or "dataset"),
                scope_key=f"stored_dataset_dashboard_{st.session_state['selected_dataset_id']}",
            )

    relationships_payload = selected_dataset_meta.get("relationships") or {}
    relationship_options = [item for item in (relationships_payload.get("relationships") or []) if isinstance(item, dict)]
    if relationship_options:
        with st.expander("Joined multi-sheet dashboard", expanded=False):
            st.caption("Use a detected relationship to analyze a merged view across sheets.")
            relationship_labels = [
                f"{rel.get('left')}.{rel.get('left_col')} -> {rel.get('right')}.{rel.get('right_col')} ({rel.get('type', 'join')})"
                for rel in relationship_options
            ]
            selected_relationship_label = st.selectbox(
                "Detected relationship",
                options=relationship_labels,
                key=f"joined_relationship_{st.session_state['selected_dataset_id']}",
            )
            selected_relationship = relationship_options[relationship_labels.index(selected_relationship_label)]
            join_how = st.selectbox(
                "Join mode",
                options=["left", "inner", "right", "outer"],
                index=0,
                key=f"joined_relationship_mode_{st.session_state['selected_dataset_id']}",
            )
            if st.button("Load joined dashboard", key="load_joined_dataset_dashboard", use_container_width=True):
                st.session_state["show_joined_dataset_dashboard"] = True

            if st.session_state.get("show_joined_dataset_dashboard"):
                _render_joined_uploaded_dataset_dashboard(
                    st.session_state["selected_dataset_id"],
                    str(selected_dataset_meta.get("name") or st.session_state.get("selected_dataset_name") or "dataset"),
                    selected_relationship,
                    join_how,
                    scope_key=f"joined_dataset_dashboard_{st.session_state['selected_dataset_id']}",
                )

with st.expander("Rules Registry Admin", expanded=False):
    st.caption("Manage deterministic/dynamic intent rules stored in Supabase `rules_registry`.")

    if st.button("Refresh rules", key="refresh_rules_registry", use_container_width=True):
        st.rerun()

    try:
        rules = _fetch_rules_registry()
    except Exception as exc:
        rules = []
        st.error(f"Failed to load rules registry: {exc}")

    if rules:
        preview_df = pd.DataFrame(rules)
        preview_cols = [c for c in ["id", "priority", "enabled", "intent", "builder", "patterns", "description"] if c in preview_df.columns]
        st.dataframe(preview_df[preview_cols] if preview_cols else preview_df, use_container_width=True)
    else:
        st.info("No rules found yet.")

    st.markdown("**Create rule**")
    c1, c2, c3 = st.columns(3)
    with c1:
        new_intent = st.text_input("Intent", key="rr_new_intent", placeholder="month_year_metric_total")
    with c2:
        new_priority = st.number_input("Priority", min_value=0, max_value=100000, value=100, step=1, key="rr_new_priority")
    with c3:
        new_enabled = st.checkbox("Enabled", value=True, key="rr_new_enabled")

    new_builder = st.text_input("Builder", key="rr_new_builder", placeholder="month_year_metric_total")
    new_sql_template = st.text_area(
        "SQL template (optional if builder is provided)",
        key="rr_new_sql_template",
        placeholder="SELECT ...",
        height=100,
    )
    new_desc = st.text_input("Description", key="rr_new_desc")
    new_patterns_text = st.text_area(
        "Patterns (one regex per line)",
        key="rr_new_patterns",
        placeholder=r"\\bdashboard\\b\n\\boverview\\b",
        height=110,
    )

    if st.button("Create rule", key="rr_create_btn", use_container_width=True):
        patterns = [line.strip() for line in (new_patterns_text or "").splitlines() if line.strip()]
        payload = {
            "intent": (new_intent or "").strip(),
            "priority": int(new_priority),
            "enabled": bool(new_enabled),
            "patterns": patterns,
            "builder": (new_builder or "").strip() or None,
            "sql_template": (new_sql_template or "").strip() or None,
            "description": (new_desc or "").strip() or None,
        }
        try:
            created = _create_rule_registry(payload)
            st.success(f"Rule created: {created.get('id', '<unknown id>')}")
            st.rerun()
        except Exception as exc:
            st.error(f"Create failed: {exc}")

    st.markdown("**Update or delete rule**")
    rule_ids = [str(r.get("id", "")) for r in rules if r.get("id")]
    selected_rule_id = st.selectbox(
        "Rule ID",
        options=([""] + rule_ids) if rule_ids else [""],
        index=0,
        key="rr_selected_rule_id",
    )
    selected_rule = next((r for r in rules if str(r.get("id", "")) == selected_rule_id), {}) if selected_rule_id else {}

    u1, u2, u3 = st.columns(3)
    with u1:
        upd_intent = st.text_input("Intent (update)", value=str(selected_rule.get("intent", "")), key="rr_upd_intent")
    with u2:
        upd_priority = st.number_input(
            "Priority (update)",
            min_value=0,
            max_value=100000,
            value=int(selected_rule.get("priority", 100) or 100),
            step=1,
            key="rr_upd_priority",
        )
    with u3:
        upd_enabled = st.checkbox("Enabled (update)", value=bool(selected_rule.get("enabled", True)), key="rr_upd_enabled")

    upd_builder = st.text_input("Builder (update)", value=str(selected_rule.get("builder", "") or ""), key="rr_upd_builder")
    upd_sql_template = st.text_area(
        "SQL template (update)",
        value=str(selected_rule.get("sql_template", "") or ""),
        key="rr_upd_sql_template",
        height=100,
    )
    upd_desc = st.text_input("Description (update)", value=str(selected_rule.get("description", "") or ""), key="rr_upd_desc")
    upd_patterns_text = st.text_area(
        "Patterns (update, one regex per line)",
        value="\n".join([str(p) for p in (selected_rule.get("patterns") or [])]),
        key="rr_upd_patterns",
        height=110,
    )

    if st.button("Update selected rule", key="rr_update_btn", use_container_width=True):
        if not selected_rule_id:
            st.error("Pick a rule ID first.")
        else:
            patterns = [line.strip() for line in (upd_patterns_text or "").splitlines() if line.strip()]
            payload = {
                "intent": (upd_intent or "").strip(),
                "priority": int(upd_priority),
                "enabled": bool(upd_enabled),
                "patterns": patterns,
                "builder": (upd_builder or "").strip() or None,
                "sql_template": (upd_sql_template or "").strip() or None,
                "description": (upd_desc or "").strip() or None,
            }
            try:
                _update_rule_registry(selected_rule_id, payload)
                st.success(f"Rule updated: {selected_rule_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Update failed: {exc}")

    if st.button("Delete selected rule", key="rr_delete_btn", use_container_width=True):
        if not selected_rule_id:
            st.error("Pick a rule ID first.")
        else:
            try:
                _delete_rule_registry(selected_rule_id)
                st.success(f"Rule deleted: {selected_rule_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")

st.divider()
st.subheader("Ask SQL analytics")

if st.session_state.get("selected_dataset_id"):
    suggested_questions = _suggest_questions_for_dataset(selected_dataset_meta)
    if suggested_questions:
        with st.expander("Probable questions you can ask", expanded=False):
            st.caption("Click any question to auto-fill the Ask box.")
            for i, s in enumerate(suggested_questions):
                if st.button(s, key=f"suggest_q_{st.session_state['selected_dataset_id']}_{i}", use_container_width=True):
                    st.session_state["ask_sql_question"] = s
                    st.session_state["ask_sql_question_by_dataset"][st.session_state["selected_dataset_id"]] = s

q = st.text_area(
    "Ask a question",
    placeholder="Example: Which state has highest profit in 2023?",
    key="ask_sql_question",
)
st.caption("Tip: After results load, use the Year filter inside Dashboard for post-query filtering.")

analyze_all_rows = st.checkbox(
    "Analyze all rows (remove LIMIT for aggregate queries)",
    value=False,
    help="Unlimited mode is applied only for aggregate-style SQL to avoid huge raw row dumps.",
)
if analyze_all_rows:
    st.caption("Analyze-all mode is enabled for aggregate SQL. Row-level queries still use a limit.")

sql_max_rows = st.number_input(
    "Max rows to fetch",
    min_value=1,
    max_value=max(1, UI_SQL_MAX_ROWS_CAP),
    value=max(1, min(DEFAULT_SQL_MAX_ROWS, UI_SQL_MAX_ROWS_CAP)),
    step=50,
    disabled=analyze_all_rows,
    help="Used when analyze-all mode is off.",
)
st.checkbox(
    "Show planner debug details",
    key="show_planner_debug",
    help="Show deterministic planner selections, candidate sheets and metrics, filters, final SQL, and failure reasons.",
)

if st.button("Run SQL analysis", use_container_width=True):
    dataset_id = st.session_state.get("selected_dataset_id", "")
    if not dataset_id:
        st.error("Select a dataset first.")
    elif not (q or "").strip():
        st.error("Enter a question first.")
    else:
        try:
            effective_question = (q or "").strip()
            st.session_state["ask_sql_question_by_dataset"][dataset_id] = effective_question
            # Clear stale result first so failed runs do not keep showing outdated output.
            st.session_state.get("sql_results_by_dataset", {}).pop(dataset_id, None)

            with st.spinner("Thinking... analyzing your data"):
                out = _ask_sql(
                    dataset_id=dataset_id,
                    question=effective_question,
                    max_rows=int(sql_max_rows),
                    analyze_all_rows=analyze_all_rows,
                )
            st.session_state["sql_results_by_dataset"][dataset_id] = {
                "question": effective_question,
                "result": out,
            }
        except Exception as exc:
            st.error(f"SQL analysis failed: {exc}")


current_dataset_id = st.session_state.get("selected_dataset_id", "")
if current_dataset_id:
    if st.button("Clear current result", use_container_width=True):
        st.session_state.get("sql_results_by_dataset", {}).pop(current_dataset_id, None)
        st.rerun()

saved_sql = st.session_state.get("sql_results_by_dataset", {}).get(current_dataset_id)
if saved_sql:
    out = saved_sql.get("result", {})
    saved_question = saved_sql.get("question", "")
    current_typed_question = (q or "").strip()
    rows = out.get("rows", [])

    if current_typed_question and saved_question and current_typed_question != saved_question:
        st.warning("The typed question has changed. Current table is from the previous run; click Run SQL analysis to refresh.")

    # Defensive UI-side filter: when question explicitly asks unbilled,
    # avoid displaying unrelated exception categories from stale/misaligned responses.
    q_norm = (saved_question or "").lower()
    asks_unbilled = any(tok in q_norm for tok in ["unbilled", "un billed", "un-billed", "uninvoiced", "not invoiced", "not billed", "without invoice"])
    asks_all_breakdown = any(tok in q_norm for tok in ["all exceptions", "all exception", "by type", "breakdown", "all types"])
    if asks_unbilled and not asks_all_breakdown and isinstance(rows, list):
        if any(isinstance(r, dict) and "exception_type" in r for r in rows):
            ui_unbilled_rows = [
                r for r in rows
                if isinstance(r, dict) and re.sub(r"[^A-Z]", "", str(r.get("exception_type", "")).upper()) == "UNBILLED"
            ]
            if ui_unbilled_rows:
                rows = ui_unbilled_rows

    rows_returned = len(rows) if isinstance(rows, list) else int(out.get("rows_returned", 0) or 0)
    applied_limit = out.get("applied_limit")
    analyze_all_requested = bool(out.get("analyze_all_requested"))
    analyze_all_applied = bool(out.get("analyze_all_applied"))

    st.subheader("Current result")
    if saved_question:
        st.markdown("**Question used for this result**")
        st.code(saved_question)

    route_intent = str(out.get("route_intent") or "unknown")
    route_source = str(out.get("route_source") or "unknown")
    query_shape = str(out.get("query_shape") or "").strip()
    generated_at = str(out.get("response_generated_at_utc") or "")
    st.caption(f"Route: {route_intent} ({route_source})")
    _render_planner_summary(out)
    if generated_at:
        st.caption(f"Generated at (UTC): {generated_at}")

    year_col1, year_col2 = st.columns([1, 1])
    result_year_key = f"{current_dataset_id}_result_year_filter"
    default_result_year = _extract_year_from_text(saved_question or q) or 2024
    if result_year_key not in st.session_state:
        st.session_state[result_year_key] = int(default_result_year)
    with year_col1:
        result_year = st.number_input(
            "Year filter (post-result)",
            min_value=1900,
            max_value=2100,
            value=int(st.session_state[result_year_key]),
            step=1,
            key=result_year_key,
            help="Re-runs this same question with the selected year.",
        )
    with year_col2:
        if st.button("Apply year filter to this result", use_container_width=True):
            try:
                rerun_question = _set_question_year(saved_question or q, int(result_year))
                rerun_out = _ask_sql(
                    dataset_id=current_dataset_id,
                    question=rerun_question,
                    max_rows=int(sql_max_rows),
                    analyze_all_rows=analyze_all_rows,
                )
                st.session_state["sql_results_by_dataset"][current_dataset_id] = {
                    "question": rerun_question,
                    "result": rerun_out,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Year filter apply failed: {exc}")

    requested_top_n = _extract_top_n_from_question(saved_question)
    if requested_top_n is not None:
        if rows_returned < requested_top_n:
            st.info(
                f"Dynamic Top-N: requested top {requested_top_n}, but only {rows_returned} matching categories are available in this dataset/filter."
            )
        else:
            st.caption(f"Dynamic Top-N: showing top {requested_top_n} as requested.")

    if route_intent == "semantic_role_sheet_summary":
        st.success(f"Result mode: Sheet semantic summary. Sheets summarized: {rows_returned}")
    elif analyze_all_requested and analyze_all_applied:
        st.success(f"Result mode: Full aggregate rows. Rows returned: {rows_returned}")
    elif applied_limit is not None:
        if rows_returned >= int(applied_limit):
            st.warning(f"Result mode: Limited rows. Returned {rows_returned} rows (limit={applied_limit}).")
        else:
            st.info(f"Result mode: Limited rows. Returned {rows_returned} rows (limit={applied_limit}).")
    else:
        st.info(f"Rows returned: {rows_returned}")

    result_note = str(out.get("result_note") or "").strip()
    if result_note:
        st.warning(result_note)

    _render_guidance_panel(out, current_dataset_id)

    st.subheader("SQL")
    st.code(out.get("sql", ""), language="sql")

    if analyze_all_requested and not analyze_all_applied:
        st.info("Analyze-all was requested, but this query was treated as row-level so limit mode was used for safety.")

    if rows:
        df = _prepare_result_dataframe_for_display(_to_display_dataframe(rows), route_intent)
        if route_intent == "semantic_role_sheet_summary":
            st.caption(
                "This table shows one summary row per sheet, including source-sheet row count and detected semantic roles."
            )
            view_df = _render_table_with_column_controls(
                df,
                scope_key=f"{current_dataset_id}_semantic_role_summary_table",
                hide_sheet_name_default=False,
            )
            st.subheader("Dashboard")
            _render_dashboard(df, scope_key=f"{current_dataset_id}_semantic_role_summary", result_payload=out)
        elif query_shape in {"aggregate_with_filters", "aggregate_scalar"} and len(df) == 1:
            st.caption("Filtered aggregate result. The table includes applied filters and the matched row count.")
            _render_table_with_column_controls(
                df,
                scope_key=f"{current_dataset_id}_aggregate_with_filters_table",
                hide_sheet_name_default=False,
                required_columns=list(df.columns),
            )
        elif "sheet_name" in df.columns and df["sheet_name"].notna().any():
            sheet_series = df["sheet_name"].astype(str)
            sheets = sorted(sheet_series.unique().tolist())

            if len(sheets) > 1:
                st.subheader("Data by sheet")
                labels = [
                    f"{s} ({int((sheet_series == s).sum())} result row{'' if int((sheet_series == s).sum()) == 1 else 's'})"
                    for s in sheets
                ]
                tabs = st.tabs(labels)
                for i, sheet in enumerate(sheets):
                    with tabs[i]:
                        sdf = df[sheet_series == sheet].copy()
                        safe_sheet = re.sub(r"[^a-zA-Z0-9_]+", "_", str(sheet)).strip("_") or "sheet"
                        dashboard_scope = f"{current_dataset_id}_{safe_sheet}"
                        view_sdf = _render_table_with_column_controls(
                            sdf,
                            scope_key=f"{dashboard_scope}_table",
                            hide_sheet_name_default=True,
                        )
                        _render_dashboard(sdf, scope_key=dashboard_scope, result_payload=out)

                with st.expander("Combined view", expanded=False):
                    view_df = _render_table_with_column_controls(
                        df,
                        scope_key=f"{current_dataset_id}_combined_table",
                        hide_sheet_name_default=False,
                    )
                    dashboard_scope = str(current_dataset_id or "default")
                    _render_dashboard(df, scope_key=f"{dashboard_scope}_combined", result_payload=out)
            else:
                only_sheet = sheets[0] if sheets else "sheet"
                safe_sheet = re.sub(r"[^a-zA-Z0-9_]+", "_", str(only_sheet)).strip("_") or "sheet"
                dashboard_scope = f"{current_dataset_id}_{safe_sheet}"
                view_df = _render_table_with_column_controls(
                    df,
                    scope_key=f"{dashboard_scope}_table",
                    hide_sheet_name_default=True,
                    required_columns=[c for c in ["product", "region", "sum_total_price", "matched_rows"] if c in df.columns],
                )
                _render_dashboard(df, scope_key=dashboard_scope, result_payload=out)
        else:
            dashboard_scope = str(current_dataset_id or "default")
            view_df = _render_table_with_column_controls(
                df,
                scope_key=f"{dashboard_scope}_table",
                hide_sheet_name_default=False,
                required_columns=[c for c in ["product", "region", "sum_total_price", "matched_rows"] if c in df.columns],
            )
            _render_dashboard(df, scope_key=dashboard_scope, result_payload=out)
    else:
        st.info("Query returned no rows.")
else:
    st.info("No saved result for this dataset yet. Run SQL analysis to display the table and dashboard.")
