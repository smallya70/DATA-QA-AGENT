from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from io import BytesIO
import os, re, json, requests
import importlib
from typing import Any
import pandas as pd
from openai import OpenAI

create_client = None
try:
    supabase_module = importlib.import_module("supabase")
    create_client = getattr(supabase_module, "create_client", None)
except Exception:
    create_client = None

load_dotenv()

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")  # safe default
oa = OpenAI(api_key=OPENAI_API_KEY)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # service role
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env")

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

# -------------------------
# APP
# -------------------------
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
                "trace": traceback.format_exc()[:4000],  # trim
                "path": str(request.url),
            },
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
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

# -------------------------
# UPLOAD ENDPOINT
# -------------------------
@app.post("/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_name: str = Form(default="")
):
    content = await file.read()
    filename = file.filename or "uploaded.csv"

    try:
        df = pd.read_csv(BytesIO(content)) if filename.lower().endswith(".csv") else pd.read_excel(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    df = df.fillna("")
    cols = list(df.columns)
    row_count, col_count = int(df.shape[0]), int(df.shape[1])

    name = dataset_name.strip() or os.path.splitext(filename)[0]

    # 1) Insert dataset metadata
    if sb is not None:
        ds_res = sb.table("datasets").insert({
            "name": name,
            "original_filename": filename,
            "row_count": row_count,
            "col_count": col_count,
            "columns": cols
        }).execute()
        dataset_id = ds_res.data[0]["id"]
    else:
        ds_rows = _rest_request(
            "POST",
            "datasets",
            payload={
                "name": name,
                "original_filename": filename,
                "row_count": row_count,
                "col_count": col_count,
                "columns": cols,
            },
            prefer="return=representation",
        )
        if not isinstance(ds_rows, list) or not ds_rows:
            raise HTTPException(status_code=502, detail="Supabase returned no dataset row after insert")
        dataset_id = ds_rows[0]["id"]

    # 2) Insert rows in chunks
    payload = []
    for i in range(row_count):
        row_dict = {c: _clean(df.iloc[i][c]) for c in cols}
        payload.append({
            "dataset_id": dataset_id,
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

    return {
        "dataset_id": dataset_id,
        "name": name,
        "filename": filename,
        "rows": row_count,
        "columns": cols
    }

# -------------------------
# LIST DATASETS
# -------------------------
@app.get("/datasets")
def list_datasets():
    if sb is not None:
        res = sb.table("datasets").select("id,name,row_count,col_count,columns,created_at").order("created_at", desc=True).execute()
        return res.data

    return _rest_request(
        "GET",
        "datasets",
        params={"select": "id,name,row_count,col_count,columns,created_at", "order": "created_at.desc"},
    )

# -------------------------
# OPTION 1: ASK (ROW SAMPLING)
# -------------------------
class AskBody(BaseModel):
    dataset_id: str
    question: str
    max_rows: int = 60


@app.post("/ask")
def ask(body: AskBody):
    dataset_id = (body.dataset_id or "").strip()
    q = (body.question or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id missing")
    if not q:
        raise HTTPException(status_code=400, detail="question missing")

    max_rows = min(max(int(body.max_rows), 10), 200)

    try:
        if sb is not None:
            meta_res = (
                sb.table("datasets")
                .select("id,name,row_count,col_count,columns")
                .eq("id", dataset_id)
                .limit(1)
                .execute()
            )
            meta_rows = meta_res.data or []

            rows_res = (
                sb.table("dataset_rows")
                .select("row_index,row_text")
                .eq("dataset_id", dataset_id)
                .order("row_index")
                .limit(max_rows)
                .execute()
            )
            rows = rows_res.data or []
        else:
            meta_rows = _rest_request(
                "GET",
                "datasets",
                params={"select": "id,name,row_count,col_count,columns", "id": f"eq.{dataset_id}", "limit": 1},
            )
            rows = _rest_request(
                "GET",
                "dataset_rows",
                params={
                    "select": "row_index,row_text",
                    "dataset_id": f"eq.{dataset_id}",
                    "order": "row_index.asc",
                    "limit": max_rows,
                },
            ) or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not isinstance(meta_rows, list) or not meta_rows or not isinstance(meta_rows[0], dict):
        raise HTTPException(status_code=404, detail="Dataset not found")
    meta = meta_rows[0]

    if not isinstance(rows, list):
        rows = []
    row_dicts = [r for r in rows if isinstance(r, dict)]
    if not row_dicts:
        raise HTTPException(status_code=404, detail="No rows found for this dataset_id")

    context = "\n".join([f"Row {r.get('row_index')}: {r.get('row_text') or ''}" for r in row_dicts])

    system = (
        "You are a data analyst. Answer using ONLY the provided rows. "
        "If you can't answer from these rows, say what extra filter/rows are needed."
    )
    user_msg = (
        f"Dataset: {meta.get('name')} | rows={meta.get('row_count')} | cols={meta.get('col_count')}\n"
        f"Columns: {meta.get('columns')}\n\n"
        f"Question: {q}\n\nRows:\n{context}"
    )

    try:
        base_args = {
            "model": TEXT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        }

        try:
            resp = oa.chat.completions.create(**base_args, max_completion_tokens=700)
        except Exception as token_err:
            if "max_completion_tokens" in str(token_err).lower() and "unsupported" in str(token_err).lower():
                resp = oa.chat.completions.create(**base_args, max_tokens=700)
            else:
                raise

        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "rows_used": len(row_dicts)}