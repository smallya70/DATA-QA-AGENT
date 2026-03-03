import streamlit as st
import pandas as pd
import requests

BACKEND = "http://127.0.0.1:8092"  # change if needed

st.set_page_config(page_title="Data QA Agent (Option 1)", layout="wide")
st.title("📊 Data QA Agent — Option 1 (Row Sampling)")

# -------------------------
# Helper: fetch datasets
# -------------------------
def fetch_datasets():
    try:
        r = requests.get(f"{BACKEND}/datasets", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend not reachable: {e}")
        return []

# -------------------------
# Upload
# -------------------------
st.header("1) Upload a dataset")
uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
ds_name = st.text_input("Dataset name (optional)", "")

if st.button("Upload to Backend", disabled=uploaded is None):
    if uploaded is None:
        st.warning("Upload a file first.")
    else:
        with st.spinner("Uploading and saving rows to Supabase..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            data = {"dataset_name": ds_name}
            try:
                res = requests.post(f"{BACKEND}/upload_csv", files=files, data=data, timeout=300)
                res.raise_for_status()
                out = res.json()
                st.success(f"Uploaded: {out['name']} ({out['rows']} rows × {len(out['columns'])} cols)")
                st.session_state["selected_dataset_id"] = out["dataset_id"]
            except Exception as e:
                st.error(f"Upload failed: {e}")

st.divider()

# -------------------------
# Dataset Picker
# -------------------------
st.header("2) Select a dataset")
datasets = fetch_datasets()
if not datasets:
    st.info("No datasets yet. Upload one above.")
else:
    options = {f"{d['name']}  —  {d['row_count']} rows": d["id"] for d in datasets}
    default_label = None
    if "selected_dataset_id" in st.session_state:
        # find matching label
        for label, did in options.items():
            if did == st.session_state["selected_dataset_id"]:
                default_label = label
                break

    selected_label = st.selectbox(
        "Choose a dataset",
        list(options.keys()),
        index=(list(options.keys()).index(default_label) if default_label in options else 0),
    )
    dataset_id = options[selected_label]
    st.session_state["selected_dataset_id"] = dataset_id

st.divider()

# -------------------------
# Ask AI (Option 1)
# -------------------------
st.header("3) Ask AI (Row Sampling)")
if "selected_dataset_id" not in st.session_state:
    st.warning("Select or upload a dataset first.")
else:
    q = st.text_input("Your question", placeholder="e.g., How many states are there in the sample?")
    max_rows = st.slider("Rows to sample for AI", min_value=10, max_value=80, value=60, step=10)

    st.caption("⚠️ Option 1 answers are based only on sampled rows (not full dataset).")

    if st.button("Ask AI", disabled=(not q.strip())):
        with st.spinner("Thinking..."):
            try:
                payload = {"dataset_id": st.session_state["selected_dataset_id"], "question": q, "max_rows": max_rows}
                r = requests.post(f"{BACKEND}/ask", json=payload, timeout=120)
                r.raise_for_status()
                out = r.json()
                st.success(f"Answered using {out.get('rows_used', 0)} rows")
                st.write(out.get("answer", ""))
            except Exception as e:
                st.error(f"Ask failed: {e}")