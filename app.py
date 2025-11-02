import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page + minimalist styling
# -----------------------------
st.set_page_config(
    page_title="DailyDeck",
    page_icon="📊",
    layout="wide",
)

HIDE_DEFAULT_SIDEBAR = """
<style>
/* Hide Streamlit's hamburger & footer for a cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}
/* Center the uploader section and make it prominent */
.upload-card {
    border: 1px solid #eaeaea;
    border-radius: 14px;
    padding: 24px 28px;
    background: #ffffff;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
}
.upload-title {
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 0.35rem;
}
.small-muted {
    font-size: 0.92rem;
    color: #6b7280;
}
</style>
"""
st.markdown(HIDE_DEFAULT_SIDEBAR, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def _read_csv_like(file_like, parse_dates=None) -> pd.DataFrame:
    """Read CSV or gz CSV from a file-like object."""
    return pd.read_csv(
        file_like,
        low_memory=False,
        parse_dates=parse_dates or [],
        infer_datetime_format=True,
        dayfirst=False,
        encoding="utf-8",
        # If your files sometimes have thousands separators in numeric cols,
        # you can add: thousands=","
    )

def load_uploaded(file, parse_date_cols=("TRN_DATE",)):
    """
    Accepts .csv, .gz or .zip (one or many CSVs inside).
    Returns a single concatenated DataFrame.
    """
    name = file.name.lower()

    # Parse date columns (if they exist)
    parse_dates_present = []
    if parse_date_cols:
        # We'll pass these names to pandas only if they exist
        # (avoids errors when a column is missing).
        # We'll detect existence after reading a sample.
        pass

    if name.endswith(".csv") or name.endswith(".gz"):
        df = _read_csv_like(file, parse_dates=None)  # do not predeclare: detect later
        # Apply date parsing only for columns that actually exist
        for c in parse_date_cols or []:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    if name.endswith(".zip"):
        # Collect all CSV files within the zip
        dataframes = []
        with zipfile.ZipFile(file) as z:
            members = [m for m in z.namelist() if m.lower().endswith((".csv", ".gz"))]
            if not members:
                st.warning("The ZIP has no CSV files.")
                return pd.DataFrame()
            for m in members:
                with z.open(m, "r") as f:
                    # If member is .gz, wrap in BytesIO directly, pandas can handle compression='infer'
                    by = io.BytesIO(f.read())
                    df_part = pd.read_csv(by, low_memory=False)
                    # Parse dates if present
                    for c in parse_date_cols or []:
                        if c in df_part.columns:
                            df_part[c] = pd.to_datetime(df_part[c], errors="coerce")
                    dataframes.append(df_part)
        return pd.concat(dataframes, ignore_index=True)

    st.error("Unsupported file type. Please upload a .csv, .gz, or .zip containing CSV(s).")
    return pd.DataFrame()


def app_header():
    st.markdown(
        """
        <div class="upload-card">
          <div class="upload-title">Upload CSV / ZIP (CSV or GZ inside) to begin</div>
          <div class="small-muted">
            Expected columns (when present): <code>STORE_NAME</code>, <code>CUST_CODE</code>, <code>SUPPLIER_NAME</code>,
            <code>CATEGORY</code>, <code>DEPARTMENT</code>, <code>ITEM_CODE</code>, <code>ITEM_NAME</code>,
            <code>TRN_DATE</code>, <code>QTY</code>, <code>NET_SALES</code>, <code>SP_PRE_VAT</code>, <code>LOYALTY_CUSTOMER_CODE</code>, <code>CASHIER_NAME</code>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# UI: Single clean uploader
# -----------------------------
app_header()
uploaded = st.file_uploader(
    " ",
    type=["csv", "zip", "gz"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    help="Drag & drop a CSV, GZ, or a ZIP containing one or more CSV files (up to 400 MB when self-hosted; 200 MB on Streamlit Cloud).",
)

if not uploaded:
    st.info("Drop a file above to start. (For 400 MB uploads, self-host and set `server.maxUploadSize=400`.)")
    st.stop()

with st.spinner("Reading your data…"):
    df = load_uploaded(uploaded, parse_date_cols=("TRN_DATE",))
    if df.empty:
        st.warning("No rows found after reading the upload.")
        st.stop()

# Normalize some likely numeric cols
for col in ["QTY", "NET_SALES", "SP_PRE_VAT"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Strip text cols
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# -----------------------------
# Your analytics tabs
# (Plug the code you already have into each tab)
# -----------------------------
tabs = st.tabs([
    "📒 Notebook Preview",
    "🛒 Supplier Basket Share",
    "📊 Category Contributions",
    "🚫 Negative Receipts",
    "🏷️ Multi-priced SKUs",
    "💚 Loyalty Overview",
])

# -------- Tab 1: Preview --------
with tabs[0]:
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

# -------- Tab 2: Supplier Basket Share --------
with tabs[1]:
    st.subheader("Supplier Basket Share")
    # 👉 Paste the final version of your single-dropdown supplier basket share code here.
    # It expects df with at least: STORE_NAME, CUST_CODE, SUPPLIER_NAME, CATEGORY, DEPARTMENT
    st.caption("Hook your existing supplier share widget here.")

# -------- Tab 3: Category Contributions --------
with tabs[2]:
    st.subheader("Category Contributions")
    # 👉 Paste your category contribution (by NET_SALES or baskets) code here.
    st.caption("Hook your existing category contribution visuals here.")

# -------- Tab 4: Negative Receipts --------
with tabs[3]:
    st.subheader("Negative Receipts")
    # 👉 Paste your 'negative receipts overview' + branch drilldown code here.
    st.caption("Hook your negative receipts table / charts here.")

# -------- Tab 5: Multi-priced SKUs --------
with tabs[4]:
    st.subheader("Multi-priced SKUs (per day)")
    # 👉 Paste your high-level store summary and detailed product drilldown code here.
    st.caption("Hook your multi-price summary & detail views here.")

# -------- Tab 6: Loyalty Overview --------
with tabs[5]:
    st.subheader("Loyalty Overview")
    # 👉 Paste your loyalty KPIs + per-store visuals here.
    st.caption("Hook your loyalty overview here.")

st.success("Loaded and ready. Explore the tabs above.")
