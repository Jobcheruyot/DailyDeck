# -------------------------------------------------------
# BranchBoard — Streamlit App
# - Big CSV uploads (pyarrow, optional chunked read)
# - Notebook Markdown Preview (hides markdown starting with "*")
# - Interactive retail analytics (Plotly)
# -------------------------------------------------------
# How to run:   streamlit run app.py
# -------------------------------------------------------

import os
import io
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page & global styling
# -----------------------------
st.set_page_config(
    page_title="BranchBoard • Retail Analytics",
    page_icon="🧺",
    layout="wide"
)

# Subtle “wow” styling
st.markdown("""
<style>
/* reduce top padding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
/* sticky filters bar */
.sticky-filters {position: sticky; top: 0; z-index: 999; background: white; padding: .5rem 0 .75rem; border-bottom: 1px solid #eee;}
/* metric badges */
.badge {display:inline-block; padding: .25rem .6rem; border-radius: 999px; background:#f5f7ff; color:#335; font-weight: 600; margin-right: .35rem; border: 1px solid #e3e7ff;}
/* table tweaks */
thead th {text-transform: uppercase; letter-spacing: .02em; font-size:.82rem;}
/* headers */
h1, h2, h3 {margin-top:.6rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers & cache
# -----------------------------

@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes, sep: str, decimal: str, encoding: str, parse_dates_cols):
    # Use pyarrow when possible for speed + memory
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            low_memory=False,
            engine="pyarrow"
        )
    except Exception:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            low_memory=False
        )
    # optional parse
    for col in parse_dates_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def read_many(files, sep, decimal, encoding, parse_dates_cols):
    frames = []
    for f in files:
        frames.append(load_csv_bytes(f.getbuffer(), sep, decimal, encoding, parse_dates_cols))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return x


def fmt_2(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return x


def to_stripped_str(s):
    return s.astype(str).str.strip()


# -----------------------------
# Sidebar: Data & notebook
# -----------------------------
st.sidebar.title("📥 Data & Notebook")

with st.sidebar.expander("Upload CSV(s) — up to ~400MB", expanded=True):
    sep = st.selectbox("Separator", [",", ";", "|", "\t"], index=0)
    decimal = st.selectbox("Decimal marker", [".", ","], index=0)
    encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
    parse_dates_cols = st.multiselect("Parse as datetimes (optional)", ["TRN_DATE"], default=["TRN_DATE"])
    files = st.file_uploader(
        "Drop one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )
    st.caption("Tip: You can also upload a zipped CSV (optional) in the widget below.")

with st.sidebar.expander("Upload ZIP of CSV(s) (optional)"):
    zip_file = st.file_uploader("ZIP file", type=["zip"])
    if zip_file is not None:
        with zipfile.ZipFile(zip_file) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            dfs = []
            for m in members:
                dfs.append(load_csv_bytes(z.read(m), sep, decimal, encoding, parse_dates_cols))
            if dfs:
                files_df = pd.concat(dfs, ignore_index=True)
                st.session_state["zip_df"] = files_df

# Load dataframe
df = pd.DataFrame()
src_pieces = []
if files:
    df = read_many(files, sep, decimal, encoding, parse_dates_cols)
    src_pieces.append(f"{len(files)} file(s)")
if "zip_df" in st.session_state:
    df_zip = st.session_state["zip_df"]
    df = pd.concat([df, df_zip], ignore_index=True) if not df.empty else df_zip
    src_pieces.append("ZIP")

if df.empty:
    st.info("Upload CSVs to begin. Columns the app uses when present: "
            "`STORE_NAME`, `CUST_CODE`, `SUPPLIER_NAME`, `CATEGORY`, `DEPARTMENT`, "
            "`NET_SALES`, `QTY`, `ITEM_CODE`, `ITEM_NAME`, `TRN_DATE`, "
            "`CAP_CUSTOMER_CODE`, `CASHIER_NAME`, `SP_PRE_VAT`, `LOYALTY_CUSTOMER_CODE`.")
else:
    st.sidebar.success(f"Loaded {fmt_int(len(df))} rows from {' + '.join(src_pieces)}")

# Optional notebook
with st.sidebar.expander("📓 Notebook markdown (optional)"):
    nb_file = st.file_uploader("Upload .ipynb to render markdown (hides lines starting with '*')",
                               type=["ipynb"], key="nb_upl")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "📓 Notebook Preview",
    "🧺 Supplier Basket Share",
    "📊 Category Contributions",
    "🔴 Negative Receipts",
    "🏷️ Multi-priced SKUs",
    "💚 Loyalty Overview"
])

# =========================================================
# 1) Notebook markdown preview (no execution, hide '*' md)
# =========================================================
with tabs[0]:
    st.header("📓 Notebook Markdown (read-only)")
    if st.session_state.get("nb_upl") is None:
        st.caption("Upload a .ipynb in the sidebar to preview markdown. "
                   "Any markdown cell that **starts with ‘*’** is hidden.")
    else:
        try:
            nb_json = json.load(st.session_state["nb_upl"])
            cells = nb_json.get("cells", [])
            shown = 0
            for c in cells:
                if c.get("cell_type") != "markdown":
                    continue
                src = "".join(c.get("source", []))
                if src.strip().startswith("*"):   # hide compute-only sections
                    continue
                st.markdown(src)
                shown += 1
            if shown == 0:
                st.warning("No markdown cells to display (or all start with '*').")
        except Exception as e:
            st.error(f"Could not read notebook: {e}")

# guard if no data
if df.empty:
    st.stop()

# Normalize common columns used in the app
for col in ["STORE_NAME","CUST_CODE","SUPPLIER_NAME","CATEGORY","DEPARTMENT",
            "CAP_CUSTOMER_CODE","CASHIER_NAME","ITEM_CODE","ITEM_NAME",
            "LOYALTY_CUSTOMER_CODE"]:
    if col in df.columns:
        df[col] = to_stripped_str(df[col])

# ============================================
# 2) Supplier Basket Share (Cat → Dept → Branch)
# ============================================
with tabs[1]:
    st.header("🧺 Supplier share of baskets")
    # Build scope filters row
    cols = st.columns([1.1, 1, 1])
    with cols[0]:
        cat_opts = ["ALL"] + sorted(df["CATEGORY"].dropna().unique()) if "CATEGORY" in df else ["ALL"]
        cat_val = st.selectbox("Category", cat_opts, key="sup_cat")
    with cols[1]:
        ddf = df if cat_val=="ALL" or "CATEGORY" not in df else df[df["CATEGORY"]==cat_val]
        dept_opts = ["ALL"] + sorted(ddf["DEPARTMENT"].dropna().unique()) if "DEPARTMENT" in ddf else ["ALL"]
        dept_val = st.selectbox("Department", dept_opts, key="sup_dept")
    with cols[2]:
        bdf = ddf if dept_val=="ALL" or "DEPARTMENT" not in ddf else ddf[ddf["DEPARTMENT"]==dept_val]
        branch_opts = ["ALL"] + sorted(bdf["STORE_NAME"].dropna().unique()) if "STORE_NAME" in bdf else ["ALL"]
        branch_val = st.selectbox("Branch", branch_opts, key="sup_branch")

    scope = df.copy()
    if "CATEGORY" in scope and cat_val != "ALL":
        scope = scope[scope["CATEGORY"] == cat_val]
    if "DEPARTMENT" in scope and dept_val != "ALL":
        scope = scope[scope["DEPARTMENT"] == dept_val]
    if "STORE_NAME" in scope and branch_val != "ALL":
        scope = scope[scope["STORE_NAME"] == branch_val]

    # Denominator: unique baskets in scope
    denom = scope[["CUST_CODE"]].drop_duplicates() if "CUST_CODE" in scope else pd.DataFrame()
    total_baskets = denom["CUST_CODE"].nunique() if "CUST_CODE" in denom else 0

    if total_baskets == 0 or "SUPPLIER_NAME" not in scope:
        st.warning("No baskets or missing columns for this view.")
    else:
        sup_tbl = (
            scope[["SUPPLIER_NAME","CUST_CODE"]]
            .drop_duplicates()
            .groupby("SUPPLIER_NAME", as_index=False)["CUST_CODE"]
            .nunique()
            .rename(columns={"CUST_CODE":"Baskets_With_Supplier"})
        )
        sup_tbl["Supplier_Share_%"] = (sup_tbl["Baskets_With_Supplier"]/total_baskets*100).round(2)
        sup_tbl = sup_tbl.sort_values("Supplier_Share_%", ascending=False).reset_index(drop=True)
        sup_tbl.insert(0, "#", range(1, len(sup_tbl)+1))

        st.caption(
            f"<span class='badge'>Category: {cat_val}</span>"
            f"<span class='badge'>Department: {dept_val}</span>"
            f"<span class='badge'>Branch: {branch_val}</span>"
            f"<span class='badge'>Denominator baskets: {fmt_int(total_baskets)}</span>",
            unsafe_allow_html=True
        )
        show = sup_tbl.copy()
        show["Baskets_With_Supplier"] = show["Baskets_With_Supplier"].map(fmt_int)
        show["Supplier_Share_%"] = show["Supplier_Share_%"].map(lambda v: f"{v:.2f}%")
        st.dataframe(show[["#","SUPPLIER_NAME","Baskets_With_Supplier","Supplier_Share_%"]],
                     use_container_width=True, hide_index=True)

        chart = sup_tbl.copy()
        fig = px.bar(
            chart,
            x="Supplier_Share_%",
            y="SUPPLIER_NAME",
            orientation="h",
            text=chart["Supplier_Share_%"].map(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=["#1f77b4"]
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(height=max(500, 22*len(chart)), margin=dict(l=260, r=30, t=20, b=20),
                          xaxis_title="% of baskets", yaxis_title="Supplier", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ======================================
# 3) Category Contributions (value/bags)
# ======================================
with tabs[2]:
    st.header("📊 Category contributions")
    if "STORE_NAME" not in df or "CATEGORY" not in df or ("NET_SALES" not in df and "CUST_CODE" not in df):
        st.warning("Need columns: STORE_NAME, CATEGORY, and either NET_SALES or CUST_CODE.")
    else:
        mode = st.radio("Metric", ["NET_SALES (value)", "BASKETS (count)"], horizontal=True, index=0)

        # Aggregate
        if mode.startswith("NET_SALES") and "NET_SALES" in df:
            agg = (df.groupby(["STORE_NAME","CATEGORY"], as_index=False)["NET_SALES"].sum())
            pivot = agg.pivot(index="STORE_NAME", columns="CATEGORY", values="NET_SALES").fillna(0)
            # convert to %
            pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)*100
        else:
            if "CUST_CODE" not in df:
                st.warning("CUST_CODE is missing.")
                st.stop()
            temp = df[["STORE_NAME","CATEGORY","CUST_CODE"]].drop_duplicates()
            agg = temp.groupby(["STORE_NAME","CATEGORY"], as_index=False)["CUST_CODE"].nunique()
            pivot = agg.pivot(index="STORE_NAME", columns="CATEGORY", values="CUST_CODE").fillna(0)
            pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)*100

        # remove NaN column label if any
        if pivot.columns.dtype == "object":
            pivot = pivot[[c for c in pivot.columns if str(c).lower() != "nan"]]

        # add TOTAL row (blank here, total is 100% per row; we’ll just show row-wise)
        pivot = pivot.sort_index()  # alphabetical branches
        show = pivot.copy().round(2)
        show.insert(0, "#", range(1, len(show)+1))
        st.dataframe(show.style.format("{:.2f}%", na_rep="-"),
                     use_container_width=True)

        # Stacked horizontal
        chart_df = pivot.reset_index().melt(id_vars="STORE_NAME", var_name="CATEGORY", value_name="PCT")
        fig = px.bar(
            chart_df.sort_values("STORE_NAME"),
            x="PCT", y="STORE_NAME", color="CATEGORY",
            orientation="h",
            title="Category share by branch (percent)"
        )
        fig.update_layout(barmode="stack", xaxis_title="% of branch", yaxis_title="Branch",
                          margin=dict(l=200, r=30, t=40, b=20), legend_title_text="Category")
        fig.update_xaxes(tickformat=".2f")
        st.plotly_chart(fig, use_container_width=True)

# ======================================
# 4) Negative Receipts (overview + drill)
# ======================================
with tabs[3]:
    st.header("🔴 Negative receipts")
    need_cols = {"STORE_NAME","CUST_CODE","NET_SALES","CAP_CUSTOMER_CODE"}
    if not need_cols.issubset(df.columns):
        st.warning(f"Need columns: {', '.join(sorted(need_cols))}.")
    else:
        # Classify CAP_CUSTOMER_CODE
        cap = df.copy()
        cap["CAP_TYPE"] = np.where(cap["CAP_CUSTOMER_CODE"].str.strip().eq(""), "General sales", "On_account sales")

        # Dedup to receipt level (sum)
        rcp = (cap.groupby(["STORE_NAME","CUST_CODE","CAP_TYPE"], as_index=False)["NET_SALES"].sum())
        neg = rcp[rcp["NET_SALES"] < 0].copy()

        # Overview: choose rank basis
        basis = st.radio("Rank by", ["Count of Receipts", "Total Negative Value (abs)"], horizontal=True)
        ov = (neg.groupby(["STORE_NAME","CAP_TYPE"], as_index=False)
              .agg(Receipts=("CUST_CODE","nunique"),
                   Neg_Value=("NET_SALES","sum")))
        ov["Neg_Value"] = ov["Neg_Value"].abs()

        if basis.startswith("Count"):
            ov = ov.sort_values(["Receipts","Neg_Value"], ascending=[False, False])
        else:
            ov = ov.sort_values(["Neg_Value","Receipts"], ascending=[False, False])

        ov.insert(0, "#", range(1, len(ov)+1))
        show = ov.copy()
        show["Receipts"] = show["Receipts"].map(fmt_int)
        show["Neg_Value"] = show["Neg_Value"].map(fmt_2)
        st.subheader("Overview (by store & CAP type)")
        st.dataframe(show[["#","STORE_NAME","CAP_TYPE","Receipts","Neg_Value"]],
                     use_container_width=True, hide_index=True)

        # Drill: branch selector
        st.markdown("---")
        st.subheader("Branch drill-down (unique receipt list)")
        bsel = st.selectbox("Branch", sorted(neg["STORE_NAME"].unique()))
        bdata = neg[neg["STORE_NAME"] == bsel].copy()
        # Dedup to one line per receipt (it already is)
        # Attach time/cashier if present: derive from original df
        extra_cols = ["TRN_DATE","CASHIER_NAME"]
        base = df[df["STORE_NAME"]==bsel][["CUST_CODE"]+ [c for c in extra_cols if c in df.columns]].drop_duplicates("CUST_CODE")
        out = bdata.merge(base, on="CUST_CODE", how="left")

        tot = out["NET_SALES"].sum()
        st.caption(f"Total negative value: **{fmt_2(abs(tot))}**")

        out_show = out.copy()
        if "TRN_DATE" in out_show:
            out_show["TRN_DATE"] = pd.to_datetime(out_show["TRN_DATE"], errors="coerce")
        out_show = out_show.sort_values("NET_SALES")
        out_show["NET_SALES"] = out_show["NET_SALES"].map(fmt_2)
        st.dataframe(out_show[["STORE_NAME","CUST_CODE","CAP_TYPE"] +
                               ([ "TRN_DATE"] if "TRN_DATE" in out_show else []) +
                               ([ "CASHIER_NAME"] if "CASHIER_NAME" in out_show else []) +
                               ["NET_SALES"]],
                     use_container_width=True, hide_index=True)

# ======================================
# 5) Multi-priced SKUs per day (summary & detail)
# ======================================
with tabs[4]:
    st.header("🏷️ Items sold at >1 price within a day")
    need_cols = {"STORE_NAME","TRN_DATE","ITEM_CODE","ITEM_NAME","QTY","SP_PRE_VAT"}
    if not need_cols.issubset(df.columns):
        st.warning(f"Need columns: {', '.join(sorted(need_cols))}.")
    else:
        dp = df.copy()
        dp["TRN_DATE"] = pd.to_datetime(dp["TRN_DATE"], errors="coerce")
        dp = dp.dropna(subset=["TRN_DATE"])
        dp["SP_PRE_VAT"] = (dp["SP_PRE_VAT"].astype(str).str.replace(",","",regex=False).str.strip())
        dp["SP_PRE_VAT"] = pd.to_numeric(dp["SP_PRE_VAT"], errors="coerce").fillna(0)
        dp["QTY"] = pd.to_numeric(dp["QTY"], errors="coerce").fillna(0)
        dp["DATE"] = dp["TRN_DATE"].dt.date

        grp = (dp.groupby(["STORE_NAME","DATE","ITEM_CODE","ITEM_NAME"], as_index=False)
                 .agg(Num_Prices=("SP_PRE_VAT", lambda s: s.dropna().nunique()),
                      Price_Min=("SP_PRE_VAT","min"),
                      Price_Max=("SP_PRE_VAT","max"),
                      Total_QTY=("QTY","sum")))
        mp = grp[grp["Num_Prices"] > 1].copy()
        mp["Price_Spread"] = (mp["Price_Max"] - mp["Price_Min"]).round(2)
        mp = mp[mp["Price_Spread"] > 0]
        mp["Diff_Value"] = (mp["Total_QTY"] * mp["Price_Spread"]).round(2)

        # Summary per store (include stores with none = 0)
        stores = sorted(dp["STORE_NAME"].dropna().unique())
        summary = (mp.groupby("STORE_NAME", as_index=False)
                     .agg(Items_with_MultiPrice=("ITEM_CODE","nunique"),
                          Total_Diff_Value=("Diff_Value","sum"),
                          Avg_Spread=("Price_Spread","mean"),
                          Max_Spread=("Price_Spread","max")))
        # fill zeros for missing stores
        miss = [s for s in stores if s not in set(summary["STORE_NAME"])]
        if miss:
            z = pd.DataFrame({"STORE_NAME": miss,
                              "Items_with_MultiPrice": [0]*len(miss),
                              "Total_Diff_Value": [0.0]*len(miss),
                              "Avg_Spread": [0.0]*len(miss),
                              "Max_Spread": [0.0]*len(miss)})
            summary = pd.concat([summary, z], ignore_index=True)

        summary = summary.sort_values("Total_Diff_Value", ascending=False).reset_index(drop=True)
        summary.insert(0, "#", range(1, len(summary)+1))

        tot_row = pd.DataFrame([{
            "#": "",
            "STORE_NAME": "TOTAL",
            "Items_with_MultiPrice": int(summary["Items_with_MultiPrice"].sum()),
            "Total_Diff_Value": float(summary["Total_Diff_Value"].sum()),
            "Avg_Spread": float(summary["Avg_Spread"].max()),
            "Max_Spread": float(summary["Max_Spread"].max())
        }])
        summary_total = pd.concat([summary, tot_row], ignore_index=True)
        show = summary_total.copy()
        show["Items_with_MultiPrice"] = show["Items_with_MultiPrice"].map(lambda x: fmt_int(x) if str(x).isdigit() or isinstance(x,int) else x)
        for c in ["Total_Diff_Value","Avg_Spread","Max_Spread"]:
            show[c] = show[c].map(lambda x: fmt_2(x) if isinstance(x,(int,float)) else x)

        st.subheader("High-level summary (per store)")
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Detail — pick a branch to see its multi-priced SKUs")
        bsel = st.selectbox("Branch", stores, key="mp_branch")
        detail = mp[mp["STORE_NAME"]==bsel].copy()
        detail = detail.sort_values("Diff_Value", ascending=False).reset_index(drop=True)
        if detail.empty:
            st.info("No multi-priced items in this branch.")
        else:
            dshow = detail.copy()
            dshow["Total_QTY"] = dshow["Total_QTY"].map(fmt_int)
            for c in ["Price_Min","Price_Max","Price_Spread","Diff_Value"]:
                dshow[c] = dshow[c].map(fmt_2)
            st.dataframe(dshow, use_container_width=True, hide_index=True)

# ======================================
# 6) Loyalty Overview
# ======================================
with tabs[5]:
    st.header("💚 Loyalty overview")
    need_cols = {"STORE_NAME","CUST_CODE","LOYALTY_CUSTOMER_CODE","NET_SALES","TRN_DATE"}
    if not need_cols.issubset(df.columns):
        st.warning(f"Need columns: {', '.join(sorted(need_cols))}.")
    else:
        d = df.copy()
        d["TRN_DATE"] = pd.to_datetime(d["TRN_DATE"], errors="coerce")
        d = d.dropna(subset=["TRN_DATE"])
        d["Is_Loyal"] = np.where(d["LOYALTY_CUSTOMER_CODE"].str.strip().eq(""), "Non-Loyalty", "Loyalty")

        receipts = (d.groupby(["STORE_NAME","CUST_CODE","Is_Loyal","LOYALTY_CUSTOMER_CODE"], as_index=False)
                      .agg(Basket_Value=("NET_SALES","sum"),
                           First_Time=("TRN_DATE","min")))

        split_counts = (receipts.groupby(["STORE_NAME","Is_Loyal"], as_index=False)
                         .agg(Receipts=("CUST_CODE","nunique")))
        piv = split_counts.pivot(index="STORE_NAME", columns="Is_Loyal", values="Receipts").fillna(0)
        for c in ["Loyalty","Non-Loyalty"]:
            if c not in piv.columns: piv[c]=0
        piv["Total_Receipts"] = piv["Loyalty"] + piv["Non-Loyalty"]
        piv["Loyalty_%"] = np.where(piv["Total_Receipts"]>0, (piv["Loyalty"]/piv["Total_Receipts"]*100).round(1), 0.0)

        avg_basket = (receipts.groupby(["STORE_NAME","Is_Loyal"], as_index=False)
                      .agg(Avg_Basket_Value=("Basket_Value","mean")))
        ap = avg_basket.pivot(index="STORE_NAME", columns="Is_Loyal", values="Avg_Basket_Value").fillna(0)
        for c in ["Loyalty","Non-Loyalty"]:
            if c not in ap.columns: ap[c]=0
        ap.rename(columns={"Loyalty":"Avg_Basket_Loyal","Non-Loyalty":"Avg_Basket_NonLoyal"}, inplace=True)

        summary = (piv[["Loyalty","Non-Loyalty","Total_Receipts","Loyalty_%"]]
                   .merge(ap[["Avg_Basket_Loyal","Avg_Basket_NonLoyal"]], left_index=True, right_index=True, how="left")
                   .reset_index())
        summary = summary.sort_values("Total_Receipts", ascending=False).reset_index(drop=True)
        summary.insert(0, "#", range(1, len(summary)+1))

        sshow = summary.copy()
        for c in ["Loyalty","Non-Loyalty","Total_Receipts"]:
            sshow[c] = sshow[c].map(fmt_int)
        for c in ["Avg_Basket_Loyal","Avg_Basket_NonLoyal"]:
            sshow[c] = sshow[c].map(fmt_2)
        sshow["Loyalty_%"] = sshow["Loyalty_%"].map(lambda v: f"{v:.1f}%")

        st.subheader("Global loyalty vs non-loyalty (by store)")
        st.dataframe(sshow, use_container_width=True, hide_index=True)

        # Quick store visual
        store = st.selectbox("Store for visuals", sorted(receipts["STORE_NAME"].unique()))
        r = receipts[receipts["STORE_NAME"]==store].copy()
        if r.empty:
            st.info("No data for this store.")
        else:
            split = (r.groupby("Is_Loyal", as_index=False)["CUST_CODE"].nunique()
                     .rename(columns={"CUST_CODE":"Receipts"}))
            split["PCT"] = 100 * split["Receipts"] / split["Receipts"].sum()
            fig1 = px.bar(split, x="PCT", y="Is_Loyal", color="Is_Loyal", orientation="h",
                          color_discrete_map={"Loyalty":"#2ca02c","Non-Loyalty":"#d62728"},
                          text=split["PCT"].round(1).astype(str) + "%")
            fig1.update_traces(textposition="inside")
            fig1.update_layout(barmode="stack", height=260, showlegend=False,
                               xaxis_title="% of receipts", yaxis_title="")
            st.plotly_chart(fig1, use_container_width=True)

            avg_store = (r.groupby("Is_Loyal", as_index=False)["Basket_Value"].mean()
                         .rename(columns={"Basket_Value":"Avg_Basket_Value"}))
            fig2 = px.bar(avg_store, x="Avg_Basket_Value", y="Is_Loyal", color="Is_Loyal", orientation="h",
                          color_discrete_map={"Loyalty":"#1f77b4","Non-Loyalty":"#9edae5"},
                          text=avg_store["Avg_Basket_Value"].round(0).map('{:,.0f}'.format))
            fig2.update_traces(textposition="outside", cliponaxis=False)
            fig2.update_layout(height=260, showlegend=False,
                               xaxis_title="Avg Basket Value", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
