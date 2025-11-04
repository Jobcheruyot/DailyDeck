
import streamlit as st
from pathlib import Path
import types, builtins, sys, io, re, pandas as pd, numpy as np

st.set_page_config(page_title="DailyDeck — Streamlit Runner", layout="wide")

st.title("DailyDeck — Streamlit Runner")
st.caption("Runs your Colab-exported .py, shows Markdown without leading asterisks, and executes code cells.")

# Upload alternative .py if you want; otherwise we'll use the bundled dailydeck.py
uploaded = st.file_uploader("Optionally upload a different .py exported from Colab", type=["py"])

if uploaded:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
else:
    raw = Path("dailydeck.py").read_text(encoding="utf-8", errors="ignore")

# --- Helpers ---
def split_cells(py_text: str):
    """
    Split a Colab-exported .py into alternating (markdown, code) cells.
    We treat triple-quoted strings as Markdown cells.
    """
    cells = []
    i = 0
    in_md = False
    buf = []
    quote = None
    while i < len(py_text):
        if not in_md:
            # look for start of triple-quote
            if py_text.startswith('"""', i) or py_text.startswith("'''", i):
                # flush code buf (if any)
                if buf:
                    cells.append(("code", "".join(buf)))
                    buf = []
                quote = py_text[i:i+3]
                i += 3
                in_md = True
            else:
                buf.append(py_text[i])
                i += 1
        else:
            # in markdown; look for closing quote
            if py_text.startswith(quote, i):
                cells.append(("md", "".join(buf)))
                buf = []
                i += 3
                in_md = False
                quote = None
            else:
                buf.append(py_text[i])
                i += 1
    # flush remainder
    if buf:
        cells.append(("code" if not in_md else "md", "".join(buf)))
    return cells

def clean_markdown(md: str, strip_bullets: bool = True):
    """
    - Convert headers like '##*Heading' -> '## Heading'
    - If strip_bullets: turn lines that start with '*' or '- *' into normal text without the asterisk.
    - Avoid altering emphasis markers inside words.
    """
    lines = md.splitlines()
    out = []
    for ln in lines:
        # Header patterns like '#*Title' or '##*Title' or '###* Title'
        ln2 = re.sub(r'^(#{1,6})\s*\*+\s*', r'\1 ', ln)
        if strip_bullets:
            # bullet only if asterisk is the very first non-space and not followed by another asterisk
            ln2 = re.sub(r'^\s*\*\s+', '', ln2)
        out.append(ln2)
    return "\n".join(out)

# Shared execution globals (persist across cells)
exec_globals = {
    "__name__": "__main__",
}

# Provide simple shims
class _DisplayShim:
    def display(self, *objs):
        for o in objs:
            st.write(o)

# pretend IPython.display.display
exec_globals["display"] = _DisplayShim().display

# fake google.colab.files shim to avoid crashes
class _FilesShim:
    def upload(self):
        st.info("Use Streamlit's file uploader at the top of this page instead of google.colab.files.upload(). Returning empty dict.")
        return {}

exec_globals["files"] = _FilesShim()

# Optional data uploader for CSVs used inside your code
st.subheader("Data files (optional)")
data_files = st.file_uploader("Upload CSVs or other files your code expects", type=None, accept_multiple_files=True)
store = {}
if data_files:
    for f in data_files:
        store[f.name] = f
st.session_state["_uploaded_store"] = store

st.write("---")

cells = split_cells(raw)
strip_bullets = st.toggle("Strip bullet asterisks in Markdown", value=True)
run_all = st.button("▶️ Run All Cells")

for idx, (kind, body) in enumerate(cells, start=1):
    with st.container(border=True):
        st.markdown(f"**Cell {idx} — {kind.upper()}**")
        if kind == "md":
            md_clean = clean_markdown(body, strip_bullets=strip_bullets)
            st.markdown(md_clean)
        else:
            # Show code (collapsed by default)
            with st.expander("Show code"):
                st.code(body, language="python")
            go = run_all or st.button(f"Run cell {idx}", key=f"run_{idx}")
            if go:
                # Sanitize known Colab-only imports to prevent errors
                sanitized = []
                for line in body.splitlines():
                    if "google.colab" in line:
                        continue
                    if "IPython.display" in line and "display" in line:
                        # already shimmed above
                        continue
                    sanitized.append(line)
                src = "\n".join(sanitized)
                try:
                    exec(src, exec_globals, exec_globals)
                    st.success("✅ Executed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
