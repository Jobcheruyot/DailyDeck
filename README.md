# DailyDeck — Streamlit Runner

This mini-app lets you run a Colab-exported `.py` file inside Streamlit and **render the embedded Markdown _without the leading asterisks_**.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

- By default it uses the included `dailydeck.py` (copied from your upload).
- You can also upload a different `.py` at runtime.
- Click **Run All Cells** or run cells one-by-one.
- Markdown headers like `##*Title` are automatically shown as `## Title`.
- Optional CSV uploader is included for data your code may expect.
