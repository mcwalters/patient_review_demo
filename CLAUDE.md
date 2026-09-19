# Working in this repo

Synthetic EHR dataset (`data/`) → DuckDB (`build_db.py`) → agentic prototype
(`prototype/`). Read `README.md` for the data and `prototype/README.md` for the
prototype architecture. What follows is only the operational stuff that is easy
to get wrong.

## Setup

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python build_db.py          # writes ehr.duckdb: 14 tables, 7 views
```

`ehr.duckdb` is derived and gitignored. Rebuild it rather than patching it.

## The agents need GCP, not an API key

Gemini 2.5 Pro on Vertex AI via **application default credentials**. There is no
`ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` anywhere, and adding one is not the fix
if something fails to authenticate.

```bash
gcloud auth application-default login   # project accorded-lake, region us-west1
```

`prototype/panel.py` and `prototype/screener.py` set `GOOGLE_GENAI_USE_VERTEXAI`,
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` at import time.

Fetching anything from Google Drive with ADC needs an explicit quota project:
`-H "X-Goog-User-Project: accorded-lake"`. Without it you get a 403 that does
not mention the real cause.

## Running the demo

```bash
./.venv/bin/python -m streamlit run prototype/app.py \
    --server.port 8501 --server.headless true --server.fileWatcherType none
```

`--server.fileWatcherType none` is deliberate. With the watcher on, **any edit to
a file under `prototype/` during a run pops a "File change / Rerun" banner that
cancels the in-flight agent run** and swallows the next button click. Do not edit
project files while an agent run is in progress.

## Verifying the app

**`curl` cannot tell you whether the Streamlit UI rendered.** Streamlit serves a
shell and builds the DOM client-side, so `curl … | grep "Cohort"` returns nothing
whether the page succeeded or failed. This produced two false timeouts on runs
that had actually completed. Use the browser tool (`get_page_text` /
`screenshot`) — it is the only valid check. `curl /_stcore/health` is fine for
"is the server up", and nothing else.

Agent runs are slow: screening 35–90s, panel review 4–6 minutes. Budget for it.

## The database lock

DuckDB allows one writer. A Jupyter kernel with `duckdb.connect("ehr.duckdb")`
open will block `build_db.py` and any read-only connection.
`prototype/tools.connect()` falls back to a temp copy when it hits the lock, so
the agents keep working, but `build_db.py` will fail outright — close the kernel
or run `conn.close()`.

## Two invariants worth not breaking

1. **No model writes SQL.** Agents select from the vocabulary in
   `prototype/vocab.py` and register structured values; `prototype/tools.py`
   builds every query. `define_criterion` rejects any code absent from the
   dataset. If you find yourself passing model output into a query string, stop.
2. **Findings travel as structured rows, not prose.** The `Findings` store in
   `prototype/panel.py` exists because prose hand-offs between agents lost data
   in both directions — 120 rows became "none found", and turning summarisation
   off made the supervisor short-circuit. Numbers shown to a user come from
   tools; model prose cites finding ids rather than restating values.

## Style

Commit messages here explain *why*, and record the failure that motivated a fix
when there was one. Several of the more surprising design choices only make
sense with that context, so keep it in the message rather than in your head.
