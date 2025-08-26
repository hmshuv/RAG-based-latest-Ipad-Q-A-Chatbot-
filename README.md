# Latest iPad Q\&A (RAG)

A small, interview-ready chatbot that answers questions about **the latest Apple iPad models** using **retrieval-augmented generation (RAG)**.
It scrapes Apple’s official pages, embeds the content, retrieves the most relevant chunks, and asks Gemini to answer **only** from those sources with inline citations.

---

## Highlights

* **RAG by default** — grounded answers with `[1]`, `[2]` style citations.
* **Compare intent** — “compare pro vs air” returns a compact markdown table; if data is missing the bot fills the cell with `—` and explains.
* **Persona & brand** — set `BRAND_NAME`, `BOT_NAME`, and `BOT_PURPOSE` to drop this onto any website.
* **Two embedding backends**

  * **Gemini** embeddings (3072-dim) with throttling + retry
  * **Local** CPU embeddings via `sentence-transformers` (384-dim) — no quotas
* **Fast, simple store** — vectors persisted in `kb.json` (no DB required).
* **Good UX** — input pinned to the top, newest messages on top, right-side “Retrieved sources,” loading spinner on each ask.

---

## Architecture (one file)

```
app.py
 ├─ Crawl & clean      -> requests + BeautifulSoup
 ├─ Chunk              -> fixed-width with overlap
 ├─ Embed              -> Gemini (text-embedding-004) or local (all-MiniLM-L6-v2)
 ├─ Store              -> kb.json (chunks, meta, vectors)
 ├─ Retrieve           -> cosine sim (numpy)
 ├─ Compare Enrichment -> adds top chunks from Pro/Air spec pages
 ├─ Generate           -> Gemini 2.5 Pro (flash fallback)
 └─ UI                 -> Streamlit (top input, newest-first chat, sources panel)
```

---

## Quick start

```bash
git clone <your-repo>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Required
export GEMINI_API_KEY=YOUR_KEY

# Optional persona
export BRAND_NAME="Your Company"
export BOT_NAME="Device Guide Assistant"
export BOT_PURPOSE="Help visitors understand the latest iPad models…"

streamlit run app.py
```

**Note (Local embeddings):** if you want the “Local (no API limits)” backend,

```bash
pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Streamlit Cloud

**Secrets (TOML):**

```toml
GEMINI_API_KEY = "your-key"
BRAND_NAME     = "Your Company"
BOT_NAME       = "Device Guide Assistant"
BOT_PURPOSE    = "Help visitors understand the latest iPad models…"
```

Push the repo → Create a new Streamlit app → set the Secrets above.

---

## How it works

1. **Seed URLs** (left sidebar) contain Apple’s official iPad pages (overview/specs/compare).
2. On first run the app **scrapes**, **chunks**, **embeds**, and saves a `kb.json`.
3. On each question:

   * Detect **meta** (“who are you, purpose”) → instant persona response.
   * Detect **compare** → enrich retrieval with top chunks from
     `/ipad-pro/specs/` and `/ipad-air/specs/`, then re-rank.
   * Build a prompt with retrieved context and call **Gemini 2.5 Pro**
     (falls back to flash if needed).
   * Show answer + citations. The **exact chunks** used appear in **Retrieved sources**.

---

## Controls

* **Embeddings backend:** use Gemini or Local. Changing this **rebuilds** the KB so dimensions match (3072 vs 384).
* **Seed URLs:** add/remove sources and click **Rebuild KB**.
* **Clear Chat:** resets session history.

---

## Customizing the assistant

Set environment variables (or Streamlit secrets):

* `BRAND_NAME` — “Acme Inc.”
* `BOT_NAME` — “Device Guide Assistant”
* `BOT_PURPOSE` — short statement rendered in “About this assistant”

---

## Tech stack

* **UI:** Streamlit
* **Scrape:** requests + BeautifulSoup
* **Embeddings:** `models/text-embedding-004` (Gemini) or `sentence-transformers/all-MiniLM-L6-v2`
* **LLM:** `models/gemini-2.5-pro` (flash fallback)
* **Vector store:** JSON file (`kb.json`) + numpy cosine similarity

---

## Troubleshooting

* **“Set GEMINI\_API\_KEY”** – Provide key via env var or Streamlit Secrets.
* **Embedding dimension mismatch** – Click **Rebuild KB** after switching backends.
* **429 / quota** – The app already throttles + retries for Gemini embeddings; switch to **Local** backend if needed.
* **“AI engine error — …”** – Generic guardrail. Usually network/SDK hiccup; retry, or see Streamlit Cloud logs.

---

## Project structure

```
app.py
requirements.txt
kb.json        # created at runtime
README.md
```

---

## Roadmap

* Add lightweight **spec schema** so comparisons are perfectly column-aligned.
* Optional **FAISS/Chroma** for larger corpora.
* **Autoscrape** on schedule and diff-aware KB rebuilds.
* Export answers as **PDF** or shareable link.
* Simple **analytics** (questions volume, answer rate, common intents).

---

## License

MIT (or your preferred license).
