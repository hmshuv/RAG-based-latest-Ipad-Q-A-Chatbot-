# app.py — Latest iPad Q&A (RAG) with robust embeddings + Gemini 2.5 Pro
# Quickstart:
#   pip install -r requirements.txt
#   export GEMINI_API_KEY=your_key_here
#   streamlit run app.py

# --- keep these 2 lines at the VERY top (avoid TF/Keras import issues for Local) ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # don't import TensorFlow/Keras
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import re
import time
import typing as t
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

# -------------------- Streamlit config MUST be first --------------------
st.set_page_config(page_title="Latest iPad Q&A (RAG)", layout="wide")
st.title("📱 Latest iPad Q&A (RAG)")

# -------------------- Config --------------------
# Keep this small to cut embedding calls; you can add more in the sidebar later.
SEED_URLS = [
    # Compare page (great for side-by-side specs)
    "https://www.apple.com/ipad/compare/",
    # iPad Pro overview + specs
    "https://www.apple.com/ipad-pro/",
    "https://www.apple.com/ipad-pro/specs/",
    # iPad Air overview + specs
    "https://www.apple.com/ipad-air/",
    "https://www.apple.com/ipad-air/specs/",
    # iPad (10th gen) overview + specs
    "https://www.apple.com/ipad-10.9/",
    "https://www.apple.com/ipad-10.9/specs/",
    # (optional) iPad mini
    "https://www.apple.com/ipad-mini/",
    "https://www.apple.com/ipad-mini/specs/",
]

# Bigger chunks & lower overlap => fewer embedding calls (helps free-tier limits).
MAX_CHARS = 4000
OVERLAP = 64
TOP_K = 3

# Embedding backends
EMBED_BACKEND_DEFAULT = "Gemini (throttled)"  # runtime default is overridden to Local if available

# Gemini model IDs (must include 'models/' prefix for the google.generativeai SDK)
EMBED_MODEL_GEMINI = "models/text-embedding-004"   # 3072-dim
CHAT_MODEL_GEMINI  = "models/gemini-2.5-pro"       # chat/answering

# Local embedding model (384-dim)
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Storage
KB_PATH = "kb.json"

# API key (env first, then Streamlit secrets)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets.get("GEMINI_API_KEY", None)  # ok if not present here; we check later
    except Exception:
        API_KEY = None

# -------------------- Helpers --------------------
def local_backend_available() -> bool:
    try:
        import sentence_transformers  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False

# Prefer Local on FIRST load if available; else Gemini
if "embed_backend" not in st.session_state:
    st.session_state["embed_backend"] = (
        "Local (no API limits)" if local_backend_available() else "Gemini (throttled)"
    )

def fetch_text(url: str) -> str:
    """Fetch & lightly clean visible text."""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)

def chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> t.List[str]:
    """Simple fixed-size chunker with overlap."""
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def _extract_embedding_vector(resp) -> np.ndarray:
    """Handle both {'embedding':[...]} and {'embedding':{'values':[...]} shapes."""
    emb = resp.get("embedding")
    vals = emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    return np.array(vals, dtype=np.float32)

# -------------------- Embedding backends --------------------
def embed_texts_gemini(
    texts: t.List[str],
    api_key: str,
    rpm: int = 20,           # respect free-tier per-minute limits
    max_retries: int = 5,
    min_delay: float = 2.1   # seconds between calls (throttle)
) -> t.List[np.ndarray]:
    """Gemini embeddings with throttling + exponential backoff on 429."""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    delay = max(min_delay, 60.0 / max(1, rpm))
    vecs: t.List[np.ndarray] = []

    for idx, ttext in enumerate(texts):
        for attempt in range(max_retries):
            try:
                r = genai.embed_content(model=EMBED_MODEL_GEMINI, content=ttext)
                vecs.append(_extract_embedding_vector(r))
                break
            except Exception as e:
                s = str(e)
                is_rate = ("429" in s) or ("Rate limit" in s) or ("quota" in s.lower()) or ("ResourceExhausted" in s)
                if is_rate and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # backoff: 1s,2s,4s,8s
                    continue
                raise
        time.sleep(delay)  # throttle
    return vecs

@st.cache_resource(show_spinner=False)
def _load_local_model(model_name: str = LOCAL_EMBED_MODEL):
    """Load sentence-transformers once (cached)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        st.error("Local embeddings selected but 'sentence-transformers' isn't installed.\n"
                 "Install: `pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu`")
        raise e
    return SentenceTransformer(model_name)

def embed_texts_local(texts: t.List[str], model_name: str = LOCAL_EMBED_MODEL) -> t.List[np.ndarray]:
    """Local CPU embeddings (no quotas)."""
    m = _load_local_model(model_name)
    arr = m.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return [np.array(v, dtype=np.float32) for v in arr]

# -------------------- RAG core --------------------
def load_or_build_kb(embed_backend: str) -> dict:
    """Load persisted KB or build it from SEED_URLS (first run)."""
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        kb = {
            "chunks": raw["chunks"],
            "metas": raw["metas"],
            "vectors": [np.array(v, dtype=np.float32) for v in raw["vectors"]],
            "meta": {
                "backend": raw.get("backend"),
                "embed_model": raw.get("embed_model"),
                "embed_dim": raw.get("embed_dim"),
            }
        }
        # Warn if backend differs from the saved one
        built_with = kb["meta"].get("backend")
        if built_with and built_with != embed_backend:
            st.warning(f"KB was built with '{built_with}', but you selected '{embed_backend}'. "
                       f"Click **Rebuild KB** to re-embed.")
        return kb

    with st.spinner("Building knowledge base… (first run only)"):
        all_chunks, metas = [], []
        for url in SEED_URLS:
            text = fetch_text(url)
            if text.startswith("ERROR"):
                st.warning(text)
                continue
            for c in chunk(text):
                all_chunks.append(c)
                metas.append({"url": url})
        if not all_chunks:
            st.error("No content fetched. Check SEED_URLS or your network.")
            return {"chunks": [], "metas": [], "vectors": [], "meta": {}}

        # Embed with chosen backend
        try:
            if embed_backend.startswith("Gemini"):
                vecs = embed_texts_gemini(all_chunks, api_key=API_KEY)
                embed_model_used = EMBED_MODEL_GEMINI
            else:
                vecs = embed_texts_local(all_chunks)
                embed_model_used = LOCAL_EMBED_MODEL
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return {"chunks": [], "metas": [], "vectors": [], "meta": {}}

        embed_dim = len(vecs[0]) if vecs else 0
        kb = {"chunks": all_chunks, "metas": metas, "vectors": vecs,
              "meta": {"backend": embed_backend, "embed_model": embed_model_used, "embed_dim": embed_dim}}

        with open(KB_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": all_chunks,
                "metas": metas,
                "vectors": [v.tolist() for v in vecs],
                "backend": embed_backend,
                "embed_model": embed_model_used,
                "embed_dim": embed_dim
            }, f)
        return kb

def embed_query_with_kb_backend(query: str, kb: dict) -> np.ndarray:
    """Always embed the query with the SAME backend used for the KB."""
    backend = (kb.get("meta") or {}).get("backend") or st.session_state.get("embed_backend", EMBED_BACKEND_DEFAULT)
    if backend.startswith("Gemini"):
        return embed_texts_gemini([query], api_key=API_KEY)[0]
    else:
        return embed_texts_local([query])[0]

def retrieve(kb: dict, query: str, k: int = TOP_K) -> t.List[dict]:
    if not kb.get("vectors"):
        return []
    qv = embed_query_with_kb_backend(query, kb)
    doc_dim = kb["vectors"][0].shape[0]
    if qv.shape[0] != doc_dim:
        st.error(f"Embedding dimension mismatch (query {qv.shape[0]} vs docs {doc_dim}). "
                 f"Click **Rebuild KB** so they match.")
        return []
    sims = [cosine_sim(qv, v) for v in kb["vectors"]]
    idxs = np.argsort(sims)[::-1][:k]
    return [
        {"rank": r+1, "score": sims[i], "chunk": kb["chunks"][i], "url": kb["metas"][i]["url"]}
        for r, i in enumerate(idxs)
    ]

def answer_with_rag(query: str, hits):
    """Ask Gemini to answer using ONLY retrieved context; cite [1],[2],… with robust fallbacks."""
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)

    def fmt_ctx(hs):
        return "\n\n".join([f"[{h['rank']}] (source: {h['url']})\n{h['chunk'][:1200]}" for h in hs])

    full_context = fmt_ctx(hits)
    short_context = fmt_ctx(hits[:2])  # retry with smaller prompt if needed

    system = (
        "You are a precise assistant answering questions about Apple iPad models. "
        "Use ONLY the provided context; if it’s not in context, say you don’t know. "
        "Always cite sources as [1], [2], etc (matching the brackets in context). "
        "If the user asks to compare models, return a concise markdown table with rows for: "
        "Chip, Display (size & tech), Storage options, Cameras, Battery, Dimensions/Weight, "
        "Apple Pencil/Magic Keyboard compatibility, and Notable features."
    )

    def call(model_name: str, context: str):
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
        )
        return model.generate_content([{
            "role": "user",
            "parts": [
                f"Question: {query}",
                "Use ONLY the context below.",
                "Context:",
                context,
                "Answer:"
            ],
        }])

    def extract_text(resp) -> str:
        txt = ""
        try:
            for cand in (resp.candidates or []):
                parts = getattr(cand.content, "parts", []) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        txt += t
            if not txt and getattr(resp, "text", None):
                txt = resp.text
        except Exception:
            pass
        return (txt or "").strip()

    # Try 1: 2.5 Pro + full context
    resp = call(CHAT_MODEL_GEMINI, full_context)
    out = extract_text(resp)
    if out:
        return out

    # Try 2: 2.5 Pro + short context
    resp = call(CHAT_MODEL_GEMINI, short_context)
    out = extract_text(resp)
    if out:
        return out

    # Try 3: fallback model (flash) + short context
    for mid in ["models/gemini-2.5-flash", "models/gemini-1.5-flash"]:
        try:
            resp = call(mid, short_context)
            out = extract_text(resp)
            if out:
                return out
        except Exception:
            continue

    reason = "unknown"
    try:
        fr = resp.candidates[0].finish_reason
        reason = getattr(fr, "name", fr)
    except Exception:
        pass
    return f"(The model returned an empty message; finish_reason={reason}. Please try again.)"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Settings & Data")

    # Build options dynamically and reflect current selection
    backends = ["Gemini (throttled)"]
    if local_backend_available():
        backends.append("Local (no API limits)")
    else:
        st.caption("To enable Local: `pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu`")

    current = st.session_state.get("embed_backend", EMBED_BACKEND_DEFAULT)
    embed_backend = st.radio(
        "Embeddings backend",
        options=backends,
        index=backends.index(current) if current in backends else 0,
        help="If you change this, click Rebuild KB so embeddings match."
    )
    st.session_state["embed_backend"] = embed_backend
    st.caption(f"Backend in use: **{embed_backend}**")

    urls_text = st.text_area("Seed URLs (one per line)", value="\n".join(SEED_URLS), height=140)
    col_a, col_b = st.columns(2)
    rebuild = col_a.button("Rebuild KB")
    clear_chat = col_b.button("Clear Chat")

    if clear_chat:
        st.session_state.pop("chat", None)
        st.success("Chat cleared.")
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    # Update URLs if changed
    new_urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if new_urls != SEED_URLS:
        SEED_URLS[:] = new_urls

    if rebuild and os.path.exists(KB_PATH):
        os.remove(KB_PATH)
        st.success("KB removed. It will rebuild now.")
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# -------------------- Preconditions --------------------
# Chat always uses Gemini -> require API key (even if embeddings are Local)
if not API_KEY:
    with st.sidebar:
        st.error(
            "GEMINI_API_KEY is not set.\n\n"
            "mac/linux:  export GEMINI_API_KEY=YOUR_KEY\n"
            "powershell: $env:GEMINI_API_KEY=\"YOUR_KEY\""
        )
    st.stop()

# -------------------- Build / Load KB --------------------
kb = load_or_build_kb(st.session_state["embed_backend"])

# Auto-rebuild if saved KB backend/dimension mismatches the selected backend
saved_meta = (kb.get("meta") or {})
saved_backend = saved_meta.get("backend")
saved_dim = saved_meta.get("embed_dim")
expected_dim = 384 if st.session_state["embed_backend"].startswith("Local") else 3072

if (saved_backend and saved_backend != st.session_state["embed_backend"]) or (saved_dim and saved_dim != expected_dim):
    if os.path.exists(KB_PATH):
        os.remove(KB_PATH)
    with st.spinner("Rebuilding KB to match the selected embeddings backend…"):
        kb = load_or_build_kb(st.session_state["embed_backend"])

# -------------------- Chat UI --------------------
left, right = st.columns([0.62, 0.38])

with left:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        st.chat_message(role).markdown(msg)

    user_q = st.chat_input("Ask about the latest iPad…")
    if user_q:
        st.session_state.chat.append(("user", user_q))
        st.chat_message("user").markdown(user_q)

        hits = retrieve(kb, user_q, k=TOP_K)
        if not hits:
            reply = "I couldn't retrieve any context. Click **Rebuild KB** or adjust the seed URLs."
        else:
            reply = answer_with_rag(user_q, hits)

        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").markdown(reply)

with right:
    st.subheader("Retrieved sources")
    st.caption("Top matching context chunks with their pages.")
    if st.session_state.get("chat"):
        last_q = next((m for m in reversed(st.session_state.chat) if m[0] == "user"), None)
        if last_q:
            for h in retrieve(kb, last_q[1], k=TOP_K):
                st.markdown(f"**[{h['rank']}]** {h['url']}")
                st.caption(f"relevance: {h['score']:.3f}")
                st.code(h["chunk"][:700])
    else:
        st.write("Ask a question to see sources here.")

st.caption("Tip: If you switch embeddings backend, the KB auto-rebuilds so query/doc vectors match.")
