import os
# avoid TF/Flax imports for local embeddings
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import re
import time
import typing as t
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ---------- App ----------
st.set_page_config(page_title="Latest iPad Q&A (RAG)", layout="wide")
st.title("📱 Latest iPad Q&A (RAG)")

# ---------- Persona / Brand ----------
BRAND_NAME = os.getenv("BRAND_NAME", "Your Company")
BOT_NAME = os.getenv("BOT_NAME", "Device Guide Assistant")
BOT_PURPOSE = os.getenv(
    "BOT_PURPOSE",
    "Help visitors understand the latest Apple iPad models using official sources. "
    "Provide comparisons and buying considerations. I don’t place orders or access personal accounts."
)

PERSONA_TEXT = f"""
You are {BOT_NAME}, a virtual assistant on {BRAND_NAME}'s website.
Purpose: {BOT_PURPOSE}
Capabilities: answer from Apple’s official pages, compare models in a small table, summarize differences.
Limits: do not invent specs; if info isn’t in the provided sources, say you don’t know. No account access or ordering.
Privacy: no personal data storage; responses are generated for this chat session only.
""".strip()

USAGE_TIPS = (
    "- Try: 'Compare iPad Pro vs Air', 'Does it support Apple Pencil Pro?', 'What are the camera specs?'\n"
    "- Say 'compare' to get a table.\n"
    "- Type 'contact' or 'handover' for human support."
)

st.caption(f"Hi! I’m **{BOT_NAME}** on **{BRAND_NAME}**’s site — ask me about the latest iPad models.")

# ---------- Config ----------
SEED_URLS = [
    "https://www.apple.com/ipad/compare/",
    "https://www.apple.com/ipad-pro/",
    "https://www.apple.com/ipad-pro/specs/",
    "https://www.apple.com/ipad-air/",
    "https://www.apple.com/ipad-air/specs/",
    "https://www.apple.com/ipad-10.9/",
    "https://www.apple.com/ipad-10.9/specs/",
    "https://www.apple.com/ipad-mini/",
    "https://www.apple.com/ipad-mini/specs/",
]
MAX_CHARS = 4000
OVERLAP = 64
TOP_K = 3

EMBED_MODEL_GEMINI = "models/text-embedding-004"   # 3072-dim
CHAT_MODEL_GEMINI  = "models/gemini-2.5-pro"
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_PATH = "kb.json"

API_KEY = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None)

# ---------- Helpers ----------
def local_backend_available() -> bool:
    try:
        import sentence_transformers  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False

if "embed_backend" not in st.session_state:
    st.session_state["embed_backend"] = "Local (no API limits)" if local_backend_available() else "Gemini (throttled)"

def fetch_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

def chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> t.List[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def _extract_embedding_vector(resp) -> np.ndarray:
    emb = resp.get("embedding")
    vals = emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    return np.array(vals, dtype=np.float32)

# meta answers (who are you, purpose, etc.)
META_PATTERNS = re.compile(
    r"\b(who\s+are\s+you|your\s+name|what\s+is\s+your\s+name|purpose|what\s+can\s+you\s+do|how\s+to\s+use|help|about\s+you)\b",
    re.I,
)
def is_meta_query(q: str) -> bool:
    return bool(META_PATTERNS.search(q or ""))

def answer_meta(_: str) -> str:
    return "\n\n".join([
        f"**I’m {BOT_NAME} — {BRAND_NAME}’s product assistant.**",
        PERSONA_TEXT,
        "\n**How to use me**\n" + USAGE_TIPS,
    ])

# compare intent + enrichment
COMPARE_PAT = re.compile(r"\b(compare|vs|versus)\b", re.I)
def is_compare_query(q: str) -> bool:
    return bool(COMPARE_PAT.search(q or ""))

def ensure_compare_coverage(kb: dict, query: str, base_hits: t.List[dict], k_total: int = 10) -> t.List[dict]:
    if not kb.get("vectors"):
        return base_hits
    qv = embed_query_with_kb_backend(query, kb)
    qv_n = qv / (np.linalg.norm(qv) + 1e-9)
    urls = [m["url"] for m in kb["metas"]]
    vectors = kb["vectors"]
    want_patterns = ["/ipad-pro/specs/", "/ipad-air/specs/"]
    extra = []
    for pat in want_patterns:
        idxs = [i for i, u in enumerate(urls) if pat in u]
        if not idxs:
            continue
        scored = sorted(
            ((i, float(np.dot(qv_n, vectors[i] / (np.linalg.norm(vectors[i]) + 1e-9)))) for i in idxs),
            key=lambda x: x[1],
            reverse=True,
        )[:2]
        for i, s in scored:
            extra.append({"rank": 0, "score": s, "chunk": kb["chunks"][i], "url": urls[i]})
    def key_of(h): return (h.get("url", ""), h.get("chunk", "")[:80])
    seen, combined = set(), []
    for h in base_hits + extra:
        k = key_of(h)
        if k in seen:
            continue
        seen.add(k)
        combined.append(h)
    combined.sort(key=lambda x: x["score"], reverse=True)
    combined = combined[:k_total]
    for r, h in enumerate(combined, start=1):
        h["rank"] = r
    return combined

# ---------- Embeddings ----------
def embed_texts_gemini(
    texts: t.List[str],
    api_key: str,
    rpm: int = 20,
    max_retries: int = 5,
    min_delay: float = 2.1
) -> t.List[np.ndarray]:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    delay = max(min_delay, 60.0 / max(1, rpm))
    vecs: t.List[np.ndarray] = []
    for ttext in texts:
        for attempt in range(max_retries):
            try:
                r = genai.embed_content(model=EMBED_MODEL_GEMINI, content=ttext)
                vecs.append(_extract_embedding_vector(r))
                break
            except Exception as e:
                s = str(e).lower()
                if ("429" in s) or ("rate" in s and "limit" in s) or ("quota" in s) or ("resourceexhausted" in s):
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                raise
        time.sleep(delay)
    return vecs

@st.cache_resource(show_spinner=False)
def _load_local_model(model_name: str = LOCAL_EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def embed_texts_local(texts: t.List[str], model_name: str = LOCAL_EMBED_MODEL) -> t.List[np.ndarray]:
    m = _load_local_model(model_name)
    arr = m.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return [np.array(v, dtype=np.float32) for v in arr]

# ---------- KB / Retrieval ----------
def load_or_build_kb(embed_backend: str) -> dict:
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
        built_with = kb["meta"].get("backend")
        if built_with and built_with != embed_backend:
            st.warning(f"KB was built with '{built_with}', but you selected '{embed_backend}'. Click **Rebuild KB**.")
        return kb

    with st.spinner("Building knowledge base…"):
        all_chunks, metas = [], []
        for url in SEED_URLS:
            text = fetch_text(url)
            if text.startswith("ERROR"):
                st.warning(text); continue
            for c in chunk(text):
                all_chunks.append(c); metas.append({"url": url})
        if not all_chunks:
            st.error("No content fetched."); return {"chunks": [], "metas": [], "vectors": [], "meta": {}}
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
                "chunks": all_chunks, "metas": metas, "vectors": [v.tolist() for v in vecs],
                "backend": embed_backend, "embed_model": embed_model_used, "embed_dim": embed_dim
            }, f)
        return kb

def embed_query_with_kb_backend(query: str, kb: dict) -> np.ndarray:
    backend = (kb.get("meta") or {}).get("backend") or st.session_state.get("embed_backend")
    return (embed_texts_gemini([query], api_key=API_KEY)[0]
            if backend.startswith("Gemini") else
            embed_texts_local([query])[0])

def retrieve(kb: dict, query: str, k: int = TOP_K) -> t.List[dict]:
    if not kb.get("vectors"):
        return []
    qv = embed_query_with_kb_backend(query, kb)
    doc_dim = kb["vectors"][0].shape[0]
    if qv.shape[0] != doc_dim:
        st.error(f"Embedding dimension mismatch (query {qv.shape[0]} vs docs {doc_dim}). Click **Rebuild KB**.")
        return []
    sims = [cosine_sim(qv, v) for v in kb["vectors"]]
    idxs = np.argsort(sims)[::-1][:k]
    return [{"rank": r+1, "score": sims[i], "chunk": kb["chunks"][i], "url": kb["metas"][i]["url"]}
            for r, i in enumerate(idxs)]

# ---------- Answering ----------
def answer_with_rag(query: str, hits):
    if is_meta_query(query):
        return answer_meta(query)

    import google.generativeai as genai
    genai.configure(api_key=API_KEY)

    def fmt_ctx(hs):
        return "\n\n".join([f"[{h['rank']}] (source: {h['url']})\n{h['chunk'][:1200]}" for h in hs])

    full_context = fmt_ctx(hits)
    short_context = fmt_ctx(hits[:2])

    system = (
        "You answer questions about Apple iPad models using ONLY the provided context. "
        "Cite sources as [1], [2], etc, matching the bracket numbers in the context. "
        "For comparisons, output a concise markdown table with rows: "
        "Chip; Display (size & tech); Storage options; Cameras; Battery; Dimensions/Weight; "
        "Apple Pencil/Magic Keyboard compatibility; Notable features. "
        "If some details are missing in the context, still produce the table with '—' for that cell and add a one-line note. "
        "If asked for suggestions, include a short 'Suggestions' section grounded to the context."
    )

    def prompt(ctx: str) -> str:
        return (
            f"{system}\n\nQuestion: {query}\n\n"
            "Use ONLY the context below. Cite as [1], [2], ... matching the brackets.\n\n"
            f"Context:\n{ctx}\n\nAnswer:"
        )

    def call(model_name: str, ctx: str):
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system,
                generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=700),
            )
            resp = model.generate_content(prompt(ctx))
        except Exception as e:
            return "", f"AI engine error — please report to the team. Details: {e}"
        txt = ""
        try:
            for cand in (resp.candidates or []):
                parts = getattr(cand.content, "parts", []) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t: txt += t
            if not txt and getattr(resp, "text", None):
                txt = resp.text
        except Exception:
            pass
        if txt: return txt.strip(), ""
        reason = "unknown"
        try:
            fr = resp.candidates[0].finish_reason
            reason = getattr(fr, "name", fr)
        except Exception:
            pass
        return "", f"AI engine error — finish_reason={reason}. Please try again."

    out, err = call(CHAT_MODEL_GEMINI, full_context)
    if out: return out
    out, err2 = call(CHAT_MODEL_GEMINI, short_context)
    if out: return out
    for mid in ["models/gemini-2.5-flash", "models/gemini-1.5-flash"]:
        out, _ = call(mid, short_context)
        if out: return out
    return err or err2 or "AI engine error — please try again."

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Settings & Data")

    backends = ["Gemini (throttled)"] + (["Local (no API limits)"] if local_backend_available() else [])
    embed_backend = st.radio(
        "Embeddings backend",
        options=backends,
        index=backends.index(st.session_state.get("embed_backend")) if st.session_state.get("embed_backend") in backends else 0,
        help="If you change this, click Rebuild KB."
    )
    st.session_state["embed_backend"] = embed_backend
    st.caption(f"Backend in use: **{embed_backend}**")

    st.markdown(f"**About this assistant**  \n{BOT_NAME} on {BRAND_NAME}  \n{BOT_PURPOSE}")

    urls_text = st.text_area("Seed URLs (one per line)", value="\n".join(SEED_URLS), height=140)
    col_a, col_b = st.columns(2)
    rebuild = col_a.button("Rebuild KB")
    clear_chat = col_b.button("Clear Chat")

    if clear_chat:
        st.session_state.pop("chat", None)
        st.session_state.pop("last_hits", None)
        st.success("Chat cleared."); st.rerun()

    new_urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if new_urls != SEED_URLS:
        SEED_URLS[:] = new_urls

    if rebuild and os.path.exists(KB_PATH):
        os.remove(KB_PATH)
        st.success("KB removed. Rebuilding…"); st.rerun()

# ---------- Preconditions ----------
if not API_KEY:
    with st.sidebar:
        st.error("Set GEMINI_API_KEY as an environment variable or secret.")
    st.stop()

# ---------- Build / Load KB ----------
kb = load_or_build_kb(st.session_state["embed_backend"])
meta = kb.get("meta") or {}
saved_backend, saved_dim = meta.get("backend"), meta.get("embed_dim")
expected_dim = 384 if st.session_state["embed_backend"].startswith("Local") else 3072
if (saved_backend and saved_backend != st.session_state["embed_backend"]) or (saved_dim and saved_dim != expected_dim):
    if os.path.exists(KB_PATH): os.remove(KB_PATH)
    with st.spinner("Rebuilding KB to match selected backend…"):
        kb = load_or_build_kb(st.session_state["embed_backend"])

# ---------- UI ----------
left, right = st.columns([0.62, 0.38])

with left:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.form("ask_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask about the latest iPad…",
            placeholder="e.g., Compare iPad Pro vs Air, or 'Which models support Apple Pencil Pro?'",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_q:
        st.session_state.chat.append(("user", user_q))
        with st.spinner("🧠 Thinking… searching sources and generating a grounded answer"):
            if is_meta_query(user_q):
                reply = answer_meta(user_q)
                st.session_state["last_hits"] = []
            else:
                base_hits = retrieve(kb, user_q, k=TOP_K)
                hits = ensure_compare_coverage(kb, user_q, base_hits, k_total=10) if is_compare_query(user_q) else base_hits
                reply = ("I couldn't retrieve any context. Click **Rebuild KB** or adjust the seed URLs."
                         if not hits else
                         answer_with_rag(user_q, hits))
                st.session_state["last_hits"] = hits
        st.session_state.chat.append(("assistant", reply))

    for role, msg in reversed(st.session_state.chat):
        st.chat_message(role).markdown(msg)

with right:
    st.subheader("Retrieved sources")
    st.caption("Top matching context chunks with their pages.")
    hits_to_show = st.session_state.get("last_hits")
    if hits_to_show:
        for h in hits_to_show:
            st.markdown(f"**[{h['rank']}]** {h['url']}")
            st.caption(f"relevance: {h['score']:.3f}")
            st.code(h["chunk"][:700])
    else:
        st.write("Ask a question to see sources here.")

st.caption("Newest messages appear at the top. Switching embeddings will rebuild the KB so vectors match.")
