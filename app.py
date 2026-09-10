import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import gdown
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# Pakistan Cyber Law RAG — Streamlit + FAISS + Grok
# Designed for Colab and Streamlit Community Cloud.
# ============================================================

APP_TITLE = "Pakistan Cyber Law RAG"
DEFAULT_DRIVE_FILE_ID = "1znmUdzzNS5gFmbSa5zEV9QFYO--PGzbr"
DEFAULT_PDF_URL = (
    f"https://drive.google.com/uc?id={DEFAULT_DRIVE_FILE_ID}&export=download"
)
DATA_DIR = Path(".rag_data")
PDF_PATH = DATA_DIR / "pakistan_cyber_law.pdf"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunks.npz"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-4.6"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1250px; padding-top: 1.5rem;}
    .law-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        background: rgba(128,128,128,.05);
    }
    .disclaimer {
        border-left: 4px solid #d97706;
        padding: 10px 14px;
        background: rgba(217,119,6,.08);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    """Read Streamlit secret first, then environment variable."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_pdf_from_drive() -> Path:
    """
    Download the user's Google Drive PDF at startup.
    The app also allows a different Drive URL/file ID from the sidebar.
    """
    ensure_data_dir()
    drive_url = st.session_state.get("drive_url", DEFAULT_PDF_URL)
    force = st.session_state.get("force_refresh", False)

    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 1000 and not force:
        return PDF_PATH

    # Prefer gdown because it handles Google Drive confirmation/permissions.
    if "drive.google.com" in drive_url:
        try:
            result = gdown.download(
                url=drive_url,
                output=str(PDF_PATH),
                quiet=True,
                fuzzy=True,
            )
            if result and PDF_PATH.exists() and PDF_PATH.stat().st_size > 1000:
                return PDF_PATH
        except Exception as exc:
            raise RuntimeError(
                "Could not download the Google Drive PDF. "
                "Make sure the file is shared as 'Anyone with the link' "
                "and that the Drive URL points to a PDF."
            ) from exc

    raise RuntimeError("Unsupported or invalid Google Drive URL.")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_chunks(text: str, page_no: int, chunk_words: int = 230, overlap_words: int = 45):
    """
    Chunk page text while preserving page metadata.
    Legal sections often span pages, so metadata includes page numbers.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)

    for start in range(0, len(words), step):
        part = words[start : start + chunk_words]
        if not part:
            break
        chunk = " ".join(part).strip()
        if len(chunk) >= 80:
            chunks.append(
                {
                    "text": chunk,
                    "page": page_no,
                    "source": PDF_PATH.name,
                }
            )
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    all_chunks: List[Dict] = []

    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        raw = clean_text(raw)
        all_chunks.extend(page_chunks(raw, page_no))

    if not all_chunks:
        raise RuntimeError(
            "No text could be extracted from the PDF. "
            "If the PDF is scanned/image-only, OCR the PDF first."
        )

    return all_chunks


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def corpus_signature(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    return hashlib.sha256(
        f"{pdf_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()


@st.cache_resource(show_spinner=False)
def build_index(pdf_signature: str, pdf_path_str: str):
    """
    Embeddings are created at startup (first app load after PDF changes).
    The index is cached by Streamlit and also persisted to disk.
    """
    pdf_path = Path(pdf_path_str)
    chunks = extract_pdf_chunks(pdf_path)
    texts = [c["text"] for c in chunks]

    model = load_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, chunks, vectors.shape[1]


def retrieve(query: str, index, chunks, model, top_k: int, threshold: float):
    qvec = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(qvec, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if float(score) < threshold:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        item["rank"] = len(results) + 1
        results.append(item)
    return results


def make_context(results: List[Dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[Source {r['rank']} | PDF page {r['page']} | "
            f"similarity {r['score']:.3f}]\n{r['text']}"
        )
    return "\n\n".join(blocks)


def grok_answer(
    api_key: str,
    model_name: str,
    question: str,
    context: str,
    technical_level: str,
    response_size: str,
    temperature: float,
) -> str:
    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)

    size_instruction = {
        "Short": "Answer in about 120–220 words.",
        "Medium": "Answer in about 250–450 words.",
        "Detailed": "Answer in about 500–800 words.",
    }[response_size]

    system_prompt = f"""
You are a Pakistan cyber-law information assistant using retrieval-augmented
generation (RAG). Your legal source of truth is ONLY the supplied retrieved
text from the user's Pakistan cyber-law PDF.

Rules:
1. Do not invent sections, penalties, authorities, dates, procedures, or legal
   conclusions that are not supported by the retrieved text.
2. If the retrieved text does not adequately answer the question, say:
   "I could not find enough support for this answer in the supplied Pakistan
   cyber-law source." Then explain what is missing.
3. Clearly distinguish the wording of the law from interpretation.
4. Cite every material legal proposition using [PDF p. X].
5. Do not claim that an answer is legal advice. Recommend a qualified Pakistani
   lawyer for case-specific advice.
6. If the user asks for wrongdoing, evasion, unauthorized access, credential
   theft, malware, or other harmful cyber activity, do not provide operational
   instructions. You may explain the relevant legal risk at a high level.
7. Jurisdiction: Pakistan. Treat the supplied document as the primary source,
   even if general knowledge differs.
8. Technical level requested by the user: {technical_level}.
9. {size_instruction}

Return a useful, structured answer with headings/bullets when appropriate.
""".strip()

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED LEGAL SOURCE:
{context}
""".strip()

    response = client.responses.create(
        model=model_name,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


def retrieval_only_answer(question: str, results: List[Dict]) -> str:
    if not results:
        return (
            "I could not find enough support for this question in the supplied "
            "Pakistan cyber-law PDF. Try a more specific question or lower the "
            "similarity threshold."
        )

    lines = [
        "**Retrieval-only mode:** the app found these relevant legal passages."
    ]
    for r in results:
        excerpt = r["text"]
        if len(excerpt) > 650:
            excerpt = excerpt[:650].rsplit(" ", 1)[0] + "…"
        lines.append(
            f"\n**PDF page {r['page']} — similarity {r['score']:.3f}**\n"
            f"{excerpt}"
        )
    return "\n".join(lines)


# ----------------------- Sidebar -----------------------

st.sidebar.title("⚙️ RAG Controls")

st.sidebar.subheader("AI / Grok")
api_key = st.sidebar.text_input(
    "xAI API key",
    value=get_secret("XAI_API_KEY"),
    type="password",
    help="Prefer Streamlit Secrets / environment variables in deployment.",
)
grok_model = st.sidebar.text_input(
    "Grok model",
    value=get_secret("GROK_MODEL", DEFAULT_GROK_MODEL),
)
use_grok = st.sidebar.toggle("Use Grok for final answer", value=True)

technical_level = st.sidebar.selectbox(
    "Technicality level",
    ["Beginner", "Intermediate", "Technical", "Legal-professional"],
    index=1,
)
response_size = st.sidebar.select_slider(
    "Response size",
    options=["Short", "Medium", "Detailed"],
    value="Medium",
)
temperature = st.sidebar.slider(
    "Creativity / temperature",
    0.0,
    1.0,
    0.2,
    0.05,
)

st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider("Top-K passages", 2, 12, 6)
threshold = st.sidebar.slider(
    "Similarity threshold",
    0.00,
    0.90,
    0.25,
    0.01,
    help="Higher = stricter evidence matching.",
)
show_sources = st.sidebar.toggle("Show retrieved passages", value=True)

st.sidebar.subheader("Corpus")
drive_url = st.sidebar.text_input(
    "Google Drive PDF URL",
    value=DEFAULT_PDF_URL,
)
st.session_state["drive_url"] = drive_url

if st.sidebar.button("🔄 Re-download + rebuild index", use_container_width=True):
    st.session_state["force_refresh"] = True
    st.cache_resource.clear()
    st.rerun()

strict_mode = st.sidebar.toggle(
    "Strict legal-source mode",
    value=True,
    help="Keeps answers grounded in the retrieved PDF and avoids unsupported legal claims.",
)

st.sidebar.caption(
    "The app is informational and is not a substitute for advice from a "
    "qualified Pakistani lawyer."
)


# ----------------------- Startup RAG build -----------------------

st.title("⚖️ Pakistan Cyber Law RAG")
st.caption(
    "Streamlit • FAISS • sentence-transformers • Grok/xAI • Google Drive PDF"
)

st.markdown(
    """
<div class="disclaimer">
<b>Important:</b> This assistant is for legal-information and research support.
It is not legal advice. The supplied PDF is treated as the primary source.
For a real case, consult a qualified lawyer in Pakistan and verify the current
official Gazette / Pakistan Code text.
</div>
""",
    unsafe_allow_html=True,
)

try:
    with st.status("Preparing legal corpus and embeddings…", expanded=False) as status:
        pdf_path = download_pdf_from_drive()
        sig = corpus_signature(pdf_path)
        index, chunks, dimension = build_index(sig, str(pdf_path))
        embedder = load_embedder()
        status.update(
            label=f"Ready — {len(chunks)} chunks indexed ({dimension}D embeddings)",
            state="complete",
        )
except Exception as exc:
    st.error(f"Startup failed: {exc}")
    st.info(
        "For Google Drive, set the PDF to 'Anyone with the link' and ensure it "
        "is a real PDF. You can also replace the Drive URL in the sidebar."
    )
    st.stop()


# ----------------------- Chat -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("📌 Corpus status", expanded=False):
    st.write(f"**PDF:** `{pdf_path.name}`")
    st.write(f"**Indexed chunks:** {len(chunks)}")
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.write(f"**Vector index:** FAISS inner-product (normalized vectors)")
    st.write(f"**Grok model:** `{grok_model}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about Pakistan cyber law… e.g. 'What does the law say about unauthorized access?'"
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant legal provisions…"):
            results = retrieve(
                question,
                index,
                chunks,
                embedder,
                top_k=top_k,
                threshold=threshold,
            )

        if not results:
            answer = (
                "I could not find enough support for that question in the supplied "
                "Pakistan cyber-law PDF. Please ask a more specific question or "
                "reduce the similarity threshold."
            )
        elif use_grok and api_key:
            try:
                context = make_context(results)
                answer = grok_answer(
                    api_key=api_key,
                    model_name=grok_model,
                    question=question,
                    context=context,
                    technical_level=technical_level,
                    response_size=response_size,
                    temperature=temperature,
                )
            except Exception as exc:
                st.warning(
                    f"Grok request failed ({type(exc).__name__}). "
                    "Showing grounded retrieval results instead."
                )
                answer = retrieval_only_answer(question, results)
        else:
            if use_grok and not api_key:
                st.info(
                    "No xAI API key was provided, so the app is using retrieval-only mode."
                )
            answer = retrieval_only_answer(question, results)

        st.markdown(answer)

        if show_sources:
            with st.expander(f"📚 Retrieved evidence ({len(results)})", expanded=False):
                for r in results:
                    st.markdown(
                        f"**#{r['rank']} — PDF page {r['page']} — "
                        f"similarity {r['score']:.3f}**"
                    )
                    st.write(r["text"])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})


st.divider()
st.caption(
    "Source-grounding note: Pakistan Code's PECA page states that its content "
    "is for information purposes and may be under review; where doubt exists, "
    "consult the original Gazette notification. This app therefore exposes "
    "the retrieved PDF evidence rather than presenting itself as a legal authority."
)
