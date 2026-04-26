# =========================
# GST bare-law chatbot streamlit app
# =========================

from __future__ import annotations

import os
import re
import uuid
import importlib
import importlib.util
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


# =========================
# app config
# =========================

APP_TITLE = "GST Bare-Law Assistant"
APP_SUBTITLE = "Ask grounded questions from the GST Acts and Rules corpus."

# Final defaults based on our latest Batch 5 setup
DEFAULT_RETRIEVAL_TOP_K = 12
DEFAULT_CONTEXT_TOP_K = 4
DEFAULT_MAX_CHARS_PER_CHUNK = 2200
DEFAULT_MAX_NEW_TOKENS = 320


# =========================
# streamlit page setup
# =========================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# ui styling
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .disclaimer-box {
        border: 1px solid rgba(156, 163, 175, 0.35);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        background: rgba(156, 163, 175, 0.08);
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    .source-card {
        border: 1px solid rgba(156, 163, 175, 0.35);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.55rem;
        background: rgba(156, 163, 175, 0.06);
    }

    .source-citation {
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .source-heading {
        font-size: 0.88rem;
        color: #9CA3AF;
    }

    .small-muted {
        font-size: 0.85rem;
        color: #9CA3AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# backend loader
# =========================

def _load_answer_func_from_module(module_name: str) -> Optional[Callable[..., Dict[str, Any]]]:
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, "answer_gst_query", None)
        if callable(fn):
            return fn
    except Exception:
        return None

    return None


def _load_answer_func_from_py_file(file_path: str) -> Optional[Callable[..., Dict[str, Any]]]:
    if not os.path.exists(file_path):
        return None

    try:
        module_name = f"gst_backend_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, "answer_gst_query", None)
        if callable(fn):
            return fn

    except Exception:
        return None

    return None


@st.cache_resource(show_spinner=False)
def load_answer_gst_query() -> Callable[..., Dict[str, Any]]:
    """
    Loads the production backend function.

    Expected backend function:

    answer_gst_query(
        query: str,
        retrieval_top_k: int = 12,
        context_top_k: int = 4,
        max_chars_per_chunk: int = 2200,
        max_new_tokens: int = 320
    ) -> dict
    """

    candidate_modules = [
        os.getenv("GST_BACKEND_MODULE", "").strip(),
        "gst_rag_backend",
        "gst_backend",
        "rag_backend",
        "app_backend",
    ]

    for module_name in candidate_modules:
        if not module_name:
            continue

        fn = _load_answer_func_from_module(module_name)
        if fn is not None:
            return fn

    candidate_files = [
        "gst_rag_backend.py",
        "gst_backend.py",
        "rag_backend.py",
        "app_backend.py",
        "03_generation_and_evaluation.py",
        os.path.join("notebooks", "03_generation_and_evaluation.py"),
    ]

    for file_path in candidate_files:
        fn = _load_answer_func_from_py_file(file_path)
        if fn is not None:
            return fn

    raise ImportError(
        "Could not find answer_gst_query. "
        "Create gst_rag_backend.py and keep answer_gst_query inside it."
    )


# =========================
# query helpers
# =========================

def basic_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())


def extract_query_refs(query: str):
    """
    Lightweight section/rule reference detector used only for follow-up detection.
    This is not the main retrieval logic.
    """
    q = str(query or "").lower()

    section_refs = re.findall(
        r"(?:section|sec\.?|s\.?)\s*([0-9]+[a-zA-Z]?)",
        q,
    )

    rule_refs = re.findall(
        r"(?:rule|r\.?)\s*([0-9]+[a-zA-Z]?)",
        q,
    )

    return section_refs, rule_refs


# =========================
# public response cleaner
# =========================

def clean_public_answer(text: str) -> str:
    """
    Removes accidental prompt/context echo if the model repeats it.
    The UI should only show answer, legal basis, source, and note.
    """
    if not text:
        return ""

    text = str(text).strip()

    cut_markers = [
        "Retrieved GST context:",
        "CONTEXT PASSED TO GENERATOR",
        "You are a GST bare-law grounded assistant.",
        "Rules you must follow:",
        "[Source 1]",
        "Question:",
    ]

    for marker in cut_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    answer_pos = text.find("Answer:")
    if answer_pos != -1:
        text = text[answer_pos:].strip()

    return text.strip()


def format_public_sources(used_sources) -> List[Dict[str, str]]:
    """
    Converts internal source objects into website-safe citation objects.
    No chunk text, prompt, context, scores, or debug traces are exposed.
    """
    public_sources = []
    seen = set()

    for src in used_sources or []:
        if not isinstance(src, dict):
            continue

        citation = str(src.get("citation_label", "") or src.get("citation", "") or "").strip()
        heading = str(src.get("full_heading", "") or src.get("heading", "") or "").strip()
        doc_family = str(src.get("doc_family", "") or "").strip()
        section_or_rule = str(
            src.get("section_or_rule_number", "")
            or src.get("section_or_rule", "")
            or ""
        ).strip()

        if not citation and not heading:
            continue

        key = (citation, heading)
        if key in seen:
            continue

        seen.add(key)

        public_sources.append(
            {
                "citation": citation,
                "heading": heading,
                "doc_family": doc_family,
                "section_or_rule": section_or_rule,
            }
        )

    return public_sources


def build_public_chatbot_response(
    out: Dict[str, Any],
    original_query: str,
    resolved_query: Optional[str] = None,
) -> Dict[str, Any]:
    out = out or {}

    answer = clean_public_answer(out.get("final_answer", ""))

    if not answer:
        answer = "The answer is not clearly available in this GST bare-law corpus."

    return {
        "ok": True,
        "query": original_query,
        "resolved_query": resolved_query or original_query,
        "mode": out.get("mode", "unknown"),
        "answer": answer,
        "sources": format_public_sources(out.get("used_sources", [])),
    }


# =========================
# chatbot history + follow-up handling
# =========================

@dataclass
class ChatTurn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: List[Dict[str, str]] = field(default_factory=list)


def _last_user_question(history: List[ChatTurn]) -> str:
    for turn in reversed(history):
        if turn.role == "user":
            return turn.content
    return ""


def _last_assistant_sources(history: List[ChatTurn]) -> List[Dict[str, str]]:
    for turn in reversed(history):
        if turn.role == "assistant" and turn.sources:
            return turn.sources
    return []


def looks_like_followup_query(query: str) -> bool:
    q = str(query or "").lower().strip()
    tokens = basic_tokenize(q)

    followup_phrases = [
        "what about",
        "and what",
        "also",
        "explain more",
        "explain this",
        "what does it mean",
        "what is the deadline",
        "what is the time limit",
        "what happens then",
        "in this section",
        "in that section",
        "same section",
        "same rule",
        "this provision",
        "that provision",
    ]

    pronoun_terms = {"it", "this", "that", "these", "those", "same", "above"}

    section_refs, rule_refs = extract_query_refs(q)
    has_explicit_ref = bool(section_refs or rule_refs)

    if has_explicit_ref:
        return False

    if any(phrase in q for phrase in followup_phrases):
        return True

    if len(tokens) <= 8 and any(token in pronoun_terms for token in tokens):
        return True

    return False


def resolve_followup_query(query: str, history: List[ChatTurn]) -> str:
    """
    Builds retrieval-friendly follow-up queries.

    This does not inject legal facts or answers.
    It only carries previous question and source citation/heading.
    """
    if not history or not looks_like_followup_query(query):
        return query

    q = str(query or "").lower().strip()

    previous_question = _last_user_question(history)
    previous_sources = _last_assistant_sources(history)

    source_hint_parts = []

    for src in previous_sources[:2]:
        citation = src.get("citation", "")
        heading = src.get("heading", "")
        hint = f"{citation} {heading}".strip()

        if hint:
            source_hint_parts.append(hint)

    source_hint = " | ".join(source_hint_parts)

    deadline_like = any(term in q for term in [
        "deadline",
        "time limit",
        "by when",
        "last date",
        "cut off",
        "cutoff",
        "within what time"
    ])

    consequence_like = any(term in q for term in [
        "what happens",
        "what happens then",
        "consequence",
        "effect",
        "fails",
        "does not",
        "not pay"
    ])

    meaning_like = any(term in q for term in [
        "what does it mean",
        "explain this",
        "explain more",
        "meaning"
    ])

    if source_hint:
        if deadline_like:
            return f"deadline or time limit mentioned in {source_hint}"

        if consequence_like:
            return f"consequence or effect mentioned in {source_hint}. Follow-up question: {query}"

        if meaning_like:
            return f"explain the meaning of the provision in {source_hint}. Follow-up question: {query}"

        return f"{query}. Relevant previous GST source: {source_hint}"

    if previous_question:
        return f"{previous_question}. Follow-up: {query}"

    return query

# =========================
# gst chatbot session
# =========================

class GSTChatSession:
    def __init__(
        self,
        answer_func: Callable[..., Dict[str, Any]],
        session_id: Optional[str] = None,
        max_history_turns: int = 12,
    ):
        self.answer_func = answer_func
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history_turns = max_history_turns
        self.history: List[ChatTurn] = []

    def ask(
        self,
        query: str,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        context_top_k: int = DEFAULT_CONTEXT_TOP_K,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Dict[str, Any]:

        query = str(query or "").strip()

        if not query:
            return {
                "ok": False,
                "query": query,
                "resolved_query": query,
                "mode": "empty_query",
                "answer": "Please enter a GST law question.",
                "sources": [],
            }

        try:
            resolved_query = resolve_followup_query(query, self.history)

            out = self.answer_func(
                query=resolved_query,
                retrieval_top_k=retrieval_top_k,
                context_top_k=context_top_k,
                max_chars_per_chunk=max_chars_per_chunk,
                max_new_tokens=max_new_tokens,
            )

            public_response = build_public_chatbot_response(
                out=out,
                original_query=query,
                resolved_query=resolved_query,
            )

        except Exception as e:
            public_response = {
                "ok": False,
                "query": query,
                "resolved_query": query,
                "mode": "system_error",
                "answer": "Sorry, the GST assistant could not process this question right now.",
                "sources": [],
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        self.history.append(ChatTurn(role="user", content=query))
        self.history.append(
            ChatTurn(
                role="assistant",
                content=public_response["answer"],
                sources=public_response.get("sources", []),
            )
        )

        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        return public_response

    def get_history_for_ui(self) -> List[Dict[str, Any]]:
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "sources": turn.sources,
            }
            for turn in self.history
        ]

    def reset(self) -> Dict[str, str]:
        self.history = []
        return {
            "session_id": self.session_id,
            "status": "reset_done",
        }


# =========================
# ui render helpers
# =========================

def render_sources(sources: List[Dict[str, str]]) -> None:
    if not sources:
        return

    st.markdown("**Sources**")

    for idx, src in enumerate(sources, start=1):
        citation = src.get("citation", "") or "Source"
        heading = src.get("heading", "")
        doc_family = src.get("doc_family", "")
        section_or_rule = src.get("section_or_rule", "")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-citation">Source {idx}: {citation}</div>
                <div class="source-heading">{heading}</div>
                <div class="small-muted">{doc_family} {section_or_rule}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_example_questions() -> None:
    st.markdown("Try examples:")

    examples = [
        "What does section 16 say?",
        "Section 16 read with rule 36 for claiming input tax credit",
        "Difference between section 22 and section 24 for registration",
        "What does the law say about delayed refunds?",
        "Inter state supply vs zero rated supply",
        "How should a business practically structure registration across states under GST?",
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_example_query"] = example
            st.rerun()


# =========================
# initialize backend and session
# =========================

try:
    answer_gst_query = load_answer_gst_query()
except Exception as backend_error:
    st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.error("Backend not found.")

    st.markdown(
        """
        Streamlit app is ready, but it needs your RAG backend function.

        Create a file named `gst_rag_backend.py` in the same folder as `streamlit_app.py`
        and keep your final `answer_gst_query()` function inside it.
        """
    )

    st.code(
        """
def answer_gst_query(
    query: str,
    retrieval_top_k: int = 12,
    context_top_k: int = 4,
    max_chars_per_chunk: int = 2200,
    max_new_tokens: int = 320,
):
    # Your existing RAG pipeline should return:
    # {
    #   "mode": "...",
    #   "final_answer": "...",
    #   "used_sources": [...]
    # }
    pass
        """,
        language="python",
    )

    with st.expander("Technical error"):
        st.exception(backend_error)

    st.stop()


if "chat_session" not in st.session_state:
    st.session_state.chat_session = GSTChatSession(answer_func=answer_gst_query)

if "last_response" not in st.session_state:
    st.session_state.last_response = None


# =========================
# sidebar
# =========================

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    retrieval_top_k = st.slider(
        "Retrieval top k",
        min_value=3,
        max_value=20,
        value=DEFAULT_RETRIEVAL_TOP_K,
        step=1,
    )

    context_top_k = st.slider(
        "Context top k",
        min_value=1,
        max_value=8,
        value=DEFAULT_CONTEXT_TOP_K,
        step=1,
    )

    max_chars_per_chunk = st.slider(
        "Max chars per chunk",
        min_value=800,
        max_value=5000,
        value=DEFAULT_MAX_CHARS_PER_CHUNK,
        step=100,
    )

    max_new_tokens = st.slider(
        "Max answer tokens",
        min_value=120,
        max_value=800,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=20,
    )

    st.divider()

    show_debug = st.toggle("Show debug info", value=False)
    show_examples = st.toggle("Show example questions", value=True)

    st.divider()

    if st.button("Reset chat", use_container_width=True):
        st.session_state.chat_session.reset()
        st.session_state.last_response = None
        st.rerun()

    st.markdown(
        """
        <div class="small-muted">
        Keep debug mode off for final public demo.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# header
# =========================

st.markdown(f'<div class="main-title">⚖️ {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer-box">
    This assistant answers only from the uploaded GST bare-law corpus.
    It does not provide legal advice, portal workflow guidance, case-law interpretation,
    or latest notification tracking unless such text exists in the corpus.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# example questions
# =========================

if show_examples and len(st.session_state.chat_session.history) == 0:
    render_example_questions()


# =========================
# render existing chat history
# =========================

for turn in st.session_state.chat_session.get_history_for_ui():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        if turn["role"] == "assistant":
            render_sources(turn.get("sources", []))


# =========================
# handle example query
# =========================

pending_query = st.session_state.pop("pending_example_query", None)


# =========================
# chat input
# =========================

user_query = pending_query or st.chat_input("Ask a GST bare-law question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching GST law and generating grounded answer..."):
            response = st.session_state.chat_session.ask(
                query=user_query,
                retrieval_top_k=retrieval_top_k,
                context_top_k=context_top_k,
                max_chars_per_chunk=max_chars_per_chunk,
                max_new_tokens=max_new_tokens,
            )

        st.session_state.last_response = response

        st.markdown(response.get("answer", ""))

        render_sources(response.get("sources", []))

        if show_debug:
            st.divider()
            st.markdown("### Debug info")

            st.json(
                {
                    "ok": response.get("ok"),
                    "mode": response.get("mode"),
                    "query": response.get("query"),
                    "resolved_query": response.get("resolved_query"),
                    "source_count": len(response.get("sources", [])),
                    "error": response.get("error"),
                }
            )

            if response.get("traceback"):
                with st.expander("Traceback"):
                    st.code(response["traceback"], language="text")