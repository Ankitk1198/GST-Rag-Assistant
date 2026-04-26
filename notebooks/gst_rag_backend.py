#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from collections import Counter, defaultdict
import json
import os
import pickle
import re
import uuid

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModel


# =========================
# PRODUCTION DEFAULTS
# =========================

DEFAULT_RETRIEVAL_TOP_K = 12
DEFAULT_CONTEXT_TOP_K = 4
DEFAULT_MAX_CHARS_PER_CHUNK = 2200
DEFAULT_MAX_NEW_TOKENS = 320

E5_MODEL_NAME = os.getenv("GST_RAG_EMBED_MODEL", "intfloat/e5-large-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma2:9b")


# =========================
# PROJECT ROOT RESOLUTION
# =========================

def find_project_root() -> Path:
    env_root = os.getenv("GST_RAG_PROJECT_ROOT")
    candidates = []

    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    file_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    cwd = Path.cwd().resolve()

    candidates.extend([
        cwd,
        cwd.parent,
        file_dir,
        file_dir.parent,
        file_dir.parent.parent,
    ])

    for path in candidates:
        if (
            (path / "chunks" / "retrieval_records_dense_v1.json").exists()
            and (path / "indexes" / "dense_embeddings_e5_v1.npy").exists()
        ):
            return path

    return cwd


PROJECT_ROOT = find_project_root()

DENSE_RECORDS_PATH = PROJECT_ROOT / "chunks" / "retrieval_records_dense_v1.json"
DENSE_EMBEDDINGS_PATH = PROJECT_ROOT / "indexes" / "dense_embeddings_e5_v1.npy"
DENSE_IDMAP_PATH = PROJECT_ROOT / "indexes" / "dense_idmap_e5_v1.json"


# =========================
# BASIC TEXT HELPERS
# =========================

def basic_tokenize(text: str):
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


def normalize_ref_number(ref_number):
    return re.sub(r"[^0-9A-Za-z]+", "", str(ref_number)).lower()


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").lower()).strip()


def _qnorm_for_generation(query: str) -> str:
    return _normalize_query(query)


def _prod_norm_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _prod_unique_keep_order(items):
    seen = set()
    out = []

    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)

    return out


def _prod_extract_number_refs(query: str, labels):
    q = _prod_norm_query(query)
    label_pat = "|".join(re.escape(x) for x in labels)

    pattern = rf"\b(?:{label_pat})\s*((?:\d+[a-z]?)(?:\s*(?:,|and|&|or)\s*\d+[a-z]?)*)\b"

    nums = []
    for m in re.finditer(pattern, q):
        nums.extend(re.findall(r"\d+[a-z]?", m.group(1)))

    return _prod_unique_keep_order(nums)


def extract_query_refs(query: str):
    section_refs = _prod_extract_number_refs(query, ["section", "sections", "sec", "s."])
    rule_refs = _prod_extract_number_refs(query, ["rule", "rules", "r."])
    return section_refs, rule_refs


# =========================
# SPARSE INDEX
# =========================

def build_sparse_text(record):
    alias_text = " ".join(record.get("reference_aliases", []))

    pieces = [
        record.get("doc_family", ""),
        record.get("doc_type", ""),
        record.get("legal_unit_type", ""),
        record.get("chapter_number", ""),
        record.get("chapter_title", ""),
        record.get("section_or_rule_number", ""),
        record.get("section_or_rule_title", ""),
        record.get("full_heading", ""),
        alias_text,
        record.get("chunk_text", "")
    ]

    return "\n".join([p for p in pieces if p])


def build_heading_text(record):
    pieces = [
        record.get("doc_family", ""),
        record.get("doc_type", ""),
        record.get("legal_unit_type", ""),
        record.get("chapter_number", ""),
        record.get("chapter_title", ""),
        record.get("section_or_rule_number", ""),
        record.get("section_or_rule_title", ""),
        record.get("full_heading", "")
    ]

    return "\n".join([p for p in pieces if p])


class SimpleBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)

        self.doc_len = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if len(self.doc_len) > 0 else 0.0

        self.term_freqs = []
        doc_freq = Counter()

        for doc in corpus_tokens:
            tf = Counter(doc)
            self.term_freqs.append(tf)

            for term in tf.keys():
                doc_freq[term] += 1

        self.idf = {}

        for term, df in doc_freq.items():
            self.idf[term] = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def get_scores(self, query_tokens, candidate_indices=None):
        if candidate_indices is None:
            candidate_indices = list(range(self.N))

        scores = np.zeros(len(candidate_indices), dtype=np.float32)
        unique_query_terms = list(dict.fromkeys(query_tokens))

        if self.avgdl == 0:
            return scores

        for term in unique_query_terms:
            if term not in self.idf:
                continue

            term_idf = self.idf[term]

            for pos, doc_idx in enumerate(candidate_indices):
                tf = self.term_freqs[doc_idx].get(term, 0)

                if tf == 0:
                    continue

                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                score = term_idf * ((tf * (self.k1 + 1.0)) / denom)
                scores[pos] += score

        return scores


# =========================
# LOAD RETRIEVAL ARTIFACTS
# =========================

def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if not DENSE_RECORDS_PATH.exists():
    raise FileNotFoundError(f"Missing retrieval records: {DENSE_RECORDS_PATH}")

if not DENSE_EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(f"Missing dense embeddings: {DENSE_EMBEDDINGS_PATH}")

if not DENSE_IDMAP_PATH.exists():
    raise FileNotFoundError(f"Missing dense id map: {DENSE_IDMAP_PATH}")

retrieval_records_dense = _load_json(DENSE_RECORDS_PATH)
dense_embeddings = np.load(DENSE_EMBEDDINGS_PATH)
dense_idmap = _load_json(DENSE_IDMAP_PATH)

if len(dense_idmap) > 0 and isinstance(dense_idmap[0], dict):
    dense_idmap_chunk_ids = [item["chunk_id"] for item in dense_idmap]
else:
    dense_idmap_chunk_ids = dense_idmap

if not (len(retrieval_records_dense) == dense_embeddings.shape[0] == len(dense_idmap_chunk_ids)):
    raise ValueError("Dense artifacts are misaligned.")

chunk_id_to_dense_idx = {chunk_id: i for i, chunk_id in enumerate(dense_idmap_chunk_ids)}

sparse_texts = [build_sparse_text(rec) for rec in retrieval_records_dense]
sparse_tokens = [basic_tokenize(text) for text in sparse_texts]
bm25_index = SimpleBM25(sparse_tokens)

heading_texts = [build_heading_text(rec) for rec in retrieval_records_dense]
heading_tokens = [basic_tokenize(text) for text in heading_texts]
heading_bm25_index = SimpleBM25(heading_tokens)

definition_global_indices = [
    i for i, rec in enumerate(retrieval_records_dense)
    if rec.get("doc_family") == "CGST Act"
    and str(rec.get("section_or_rule_number")) == "2"
]

definition_texts = [
    build_sparse_text(retrieval_records_dense[i])
    for i in definition_global_indices
]

definition_tokens = [basic_tokenize(text) for text in definition_texts]
definition_bm25_index = SimpleBM25(definition_tokens)


# =========================
# LAZY E5 ENCODER
# =========================

_tokenizer = None
_model = None
_device = None


def get_encoder():
    global _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    _model = AutoModel.from_pretrained(E5_MODEL_NAME).to(_device)
    _model.eval()

    return _tokenizer, _model, _device


def average_pool(last_hidden_state, attention_mask):
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def encode_queries_e5(queries, batch_size=8, max_length=512):
    tokenizer, model, device = get_encoder()
    all_embeddings = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        pooled = average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


# =========================
# QUERY PROFILE
# =========================

QUERY_PROFILE_STOPWORDS = {
    "what", "does", "do", "say", "says", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "can", "could", "should", "would",
    "i", "we", "you", "the", "a", "an", "of", "for", "to", "under", "about",
    "and", "or", "with", "on", "in", "by", "from", "as", "this", "that",
    "gst", "law", "provision", "provisions"
}

FOCUS_DROP_TOKENS = {
    "claim", "claiming", "claimed", "liable", "liability", "eligible", "eligibility",
    "opt", "opted", "register", "registered", "required", "requirement",
    "difference", "between", "compare", "versus", "vs", "add", "adds"
}

_DEFINED_TERM_PHRASES = {
    "aggregate turnover",
    "composite supply",
    "mixed supply",
    "principal supply",
    "reverse charge",
    "zero rated supply",
    "zero-rated supply",
    "exempt supply",
    "non taxable supply",
    "non-taxable supply",
    "input tax credit",
    "tax invoice",
    "turnover in state",
    "turnover in union territory",
}

_CONCEPT_PHRASE_MAP = [
    ("input tax credit", "input_tax_credit"),
    ("itc", "input_tax_credit"),
    ("turnover in state", "turnover_definition"),
    ("turnover in union territory", "turnover_definition"),
    ("aggregate turnover", "aggregate_turnover"),
    ("composition levy", "composition_levy"),
    ("composition scheme", "composition_levy"),
    ("tax invoice", "tax_invoice"),
    ("invoice particulars", "tax_invoice"),
    ("refund application", "refund"),
    ("refund", "refund"),
    ("registration", "registration"),
    ("register", "registration"),
    ("reverse charge", "reverse_charge"),
    ("zero rated supply", "zero_rated_supply"),
    ("zero-rated supply", "zero_rated_supply"),
    ("inter-state supply", "inter_intra_supply"),
    ("intra-state supply", "inter_intra_supply"),
    ("inter state supply", "inter_intra_supply"),
    ("intra state supply", "inter_intra_supply"),
    ("composite supply", "supply_classification"),
    ("mixed supply", "supply_classification"),
    ("principal supply", "supply_classification"),
    ("exempt supply", "definition_supply_types"),
    ("non-taxable supply", "definition_supply_types"),
    ("non taxable supply", "definition_supply_types"),
]

_REGISTRATION_EXCEPTION_TERMS = {
    "not liable",
    "need not register",
    "who need not register",
    "who is not liable",
    "not required to register",
    "exception",
    "exceptions",
    "exempt from registration",
}

_REGISTRATION_LIABILITY_TERMS = {
    "liable",
    "liable for registration",
    "who is liable",
    "who must register",
    "registration required",
    "required to register",
    "need to register",
    "threshold",
}

_CONDITION_TERMS = {
    "when can",
    "conditions",
    "eligibility",
    "eligible",
    "who can",
    "who must",
    "who is liable",
    "liable",
    "when is",
    "when does",
    "can i claim",
    "can i take",
    "when do i need",
    "subject to",
    "opted",
    "opt for",
    "threshold",
}

_PROCEDURAL_TERMS = {
    "procedure",
    "process",
    "application",
    "particulars",
    "documentary",
    "documents",
    "evidence",
    "form",
    "format",
    "furnish",
    "issue of invoice",
    "tax invoice particulars",
    "how is",
}


def _clean_phrase(text: str) -> str:
    toks = [
        t for t in basic_tokenize(text)
        if t not in QUERY_PROFILE_STOPWORDS
        and t not in {"act", "acts", "rule", "rules", "section", "sec", "s", "r", "cgst", "igst", "utgst"}
    ]
    toks = [t for t in toks if t not in FOCUS_DROP_TOKENS]
    return " ".join(toks).strip()


def _extract_comparison_sides(query: str):
    q = _normalize_query(query)

    patterns = [
        r"\b(.+?)\s+vs\.?\s+(.+)$",
        r"\b(.+?)\s+versus\s+(.+)$",
        r"\bdifference between\s+(.+?)\s+and\s+(.+)$",
        r"\bcompare\s+(.+?)\s+and\s+(.+)$",
    ]

    for pat in patterns:
        m = re.search(pat, q)
        if m:
            left = _clean_phrase(m.group(1))
            right = _clean_phrase(m.group(2))
            if left and right:
                return left, right

    return None, None


def _phrase_present(query: str, phrase_set) -> bool:
    q = query.lower()
    return any(p in q for p in phrase_set)


def _extract_focus_phrases(query: str, left_phrase=None, right_phrase=None):
    phrases = []

    if left_phrase:
        phrases.append(left_phrase)

    if right_phrase:
        phrases.append(right_phrase)

    q = _normalize_query(query)

    for phrase, _concept in _CONCEPT_PHRASE_MAP:
        if phrase in q:
            phrases.append(phrase)

    tokens = [
        t for t in basic_tokenize(q)
        if t not in QUERY_PROFILE_STOPWORDS
        and t not in {"act", "acts", "rule", "rules", "section", "sec", "s", "r", "cgst", "igst", "utgst"}
        and t not in FOCUS_DROP_TOKENS
    ]

    if len(tokens) >= 3:
        phrases.append(" ".join(tokens[:3]))
    if len(tokens) >= 2:
        phrases.append(" ".join(tokens[:2]))
    if len(tokens) >= 1:
        phrases.append(tokens[0])

    seen = set()
    clean = []

    for p in phrases:
        p = p.strip()
        if p and p not in seen:
            clean.append(p)
            seen.add(p)

    return clean[:5]


def _detect_primary_concept(query: str):
    q = _normalize_query(query)

    for phrase, concept in _CONCEPT_PHRASE_MAP:
        if phrase in q:
            return concept

    return None


def _is_defined_term_phrase(phrase: str) -> bool:
    if not phrase:
        return False

    phrase = _normalize_query(phrase)

    if phrase in _DEFINED_TERM_PHRASES:
        return True

    phrase_no_hyphen = phrase.replace("-", " ")
    defined_no_hyphen = {p.replace("-", " ") for p in _DEFINED_TERM_PHRASES}

    return phrase_no_hyphen in defined_no_hyphen


def _detect_explicit_family_hints(query: str):
    q = _normalize_query(query)
    hints = []

    if "cgst" in q or "central goods and services tax" in q:
        hints.append("CGST Act")

    if "igst" in q or "integrated goods and services tax" in q:
        hints.append("IGST Act")

    if "utgst" in q or "union territory goods and services tax" in q:
        hints.append("UTGST Act")

    if "compensation act" in q or "compensation cess" in q:
        hints.append("Compensation Act")

    if "cgst rule" in q or "cgst rules" in q:
        hints.append("CGST Rules")

    if "igst rule" in q or "igst rules" in q:
        hints.append("IGST Rules")

    return _prod_unique_keep_order(hints)


@lru_cache(maxsize=2048)
def infer_query_profile(query: str) -> dict:
    q = _normalize_query(query)
    tokens = basic_tokenize(q)
    token_set = set(tokens)

    section_refs, rule_refs = extract_query_refs(q)
    left_phrase, right_phrase = _extract_comparison_sides(q)

    primary_concept = _detect_primary_concept(q)

    portal_like = any(t in token_set for t in {
        "portal", "gstn", "login", "upload", "submit", "filing", "file", "workflow"
    })

    operational_like = any(t in token_set for t in {
        "how", "steps", "step", "process", "procedure"
    }) or any(term in q for term in {
        "how do i file", "step by step", "compliance process"
    })

    latest_regulatory_like = (
        any(t in token_set for t in {"latest", "recent", "current", "new", "newest", "updated"})
        and any(t in token_set for t in {
            "circular", "notification", "notifications",
            "amendment", "update", "updates",
            "press", "release", "advisory"
        })
    )

    judicial_like = (
        any(t in token_set for t in {
            "court", "courts", "judgment", "judgement",
            "judgments", "judgements", "precedent"
        })
        or "case law" in q
        or "interpretation by court" in q
        or "what do courts say" in q
    )

    if "advance ruling" in q:
        judicial_like = False

    if (section_refs or rule_refs) and not any(term in q for term in [
        "case law", "court", "courts", "judgment", "judgement",
        "judgments", "judgements", "precedent", "what do courts say"
    ]):
        judicial_like = False

    scope = "in_scope"
    out_of_scope_reason = None

    if judicial_like:
        scope = "out_of_scope"
        out_of_scope_reason = "case_law_or_judicial_interpretation"
    elif latest_regulatory_like:
        scope = "out_of_scope"
        out_of_scope_reason = "latest_regulatory_update"
    elif portal_like and operational_like:
        scope = "out_of_scope"
        out_of_scope_reason = "portal_or_operational_workflow"

    comparison_signal = (
        bool(left_phrase and right_phrase)
        or " vs " in q
        or " versus " in q
        or "difference between" in q
        or "compare " in q
    )

    act_rule_signal = (
        (("act" in token_set or "acts" in token_set) and ("rule" in token_set or "rules" in token_set))
        or "act vs rule" in q
        or "act versus rule" in q
        or "what do the rules add" in q
        or "rules add" in q
        or "act says" in q
        or "act say" in q
        or "read with rule" in q
    )

    definition_signal = (
        q.startswith("what is ")
        or q.startswith("what are ")
        or q.startswith("define ")
        or q.startswith("meaning of ")
    )

    procedural_signal = any(term in q for term in _PROCEDURAL_TERMS)
    condition_signal = any(term in q for term in _CONDITION_TERMS)

    if scope == "out_of_scope":
        intent = "out_of_scope"
    elif act_rule_signal:
        intent = "act_vs_rule_linking"
    elif comparison_signal:
        intent = "comparison"
    elif section_refs or rule_refs:
        intent = "reference"
    elif definition_signal:
        intent = "definition_lookup"
    elif procedural_signal:
        intent = "procedural_rule"
    elif condition_signal:
        intent = "conditions_or_eligibility"
    else:
        intent = "semantic"

    registration_mode = None

    if primary_concept == "registration" or "registration" in token_set or "register" in token_set:
        if _phrase_present(q, _REGISTRATION_EXCEPTION_TERMS):
            registration_mode = "exception"
        elif _phrase_present(q, _REGISTRATION_LIABILITY_TERMS):
            registration_mode = "liability"

    definition_comparison = bool(
        intent == "comparison"
        and left_phrase
        and right_phrase
        and _is_defined_term_phrase(left_phrase)
        and _is_defined_term_phrase(right_phrase)
        and not section_refs
        and not rule_refs
    )

    prefer_definition_section = bool(
        intent == "definition_lookup"
        or definition_comparison
    )

    preferred_doc_type = None

    if intent == "definition_lookup":
        preferred_doc_type = "Act"
    elif intent == "procedural_rule":
        preferred_doc_type = "Rules"
    elif intent == "conditions_or_eligibility":
        preferred_doc_type = "Act"
    elif intent == "reference":
        if rule_refs:
            preferred_doc_type = "Rules"
        elif section_refs:
            preferred_doc_type = "Act"
    elif intent == "comparison" and not rule_refs:
        preferred_doc_type = "Act"
    elif intent == "act_vs_rule_linking":
        preferred_doc_type = "Both"

    family_hints = _detect_explicit_family_hints(query)

    normalized_targets = []
    normalized_targets.extend([f"section_{normalize_ref_number(x)}" for x in section_refs])
    normalized_targets.extend([f"rule_{normalize_ref_number(x)}" for x in rule_refs])

    return {
        "raw_query": query,
        "scope": scope,
        "out_of_scope_reason": out_of_scope_reason,
        "intent": intent,
        "query_type": intent,
        "primary_concept": primary_concept,
        "section_refs": section_refs,
        "rule_refs": rule_refs,
        "comparison_left": left_phrase,
        "comparison_right": right_phrase,
        "focus_phrases": _extract_focus_phrases(query=q, left_phrase=left_phrase, right_phrase=right_phrase),
        "preferred_doc_type": preferred_doc_type,
        "definition_comparison": definition_comparison,
        "prefer_definition_section": prefer_definition_section,
        "prefer_opening_provision": True,
        "registration_mode": registration_mode,
        "doc_family_hints": family_hints,
        "normalized_targets": _prod_unique_keep_order(normalized_targets),
        "explicit_family_mentioned": bool(family_hints),
        "prefer_cgst_default": False,
    }


def is_out_of_scope_query(query: str) -> bool:
    return infer_query_profile(query)["scope"] == "out_of_scope"


def looks_like_definition_query(query: str) -> bool:
    q = query.lower().strip()
    return q.startswith("what is ") or q.startswith("define ") or q.startswith("meaning of ")


def looks_like_definition_generation_query(query: str) -> bool:
    return infer_query_profile(query)["intent"] == "definition_lookup"


def looks_like_procedural_rule_query(query: str) -> bool:
    return infer_query_profile(query)["intent"] == "procedural_rule"


def looks_like_act_rule_link_query(query: str) -> bool:
    return infer_query_profile(query)["intent"] == "act_vs_rule_linking"


def looks_like_condition_query(query: str) -> bool:
    return infer_query_profile(query)["intent"] == "conditions_or_eligibility"


def looks_like_comparison_query(query: str) -> bool:
    return infer_query_profile(query)["intent"] == "comparison"


# =========================
# QUERY STRUCTURE FOR RETRIEVAL
# =========================

RETRIEVAL_STOPWORDS = {
    "what", "when", "where", "who", "why", "how", "is", "are", "was", "were",
    "can", "i", "we", "you", "the", "a", "an", "of", "for", "to", "under",
    "in", "on", "and", "or", "with", "about", "do", "does", "did", "be",
    "by", "from", "as", "it", "this", "that"
}


def expanded_query_tokens(query: str):
    tokens = basic_tokenize(query)

    cleaned = {
        t for t in tokens
        if t not in RETRIEVAL_STOPWORDS
        and t not in {"section", "sec", "rule", "s", "r"}
    }

    if "claim" in cleaned:
        cleaned.update({"eligibility", "take", "avail"})

    if "itc" in cleaned:
        cleaned.update({"input", "tax", "credit", "eligibility", "take", "avail"})

    if {"input", "tax", "credit"}.issubset(cleaned):
        cleaned.update({"itc", "eligibility", "take", "avail"})

    return cleaned


def token_overlap_count(query_tokens, text):
    if not text:
        return 0

    text_tokens = set(basic_tokenize(text))
    return len(query_tokens & text_tokens)


def parse_query_structure(query: str):
    q = query.lower()

    section_refs, rule_refs = extract_query_refs(q)

    has_cgst = ("cgst" in q) or ("central goods and services tax" in q)
    has_igst = ("igst" in q) or ("integrated goods and services tax" in q)
    has_utgst = ("utgst" in q) or ("union territory goods and services tax" in q)
    has_comp = ("compensation" in q) and ("states" in q or "state" in q or "cess" in q)

    explicit_family_mentioned = any([has_cgst, has_igst, has_utgst, has_comp])

    doc_family_hints = set()

    if section_refs:
        if has_cgst:
            doc_family_hints.add("CGST Act")
        elif has_igst:
            doc_family_hints.add("IGST Act")
        elif has_utgst:
            doc_family_hints.add("UTGST Act")
        elif has_comp:
            doc_family_hints.add("Compensation Act")
        else:
            doc_family_hints.update(["CGST Act", "IGST Act", "UTGST Act", "Compensation Act"])

    if rule_refs:
        if has_igst:
            doc_family_hints.add("IGST Rules")
        else:
            doc_family_hints.add("CGST Rules")

    if not section_refs and not rule_refs:
        if has_cgst:
            doc_family_hints.update(["CGST Act", "CGST Rules"])
        if has_igst:
            doc_family_hints.update(["IGST Act", "IGST Rules"])
        if has_utgst:
            doc_family_hints.add("UTGST Act")
        if has_comp:
            doc_family_hints.add("Compensation Act")

    normalized_targets = set()

    for s in section_refs:
        normalized_targets.add(f"section_{normalize_ref_number(s)}")

    for r in rule_refs:
        normalized_targets.add(f"rule_{normalize_ref_number(r)}")

    if section_refs or rule_refs:
        query_type = "reference"
    else:
        heading_like_terms = [
            "input tax credit", "blocked credit", "place of supply", "registration",
            "refund", "tax invoice", "composition levy", "reverse charge",
            "zero rated supply", "mixed supply", "composite supply"
        ]

        if any(term in q for term in heading_like_terms):
            query_type = "heading_or_topic"
        else:
            query_type = "semantic"

    prefer_cgst_default = bool(section_refs) and not explicit_family_mentioned and not rule_refs

    return {
        "raw_query": query,
        "query_type": query_type,
        "section_refs": section_refs,
        "rule_refs": rule_refs,
        "doc_family_hints": sorted(doc_family_hints),
        "normalized_targets": sorted(normalized_targets),
        "explicit_family_mentioned": explicit_family_mentioned,
        "prefer_cgst_default": prefer_cgst_default
    }


def build_branch_topic_query(query: str) -> str:
    drop_terms = {
        "what", "does", "the", "act", "rule", "rules", "say", "add",
        "about", "and", "do"
    }

    toks = [t for t in basic_tokenize(query) if t not in drop_terms]
    return " ".join(toks) if toks else query


def get_candidate_indices(query_info):
    indices = list(range(len(retrieval_records_dense)))

    if query_info["doc_family_hints"]:
        allowed = set(query_info["doc_family_hints"])
        indices = [
            i for i in indices
            if retrieval_records_dense[i].get("doc_family") in allowed
        ]

    if query_info["normalized_targets"]:
        targets = set(query_info["normalized_targets"])
        ref_filtered = [
            i for i in indices
            if retrieval_records_dense[i].get("normalized_section_or_rule_number") in targets
        ]

        if ref_filtered:
            indices = ref_filtered

    return indices


# =========================
# DENSE + SPARSE RETRIEVAL
# =========================

def dense_search(query, candidate_indices=None, top_k=20):
    query_text = f"query: {query}"
    query_vec = encode_queries_e5([query_text])[0]

    if candidate_indices is None:
        candidate_indices = list(range(len(retrieval_records_dense)))

    candidate_matrix = dense_embeddings[candidate_indices]
    scores = candidate_matrix @ query_vec

    ranked_positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, pos in enumerate(ranked_positions, start=1):
        global_idx = candidate_indices[pos]

        results.append({
            "rank": rank,
            "dense_score": float(scores[pos]),
            "global_idx": global_idx,
            "chunk_id": retrieval_records_dense[global_idx]["chunk_id"]
        })

    return results


def sparse_search(query, candidate_indices=None, top_k=20):
    query_tokens = basic_tokenize(query)

    if candidate_indices is None:
        candidate_indices = list(range(len(retrieval_records_dense)))

    scores = bm25_index.get_scores(query_tokens, candidate_indices=candidate_indices)
    ranked_positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, pos in enumerate(ranked_positions, start=1):
        global_idx = candidate_indices[pos]

        results.append({
            "rank": rank,
            "sparse_score": float(scores[pos]),
            "global_idx": global_idx,
            "chunk_id": retrieval_records_dense[global_idx]["chunk_id"]
        })

    return results


def heading_search(query, candidate_indices=None, top_k=20):
    query_tokens = list(expanded_query_tokens(query))

    if not query_tokens:
        query_tokens = basic_tokenize(query)

    if candidate_indices is None:
        candidate_indices = list(range(len(retrieval_records_dense)))

    scores = heading_bm25_index.get_scores(query_tokens, candidate_indices=candidate_indices)
    ranked_positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, pos in enumerate(ranked_positions, start=1):
        global_idx = candidate_indices[pos]

        results.append({
            "rank": rank,
            "heading_score": float(scores[pos]),
            "global_idx": global_idx,
            "chunk_id": retrieval_records_dense[global_idx]["chunk_id"]
        })

    return results


def definition_search(query, top_k=20):
    profile = infer_query_profile(query)

    is_definition_style = looks_like_definition_query(query)
    is_definition_comparison = profile.get("definition_comparison", False)

    if not (is_definition_style or is_definition_comparison):
        return []

    if len(definition_global_indices) == 0:
        return []

    query_tokens = list(expanded_query_tokens(query))

    if not query_tokens:
        query_tokens = basic_tokenize(query)

    scores = definition_bm25_index.get_scores(query_tokens)
    ranked_positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, local_pos in enumerate(ranked_positions, start=1):
        global_idx = definition_global_indices[local_pos]

        results.append({
            "rank": rank,
            "definition_score": float(scores[local_pos]),
            "global_idx": global_idx,
            "chunk_id": retrieval_records_dense[global_idx]["chunk_id"]
        })

    return results


def reciprocal_rank_fusion(rank_lists, k=60):
    rrf_scores = defaultdict(float)

    for rank_list in rank_lists:
        for item in rank_list:
            rrf_scores[item["global_idx"]] += 1.0 / (k + item["rank"])

    return dict(rrf_scores)


# =========================
# SOURCE SCORING
# =========================

GEN_STOPWORDS = {
    "what", "does", "do", "say", "says", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "can", "could", "should", "would",
    "i", "we", "you", "the", "a", "an", "of", "for", "to", "under", "about",
    "and", "or", "with", "on", "in", "by", "from", "as", "this", "that"
}


def _infer_doc_type(item: dict) -> str:
    doc_type = item.get("doc_type")

    if doc_type in {"Act", "Rules"}:
        return doc_type

    doc_family = (item.get("doc_family", "") or "").lower()
    citation = (item.get("citation_label", "") or "").lower()

    if "rules" in doc_family or "rule" in citation:
        return "Rules"

    if "act" in doc_family or "section" in citation:
        return "Act"

    return ""


def content_tokens(query: str):
    toks = basic_tokenize(query)

    return [
        t for t in toks
        if t not in GEN_STOPWORDS
        and t not in {"section", "rule", "sec", "s", "r", "act", "acts", "cgst", "igst", "utgst"}
    ]


def heading_overlap_count(query: str, record: dict) -> int:
    q_tokens = set(content_tokens(query))
    h_text = (record.get("full_heading", "") or "") + " " + (record.get("citation_label", "") or "")
    h_tokens = set(basic_tokenize(h_text))
    return len(q_tokens & h_tokens)


def is_opening_provision_chunk(item: dict) -> bool:
    text = (item.get("chunk_text", "") or "").lstrip()
    ref = str(item.get("section_or_rule_number", "")).strip()

    return text.startswith(f"Section {ref}.") or text.startswith(f"Rule {ref}.")


def _title_text(item: dict) -> str:
    heading = item.get("full_heading", "") or ""
    parts = [p.strip() for p in heading.split(">") if p.strip()]
    return parts[-1].lower() if parts else heading.lower()


def _source_text_for_scoring(item: dict) -> str:
    heading = item.get("full_heading", "") or ""
    citation = item.get("citation_label", "") or ""
    text = item.get("chunk_text", "") or ""
    return f"{heading} {citation} {text[:1200]}".lower()


def _phrase_tokens(text: str):
    return [
        t for t in basic_tokenize(text)
        if t not in GEN_STOPWORDS
        and t not in {"section", "rule", "sec", "s", "r", "act", "acts", "cgst", "igst", "utgst"}
    ]


def _looks_like_definition_section_item(item: dict) -> bool:
    heading_lower = (item.get("full_heading", "") or "").lower()

    return (
        _infer_doc_type(item) == "Act"
        and str(item.get("section_or_rule_number", "")).lower() == "2"
        and "definitions" in heading_lower
    )


def _legal_unit_key(item: dict):
    normalized = item.get("normalized_section_or_rule_number")

    if normalized:
        ref = str(normalized).lower()
    else:
        ref = str(item.get("section_or_rule_number", "")).lower()

    return (
        item.get("doc_family", ""),
        ref
    )


def _phrase_match_score(item: dict, phrase: str) -> float:
    if not phrase:
        return 0.0

    phrase = phrase.lower().strip()

    if not phrase:
        return 0.0

    title = _title_text(item)
    full_text = _source_text_for_scoring(item)

    score = 0.0

    if title.startswith(phrase):
        score += 4.0
    elif phrase in title:
        score += 2.5

    if phrase in full_text[:350]:
        score += 1.75
    elif phrase in full_text:
        score += 1.0

    phrase_tok = set(_phrase_tokens(phrase))

    if phrase_tok:
        text_tok = set(basic_tokenize(full_text))
        score += 0.40 * len(phrase_tok & text_tok)

    return score


def _source_preference_score(item: dict, query: str, profile: dict) -> float:
    score = 0.0
    q = query.lower()
    title_lower = _title_text(item)
    doc_type = _infer_doc_type(item)
    ref_num = str(item.get("section_or_rule_number", "")).lower()

    if profile.get("prefer_opening_provision", False) and is_opening_provision_chunk(item):
        score += 1.5
    elif is_opening_provision_chunk(item):
        score += 0.75

    preferred_doc_type = profile.get("preferred_doc_type")

    if preferred_doc_type and doc_type == preferred_doc_type:
        score += 1.5

    score += 0.30 * heading_overlap_count(query, item)

    for phrase in profile.get("focus_phrases", [])[:5]:
        score += _phrase_match_score(item, phrase)

    if profile.get("prefer_definition_section", False):
        if _looks_like_definition_section_item(item):
            score += 5.0
        elif doc_type == "Act" and ref_num != "2":
            score -= 0.5

    if profile.get("definition_comparison", False):
        if _looks_like_definition_section_item(item):
            score += 3.0

    if profile.get("intent") == "definition_lookup":
        if "tax liability on" in title_lower:
            score -= 2.0
        if "procedure" in title_lower or "particulars" in title_lower:
            score -= 1.0

    if profile.get("intent") == "procedural_rule" and doc_type == "Rules":
        score += 1.0

    if profile.get("intent") == "conditions_or_eligibility" and doc_type == "Act":
        score += 0.75

    reg_mode = profile.get("registration_mode")

    if reg_mode == "liability":
        if "not liable for registration" in title_lower:
            score -= 5.0
        if "liable for registration" in title_lower or "persons liable for registration" in title_lower:
            score += 4.5

    elif reg_mode == "exception":
        if "not liable for registration" in title_lower:
            score += 4.5
        if "liable for registration" in title_lower or "persons liable for registration" in title_lower:
            score -= 2.0

    if profile.get("intent") == "act_vs_rule_linking":
        if "special cases" in title_lower and "special" not in q:
            score -= 2.5

    return score


def metadata_bonus(record, query_info):
    bonus = 0.0

    targets = set(query_info.get("normalized_targets", []))
    hinted_families = set(query_info.get("doc_family_hints", []))

    if targets and record.get("normalized_section_or_rule_number") in targets:
        bonus += 2.0

        if record.get("is_subchunk"):
            sub_idx = record.get("subchunk_index")
            if sub_idx is not None and float(sub_idx) == 1.0:
                bonus += 0.65
        else:
            bonus += 0.80

    if hinted_families and record.get("doc_family") in hinted_families:
        bonus += 0.20

    if query_info.get("prefer_cgst_default", False):
        if record.get("doc_family") == "CGST Act":
            bonus += 0.30

    q_tokens = expanded_query_tokens(query_info.get("raw_query", ""))

    if q_tokens:
        title_overlap = token_overlap_count(q_tokens, record.get("section_or_rule_title", ""))
        heading_overlap = token_overlap_count(q_tokens, record.get("full_heading", ""))

        overlap_bonus = (0.12 * title_overlap) + (0.05 * heading_overlap)
        bonus += min(0.60, overlap_bonus)

    if looks_like_definition_query(query_info.get("raw_query", "")):
        if (
            record.get("doc_family") == "CGST Act"
            and str(record.get("section_or_rule_number", "")).strip().lower() == "2"
        ):
            local_chunk_text = record.get("chunk_text", "") or ""
            local_definition_text = " ".join([
                record.get("section_or_rule_title", ""),
                local_chunk_text[:600]
            ])

            local_overlap = token_overlap_count(q_tokens, local_definition_text)

            if local_overlap >= 2:
                bonus += min(1.20, 0.60 + (0.20 * local_overlap))

            query_phrase = query_info["raw_query"].lower().strip()

            for prefix in ["what is ", "define ", "meaning of "]:
                if query_phrase.startswith(prefix):
                    query_phrase = query_phrase[len(prefix):].strip()

            local_lower = local_chunk_text.lower()[:500]

            if f'"{query_phrase}"' in local_lower:
                bonus += 0.60
            elif f"{query_phrase} means" in local_lower:
                bonus += 0.60
            elif query_phrase in local_lower:
                bonus += 0.30

    return bonus


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_search_on_candidates(query, candidate_indices, query_info, candidate_top_k=12, rrf_k=60):
    if not candidate_indices:
        return []

    heading_results = heading_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)
    sparse_results = sparse_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)
    dense_results = dense_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)

    rrf_scores = reciprocal_rank_fusion(
        [heading_results, sparse_results, dense_results],
        k=rrf_k
    )

    heading_rank_map = {item["global_idx"]: item["rank"] for item in heading_results}
    sparse_rank_map = {item["global_idx"]: item["rank"] for item in sparse_results}
    dense_rank_map = {item["global_idx"]: item["rank"] for item in dense_results}

    sparse_score_map = {item["global_idx"]: item["sparse_score"] for item in sparse_results}
    dense_score_map = {item["global_idx"]: item["dense_score"] for item in dense_results}

    combined = []

    for global_idx, rrf_score in rrf_scores.items():
        rec = retrieval_records_dense[global_idx]
        bonus = metadata_bonus(rec, query_info)
        final_score = rrf_score + bonus

        combined.append({
            "global_idx": global_idx,
            "chunk_id": rec["chunk_id"],
            "doc_family": rec.get("doc_family"),
            "doc_type": _infer_doc_type(rec),
            "section_or_rule_number": rec.get("section_or_rule_number"),
            "normalized_section_or_rule_number": rec.get("normalized_section_or_rule_number"),
            "citation_label": rec.get("citation_label"),
            "full_heading": rec.get("full_heading"),
            "chunk_text": rec.get("chunk_text"),
            "rrf_score": float(rrf_score),
            "metadata_bonus": float(bonus),
            "final_score": float(final_score),
            "heading_rank": heading_rank_map.get(global_idx),
            "sparse_rank": sparse_rank_map.get(global_idx),
            "dense_rank": dense_rank_map.get(global_idx),
            "definition_rank": None,
            "sparse_score": sparse_score_map.get(global_idx),
            "dense_score": dense_score_map.get(global_idx)
        })

    return sorted(combined, key=lambda x: x["final_score"], reverse=True)


def retrieve_hybrid(query, top_k=DEFAULT_RETRIEVAL_TOP_K, candidate_top_k=25, rrf_k=60):
    query_info = parse_query_structure(query)
    profile = infer_query_profile(query)

    candidate_indices = get_candidate_indices(query_info)

    if len(candidate_indices) == 0:
        candidate_indices = list(range(len(retrieval_records_dense)))

    if profile["intent"] == "definition_lookup":
        act_only = [
            i for i in candidate_indices
            if _infer_doc_type(retrieval_records_dense[i]) == "Act"
        ]

        if act_only:
            candidate_indices = act_only

    elif profile["intent"] == "procedural_rule":
        rule_only = [
            i for i in candidate_indices
            if _infer_doc_type(retrieval_records_dense[i]) == "Rules"
        ]

        if rule_only:
            candidate_indices = rule_only

    elif profile.get("definition_comparison", False):
        act_only = [
            i for i in candidate_indices
            if _infer_doc_type(retrieval_records_dense[i]) == "Act"
        ]

        if act_only:
            candidate_indices = act_only

    if looks_like_act_rule_link_query(query):
        branch_query = build_branch_topic_query(query)

        act_candidate_indices = [
            i for i in candidate_indices
            if _infer_doc_type(retrieval_records_dense[i]) == "Act"
        ]

        rule_candidate_indices = [
            i for i in candidate_indices
            if _infer_doc_type(retrieval_records_dense[i]) == "Rules"
        ]

        if not query_info.get("explicit_family_mentioned", False):
            cgst_act_candidates = [
                i for i in act_candidate_indices
                if retrieval_records_dense[i].get("doc_family") == "CGST Act"
            ]

            cgst_rule_candidates = [
                i for i in rule_candidate_indices
                if retrieval_records_dense[i].get("doc_family") == "CGST Rules"
            ]

            if cgst_act_candidates:
                act_candidate_indices = cgst_act_candidates

            if cgst_rule_candidates:
                rule_candidate_indices = cgst_rule_candidates

        act_results = hybrid_search_on_candidates(
            query=branch_query,
            candidate_indices=act_candidate_indices,
            query_info=query_info,
            candidate_top_k=candidate_top_k,
            rrf_k=rrf_k
        )

        rule_results = hybrid_search_on_candidates(
            query=branch_query,
            candidate_indices=rule_candidate_indices,
            query_info=query_info,
            candidate_top_k=candidate_top_k,
            rrf_k=rrf_k
        )

        def _branch_title_focus_bonus(item, local_profile):
            title = _title_text(item)
            title_tokens = set(_phrase_tokens(title))
            score = 0.0

            for phrase in local_profile.get("focus_phrases", [])[:5]:
                phrase_tokens = set(_phrase_tokens(phrase))

                if not phrase_tokens:
                    continue

                overlap = len(phrase_tokens & title_tokens)
                coverage = overlap / len(phrase_tokens)

                score += 1.25 * overlap

                if coverage == 1.0:
                    score += 2.0
                elif coverage >= 0.5:
                    score += 0.75

            if is_opening_provision_chunk(item):
                score += 0.5

            return score

        act_results = sorted(
            act_results,
            key=lambda x: (
                _branch_title_focus_bonus(x, {**profile, "preferred_doc_type": "Act"}),
                _source_preference_score(x, query, {**profile, "preferred_doc_type": "Act"}),
                x.get("final_score", 0.0)
            ),
            reverse=True
        )

        rule_results = sorted(
            rule_results,
            key=lambda x: (
                _branch_title_focus_bonus(x, {**profile, "preferred_doc_type": "Rules"}),
                _source_preference_score(x, query, {**profile, "preferred_doc_type": "Rules"}),
                x.get("final_score", 0.0)
            ),
            reverse=True
        )

        combined = []
        seen = set()

        if act_results:
            best_act = act_results[0]
            best_act["doc_type"] = _infer_doc_type(best_act)
            combined.append(best_act)
            seen.add(best_act["chunk_id"])

        if rule_results:
            best_rule = rule_results[0]

            if best_rule["chunk_id"] not in seen:
                best_rule["doc_type"] = _infer_doc_type(best_rule)
                combined.append(best_rule)
                seen.add(best_rule["chunk_id"])

        merged_rest = []
        max_len = max(len(act_results), len(rule_results))

        for i in range(1, max_len):
            if i < len(act_results):
                merged_rest.append(act_results[i])

            if i < len(rule_results):
                merged_rest.append(rule_results[i])

        for item in merged_rest:
            if item["chunk_id"] not in seen:
                item["doc_type"] = _infer_doc_type(item)
                combined.append(item)
                seen.add(item["chunk_id"])

            if len(combined) >= top_k:
                break

        return {
            "query_info": {
                **query_info,
                "query_type": "act_vs_rule_linking"
            },
            "candidate_count": len(candidate_indices),
            "results": combined[:top_k],
            "candidates": combined[:top_k],
        }

    heading_results = heading_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)
    sparse_results = sparse_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)
    dense_results = dense_search(query, candidate_indices=candidate_indices, top_k=candidate_top_k)
    definition_results = definition_search(query, top_k=candidate_top_k)

    rank_lists = [heading_results, sparse_results, dense_results]

    if definition_results:
        rank_lists.append(definition_results)

    rrf_scores = reciprocal_rank_fusion(rank_lists, k=rrf_k)

    heading_rank_map = {item["global_idx"]: item["rank"] for item in heading_results}
    sparse_rank_map = {item["global_idx"]: item["rank"] for item in sparse_results}
    dense_rank_map = {item["global_idx"]: item["rank"] for item in dense_results}
    definition_rank_map = {item["global_idx"]: item["rank"] for item in definition_results} if definition_results else {}

    sparse_score_map = {item["global_idx"]: item["sparse_score"] for item in sparse_results}
    dense_score_map = {item["global_idx"]: item["dense_score"] for item in dense_results}

    combined = []

    for global_idx, rrf_score in rrf_scores.items():
        rec = retrieval_records_dense[global_idx]

        item = {
            "global_idx": global_idx,
            "chunk_id": rec["chunk_id"],
            "doc_family": rec.get("doc_family"),
            "doc_type": _infer_doc_type(rec),
            "section_or_rule_number": rec.get("section_or_rule_number"),
            "normalized_section_or_rule_number": rec.get("normalized_section_or_rule_number"),
            "citation_label": rec.get("citation_label"),
            "full_heading": rec.get("full_heading"),
            "chunk_text": rec.get("chunk_text"),
            "rrf_score": float(rrf_score),
            "heading_rank": heading_rank_map.get(global_idx),
            "sparse_rank": sparse_rank_map.get(global_idx),
            "dense_rank": dense_rank_map.get(global_idx),
            "definition_rank": definition_rank_map.get(global_idx),
            "sparse_score": sparse_score_map.get(global_idx),
            "dense_score": dense_score_map.get(global_idx)
        }

        bonus = metadata_bonus(rec, query_info)
        profile_bonus = 0.18 * _source_preference_score(item, query, profile)
        final_score = rrf_score + bonus + profile_bonus

        item["metadata_bonus"] = float(bonus)
        item["profile_bonus"] = float(profile_bonus)
        item["final_score"] = float(final_score)

        combined.append(item)

    combined = sorted(combined, key=lambda x: x["final_score"], reverse=True)[:top_k]

    return {
        "query_info": {
            **query_info,
            "query_type": profile["intent"]
        },
        "candidate_count": len(candidate_indices),
        "results": combined,
        "candidates": combined,
    }


# =========================
# STRUCTURAL CONTEXT HELPERS
# =========================

def extract_subunit_chain(query: str):
    q = _normalize_query(query)

    m = re.search(
        r"\b(?:section|sec|s\.|rule|r\.)\s*\d+[a-z]?\s*((?:\([^)]+\))+)​?",
        q,
        flags=re.IGNORECASE
    )

    if not m:
        m = re.search(
            r"\b\d+[a-z]?\s*((?:\([^)]+\))+)",
            q,
            flags=re.IGNORECASE
        )

    if not m:
        return []

    return re.findall(r"\([^)]+\)", m.group(1))


def is_subunit_reference_query(query: str) -> bool:
    return len(extract_subunit_chain(query)) > 0


def _contains_subunit_chain(text: str, chain) -> bool:
    if not text or not chain:
        return False

    lower_text = text.lower()
    search_from = 0

    for marker in chain:
        idx = lower_text.find(marker.lower(), search_from)

        if idx == -1:
            return False

        search_from = idx + len(marker)

    return True


def _reference_query_terms(query: str):
    drop = QUERY_PROFILE_STOPWORDS | {
        "section", "sec", "s", "rule", "r",
        "act", "acts", "rules",
        "cgst", "igst", "utgst",
        "say", "says", "state", "explain", "show",
        "tell", "listed", "about", "under"
    }

    toks = basic_tokenize(_normalize_query(query))
    return [t for t in toks if t not in drop]


def _query_has_temporal_focus(query: str) -> bool:
    q = _normalize_query(query)

    triggers = [
        "by when",
        "deadline",
        "time limit",
        "cut off",
        "cutoff",
        "last date",
        "before when",
        "within what time"
    ]

    return any(t in q for t in triggers)


def _query_has_consequence_focus(query: str) -> bool:
    q = _normalize_query(query)

    triggers = [
        "what happens if",
        "consequence",
        "effect if",
        "fails to",
        "does not",
        "not pay"
    ]

    return any(t in q for t in triggers)


def _split_legal_blocks(text: str):
    if not text:
        return []

    text = text.strip()

    parts = re.split(
        r"(?=\n(?:\(\d+[a-z]?\)|\([a-z]+\)|Provided\b|Explanation\b|Illustration\b))",
        "\n" + text
    )

    blocks = [p.strip() for p in parts if p.strip()]

    if len(blocks) >= 2:
        return blocks

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    if paras:
        return paras

    return [text]


def _subunit_forward_excerpt(text: str, query: str, max_chars: int):
    if not text:
        return None

    chain = extract_subunit_chain(query)

    if not chain:
        return None

    lower_text = text.lower()
    search_from = 0
    found_positions = []

    for marker in chain:
        idx = lower_text.find(marker.lower(), search_from)

        if idx == -1:
            return None

        found_positions.append(idx)
        search_from = idx + len(marker)

    anchor_idx = found_positions[-1]

    start = max(0, anchor_idx - 250)
    end = min(len(text), start + max_chars)

    excerpt = text[start:end]

    if start > 0:
        excerpt = "... " + excerpt

    if end < len(text):
        excerpt = excerpt + " ..."

    return excerpt


def _legal_block_score(block: str, query: str) -> float:
    lower_block = block.lower()
    q_terms = _reference_query_terms(query)

    if not q_terms:
        return 0.0

    block_tokens = set(basic_tokenize(lower_block))

    overlap = sum(1 for t in q_terms if t in block_tokens)
    freq_bonus = sum(lower_block.count(t) for t in q_terms) * 0.15

    temporal_bonus = 0.0

    if _query_has_temporal_focus(query):
        temporal_markers = [
            "before", "after", "within", "expiry", "period",
            "day", "days", "month", "november", "annual return",
            "furnishing", "due date"
        ]
        temporal_bonus = 0.20 * sum(1 for m in temporal_markers if m in lower_block)

    consequence_bonus = 0.0

    if _query_has_consequence_focus(query):
        consequence_markers = [
            "fails", "failure", "not pay", "shall be added",
            "output tax liability", "interest", "re-avail",
            "payment made", "along with"
        ]
        consequence_bonus = 0.20 * sum(1 for m in consequence_markers if m in lower_block)

    return overlap + freq_bonus + temporal_bonus + consequence_bonus


def _best_query_focused_excerpt(text: str, query: str, max_chars: int):
    blocks = _split_legal_blocks(text)

    if not blocks:
        return None

    scored = [
        (_legal_block_score(block, query), idx, block)
        for idx, block in enumerate(blocks)
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_idx, best_block = scored[0]

    if best_score <= 0:
        return None

    selected = [best_block]
    total_len = len(best_block)

    j = best_idx + 1

    while j < len(blocks) and total_len + len(blocks[j]) + 2 <= max_chars:
        selected.append(blocks[j])
        total_len += len(blocks[j]) + 2
        j += 1

    k = best_idx - 1

    while k >= 0 and total_len + len(blocks[k]) + 2 <= max_chars:
        selected.insert(0, blocks[k])
        total_len += len(blocks[k]) + 2
        k -= 1

    excerpt = "\n\n".join(selected).strip()

    if len(excerpt) < min(600, max_chars * 0.25) and len(text) > len(excerpt):
        return None

    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]

    return excerpt


def _head_tail_excerpt(text: str, max_chars: int, head_ratio: float = 0.55):
    if len(text) <= max_chars:
        return text

    separator = "\n\n... [middle omitted for brevity] ...\n\n"
    usable = max_chars - len(separator)

    if usable <= 200:
        return text[:max_chars]

    head_chars = int(usable * head_ratio)
    tail_chars = usable - head_chars

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()

    return head + separator + tail


def _context_excerpt_for_query(text: str, query: str, max_chars: int):
    if not text:
        return ""

    text = text.strip()

    subunit_excerpt = _subunit_forward_excerpt(text, query, max_chars)

    if subunit_excerpt is not None:
        return subunit_excerpt

    focused_excerpt = _best_query_focused_excerpt(text, query, max_chars)

    if focused_excerpt is not None:
        return focused_excerpt

    if len(text) <= max_chars:
        return text

    return _head_tail_excerpt(text, max_chars)


# =========================
# SOURCE PRUNING
# =========================

def _merge_same_unit_chunks(primary_item: dict, continuation_item: dict) -> dict:
    merged = dict(primary_item)

    primary_text = (primary_item.get("chunk_text", "") or "").strip()
    continuation_text = (continuation_item.get("chunk_text", "") or "").strip()

    if continuation_text and continuation_text not in primary_text:
        merged["chunk_text"] = f"{primary_text}\n\n{continuation_text}"
    else:
        merged["chunk_text"] = primary_text

    merged["merged_chunk_ids"] = [
        primary_item.get("chunk_id"),
        continuation_item.get("chunk_id")
    ]

    return merged


def _best_same_unit_continuation(candidates, base_item, query: str, profile: dict):
    base_unit = _legal_unit_key(base_item)
    base_chunk_id = base_item.get("chunk_id")

    same_unit_candidates = []

    for item in candidates:
        if item.get("chunk_id") == base_chunk_id:
            continue

        if _legal_unit_key(item) != base_unit:
            continue

        score = _source_preference_score(item, query, profile)

        if not is_opening_provision_chunk(item):
            score += 0.75

        same_unit_candidates.append((score, item))

    if not same_unit_candidates:
        return None

    same_unit_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = same_unit_candidates[0]

    if best_score >= 1.25:
        return best_item

    return None


def _merge_same_unit_chunks_for_reference(
    matched_items,
    base_item,
    query: str,
    profile: dict,
    max_extra_chunks=8
):
    merged = dict(base_item)

    merged_text = (base_item.get("chunk_text", "") or "").strip()
    merged_ids = [base_item.get("chunk_id")]
    base_unit = _legal_unit_key(base_item)
    target_chain = extract_subunit_chain(query)

    same_unit_ranked = []

    for item in matched_items:
        if item.get("chunk_id") == base_item.get("chunk_id"):
            continue

        if _legal_unit_key(item) != base_unit:
            continue

        score = _source_preference_score(item, query, profile)

        if not is_opening_provision_chunk(item):
            score += 0.75

        same_unit_ranked.append((score, item))

    same_unit_ranked.sort(key=lambda x: x[0], reverse=True)

    for _, item in same_unit_ranked[:max_extra_chunks]:
        part = (item.get("chunk_text", "") or "").strip()

        if part and part not in merged_text:
            merged_text += "\n\n" + part
            merged_ids.append(item.get("chunk_id"))

        if target_chain and _contains_subunit_chain(merged_text, target_chain):
            break

    merged["chunk_text"] = merged_text
    merged["merged_chunk_ids"] = merged_ids

    return merged


def prune_generation_sources(query: str, retrieval_output: dict, max_chunks=DEFAULT_CONTEXT_TOP_K):
    results = retrieval_output.get("results", [])

    if not results:
        return []

    query_info = retrieval_output.get("query_info", {})
    explicit_family_mentioned = query_info.get("explicit_family_mentioned", False)
    profile = infer_query_profile(query)

    section_refs, rule_refs = extract_query_refs(query)
    comparison = looks_like_comparison_query(query) or (len(section_refs) + len(rule_refs) >= 2)

    pruned = []
    seen_chunk_ids = set()
    seen_unit_keys = set()

    def add_item(item, allow_same_unit=False):
        chunk_id = item.get("chunk_id")
        unit_key = _legal_unit_key(item)

        if chunk_id in seen_chunk_ids:
            return False

        if unit_key in seen_unit_keys and not allow_same_unit:
            return False

        pruned.append(item)
        seen_chunk_ids.add(chunk_id)
        seen_unit_keys.add(unit_key)

        return True

    if profile["intent"] == "definition_lookup":
        definition_candidates = [item for item in results if _looks_like_definition_section_item(item)]

        if definition_candidates:
            best_def = max(
                definition_candidates,
                key=lambda x: _source_preference_score(x, query, profile)
            )
            add_item(best_def)
            return pruned[:1]

    if profile.get("registration_mode") == "liability":
        liability_items = [
            item for item in results
            if "liable for registration" in _title_text(item)
            or "persons liable for registration" in _title_text(item)
        ]

        if liability_items:
            best_item = max(
                liability_items,
                key=lambda x: _source_preference_score(x, query, profile)
            )
            add_item(best_item)
            return pruned[:1]

    if profile.get("registration_mode") == "exception":
        exception_items = [
            item for item in results
            if "not liable for registration" in _title_text(item)
        ]

        if exception_items:
            best_item = max(
                exception_items,
                key=lambda x: _source_preference_score(x, query, profile)
            )
            add_item(best_item)
            return pruned[:1]

    if profile["intent"] == "act_vs_rule_linking":
        act_items = [item for item in results if _infer_doc_type(item) == "Act"]
        rule_items = [item for item in results if _infer_doc_type(item) == "Rules"]

        if not explicit_family_mentioned:
            cgst_act_items = [item for item in act_items if item.get("doc_family") == "CGST Act"]
            cgst_rule_items = [item for item in rule_items if item.get("doc_family") == "CGST Rules"]

            if cgst_act_items:
                act_items = cgst_act_items

            if cgst_rule_items:
                rule_items = cgst_rule_items

        if act_items:
            best_act = max(act_items, key=lambda x: _source_preference_score(x, query, profile))
            add_item(best_act)

        if rule_items:
            best_rule = max(rule_items, key=lambda x: _source_preference_score(x, query, profile))
            add_item(best_rule)

        return pruned[:max_chunks]

    if comparison:
        if section_refs or rule_refs:
            target_refs = {str(x).lower() for x in section_refs + rule_refs}
            dominant_family = results[0].get("doc_family")

            matched = [
                item for item in results
                if item.get("doc_family") == dominant_family
                and str(item.get("section_or_rule_number", "")).lower() in target_refs
            ]

            matched = sorted(
                matched,
                key=lambda x: _source_preference_score(x, query, profile),
                reverse=True
            )

            for item in matched:
                add_item(item)

                if len(pruned) >= max_chunks:
                    break

            return pruned[:max_chunks]

        left_phrase = profile.get("comparison_left")
        right_phrase = profile.get("comparison_right")

        definition_candidates = [item for item in results if _looks_like_definition_section_item(item)]

        if left_phrase and right_phrase and definition_candidates:
            left_profile = {**profile, "focus_phrases": [left_phrase]}
            right_profile = {**profile, "focus_phrases": [right_phrase]}

            left_ranked = sorted(
                definition_candidates,
                key=lambda x: _source_preference_score(x, query, left_profile),
                reverse=True
            )

            right_ranked = sorted(
                definition_candidates,
                key=lambda x: _source_preference_score(x, query, right_profile),
                reverse=True
            )

            if left_ranked:
                add_item(left_ranked[0])

            for item in right_ranked:
                if item.get("chunk_id") not in seen_chunk_ids:
                    add_item(item, allow_same_unit=True)
                    break

            if pruned:
                return pruned[:max_chunks]

        ranked = sorted(
            results,
            key=lambda x: _source_preference_score(x, query, profile),
            reverse=True
        )

        for item in ranked:
            add_item(item)

            if len(pruned) >= min(max_chunks, 2):
                break

        return pruned[:max_chunks]

    if (section_refs or rule_refs) and not comparison:
        dominant_family = results[0].get("doc_family")
        dominant_ref = str(results[0].get("section_or_rule_number", "")).lower()

        matched = [
            item for item in results
            if item.get("doc_family") == dominant_family
            and str(item.get("section_or_rule_number", "")).lower() == dominant_ref
        ]

        if matched:
            best_item = max(
                matched,
                key=lambda x: _source_preference_score(x, query, profile)
            )
            add_item(best_item)

            should_merge_same_unit = (
                profile["intent"] in {"reference", "conditions_or_eligibility", "procedural_rule"}
                or is_subunit_reference_query(query)
            )

            if should_merge_same_unit:
                max_extra_chunks = 8 if is_subunit_reference_query(query) else 5
                pruned[-1] = _merge_same_unit_chunks_for_reference(
                    matched_items=matched,
                    base_item=pruned[-1],
                    query=query,
                    profile=profile,
                    max_extra_chunks=max_extra_chunks
                )

            return pruned[:1]

    ranked = sorted(
        results,
        key=lambda x: _source_preference_score(x, query, profile),
        reverse=True
    )

    add_item(ranked[0])

    if (
        max_chunks > 1
        and profile["intent"] in {"conditions_or_eligibility", "procedural_rule"}
    ):
        continuation = _best_same_unit_continuation(ranked, ranked[0], query, profile)

        if continuation is not None:
            pruned[-1] = _merge_same_unit_chunks(pruned[-1], continuation)
            seen_chunk_ids.add(continuation.get("chunk_id"))

    return pruned[:max_chunks]


# =========================
# PRODUCTION SOURCE REPAIR
# =========================

_PROD_SOURCE_STOPWORDS = {
    "what", "does", "do", "say", "says", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "can", "could", "should", "would",
    "i", "we", "you", "the", "a", "an", "of", "for", "to", "under", "about",
    "and", "or", "with", "on", "in", "by", "from", "as", "this", "that",
    "gst", "law", "provision", "provisions", "act", "acts", "rule", "rules",
    "section", "sections", "sec", "s", "r", "read", "add", "adds", "added",
    "difference", "between", "compare", "versus", "vs"
}

_PROD_TOPIC_MISMATCH_TERMS = {
    "cancellation", "cancel", "revocation", "revoke", "amendment", "amend",
    "deemed", "refund", "detention", "seizure", "appeal", "assessment",
    "return", "payment", "penalty", "audit", "inspection"
}


def _prod_norm_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _prod_singularize(token):
    token = str(token or "").lower().strip()

    if len(token) > 4 and token.endswith("s"):
        return token[:-1]

    return token


def _prod_tokens(text):
    raw = re.findall(r"[a-z0-9]+", _prod_norm_text(text))
    return [_prod_singularize(t) for t in raw if t and t not in _PROD_SOURCE_STOPWORDS]


def _prod_query_focus_tokens(query):
    tokens = set(_prod_tokens(query))

    expansions = {
        "application": {"apply", "applicant", "procedure"},
        "apply": {"application", "applicant", "procedure"},
        "applicant": {"application", "apply"},
        "registration": {"register", "registered"},
        "register": {"registration", "registered"},
        "registered": {"registration", "register"},
        "cancellation": {"cancel", "cancelled"},
        "cancel": {"cancellation", "cancelled"},
        "refund": {"refunded", "refunds"},
        "detention": {"detain", "detained"},
        "seizure": {"seize", "seized"},
        "supply": {"supplies"},
    }

    expanded = set(tokens)

    for tok in list(tokens):
        for key, vals in expansions.items():
            if tok == key:
                expanded.update(vals)

    return expanded


def _prod_field(obj, *names, default=""):
    if not isinstance(obj, dict):
        return default

    for name in names:
        val = obj.get(name)

        if val not in [None, ""]:
            return val

    metadata = obj.get("metadata")

    if isinstance(metadata, dict):
        for name in names:
            val = metadata.get(name)

            if val not in [None, ""]:
                return val

    return default


def _prod_candidate_list(retrieval_output):
    if not isinstance(retrieval_output, dict):
        return []

    for key in ["candidates", "results", "retrieved_results", "ranked_results"]:
        val = retrieval_output.get(key)

        if isinstance(val, list):
            return val

    return []


def _prod_doc_family(candidate):
    return str(_prod_field(candidate, "doc_family", "document_name", default="")).strip()


def _prod_doc_type(candidate):
    raw = str(_prod_field(candidate, "doc_type", default="")).strip()
    heading = _prod_norm_text(_prod_field(candidate, "full_heading", "heading", default=""))
    family = _prod_norm_text(_prod_doc_family(candidate))

    if raw.lower() in {"act", "acts"}:
        return "Act"

    if raw.lower() in {"rule", "rules"}:
        return "Rules"

    if "rules" in family or " rule " in f" {heading} ":
        return "Rules"

    if "act" in family or " section " in f" {heading} ":
        return "Act"

    return raw


def _prod_number(candidate):
    return str(_prod_field(
        candidate,
        "section_or_rule_number",
        "section_number",
        "rule_number",
        default=""
    )).strip().lower()


def _prod_heading(candidate):
    return str(_prod_field(candidate, "full_heading", "heading", "title", default=""))


def _prod_citation(candidate):
    return str(_prod_field(candidate, "citation_label", "citation", default=""))


def _prod_text(candidate):
    return str(_prod_field(candidate, "chunk_text", "text", "chunk_preview", "preview", default=""))


def _prod_source_key(src):
    chunk_id = str(_prod_field(src, "chunk_id", default="")).strip()

    if chunk_id:
        return ("chunk_id", chunk_id)

    return (
        _prod_doc_family(src),
        _prod_doc_type(src),
        _prod_number(src),
        _prod_citation(src)
    )


def _prod_dedupe_sources(sources):
    seen = set()
    out = []

    for src in sources or []:
        key = _prod_source_key(src)

        if key not in seen:
            out.append(src)
            seen.add(key)

    return out


def _prod_family_matches(candidate, family_hints):
    if not family_hints:
        return True

    cand_family = _prod_norm_text(_prod_doc_family(candidate))
    cand_heading = _prod_norm_text(_prod_heading(candidate))
    cand_text = cand_family + " " + cand_heading

    for hint in family_hints:
        h = _prod_norm_text(hint)

        if h and h in cand_text:
            return True

        if "cgst" in h and "cgst" in cand_text:
            return True
        if "igst" in h and "igst" in cand_text:
            return True
        if "utgst" in h and "utgst" in cand_text:
            return True
        if "compensation" in h and "compensation" in cand_text:
            return True

    return False


def _prod_phrase_bonus(query, heading_text, body_text):
    q_tokens = _prod_tokens(query)
    heading = _prod_norm_text(heading_text)
    body = _prod_norm_text(body_text)

    bonus = 0.0
    phrases = []

    for n in [2, 3]:
        for i in range(0, max(0, len(q_tokens) - n + 1)):
            phrase = " ".join(q_tokens[i:i+n])

            if phrase:
                phrases.append(phrase)

    for phrase in phrases:
        if phrase in heading:
            bonus += 4.0
        elif phrase in body:
            bonus += 1.0

    return bonus


def _prod_topic_mismatch_penalty(query, heading_text):
    q_tokens = _prod_query_focus_tokens(query)
    h_tokens = set(_prod_tokens(heading_text))
    penalty = 0.0

    for term in _PROD_TOPIC_MISMATCH_TERMS:
        t = _prod_singularize(term)

        if t in h_tokens and t not in q_tokens:
            penalty += 1.2

    h = _prod_norm_text(heading_text)

    if "power to make rules" in h and "power" not in q_tokens:
        penalty += 3.0

    if "laying of rules" in h and "laying" not in q_tokens:
        penalty += 3.0

    return penalty


def _prod_candidate_score(
    query,
    candidate,
    desired_doc_type=None,
    target_number=None,
    family_hints=None
):
    heading = _prod_heading(candidate)
    citation = _prod_citation(candidate)
    body = _prod_text(candidate)

    q_tokens = _prod_query_focus_tokens(query)
    heading_tokens = set(_prod_tokens(heading))
    citation_tokens = set(_prod_tokens(citation))
    body_tokens = set(_prod_tokens(body[:3500]))

    score = 0.0

    if desired_doc_type:
        if _prod_doc_type(candidate).lower() == desired_doc_type.lower():
            score += 4.0
        else:
            score -= 4.0

    if target_number:
        if _prod_number(candidate) == str(target_number).lower():
            score += 8.0
        else:
            score -= 3.0

    if family_hints:
        if _prod_family_matches(candidate, family_hints):
            score += 3.0
        else:
            score -= 2.0

    heading_overlap = len(q_tokens & heading_tokens)
    citation_overlap = len(q_tokens & citation_tokens)
    body_overlap = len(q_tokens & body_tokens)

    score += heading_overlap * 5.0
    score += citation_overlap * 2.0
    score += min(body_overlap * 0.7, 4.0)

    score += _prod_phrase_bonus(query, heading, body)
    score -= _prod_topic_mismatch_penalty(query, heading)

    raw_retrieval_score = _prod_field(candidate, "final_score", "score", default=0)

    try:
        score += min(float(raw_retrieval_score), 5.0) * 0.15
    except Exception:
        pass

    return score


def _prod_select_best_candidate(
    query,
    candidates,
    desired_doc_type=None,
    target_number=None,
    family_hints=None,
    min_score=None
):
    scored = []

    for cand in candidates:
        score = _prod_candidate_score(
            query=query,
            candidate=cand,
            desired_doc_type=desired_doc_type,
            target_number=target_number,
            family_hints=family_hints
        )
        scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best_score, best_cand = scored[0]

    if min_score is not None and best_score < min_score:
        return None

    return best_cand


def infer_force_targets_for_query(query, profile=None):
    profile = profile or infer_query_profile(query)
    targets = []

    for sec in profile.get("section_refs", []) or []:
        targets.append({
            "target_type": "section",
            "number": str(sec),
            "desired_doc_type": "Act",
            "family_hints": profile.get("doc_family_hints", []) or []
        })

    for rule in profile.get("rule_refs", []) or []:
        targets.append({
            "target_type": "rule",
            "number": str(rule),
            "desired_doc_type": "Rules",
            "family_hints": profile.get("doc_family_hints", []) or []
        })

    return targets


def is_multi_target_query(query):
    profile = infer_query_profile(query)

    explicit_count = len(profile.get("section_refs", []) or []) + len(profile.get("rule_refs", []) or [])

    if explicit_count >= 2:
        return True

    qtype = profile.get("query_type") or profile.get("intent")

    return qtype in {"comparison", "act_vs_rule_linking"}


def force_include_target_sources(
    query,
    retrieval_output,
    used_sources,
    context_top_k=DEFAULT_CONTEXT_TOP_K
):
    candidates = _prod_candidate_list(retrieval_output)
    used_sources = list(used_sources or [])

    if not candidates:
        return _prod_dedupe_sources(used_sources)[:context_top_k]

    profile = infer_query_profile(query)
    query_type = profile.get("query_type") or profile.get("intent")

    forced = []
    explicit_targets = infer_force_targets_for_query(query, profile)

    for target in explicit_targets:
        best = _prod_select_best_candidate(
            query=query,
            candidates=candidates,
            desired_doc_type=target["desired_doc_type"],
            target_number=target["number"],
            family_hints=target.get("family_hints", []),
            min_score=None
        )

        if best is not None:
            forced.append(best)

    if not explicit_targets and query_type == "act_vs_rule_linking":
        best_act = _prod_select_best_candidate(
            query=query,
            candidates=candidates,
            desired_doc_type="Act",
            target_number=None,
            family_hints=profile.get("doc_family_hints", []) or [],
            min_score=3.0
        )

        best_rule = _prod_select_best_candidate(
            query=query,
            candidates=candidates,
            desired_doc_type="Rules",
            target_number=None,
            family_hints=profile.get("doc_family_hints", []) or [],
            min_score=3.0
        )

        if best_act is not None:
            forced.append(best_act)

        if best_rule is not None:
            forced.append(best_rule)

    if not explicit_targets and query_type == "comparison":
        ranked = []

        for cand in candidates:
            score = _prod_candidate_score(query, cand)
            ranked.append((score, cand))

        ranked.sort(key=lambda x: x[0], reverse=True)

        distinct_keys = set()

        for score, cand in ranked:
            if score < 3.0:
                continue

            key = (
                _prod_doc_family(cand),
                _prod_doc_type(cand),
                _prod_number(cand),
                _prod_heading(cand)
            )

            if key in distinct_keys:
                continue

            forced.append(cand)
            distinct_keys.add(key)

            if len(forced) >= max(2, context_top_k):
                break

    final_sources = _prod_dedupe_sources(forced + used_sources)

    return final_sources[:context_top_k]


def remove_irrelevant_comparison_contexts(query, profile, used_sources):
    q = query.lower()

    intent = (
        profile.get("intent")
        or profile.get("query_type")
        or ""
    )

    comparison_like = (
        intent == "comparison"
        or " vs " in q
        or " versus " in q
        or "difference between" in q
        or "compare" in q
    )

    if not comparison_like:
        return used_sources

    user_asked_wrong_tax_topic = any(term in q for term in [
        "wrong", "wrongly", "mistake", "incorrect", "refund", "correction", "wrongfully"
    ])

    if user_asked_wrong_tax_topic:
        return used_sources

    filtered_sources = []

    for source in used_sources:
        citation = source.get("citation_label", "").lower()
        heading = source.get("full_heading", "").lower()
        text = source.get("chunk_text", "").lower()

        is_wrong_tax_context = (
            "cgst act, section 77" in citation
            or "igst act, section 19" in citation
            or "wrongfully collected" in heading
            or "tax wrongfully collected" in heading
            or "wrongfully collected" in text[:1200]
            or "tax wrongfully collected" in text[:1200]
        )

        if not is_wrong_tax_context:
            filtered_sources.append(source)

    return filtered_sources if filtered_sources else used_sources


# =========================
# OUT OF SCOPE / ABSTAIN
# =========================

def looks_like_exact_bare_law_lookup(query: str) -> bool:
    q = _qnorm_for_generation(query)

    if re.search(r"\b(section|sec|s\.)\s*\d+[a-z]?\b", q):
        return True

    if re.search(r"\b(rule|r\.)\s*\d+[a-z]?\b", q):
        return True

    if "what does" in q and ("section" in q or "rule" in q):
        return True

    return False


def should_abstain_for_practical_advisory(query: str) -> bool:
    q = _qnorm_for_generation(query)

    if looks_like_exact_bare_law_lookup(query):
        return False

    notification_terms = [
        "latest notification",
        "recent notification",
        "current notification",
        "new notification",
        "latest circular",
        "recent circular",
    ]

    if any(term in q for term in notification_terms):
        return True

    practical_terms = [
        "how should",
        "practically",
        "best way",
        "strategy",
        "structure",
        "plan",
        "advise",
        "advice",
        "what should a business do",
    ]

    business_terms = [
        "business",
        "company",
        "firm",
        "registration across states",
        "across states",
        "structure registration",
    ]

    if any(term in q for term in practical_terms) and any(term in q for term in business_terms):
        return True

    return False


# =========================
# DETERMINISTIC LIST HELPERS
# =========================

def looks_like_listing_generation_query(query: str) -> bool:
    q = _qnorm_for_generation(query)

    if re.search(r"\b(?:section|sec|s\.)?\s*\d+[a-z]?\s*\([^)]*\)\s*\([^)]*\)", q):
        return False

    listing_terms = [
        "list",
        "listed",
        "lists",
        "categories",
        "category",
        "who are required",
        "who is required",
        "required to be registered",
        "compulsory registration",
        "blocked credits",
        "what blocked credits",
    ]

    return any(term in q for term in listing_terms)


def looks_like_consequence_generation_query(query: str) -> bool:
    q = _qnorm_for_generation(query)

    consequence_terms = [
        "what happens if",
        "fails to",
        "does not pay",
        "not pay",
        "consequence",
        "effect if",
        "reclaim",
        "re-avail",
        "reavail",
        "restore",
    ]

    return any(term in q for term in consequence_terms)


def _clean_hint_text(text: str) -> str:
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def _get_source_text_for_list_answer(source):
    return (
        source.get("chunk_text")
        or source.get("text")
        or source.get("content")
        or source.get("chunk_preview")
        or ""
    )


def _remove_amendment_noise_but_keep_later_items(text):
    text = str(text)

    text = re.sub(
        r"(?m)^\s*\d+\[\s*(\((?:[ivxlcdm]+[a-z]?|[a-z])\))",
        r"\1",
        text
    )

    noisy_line_pattern = re.compile(
        r"^\s*\d+\s*(Inserted|Substituted|Omitted|Renumbered|Corrigendum|Inserted vide|Substituted vide|Omitted vide)\b.*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    text = noisy_line_pattern.sub("", text)

    spill_line_pattern = re.compile(
        r"^\s*.*\b(notified through Notification|w\.e\.f\.|prior to its|dated\s+\d{2}\.\d{2}\.\d{4}|No\.\s*\d+/\d+|Corrigendum)\b.*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    text = spill_line_pattern.sub("", text)

    return text


def _clean_legal_list_item_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()

    text = re.sub(r"\b\d+\[\s*\*\*\*\s*\]", "", text)
    text = re.sub(r"\[\s*\*\*\*\s*\]", "", text)
    text = re.sub(r"\b\d+\[", "", text)

    text = re.sub(r"\(\s*Corrigendum.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*$", "", text)

    spill_patterns = [
        r"\bInserted vide\b.*$",
        r"\bSubstituted vide\b.*$",
        r"\bOmitted vide\b.*$",
        r"\bRenumbered\b.*$",
        r"\bCorrigendum\b.*$",
        r"\bnotified through Notification\b.*$",
        r"\bdated\s+\d{2}\.\d{2}\.\d{4}.*$",
        r"\bw\.e\.f\..*$",
        r"\bprior to its\b.*$",
    ]

    for pat in spill_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    text = text.replace("sub- section", "sub-section")
    text = text.replace("sub section", "sub-section")
    text = text.replace("[", "").replace("]", "").strip()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*\.?\s*$", "", text)
    text = text.strip(" ;,.")

    return text


def _extract_visible_statutory_list_items(text, max_items: int = 18):
    if not text:
        return []

    text = _remove_amendment_noise_but_keep_later_items(text)

    marker_pattern = r"\((?:[ivxlcdm]+[a-z]?|[a-z])\)"

    pattern = re.compile(
        rf"(?ms)^\s*({marker_pattern})\s+(.+?)(?=^\s*{marker_pattern}\s+|\Z)"
    )

    items = []

    for m in pattern.finditer(text):
        marker = m.group(1).strip()
        raw_item = m.group(2).strip()
        cleaned = _clean_legal_list_item_text(raw_item)

        if not cleaned or len(cleaned) < 8:
            continue

        if re.search(
            r"^(inserted|substituted|omitted|renumbered|notified|dated)\b",
            cleaned,
            re.IGNORECASE
        ):
            continue

        items.append((marker, cleaned))

        if len(items) >= max_items:
            break

    seen = set()
    unique_items = []

    for marker, item in items:
        key = (marker.lower(), item.lower())

        if key not in seen:
            unique_items.append((marker, item))
            seen.add(key)

    return unique_items


def extract_visible_list_items(text: str, max_items: int = 18, max_chars_per_item: int = 320):
    items = _extract_visible_statutory_list_items(text, max_items=max_items)

    output = []

    for marker, item in items:
        if len(item) > max_chars_per_item:
            item = item[:max_chars_per_item].rstrip() + "..."

        output.append(f"{marker} {item}")

    return output


def clean_legal_list_answer_text(answer: str) -> str:
    if not isinstance(answer, str):
        return answer

    cleaned_lines = []

    for line in answer.splitlines():
        stripped = line.strip()

        if stripped.startswith("*"):
            line = re.sub(r"\s*\[?\*{3,}\]?.*$", ";", line)
            line = re.sub(r"\s+\d+\[", " ", line)
            line = line.replace("[", "").replace("]", "")
            line = re.sub(r";\s*and\s*;", ";", line)
            line = re.sub(r"\band\s*;", ";", line)
            line = re.sub(r";\s*;", ";", line)
            line = re.sub(r"\s+", " ", line).strip()

            if not line.startswith("*"):
                line = "* " + line.lstrip("* ").strip()

        cleaned_lines.append(line)

    answer = "\n".join(cleaned_lines)
    answer = answer.replace("sub- section", "sub-section")
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


def _is_clean_person_category_list_query(query: str) -> bool:
    q = _qnorm_for_generation(query)

    clean_terms = [
        "compulsory registration",
        "required to be registered",
        "who must register",
        "who is required to register",
        "who are required to register",
        "categories of persons",
        "category of persons",
    ]

    return any(term in q for term in clean_terms)


def _is_clean_person_category_list_source(src) -> bool:
    text = _qnorm_for_generation(_get_source_text_for_list_answer(src))
    heading = _qnorm_for_generation(src.get("full_heading", ""))

    source_has_clean_registration_list = (
        "following categories of persons" in text
        and "required to be registered" in text
    )

    heading_matches = "compulsory registration" in heading

    return source_has_clean_registration_list or (
        heading_matches and "required to be registered" in text
    )


def should_use_deterministic_list_answer(query: str, used_sources) -> bool:
    if not _is_clean_person_category_list_query(query):
        return False

    for src in used_sources or []:
        if not _is_clean_person_category_list_source(src):
            continue

        text = _get_source_text_for_list_answer(src)
        items = _extract_visible_statutory_list_items(text)

        if len(items) >= 3:
            return True

    return False


def build_deterministic_list_answer(query, used_sources):
    if not used_sources:
        return ""

    source = used_sources[0]
    text = _get_source_text_for_list_answer(source)

    citation = source.get("citation_label", "Retrieved source")
    heading = source.get("full_heading", "")
    section_num = source.get("section_or_rule_number", "")
    doc_family = source.get("doc_family", "")

    items = _extract_visible_statutory_list_items(text)

    if not items:
        return ""

    items = items[:15]

    if heading and ">" in heading:
        title = heading.split(">")[-1].strip(" .-")
    elif heading:
        title = heading.strip(" .-")
    else:
        title = ""

    if section_num and title:
        intro = f"Section {section_num} ({title}) lists the following statutory items:"
    elif section_num:
        intro = f"Section {section_num} lists the following statutory items:"
    else:
        intro = "The retrieved provision lists the following statutory items:"

    q = _qnorm_for_generation(query)
    title_lower = title.lower()

    if "registration" in q or "registration" in title_lower:
        intro = f"Section {section_num} lists the following categories of persons who are required to be registered under the Act:"

    bullet_lines = []

    for idx, (marker, item) in enumerate(items):
        ending = "." if idx == len(items) - 1 else ";"
        bullet_lines.append(f"* {marker} {item}{ending}")

    legal_basis = (
        f"{doc_family}, Section {section_num}"
        if doc_family and section_num
        else (heading or citation)
    )

    answer = (
        "Answer:\n"
        + intro
        + "\n\n"
        + "\n".join(bullet_lines)
        + "\n\nLegal basis:\n"
        + legal_basis
        + "\n\nSource:\n"
        + citation
    )

    return clean_legal_list_answer_text(answer)


def extract_consequence_passages(text: str, max_passages: int = 4, max_chars_per_passage: int = 420):
    if not text:
        return []

    blocks = _split_legal_blocks(text)

    keywords = [
        "fails to pay",
        "failure",
        "within a period",
        "one hundred and eighty",
        "paid by him",
        "interest",
        "entitled to avail",
        "avail of the credit",
        "payment made",
        "re-avail",
        "reavail",
        "restore",
        "restoration",
    ]

    passages = []

    for block in blocks:
        lower = block.lower()

        if any(k in lower for k in keywords):
            cleaned = _clean_hint_text(block)

            if len(cleaned) > max_chars_per_passage:
                cleaned = cleaned[:max_chars_per_passage].rstrip() + "..."

            passages.append(cleaned)

            if len(passages) >= max_passages:
                break

    return passages


def build_generation_structure_hints(query: str, used_sources) -> str:
    hint_blocks = []

    if should_use_deterministic_list_answer(query, used_sources):
        for source_idx, src in enumerate(used_sources or [], start=1):
            text = _get_source_text_for_list_answer(src)
            items = extract_visible_list_items(text)

            if items:
                hint_blocks.append(
                    "Visible statutory list items from Source "
                    + str(source_idx)
                    + ":\n"
                    + "\n".join([f"- {item}" for item in items])
                )

    if looks_like_consequence_generation_query(query):
        for source_idx, src in enumerate(used_sources or [], start=1):
            text = _get_source_text_for_list_answer(src)
            passages = extract_consequence_passages(text)

            if passages:
                hint_blocks.append(
                    "Visible consequence / restoration passages from Source "
                    + str(source_idx)
                    + ":\n"
                    + "\n".join([f"- {passage}" for passage in passages])
                )

    if not hint_blocks:
        return ""

    return "\n\n".join(hint_blocks)


# =========================
# CONTEXT BLOCK
# =========================

def build_context_block(query, used_sources, max_chars_per_chunk=2200):
    q = _qnorm_for_generation(query)

    needs_full_start = (
        " vs " in q
        or " versus " in q
        or "difference between" in q
        or "read with" in q
        or "what does the act say and what do the rules add" in q
        or ("act say" in q and "rules add" in q)
    )

    blocks = []

    for i, item in enumerate(used_sources or [], start=1):
        raw_text = item.get("chunk_text", "") or ""

        if needs_full_start:
            excerpt = str(raw_text).strip()

            if len(excerpt) > max_chars_per_chunk:
                excerpt = excerpt[:max_chars_per_chunk].rstrip() + "\n...[truncated]"
        else:
            excerpt = _context_excerpt_for_query(raw_text, query, max_chars_per_chunk)

        block = f"""[Source {i}]
Citation: {item.get("citation_label", "")}
Heading: {item.get("full_heading", "")}
Text:
{excerpt}"""
        blocks.append(block)

    return "\n\n".join(blocks)


# =========================
# PROMPT BUILDER
# =========================

def build_grounded_prompt(query: str, context_block: str) -> str:
    profile = infer_query_profile(query)
    intent = profile["intent"]
    q = _normalize_query(query)

    blocks = []

    if intent == "definition_lookup":
        blocks.append("""
Extra instruction for this question type:
- This is a definition-style question.
- Give the statutory meaning faithfully.
- If a definition chunk is present, answer from that definition, not from a downstream consequence provision.
- Do not add items that are not clearly present in the retrieved text.
- If the definition contains inclusions, exclusions, or conditions, preserve them correctly.
""".strip())

    if intent == "conditions_or_eligibility":
        blocks.append("""
Extra instruction for this question type:
- This is a condition / eligibility question.
- State the main statutory conditions clearly.
- Include the important statutory requirements that appear in the context.
- Do not reduce the answer to one vague sentence if multiple conditions are present.
- Prefer the main substantive provision over nearby procedural provisions if both are present.
""".strip())

    if intent == "procedural_rule":
        blocks.append("""
Extra instruction for this question type:
- This is a procedural-rule question.
- Summarize the main process or rule position.
- If the context mentions forms, documentary requirements, evidentiary requirements, or important conditions, include the main ones.
- Do not give only one thin line if the retrieved rule contains more substance.
""".strip())

    if intent == "comparison":
        blocks.append("""
Extra instruction for this question type:
- This is a comparison question.
- Explain each provision or concept separately first.
- Then state the key difference clearly.
- If both compared concepts are statutory definitions, preserve the definition-level distinction, not only downstream consequence.
""".strip())

    if intent == "act_vs_rule_linking":
        blocks.append("""
Extra instruction for this question type:
- This question asks what the Act says and what the Rules add.
- First state the Act-level legal position.
- Then state what the Rule adds procedurally or operationally.
- Do not answer with Rule-only content if an Act source is present in the context.
- Cite both the Act source and the Rule source separately.
""".strip())

    if intent == "reference":
        blocks.append("""
Extra instruction for this question type:
- This is a section / rule lookup question.
- Summarize the core legal position of the retrieved provision.
- Do not drift to nearby provisions unless they are clearly part of the same retrieved context.
""".strip())

    if profile.get("registration_mode") == "liability":
        blocks.append("""
Extra instruction for this question type:
- This is a registration-liability question.
- Focus on the core liability rule.
- State the turnover threshold rule clearly and directly.
- Do not add exclusions, exemptions, or transition/history points unless the question explicitly asks for them or they are central in the retrieved context.
""".strip())

    if is_subunit_reference_query(query):
        blocks.append("""
Extra instruction for this question type:
- The user is asking about a specific subsection or clause.
- If that exact subunit is visible in the context, answer from that exact subunit.
- Do not answer from the broader section if the requested subunit is present.
- Do not say it is missing if the visible text contains that subunit.
- In the Legal basis line, mention the exact subsection or clause if visible.
""".strip())

    if any(term in q for term in ["by when", "time limit", "deadline", "last date", "cut off", "cutoff"]):
        blocks.append("""
Extra instruction for this question type:
- This is a deadline or cutoff question.
- If the context contains a specific time limit, state it exactly.
- If the context gives two alternative cutoff points, include both and preserve phrases like "whichever is earlier" exactly.
- Do not answer only with general eligibility conditions if a specific cutoff is visible.
- In the Legal basis line, mention the exact subsection if visible.
""".strip())

    if any(term in q for term in ["what happens if", "fails to", "does not pay", "not pay", "consequence"]):
        blocks.append("""
Extra instruction for this question type:
- This is a consequence question.
- Read the nearby provisos carefully.
- If the context contains both an immediate consequence and a later restoration / re-availment / reclaim rule, the Answer must include both.
- Structure the Answer in two parts:
  1. Immediate consequence
  2. Later entitlement / restoration, if visible
- If the text says the recipient is entitled to avail credit after making payment to the supplier, mention that clearly.
- Do not stop after only repayment, reversal, penalty, or interest if the context also gives a later entitlement rule.
- For Legal basis, do not invent subsection numbers. If the text appears in a proviso, write "proviso to Section ..." or "proviso under Section ..." instead of guessing a subsection number.
""".strip())

    if looks_like_listing_generation_query(query):
        blocks.append("""
Extra instruction for this question type:
- This question requires a statutory list if the retrieved context contains numbered or lettered items like (i), (ii), (a), (b), etc.
- In the Answer section, you must include the main visible listed items as bullet points.
- Do not answer only with a generic sentence such as "the section lists categories".
- If the list is long, group or summarize the items concisely, but still mention the main categories visible in the context.
- Do not invent categories that are not present in the retrieved text.
""".strip())

    extra_blocks = "\n\n".join(blocks)

    return f"""
You are a GST bare-law grounded assistant.

Your job:
answer only from the retrieved GST legal text provided below.

Rules you must follow:
1. Stay grounded in the provided context only.
2. Do not give portal workflow, filing steps, compliance advice, case-law interpretation, or practical advisory unless it is explicitly present in the context.
3. If the answer is not clearly supported by the context, say exactly:
   "The answer is not clearly available in this GST bare-law corpus."
4. Prefer faithful paraphrasing over copying long legal text.
5. You may quote only a small critical phrase when exact wording matters.
6. Always include source citations at the end.
7. Do not silently add or remove legal conditions, inclusions, exclusions, scope limitations, deadlines, or exceptions that are visible in the context.
8. Unless the question specifically asks about amendment history, effective dates, substitutions, omissions, or prior wording, treat amendment footnotes as supporting history only and answer from the operative provision text.
9. In the Legal basis line, do not guess subsection numbers. Use an exact subsection or clause only if it is visible in the context. If the relevant text is in a proviso, say "proviso to Section/Rule X" rather than inventing a subsection number.
10. When a provision defines a term using words like "means", treat that definition clause as the controlling definition.
11. Do not define a statutory term from later consequence clauses, provisos, explanations, amendment notes, or phrases like "notwithstanding that...".
12. Do not describe "zero-rated supply" as exempt from GST, exempt from IGST, tax-free, or non-taxable. For definition or comparison questions, explain zero-rated supply from the statutory definition itself: export of goods or services or both, or supply to an SEZ developer or SEZ unit for authorised operations. Mention ITC/refund treatment separately only if it is clearly relevant.
13. When summarizing an enforcement, penalty, detention, seizure, registration cancellation, refund, or procedural provision, do not stop at only the opening condition. If the retrieved provision contains release conditions, penalty/security conditions, notice requirements, time limits, order requirements, or opportunity of hearing, include those key safeguards briefly.

{extra_blocks}

Output format:
Answer:
<your grounded answer in plain language>

Legal basis:
<brief mention of the key section/rule used>

Source:
<Act/Rule family + section/rule + page number or page range>

Note:
<only if needed, for limitation / ambiguity / abstain>

Question:
{query}

Retrieved GST context:
{context_block}
""".strip()


# =========================
# OLLAMA GENERATION
# =========================

def check_ollama_ready():
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=20)
    r.raise_for_status()
    return r.json()


def generate_with_ollama(prompt, max_new_tokens=220, temperature=0.0, num_ctx=4096):
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "num_ctx": num_ctx
        }
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600
    )

    if response.status_code >= 400:
        response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


# =========================
# MAIN ANSWER FUNCTION
# =========================

def answer_gst_query(
    query,
    retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
    context_top_k=DEFAULT_CONTEXT_TOP_K,
    max_chars_per_chunk=DEFAULT_MAX_CHARS_PER_CHUNK,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS
):
    query = str(query or "").strip()
    profile = infer_query_profile(query)

    if should_abstain_for_practical_advisory(query):
        final_answer = (
            "Answer:\n"
            "The answer is not clearly available in this GST bare-law corpus.\n\n"
            "Legal basis:\n"
            "This question asks for practical advisory or latest notification guidance, not only bare-law text.\n\n"
            "Source:\n"
            "No direct statutory source retrieved for this advisory request.\n\n"
            "Note:\n"
            "This assistant is grounded in GST Acts and Rules text, not practical structuring advice or latest notification tracking."
        )

        return {
            "query": query,
            "mode": "out_of_scope_abstain",
            "answer": final_answer,
            "final_answer": final_answer,
            "retrieval_output": {"results": [], "candidates": []},
            "used_sources": [],
            "query_info": profile,
            "context_block": "",
            "prompt": ""
        }

    if is_out_of_scope_query(query) and not looks_like_exact_bare_law_lookup(query):
        final_answer = (
            "Answer:\n"
            "The answer is not clearly available in this GST bare-law corpus.\n\n"
            "Legal basis:\n"
            "This corpus is designed for bare-law provisions from GST Acts and Rules, not portal workflow, filing process, circular-based practice, or broader advisory guidance.\n\n"
            "Source:\n"
            "No direct statutory source retrieved for this practical/advisory request.\n\n"
            "Note:\n"
            "This assistant is grounded in bare-law text, not operational GST portal procedures."
        )

        return {
            "query": query,
            "mode": "out_of_scope_abstain",
            "answer": final_answer,
            "final_answer": final_answer,
            "retrieval_output": {"results": [], "candidates": []},
            "used_sources": [],
            "query_info": profile,
            "context_block": "",
            "prompt": ""
        }

    if looks_like_definition_generation_query(query):
        context_top_k = 1
        max_chars_per_chunk = min(max_chars_per_chunk, 1400)
        max_new_tokens = min(max_new_tokens, 220)

    elif is_subunit_reference_query(query):
        retrieval_top_k = max(retrieval_top_k, 12)
        context_top_k = 1
        max_chars_per_chunk = max(max_chars_per_chunk, 7500)
        max_new_tokens = min(max_new_tokens, 260)

    elif profile.get("intent") == "reference":
        retrieval_top_k = max(retrieval_top_k, 12)

        if is_multi_target_query(query):
            targets = infer_force_targets_for_query(query, profile)
            context_top_k = max(context_top_k, len(targets), 2)
            max_chars_per_chunk = max(max_chars_per_chunk, 5200)
            max_new_tokens = max(max_new_tokens, 360)
        else:
            context_top_k = 1
            max_chars_per_chunk = max(max_chars_per_chunk, 7500)
            max_new_tokens = min(max_new_tokens, 260)

    elif looks_like_condition_query(query):
        retrieval_top_k = max(retrieval_top_k, 8)
        context_top_k = min(context_top_k, 2)
        max_chars_per_chunk = max(max_chars_per_chunk, 4200)
        max_new_tokens = min(max_new_tokens, 260)

    elif looks_like_procedural_rule_query(query):
        retrieval_top_k = max(retrieval_top_k, 8)
        context_top_k = min(context_top_k, 2)
        max_chars_per_chunk = max(max_chars_per_chunk, 3800)
        max_new_tokens = min(max_new_tokens, 260)

    elif looks_like_comparison_query(query):
        context_top_k = min(context_top_k, 2)
        max_chars_per_chunk = max(max_chars_per_chunk, 2600)
        max_new_tokens = min(max_new_tokens, 280)

    retrieval_output = retrieve_hybrid(query, top_k=retrieval_top_k)

    if isinstance(retrieval_output, dict):
        retrieval_output["query_info"] = profile

    used_sources = prune_generation_sources(
        query=query,
        retrieval_output=retrieval_output,
        max_chunks=context_top_k
    )

    used_sources = force_include_target_sources(
        query=query,
        retrieval_output=retrieval_output,
        used_sources=used_sources,
        context_top_k=context_top_k
    )

    used_sources = remove_irrelevant_comparison_contexts(
        query=query,
        profile=profile,
        used_sources=used_sources
    )

    if should_use_deterministic_list_answer(query, used_sources):
        final_answer = build_deterministic_list_answer(query, used_sources)

        if isinstance(final_answer, str) and final_answer.strip():
            return {
                "query": query,
                "mode": "grounded_generation",
                "answer": final_answer,
                "final_answer": final_answer,
                "used_sources": used_sources,
                "retrieval_output": retrieval_output,
                "query_info": retrieval_output.get("query_info", profile),
                "context_block": "",
                "prompt": ""
            }

    context_block = build_context_block(
        query=query,
        used_sources=used_sources,
        max_chars_per_chunk=max_chars_per_chunk
    )

    generation_hints = build_generation_structure_hints(
        query=query,
        used_sources=used_sources
    )

    if generation_hints:
        context_block = (
            context_block
            + "\n\n[Structure hints extracted from the retrieved text]\n"
            + generation_hints
        )

    if looks_like_listing_generation_query(query):
        max_new_tokens = max(max_new_tokens, 520)

    if looks_like_consequence_generation_query(query):
        max_new_tokens = max(max_new_tokens, 360)

    prompt = build_grounded_prompt(query, context_block)

    final_answer = generate_with_ollama(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )

    return {
        "query": query,
        "mode": "grounded_generation",
        "answer": final_answer,
        "final_answer": final_answer,
        "retrieval_output": retrieval_output,
        "used_sources": used_sources,
        "query_info": retrieval_output.get("query_info", profile),
        "context_block": context_block,
        "prompt": prompt
    }


# =========================
# PUBLIC CHATBOT FORMATTER
# =========================

def clean_public_answer(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    bad_sections = [
        "Retrieved GST context:",
        "CONTEXT PASSED TO GENERATOR",
        "You are a GST bare-law grounded assistant.",
        "Rules you must follow:",
    ]

    for marker in bad_sections:
        if marker in text:
            text = text.split(marker)[0].strip()

    answer_pos = text.find("Answer:")

    if answer_pos != -1:
        text = text[answer_pos:].strip()

    return text


def format_public_sources(used_sources):
    public_sources = []
    seen = set()

    for src in used_sources or []:
        if not isinstance(src, dict):
            continue

        citation = str(src.get("citation_label", "") or "").strip()
        heading = str(src.get("full_heading", "") or "").strip()
        doc_family = str(src.get("doc_family", "") or "").strip()
        section_or_rule = str(src.get("section_or_rule_number", "") or "").strip()

        if not citation and not heading:
            continue

        key = (citation, heading)

        if key in seen:
            continue

        seen.add(key)

        public_sources.append({
            "citation": citation,
            "heading": heading,
            "doc_family": doc_family,
            "section_or_rule": section_or_rule
        })

    return public_sources


def build_public_chatbot_response(out: dict, original_query: str, resolved_query: str = None) -> dict:
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
        "sources": format_public_sources(out.get("used_sources", []))
    }


# =========================
# CHATBOT HISTORY + FOLLOW-UP HANDLING
# =========================

@dataclass
class ChatTurn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: list = field(default_factory=list)


def _last_user_question(history):
    for turn in reversed(history):
        if turn.role == "user":
            return turn.content

    return ""


def _last_assistant_sources(history):
    for turn in reversed(history):
        if turn.role == "assistant" and turn.sources:
            return turn.sources

    return []


def looks_like_followup_query(query: str) -> bool:
    q = query.lower().strip()
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
        "that provision"
    ]

    pronoun_terms = {"it", "this", "that", "these", "those", "same", "above"}

    has_explicit_ref = bool(extract_query_refs(q)[0] or extract_query_refs(q)[1])

    if has_explicit_ref:
        return False

    if any(p in q for p in followup_phrases):
        return True

    if len(tokens) <= 8 and any(t in pronoun_terms for t in tokens):
        return True

    return False


def resolve_followup_query(query: str, history) -> str:
    """
    Converts short follow-up questions into retrieval-friendly queries.

    This does not inject legal facts or answers.
    It only carries forward the previous question/source hint so the backend
    can retrieve and answer from the same legal provision.
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

        if citation or heading:
            source_hint_parts.append(f"{citation} {heading}".strip())

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


class GSTChatSession:
    def __init__(self, session_id=None, max_history_turns=12):
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history_turns = max_history_turns
        self.history = []

    def ask(
        self,
        query: str,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        context_top_k: int = DEFAULT_CONTEXT_TOP_K,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ):
        query = str(query or "").strip()

        if not query:
            return {
                "ok": False,
                "query": query,
                "resolved_query": query,
                "mode": "empty_query",
                "answer": "Please enter a GST law question.",
                "sources": []
            }

        try:
            resolved_query = resolve_followup_query(query, self.history)

            out = answer_gst_query(
                query=resolved_query,
                retrieval_top_k=retrieval_top_k,
                context_top_k=context_top_k,
                max_chars_per_chunk=max_chars_per_chunk,
                max_new_tokens=max_new_tokens
            )

            public_response = build_public_chatbot_response(
                out=out,
                original_query=query,
                resolved_query=resolved_query
            )

        except Exception as e:
            public_response = {
                "ok": False,
                "query": query,
                "resolved_query": query,
                "mode": "system_error",
                "answer": "Sorry, the GST assistant could not process this question right now.",
                "sources": [],
                "error": str(e)
            }

        self.history.append(ChatTurn(role="user", content=query))
        self.history.append(
            ChatTurn(
                role="assistant",
                content=public_response["answer"],
                sources=public_response.get("sources", [])
            )
        )

        max_messages = self.max_history_turns * 2

        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        return public_response

    def get_history_for_ui(self):
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "sources": turn.sources
            }
            for turn in self.history
        ]

    def reset(self):
        self.history = []

        return {
            "session_id": self.session_id,
            "status": "reset_done"
        }