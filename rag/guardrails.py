"""Security guardrails around the LLM call.

Defence in depth, because no single layer stops prompt injection:
  1. Input  : size limits, control-char stripping, high-confidence injection patterns.
  2. Prompt : retrieved text is wrapped as untrusted <document> data; the system
              prompt tells the model never to follow instructions found inside it.
  3. Output : structured JSON only, citations must reference chunks we actually
              retrieved, and a canary token detects system-prompt leakage.
  4. Logs   : PII is redacted before anything is written.
"""
from __future__ import annotations

import re
import unicodedata

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 6

# Deliberately narrow: block only patterns that are almost never legitimate in
# a Q&A about this project. Everything else is handled by the prompt/output layers.
_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|all|earlier)\b.{0,20}\b(instruction|rule|prompt|direction)",
    r"\bdisregard\b.{0,40}\b(instruction|rule|prompt|guideline)",
    r"\b(reveal|print|show|repeat|output|leak|tell me)\b.{0,40}\b(system prompt|hidden prompt|your instructions|initial prompt|developer message)",
    r"\byou are now\b.{0,40}\b(dan|unrestricted|jailbroken|developer mode|no restrictions)",
    r"\b(developer|god|jailbreak|dan)\s+mode\b",
    r"\bpretend\b.{0,40}\b(no|without)\b.{0,20}\b(rules|restrictions|guidelines|filters)",
    r"<\s*/?\s*(system|document|documents|instructions)\s*>",
    r"\bapi[_ ]?key\b|\banthropic_api_key\b|\benv(ironment)? variables?\b",
]
_INJECTION_RE = [re.compile(p, re.I | re.S) for p in _INJECTION_PATTERNS]

_PII_RE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]+"), "[SECRET]"),
]


class GuardrailViolation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sanitize(text: str) -> str:
    """Normalise unicode (defeats full-width/homoglyph tricks) and drop control chars."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()


def check_input(question: str) -> str:
    q = sanitize(question)
    if not q:
        raise GuardrailViolation("empty", "Question is empty.")
    if len(q) > MAX_QUESTION_CHARS:
        raise GuardrailViolation("too_long", f"Question exceeds {MAX_QUESTION_CHARS} characters.")
    if looks_like_injection(q):
        raise GuardrailViolation("prompt_injection",
                                 "This request looks like an attempt to override the assistant's instructions.")
    return q


def looks_like_injection(text: str) -> bool:
    return any(r.search(text) for r in _INJECTION_RE)


def redact_pii(text: str) -> str:
    for rx, repl in _PII_RE:
        text = rx.sub(repl, text)
    return text


def escape_document(text: str) -> str:
    """Stop retrieved text from closing our XML wrapper and posing as instructions."""
    return text.replace("<", "&lt;").replace(">", "&gt;")
