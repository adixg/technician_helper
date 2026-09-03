"""Pure metric functions. No I/O, no network — safe to run anywhere.

Every function returns a float in [0, 1] (higher is better) unless noted, and
treats an empty expectation as vacuously satisfied (1.0) so that partially
labelled golden cases still score sensibly.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected chunk ids present in the top-k retrieved ids."""
    exp = set(expected)
    if not exp:
        return 1.0
    return len(exp & set(retrieved[:k])) / len(exp)


def hit_rate_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected id is in the top-k, else 0.0."""
    exp = set(expected)
    if not exp:
        return 1.0
    return 1.0 if exp & set(retrieved[:k]) else 0.0


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the first expected id in the retrieved list."""
    exp = set(expected)
    if not exp:
        return 1.0
    for i, rid in enumerate(retrieved, start=1):
        if rid in exp:
            return 1.0 / i
    return 0.0


def schema_valid(result: dict[str, Any] | None) -> float:
    """1.0 if the answer object passes the pipeline's schema validation."""
    if not result:
        return 0.0
    from technician_helper.pipeline.rag_fusion import validate_output

    try:
        validate_output(dict(result))
        return 1.0
    except Exception:
        return 0.0


def groundedness(result: dict[str, Any] | None, evidence_texts: list[str]) -> float:
    """Fraction of generated claims whose salient tokens appear in the evidence.

    A claim counts as supported when at least half of its content tokens are
    found somewhere in the retrieved chunk text.
    """
    if not result:
        return 0.0

    claims: list[str] = []
    claims += [str(c.get("cause", "")) for c in result.get("likely_causes", [])]
    claims += [str(r.get("section_title", "")) for r in result.get("manual_references", [])]
    claims += [str(s.get("summary", "")) for s in result.get("similar_incidents", [])]
    claims = [c for c in claims if c.strip()]
    if not claims:
        return 1.0

    evidence = set()
    for text in evidence_texts:
        evidence |= _tokens(text)

    supported = 0
    scored = 0
    for claim in claims:
        ct = _tokens(claim)
        if not ct:
            continue
        scored += 1
        if len(ct & evidence) / len(ct) >= 0.5:
            supported += 1
    return supported / scored if scored else 1.0


def field_match(result: dict[str, Any] | None, expected: dict[str, Any]) -> float:
    """Fraction of the expectations in ``expected`` that the answer satisfies.

    Recognised keys: ``escalation_needed`` (bool), ``confidence`` (str or list of
    allowed values), ``must_mention`` (list of substrings, case-insensitive),
    ``manual_refs_expected`` (bool).
    """
    if result is None:
        return 0.0

    checks: list[bool] = []
    if "escalation_needed" in expected:
        checks.append(bool(result.get("escalation_needed")) == bool(expected["escalation_needed"]))
    if "confidence" in expected:
        allowed = expected["confidence"]
        allowed = [allowed] if isinstance(allowed, str) else list(allowed)
        checks.append(result.get("confidence") in allowed)
    if "must_mention" in expected:
        blob = json.dumps(result).lower()
        checks.append(all(str(term).lower() in blob for term in expected["must_mention"]))
    if "manual_refs_expected" in expected:
        has_refs = len(result.get("manual_references", [])) > 0
        checks.append(has_refs == bool(expected["manual_refs_expected"]))

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


def completeness(result: dict[str, Any] | None) -> float:
    """1.0 when both likely_causes and recommended_checks are non-empty."""
    if not result:
        return 0.0
    filled = sum(1 for key in ("likely_causes", "recommended_checks") if result.get(key))
    return filled / 2
