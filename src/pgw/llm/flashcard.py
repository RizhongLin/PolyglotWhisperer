"""LLM-driven flashcard refinement.

Takes batches of raw cards (front, surface, language, optional context
sentence) and produces structured output — lemma, POS, polished
definition, an example sentence pair, optional mnemonic. Output is
keyed JSON so we can refine 20 cards in one call and self-anchor
against silent merges/drops, mirroring ``llm/translator.py``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from pgw.core.config import LLMConfig
from pgw.llm.client import complete

logger = logging.getLogger(__name__)


# Output length caps — LLM is told and the schema enforces.
_LEMMA_MAX = 64
_POS_MAX = 16
_DEFINITION_MAX = 140
_EXAMPLE_MAX = 240
_MNEMONIC_MAX = 200


@dataclass(frozen=True)
class RefineInput:
    """Single card seed handed to the LLM."""

    surface: str
    language: str
    target_language: str
    context: str | None = None
    prior_back: str | None = None


@dataclass(frozen=True)
class RefineOutput:
    """Structured result for one card."""

    lemma: str
    pos: str
    definition: str
    example_source: str
    example_target: str
    mnemonic: str | None


def build_flashcard_schema(count: int, *, want_mnemonic: bool) -> dict:
    """Strict json_schema for batched flashcard refinement.

    Mirrors the keyed-JSON convention of ``build_translation_schema``:
    keys ``"1".."{count}"`` are required + ``additionalProperties:false``
    so a silent merge / drop / extra is structurally visible.
    """
    if count < 1:
        raise ValueError("count must be >= 1")

    item: dict = {
        "type": "object",
        "additionalProperties": False,
        "required": ["lemma", "pos", "definition", "example_source", "example_target", "mnemonic"],
        "properties": {
            "lemma": {"type": "string", "minLength": 1, "maxLength": _LEMMA_MAX},
            "pos": {"type": "string", "maxLength": _POS_MAX},
            "definition": {"type": "string", "minLength": 1, "maxLength": _DEFINITION_MAX},
            "example_source": {"type": "string", "minLength": 1, "maxLength": _EXAMPLE_MAX},
            "example_target": {"type": "string", "minLength": 1, "maxLength": _EXAMPLE_MAX},
            "mnemonic": (
                {
                    "type": ["string", "null"],
                    "maxLength": _MNEMONIC_MAX,
                }
                if want_mnemonic
                else {"type": "null"}
            ),
        },
    }

    keys = [str(i) for i in range(1, count + 1)]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "flashcard_refine_batch",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": keys,
                "properties": {k: item for k in keys},
            },
            "strict": True,
        },
    }


def _system_prompt(source_lang: str, target_lang: str, *, want_mnemonic: bool) -> str:
    """Static system message — same shape as translator/refine."""
    base = (
        f"You are a {source_lang}->{target_lang} flashcard editor.\n"
        "For each input item keyed 1..N, return a JSON object with the keys:\n"
        f"- lemma: dictionary form in {source_lang}\n"
        "- pos: spaCy UPOS tag (NOUN, VERB, ADJ, ADV, PROPN, ...). Use UNKNOWN if unclear.\n"
        f"- definition: one short {target_lang} gloss, no examples, no quotes, "
        f"<= {_DEFINITION_MAX} characters\n"
        f"- example_source: a natural {source_lang} sentence using the lemma, "
        f"<= 18 words. Prefer the provided context sentence verbatim if it is good; "
        "otherwise invent one.\n"
        f"- example_target: faithful {target_lang} translation of example_source\n"
    )
    if want_mnemonic:
        base += (
            f"- mnemonic: optional 1-line memory hook in {target_lang}, "
            f"<= {_MNEMONIC_MAX} characters; null if none\n"
        )
    else:
        base += "- mnemonic: always null\n"
    base += (
        "\n"
        "Never invent items not in the input. Never merge items. Return only "
        "the JSON object, no commentary."
    )
    return base


def _user_payload(items: list[RefineInput]) -> str:
    """Format the batch as keyed JSON the model echoes the structure of."""
    payload = {
        str(i + 1): {
            "surface": it.surface,
            "language": it.language,
            "target_language": it.target_language,
            **({"context": it.context} if it.context else {}),
            **({"prior_back": it.prior_back} if it.prior_back else {}),
        }
        for i, it in enumerate(items)
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def refine_batch(
    items: list[RefineInput],
    config: LLMConfig,
    *,
    want_mnemonic: bool = False,
) -> list[RefineOutput]:
    """Refine up to ~20 cards in a single LLM call.

    Returns the same length as ``items`` — order preserved by the keyed
    JSON. Raises ``ValueError`` if the model returns a malformed payload
    that can't be reconciled.
    """
    if not items:
        return []

    # Items typically share (source, target) language; if not we still
    # want to honour the first card's pair in the prompt — heterogeneous
    # batches are an upstream-orchestrator decision and shouldn't reach
    # here in practice.
    source_lang = items[0].language
    target_lang = items[0].target_language

    messages = [
        {
            "role": "system",
            "content": _system_prompt(source_lang, target_lang, want_mnemonic=want_mnemonic),
        },
        {"role": "user", "content": _user_payload(items)},
    ]

    raw = complete(
        messages,
        config,
        json_schema=build_flashcard_schema(len(items), want_mnemonic=want_mnemonic),
        expected_count=len(items),
    )
    parsed = _parse(raw, expected=len(items))
    return [_to_output(parsed[str(i + 1)]) for i in range(len(items))]


def _parse(raw: str, *, expected: int) -> dict[str, dict]:
    """Robust JSON parse: keyed object first, then array fallback.

    Mirrors ``translator.parse_response``'s tolerance — some models
    occasionally wrap in ``{"items": [...]}`` even with strict schema.
    """
    raw = raw.strip()
    # Strip Markdown fences just in case.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"flashcard refine: invalid JSON: {exc}") from exc

    if isinstance(obj, dict) and all(str(i + 1) in obj for i in range(expected)):
        return {str(i + 1): obj[str(i + 1)] for i in range(expected)}

    # Array fallback — wrap into keyed dict.
    if isinstance(obj, dict):
        for k in ("items", "cards", "results"):
            arr = obj.get(k)
            if isinstance(arr, list) and len(arr) == expected:
                return {str(i + 1): arr[i] for i in range(expected)}
    if isinstance(obj, list) and len(obj) == expected:
        return {str(i + 1): obj[i] for i in range(expected)}

    raise ValueError(f"flashcard refine: response shape mismatch (expected {expected})")


def _to_output(item: dict) -> RefineOutput:
    """Coerce one parsed item into a ``RefineOutput`` with length safety."""

    def _str(key: str, *, max_len: int) -> str:
        val = item.get(key)
        if not isinstance(val, str):
            raise ValueError(f"flashcard refine: missing/invalid {key!r}")
        return val.strip()[:max_len]

    mnemonic_raw = item.get("mnemonic")
    mnemonic = (
        mnemonic_raw.strip()[:_MNEMONIC_MAX]
        if isinstance(mnemonic_raw, str) and mnemonic_raw.strip()
        else None
    )
    return RefineOutput(
        lemma=_str("lemma", max_len=_LEMMA_MAX),
        pos=_str("pos", max_len=_POS_MAX),
        definition=_str("definition", max_len=_DEFINITION_MAX),
        example_source=_str("example_source", max_len=_EXAMPLE_MAX),
        example_target=_str("example_target", max_len=_EXAMPLE_MAX),
        mnemonic=mnemonic,
    )
