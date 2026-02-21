"""Vocabulary summary generation for subtitle segments.

Analyzes word frequency and CEFR difficulty from transcribed subtitles,
producing a per-video vocabulary profile.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pgw.core.models import SubtitleSegment
from pgw.utils.spacy import load_spacy_model

# POS tags to skip during vocabulary extraction
_SKIP_POS = {"PUNCT", "SPACE", "NUM", "SYM", "X"}

# CEFR bins based on wordfreq zipf_frequency values.
# zipf ≈ log10(frequency_per_billion). Higher = more common.
_CEFR_BINS = [
    (5.0, "A1"),  # very common: the, is, I, de, le
    (4.0, "A2"),  # common: house, eat, big
    (3.0, "B1"),  # intermediate: opportunity, develop
    (2.0, "B2"),  # upper-intermediate: comprehensive, deteriorate
    (1.0, "C1"),  # advanced: ubiquitous, ephemeral
]

_CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def zipf_to_cefr(zipf: float) -> str:
    """Map a wordfreq zipf_frequency value to an estimated CEFR level."""
    for threshold, level in _CEFR_BINS:
        if zipf > threshold:
            return level
    return "C2"


@dataclass
class WordInfo:
    """Collected info about a unique word (lemma + POS)."""

    word: str  # surface form (first occurrence)
    lemma: str
    pos: str
    zipf: float
    cefr: str
    count: int
    context: str  # segment text where word first appeared
    translation: str  # translated segment text (if available)


def generate_vocab_summary(
    segments: list[SubtitleSegment],
    language: str,
    translated_segments: list[SubtitleSegment] | None = None,
    top_n: int = 30,
) -> dict:
    """Analyze vocabulary from subtitle segments.

    Args:
        segments: Original subtitle segments.
        language: ISO 639-1 language code.
        translated_segments: Paired translated segments (same length), optional.
        top_n: Number of rarest words to include.

    Returns:
        Dict with vocabulary statistics and top rare words.
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        raise ImportError("wordfreq is not installed. Install with: uv sync --extra vocab")

    nlp = load_spacy_model(language, enable_lemmatizer=True)
    if nlp is None:
        raise RuntimeError(f"No spaCy model available for language '{language}'")

    texts = [seg.text for seg in segments]
    trans_texts = [seg.text for seg in translated_segments] if translated_segments else None

    # Track unique lemmas and their info
    lemma_key_to_info: dict[tuple[str, str], WordInfo] = {}  # (lemma, pos) → WordInfo
    total_words = 0
    cefr_counts: Counter[str] = Counter()

    for doc_idx, doc in enumerate(nlp.pipe(texts, batch_size=50)):
        for token in doc:
            if token.pos_ in _SKIP_POS or token.is_space:
                continue

            total_words += 1
            lemma = token.lemma_.lower()
            pos = token.pos_
            key = (lemma, pos)

            if key in lemma_key_to_info:
                lemma_key_to_info[key].count += 1
                continue

            # First occurrence — compute frequency and CEFR
            zipf = zipf_frequency(lemma, language)
            if zipf == 0:
                # Fallback: try surface form
                zipf = zipf_frequency(token.text.lower(), language)
            cefr = zipf_to_cefr(zipf)
            cefr_counts[cefr] += 1

            context = texts[doc_idx]
            translation = trans_texts[doc_idx] if trans_texts and doc_idx < len(trans_texts) else ""

            lemma_key_to_info[key] = WordInfo(
                word=token.text,
                lemma=lemma,
                pos=pos,
                zipf=zipf,
                cefr=cefr,
                count=1,
                context=context,
                translation=translation,
            )

    # Compute estimated video level (weighted average CEFR)
    if lemma_key_to_info:
        total_level = sum(_CEFR_ORDER[info.cefr] for info in lemma_key_to_info.values())
        avg_level = total_level / len(lemma_key_to_info)
        # Round to nearest CEFR
        level_names = list(_CEFR_ORDER.keys())
        level_idx = min(max(round(avg_level) - 1, 0), len(level_names) - 1)
        estimated_level = level_names[level_idx]
    else:
        estimated_level = "A1"

    # Sort by zipf ascending (rarest first), then by count descending
    sorted_words = sorted(
        lemma_key_to_info.values(),
        key=lambda w: (w.zipf, -w.count),
    )
    top_rare = sorted_words[:top_n]

    return {
        "language": language,
        "total_words": total_words,
        "unique_words": len({info.word.lower() for info in lemma_key_to_info.values()}),
        "unique_lemmas": len(lemma_key_to_info),
        "cefr_distribution": {level: cefr_counts.get(level, 0) for level in _CEFR_ORDER},
        "estimated_level": estimated_level,
        "top_rare_words": [
            {
                "word": info.word,
                "lemma": info.lemma,
                "pos": info.pos,
                "cefr": info.cefr,
                "zipf": round(info.zipf, 2),
                "count": info.count,
                "context": info.context,
                "translation": info.translation,
            }
            for info in top_rare
        ],
    }
