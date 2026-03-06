"""Post-transcription subtitle segmentation and linguistic fixes.

Rebuilds raw Whisper segments into subtitle-friendly chunks using
word-level timestamps, then applies spaCy POS tagging to fix
dangling function words at segment boundaries.
"""

from __future__ import annotations

import copy

from pgw.core.models import SubtitleSegment
from pgw.utils.console import warning
from pgw.utils.spacy import load_spacy_model
from pgw.utils.text import (
    APOSTROPHES,
    MAX_MERGE_TRAIL_WORDS,
    MAX_SEGMENT_CHARS,
    MAX_SEGMENT_DURATION,
    MERGE_CHAR_SLACK,
    MERGE_GAP_THRESHOLD,
    SENTENCE_END_CHARS,
    SPEECH_GAP_THRESHOLD,
)

# POS tags that should not dangle at the end of a subtitle segment.
# DET   = determiners (le/la/les/the/der/die/das/el/la/los...)
# ADP   = adpositions/prepositions (de/en/à/of/in/to/von/mit...)
# CCONJ = coordinating conjunctions (et/ou/mais/and/or/but/und/oder...)
# SCONJ = subordinating conjunctions (que/qui/parce/when/if/dass/weil...)
# AUX   = auxiliaries (est/a/sera/is/has/will/ist/hat...)
_DANGLING_POS = {"DET", "ADP", "CCONJ", "SCONJ", "AUX"}


def regroup_for_subtitles(result, max_chars: int = MAX_SEGMENT_CHARS) -> None:
    """Rebuild segments from word-level timestamps for subtitle display.

    Merges all segments, then re-splits using punctuation, speech gaps,
    and length constraints.  This produces subtitle-friendly segments
    with correct word-level timestamps.
    """
    result.ignore_special_periods()
    result.clamp_max()
    result.merge_all_segments()
    result.split_by_punctuation([(".", " "), "?", "!", "。", "？", "！"])
    result.split_by_gap(SPEECH_GAP_THRESHOLD)
    result.split_by_punctuation([(",", " "), ";", "，", "；"], min_words=4)
    result.split_by_length(max_chars=max_chars)
    try:
        result.split_by_duration(max_dur=MAX_SEGMENT_DURATION)
    except ValueError:
        warning("split_by_duration skipped (1-word segment edge case).")
    result.merge_by_gap(MERGE_GAP_THRESHOLD, max_words=3, max_chars=max_chars, is_sum_max=True)
    _merge_short_trailing(result, max_chars)
    result.clamp_max()


def _merge_short_trailing(
    result, max_chars: int, max_trail_words: int = MAX_MERGE_TRAIL_WORDS
) -> None:
    """Merge short trailing fragments into the previous segment.

    Handles cases where split_by_length or split_by_duration creates short
    trailing fragments that complete the previous segment's sentence.
    Only merges when the previous segment does NOT end with sentence
    punctuation (indicating the fragment is a continuation, not a new
    sentence).  Uses MERGE_CHAR_SLACK to allow slightly longer combined
    segments rather than leaving dangling fragments.

    Operates on stable-ts WhisperResult segments with word-level data.
    """
    segments = result.segments
    i = len(segments) - 1
    while i > 0:
        seg = segments[i]
        prev = segments[i - 1]

        if len(seg.words) <= max_trail_words and len(prev.words) >= 1:
            prev_text = prev.text.strip()
            seg_text = seg.text.strip()
            gap = seg.start - prev.end if hasattr(seg, "start") else 0.0
            if (
                prev_text
                and gap < SPEECH_GAP_THRESHOLD
                and prev_text[-1] not in SENTENCE_END_CHARS
                and len(prev_text) + 1 + len(seg_text) <= max_chars + MERGE_CHAR_SLACK
            ):
                # Move all words from seg into prev
                for w in seg.words:
                    prev.words.append(w)
                    w.segment = prev
                prev.reassign_ids()
                segments.pop(i)
        i -= 1


def fix_overlapping_timestamps(
    segments: list[SubtitleSegment],
) -> list[SubtitleSegment]:
    """Fix overlapping timestamps between adjacent segments.

    When seg[n].start < seg[n-1].end, snaps seg[n].start to seg[n-1].end.
    Common with API transcription where word timestamps can slightly overlap.
    """
    if len(segments) <= 1:
        return segments

    fixed = 0
    for i in range(1, len(segments)):
        if segments[i].start < segments[i - 1].end:
            segments[i].start = segments[i - 1].end
            segments[i].end = max(segments[i].end, segments[i].start)
            fixed += 1

    if fixed:
        from pgw.utils.console import debug

        debug(f"Fixed {fixed} overlapping timestamp(s).")

    return segments


def postprocess_segments(
    segments: list[SubtitleSegment],
    language: str,
) -> list[SubtitleSegment]:
    """Apply standard postprocessing: fix overlaps, then fix dangling clitics."""
    segments = fix_overlapping_timestamps(segments)
    segments = fix_dangling_clitics(segments, language)
    return segments


def fix_dangling_function_words(result, language: str) -> None:
    """Move dangling function words from segment ends to the next segment.

    Uses spaCy POS tagging to detect function words (DET, ADP, CCONJ,
    SCONJ, AUX) at segment boundaries.  Works for any language with a
    spaCy model.  Falls back silently if spaCy is not installed.
    """
    nlp = load_spacy_model(language)
    if nlp is None:
        return

    segments = result.segments
    for i in range(len(segments) - 1):
        words = segments[i].words
        if len(words) <= 1:
            continue  # Don't empty a segment

        # Check if the last Whisper word ends with an apostrophe (clitic)
        last_word_text = words[-1].word.strip()
        if last_word_text and last_word_text[-1] in APOSTROPHES:
            word = words.pop()
            segments[i].reassign_ids()
            segments[i + 1].words.insert(0, word)
            word.segment = segments[i + 1]
            segments[i + 1].reassign_ids()
            continue

        # Run POS tagger on full segment text for context-aware tagging
        segment_text = segments[i].text.strip()
        if not segment_text:
            continue

        doc = nlp(segment_text)
        if not doc:
            continue

        # Check if the last token is a function word (DET or ADP)
        last_token = doc[-1]
        if last_token.pos_ not in _DANGLING_POS:
            continue

        # Move word from end of current segment to start of next
        word = words.pop()
        segments[i].reassign_ids()

        segments[i + 1].words.insert(0, word)
        word.segment = segments[i + 1]
        segments[i + 1].reassign_ids()

    # Drop segments that became empty (all words moved out)
    result.segments = [s for s in result.segments if s.words]


def fix_dangling_clitics(segments: list[SubtitleSegment], language: str) -> list[SubtitleSegment]:
    """Move dangling clitics/function words from segment ends to the next segment.

    Text-level version of fix_dangling_function_words — operates on
    SubtitleSegment text (no WhisperResult/word objects needed). Suitable
    for the API transcription path.

    Detects two patterns:
    1. Text ending with an apostrophe (Romance clitics: l', d', qu', etc.)
       — uses simple text check, handles spaCy splitting "l'" into ["l", "'"]
    2. Trailing function word POS tags (DET, ADP, CCONJ, SCONJ, AUX) — uses spaCy
    """
    nlp = load_spacy_model(language)
    if nlp is None:
        return segments

    fixed = [copy.copy(seg) for seg in segments]

    for i in range(len(fixed) - 1):
        text = fixed[i].text.strip()
        if not text:
            continue

        # Pattern 1: text ends with apostrophe (clitic like l', d', qu')
        # Move the trailing "word'" to the next segment via text splitting
        if text[-1] in APOSTROPHES:
            space_idx = text.rfind(" ")
            if space_idx >= 0:
                dangling = text[space_idx + 1 :]
                fixed[i].text = text[:space_idx].rstrip()
                fixed[i + 1].text = dangling + fixed[i + 1].text.lstrip()
            else:
                # Entire segment is a clitic (e.g. "l'")
                fixed[i].text = ""
                fixed[i + 1].text = text + fixed[i + 1].text.lstrip()
            continue

        # Pattern 2: trailing DET/ADP via spaCy POS tagging
        doc = nlp(text)
        if not doc or len(doc) <= 1:
            continue

        last_token = doc[-1]
        if last_token.pos_ not in _DANGLING_POS:
            continue

        # Move the dangling token text to the next segment
        fixed[i].text = text[: last_token.idx].rstrip()
        fixed[i + 1].text = last_token.text.strip() + " " + fixed[i + 1].text.lstrip()

    return [seg for seg in fixed if seg.text.strip()]
