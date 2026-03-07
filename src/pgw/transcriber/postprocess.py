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
    MAX_WORDS_GAP_MERGE,
    MERGE_CHAR_SLACK,
    MERGE_GAP_THRESHOLD,
    MIN_WORDS_CLAUSE_SPLIT,
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


def _is_dangling(token) -> bool:
    """Check if a token should not dangle at the end of a subtitle segment.

    Catches function words (DET, ADP, CCONJ, SCONJ, AUX) and relative
    pronouns (qui, où, dont, laquelle, etc. — identified by PronType=Rel).
    """
    if token.pos_ in _DANGLING_POS:
        return True
    if token.pos_ == "PRON" and token.morph.get("PronType") == ["Rel"]:
        return True
    return False


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
    result.split_by_punctuation([(",", " "), ";", "，", "；"], min_words=MIN_WORDS_CLAUSE_SPLIT)
    result.split_by_length(max_chars=max_chars)
    try:
        result.split_by_duration(max_dur=MAX_SEGMENT_DURATION)
    except ValueError:
        warning("split_by_duration skipped (1-word segment edge case).")
    result.merge_by_gap(
        MERGE_GAP_THRESHOLD, max_words=MAX_WORDS_GAP_MERGE, max_chars=max_chars, is_sum_max=True
    )
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


def fix_false_sentence_breaks(
    segments: list[SubtitleSegment],
    language: str,
) -> list[SubtitleSegment]:
    """Merge segments that were falsely split at abbreviation periods.

    Uses spaCy's sentence segmenter as the oracle: joins each segment's
    text with the next, and if spaCy sees them as one sentence (i.e. the
    first token of the next segment is NOT a sentence start), merges the
    segments back together — provided the combined length stays within
    MAX_SEGMENT_CHARS + MERGE_CHAR_SLACK.

    This fixes splits like ``M. | Macron`` or ``Dr. | Dupont`` without
    hardcoding abbreviation lists.
    """
    nlp = load_spacy_model(language, enable_parser=True)
    if nlp is None:
        return segments

    merge_limit = MAX_SEGMENT_CHARS + MERGE_CHAR_SLACK
    fixed = list(segments)
    i = 0
    merged_count = 0

    while i < len(fixed) - 1:
        cur_text = fixed[i].text.strip()
        next_text = fixed[i + 1].text.strip()

        # Only check segments where current ends with sentence-end punctuation
        if not cur_text or cur_text[-1] not in SENTENCE_END_CHARS:
            i += 1
            continue

        combined = cur_text + " " + next_text
        if len(combined) > merge_limit:
            i += 1
            continue

        doc = nlp(combined)
        # Find the token that starts the next segment's text
        cur_char_end = len(cur_text) + 1  # +1 for the joining space
        false_break = False
        for token in doc:
            if token.idx >= cur_char_end:
                # If this token is NOT a sentence start, the split was false.
                # is_sent_start is True/False/None — only True means real boundary.
                if token.is_sent_start is not True:
                    false_break = True
                break

        if false_break:
            # Merge: combine text, take timing from both
            fixed[i] = SubtitleSegment(
                text=combined,
                start=fixed[i].start,
                end=fixed[i + 1].end,
                speaker=fixed[i].speaker,
            )
            fixed.pop(i + 1)
            merged_count += 1
            # Don't advance — check if next segment also merges
        else:
            i += 1

    if merged_count:
        from pgw.utils.console import debug

        debug(f"Merged {merged_count} false sentence break(s) (abbreviations).")

    return fixed


def postprocess_segments(
    segments: list[SubtitleSegment],
    language: str,
) -> list[SubtitleSegment]:
    """Apply standard postprocessing: fix overlaps, abbreviations, then danglers."""
    segments = fix_overlapping_timestamps(segments)
    segments = fix_false_sentence_breaks(segments, language)
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

        # Check if the last token is a dangling function word or relative pronoun
        last_token = doc[-1]
        if not _is_dangling(last_token):
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

    Detects three patterns:
    1. Text ending with an apostrophe (Romance clitics: l', d', qu', etc.)
       — uses simple text check, handles spaCy splitting "l'" into ["l", "'"]
    2. Trailing function word POS tags (DET, ADP, CCONJ, SCONJ, AUX) — uses spaCy
    3. Trailing relative pronouns (qui, où, dont, etc. — PronType=Rel) — uses spaCy
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

        # Pattern 2: trailing function word or relative pronoun via spaCy
        doc = nlp(text)
        if not doc or len(doc) <= 1:
            continue

        last_token = doc[-1]
        if not _is_dangling(last_token):
            continue

        # Move the dangling token text to the next segment
        fixed[i].text = text[: last_token.idx].rstrip()
        fixed[i + 1].text = last_token.text.strip() + " " + fixed[i + 1].text.lstrip()

    return [seg for seg in fixed if seg.text.strip()]
