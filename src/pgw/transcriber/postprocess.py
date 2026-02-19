"""Post-transcription subtitle segmentation and linguistic fixes.

Rebuilds raw Whisper segments into subtitle-friendly chunks using
word-level timestamps, then applies spaCy POS tagging to fix
dangling function words at segment boundaries.
"""

from __future__ import annotations

from pgw.utils.console import console

# POS tags that should not dangle at the end of a subtitle segment.
# DET = determiners (le/la/les/the/der/die/das/el/la/los...)
# ADP = adpositions/prepositions (de/en/à/of/in/to/von/mit...)
_DANGLING_POS = {"DET", "ADP"}

# Mapping from pgw language codes to spaCy model names.
# Languages without a model here gracefully skip the function-word fix.
_SPACY_MODELS: dict[str, str] = {
    "ca": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "hr": "hr_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "nl": "nl_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "zh": "zh_core_web_sm",
}

_spacy_cache: dict[str, object] = {}


def _load_spacy_model(language: str):
    """Load a spaCy model for POS tagging, auto-downloading if needed.

    Returns the loaded model, or None if spaCy is not installed or the
    language has no model available.  Results are cached per language.
    """
    if language in _spacy_cache:
        return _spacy_cache[language]

    try:
        import spacy
    except ImportError:
        _spacy_cache[language] = None
        return None

    model_name = _SPACY_MODELS.get(language)
    if model_name is None:
        _spacy_cache[language] = None
        return None

    try:
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "ner"])
    except OSError:
        # Model not installed — auto-download
        console.print(f"[bold]Downloading spaCy model:[/bold] {model_name}")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "ner"])
        except (SystemExit, Exception):
            console.print(f"[yellow]Could not load spaCy model {model_name}, skipping.[/yellow]")
            _spacy_cache[language] = None
            return None

    _spacy_cache[language] = nlp
    return nlp


def regroup_for_subtitles(result, max_chars: int = 50) -> None:
    """Rebuild segments from word-level timestamps for subtitle display.

    Merges all segments, then re-splits using punctuation, speech gaps,
    and length constraints.  This produces subtitle-friendly segments
    with correct word-level timestamps.
    """
    result.ignore_special_periods()
    result.clamp_max()
    result.merge_all_segments()
    result.split_by_punctuation([(".", " "), "?", "!", "。", "？", "！"])
    result.split_by_gap(0.5)
    result.split_by_punctuation([(",", " "), ";", "，", "；"], min_words=4)
    result.split_by_length(max_chars=max_chars)
    result.split_by_duration(max_dur=8.0)
    result.merge_by_gap(0.15, max_words=3, max_chars=max_chars, is_sum_max=True)
    result.clamp_max()


def fix_dangling_function_words(result, language: str) -> None:
    """Move dangling function words from segment ends to the next segment.

    Uses spaCy POS tagging to detect determiners (DET) and adpositions
    (ADP) at segment boundaries.  Works for any language with a spaCy
    model.  Falls back silently if spaCy is not installed.
    """
    nlp = _load_spacy_model(language)
    if nlp is None:
        return

    segments = result.segments
    for i in range(len(segments) - 1):
        words = segments[i].words
        if len(words) <= 1:
            continue  # Don't empty a segment

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
