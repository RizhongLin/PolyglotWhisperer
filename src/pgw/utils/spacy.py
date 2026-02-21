"""Shared spaCy model loading with auto-download support.

Provides a unified loader for spaCy models in two configurations:
- POS-only (disable lemmatizer) — used by postprocess.py for segmentation
- POS + lemmatizer — used by vocab summary for word analysis

Models are cached per (language, configuration) to avoid redundant loading.
"""

from __future__ import annotations

from pgw.utils.console import console

# Mapping from pgw language codes to spaCy model names.
# Languages without a model here gracefully skip NLP features.
SPACY_MODELS: dict[str, str] = {
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

# Separate caches for different configurations
_cache_pos_only: dict[str, object] = {}
_cache_with_lemma: dict[str, object] = {}


def load_spacy_model(language: str, enable_lemmatizer: bool = False):
    """Load a spaCy model, auto-downloading if needed.

    Args:
        language: ISO 639-1 language code (e.g. "fr", "en").
        enable_lemmatizer: If True, keeps the lemmatizer enabled (for vocab
            analysis). If False, disables it for faster POS-only processing.

    Returns:
        The loaded spaCy Language model, or None if spaCy is not installed
        or the language has no model available.
    """
    cache = _cache_with_lemma if enable_lemmatizer else _cache_pos_only

    if language in cache:
        return cache[language]

    try:
        import spacy
    except ImportError:
        cache[language] = None
        return None

    model_name = SPACY_MODELS.get(language)
    if model_name is None:
        cache[language] = None
        return None

    disable = ["parser", "ner"] if enable_lemmatizer else ["parser", "lemmatizer", "ner"]

    try:
        nlp = spacy.load(model_name, disable=disable)
    except OSError:
        # Model not installed — auto-download
        console.print(f"[bold]Downloading spaCy model:[/bold] {model_name}")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name, disable=disable)
        except (SystemExit, Exception):
            console.print(f"[yellow]Could not load spaCy model {model_name}, skipping.[/yellow]")
            cache[language] = None
            return None

    cache[language] = nlp
    return nlp
