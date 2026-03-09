"""Shared spaCy model loading with auto-download support.

Provides a unified loader for spaCy models in two configurations:
- POS-only (disable lemmatizer) — used by postprocess.py for segmentation
- POS + lemmatizer — used by vocab summary for word analysis

Models are cached per (language, configuration) to avoid redundant loading.
"""

from __future__ import annotations

import subprocess
import sys

from pgw.utils.console import stage, warning

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
    "nn": "nb_core_news_sm",  # Nynorsk → Bokmål model
    "nl": "nl_core_news_sm",
    "no": "nb_core_news_sm",  # Norwegian → Bokmål model
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "zh": "zh_core_web_sm",
}

# Larger models for vocabulary analysis (better lemmatization + POS accuracy).
# Used when enable_lemmatizer=True; falls back to SPACY_MODELS if unavailable.
SPACY_VOCAB_MODELS: dict[str, str] = {
    "ca": "ca_core_news_md",
    "da": "da_core_news_md",
    "de": "de_core_news_md",
    "el": "el_core_news_md",
    "en": "en_core_web_md",
    "es": "es_core_news_md",
    "fi": "fi_core_news_md",
    "fr": "fr_core_news_md",
    "hr": "hr_core_news_md",
    "it": "it_core_news_md",
    "ja": "ja_core_news_md",
    "ko": "ko_core_news_md",
    "lt": "lt_core_news_md",
    "mk": "mk_core_news_md",
    "nb": "nb_core_news_md",
    "nl": "nl_core_news_md",
    "pl": "pl_core_news_md",
    "pt": "pt_core_news_md",
    "ro": "ro_core_news_md",
    "ru": "ru_core_news_md",
    "sl": "sl_core_news_md",
    "sv": "sv_core_news_md",
    "uk": "uk_core_news_md",
    "zh": "zh_core_web_md",
}

# Separate caches for different configurations
_cache_pos_only: dict[str, object] = {}
_cache_with_lemma: dict[str, object] = {}
_cache_with_parser: dict[str, object] = {}


def _install_spacy_model(model_name: str) -> None:
    """Install a spaCy model via pip in the current environment."""
    cmd = [sys.executable, "-m", "pip", "install", model_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def load_spacy_model(
    language: str,
    enable_lemmatizer: bool = False,
    enable_parser: bool = False,
):
    """Load a spaCy model, auto-downloading if needed.

    Args:
        language: ISO 639-1 language code (e.g. "fr", "en").
        enable_lemmatizer: If True, keeps the lemmatizer enabled (for vocab
            analysis). If False, disables it for faster POS-only processing.
        enable_parser: If True, keeps the parser enabled (for sentence
            boundary detection). If False, disables it for faster processing.

    Returns:
        The loaded spaCy Language model, or None if spaCy is not installed
        or the language has no model available.
    """
    if enable_parser:
        cache = _cache_with_parser
    elif enable_lemmatizer:
        cache = _cache_with_lemma
    else:
        cache = _cache_pos_only

    if language in cache:
        return cache[language]

    try:
        import spacy
    except ImportError:
        cache[language] = None
        return None

    # Use larger _md models for vocab analysis (better lemmatization)
    if enable_lemmatizer:
        model_name = SPACY_VOCAB_MODELS.get(language) or SPACY_MODELS.get(language)
    else:
        model_name = SPACY_MODELS.get(language)
    if model_name is None:
        cache[language] = None
        return None

    disable = ["ner"]
    if not enable_parser:
        disable.append("parser")
    if not enable_lemmatizer:
        disable.append("lemmatizer")

    try:
        nlp = spacy.load(model_name, disable=disable)
    except OSError:
        # Model not installed — auto-download
        stage("Downloading spaCy model", model_name)
        try:
            _install_spacy_model(model_name)
            nlp = spacy.load(model_name, disable=disable)
        except (SystemExit, Exception):
            # _md model may not exist; fall back to _sm
            fallback = SPACY_MODELS.get(language)
            if fallback and fallback != model_name:
                try:
                    nlp = spacy.load(fallback, disable=disable)
                except OSError:
                    stage("Downloading spaCy model", fallback)
                    try:
                        _install_spacy_model(fallback)
                        nlp = spacy.load(fallback, disable=disable)
                    except (SystemExit, Exception):
                        warning(f"Could not load spaCy model {fallback}, skipping.")
                        cache[language] = None
                        return None
            else:
                warning(f"Could not load spaCy model {model_name}, skipping.")
                cache[language] = None
                return None

    cache[language] = nlp
    return nlp
