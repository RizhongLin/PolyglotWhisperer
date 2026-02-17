"""Whisper-supported language definitions.

Language codes and names sourced from OpenAI Whisper (tokenizer.py).
stable-ts provides word-level alignment natively for all languages.

Reference: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
"""

from __future__ import annotations

# fmt: off
WHISPER_LANGUAGES: dict[str, str] = {
    "af": "afrikaans",   "am": "amharic",        "ar": "arabic",
    "as": "assamese",    "az": "azerbaijani",     "ba": "bashkir",
    "be": "belarusian",  "bg": "bulgarian",       "bn": "bengali",
    "bo": "tibetan",     "br": "breton",          "bs": "bosnian",
    "ca": "catalan",     "cs": "czech",           "cy": "welsh",
    "da": "danish",      "de": "german",          "el": "greek",
    "en": "english",     "es": "spanish",         "et": "estonian",
    "eu": "basque",      "fa": "persian",         "fi": "finnish",
    "fo": "faroese",     "fr": "french",          "gl": "galician",
    "gu": "gujarati",    "ha": "hausa",           "haw": "hawaiian",
    "he": "hebrew",      "hi": "hindi",           "hr": "croatian",
    "ht": "haitian creole", "hu": "hungarian",    "hy": "armenian",
    "id": "indonesian",  "is": "icelandic",       "it": "italian",
    "ja": "japanese",    "jw": "javanese",        "ka": "georgian",
    "kk": "kazakh",      "km": "khmer",           "kn": "kannada",
    "ko": "korean",      "la": "latin",           "lb": "luxembourgish",
    "ln": "lingala",     "lo": "lao",             "lt": "lithuanian",
    "lv": "latvian",     "mg": "malagasy",        "mi": "maori",
    "mk": "macedonian",  "ml": "malayalam",       "mn": "mongolian",
    "mr": "marathi",     "ms": "malay",           "mt": "maltese",
    "my": "myanmar",     "ne": "nepali",          "nl": "dutch",
    "nn": "nynorsk",     "no": "norwegian",       "oc": "occitan",
    "pa": "punjabi",     "pl": "polish",          "ps": "pashto",
    "pt": "portuguese",  "ro": "romanian",        "ru": "russian",
    "sa": "sanskrit",    "sd": "sindhi",          "si": "sinhala",
    "sk": "slovak",      "sl": "slovenian",       "sn": "shona",
    "so": "somali",      "sq": "albanian",        "sr": "serbian",
    "su": "sundanese",   "sv": "swedish",         "sw": "swahili",
    "ta": "tamil",       "te": "telugu",          "tg": "tajik",
    "th": "thai",        "tk": "turkmen",         "tl": "tagalog",
    "tr": "turkish",     "tt": "tatar",           "uk": "ukrainian",
    "ur": "urdu",        "uz": "uzbek",           "vi": "vietnamese",
    "yi": "yiddish",     "yo": "yoruba",          "yue": "cantonese",
    "zh": "chinese",
}
# fmt: on

# Languages historically known to have strong word-level alignment support.
# stable-ts provides native alignment for all languages, but these have
# been most extensively tested.
ALIGNMENT_LANGUAGES: set[str] = {
    # torchaudio pipeline models
    "en",
    "fr",
    "de",
    "es",
    "it",
    # HuggingFace wav2vec2 models
    "ja",
    "zh",
    "nl",
    "uk",
    "pt",
    "ar",
    "cs",
    "ru",
    "pl",
    "hu",
    "fi",
    "fa",
    "el",
    "tr",
    "da",
    "he",
    "vi",
    "ko",
    "ur",
    "te",
    "hi",
    "ca",
    "ml",
    "no",
    "nn",
    "sk",
    "sl",
    "hr",
    "ro",
    "eu",
    "gl",
    "ka",
    "lv",
    "tl",
    "sv",
}


def is_valid_language(code: str) -> bool:
    """Check if a language code is supported by Whisper."""
    return code in WHISPER_LANGUAGES


def language_name(code: str) -> str:
    """Get the full language name for a code, or the code itself if unknown."""
    return WHISPER_LANGUAGES.get(code, code)


def validate_language(code: str) -> str:
    """Validate a language code and return it, raising ValueError if invalid."""
    if code not in WHISPER_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{code}'. "
            f"Run 'pgw languages' to see all {len(WHISPER_LANGUAGES)} supported languages."
        )
    return code
