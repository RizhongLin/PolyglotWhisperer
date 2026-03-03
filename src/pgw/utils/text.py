"""Shared text processing constants.

Centralised definitions for punctuation sets, apostrophes, and timing
thresholds used across the transcriber and LLM modules.
"""

# --- Punctuation sets ---

# Sentence-ending punctuation (Western + CJK)
SENTENCE_END_CHARS = frozenset(".!?。？！")

# Clause-level punctuation for secondary split points
CLAUSE_PUNCT = frozenset(",;，；")

# Apostrophe characters used in Romance language clitics (l', d', qu', etc.)
APOSTROPHES = frozenset({"'", "\u2019"})

# --- Timing thresholds (seconds) ---

# Split segments at speech pauses longer than this
SPEECH_GAP_THRESHOLD = 0.5

# Merge short fragments with neighbors when gap is below this
MERGE_GAP_THRESHOLD = 0.15

# Natural break between segments for translation chunk boundaries
TIMING_GAP_THRESHOLD = 1.0

# Maximum duration for a single subtitle segment
MAX_SEGMENT_DURATION = 8.0

# Extra character allowance when merging short trailing fragments back.
# Keeps the initial split tight for readability while preventing
# dangling 1-2 word fragments at segment boundaries.
MERGE_CHAR_SLACK = 15
