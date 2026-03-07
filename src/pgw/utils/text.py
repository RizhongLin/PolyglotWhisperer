"""Shared text processing constants.

Centralised definitions for punctuation sets, apostrophes, timing
thresholds, and formatting constants used across multiple modules.
"""

# --- Unit conversion ---

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

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

# --- Segment length limits ---

# Maximum characters per subtitle segment (split point)
MAX_SEGMENT_CHARS = 72

# Extra character allowance when merging short trailing fragments back.
# Keeps the initial split tight for readability while preventing
# dangling fragments at segment boundaries.
MERGE_CHAR_SLACK = 15

# Maximum words in a trailing fragment eligible for merge-back
MAX_MERGE_TRAIL_WORDS = 4

# Maximum words in a leading fragment eligible for merge-forward
MAX_MERGE_LEAD_WORDS = 2

# Maximum combined words when merging a leading fragment into the next group
MAX_LEAD_MERGE_COMBINED = 10

# Minimum words in a segment before splitting at clause punctuation
MIN_WORDS_CLAUSE_SPLIT = 4

# Maximum words in a gap-based merge (merge_by_gap)
MAX_WORDS_GAP_MERGE = 3
