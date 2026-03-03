"""Tests for LLM prompt parsing edge cases."""

from pgw.llm.prompts import (
    UNTRANSLATED_MARKER,
    format_bilingual_context,
    format_history_context,
    parse_json_response,
    parse_numbered_response,
)


def test_parse_extra_lines_truncated():
    response = "1. A\n2. B\n3. C\n4. D"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["A", "B"]
    assert exact is False


def test_parse_fewer_lines_padded():
    response = "1. Only one"
    result, exact = parse_numbered_response(response, 3)
    assert result == ["Only one", "", ""]
    assert exact is False


def test_parse_exact_match():
    response = "1. Hello\n2. World"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_parse_blank_lines_ignored():
    response = "1. Hello\n\n2. World\n\n"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_parse_no_numbering_ignored():
    """Non-numbered lines are ignored to avoid counting context/explanations."""
    response = "Just plain text\nAnother line"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["", ""]
    assert exact is False


def test_parse_mixed_numbered_and_plain():
    """Only numbered lines are extracted, plain text is ignored."""
    response = "Here are the translations:\n1. Hello\n2. World\nHope this helps!"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_format_history_context_empty():
    assert format_history_context([], []) == ""


def test_format_history_context_pairs():
    result = format_history_context(["Bonjour"], ["Hello"])
    assert "Bonjour" in result
    assert "Hello" in result
    assert "Previous translations" in result


# --- JSON response parser tests ---


def test_parse_json_response_translations_object():
    response = '{"translations": ["Hello", "World"]}'
    result, exact = parse_json_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_parse_json_response_with_code_fences():
    response = '```json\n{"translations": ["Hello", "World"]}\n```'
    result, exact = parse_json_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_parse_json_response_plain_array_rejected():
    """Plain arrays are not accepted — must be wrapped in an object."""
    response = '["Hello", "World"]'
    result, exact = parse_json_response(response, 2)
    assert result == []
    assert exact is False


def test_parse_json_response_invalid_json():
    response = "Not JSON at all"
    result, exact = parse_json_response(response, 2)
    assert result == []
    assert exact is False


def test_parse_json_response_not_array_or_translations():
    response = '{"key": "value"}'
    result, exact = parse_json_response(response, 2)
    assert result == []
    assert exact is False


def test_parse_json_response_count_mismatch():
    response = '{"translations": ["A", "B", "C"]}'
    result, exact = parse_json_response(response, 2)
    assert result == ["A", "B"]
    assert exact is False


def test_parse_json_response_keyed_format():
    response = '{"1": "Hello", "2": "World"}'
    result, exact = parse_json_response(response, 2)
    assert result == ["Hello", "World"]
    assert exact is True


def test_parse_json_response_keyed_format_mismatch():
    response = '{"1": "A", "2": "B", "3": "C"}'
    result, exact = parse_json_response(response, 2)
    assert result == ["A", "B"]
    assert exact is False


# --- Bilingual context formatter tests ---


def test_format_bilingual_context_empty():
    assert format_bilingual_context([], []) == ""


def test_format_bilingual_context_pairs():
    result = format_bilingual_context(["Bonjour", "Merci"], ["Hello", "Thanks"])
    assert "preceding:" in result
    assert '"Bonjour": "Hello"' in result
    assert '"Merci": "Thanks"' in result


# --- Untranslated marker constant ---


def test_untranslated_marker_constant():
    assert UNTRANSLATED_MARKER.startswith("[?]")
