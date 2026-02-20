"""Tests for LLM prompt parsing edge cases."""

from pgw.llm.prompts import format_history_context, parse_numbered_response


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


def test_parse_no_numbering_fallback():
    response = "Just plain text\nAnother line"
    result, exact = parse_numbered_response(response, 2)
    assert result == ["Just plain text", "Another line"]
    assert exact is True


def test_format_history_context_empty():
    assert format_history_context([], []) == ""


def test_format_history_context_pairs():
    result = format_history_context(["Bonjour"], ["Hello"])
    assert "Bonjour" in result
    assert "Hello" in result
    assert "Previous translations" in result
