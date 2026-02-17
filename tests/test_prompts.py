"""Tests for LLM prompt parsing edge cases."""

from pgw.llm.prompts import parse_numbered_response


def test_parse_extra_lines_truncated():
    response = "1. A\n2. B\n3. C\n4. D"
    result = parse_numbered_response(response, 2)
    assert result == ["A", "B"]


def test_parse_fewer_lines_padded():
    response = "1. Only one"
    result = parse_numbered_response(response, 3)
    assert result == ["Only one", "", ""]


def test_parse_blank_lines_ignored():
    response = "1. Hello\n\n2. World\n\n"
    result = parse_numbered_response(response, 2)
    assert result == ["Hello", "World"]


def test_parse_no_numbering_fallback():
    response = "Just plain text\nAnother line"
    result = parse_numbered_response(response, 2)
    assert result == ["Just plain text", "Another line"]
