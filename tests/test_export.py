"""Tests for parallel text export HTML building."""

import html

from conftest import make_segments

from pgw.core.models import SubtitleSegment
from pgw.subtitles.export import build_parallel_html


class TestBuildParallelHtml:

    def test_basic_structure(self):
        orig = make_segments(["Bonjour"])
        trans = make_segments(["Hello"])
        result = build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "<!DOCTYPE html>" in result
        assert "<table>" in result
        assert "</table>" in result
        assert "French" in result
        assert "English" in result

    def test_title_escaped(self):
        orig = make_segments(["Bonjour"])
        trans = make_segments(["Hello"])
        result = build_parallel_html(orig, trans, "fr", "en", "Title <script>")
        assert html.escape("Title <script>") in result
        assert "<script>" not in result.split("<style>")[0].split("</style>")[-1].replace(
            html.escape("<script>"), ""
        )

    def test_text_html_escaped(self):
        orig = make_segments(["<b>bold</b>"])
        trans = make_segments(["<i>italic</i>"])
        result = build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "&lt;b&gt;bold&lt;/b&gt;" in result
        assert "&lt;i&gt;italic&lt;/i&gt;" in result

    def test_timestamps_as_mmss(self):
        orig = [SubtitleSegment(text="Hi", start=65.0, end=67.0)]
        trans = [SubtitleSegment(text="Salut", start=65.0, end=67.0)]
        result = build_parallel_html(orig, trans, "en", "fr", "Test")
        assert "01:05" in result

    def test_segment_count_in_meta(self):
        orig = make_segments(["A", "B", "C"])
        trans = make_segments(["X", "Y", "Z"])
        result = build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "3 segments" in result

    def test_correct_number_of_rows(self):
        orig = make_segments(["A", "B", "C", "D"])
        trans = make_segments(["1", "2", "3", "4"])
        result = build_parallel_html(orig, trans, "fr", "en", "Test")
        # 4 data rows + 1 header row = 5 total <tr>
        assert result.count("<tr>") == 5

    def test_empty_segments(self):
        result = build_parallel_html([], [], "fr", "en", "Test")
        assert "<tbody>" in result
        assert "0 segments" in result
