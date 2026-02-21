"""Tests for parallel text export HTML building."""

import html

from pgw.core.models import SubtitleSegment
from pgw.subtitles.export import _build_parallel_html


class TestBuildParallelHtml:
    def _make_segments(self, texts, start=0.0, duration=2.0):
        return [
            SubtitleSegment(text=t, start=start + i * duration, end=start + (i + 1) * duration)
            for i, t in enumerate(texts)
        ]

    def test_basic_structure(self):
        orig = self._make_segments(["Bonjour"])
        trans = self._make_segments(["Hello"])
        result = _build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "<!DOCTYPE html>" in result
        assert "<table>" in result
        assert "</table>" in result
        assert "FR" in result
        assert "EN" in result

    def test_title_escaped(self):
        orig = self._make_segments(["Bonjour"])
        trans = self._make_segments(["Hello"])
        result = _build_parallel_html(orig, trans, "fr", "en", "Title <script>")
        assert html.escape("Title <script>") in result
        assert "<script>" not in result.split("<style>")[0].split("</style>")[-1].replace(
            html.escape("<script>"), ""
        )

    def test_text_html_escaped(self):
        orig = self._make_segments(["<b>bold</b>"])
        trans = self._make_segments(["<i>italic</i>"])
        result = _build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "&lt;b&gt;bold&lt;/b&gt;" in result
        assert "&lt;i&gt;italic&lt;/i&gt;" in result

    def test_timestamps_as_mmss(self):
        orig = [SubtitleSegment(text="Hi", start=65.0, end=67.0)]
        trans = [SubtitleSegment(text="Salut", start=65.0, end=67.0)]
        result = _build_parallel_html(orig, trans, "en", "fr", "Test")
        assert "01:05" in result

    def test_segment_count_in_meta(self):
        orig = self._make_segments(["A", "B", "C"])
        trans = self._make_segments(["X", "Y", "Z"])
        result = _build_parallel_html(orig, trans, "fr", "en", "Test")
        assert "3 segments" in result

    def test_correct_number_of_rows(self):
        orig = self._make_segments(["A", "B", "C", "D"])
        trans = self._make_segments(["1", "2", "3", "4"])
        result = _build_parallel_html(orig, trans, "fr", "en", "Test")
        # 4 data rows + 1 header row = 5 total <tr>
        assert result.count("<tr>") == 5

    def test_empty_segments(self):
        result = _build_parallel_html([], [], "fr", "en", "Test")
        assert "<tbody>" in result
        assert "0 segments" in result
