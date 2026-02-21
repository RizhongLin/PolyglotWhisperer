"""Shared CLI utilities."""

from __future__ import annotations

from pathlib import Path


def expand_inputs(inputs: list[str]) -> list[str]:
    """Expand glob patterns and URL list files into individual paths/URLs."""
    expanded = []
    for inp in inputs:
        # URL — pass through
        if inp.startswith(("http://", "https://", "ftp://")):
            expanded.append(inp)
            continue

        path = Path(inp)

        # .txt file — read as URL/path list (one per line)
        if path.suffix == ".txt" and path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    expanded.append(line)
            continue

        # Try as glob pattern if it contains wildcards
        if any(c in inp for c in "*?["):
            matches = sorted(Path(".").glob(inp))
            if matches:
                expanded.extend(str(m) for m in matches)
                continue

        # Regular file/path
        expanded.append(inp)

    return expanded
