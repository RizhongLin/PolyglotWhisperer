"""Shared CLI utilities."""

from __future__ import annotations

from pathlib import Path

from rich.table import Table

from pgw.utils.console import console


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


def build_config_overrides(
    language: str,
    device: str,
    whisper_model: str | None = None,
    llm_model: str | None = None,
    llm_backend: str | None = None,
    backend: str | None = None,
    translate: str | None = None,
    subs: bool = False,
) -> dict[str, object]:
    """Build config override dict from CLI flags."""
    overrides: dict[str, object] = {
        "whisper.language": language,
        "whisper.device": device,
    }
    if whisper_model is not None:
        model_key = "whisper.api_model" if backend == "api" else "whisper.local_model"
        overrides[model_key] = whisper_model
    if llm_model is not None:
        model_key = "llm.api_model" if llm_backend == "api" else "llm.local_model"
        overrides[model_key] = llm_model
    if llm_backend is not None:
        overrides["llm.backend"] = llm_backend
    if backend is not None:
        overrides["whisper.backend"] = backend
    if translate is not None:
        overrides["llm.target_language"] = translate
    if subs:
        overrides["download.subtitles"] = True
    return overrides


def print_batch_summary(
    results: list[tuple[str, str, str]],
    total: int,
    show_output: bool = False,
) -> None:
    """Print a Rich batch results summary table.

    Args:
        results: List of (input, status, detail) tuples.
        total: Total number of inputs processed.
        show_output: If True, adds a fourth "Output" column with the detail field.
    """
    table = Table(title=f"Batch Results ({total} files)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Input", max_width=50, no_wrap=True)
    table.add_column("Status")
    if show_output:
        table.add_column("Output", max_width=50, no_wrap=True)

    succeeded = 0
    for i, (inp, status, detail) in enumerate(results, 1):
        style = "green" if status == "success" else "red"
        row = [str(i), inp, f"[{style}]{status}[/{style}]"]
        if show_output:
            row.append(detail)
        table.add_row(*row)
        if status == "success":
            succeeded += 1

    console.print(table)
    console.print(f"\n[bold]{succeeded}/{total} succeeded[/bold]")
