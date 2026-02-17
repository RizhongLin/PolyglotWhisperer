"""PolyglotWhisperer CLI entry point."""

import typer

from pgw.cli.transcribe import transcribe

app = typer.Typer(
    name="pgw",
    help="PolyglotWhisperer — Video transcription & translation for language learners.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """PolyglotWhisperer — Video transcription & translation for language learners."""


app.command("transcribe")(transcribe)
