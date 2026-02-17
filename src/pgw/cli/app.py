"""PolyglotWhisperer CLI entry point."""

import typer

from pgw.cli.languages import languages
from pgw.cli.play import play
from pgw.cli.run import run
from pgw.cli.transcribe import transcribe
from pgw.cli.translate import translate

app = typer.Typer(
    name="pgw",
    help="PolyglotWhisperer — Video transcription & translation for language learners.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """PolyglotWhisperer — Video transcription & translation for language learners."""


app.command("run")(run)
app.command("transcribe")(transcribe)
app.command("translate")(translate)
app.command("play")(play)
app.command("languages")(languages)
