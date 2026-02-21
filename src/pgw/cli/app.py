"""PolyglotWhisperer CLI entry point."""

from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

from pgw import __version__
from pgw.cli.languages import languages
from pgw.cli.play import play
from pgw.cli.run import run
from pgw.cli.serve import serve
from pgw.cli.transcribe import transcribe
from pgw.cli.translate import translate
from pgw.cli.vocab import vocab

app = typer.Typer(
    name="pgw",
    help="PolyglotWhisperer — Video transcription & translation for language learners.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pgw {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """PolyglotWhisperer — Video transcription & translation for language learners."""
    # Load .env file for API keys (GROQ_API_KEY, OPENAI_API_KEY, etc.)
    # Does not override existing env vars — shell exports take precedence
    load_dotenv(override=False)


app.command("run")(run)
app.command("transcribe")(transcribe)
app.command("translate")(translate)
app.command("play")(play)
app.command("serve")(serve)
app.command("languages")(languages)
app.command("vocab")(vocab)
