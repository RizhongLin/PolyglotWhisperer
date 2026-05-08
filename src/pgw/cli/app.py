"""PolyglotWhisperer CLI entry point."""

from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

from pgw import __version__
from pgw.cli.admin import admin_app
from pgw.cli.clean import clean
from pgw.cli.export import export
from pgw.cli.languages import languages
from pgw.cli.maintenance import maintenance_app
from pgw.cli.play import play
from pgw.cli.run import run
from pgw.cli.serve import serve
from pgw.cli.transcribe import transcribe
from pgw.cli.translate import translate
from pgw.cli.vocab import vocab
from pgw.cli.worker import worker_app

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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug output."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-error output."),
    ] = False,
) -> None:
    """PolyglotWhisperer — Video transcription & translation for language learners."""
    load_dotenv(override=False)

    from pgw.utils.logging import set_quiet, set_verbose, setup_logging

    setup_logging()
    if verbose:
        set_verbose(True)
    if quiet:
        set_quiet(True)


app.command("run")(run)
app.command("transcribe")(transcribe)
app.command("translate")(translate)
app.command("play")(play)
app.command("serve")(serve)
app.command("languages")(languages)
app.command("vocab")(vocab)
app.command("clean")(clean)
app.command("export")(export)
app.add_typer(worker_app)
app.add_typer(admin_app)
app.add_typer(maintenance_app)
