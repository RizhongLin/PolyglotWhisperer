"""PolyglotWhisperer CLI entry point."""

import typer
from dotenv import load_dotenv

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


@app.callback()
def main() -> None:
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
