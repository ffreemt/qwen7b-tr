r"""
Define qwen7b_tr.

%%time
question = "Translate the following text into German. List 10 variants:\n Life sucks, then you die. "

result = client.predict(
    question,
    256,
    0.81,
    1.1,
    0,
    0.95,
    "You are a helpful assistant. ",
    None,
    api_name="/api"
)
print(result)
"""
# pylint: disable=invalid-name, too-many-arguments,
# import sys
from pathlib import Path
from typing import List, Optional

import pyperclip
import typer
from graio_client import Client
from loguru import logger

from qwen7b_tr import __version__

# del sys  # set/export LOGURU_LEVEL=TRACE
# logger.remove()
# logger.add(sys.stderr, level="TRACE")

del Path

app = typer.Typer(
    name="qwen7b-tr",
    add_completion=False,
    help="Translate via qwen7b-chat huggingface api",
)

client = Client("https://mikeee-qwen-7b-chat.hf.space/")


def qwen7b_tr(
    text: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.81,
    repetition_penalty: float = 1.1,
    top_k: float =0.,
    top_p: float = 0.9,
    system_prompt: str = "You are a helpful assistant",
) -> str:
    """
    Fetch result from api.

    Args:
    ----
    text: user prompt or question
    max_new_tokens: 256 (numeric value between 1 and 2048)
    temperature: 0.81
    repetition_penalty: 1.1
    top_k: 0
    top_p: 0.9
    system_prompt: "You are a helpful assistant"

    Returns:
    -------
    response
    """
    try:
        res = client.predict(
            text,
            max_new_tokens,
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            system_prompt,
            None,  # bot_history
            api_name="/api",
        )
    except Exception as exc:
        logger(exc)
        res = str(exc)

    return res

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{app.info.name} v.{__version__} -- ...")
        raise typer.Exit()


@app.command()
def main(
    question: Optional[List[str]] = typer.Argument(
        None,
        help="Source text or question.",
    ),
    clipb: Optional[bool] = typer.Option(
        None,
        "--clipb",
        "-c",
        help="Use clipboard content if set or if `text` is empty.",
    ),
    to_lang: str = typer.Option(
        "中文", "--to-lang", "-t", help="Target language when using the default prompt."
    ),
    numb: int = typer.Option(
        3,
        "--numb",
        "-n",
        help="number of translation variants when using the default prompt.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        "-p",
        help="User defined system prompt. [default: 'You are a helpful assistant.']",
        show_default=False,
    ),
    version: Optional[bool] = typer.Option(  # pylint: disable=(unused-argument
        None,
        "--version",
        "-v",
        "-V",
        help="Show version info and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> str:
    """
    Define qwen7b_tr.

    Args:
    ----
    question: source test or a question
    clipb: if True, copy from the clipboard
    to_lang: target language
    numb: the number translaton variants when using the default user prompt.
    system_prompt: str
    version: info

    Returns:
    -------
    translated text
    """
    logger.trace(f" entry {question=} ")
    text = question[:]

    # if clip is set use it
    if clipb:
        text_str = pyperclip.paste()
    else:
        if text is None:
            # if no text provided, copy from clipboard
            text_str = pyperclip.paste()
        else:
            text_str = " ".join(text).strip()
        if not text_str:
            text_str = pyperclip.paste()
    try:
        text_str = text_str.strip()
    except Exception as exc:
        logger.error(exc)
        text_str = ""
    if not text_str:
        raise typer.Exit(1)

    logger.trace(f"text_str: {text_str}")


if __name__ == "__main__":
    app()
