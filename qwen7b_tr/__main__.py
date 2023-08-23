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
# pylint: disable=invalid-name, too-many-arguments, redefined-builtin, unused-argument, too-many-locals, too-many-branches
# import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import pyperclip
import typer
from gradio_client import Client
from loguru import logger
from rich import print

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

# client = Client("https://mikeee-qwen-7b-chat.hf.space/")

param_def = dict(
    zip(
        [
            "max_new_tokens",
            "temperature",
            "repetition_penalty",
            "top_k",
            "top_p",
            "system_prompt",
        ],
        [256, 0.81, 1.1, 0, 0.9, "You are a helpful assistang"],
    )
)
api_url = "https://mikeee-qwen-7b-chat.hf.space/"


def qwen7b_tr(
    text: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    top_k: Optional[float] = None,
    top_p: Optional[float] = None,
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
        qwen7b_tr.client
    except Exception:
        qwen7b_tr.client = Client(api_url)
    client = qwen7b_tr.client

    # make a copy of locals()
    locals_ = locals()
    # assign default value for Nones
    for elm in param_def:
        if locals_[elm] is None:
            locals_[f"{elm}"] = param_def.get(elm)
    logger.trace(f"{locals_=}")

    # params to be used
    # _ = [locals_[f"{elm}"] for elm in param_def]
    # logger.trace(f"params: {_}")

    # use the param_  below e.g.
    # max_new_tokens_, temperature_ ...

    # handle top_k
    try:
        repetition_penalty_ = float(locals_["repetition_penalty"])
    except Exception:
        repetition_penalty_ = 1.1
    try:
        top_k_ = int(locals_["top_k"])
    except Exception:
        top_k_ = 0.0

    try:
        res = client.predict(
            text,
            locals_["max_new_tokens"],
            locals_["temperature"],
            repetition_penalty_,
            top_k_,
            locals_["top_p"],
            locals_["system_prompt"],
            None,  # bot_history
            api_name="/api",
        )
    except Exception as exc:
        logger.error(exc)
        res = str(exc)

    return res


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(
            f"{app.info.name} v.{__version__} -- translate/chat via qwen-7b huggingface api"
        )
        raise typer.Exit()


NUMB = 3
TO_LANG = "中文"
USER_PROMPT_TEMPL = "翻成{to_lang}。列出{numb}个版本。"


@app.command()
def main(
    question: Optional[List[str]] = typer.Argument(
        None,
        help="Source text or question.",
        show_default=False,
    ),
    clipb: Optional[bool] = typer.Option(
        None,
        "--clipb",
        "-c",
        help="Use clipboard content if set or if `question` is empty.",
    ),
    to_lang: str = typer.Option(
        None,
        "--to-lang",
        "-t",
        help=f"Target language when using the default prompt. [default: {TO_LANG}]",
        show_default=False,
    ),
    numb: int = typer.Option(
        None,
        "--numb",
        "-n",
        help=f"number of translation variants when using the default prompt. [default {NUMB}]",
        show_default=False,
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        "--max-new-tokens",
        "-m",
        help=f"Max new tokens. [default: {param_def.get('max_new_tokens')}]",
        show_default=False,
    ),
    temperature: Optional[int] = typer.Option(
        None,
        "--temperature",
        "--temp",
        help=f"Temperature. [default: {param_def.get('temperature')}]",
        show_default=False,
    ),
    repetition_penalty: Optional[float] = typer.Option(
        None,
        "--repetition-penalty",
        "--rep",
        help=f"Repetition penalty. [default: {param_def.get('repetition_penalty')}]",
        show_default=False,
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "--top_k",
        help=f"Top_k. [default: {param_def.get('top_k')}]",
        show_default=False,
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        "--top_p",
        help=f"Top_p. [default: {param_def.get('top_p')}]",
        show_default=False,
    ),
    user_prompt: Optional[str] = typer.Option(
        None,
        "--user-prompt",
        help=f"User prompt. [default: '翻成{TO_LANG}，列出{NUMB}个版本.']",
        show_default=False,
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
):
    """Translate via qwen-7b-chat huggingface API."""
    logger.trace(f" entry {question=} ")
    if numb is None:
        numb = NUMB
    if to_lang is None:
        to_lang = TO_LANG
    if user_prompt is None:
        user_prompt = USER_PROMPT_TEMPL.format(
            numb=numb,
            to_lang=to_lang,
        )
    logger.trace(f"{user_prompt=}")

    if question is not None:
        text = question[:]
    else:
        text = []

    # if clip is set use it
    if clipb:
        logger.trace(
            "clipb is set to True, translating the content of the clipboard..."
        )
        text_str = pyperclip.paste()
    else:
        if not text:
            # if no text provided, copy from clipboard
            logger.trace(
                "No text provided, translating the content of the clipboard..."
            )
            print(
                "\t[yellow]No text provided[/yellow], [green]translating the content of the clipboard[/green]..."
            )
            text_str = pyperclip.paste()
        else:
            text_str = " ".join(text).strip()
        if not text_str:
            # somehow still no text collected
            logger.trace(
                "\tNo text provided, translating the content of the clipboard..."
            )
            print(
                "\t[yellow]No text provided[/yellow], [green]translating the content of the clipboard[/green]..."
            )
            text_str = pyperclip.paste()
    try:
        text_str = text_str.strip()
    except Exception as exc:
        logger.error(exc)
        text_str = ""
    if not text_str:
        print("[yellow]Nothing to do...[/yellow]")
        raise typer.Exit(1)

    logger.trace(f"text_str: {text_str}")

    text = dedent(
        f"""
        {user_prompt}
        {text_str}
        """
    ).strip()
    logger.trace(f"{text=}")

    typer.secho("\tdiggin...", fg=typer.colors.MAGENTA)
    try:
        res = qwen7b_tr(text)
    except Exception as exc:
        logger.error(exc)
        raise typer.Exit()
    print()
    print(text_str)
    print()
    print(res)


if __name__ == "__main__":
    app()
