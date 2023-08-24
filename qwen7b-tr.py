"""Prep for nuitka build."""
# ruff: noqa: F401
# import gradio_client  # nuitka
# import loguru
# import pyperclip
# import rich
# import typer

from qwen7b_tr.__main__ import app

if __name__ == "__main__":
    app()
