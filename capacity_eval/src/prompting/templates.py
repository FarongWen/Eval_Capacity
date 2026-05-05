from __future__ import annotations


TEXT_ONLY_TEMPLATE = """You are answering a multiple-choice question. Choose the single best answer.
Question:
{question}

Options:
{options}

Return only the option letter."""

MULTIMODAL_TEMPLATE = """You are answering a multiple-choice question based on the image and text.
Choose the single best answer.
Question:
{question}

Options:
{options}

Return only the option letter."""


def format_options(choices: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(choices.items()))


def build_prompt(question: str, choices: dict[str, str], has_image: bool = False) -> str:
    opts = format_options(choices)
    template = MULTIMODAL_TEMPLATE if has_image else TEXT_ONLY_TEMPLATE
    return template.format(question=question, options=opts)
