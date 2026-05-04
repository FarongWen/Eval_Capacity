from __future__ import annotations
import re
from typing import Optional


def parse_mcq_answer(raw_text: str, valid_options: set[str] | None = None) -> Optional[str]:
    """Robustly parse an MCQ answer letter from model output.

    Handles:
    - "A"
    - "Answer: A"
    - "The answer is (A)"
    - "I choose A."
    - Chinese colons "："
    - Mixed formatting with explanation
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # Direct single letter
    if len(text) == 1 and text.isalpha():
        letter = text.upper()
        if valid_options is None or letter in valid_options:
            return letter

    # Pattern: "The answer is (A)" / "The answer is A" / "答案是(A)" / "答案：A"
    patterns = [
        r"(?:the answer is|the correct answer is|答案[是为：:]*)\s*[\(（]?\s*([A-Ha-h])\s*[\)）]?",
        r"(?:I choose|I select|我选)\s*[\(（]?\s*([A-Ha-h])\s*[\)）]?",
        r"(?:Answer|答案|选项)\s*[:：]\s*[\(（]?\s*([A-Ha-h])\s*[\)）]?",
        r"[\(（]\s*([A-Ha-h])\s*[\)）]",
        r"(?:choose|select|选择)\s*[\(（]?\s*([A-Ha-h])\s*[\)）]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if valid_options is None or letter in valid_options:
                return letter

    # Fallback: find first standalone option letter
    # Match a letter that is either at start, after whitespace/punctuation, or before punctuation
    standalone = re.findall(r'(?:^|[\s,.\-:：，。、])\s*([A-Ha-h])\s*(?:[.,，。\)）\s]|$)', text)
    for letter in standalone:
        letter = letter.upper()
        if valid_options is None or letter in valid_options:
            return letter

    # Last resort: find any capital letter A-H that could be an option
    all_caps = re.findall(r'\b([A-H])\b', text)
    if all_caps:
        for letter in all_caps:
            if valid_options is None or letter in valid_options:
                return letter

    return None
