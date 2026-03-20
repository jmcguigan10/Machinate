from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

CANCEL_SENTINEL = "__cancel__"
BACK_SENTINEL = "__back__"


@dataclass(frozen=True)
class MenuChoice:
    label: str
    value: str
    styled_title: Any | None = None


class OperationCancelled(Exception):
    pass


def load_questionary() -> tuple[Any | None, Any | None]:
    try:
        import questionary  # type: ignore
        from prompt_toolkit.styles import Style  # type: ignore

        return questionary, Style
    except ModuleNotFoundError:
        return None, None


QUESTIONARY, STYLE_CLASS = load_questionary()
QUESTIONARY_STYLE = (
    STYLE_CLASS.from_dict(
        {
            "managed-venv": "fg:#d70000 bold",
            "library-detail": "fg:#6b7280",
            "library-existing": "fg:#005faf bold",
            "library-add": "fg:#2e8b57 bold",
            "library-edit": "fg:#b58900 bold",
            "library-remove": "fg:#d70000 bold",
            "library-missing": "fg:#ffffff",
            "finish": "fg:#005f87 bold",
            "custom-input": "fg:#875f00 bold",
            "warning": "fg:#d70000 bold",
            "answer": "fg:#005f87 bold",
            "selected": "fg:#2e8b57 bold",
            "question": "bold",
        }
    )
    if STYLE_CLASS is not None
    else None
)


def can_prompt_interactively() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def use_questionary() -> bool:
    return QUESTIONARY is not None and can_prompt_interactively()


def fallback_prompt_text(message: str, default: str | None = None) -> str:
    prompt = message
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    response = input(prompt).strip()
    if response:
        return response
    if default is not None:
        return default
    raise ValueError("No value entered.")


def prompt_text(message: str, default: str | None = None) -> str:
    if use_questionary():
        if default is None:
            response = QUESTIONARY.text(message, style=QUESTIONARY_STYLE).ask()
        else:
            response = QUESTIONARY.text(message, default=default, style=QUESTIONARY_STYLE).ask()
        if response is None or not response.strip():
            raise ValueError("No value entered.")
        return response.strip()
    return fallback_prompt_text(message, default=default)


def prompt_optional_text(message: str, default: str = "") -> str:
    if use_questionary():
        response = QUESTIONARY.text(message, default=default, style=QUESTIONARY_STYLE).ask()
        if response is None:
            raise ValueError("Input cancelled.")
        return response.strip()

    prompt = message
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    return input(prompt).strip()


def prompt_multiline(message: str) -> str:
    print(message)
    print("Finish with a blank line.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_select(
    message: str,
    choices: list[MenuChoice],
    default: str | None = None,
    *,
    use_shortcuts: bool = True,
    use_search_filter: bool = False,
) -> str:
    if not choices:
        raise ValueError("A non-empty choice list is required.")

    if use_questionary():
        questionary_choices = [
            QUESTIONARY.Choice(title=choice.styled_title or choice.label, value=choice.value)
            for choice in choices
        ]
        response = QUESTIONARY.select(
            message,
            choices=questionary_choices,
            default=default,
            use_shortcuts=use_shortcuts,
            use_search_filter=use_search_filter,
            use_jk_keys=not use_search_filter,
            style=QUESTIONARY_STYLE,
        ).ask()
        if response is None:
            raise ValueError("Selection cancelled.")
        return response

    print(message)
    for index, choice in enumerate(choices, start=1):
        print(f"  {index}. {choice.label}")
    raw = fallback_prompt_text("Select an option by number or value", default=default)
    if raw.isdigit():
        choice_index = int(raw) - 1
        if choice_index < 0 or choice_index >= len(choices):
            raise ValueError(f"Selection {raw} is out of range.")
        return choices[choice_index].value
    valid_values = {choice.value for choice in choices}
    if raw not in valid_values:
        raise ValueError(f"Unknown selection `{raw}`.")
    return raw


def prompt_multiselect(message: str, choices: list[str]) -> list[str]:
    if not choices:
        raise ValueError("A non-empty choice list is required.")

    if use_questionary():
        response = QUESTIONARY.checkbox(message, choices=choices, style=QUESTIONARY_STYLE).ask()
        if response is None or not response:
            raise ValueError("At least one selection is required.")
        return [str(item) for item in response]

    print(message)
    for index, choice in enumerate(choices, start=1):
        print(f"  {index}. {choice}")
    response = input("Select one or more by comma-separated numbers or names: ").strip()
    if not response:
        raise ValueError("At least one selection is required.")
    selected: list[str] = []
    for chunk in response.split(","):
        item = chunk.strip()
        if not item:
            continue
        if item.isdigit():
            selected.append(choices[int(item) - 1])
        elif item in choices:
            selected.append(item)
        else:
            raise ValueError(f"Unknown selection `{item}`.")
    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return deduped


def prompt_confirm(message: str, default: bool = False) -> bool:
    if use_questionary():
        response = QUESTIONARY.confirm(message, default=default, style=QUESTIONARY_STYLE).ask()
        if response is None:
            raise ValueError("Confirmation cancelled.")
        return bool(response)

    default_label = "Y/n" if default else "y/N"
    response = input(f"{message} [{default_label}]: ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes", "true", "1"}
