from __future__ import annotations

from pathlib import Path


def load_keybind_config(path: str | Path) -> dict[str, dict[str, str]]:
    config_path = Path(path)
    config: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    lines: list[tuple[int, str]] = []

    for line_number, raw_line in enumerate(config_path.read_text().splitlines(), 1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        lines.append((line_number, line))

    common_indent = min((_indent_width(line) for _, line in lines), default=0)

    for line_number, raw_line in lines:
        line = raw_line[common_indent:]
        if not line.startswith((" ", "\t")):
            if not line.endswith(":"):
                raise ValueError(f"{config_path}:{line_number}: expected a section ending with ':'")
            current_section = line[:-1].strip()
            config[current_section] = {}
            continue

        if current_section is None:
            raise ValueError(f"{config_path}:{line_number}: key outside a section")

        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"{config_path}:{line_number}: expected 'key: value'")
        config[current_section][key.strip()] = _unquote(value.strip())

    return config


def sensitivity_config(config: dict[str, dict[str, str]]) -> dict[str, float]:
    section = config.get("sensitivity", {})
    values = {
        "default_sensitivity": _float_setting(section, "default_sensitivity", 5.0),
        "min_sensitivity": _float_setting(section, "min_sensitivity", 0.1),
        "max_sensitivity": _float_setting(section, "max_sensitivity", 15.0),
    }
    validate_sensitivity_bounds(values["min_sensitivity"], values["max_sensitivity"])
    return values


def validate_sensitivity_bounds(min_sensitivity: float, max_sensitivity: float):
    if min_sensitivity <= 0.0:
        raise ValueError("min_sensitivity must be greater than 0")
    if max_sensitivity < min_sensitivity:
        raise ValueError("max_sensitivity must be greater than or equal to min_sensitivity")


def _float_setting(section: dict[str, str], key: str, default: float) -> float:
    raw_value = section.get(key)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"sensitivity.{key} must be a number, got {raw_value!r}") from exc


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
