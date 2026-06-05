from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def json_safe(value: Any) -> Any:
    """Convert common NumPy/scalar values into JSON-serializable objects."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON file with stable indentation."""

    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Iterable[str] | None = None) -> Path:
    """Write rows to CSV, preserving a stable union of row fields."""

    output = Path(path)
    ensure_dir(output.parent)
    rows_list = [dict(row) for row in rows]
    if fieldnames is None:
        names: list[str] = []
        for row in rows_list:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    else:
        fieldnames = list(fieldnames)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(text: str, path: str | Path) -> Path:
    """Write text to a file."""

    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")
    return output


def _csv_safe(value: Any) -> Any:
    value = json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value
