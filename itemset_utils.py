"""Safe serialization and parsing of association-rule itemsets.

New exports use `` || `` between items.  The parser also supports legacy CSV
files whose itemsets used commas, including product names that themselves
contain commas, by resolving tokens against the exported product universe.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
ITEMSET_SEPARATOR = " || "


def load_product_names(base_dir: str | Path = BASE_DIR) -> frozenset[str]:
    names: set[str] = set()
    for path in Path(base_dir).glob("ALL_PRODUCTS*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            # Project product-universe files are exported by data.table with
            # sep=";".  A one-column header contains no visible delimiter, so
            # delimiter sniffing would incorrectly treat commas in names as CSV
            # separators.
            for row in csv.DictReader(handle, delimiter=";"):
                value = row.get("Product")
                if value and value.strip():
                    names.add(value.strip())
    return frozenset(names)


PRODUCT_NAMES = load_product_names()


def serialize_itemset(items: Iterable[object]) -> str:
    return ITEMSET_SEPARATOR.join(str(item).strip() for item in items if str(item).strip())


def parse_itemset(
    value: object,
    product_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    text = str(value).strip().strip("{}")
    if not text:
        return ()

    if "||" in text:
        return tuple(item.strip() for item in text.split("||") if item.strip())

    known = frozenset(product_names) if product_names is not None else PRODUCT_NAMES
    if text in known:
        return (text,)

    tokens = tuple(token.strip() for token in text.split(","))
    if not known:
        return tuple(token for token in tokens if token)

    solutions: dict[int, list[tuple[str, ...]]] = {0: [()]}
    for start in range(len(tokens)):
        for prefix in solutions.get(start, []):
            for end in range(start + 1, len(tokens) + 1):
                candidate = ", ".join(tokens[start:end])
                if candidate in known:
                    solutions.setdefault(end, []).append(prefix + (candidate,))

    parsed = solutions.get(len(tokens), [])
    unique = list(dict.fromkeys(parsed))
    if len(unique) != 1:
        raise ValueError(
            f"Cannot uniquely parse itemset {text!r}. "
            "Use the ' || ' separator or provide the complete product universe."
        )
    return unique[0]
