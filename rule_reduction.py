"""Compare confidence, lift, and Lift x Confidence rule reduction.

Rules are compared only when they have the same consequent. An expanded rule
is removed when an already retained subset of its antecedent improves the
selected score by less than ``delta``. The manuscript comparison uses the same
delta = 0.05 for all three criteria.
"""

from __future__ import annotations

import argparse
import platform
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import polars as pl

from itemset_utils import parse_itemset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Rules_For_Python.csv"
DEFAULT_DELTA = 0.05
REQUIRED_COLUMNS = ("Premises", "Conclusion", "Confidence", "Lift")
NUMERIC_COLUMNS = ("Support", "Confidence", "Lift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce rules using confidence, lift, or their product."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Apriori rules CSV (default: Rules_For_Python.csv beside this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Reduced-rules CSV (default: INPUT stem plus _REDUCED_METRIC.csv).",
    )
    parser.add_argument(
        "--metric",
        choices=("confidence", "lift", "product"),
        default="confidence",
        help="Reduction score: confidence, lift, or product (default: confidence).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help="Minimum selected-score improvement (default: 0.05).",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if not 0 <= args.delta <= 1:
        raise ValueError("--delta must be between 0 and 1.")

    input_file = args.input.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input rules file not found: {input_file}")

    output_file = (
        args.output.expanduser().resolve()
        if args.output
        else input_file.with_name(
            f"{input_file.stem}_REDUCED_{args.metric.upper()}.csv"
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metric_label = args.metric.upper()
    timing_file = output_file.parent / f"RULE_REDUCTION_{metric_label}_TIMING_SUMMARY.csv"
    metadata_file = output_file.parent / f"RULE_REDUCTION_{metric_label}_RUN_METADATA.csv"
    return input_file, output_file, timing_file, metadata_file


def read_rules_csv(path: Path) -> pl.DataFrame:
    """Read either the semicolon exports used by the pipeline or legacy CSV."""
    errors: list[str] = []
    for separator in (";", ","):
        try:
            frame = pl.read_csv(path, separator=separator)
        except Exception as error:  # Keep both parser errors for a useful message.
            errors.append(f"separator {separator!r}: {error}")
            continue

        stripped = [str(column).strip() for column in frame.columns]
        if len(stripped) != len(set(stripped)):
            raise ValueError("Column names are duplicated after trimming whitespace.")
        frame.columns = stripped
        if all(column in frame.columns for column in REQUIRED_COLUMNS):
            return frame

    detail = "; ".join(errors) if errors else "required columns were not found"
    raise ValueError(f"Could not read association-rule CSV {path}: {detail}")


def prepare_rules_dataframe(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input rules are missing columns: {', '.join(missing)}")
    if frame.height == 0:
        raise ValueError("Input rules file is empty.")

    if "RuleID" not in frame.columns:
        frame = frame.with_row_index(name="RuleID")
    elif frame["RuleID"].n_unique() != frame.height:
        raise ValueError("RuleID values must be unique.")

    # Decimal commas are supported for legacy exports; new R output is numeric.
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column)
                .cast(pl.String)
                .str.replace(",", ".", literal=True)
                .cast(pl.Float64, strict=False)
                .alias(column)
            )
            if frame[column].null_count() > 0:
                raise ValueError(f"Column {column} contains missing or invalid numbers.")

    invalid_confidence = frame.filter(
        (pl.col("Confidence") < 0) | (pl.col("Confidence") > 1)
    ).height
    if invalid_confidence:
        raise ValueError("Confidence values must be between 0 and 1.")

    return frame


def canonical_itemset(value: object, label: str, rule_id: object) -> tuple[str, ...]:
    try:
        items = tuple(sorted(parse_itemset(value)))
    except ValueError as error:
        raise ValueError(f"Cannot parse {label} for RuleID {rule_id}: {error}") from error
    if not items:
        raise ValueError(f"RuleID {rule_id} has an empty {label}.")
    return items


def process_consequent_group(rules: list[dict], delta: float) -> set[object]:
    """Return IDs removed from one group of rules sharing a consequent."""
    removed: set[object] = set()
    retained_by_size: dict[int, dict[tuple[str, ...], float]] = defaultdict(dict)

    # Simpler antecedents must be evaluated first because they are the reference
    # rules against which score improvement of expanded antecedents is tested.
    rules.sort(key=lambda rule: (rule["antecedent_size"], rule["antecedent"]))

    for rule in rules:
        should_remove = False

        for subset_size in range(1, rule["antecedent_size"]):
            retained_subsets = retained_by_size.get(subset_size)
            if not retained_subsets:
                continue

            for subset in combinations(rule["antecedent"], subset_size):
                parent_score = retained_subsets.get(subset)
                if parent_score is None:
                    continue

                improvement = rule["score"] - parent_score
                if improvement < delta:
                    removed.add(rule["id"])
                    should_remove = True
                    break

            if should_remove:
                break

        if should_remove:
            continue

        # Duplicate antecedent/consequent forms, if present, retain the strongest
        # score as the reference for rules with larger antecedents.
        same_size = retained_by_size[rule["antecedent_size"]]
        previous_best = same_size.get(rule["antecedent"])
        if previous_best is None or rule["score"] > previous_best:
            same_size[rule["antecedent"]] = rule["score"]

    return removed


def reduction_score(confidence: float, lift: float, metric: str) -> float:
    if metric == "confidence":
        return confidence
    if metric == "lift":
        return lift
    return confidence * lift


def reduce_rules(
    frame: pl.DataFrame,
    delta: float,
    metric: str,
) -> tuple[pl.DataFrame, int]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)

    for rule_id, premises, conclusion, confidence, lift in frame.select(
        "RuleID", "Premises", "Conclusion", "Confidence", "Lift"
    ).iter_rows():
        antecedent = canonical_itemset(premises, "antecedent", rule_id)
        consequent = canonical_itemset(conclusion, "consequent", rule_id)
        groups[consequent].append(
            {
                "id": rule_id,
                "antecedent": antecedent,
                "antecedent_size": len(antecedent),
                "score": reduction_score(confidence, lift, metric),
            }
        )

    removed: set[object] = set()
    for group in groups.values():
        if len(group) > 1:
            removed.update(process_consequent_group(group, delta))

    reduced = frame.filter(~pl.col("RuleID").is_in(removed))
    return reduced, len(groups)


def write_csv_safely(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_csv(temporary, separator=";")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    input_file, output_file, timing_file, metadata_file = resolve_paths(args)
    total_start = perf_counter()

    step_start = perf_counter()
    frame = read_rules_csv(input_file)
    load_seconds = perf_counter() - step_start

    step_start = perf_counter()
    prepared = prepare_rules_dataframe(frame)
    prepare_seconds = perf_counter() - step_start

    step_start = perf_counter()
    reduced, consequent_groups = reduce_rules(prepared, args.delta, args.metric)
    reduction_seconds = perf_counter() - step_start

    step_start = perf_counter()
    write_csv_safely(reduced, output_file)
    export_seconds = perf_counter() - step_start
    total_seconds = perf_counter() - total_start

    timing = pl.DataFrame(
        {
            "Step": [
                "Load rules",
                "Prepare rules",
                "Rule reduction",
                "Export reduced rules",
                "Total pipeline",
            ],
            "Seconds": [
                round(load_seconds, 2),
                round(prepare_seconds, 2),
                round(reduction_seconds, 2),
                round(export_seconds, 2),
                round(total_seconds, 2),
            ],
        }
    )
    write_csv_safely(timing, timing_file)

    input_rules = prepared.height
    output_rules = reduced.height
    metadata = pl.DataFrame(
        {
            "Metric": [
                "Input file",
                "Output file",
                "Reduction metric",
                "Improvement threshold delta",
                "Consequent groups",
                "Input rules",
                "Removed rules",
                "Output rules",
                "Retained percent",
                "Python version",
                "Polars version",
            ],
            "Value": [
                str(input_file),
                str(output_file),
                args.metric,
                str(args.delta),
                str(consequent_groups),
                str(input_rules),
                str(input_rules - output_rules),
                str(output_rules),
                f"{100 * output_rules / input_rules:.4f}",
                platform.python_version(),
                pl.__version__,
            ],
        }
    )
    write_csv_safely(metadata, metadata_file)

    print("Multi-metric rule reduction completed.")
    print(f"Metric: {args.metric}")
    print(f"Delta: {args.delta}")
    print(f"Input rules: {input_rules}")
    print(f"Removed rules: {input_rules - output_rules}")
    print(f"Output rules: {output_rules}")
    print(f"Reduced rules saved to: {output_file}")
    for row in timing.iter_rows(named=True):
        print(f"{row['Step']}: {row['Seconds']} s")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
