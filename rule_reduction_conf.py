# ============================================================
# RULE REDUCTION -> Remove Redundant Rules
# ============================================================

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import polars as pl


# ==============================
# SETTINGS
# ==============================

IMPROVEMENT_THRESHOLD = 0.05
BASE_DIR = Path(__file__).resolve().parent


# ==============================
# SMALL HELPERS
# ==============================

def parse_premise_items(value):
    return tuple(
        sorted(item.strip() for item in str(value).split(",") if item.strip())
    )


def process_single_group(rules):
    rules_to_remove = set()
    kept_rules_by_size = defaultdict(dict)

    rules.sort(key=lambda item: item["premise_len"])

    for rule in rules:
        removable = False

        for subset_size in range(1, rule["premise_len"]):
            kept_rules = kept_rules_by_size.get(subset_size)

            if not kept_rules:
                continue

            for subset in combinations(rule["premise"], subset_size):
                parent_confidence = kept_rules.get(subset)

                if parent_confidence is None:
                    continue

                improvement = rule["confidence"] - parent_confidence

                if improvement < IMPROVEMENT_THRESHOLD:
                    rules_to_remove.add(rule["id"])
                    removable = True
                    break

            if removable:
                break

        if removable:
            continue

        current_best = kept_rules_by_size[rule["premise_len"]].get(rule["premise"])

        if current_best is None or rule["confidence"] > current_best:
            kept_rules_by_size[rule["premise_len"]][rule["premise"]] = (
                rule["confidence"]
            )

    return rules_to_remove


# ==============================
# FILE LOADING
# ==============================

def load_rules_file(filepath):
    try:
        df = pl.read_csv(filepath, separator=";")
        if "Premises" in df.columns:
            return df

        df = pl.read_csv(filepath, separator=",")
        if "Premises" in df.columns:
            return df
    except Exception as error:
        print(f"Error while loading file: {error}")
        return None

    return None


# ==============================
# DATA PREPARATION
# ==============================

def prepare_rules_dataframe(df):
    df = df.rename({column: column.strip() for column in df.columns})

    if "RuleID" not in df.columns:
        df = df.with_row_index(name="RuleID")

    numeric_columns = ["Confidence", "Lift", "Support"]

    for column in numeric_columns:
        if column in df.columns and df[column].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(column).str.replace(",", ".").cast(pl.Float64)
            )

    return df


# ==============================
# REDUCTION LOGIC
# ==============================

def reduce_rules(df):
    grouped_rules = defaultdict(list)

    for rule_id, premises, conclusion, confidence in df.select(
        ["RuleID", "Premises", "Conclusion", "Confidence"]
    ).iter_rows():
        premise_items = parse_premise_items(premises)

        grouped_rules[str(conclusion)].append(
            {
                "id": rule_id,
                "premise": premise_items,
                "premise_len": len(premise_items),
                "confidence": confidence,
            }
        )

    rules_to_remove = set()

    for rules in grouped_rules.values():
        if len(rules) < 2:
            continue

        rules_to_remove.update(process_single_group(rules))

    return df.filter(~pl.col("RuleID").is_in(rules_to_remove))


# ==============================
# ENTRY POINT
# ==============================

def main():
    input_file = BASE_DIR / "Rules_For_Python.csv"

    df = load_rules_file(input_file)

    if df is None:
        print("Failed to load file.")
        return

    reduced_df = reduce_rules(prepare_rules_dataframe(df))
    output_file = input_file.with_name(f"{input_file.stem}_REDUCED.csv")
    reduced_df.write_csv(output_file, separator=";")


if __name__ == "__main__":
    main()
