# ============================================================
# RULE REDUCTION -> Remove Redundant Rules (MULTI-METRIC VERSION)
# ============================================================

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import polars as pl


# ==============================
# SETTINGS
# ==============================

IMPROVEMENT_THRESHOLD = 0.05

# Reduction metric selection:
# options: "confidence", "lift", "product"
REDUCTION_METRIC = "confidence"

BASE_DIR = Path(__file__).resolve().parent


# ==============================
# SCORE FUNCTION
# ==============================

def get_rule_score(rule):
    if REDUCTION_METRIC == "confidence":
        return rule["confidence"]
    elif REDUCTION_METRIC == "lift":
        return rule["lift"]
    elif REDUCTION_METRIC == "product":
        return rule["confidence"] * rule["lift"]
    else:
        raise ValueError(f"Unknown REDUCTION_METRIC: {REDUCTION_METRIC}")


# ==============================
# SMALL HELPERS
# ==============================

def parse_premise_items(value):
    return tuple(
        sorted(item.strip() for item in str(value).split(",") if item.strip())
    )


# ==============================
# CORE REDUCTION LOGIC
# ==============================

def process_single_group(rules):
    rules_to_remove = set()
    kept_rules_by_size = defaultdict(dict)

    # Sort rules by premise size.
    rules.sort(key=lambda item: item["premise_len"])

    for rule in rules:
        removable = False
        current_score = get_rule_score(rule)

        for subset_size in range(1, rule["premise_len"]):
            kept_rules = kept_rules_by_size.get(subset_size)

            if not kept_rules:
                continue

            for subset in combinations(rule["premise"], subset_size):
                parent_score = kept_rules.get(subset)

                if parent_score is None:
                    continue

                improvement = current_score - parent_score

                if improvement < IMPROVEMENT_THRESHOLD:
                    rules_to_remove.add(rule["id"])
                    removable = True
                    break

            if removable:
                break

        if removable:
            continue

        # Keep the best score seen for this exact premise.
        current_best = kept_rules_by_size[rule["premise_len"]].get(rule["premise"])

        if current_best is None or current_score > current_best:
            kept_rules_by_size[rule["premise_len"]][rule["premise"]] = current_score

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

    for rule_id, premises, conclusion, confidence, lift in df.select(
        ["RuleID", "Premises", "Conclusion", "Confidence", "Lift"]
    ).iter_rows():

        premise_items = parse_premise_items(premises)

        grouped_rules[str(conclusion)].append(
            {
                "id": rule_id,
                "premise": premise_items,
                "premise_len": len(premise_items),
                "confidence": confidence,
                "lift": lift,
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

    df_prepared = prepare_rules_dataframe(df)
    reduced_df = reduce_rules(df_prepared)

    output_file = input_file.with_name(
        f"{input_file.stem}_REDUCED_{REDUCTION_METRIC.upper()}.csv"
    )

    reduced_df.write_csv(output_file, separator=";")

    print("===================================")
    print(f"Reduction metric: {REDUCTION_METRIC}")
    print(f"Input rules: {df_prepared.shape[0]}")
    print(f"Output rules: {reduced_df.shape[0]}")
    print(f"Saved to: {output_file}")
    print("===================================")


if __name__ == "__main__":
    main()
