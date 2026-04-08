# ============================================================
# TOP RULES -> Export the Strongest Reduced Rules
# ============================================================

from pathlib import Path
import sys

import pandas as pd

from graph_utils import DEFAULT_REDUCED_RULES_FILE


# ==============================
# SETTINGS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = DEFAULT_REDUCED_RULES_FILE
TOP_N = 1000

# ==============================
# SMALL HELPERS
# ==============================

def resolve_top_n() -> int:
    if len(sys.argv) < 2:
        return TOP_N

    value = int(sys.argv[1])

    if value <= 0:
        raise ValueError("TOP_N must be greater than 0.")

    return value


def count_rule_length(premises: object, conclusion: object) -> int:
    premise_count = len([item for item in str(premises).split(",") if item.strip()])
    conclusion_count = len([item for item in str(conclusion).split(",") if item.strip()])
    return premise_count + conclusion_count


# ==============================
# LOAD RULES
# ==============================

top_n = resolve_top_n()
df = pd.read_csv(INPUT_FILE, sep=";")

for column in ["Lift", "Confidence", "Support"]:
    if column in df.columns:
        df[column] = (
            df[column].astype(str).str.replace(",", ".", regex=False).astype(float)
        )


# ==============================
# BUILD TOP LIST
# ==============================

df["Weight"] = df["Lift"] * df["Confidence"]
df["Rule_Length"] = df.apply(
    lambda row: count_rule_length(row["Premises"], row["Conclusion"]),
    axis=1,
)
top_rules = df.sort_values(by="Weight", ascending=False).head(top_n).copy()
top_rules.insert(0, "Rank", range(1, len(top_rules) + 1))
rule_length_column = top_rules.pop("Rule_Length")
top_rules.insert(1, "Rule_Length", rule_length_column)


# ==============================
# EXPORT AND PRINT
# ==============================

output_file = BASE_DIR / f"top{top_n}.csv"
top_rules.to_csv(output_file, sep=";", index=False)
