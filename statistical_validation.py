"""Statistical validation of retained association rules.

For each maxlen-specific retained-rule file, this script reconstructs the 2 x 2
contingency table from Support, Confidence, Lift, and the number of baskets. It
then computes a one-sided Fisher exact test for positive association and applies
the Benjamini-Hochberg false-discovery-rate correction within each maxlen set.

The implementation uses only the Python standard library so it can run without
installing scipy, pandas, or statsmodels.
"""

from __future__ import print_function

import argparse
import csv
import math
import shutil
import statistics
import platform
from pathlib import Path
from time import perf_counter


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ALPHA = 0.05
MAXLENS = (3, 4, 5, 6)
REQUIRED_COLUMNS = ("Premises", "Conclusion", "Support", "Confidence", "Lift")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 2x2 tables, one-sided Fisher tests, and BH correction."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BASE_DIR / "timing_runs",
        help="Directory containing maxlen_<N> run folders (legacy flat layout is also accepted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "statistical_validation",
        help="Directory for retained-rule copies and validation outputs.",
    )
    parser.add_argument(
        "--n-transactions",
        type=int,
        default=None,
        help="Override basket count. By default it is read from each Apriori run metadata file.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BH-FDR significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--maxlen", nargs="+", type=int, default=list(MAXLENS),
        help="One or more maxlen values (default: 3 4 5 6).",
    )
    return parser.parse_args()


def resolve_input(input_dir, maxlen):
    candidates = [
        input_dir / "maxlen_{}".format(maxlen) / "Rules_For_Python_REDUCED_CONFIDENCE.csv",
        input_dir / "Rules_For_Python_REDUCED_CONFIDENCE_{}.csv".format(maxlen),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Retained-rule file not found; checked: {}".format(
        ", ".join(map(str, candidates))
    ))


def infer_n_transactions(input_dir, maxlen):
    metadata_path = input_dir / "maxlen_{}".format(maxlen) / "APRIORI_RUN_METADATA.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Cannot infer basket count because metadata is missing: {}. "
            "Supply --n-transactions explicitly.".format(metadata_path)
        )
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row.get("Metric") == "Baskets":
                return int(row["Value"])
    raise ValueError("Baskets entry not found in {}".format(metadata_path))


def round_count(value):
    """Round a non-negative reconstructed count to the nearest integer."""
    return int(math.floor(value + 0.5))


def reconstruct_table(support, confidence, lift, n_transactions):
    if not (0.0 < support <= 1.0):
        raise ValueError("Support must be in (0, 1].")
    if not (0.0 < confidence <= 1.0):
        raise ValueError("Confidence must be in (0, 1].")
    if lift <= 0.0:
        raise ValueError("Lift must be greater than zero.")

    n11 = round_count(support * n_transactions)
    antecedent_total = round_count((support / confidence) * n_transactions)
    consequent_total = round_count((confidence / lift) * n_transactions)
    n10 = antecedent_total - n11
    n01 = consequent_total - n11
    n00 = n_transactions - n11 - n10 - n01

    table = (n11, n10, n01, n00)
    if min(table) < 0 or sum(table) != n_transactions:
        raise ValueError("Reconstructed contingency table is invalid: {}".format(table))
    return table


def log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def fisher_exact_greater(n11, n10, n01, n00):
    """One-sided Fisher exact p-value P(A >= n11) with fixed margins."""
    row_1 = n11 + n10
    col_1 = n11 + n01
    total_n = n11 + n10 + n01 + n00
    upper = min(row_1, col_1)

    log_probability = (
        log_comb(row_1, n11)
        + log_comb(total_n - row_1, col_1 - n11)
        - log_comb(total_n, col_1)
    )

    # Values below the smallest representable float are reported as zero, as in
    # common scientific libraries. They remain significant under BH correction.
    if log_probability < -745.0:
        return 0.0

    term = math.exp(log_probability)
    p_value = term
    compensation = 0.0
    k = n11

    while k < upper:
        numerator = (row_1 - k) * (col_1 - k)
        denominator = (k + 1) * (total_n - row_1 - col_1 + k + 1)
        if numerator == 0:
            break

        ratio = numerator / float(denominator)
        term *= ratio
        k += 1

        # Kahan summation reduces loss of precision in long hypergeometric tails.
        adjusted = term - compensation
        updated = p_value + adjusted
        compensation = (updated - p_value) - adjusted
        p_value = updated

        if ratio < 1.0 and term <= p_value * 1e-15:
            break

    return min(1.0, p_value)


def benjamini_hochberg(p_values):
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_minimum = 1.0

    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        candidate = p_values[original_index] * count / float(rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[original_index] = min(1.0, running_minimum)

    return adjusted


def read_rules(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, ", ".join(missing)))
        return list(reader), list(reader.fieldnames)


def write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def validate_maxlen(maxlen, input_path, output_dir, n_transactions, alpha):
    rows, original_fields = read_rules(input_path)
    p_values = []

    for line_number, row in enumerate(rows, start=2):
        try:
            support = float(row["Support"].replace(",", "."))
            confidence = float(row["Confidence"].replace(",", "."))
            lift = float(row["Lift"].replace(",", "."))
            table = reconstruct_table(support, confidence, lift, n_transactions)
            p_value = fisher_exact_greater(*table)
        except (TypeError, ValueError) as error:
            raise ValueError("{} line {}: {}".format(input_path, line_number, error))

        row["Maxlen"] = str(maxlen)
        row["n11"] = str(table[0])
        row["n10"] = str(table[1])
        row["n01"] = str(table[2])
        row["n00"] = str(table[3])
        row["FisherPValue"] = "{:.17g}".format(p_value)
        p_values.append(p_value)

    adjusted = benjamini_hochberg(p_values)
    for row, adjusted_p in zip(rows, adjusted):
        row["BHAdjustedPValue"] = "{:.17g}".format(adjusted_p)
        row["SignificantBH005"] = "TRUE" if adjusted_p < alpha else "FALSE"

    extra_fields = [
        "Maxlen", "n11", "n10", "n01", "n00", "FisherPValue",
        "BHAdjustedPValue", "SignificantBH005",
    ]
    output_path = output_dir / "retained_rules_maxlen_{}_with_fisher_bh.csv".format(maxlen)
    write_rows(output_path, original_fields + extra_fields, rows)

    canonical_path = output_dir / "retained_rules_maxlen_{}.csv".format(maxlen)
    shutil.copyfile(str(input_path), str(canonical_path))

    significant_count = sum(value < alpha for value in adjusted)
    summary = {
        "Maxlen": str(maxlen),
        "RetainedRules": str(len(rows)),
        "SignificantAfterBH005": str(significant_count),
        "PercentSignificant": "{:.6f}".format(100.0 * significant_count / len(rows)),
        "MinAdjustedPValue": "{:.17g}".format(min(adjusted)),
        "MedianAdjustedPValue": "{:.17g}".format(statistics.median(adjusted)),
        "NTransactions": str(n_transactions),
    }

    # Manuscript Table 5 reports the five baseline (maxlen=3) rules ranked by
    # Lift, rather than the graph edge score Lift x Confidence.
    top_rows = sorted(
        rows,
        key=lambda row: float(row["Lift"].replace(",", ".")),
        reverse=True,
    )[:5]
    for rank, row in enumerate(top_rows, start=1):
        row["Rank"] = str(rank)
        row["RankingMetric"] = "Lift"
        row["LiftXConfidence"] = "{:.17g}".format(
            float(row["Lift"].replace(",", "."))
            * float(row["Confidence"].replace(",", "."))
        )

    return summary, top_rows


def main():
    args = parse_args()
    if args.n_transactions is not None and args.n_transactions <= 0:
        raise ValueError("--n-transactions must be greater than zero.")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be between zero and one.")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    summaries = []
    table5_rows = []

    transaction_counts = []
    for maxlen in args.maxlen:
        input_path = resolve_input(input_dir, maxlen)
        n_transactions = (
            args.n_transactions
            if args.n_transactions is not None
            else infer_n_transactions(input_dir, maxlen)
        )
        transaction_counts.append(n_transactions)
        summary, top_rows = validate_maxlen(
            maxlen, input_path, output_dir, n_transactions, args.alpha
        )
        summaries.append(summary)
        if maxlen == 3:
            table5_rows = top_rows
        print(
            "maxlen={}: {} retained, {} significant after BH-FDR".format(
                maxlen, summary["RetainedRules"], summary["SignificantAfterBH005"]
            )
        )

    summary_fields = [
        "Maxlen", "RetainedRules", "SignificantAfterBH005",
        "PercentSignificant", "MinAdjustedPValue", "MedianAdjustedPValue",
        "NTransactions",
    ]
    write_rows(output_dir / "statistical_validation_summary.csv", summary_fields, summaries)

    top_fields = [
        "Maxlen", "Rank", "RankingMetric", "RuleID", "Premises", "Conclusion", "Support", "Confidence",
        "Lift", "LiftXConfidence", "n11", "n10", "n01", "n00",
        "FisherPValue", "BHAdjustedPValue", "SignificantBH005",
    ]
    # Keep the original helper filename for compatibility, while also exporting
    # a manuscript-specific filename that makes the provenance explicit.
    write_rows(output_dir / "top5_statistical_validation.csv", top_fields, table5_rows)
    write_rows(output_dir / "table5_statistical_validation.csv", top_fields, table5_rows)

    elapsed = perf_counter() - start
    write_rows(
        output_dir / "STATISTICAL_VALIDATION_RUN_METADATA.csv",
        ["Metric", "Value"],
        [
            {"Metric": "Maxlen values", "Value": ", ".join(map(str, args.maxlen))},
            {"Metric": "Basket counts", "Value": ", ".join(map(str, transaction_counts))},
            {"Metric": "Test", "Value": "One-sided Fisher exact test (greater)"},
            {"Metric": "Multiple-testing correction", "Value": "Benjamini-Hochberg within maxlen"},
            {"Metric": "Alpha", "Value": args.alpha},
            {"Metric": "Python version", "Value": platform.python_version()},
        ],
    )
    write_rows(
        output_dir / "STATISTICAL_VALIDATION_TIMING.csv",
        ["Step", "Seconds"],
        [{"Step": "Total pipeline", "Seconds": "{:.4f}".format(elapsed)}],
    )

    print("Outputs saved to: {}".format(output_dir))
    print("Elapsed seconds: {:.2f}".format(elapsed))


if __name__ == "__main__":
    main()
