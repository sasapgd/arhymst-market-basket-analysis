"""Run delta and support/confidence sensitivity experiments reproducibly.

Support/confidence scenarios rerun Apriori on the anonymized data. Delta
scenarios reuse the baseline Apriori rules because delta acts only during the
post-mining confidence-improvement reduction.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from graph_utils import build_full_graph_from_rules, export_product_graph
from mst_network_analysis import build_mst_from_graph, export_mst


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RSCRIPT = Path(r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe")
THRESHOLD_SCENARIOS = (
    ("Baseline", 0.0010, 0.30),
    ("Lower support", 0.0005, 0.30),
    ("Higher support", 0.0020, 0.30),
    ("Lower confidence", 0.0010, 0.20),
    ("Higher confidence", 0.0010, 0.40),
    ("Lower support + confidence", 0.0005, 0.20),
    ("Higher support + confidence", 0.0020, 0.40),
)
DEFAULT_DELTAS = tuple(value / 100 for value in range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run delta and support/confidence sensitivity experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "Data")
    parser.add_argument(
        "--baseline-dir", type=Path, default=BASE_DIR / "timing_runs" / "maxlen_3"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "sensitivity_experiments"
    )
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS))
    parser.add_argument(
        "--reuse-threshold-runs",
        action="store_true",
        help="Reuse complete scenario directories instead of rerunning Apriori.",
    )
    parser.add_argument("--skip-delta", action="store_true")
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument(
        "--scenario", nargs="+", type=int, choices=range(1, 8),
        help="Run only selected support/confidence scenario numbers.",
    )
    return parser.parse_args()


def run(command: list[object], cwd: Path) -> None:
    print("> " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def read_rule_count(path: Path) -> int:
    # Apriori exports do not require RuleID; reduction scripts add it when absent.
    return len(pd.read_csv(path, sep=";", usecols=[0]))


def canonical_edges(graph: nx.Graph) -> set[tuple[str, str]]:
    return {tuple(sorted((str(a), str(b)))) for a, b in graph.edges()}


def top_hub(tree: nx.Graph) -> str:
    return sorted(tree.degree(), key=lambda item: (-item[1], item[0]))[0][0]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def build_network(reduced_file: Path, output_dir: Path) -> tuple[nx.Graph, nx.Graph, float]:
    start = perf_counter()
    graph = build_full_graph_from_rules(reduced_file)
    export_product_graph(graph, output_dir / "PRODUCT_GRAPH.csv")
    tree = build_mst_from_graph(graph)
    export_mst(tree, output_dir / "MST.csv")
    return graph, tree, perf_counter() - start


def reduce_rules(raw_file: Path, output_dir: Path, delta: float) -> tuple[Path, float]:
    reduced_file = output_dir / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
    start = perf_counter()
    run(
        [
            sys.executable,
            BASE_DIR / "rule_reduction_conf.py",
            "--input",
            raw_file,
            "--output",
            reduced_file,
            "--delta",
            delta,
        ],
        BASE_DIR,
    )
    wall_seconds = perf_counter() - start
    timing_file = output_dir / "RULE_REDUCTION_TIMING_SUMMARY.csv"
    if timing_file.is_file():
        timing = pd.read_csv(timing_file, sep=";")
        pipeline_seconds = float(
            timing.loc[timing["Step"] == "Total pipeline", "Seconds"].iloc[0]
        )
    else:
        pipeline_seconds = wall_seconds
    return reduced_file, pipeline_seconds


def prepare_baseline(baseline_dir: Path, scenario_dir: Path) -> tuple[Path, Path]:
    raw_source = baseline_dir / "Rules_For_Python.csv"
    reduced_source = baseline_dir / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
    if not raw_source.is_file() or not reduced_source.is_file():
        raise FileNotFoundError(f"Incomplete baseline directory: {baseline_dir}")
    scenario_dir.mkdir(parents=True, exist_ok=True)
    raw_file = scenario_dir / raw_source.name
    reduced_file = scenario_dir / reduced_source.name
    shutil.copy2(raw_source, raw_file)
    shutil.copy2(reduced_source, reduced_file)
    return raw_file, reduced_file


def threshold_experiments(args: argparse.Namespace) -> pd.DataFrame:
    threshold_root = args.output_dir / "support_confidence"
    baseline_dir = threshold_root / "scenario_01_baseline"
    baseline_raw, baseline_reduced = prepare_baseline(args.baseline_dir, baseline_dir)
    baseline_graph, baseline_tree, baseline_network_seconds = build_network(
        baseline_reduced, baseline_dir
    )
    baseline_edges = canonical_edges(baseline_tree)

    rows: list[dict[str, object]] = []
    for number, (label, support, confidence) in enumerate(THRESHOLD_SCENARIOS, start=1):
        if args.scenario and number not in args.scenario:
            continue
        slug = label.lower().replace(" + ", "_").replace(" ", "_")
        scenario_dir = threshold_root / f"scenario_{number:02d}_{slug}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if number == 1:
            raw_file, reduced_file = baseline_raw, baseline_reduced
            graph, tree = baseline_graph, baseline_tree
            apriori_seconds = float(
                pd.read_csv(
                    args.baseline_dir / "APRIORI_TIMING_SUMMARY.csv", sep=";"
                ).loc[lambda frame: frame["Step"] == "Total pipeline", "Seconds"].iloc[0]
            )
            reduction_seconds = float(
                pd.read_csv(
                    args.baseline_dir / "RULE_REDUCTION_CONFIDENCE_TIMING_SUMMARY.csv",
                    sep=";",
                ).loc[lambda frame: frame["Step"] == "Total pipeline", "Seconds"].iloc[0]
            )
            network_seconds = baseline_network_seconds
        else:
            raw_file = scenario_dir / "Rules_For_Python.csv"
            reduced_file = scenario_dir / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
            apriori_complete = raw_file.is_file() and (
                scenario_dir / "APRIORI_TIMING_SUMMARY.csv"
            ).is_file()
            if not (args.reuse_threshold_runs and apriori_complete):
                run(
                    [
                        args.rscript,
                        BASE_DIR / "apriori.R",
                        f"--input-dir={args.data_dir}",
                        f"--output-dir={scenario_dir}",
                        "--maxlen=3",
                        f"--support={support}",
                        f"--confidence={confidence}",
                        "--min-lift=1",
                    ],
                    BASE_DIR,
                )
            if not (args.reuse_threshold_runs and reduced_file.is_file()):
                reduced_file, reduction_seconds = reduce_rules(raw_file, scenario_dir, 0.05)
            else:
                reduction_timing = pd.read_csv(
                    scenario_dir / "RULE_REDUCTION_TIMING_SUMMARY.csv", sep=";"
                )
                reduction_seconds = float(
                    reduction_timing.loc[
                        reduction_timing["Step"] == "Total pipeline", "Seconds"
                    ].iloc[0]
                )
            apriori_timing = pd.read_csv(
                scenario_dir / "APRIORI_TIMING_SUMMARY.csv", sep=";"
            )
            apriori_seconds = float(
                apriori_timing.loc[
                    apriori_timing["Step"] == "Total pipeline", "Seconds"
                ].iloc[0]
            )
            graph, tree, network_seconds = build_network(reduced_file, scenario_dir)

        tree_edges = canonical_edges(tree)
        rows.append(
            {
                "Scenario": label,
                "Support": support,
                "Confidence": confidence,
                "InitialRules": read_rule_count(raw_file),
                "RetainedRules": read_rule_count(reduced_file),
                "MaxSTNodes": tree.number_of_nodes(),
                "MaxSTEdges": tree.number_of_edges(),
                "SharedEdgesWithBaseline": len(tree_edges & baseline_edges),
                "TopHub": top_hub(tree),
                "AprioriTotalSeconds": round(apriori_seconds, 4),
                "ReductionTotalSeconds": round(reduction_seconds, 4),
                "NetworkTotalSeconds": round(network_seconds, 4),
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, threshold_root / "SUPPORT_CONFIDENCE_SENSITIVITY.csv")
    return result


def delta_experiments(args: argparse.Namespace) -> pd.DataFrame:
    delta_root = args.output_dir / "delta"
    raw_file = args.baseline_dir / "Rules_For_Python.csv"
    baseline_reduced = args.baseline_dir / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
    _, baseline_tree, _ = build_network(baseline_reduced, delta_root / "baseline")
    baseline_edges = canonical_edges(baseline_tree)

    rows: list[dict[str, object]] = []
    for delta in args.deltas:
        if not 0 <= delta <= 1:
            raise ValueError("Every delta must be between zero and one.")
        scenario_dir = delta_root / f"delta_{delta:.2f}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        reduced_file, reduction_seconds = reduce_rules(raw_file, scenario_dir, delta)
        graph, tree, network_seconds = build_network(reduced_file, scenario_dir)
        edges = canonical_edges(tree)
        rows.append(
            {
                "Delta": delta,
                "InitialRules": read_rule_count(raw_file),
                "RetainedRules": read_rule_count(reduced_file),
                "MaxSTNodes": tree.number_of_nodes(),
                "MaxSTEdges": tree.number_of_edges(),
                "SharedEdgesWithBaseline": len(edges & baseline_edges),
                "MaxSTUnchanged": edges == baseline_edges,
                "TopHub": top_hub(tree),
                "ReductionTotalSeconds": round(reduction_seconds, 4),
                "NetworkTotalSeconds": round(network_seconds, 4),
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, delta_root / "DELTA_SENSITIVITY.csv")
    return result


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.baseline_dir = args.baseline_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.rscript = args.rscript.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_thresholds and not args.rscript.is_file():
        raise FileNotFoundError(f"Rscript not found: {args.rscript}")

    if not args.skip_delta:
        print("\nDELTA SENSITIVITY", flush=True)
        print(delta_experiments(args).to_string(index=False))
    if not args.skip_thresholds:
        print("\nSUPPORT/CONFIDENCE SENSITIVITY", flush=True)
        print(threshold_experiments(args).to_string(index=False))


if __name__ == "__main__":
    main()
