"""Compare MaxSTs obtained from the same reduced rule set with different weights.

This is a small robustness experiment: Apriori and rule reduction are not
repeated.  The script reads one reduced rule file, projects it into three
product graphs, and changes only the edge weight used for duplicate-edge
selection and Maximum Spanning Tree extraction:

* Lift
* Confidence
* Lift x Confidence

The Lift x Confidence variant corresponds to the baseline graph definition used
in the main analysis pipeline.
"""

from __future__ import annotations

import argparse
import math
import platform
from itertools import combinations
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from graph_utils import BASE_DIR, get_short_name, load_rules_dataframe
from itemset_utils import load_product_names, parse_itemset


DEFAULT_INPUT = BASE_DIR / "Rules_For_Python_REDUCED.csv"
WEIGHT_DEFINITIONS = {
    "lift": {
        "label": "Lift",
        "column": "Weight_Lift",
    },
    "confidence": {
        "label": "Confidence",
        "column": "Weight_Confidence",
    },
    "lift_confidence": {
        "label": "Lift x Confidence",
        "column": "Weight_Lift_x_Confidence",
    },
}
BASELINE_WEIGHT = "lift_confidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MaxSTs from the same reduced rule set using Lift, Confidence, "
            "and Lift x Confidence as alternative edge weights."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Reduced-rules CSV (default: Rules_For_Python_REDUCED.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: mst_weight_sensitivity beside input).",
    )
    return parser.parse_args()


def _safe_name(name: str) -> str:
    return name.upper().replace("_", "_")


def _edge_key(product_1: str, product_2: str) -> tuple[str, str]:
    return tuple(sorted((product_1, product_2)))


def _weight_value(row: object, weight_name: str) -> float:
    lift = float(row.Lift)
    confidence = float(row.Confidence)
    if weight_name == "lift":
        return lift
    if weight_name == "confidence":
        return confidence
    if weight_name == "lift_confidence":
        return lift * confidence
    raise ValueError(f"Unknown weight: {weight_name}")


def _product_names_for_rules(input_file: Path) -> frozenset[str]:
    """Load product universe from the input folder first.

    This keeps legacy comma-separated itemsets parseable even when the repository
    root has been cleaned of generated files.  Newer files with `` || ``
    separators do not depend on this, but the extra lookup is harmless.
    """

    names = set(load_product_names(input_file.parent))
    names.update(load_product_names(BASE_DIR))
    return frozenset(names)


def build_weighted_graph(
    rules: pd.DataFrame,
    weight_name: str,
    product_names: frozenset[str],
) -> nx.Graph:
    """Project rules into a graph using one selected edge-weight definition."""

    graph = nx.Graph()

    for row_number, row in enumerate(rules.itertuples(index=False), start=2):
        try:
            premises = parse_itemset(row.Premises, product_names)
            conclusions = parse_itemset(row.Conclusion, product_names)
        except ValueError as error:
            raise ValueError(f"Cannot parse itemset on CSV row {row_number}: {error}") from error

        if not premises or not conclusions:
            raise ValueError(f"CSV row {row_number} contains an empty itemset.")

        lift = float(row.Lift)
        confidence = float(row.Confidence)
        support = float(getattr(row, "Support", math.nan))
        rule_id = getattr(row, "RuleID", "")
        weight = _weight_value(row, weight_name)

        for premise in premises:
            for conclusion in conclusions:
                if premise == conclusion:
                    continue

                attributes = {
                    "weight": weight,
                    "lift": lift,
                    "confidence": confidence,
                    "support": support,
                    "lift_confidence": lift * confidence,
                    "rule_id": rule_id,
                    "weight_name": weight_name,
                }

                if graph.has_edge(premise, conclusion):
                    # For each pair, keep the strongest rule under the current
                    # weight definition.  This is the only intentional difference
                    # among the three graphs.
                    if weight > graph[premise][conclusion]["weight"]:
                        graph[premise][conclusion].update(attributes)
                    continue

                attributes["edge_order"] = graph.number_of_edges()
                graph.add_edge(premise, conclusion, **attributes)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Rule projection produced an empty graph.")
    if not nx.is_connected(graph):
        components = nx.number_connected_components(graph)
        raise ValueError(
            f"Projected graph for {weight_name} is disconnected ({components} components)."
        )
    return graph


def graph_to_dataframe(graph: nx.Graph, weight_name: str) -> pd.DataFrame:
    weight_column = WEIGHT_DEFINITIONS[weight_name]["column"]
    rows = []
    for product_1, product_2, data in graph.edges(data=True):
        rows.append(
            {
                "EdgeOrder": data.get("edge_order", len(rows)),
                "Product_1": product_1,
                "Product_2": product_2,
                "Lift": data["lift"],
                "Confidence": data["confidence"],
                "Support": data.get("support", math.nan),
                "Weight_Lift": data["lift"],
                "Weight_Confidence": data["confidence"],
                "Weight_Lift_x_Confidence": data["lift_confidence"],
                weight_column: data["weight"],
                "SelectedWeight": data["weight"],
                "RuleID": data.get("rule_id", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("EdgeOrder", kind="stable")


def mst_to_dataframe(mst: nx.Graph, weight_name: str) -> pd.DataFrame:
    weight_column = WEIGHT_DEFINITIONS[weight_name]["column"]
    rows = []
    for product_1, product_2, data in mst.edges(data=True):
        rows.append(
            {
                "Product_1": product_1,
                "Product_2": product_2,
                "Lift": data["lift"],
                "Confidence": data["confidence"],
                "Support": data.get("support", math.nan),
                "Weight_Lift": data["lift"],
                "Weight_Confidence": data["confidence"],
                "Weight_Lift_x_Confidence": data["lift_confidence"],
                weight_column: data["weight"],
                "SelectedWeight": data["weight"],
                "RuleID": data.get("rule_id", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("SelectedWeight", ascending=False, kind="stable")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def top_hubs(mst: nx.Graph, limit: int = 5) -> str:
    ranked = sorted(mst.degree(), key=lambda item: (-item[1], item[0]))[:limit]
    return "; ".join(f"{get_short_name(node)}={degree}" for node, degree in ranked)


def edge_set(mst: nx.Graph) -> set[tuple[str, str]]:
    return {_edge_key(product_1, product_2) for product_1, product_2 in mst.edges()}


def run_experiment(args: argparse.Namespace) -> None:
    input_file = args.input.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_file.parent / "mst_weight_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = perf_counter()

    step_start = perf_counter()
    rules = load_rules_dataframe(input_file)
    product_names = _product_names_for_rules(input_file)
    load_seconds = perf_counter() - step_start

    rows = []
    graphs: dict[str, nx.Graph] = {}
    msts: dict[str, nx.Graph] = {}

    for weight_name, definition in WEIGHT_DEFINITIONS.items():
        label = definition["label"]
        file_stem = _safe_name(weight_name)

        step_start = perf_counter()
        graph = build_weighted_graph(rules, weight_name, product_names)
        graph_seconds = perf_counter() - step_start

        step_start = perf_counter()
        mst = nx.maximum_spanning_tree(graph, weight="weight", algorithm="kruskal")
        mst_seconds = perf_counter() - step_start

        graph_file = output_dir / f"PRODUCT_GRAPH_WEIGHT_{file_stem}.csv"
        mst_file = output_dir / f"MST_WEIGHT_{file_stem}.csv"

        step_start = perf_counter()
        write_csv(graph_to_dataframe(graph, weight_name), graph_file)
        write_csv(mst_to_dataframe(mst, weight_name), mst_file)
        export_seconds = perf_counter() - step_start

        graphs[weight_name] = graph
        msts[weight_name] = mst

        rows.append(
            {
                "Weight": label,
                "InputReducedRules": len(rules),
                "GraphNodes": graph.number_of_nodes(),
                "GraphEdges": graph.number_of_edges(),
                "MaxSTNodes": mst.number_of_nodes(),
                "MaxSTEdges": mst.number_of_edges(),
                "MaxSTTotalSelectedWeight": sum(
                    data["weight"] for _, _, data in mst.edges(data=True)
                ),
                "TopHubsByDegree": top_hubs(mst),
                "LoadRulesSeconds": load_seconds,
                "BuildGraphSeconds": graph_seconds,
                "BuildMaxSTSeconds": mst_seconds,
                "ExportSeconds": export_seconds,
                "GraphFile": str(graph_file),
                "MaxSTFile": str(mst_file),
            }
        )

        print(
            f"{label}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} graph edges, "
            f"{mst.number_of_edges()} MaxST edges."
        )

    baseline_edges = edge_set(msts[BASELINE_WEIGHT])
    for row in rows:
        weight_key = next(
            key for key, definition in WEIGHT_DEFINITIONS.items() if definition["label"] == row["Weight"]
        )
        shared = len(edge_set(msts[weight_key]) & baseline_edges)
        row["SharedEdgesWithLift_x_Confidence"] = shared
        row["DifferingEdgesFromLift_x_Confidence"] = (
            msts[weight_key].number_of_edges() - shared
        )

    summary = pd.DataFrame(rows)
    seconds_columns = [column for column in summary.columns if column.endswith("Seconds")]
    summary[seconds_columns] = summary[seconds_columns].round(4)
    summary["MaxSTTotalSelectedWeight"] = summary["MaxSTTotalSelectedWeight"].round(12)
    write_csv(summary, output_dir / "MST_WEIGHT_SENSITIVITY_SUMMARY.csv")

    overlap_rows = []
    for left, right in combinations(WEIGHT_DEFINITIONS, 2):
        left_edges = edge_set(msts[left])
        right_edges = edge_set(msts[right])
        shared = len(left_edges & right_edges)
        overlap_rows.append(
            {
                "Weight_1": WEIGHT_DEFINITIONS[left]["label"],
                "Weight_2": WEIGHT_DEFINITIONS[right]["label"],
                "Edges_1": len(left_edges),
                "Edges_2": len(right_edges),
                "SharedEdges": shared,
                "DifferentEdgesInEachTree": len(left_edges) - shared,
                "JaccardSimilarity": shared / len(left_edges | right_edges),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap["JaccardSimilarity"] = overlap["JaccardSimilarity"].round(6)
    write_csv(overlap, output_dir / "MST_WEIGHT_EDGE_OVERLAP.csv")

    metadata = pd.DataFrame(
        {
            "Metric": [
                "Input reduced-rules file",
                "Output directory",
                "Experiment",
                "Weights compared",
                "Duplicate-edge rule",
                "MaxST algorithm",
                "Product universe entries",
                "Total seconds",
                "Python version",
                "pandas version",
                "NetworkX version",
            ],
            "Value": [
                str(input_file),
                str(output_dir),
                "Same reduced rule set, alternative MaxST edge weights",
                ", ".join(definition["label"] for definition in WEIGHT_DEFINITIONS.values()),
                "maximum selected weight per product pair",
                "Kruskal maximum spanning tree",
                str(len(product_names)),
                f"{perf_counter() - total_start:.4f}",
                platform.python_version(),
                pd.__version__,
                nx.__version__,
            ],
        }
    )
    write_csv(metadata, output_dir / "MST_WEIGHT_SENSITIVITY_RUN_METADATA.csv")

    print(f"Summary: {output_dir / 'MST_WEIGHT_SENSITIVITY_SUMMARY.csv'}")
    print(f"Edge overlap: {output_dir / 'MST_WEIGHT_EDGE_OVERLAP.csv'}")


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
