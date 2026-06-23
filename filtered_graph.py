"""Retain selected percentages of the strongest projected graph edges."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from graph_utils import (
    BASE_DIR,
    DEFAULT_PRODUCT_GRAPH_FILE,
    get_short_name,
    load_projected_graph,
)


OUTPUT_CSV = BASE_DIR / "Filtered_Graph.csv"
FILTERED_COLUMNS = (
    "EdgeOrder",
    "Product_1",
    "Product_2",
    "Short_1",
    "Short_2",
    "Lift",
    "Confidence",
    "Support",
    "Weight_Lift_x_Confidence",
    "RuleID",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export graph variants containing the strongest edges."
    )
    parser.add_argument(
        "--input-graph",
        type=Path,
        default=DEFAULT_PRODUCT_GRAPH_FILE,
        help="Projected product-graph CSV (default: PRODUCT_GRAPH.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: directory containing the input graph).",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[100.0],
        help="Percentages of strongest edges to retain (default: 100).",
    )
    return parser.parse_args()


def normalize_percentages(values: list[float]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("Each percentage must be greater than 0 and at most 100.")
        fraction = value / 100.0
        if fraction not in normalized:
            normalized.append(fraction)
    return normalized


def format_percent_label(top_percent: float) -> str:
    percent_value = top_percent * 100
    if float(percent_value).is_integer():
        return str(int(percent_value))
    return f"{percent_value:.2f}".rstrip("0").rstrip(".").replace(".", "_")


def build_filtered_graph_from_graph(graph: nx.Graph, top_percent: float) -> nx.Graph:
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be greater than 0 and at most 1.")
    if graph.number_of_edges() == 0:
        raise ValueError("Cannot filter a graph with no edges.")

    # Edge order resolves equal-weight ties reproducibly at percentage cutoffs.
    edges_sorted = sorted(
        graph.edges(data=True),
        key=lambda edge: (-edge[2]["weight"], edge[2].get("edge_order", 0)),
    )
    keep_n = max(1, int(len(edges_sorted) * top_percent))
    if top_percent == 1:
        keep_n = len(edges_sorted)

    filtered = nx.Graph()
    for product_1, product_2, data in edges_sorted[:keep_n]:
        filtered.add_edge(product_1, product_2, **data)

    # Isolated nodes remain part of the filtered graph so node coverage can be
    # reported even when none of their edges pass a strict percentage cutoff.
    filtered.add_nodes_from(graph.nodes(data=True))
    return filtered


def filtered_graph_to_dataframe(filtered_graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for product_1, product_2, data in filtered_graph.edges(data=True):
        rows.append(
            {
                "EdgeOrder": data.get("edge_order", len(rows)),
                "Product_1": product_1,
                "Product_2": product_2,
                "Short_1": get_short_name(product_1),
                "Short_2": get_short_name(product_2),
                "Lift": data.get("lift", math.nan),
                "Confidence": data.get("confidence", math.nan),
                "Support": data.get("support", math.nan),
                "Weight_Lift_x_Confidence": data["weight"],
                "RuleID": data.get("rule_id", ""),
            }
        )
    return pd.DataFrame(rows, columns=FILTERED_COLUMNS).sort_values(
        ["Weight_Lift_x_Confidence", "EdgeOrder"],
        ascending=[False, True],
        kind="stable",
    )


def _write_dataframe_safely(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def export_filtered_graph(
    filtered_graph: nx.Graph,
    output_csv: str | Path = OUTPUT_CSV,
) -> pd.DataFrame:
    frame = filtered_graph_to_dataframe(filtered_graph)
    _write_dataframe_safely(frame, Path(output_csv))
    return frame


def get_isolated_nodes(filtered_graph: nx.Graph) -> list[str]:
    return list(nx.isolates(filtered_graph))


def run_filtering(args: argparse.Namespace) -> None:
    input_graph = args.input_graph.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_graph.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    percentages = normalize_percentages(args.percentages)
    total_start = perf_counter()

    step_start = perf_counter()
    graph = load_projected_graph(input_graph)
    load_seconds = perf_counter() - step_start

    summary_rows = []
    timing_rows = [{"Step": "Load projected graph", "Seconds": load_seconds}]
    for percentage in percentages:
        label = format_percent_label(percentage)
        step_start = perf_counter()
        filtered = build_filtered_graph_from_graph(graph, percentage)
        output_csv = output_dir / f"Filtered_Graph_{label}.csv"
        export_filtered_graph(filtered, output_csv)
        elapsed = perf_counter() - step_start

        isolated = get_isolated_nodes(filtered)
        active_nodes = filtered.number_of_nodes() - len(isolated)
        summary_rows.append(
            {
                "Percentage": percentage * 100,
                "TotalGraphNodes": graph.number_of_nodes(),
                "TotalGraphEdges": graph.number_of_edges(),
                "RetainedEdges": filtered.number_of_edges(),
                "ActiveNodes": active_nodes,
                "IsolatedNodes": len(isolated),
                "OutputFile": str(output_csv),
            }
        )
        timing_rows.append(
            {"Step": f"Filter/export top {label}%", "Seconds": elapsed}
        )

    total_seconds = perf_counter() - total_start
    timing_rows.append({"Step": "Total pipeline", "Seconds": total_seconds})

    summary = pd.DataFrame(summary_rows)
    timing = pd.DataFrame(timing_rows)
    timing["Seconds"] = timing["Seconds"].round(4)
    _write_dataframe_safely(summary, output_dir / "FILTERED_GRAPH_SUMMARY.csv")
    _write_dataframe_safely(timing, output_dir / "FILTERED_GRAPH_TIMING_SUMMARY.csv")

    print("Filtered-graph export completed.")
    print(summary.to_string(index=False))
    print(f"Total time: {total_seconds:.4f} s")


def main() -> None:
    run_filtering(parse_args())


if __name__ == "__main__":
    main()
