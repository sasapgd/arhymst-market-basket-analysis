"""Extract a Maximum Spanning Tree from a projected product graph."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from graph_utils import (
    BASE_DIR,
    DEFAULT_PRODUCT_GRAPH_FILE,
    build_full_graph_from_rules,
    load_projected_graph,
)


OUTPUT_FILE = BASE_DIR / "MST.csv"
MST_COLUMNS = (
    "Product_1",
    "Product_2",
    "Lift",
    "Confidence",
    "Weight_Lift_x_Confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a Maximum Spanning Tree from a product graph."
    )
    parser.add_argument(
        "--input-graph",
        type=Path,
        default=DEFAULT_PRODUCT_GRAPH_FILE,
        help="Projected product-graph CSV (default: PRODUCT_GRAPH.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="MaxST CSV (default: MST.csv beside the input graph).",
    )
    return parser.parse_args()


def build_mst_from_graph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot build a MaxST from an empty graph.")
    if not nx.is_connected(graph):
        components = nx.number_connected_components(graph)
        raise ValueError(
            f"Projected graph is disconnected ({components} components); "
            "a single spanning tree does not exist."
        )

    # Edge weights represent association strength, so the maximum—not minimum—
    # spanning tree retains the strongest globally connecting product links.
    return nx.maximum_spanning_tree(graph, weight="weight", algorithm="kruskal")


def mst_to_dataframe(mst: nx.Graph) -> pd.DataFrame:
    rows = []
    for product_1, product_2, data in mst.edges(data=True):
        rows.append(
            {
                "Product_1": product_1,
                "Product_2": product_2,
                "Lift": data["lift"],
                "Confidence": data["confidence"],
                "Weight_Lift_x_Confidence": data["weight"],
            }
        )

    return pd.DataFrame(rows, columns=MST_COLUMNS).sort_values(
        "Weight_Lift_x_Confidence",
        ascending=False,
        kind="stable",
    )


def _write_dataframe_safely(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def export_mst(mst: nx.Graph, output_file: str | Path = OUTPUT_FILE) -> pd.DataFrame:
    frame = mst_to_dataframe(mst)
    _write_dataframe_safely(frame, Path(output_file))
    return frame


def build_mst_from_rules(
    filepath: str | Path,
    output_file: str | Path = OUTPUT_FILE,
) -> nx.Graph:
    """Compatibility helper for older experimental scripts.

    The reproducible main pipeline should use graph_utils.py first and pass its
    PRODUCT_GRAPH.csv output to this script.
    """
    graph = build_full_graph_from_rules(filepath)
    mst = build_mst_from_graph(graph)
    export_mst(mst, output_file)
    return mst


def run_mst(args: argparse.Namespace) -> None:
    input_graph = args.input_graph.expanduser().resolve()
    output_file = (
        args.output.expanduser().resolve()
        if args.output
        else input_graph.with_name("MST.csv")
    )
    timing_file = output_file.parent / "MST_TIMING_SUMMARY.csv"
    metadata_file = output_file.parent / "MST_RUN_METADATA.csv"
    total_start = perf_counter()

    step_start = perf_counter()
    graph = load_projected_graph(input_graph)
    load_seconds = perf_counter() - step_start

    step_start = perf_counter()
    mst = build_mst_from_graph(graph)
    mst_seconds = perf_counter() - step_start

    step_start = perf_counter()
    export_mst(mst, output_file)
    export_seconds = perf_counter() - step_start
    total_seconds = perf_counter() - total_start

    timing = pd.DataFrame(
        {
            "Step": ["Load projected graph", "Build MaxST", "Export MaxST", "Total pipeline"],
            "Seconds": [load_seconds, mst_seconds, export_seconds, total_seconds],
        }
    )
    timing["Seconds"] = timing["Seconds"].round(4)
    _write_dataframe_safely(timing, timing_file)

    total_weight = sum(data["weight"] for _, _, data in mst.edges(data=True))
    metadata = pd.DataFrame(
        {
            "Metric": [
                "Input graph",
                "Output MaxST",
                "Algorithm",
                "Edge weight",
                "Input graph nodes",
                "Input graph edges",
                "MaxST nodes",
                "MaxST edges",
                "MaxST total weight",
                "Python version",
                "pandas version",
                "NetworkX version",
            ],
            "Value": [
                str(input_graph),
                str(output_file),
                "Kruskal maximum spanning tree",
                "Lift x Confidence",
                str(graph.number_of_nodes()),
                str(graph.number_of_edges()),
                str(mst.number_of_nodes()),
                str(mst.number_of_edges()),
                f"{total_weight:.12g}",
                platform.python_version(),
                pd.__version__,
                nx.__version__,
            ],
        }
    )
    _write_dataframe_safely(metadata, metadata_file)

    print("Maximum Spanning Tree extraction completed.")
    print(f"Input graph nodes: {graph.number_of_nodes()}")
    print(f"Input graph edges: {graph.number_of_edges()}")
    print(f"MaxST nodes: {mst.number_of_nodes()}")
    print(f"MaxST edges: {mst.number_of_edges()}")
    print(f"MaxST saved to: {output_file}")
    for row in timing.itertuples(index=False):
        print(f"{row.Step}: {row.Seconds:.4f} s")


def main() -> None:
    run_mst(parse_args())


if __name__ == "__main__":
    main()
