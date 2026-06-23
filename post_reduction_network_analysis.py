"""Create manuscript-ready network figures from existing graph and MaxST CSVs.

This stage performs visualization and MaxST centrality analysis only. Graph
projection, MaxST extraction, and edge filtering remain separate reproducible
steps and are not silently repeated here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from filtered_graph import format_percent_label, normalize_percentages
from graph_utils import (
    BASE_DIR,
    DEFAULT_PRODUCT_GRAPH_FILE,
    get_short_name,
    load_projected_graph,
)


DEFAULT_MST_FILE = BASE_DIR / "MST.csv"
TOP_HUBS_RED = 5
TOP_HUBS_ORANGE = 5
EDGE_LABEL_OFFSET = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an existing MaxST and filtered product graphs."
    )
    parser.add_argument(
        "--product-graph",
        type=Path,
        default=DEFAULT_PRODUCT_GRAPH_FILE,
        help="Projected graph CSV (used for metadata validation).",
    )
    parser.add_argument(
        "--mst",
        type=Path,
        help="MaxST CSV (default: MST.csv beside the product graph).",
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        help="Directory containing Filtered_Graph_PERCENT.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Figure/output directory (default: product-graph directory).",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[100.0],
        help="Filtered graph percentages to visualize (default: 100).",
    )
    parser.add_argument(
        "--top-edge-labels",
        type=int,
        default=5,
        help="Strongest filtered edges to label (default: 5).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Image DPI (default: 600).")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_edge_labels < 0:
        raise ValueError("--top-edge-labels must be greater than or equal to 0.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")


def _write_dataframe_safely(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def draw_edge_weight_labels_above(
    axis,
    graph: nx.Graph,
    positions: dict,
    offset: float = EDGE_LABEL_OFFSET,
) -> None:
    for product_1, product_2, data in graph.edges(data=True):
        x1, y1 = positions[product_1]
        x2, y2 = positions[product_2]
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        delta_x = x2 - x1
        delta_y = y2 - y1
        length = math.hypot(delta_x, delta_y)
        offset_x, offset_y = (0, offset) if length == 0 else (
            -delta_y / length * offset,
            delta_x / length * offset,
        )
        axis.text(
            midpoint_x + offset_x,
            midpoint_y + offset_y,
            f"{data['weight']:.2f}",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
            zorder=5,
        )


def ranked_degrees(graph: nx.Graph) -> list[tuple[str, int]]:
    return sorted(graph.degree(), key=lambda item: (-item[1], item[0]))


def visualize_mst_bfs(
    graph: nx.Graph,
    output_image: str | Path,
    dpi: int = 600,
) -> None:
    if not nx.is_tree(graph):
        raise ValueError("MST visualization input must be a connected tree.")

    degree = dict(graph.degree())
    degree_ranking = ranked_degrees(graph)
    root = degree_ranking[0][0]
    top_hubs = {node for node, _ in degree_ranking[:TOP_HUBS_RED]}
    positions = nx.bfs_layout(graph, start=root)

    figure, axis = plt.subplots(figsize=(14, 10))
    node_sizes = [400 + degree[node] * 250 for node in graph.nodes()]
    node_colors = ["red" if node in top_hubs else "skyblue" for node in graph.nodes()]
    labels = {node: get_short_name(node) for node in graph.nodes()}
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    max_weight = max(weights)
    edge_widths = [0.6 + (weight / max_weight) * 1.4 for weight in weights]

    nx.draw_networkx_nodes(
        graph, positions, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=axis
    )
    nx.draw_networkx_edges(graph, positions, width=edge_widths, alpha=0.7, ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=axis)
    draw_edge_weight_labels_above(axis, graph, positions)
    axis.set_title("Maximum Spanning Tree of Product Associations")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_image, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def export_mst_centrality(mst: nx.Graph, output_file: str | Path) -> pd.DataFrame:
    degree = dict(mst.degree())
    degree_centrality = nx.degree_centrality(mst)
    # Weights represent strength rather than distance, so manuscript
    # betweenness is calculated on the unweighted MaxST topology.
    betweenness = nx.betweenness_centrality(mst, normalized=True, weight=None)
    rows = [
        {
            "Product": node,
            "ShortName": get_short_name(node),
            "Degree": degree[node],
            "DegreeCentrality": degree_centrality[node],
            "BetweennessCentrality": betweenness[node],
        }
        for node in mst.nodes()
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["Degree", "BetweennessCentrality", "Product"],
        ascending=[False, False, True],
        kind="stable",
    )
    _write_dataframe_safely(frame, Path(output_file))
    return frame


def draw_filtered_graph(
    graph: nx.Graph,
    top_percent: float,
    output_png: str | Path,
    top_edge_labels: int = 5,
    dpi: int = 600,
) -> None:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Filtered graph must contain nodes and edges.")

    degree = dict(graph.degree())
    degree_ranking = ranked_degrees(graph)
    top_red = {node for node, _ in degree_ranking[:TOP_HUBS_RED]}
    top_orange = {
        node
        for node, _ in degree_ranking[TOP_HUBS_RED : TOP_HUBS_RED + TOP_HUBS_ORANGE]
    }
    positions = nx.spring_layout(
        graph,
        weight="weight",
        k=2.5,
        iterations=300,
        seed=42,
    )

    figure, axis = plt.subplots(figsize=(20, 15))
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    max_weight = max(weights)
    edge_widths = [0.5 + (weight / max_weight) * 2 for weight in weights]
    max_degree = max(degree.values())
    node_sizes = [600 + (degree[node] / max_degree) * 1500 for node in graph.nodes()]
    node_colors = [
        "red" if node in top_red else "orange" if node in top_orange else "skyblue"
        for node in graph.nodes()
    ]
    labels = {node: get_short_name(node) for node in graph.nodes()}

    nx.draw_networkx_edges(graph, positions, width=edge_widths, alpha=0.5, ax=axis)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=10,
        font_weight="bold",
        ax=axis,
    )

    strongest = sorted(
        graph.edges(data=True),
        key=lambda edge: (-edge[2]["weight"], edge[2].get("edge_order", 0)),
    )[:top_edge_labels]
    edge_labels = {
        (product_1, product_2): f"{data['weight']:.2f}"
        for product_1, product_2, data in strongest
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=8,
            font_color="black",
            label_pos=0.5,
            ax=axis,
        )

    axis.set_title(
        f"Filtered Product Network (Top {format_percent_label(top_percent)}% Edges)\n"
        f"Edge labels shown for the {top_edge_labels} strongest connections"
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> None:
    validate_args(args)
    product_graph_file = args.product_graph.expanduser().resolve()
    mst_file = (
        args.mst.expanduser().resolve()
        if args.mst
        else product_graph_file.with_name("MST.csv")
    )
    filtered_dir = (
        args.filtered_dir.expanduser().resolve()
        if args.filtered_dir
        else product_graph_file.parent
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else product_graph_file.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    percentages = normalize_percentages(args.percentages)
    total_start = perf_counter()
    timing_rows = []

    step_start = perf_counter()
    product_graph = load_projected_graph(product_graph_file)
    mst = load_projected_graph(mst_file)
    if set(mst.nodes()) != set(product_graph.nodes()):
        raise ValueError("MaxST and product graph do not contain the same nodes.")
    timing_rows.append({"Step": "Load graph and MaxST", "Seconds": perf_counter() - step_start})

    step_start = perf_counter()
    visualize_mst_bfs(mst, output_dir / "MST.png", dpi=args.dpi)
    centrality = export_mst_centrality(mst, output_dir / "MST_CENTRALITY.csv")
    timing_rows.append(
        {"Step": "MST figure and centrality", "Seconds": perf_counter() - step_start}
    )

    for percentage in percentages:
        label = format_percent_label(percentage)
        filtered_file = filtered_dir / f"Filtered_Graph_{label}.csv"
        step_start = perf_counter()
        filtered_graph = load_projected_graph(filtered_file)
        draw_filtered_graph(
            filtered_graph,
            percentage,
            output_dir / f"Filtered_Graph_{label}.png",
            top_edge_labels=args.top_edge_labels,
            dpi=args.dpi,
        )
        timing_rows.append(
            {"Step": f"Draw filtered graph top {label}%", "Seconds": perf_counter() - step_start}
        )

    timing_rows.append({"Step": "Total pipeline", "Seconds": perf_counter() - total_start})
    timing = pd.DataFrame(timing_rows)
    timing["Seconds"] = timing["Seconds"].round(4)
    _write_dataframe_safely(timing, output_dir / "NETWORK_VISUALIZATION_TIMING_SUMMARY.csv")

    metadata = pd.DataFrame(
        {
            "Metric": [
                "Product graph",
                "MaxST",
                "Visualized percentages",
                "Labeled filtered edges",
                "Product graph nodes",
                "Product graph edges",
                "MaxST nodes",
                "MaxST edges",
                "Top degree hub",
            ],
            "Value": [
                str(product_graph_file),
                str(mst_file),
                ", ".join(format_percent_label(value) for value in percentages),
                str(args.top_edge_labels),
                str(product_graph.number_of_nodes()),
                str(product_graph.number_of_edges()),
                str(mst.number_of_nodes()),
                str(mst.number_of_edges()),
                str(centrality.iloc[0]["Product"]),
            ],
        }
    )
    _write_dataframe_safely(metadata, output_dir / "NETWORK_VISUALIZATION_RUN_METADATA.csv")

    print("Network visualization completed.")
    print(f"MST image: {output_dir / 'MST.png'}")
    print(f"MST centrality: {output_dir / 'MST_CENTRALITY.csv'}")
    print(f"Top degree hub: {centrality.iloc[0]['Product']}")
    for row in timing.itertuples(index=False):
        print(f"{row.Step}: {row.Seconds:.4f} s")


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
