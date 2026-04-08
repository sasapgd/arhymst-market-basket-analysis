# ============================================================
# POST-REDUCTION NETWORK ANALYSIS -> Export and Visualize Graphs
# ============================================================

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from filtered_graph import (
    build_filtered_graph_from_graph,
    export_filtered_graph,
    get_isolated_nodes,
)
from graph_utils import (
    BASE_DIR,
    DEFAULT_REDUCED_RULES_FILE,
    build_full_graph_from_rules,
    get_short_name,
)
from mst_network_analysis import build_mst_from_graph, export_mst


# ==============================
# SETTINGS
# ==============================

INPUT_FILE = DEFAULT_REDUCED_RULES_FILE
MST_OUTPUT_CSV = BASE_DIR / "MST.csv"
MST_OUTPUT_IMAGE = BASE_DIR / "MST.png"
FILTER_START_PERCENT = 0.20
FILTER_END_PERCENT = 0.30
FILTER_STEP_PERCENT = 0.05

TOP_HUBS_TO_HIGHLIGHT = 5
EDGE_LABEL_OFFSET = 0.035
TOP_HUBS_RED = 5
TOP_HUBS_ORANGE = 5
TOP_LABEL_EDGES = 10


# ==============================
# DRAW EDGE LABELS
# ==============================

def draw_edge_weight_labels_above(ax, graph, pos, offset=EDGE_LABEL_OFFSET):
    for u, v, data in graph.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length == 0:
            ox, oy = 0, offset
        else:
            ox = -dy / length * offset
            oy = dx / length * offset

        ax.text(
            mx + ox,
            my + oy,
            f"{data['weight']:.2f}",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
            zorder=5,
        )


# ==============================
# FILTER SETTINGS HELPERS
# ==============================

def build_filter_percentages(
    start_percent: float = FILTER_START_PERCENT,
    end_percent: float = FILTER_END_PERCENT,
    step_percent: float = FILTER_STEP_PERCENT,
) -> list[float]:
    percentages = []
    current = start_percent

    while current <= end_percent + 1e-9:
        percentages.append(round(current, 4))
        current += step_percent

    return percentages


def format_percent_label(top_percent: float) -> str:
    percent_value = top_percent * 100

    if float(percent_value).is_integer():
        return str(int(percent_value))

    return f"{percent_value:.1f}".replace(".", "_")


# ==============================
# MST VISUALIZATION
# ==============================

def visualize_mst_bfs(graph: nx.Graph, output_image: str | Path = MST_OUTPUT_IMAGE):
    degree_dict = dict(graph.degree())
    sorted_degree = sorted(
        degree_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    root = sorted_degree[0][0]
    top_hubs = [node for node, _ in sorted_degree[:TOP_HUBS_TO_HIGHLIGHT]]

    pos = nx.bfs_layout(graph, start=root)
    fig, ax = plt.subplots(figsize=(14, 10))

    node_sizes = [400 + degree_dict[node] * 250 for node in graph.nodes()]
    node_colors = [
        "red" if node in top_hubs else "skyblue"
        for node in graph.nodes()
    ]
    labels = {node: get_short_name(node) for node in graph.nodes()}

    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    max_weight = max(weights)
    edge_widths = [0.6 + (weight / max_weight) * 1.4 for weight in weights]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        ax=ax,
    )

    draw_edge_weight_labels_above(ax, graph, pos)

    ax.set_title("Maximum Spanning Tree of Product Associations\n(Hierarchical BFS Layout)")
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.show()


# ==============================
# FILTERED GRAPH VISUALIZATION
# ==============================

def draw_filtered_graph(
    filtered_graph: nx.Graph,
    top_percent: float,
    output_png: str | Path,
) -> None:
    graph_to_draw = filtered_graph.copy()
    graph_to_draw.remove_nodes_from(get_isolated_nodes(filtered_graph))

    if graph_to_draw.number_of_nodes() == 0:
        return

    degree_dict = dict(graph_to_draw.degree())
    sorted_degree = sorted(
        degree_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_red = [node for node, _ in sorted_degree[:TOP_HUBS_RED]]
    top_orange = [
        node
        for node, _ in sorted_degree[TOP_HUBS_RED:TOP_HUBS_RED + TOP_HUBS_ORANGE]
    ]

    pos = nx.spring_layout(
        graph_to_draw,
        weight="weight",
        k=2.5,
        iterations=300,
        seed=42,
    )

    plt.figure(figsize=(20, 15))

    weights = [graph_to_draw[u][v]["weight"] for u, v in graph_to_draw.edges()]
    max_weight = max(weights) if weights else 1.0
    edge_widths = [0.5 + (weight / max_weight) * 2 for weight in weights]

    max_degree = max(degree_dict.values()) if degree_dict else 1
    node_sizes = [
        600 + (degree_dict[node] / max_degree) * 1500
        for node in graph_to_draw.nodes()
    ]

    node_colors = []
    for node in graph_to_draw.nodes():
        if node in top_red:
            node_colors.append("red")
        elif node in top_orange:
            node_colors.append("orange")
        else:
            node_colors.append("skyblue")

    labels = {node: get_short_name(node) for node in graph_to_draw.nodes()}

    nx.draw_networkx_edges(graph_to_draw, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_nodes(
        graph_to_draw,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        graph_to_draw,
        pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
    )

    edges_sorted = sorted(
        graph_to_draw.edges(data=True),
        key=lambda edge: edge[2]["weight"],
        reverse=True,
    )
    top_edges = edges_sorted[:TOP_LABEL_EDGES]
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in top_edges}

    nx.draw_networkx_edge_labels(
        graph_to_draw,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="black",
        label_pos=0.5,
    )

    plt.title(
        f"Filtered Product Network (Top {format_percent_label(top_percent)}% Edges)\n"
        f"Edge labels shown for top {TOP_LABEL_EDGES} strongest connections"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


# ==============================
# ENTRY POINT
# ==============================

def main():
    full_graph = build_full_graph_from_rules(INPUT_FILE)

    mst = build_mst_from_graph(full_graph)
    export_mst(mst, MST_OUTPUT_CSV)
    visualize_mst_bfs(mst, MST_OUTPUT_IMAGE)

    for top_percent in build_filter_percentages():
        percent_label = format_percent_label(top_percent)
        filtered_output_csv = BASE_DIR / f"Filtered_Graph_{percent_label}.csv"
        filtered_output_image = BASE_DIR / f"Filtered_Graph_{percent_label}.png"

        filtered_graph = build_filtered_graph_from_graph(full_graph, top_percent)
        export_filtered_graph(filtered_graph, filtered_output_csv)
        draw_filtered_graph(filtered_graph, top_percent, filtered_output_image)


if __name__ == "__main__":
    main()
