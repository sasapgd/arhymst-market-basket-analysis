# ============================================================
# FILTERED GRAPH -> Keep Strongest Edges
# ============================================================

import networkx as nx
import pandas as pd

from graph_utils import BASE_DIR, DEFAULT_REDUCED_RULES_FILE, get_short_name


# ==============================
# SETTINGS
# ==============================

INPUT_FILE = DEFAULT_REDUCED_RULES_FILE
OUTPUT_CSV = BASE_DIR / "Filtered_Graph.csv"


# ==============================
# FILTER THE STRONGEST EDGES
# ==============================

def build_filtered_graph_from_graph(graph: nx.Graph, top_percent: float) -> nx.Graph:
    edges_sorted = sorted(
        graph.edges(data=True),
        key=lambda edge: edge[2]["weight"],
        reverse=True,
    )

    keep_n = max(1, int(len(edges_sorted) * top_percent))
    filtered_graph = nx.Graph()

    for u, v, data in edges_sorted[:keep_n]:
        filtered_graph.add_edge(u, v, **data)

    for node in graph.nodes():
        if node not in filtered_graph:
            filtered_graph.add_node(node)

    return filtered_graph


# ==============================
# CSV EXPORT
# ==============================

def export_filtered_graph(filtered_graph: nx.Graph, output_csv=OUTPUT_CSV) -> pd.DataFrame:
    rows = []

    for u, v, data in filtered_graph.edges(data=True):
        rows.append(
            {
                "Product_1": u,
                "Product_2": v,
                "Short_1": get_short_name(u),
                "Short_2": get_short_name(v),
                "Lift": data.get("lift"),
                "Confidence": data.get("confidence"),
                "Weight_Lift_x_Confidence": data.get("weight"),
            }
        )

    export_df = pd.DataFrame(rows).sort_values(
        by="Weight_Lift_x_Confidence",
        ascending=False,
    )
    export_df.to_csv(output_csv, sep=";", index=False)
    return export_df


# ==============================
# ISOLATED NODES
# ==============================

def get_isolated_nodes(filtered_graph: nx.Graph) -> list[str]:
    return list(nx.isolates(filtered_graph))
