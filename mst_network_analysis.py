# ============================================================
# MST NETWORK ANALYSIS -> Build and Export MST
# ============================================================

import networkx as nx
import pandas as pd

from graph_utils import BASE_DIR, DEFAULT_REDUCED_RULES_FILE, build_full_graph_from_rules


# ==============================
# SETTINGS
# ==============================

INPUT_FILE = DEFAULT_REDUCED_RULES_FILE
OUTPUT_FILE = BASE_DIR / "Rules_For_Python_MST_FINAL.csv"


# ==============================
# MST COMPUTATION
# ==============================

def build_mst_from_graph(graph: nx.Graph) -> nx.Graph:
    return nx.maximum_spanning_tree(graph, weight="weight")


# ==============================
# CSV EXPORT
# ==============================

def export_mst(mst: nx.Graph, output_file=OUTPUT_FILE) -> pd.DataFrame:
    mst_data = []

    for u, v, data in mst.edges(data=True):
        mst_data.append(
            {
                "Product_1": u,
                "Product_2": v,
                "Lift": data["lift"],
                "Confidence": data["confidence"],
                "Weight_Lift_x_Confidence": data["weight"],
            }
        )

    mst_df = pd.DataFrame(mst_data).sort_values(
        by="Weight_Lift_x_Confidence",
        ascending=False,
    )
    mst_df.to_csv(output_file, sep=";", index=False)
    return mst_df


# ==============================
# FULL PIPELINE FOR MST ONLY
# ==============================

def build_mst_from_rules(filepath=INPUT_FILE, output_file=OUTPUT_FILE) -> nx.Graph:
    graph = build_full_graph_from_rules(filepath)
    mst = build_mst_from_graph(graph)
    export_mst(mst, output_file)
    return mst


if __name__ == "__main__":
    build_mst_from_rules(INPUT_FILE)
