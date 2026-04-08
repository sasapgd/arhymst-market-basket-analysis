from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ============================================================
# MST COMPARISON ANALYSIS -> Compare Reduction Variants
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ==============================
# SETTINGS
# ==============================

VARIANT = "COMMON"  # Change to CONF / LIFT / PRODUCT / COMMON.

file_conf = BASE_DIR / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
file_lift = BASE_DIR / "Rules_For_Python_REDUCED_LIFT.csv"
file_prod = BASE_DIR / "Rules_For_Python_REDUCED_PRODUCT.csv"


# ==============================
# LOAD FILES
# ==============================

df_conf = pd.read_csv(file_conf, sep=";")
df_lift = pd.read_csv(file_lift, sep=";")
df_prod = pd.read_csv(file_prod, sep=";")


# ==============================
# RULE KEY
# ==============================

def create_rule_key(df):
    return df["Premises"].str.strip() + " -> " + df["Conclusion"].str.strip()


df_conf["RuleKey"] = create_rule_key(df_conf)
df_lift["RuleKey"] = create_rule_key(df_lift)
df_prod["RuleKey"] = create_rule_key(df_prod)


# ==============================
# COMMON-RULE INTERSECTION
# ==============================

common_rules = set(df_conf["RuleKey"]) \
    & set(df_lift["RuleKey"]) \
    & set(df_prod["RuleKey"])

df_common = df_conf[df_conf["RuleKey"].isin(common_rules)].copy()

print(f"Common rules: {len(df_common)}")


# ==============================
# SELECT DATASET
# ==============================

variant_key = VARIANT.strip().upper()
variant_frames = {
    "CONF": df_conf,
    "CONFIDENCE": df_conf,
    "LIFT": df_lift,
    "PRODUCT": df_prod,
    "COMMON": df_common,
}

if variant_key not in variant_frames:
    raise ValueError(
        f"Unsupported VARIANT '{VARIANT}'. Use CONF, LIFT, PRODUCT, or COMMON."
    )

selected_df = variant_frames[variant_key]


# ==============================
# GRAPH
# ==============================

def split_items(x):
    return [i.strip() for i in str(x).split(",") if i.strip()]


G = nx.Graph()

for _, row in selected_df.iterrows():
    premises = split_items(row["Premises"])
    conclusions = split_items(row["Conclusion"])

    weight = float(row["Lift"]) * float(row["Confidence"])

    for p in premises:
        for c in conclusions:
            if p == c:
                continue

            if G.has_edge(p, c):
                if G[p][c]["weight"] < weight:
                    G[p][c]["weight"] = weight
            else:
                G.add_edge(p, c, weight=weight)


# ==============================
# MST
# ==============================

mst = nx.maximum_spanning_tree(G, weight="weight")

print(f"MST nodes: {mst.number_of_nodes()}")
print(f"MST edges: {mst.number_of_edges()}")


# ==============================
# BFS HORIZONTAL LAYOUT
# ==============================

def bfs_layout_horizontal(tree):
    root = max(tree.degree, key=lambda x: x[1])[0]

    pos = {}
    levels = {}
    visited = set([root])
    queue = [(root, 0)]

    while queue:
        node, level = queue.pop(0)
        levels.setdefault(level, []).append(node)

        for neighbor in tree.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))

    for level, nodes in levels.items():
        height = len(nodes)
        for i, node in enumerate(nodes):
            x = level * 2
            y = -(i - height / 2)
            pos[node] = (x, y)

    return pos


pos = bfs_layout_horizontal(mst)


# ==============================
# NODE RANKING
# ==============================

degree = dict(mst.degree())
sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)

top5 = [n for n, _ in sorted_nodes[:5]]
next5 = [n for n, _ in sorted_nodes[5:10]]


# ==============================
# NODE COLORS
# ==============================

node_colors = []
node_sizes = []

for node in mst.nodes():
    if node in top5:
        node_colors.append("red")
        node_sizes.append(1200)
    elif node in next5:
        node_colors.append("orange")
        node_sizes.append(800)
    else:
        node_colors.append("lightblue")
        node_sizes.append(400)


# ==============================
# EDGE WIDTHS
# ==============================

weights = [mst[u][v]["weight"] for u, v in mst.edges()]
max_w = max(weights)

edge_widths = [1 + (w / max_w) * 3 for w in weights]


# ==============================
# DRAW
# ==============================

plt.figure(figsize=(16, 10))

nx.draw_networkx_nodes(mst, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_edges(mst, pos, width=edge_widths, alpha=0.7)
nx.draw_networkx_labels(mst, pos, font_size=8)

plt.title(f"MST ({VARIANT}) - BFS Layout", fontsize=14)
plt.axis("off")

plt.tight_layout()
plt.show()


# ==============================
# EXPORT CSV
# ==============================

# EDGES
edges_data = []

for u, v, data in mst.edges(data=True):
    edges_data.append({
        "Source": u,
        "Target": v,
        "Weight": data["weight"],
    })

df_edges = pd.DataFrame(edges_data)
edges_file = BASE_DIR / f"MST_{VARIANT}_EDGES.csv"
df_edges.to_csv(edges_file, index=False)

print(f"Saved edges: {edges_file}")


# NODES
nodes_data = []

for node, deg in mst.degree():
    nodes_data.append({
        "Node": node,
        "Degree": deg
    })

df_nodes = pd.DataFrame(nodes_data)
nodes_file = BASE_DIR / f"MST_{VARIANT}_NODES.csv"
df_nodes.to_csv(nodes_file, index=False)

print(f"Saved nodes: {nodes_file}")
