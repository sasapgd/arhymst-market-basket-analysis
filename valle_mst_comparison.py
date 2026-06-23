"""Compare the rule-weighted MaxST with a Valle-type correlation MST.

The Valle tree is built directly from binary basket/category occurrences:

    phi(i, j) = Pearson correlation of two binary purchase vectors
    distance(i, j) = sqrt(2 * (1 - phi(i, j)))

The script does not rerun Apriori.  It reuses the existing maxlen-specific
Lift x Confidence MaxST files and scans the anonymized transactions only once
to create a compact table of pairwise 2 x 2 counts.  Subsequent runs reuse that
table unless --rebuild-pairs is supplied.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
from itertools import combinations
from pathlib import Path
from time import perf_counter

import networkx as nx
import polars as pl

from itemset_utils import parse_itemset


BASE_DIR = Path(__file__).resolve().parent
MAXLENS = (3, 4, 5, 6)
BASKET_KEYS = ("Person", "Date", "Channel")
EXPECTED_N_TRANSACTIONS = 2_636_756


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Valle phi-distance MST and compare it with Lift x Confidence MaxSTs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "Data",
        help="Folder containing anonymized transaction CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "valle_mst_comparison",
        help="Folder for pair counts, trees, and comparison tables.",
    )
    parser.add_argument(
        "--rebuild-pairs",
        action="store_true",
        help="Recompute pair counts even if the compact pair-count file exists.",
    )
    parser.add_argument(
        "--keep-basket-cache",
        action="store_true",
        help="Keep the temporary deduplicated basket-item Parquet file.",
    )
    parser.add_argument(
        "--maxlen", nargs="+", type=int, default=list(MAXLENS),
        help="One or more MaxST maxlen variants (default: 3 4 5 6).",
    )
    parser.add_argument(
        "--maxst-dir",
        type=Path,
        default=BASE_DIR / "timing_runs" / "mst_variants",
        help="Directory containing MST_MAXLEN_<N>_CONFIDENCE.csv files.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def canonical_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_pair_counts(data_dir: Path, output_dir: Path, keep_cache: bool) -> tuple[int, list[dict]]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    cache_path = output_dir / "_basket_items_cache.parquet"
    print(f"Scanning {len(files)} transaction files...", flush=True)
    raw = pl.scan_csv(
        [str(path) for path in files],
        infer_schema_length=0,
        schema_overrides={
            "PERSON_PUBLIC_KEY": pl.String,
            "DATE": pl.String,
            "CHANNEL": pl.String,
            "PRODUCT_CATEGORY": pl.String,
        },
    )
    basket_items = (
        raw.select(
            pl.col("PERSON_PUBLIC_KEY").str.strip_chars().alias("Person"),
            pl.col("DATE").str.strip_chars().alias("Date"),
            pl.col("CHANNEL").str.strip_chars().str.to_uppercase().alias("Channel"),
            pl.col("PRODUCT_CATEGORY").str.strip_chars().alias("Product"),
        )
        .filter(
            pl.all_horizontal(
                pl.col("Person").is_not_null() & (pl.col("Person") != ""),
                pl.col("Date").is_not_null() & (pl.col("Date") != ""),
                pl.col("Product").is_not_null() & (pl.col("Product") != ""),
            )
        )
        .with_columns(
            pl.when(pl.col("Channel").is_null() | (pl.col("Channel") == ""))
            .then(pl.lit("UNKNOWN"))
            .otherwise(pl.col("Channel"))
            .alias("Channel")
        )
        .unique(subset=[*BASKET_KEYS, "Product"])
    )
    basket_items.sink_parquet(cache_path, compression="zstd", mkdir=True)

    cached = pl.scan_parquet(cache_path)
    n_transactions = (
        cached.select(*BASKET_KEYS)
        .unique()
        .select(pl.len().alias("N"))
        .collect(engine="streaming")
        .item()
    )
    item_counts_df = (
        cached.group_by("Product")
        .agg(pl.len().alias("ItemCount"))
        .sort("Product")
        .collect(engine="streaming")
    )

    left = cached.rename({"Product": "Product_1"})
    right = cached.rename({"Product": "Product_2"})
    observed_pairs_df = (
        left.join(right, on=list(BASKET_KEYS), how="inner")
        .filter(pl.col("Product_1") < pl.col("Product_2"))
        .group_by("Product_1", "Product_2")
        .agg(pl.len().alias("n11"))
        .sort("Product_1", "Product_2")
        .collect(engine="streaming")
    )

    item_counts = {
        row["Product"]: int(row["ItemCount"])
        for row in item_counts_df.iter_rows(named=True)
    }
    observed_pairs = {
        (row["Product_1"], row["Product_2"]): int(row["n11"])
        for row in observed_pairs_df.iter_rows(named=True)
    }

    pair_rows = []
    for product_1, product_2 in combinations(sorted(item_counts), 2):
        n11 = observed_pairs.get((product_1, product_2), 0)
        n10 = item_counts[product_1] - n11
        n01 = item_counts[product_2] - n11
        n00 = n_transactions - n11 - n10 - n01
        pair_rows.append(make_pair_row(product_1, product_2, n11, n10, n01, n00))

    if not keep_cache:
        cache_path.unlink(missing_ok=True)
    return int(n_transactions), pair_rows


def make_pair_row(product_1: str, product_2: str, n11: int, n10: int, n01: int, n00: int) -> dict:
    denominator = math.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    phi = ((n11 * n00) - (n10 * n01)) / denominator if denominator else 0.0
    phi = min(1.0, max(-1.0, phi))
    distance = math.sqrt(2.0 * (1.0 - phi))
    return {
        "Product_1": product_1,
        "Product_2": product_2,
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "Phi": phi,
        "ValleDistance": distance,
    }


def load_or_build_pairs(args: argparse.Namespace) -> tuple[int, list[dict]]:
    pair_path = args.output_dir / "VALLE_PAIR_COUNTS.csv"
    metadata_path = args.output_dir / "VALLE_METADATA.csv"
    if pair_path.exists() and metadata_path.exists() and not args.rebuild_pairs:
        metadata = read_csv(metadata_path)[0]
        rows = read_csv(pair_path)
        for row in rows:
            for column in ("n11", "n10", "n01", "n00"):
                row[column] = int(row[column])
            for column in ("Phi", "ValleDistance"):
                row[column] = float(row[column])
        print(f"Reusing compact pair counts: {pair_path}", flush=True)
        return int(metadata["NTransactions"]), rows

    n_transactions, rows = build_pair_counts(
        args.data_dir.resolve(), args.output_dir, args.keep_basket_cache
    )
    write_csv(
        pair_path,
        ["Product_1", "Product_2", "n11", "n10", "n01", "n00", "Phi", "ValleDistance"],
        rows,
    )
    write_csv(
        metadata_path,
        ["NTransactions", "Products", "Pairs", "ExpectedNTransactions", "MatchesExpectedN"],
        [{
            "NTransactions": n_transactions,
            "Products": len({row["Product_1"] for row in rows} | {row["Product_2"] for row in rows}),
            "Pairs": len(rows),
            "ExpectedNTransactions": EXPECTED_N_TRANSACTIONS,
            "MatchesExpectedN": n_transactions == EXPECTED_N_TRANSACTIONS,
        }],
    )
    return n_transactions, rows


def build_valle_graph(pair_rows: list[dict], nodes: set[str] | None = None) -> nx.Graph:
    graph = nx.Graph()
    all_nodes = sorted(
        {row["Product_1"] for row in pair_rows} | {row["Product_2"] for row in pair_rows}
    )
    selected_nodes = set(all_nodes) if nodes is None else set(nodes)
    graph.add_nodes_from(sorted(selected_nodes))
    for row in pair_rows:
        a, b = row["Product_1"], row["Product_2"]
        if a in selected_nodes and b in selected_nodes:
            graph.add_edge(
                a,
                b,
                distance=float(row["ValleDistance"]),
                phi=float(row["Phi"]),
                n11=int(row["n11"]),
                n10=int(row["n10"]),
                n01=int(row["n01"]),
                n00=int(row["n00"]),
            )
    return graph


def minimum_tree(graph: nx.Graph) -> nx.Graph:
    return nx.minimum_spanning_tree(graph, weight="distance", algorithm="kruskal")


def export_lift_confidence_tree(tree: nx.Graph, path: Path) -> None:
    rows = []
    for a, b, data in sorted(tree.edges(data=True), key=lambda item: item[2]["weight"], reverse=True):
        rows.append({
            "Product_1": a,
            "Product_2": b,
            "Lift": data["lift"],
            "Confidence": data["confidence"],
            "Weight_Lift_x_Confidence": data["weight"],
        })
    write_csv(
        path,
        ["Product_1", "Product_2", "Lift", "Confidence", "Weight_Lift_x_Confidence"],
        rows,
    )


def load_our_maxst(maxlen: int, product_names: set[str], output_dir: Path, maxst_dir: Path) -> nx.Graph:
    """Load the exact MaxST emitted by the reproducible main pipeline."""
    path = maxst_dir / f"MST_MAXLEN_{maxlen}_CONFIDENCE.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Lift x Confidence MaxST file: {path}")
    graph = nx.Graph()
    for row in read_csv(path):
        premise = row["Product_1"].strip()
        conclusion = row["Product_2"].strip()
        if premise not in product_names or conclusion not in product_names:
            raise ValueError(f"Unknown product in {path}: {premise}, {conclusion}")
        lift = float(row["Lift"].replace(",", "."))
        confidence = float(row["Confidence"].replace(",", "."))
        weight = float(row.get("Weight_Lift_x_Confidence", lift * confidence))
        graph.add_edge(premise, conclusion, lift=lift, confidence=confidence, weight=weight)
    if not nx.is_tree(graph):
        raise ValueError(f"Lift x Confidence input for maxlen={maxlen} is not a tree")
    export_lift_confidence_tree(
        graph, output_dir / f"LIFT_CONFIDENCE_MAXST_MAXLEN_{maxlen}.csv"
    )
    return graph


def edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {canonical_edge(a, b) for a, b in graph.edges()}


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def pearson(values_x: list[float], values_y: list[float]) -> float:
    if len(values_x) != len(values_y) or not values_x:
        return float("nan")
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in values_x)
        * sum((y - mean_y) ** 2 for y in values_y)
    )
    return numerator / denominator if denominator else float("nan")


def rank_correlation(tree_a: nx.Graph, tree_b: nx.Graph, nodes: list[str]) -> float:
    ranks_a = average_ranks({node: tree_a.degree(node) for node in nodes})
    ranks_b = average_ranks({node: tree_b.degree(node) for node in nodes})
    return pearson([ranks_a[node] for node in nodes], [ranks_b[node] for node in nodes])


def path_correlation(tree_a: nx.Graph, tree_b: nx.Graph, nodes: list[str]) -> float:
    distances_a = dict(nx.all_pairs_shortest_path_length(tree_a))
    distances_b = dict(nx.all_pairs_shortest_path_length(tree_b))
    values_a, values_b = [], []
    for a, b in combinations(nodes, 2):
        values_a.append(float(distances_a[a][b]))
        values_b.append(float(distances_b[a][b]))
    return pearson(values_a, values_b)


def top_nodes(tree: nx.Graph, count: int = 5) -> list[str]:
    return [
        node for node, _ in sorted(tree.degree(), key=lambda item: (-item[1], item[0]))[:count]
    ]


def export_valle_tree(tree: nx.Graph, path: Path) -> None:
    rows = []
    for a, b, data in sorted(tree.edges(data=True), key=lambda item: item[2]["distance"]):
        rows.append({
            "Product_1": a,
            "Product_2": b,
            "Phi": data["phi"],
            "ValleDistance": data["distance"],
            "n11": data["n11"],
            "n10": data["n10"],
            "n01": data["n01"],
            "n00": data["n00"],
        })
    write_csv(
        path,
        ["Product_1", "Product_2", "Phi", "ValleDistance", "n11", "n10", "n01", "n00"],
        rows,
    )


def compare_trees(maxlen: int, our_tree: nx.Graph, pair_rows: list[dict], output_dir: Path) -> dict:
    all_valle_nodes = {
        row["Product_1"] for row in pair_rows
    } | {row["Product_2"] for row in pair_rows}
    common_nodes = sorted(set(our_tree.nodes()) & all_valle_nodes)
    valle_common = minimum_tree(build_valle_graph(pair_rows, set(common_nodes)))
    our_edges = edge_set(our_tree)
    valle_edges = edge_set(valle_common)
    intersection = our_edges & valle_edges
    union = our_edges | valle_edges
    pair_lookup = {
        canonical_edge(row["Product_1"], row["Product_2"]): row for row in pair_rows
    }

    edge_rows = []
    for edge in sorted(union):
        pair = pair_lookup[edge]
        our_data = our_tree.get_edge_data(*edge) or {}
        edge_rows.append({
            "Product_1": edge[0],
            "Product_2": edge[1],
            "InValleCommonNodeMST": edge in valle_edges,
            "InLiftConfidenceMaxST": edge in our_edges,
            "Phi": pair["Phi"],
            "ValleDistance": pair["ValleDistance"],
            "Lift": our_data.get("lift", ""),
            "Confidence": our_data.get("confidence", ""),
            "LiftXConfidence": our_data.get("weight", ""),
        })
    write_csv(
        output_dir / f"EDGE_COMPARISON_MAXLEN_{maxlen}.csv",
        [
            "Product_1", "Product_2", "InValleCommonNodeMST", "InLiftConfidenceMaxST",
            "Phi", "ValleDistance", "Lift", "Confidence", "LiftXConfidence",
        ],
        edge_rows,
    )

    ranks_valle = average_ranks({node: valle_common.degree(node) for node in common_nodes})
    ranks_ours = average_ranks({node: our_tree.degree(node) for node in common_nodes})
    hub_rows = [{
        "Product": node,
        "ValleDegree": valle_common.degree(node),
        "ValleDegreeRank": ranks_valle[node],
        "LiftConfidenceDegree": our_tree.degree(node),
        "LiftConfidenceDegreeRank": ranks_ours[node],
    } for node in common_nodes]
    hub_rows.sort(key=lambda row: (row["ValleDegreeRank"], row["LiftConfidenceDegreeRank"], row["Product"]))
    write_csv(
        output_dir / f"HUB_COMPARISON_MAXLEN_{maxlen}.csv",
        ["Product", "ValleDegree", "ValleDegreeRank", "LiftConfidenceDegree", "LiftConfidenceDegreeRank"],
        hub_rows,
    )

    valle_top = top_nodes(valle_common)
    our_top = top_nodes(our_tree)
    return {
        "Maxlen": maxlen,
        "ValleAllProducts": len(all_valle_nodes),
        "OurMaxSTProducts": our_tree.number_of_nodes(),
        "CommonProducts": len(common_nodes),
        "EdgesPerCommonNodeTree": len(common_nodes) - 1,
        "CommonEdges": len(intersection),
        "EdgeOverlapPercent": 100.0 * len(intersection) / max(1, len(our_edges)),
        "EdgeJaccard": len(intersection) / max(1, len(union)),
        "DegreeRankCorrelation": rank_correlation(valle_common, our_tree, common_nodes),
        "TreePathLengthCorrelation": path_correlation(valle_common, our_tree, common_nodes),
        "Top5HubOverlap": len(set(valle_top) & set(our_top)),
        "ValleTopHub": valle_top[0],
        "LiftConfidenceTopHub": our_top[0],
        "ValleTop5Hubs": " | ".join(valle_top),
        "LiftConfidenceTop5Hubs": " | ".join(our_top),
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.maxst_dir = args.maxst_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    n_transactions, pair_rows = load_or_build_pairs(args)
    valle_full = minimum_tree(build_valle_graph(pair_rows))
    export_valle_tree(valle_full, args.output_dir / "VALLE_FULL_MST.csv")

    summary_rows = []
    product_names = set(valle_full.nodes())
    for maxlen in args.maxlen:
        print(f"Comparing maxlen={maxlen}...", flush=True)
        summary_rows.append(
            compare_trees(
                maxlen,
                load_our_maxst(maxlen, product_names, args.output_dir, args.maxst_dir),
                pair_rows,
                args.output_dir,
            )
        )

    summary_fields = [
        "Maxlen", "ValleAllProducts", "OurMaxSTProducts", "CommonProducts",
        "EdgesPerCommonNodeTree", "CommonEdges", "EdgeOverlapPercent", "EdgeJaccard",
        "DegreeRankCorrelation", "TreePathLengthCorrelation", "Top5HubOverlap",
        "ValleTopHub", "LiftConfidenceTopHub", "ValleTop5Hubs", "LiftConfidenceTop5Hubs",
    ]
    write_csv(args.output_dir / "VALLE_VS_LIFT_CONFIDENCE_SUMMARY.csv", summary_fields, summary_rows)

    elapsed = perf_counter() - start
    write_csv(
        args.output_dir / "VALLE_RUN_METADATA.csv",
        ["Metric", "Value"],
        [
            {"Metric": "Data directory", "Value": args.data_dir},
            {"Metric": "MaxST directory", "Value": args.maxst_dir},
            {"Metric": "Maxlen values", "Value": ", ".join(map(str, args.maxlen))},
            {"Metric": "Transactions", "Value": n_transactions},
            {"Metric": "Products", "Value": valle_full.number_of_nodes()},
            {"Metric": "Valle distance", "Value": "sqrt(2 * (1 - phi))"},
            {"Metric": "Python version", "Value": platform.python_version()},
            {"Metric": "NetworkX version", "Value": nx.__version__},
            {"Metric": "Polars version", "Value": pl.__version__},
        ],
    )
    write_csv(
        args.output_dir / "VALLE_TIMING.csv",
        ["Step", "Seconds"],
        [{"Step": "Total pipeline", "Seconds": f"{elapsed:.4f}"}],
    )

    print(f"N transactions: {n_transactions}")
    print(f"Products in Valle tree: {valle_full.number_of_nodes()}")
    print(f"Valle MST edges: {valle_full.number_of_edges()}")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
