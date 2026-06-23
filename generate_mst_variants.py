"""Generate product graphs and MaxSTs for all rule-reduction criteria.

The reduction criterion selects which rules survive.  The graph edge weight is
always Lift x Confidence, exactly as defined in the manuscript.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from graph_utils import BASE_DIR, build_full_graph_from_rules, export_product_graph, get_short_name
from mst_network_analysis import build_mst_from_graph, export_mst


CRITERIA = ("confidence", "lift", "product")
DEFAULT_MAXLENS = (3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confidence-, lift-, and product-reduction MaxST variants."
    )
    parser.add_argument(
        "--maxlen",
        nargs="+",
        type=int,
        default=list(DEFAULT_MAXLENS),
        help="One or more Apriori maxlen values (default: 3 4 5 6).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=BASE_DIR / "timing_runs",
        help="Directory containing maxlen_<N> experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root (default: <runs-dir>/mst_variants).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation; CSV and comparison outputs are unaffected.",
    )
    parser.add_argument(
        "--dpi", type=int, default=600, help="PNG resolution (default: 600)."
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def _draw_mst(mst: nx.Graph, criterion: str, maxlen: int, output: Path, dpi: int) -> None:
    """Create a deterministic presentation figure; CSV is the analysis output."""
    degree = dict(mst.degree())
    hubs = {node for node, _ in sorted(degree.items(), key=lambda x: (-x[1], x[0]))[:5]}
    positions = nx.spring_layout(mst, weight="weight", seed=42, iterations=300)
    weights = [data["weight"] for _, _, data in mst.edges(data=True)]
    maximum = max(weights)

    fig, axis = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(
        mst,
        positions,
        node_size=[400 + 250 * degree[node] for node in mst.nodes()],
        node_color=["#d62728" if node in hubs else "#87ceeb" for node in mst.nodes()],
        alpha=0.9,
        ax=axis,
    )
    nx.draw_networkx_edges(
        mst,
        positions,
        width=[0.8 + 2.2 * weight / maximum for weight in weights],
        alpha=0.7,
        ax=axis,
    )
    nx.draw_networkx_labels(
        mst,
        positions,
        labels={node: get_short_name(node) for node in mst.nodes()},
        font_size=8,
        ax=axis,
    )
    axis.set_title(
        f"Maximum Spanning Tree - {criterion.upper()} reduction, maxlen={maxlen}\n"
        "Edge weight = Lift x Confidence"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_one(
    input_file: Path,
    output_dir: Path,
    criterion: str,
    maxlen: int,
    make_plot: bool,
    dpi: int,
) -> dict[str, object]:
    if not input_file.is_file():
        raise FileNotFoundError(f"Reduced-rules file not found: {input_file}")

    start = perf_counter()
    step = perf_counter()
    graph = build_full_graph_from_rules(input_file)
    graph_seconds = perf_counter() - step

    stem = f"MAXLEN_{maxlen}_{criterion.upper()}"
    graph_file = output_dir / f"PRODUCT_GRAPH_{stem}.csv"
    mst_file = output_dir / f"MST_{stem}.csv"
    png_file = output_dir / f"MST_{stem}.png"

    step = perf_counter()
    export_product_graph(graph, graph_file)
    graph_export_seconds = perf_counter() - step

    step = perf_counter()
    mst = build_mst_from_graph(graph)
    mst_seconds = perf_counter() - step

    step = perf_counter()
    export_mst(mst, mst_file)
    mst_export_seconds = perf_counter() - step

    plot_seconds = 0.0
    if make_plot:
        step = perf_counter()
        _draw_mst(mst, criterion, maxlen, png_file, dpi)
        plot_seconds = perf_counter() - step

    total_seconds = perf_counter() - start
    return {
        "Maxlen": maxlen,
        "ReductionCriterion": criterion,
        "InputRules": len(pd.read_csv(input_file, sep=";", usecols=["RuleID"])),
        "GraphNodes": graph.number_of_nodes(),
        "GraphEdges": graph.number_of_edges(),
        "MaxSTNodes": mst.number_of_nodes(),
        "MaxSTEdges": mst.number_of_edges(),
        "MaxSTTotalWeight": sum(data["weight"] for _, _, data in mst.edges(data=True)),
        "BuildGraphSeconds": graph_seconds,
        "ExportGraphSeconds": graph_export_seconds,
        "BuildMaxSTSeconds": mst_seconds,
        "ExportMaxSTSeconds": mst_export_seconds,
        "PlotSeconds": plot_seconds,
        "TotalSeconds": total_seconds,
        "InputFile": str(input_file),
        "GraphFile": str(graph_file),
        "MaxSTFile": str(mst_file),
    }


def main() -> None:
    args = parse_args()
    if any(value < 2 for value in args.maxlen):
        raise ValueError("Every maxlen value must be at least 2.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")

    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else runs_dir / "mst_variants"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for maxlen in args.maxlen:
        input_dir = runs_dir / f"maxlen_{maxlen}"
        for criterion in CRITERIA:
            input_file = input_dir / f"Rules_For_Python_REDUCED_{criterion.upper()}.csv"
            row = generate_one(
                input_file, output_dir, criterion, maxlen, not args.no_plots, args.dpi
            )
            rows.append(row)
            print(
                f"maxlen={maxlen}, {criterion}: "
                f"{row['GraphNodes']} nodes, {row['GraphEdges']} graph edges, "
                f"{row['MaxSTEdges']} MaxST edges, {row['TotalSeconds']:.4f} s"
            )

    summary = pd.DataFrame(rows)
    numeric_seconds = [column for column in summary if column.endswith("Seconds")]
    summary[numeric_seconds] = summary[numeric_seconds].round(4)
    summary["MaxSTTotalWeight"] = summary["MaxSTTotalWeight"].round(12)
    _write_csv(summary, output_dir / "MST_VARIANTS_SUMMARY.csv")

    metadata = pd.DataFrame(
        {
            "Metric": [
                "Reduction criteria",
                "Maxlen values",
                "Graph/MaxST edge weight",
                "MaxST algorithm",
                "Python version",
                "pandas version",
                "NetworkX version",
            ],
            "Value": [
                ", ".join(CRITERIA),
                ", ".join(map(str, args.maxlen)),
                "Lift x Confidence",
                "Kruskal maximum spanning tree",
                platform.python_version(),
                pd.__version__,
                nx.__version__,
            ],
        }
    )
    _write_csv(metadata, output_dir / "MST_VARIANTS_RUN_METADATA.csv")
    print(f"Summary: {output_dir / 'MST_VARIANTS_SUMMARY.csv'}")


if __name__ == "__main__":
    main()
