"""Project reduced association rules into a weighted product graph.

Each product category becomes a node. Every antecedent-consequent product pair
becomes an undirected edge weighted by Lift x Confidence. If several rules
produce the same product pair, only the rule with the largest weight is kept.
"""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from itemset_utils import parse_itemset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REDUCED_RULES_FILE = BASE_DIR / "Rules_For_Python_REDUCED.csv"
DEFAULT_PRODUCT_GRAPH_FILE = BASE_DIR / "PRODUCT_GRAPH.csv"
REQUIRED_RULE_COLUMNS = ("Premises", "Conclusion", "Lift", "Confidence")
NUMERIC_RULE_COLUMNS = ("Lift", "Confidence", "Support")
PROJECTED_GRAPH_COLUMNS = (
    "EdgeOrder",
    "Product_1",
    "Product_2",
    "Lift",
    "Confidence",
    "Support",
    "Weight_Lift_x_Confidence",
    "RuleID",
)


# Short labels are presentation-only; full product names remain in all CSVs.
SHORT_NAME_MAP = {
    "BAKERY PRODUCTS": "BAKER",
    "BEAUTY AND PERSONAL CARE": "BEAUT",
    "BEER AND ALCOPOPS": "BEER",
    "BREAKFAST PRODUCTS": "BRKF",
    "CANNED FOODS": "CANNE",
    "CHEESE": "CHEES",
    "CHILDREN'S WORLD (KIDS' PRODUCTS)": "KIDS",
    "CONSUMER ELECTRONICS / TECHNOLOGY": "TECH",
    "DAIRY PRODUCTS": "DAIRY",
    "DECORATIVE AND GIFT PROGRAM": "DECOR",
    "DELICATESSEN - PROCESSED MEAT PRODUCTS": "DELI",
    "DRIED FRUITS AND VEGETABLES": "DRYFV",
    "ECONOMAT / STORE SUPPLIES": "ECONO",
    "FRESH FISH": "FISH",
    "FRESH MEAT (COUNTER / UNPACKAGED)": "MEAT",
    "FRESH PACKAGED MEAT": "PMEAT",
    "FROZEN PRODUCTS": "FROZE",
    "FRUITS": "FRUIT",
    "GASTRO PROGRAM (FOOD SERVICE PRODUCTS)": "GASTR",
    "HEALTH FOOD": "HEALT",
    "HOUSEHOLD ESSENTIALS AND CANDLES": "HOUSE",
    "HYGIENE AND PAPER PRODUCTS": "HYGI",
    "INGREDIENTS FOR MEAL PREPARATION": "MEAL",
    "KIOSK PRODUCTS": "KIOSK",
    "LAUNDRY AND CLEANING PRODUCTS": "CLEAN",
    "NON-ALCOHOLIC BEVERAGES": "DRINK",
    "PARTY SUPPLIES": "PARTY",
    "PET SUPPLIES": "PET",
    "POULTRY": "POULT",
    "SCHOOL, OFFICE AND MEDIA SUPPLIES": "SCHOM",
    "SPORTS AND LEISURE": "SPORT",
    "SWEETS AND SNACKS": "SNACK",
    "TABLEWARE / KITCHENWARE": "TABLE",
    "TOYS": "TOYS",
    "VEGETABLES": "VEG",
    "WINE AND SPIRITS": "WINE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project reduced association rules into a product graph."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_REDUCED_RULES_FILE,
        help="Reduced-rules CSV (default: Rules_For_Python_REDUCED.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Projected product-graph CSV (default: PRODUCT_GRAPH.csv beside input).",
    )
    return parser.parse_args()


def _read_csv_with_required_columns(
    path: Path, required_columns: tuple[str, ...]
) -> pd.DataFrame:
    errors: list[str] = []
    for separator in (";", ","):
        try:
            frame = pd.read_csv(path, sep=separator)
        except Exception as error:
            errors.append(f"separator {separator!r}: {error}")
            continue

        frame.columns = [str(column).strip() for column in frame.columns]
        if all(column in frame.columns for column in required_columns):
            return frame

    detail = "; ".join(errors) if errors else "required columns were not found"
    raise ValueError(f"Could not read CSV {path}: {detail}")


def load_rules_dataframe(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reduced-rules file not found: {path}")

    frame = _read_csv_with_required_columns(path, REQUIRED_RULE_COLUMNS)
    if frame.empty:
        raise ValueError("Reduced-rules file is empty.")

    for column in NUMERIC_RULE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if frame[column].isna().any():
                raise ValueError(f"Column {column} contains missing or invalid numbers.")

    if ((frame["Confidence"] < 0) | (frame["Confidence"] > 1)).any():
        raise ValueError("Confidence values must be between 0 and 1.")
    if (frame["Lift"] <= 0).any():
        raise ValueError("Lift values must be greater than 0.")

    return frame


def split_items(value: object) -> list[str]:
    return list(parse_itemset(value))


def get_short_name(product_name: str) -> str:
    return SHORT_NAME_MAP.get(product_name, product_name[:5].upper())


def build_product_graph(frame: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        try:
            premises = split_items(row.Premises)
            conclusions = split_items(row.Conclusion)
        except ValueError as error:
            raise ValueError(f"Cannot parse itemset on CSV row {row_number}: {error}") from error
        if not premises or not conclusions:
            raise ValueError(f"CSV row {row_number} contains an empty itemset.")

        lift = float(row.Lift)
        confidence = float(row.Confidence)
        support = float(getattr(row, "Support", math.nan))
        rule_id = getattr(row, "RuleID", "")
        weight = lift * confidence

        for premise in premises:
            for conclusion in conclusions:
                if premise == conclusion:
                    continue

                attributes = {
                    "weight": weight,
                    "lift": lift,
                    "confidence": confidence,
                    "support": support,
                    "rule_id": rule_id,
                }

                if graph.has_edge(premise, conclusion):
                    # Strict comparison preserves the first rule when weights tie.
                    if weight > graph[premise][conclusion]["weight"]:
                        graph[premise][conclusion].update(attributes)
                    continue

                attributes["edge_order"] = graph.number_of_edges()
                graph.add_edge(premise, conclusion, **attributes)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Rule projection produced an empty graph.")
    return graph


def build_full_graph_from_rules(filepath: str | Path) -> nx.Graph:
    """Compatibility helper used by visualization and experimental scripts."""
    return build_product_graph(load_rules_dataframe(filepath))


def graph_to_dataframe(graph: nx.Graph) -> pd.DataFrame:
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
                "Weight_Lift_x_Confidence": data["weight"],
                "RuleID": data.get("rule_id", ""),
            }
        )
    return pd.DataFrame(rows, columns=PROJECTED_GRAPH_COLUMNS).sort_values(
        "EdgeOrder", kind="stable"
    )


def _write_dataframe_safely(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep=";", index=False)
    temporary.replace(path)


def export_product_graph(graph: nx.Graph, output_file: str | Path) -> pd.DataFrame:
    frame = graph_to_dataframe(graph)
    _write_dataframe_safely(frame, Path(output_file))
    return frame


def load_projected_graph(filepath: str | Path) -> nx.Graph:
    path = Path(filepath).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Projected product graph not found: {path}")

    required = ("Product_1", "Product_2", "Weight_Lift_x_Confidence")
    frame = _read_csv_with_required_columns(path, required)
    if frame.empty:
        raise ValueError("Projected product graph is empty.")
    if "EdgeOrder" in frame.columns:
        frame = frame.sort_values("EdgeOrder", kind="stable")

    graph = nx.Graph()
    for fallback_order, row in enumerate(frame.itertuples(index=False)):
        product_1 = str(row.Product_1).strip()
        product_2 = str(row.Product_2).strip()
        if not product_1 or not product_2 or product_1 == product_2:
            raise ValueError("Projected graph contains an invalid edge.")

        weight = float(row.Weight_Lift_x_Confidence)
        if not math.isfinite(weight):
            raise ValueError("Projected graph contains a non-finite edge weight.")
        graph.add_edge(
            product_1,
            product_2,
            weight=weight,
            lift=float(getattr(row, "Lift", math.nan)),
            confidence=float(getattr(row, "Confidence", math.nan)),
            support=float(getattr(row, "Support", math.nan)),
            rule_id=getattr(row, "RuleID", ""),
            edge_order=int(getattr(row, "EdgeOrder", fallback_order)),
        )

    return graph


def run_projection(args: argparse.Namespace) -> None:
    input_file = args.input.expanduser().resolve()
    output_file = (
        args.output.expanduser().resolve()
        if args.output
        else input_file.with_name("PRODUCT_GRAPH.csv")
    )
    timing_file = output_file.parent / "GRAPH_PROJECTION_TIMING_SUMMARY.csv"
    metadata_file = output_file.parent / "GRAPH_PROJECTION_RUN_METADATA.csv"
    total_start = perf_counter()

    step_start = perf_counter()
    rules = load_rules_dataframe(input_file)
    load_seconds = perf_counter() - step_start

    step_start = perf_counter()
    graph = build_product_graph(rules)
    projection_seconds = perf_counter() - step_start

    step_start = perf_counter()
    export_product_graph(graph, output_file)
    export_seconds = perf_counter() - step_start
    total_seconds = perf_counter() - total_start

    timing = pd.DataFrame(
        {
            "Step": ["Load reduced rules", "Project rules to graph", "Export graph", "Total pipeline"],
            "Seconds": [load_seconds, projection_seconds, export_seconds, total_seconds],
        }
    )
    timing["Seconds"] = timing["Seconds"].round(4)
    _write_dataframe_safely(timing, timing_file)

    metadata = pd.DataFrame(
        {
            "Metric": [
                "Input file",
                "Output file",
                "Edge weight",
                "Duplicate-edge rule",
                "Input reduced rules",
                "Graph nodes",
                "Graph edges",
                "Connected components",
                "Python version",
                "pandas version",
                "NetworkX version",
            ],
            "Value": [
                str(input_file),
                str(output_file),
                "Lift x Confidence",
                "maximum weight",
                str(len(rules)),
                str(graph.number_of_nodes()),
                str(graph.number_of_edges()),
                str(nx.number_connected_components(graph)),
                platform.python_version(),
                pd.__version__,
                nx.__version__,
            ],
        }
    )
    _write_dataframe_safely(metadata, metadata_file)

    print("Product-graph projection completed.")
    print(f"Input reduced rules: {len(rules)}")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")
    print(f"Projected graph saved to: {output_file}")
    for row in timing.itertuples(index=False):
        print(f"{row.Step}: {row.Seconds:.4f} s")


def main() -> None:
    run_projection(parse_args())


if __name__ == "__main__":
    main()
