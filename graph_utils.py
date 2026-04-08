# ============================================================
# GRAPH UTILITIES -> Shared Loading and Graph Construction
# ============================================================

from pathlib import Path

import networkx as nx
import pandas as pd


# ==============================
# PATHS AND SHARED CONSTANTS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REDUCED_RULES_FILE = (
    BASE_DIR / "Rules_For_Python_REDUCED.csv"
    if (BASE_DIR / "Rules_For_Python_REDUCED.csv").exists()
    else BASE_DIR / "Rules_For_Python_REDUCED_CONFIDENCE.csv"
)
NUMERIC_COLUMNS = ("Lift", "Confidence", "Support")


# ==============================
# PRODUCT LABELS
# ==============================

SHORT_NAME_MAP = {
    "FRESH MEAT RNF - MR": "MEAT",
    "DELICATESSEN MEAT PRODUCTS": "DELI",
    "DAIRY PRODUCTS - SM": "DAIRY",
    "SOFT DRINKS - PB": "DRINK",
    "BEER AND ALCOPOPS - PA": "BEER",
    "VEGETABLES - VP": "VEG",
    "FRUITS - VV": "FRUIT",
    "POULTRY - MP": "POULT",
    "CHEESE - SD": "CHEES",
    "SNACKS AND SWEETS - PS": "SNACK",
    "BAKERY PRODUCTS - SP": "BAKER",
    "CLEANING PRODUCTS": "CLEAN",
    "HYGIENE AND PAPER PRODUCTS": "HYGI",
    "BEAUTY AND CARE": "BEAUT",
    "HOUSEHOLD ITEMS": "HOUSE",
    "PET PRODUCTS": "PET",
    "FROZEN PRODUCTS": "FROZE",
    "FISH - SR": "FISH",
    "READY MEAL INGREDIENTS - P": "MEAL",
    "DRIED FRUITS AND VEGETABLES - VS": "DRYFV",
    "BREAKFAST PRODUCTS - PD": "BRKF",
    "WINE AND SPIRITS - PV": "WINE",
    "CHILDREN PRODUCTS - ND": "KIDS",
    "PARTY PROGRAM - NP": "PARTY",
    "GASTRO PROGRAM - SG": "GASTR",
    "KIOSK PRODUCTS - NT": "KIOSK",
    "OFFICE SUPPLIES": "OFFIC",
    "SCHOOL SUPPLIES": "SCHOO",
    "MEDIA PRODUCTS": "MEDIA",
    "TECHNOLOGY - DB": "TECH",
    "SPORTS AND LEISURE - DT": "SPORT",
    "ECONOMAT - EK": "ECONO",
    "HOME DECOR AND GIFTS - DS": "DECOR",
    "PACKED MEAT - MS": "PMEAT",
    "HEALTH FOOD - PZ": "HEALT",
    "WHOLESALE TRANSIT": "WST",
    "WHOLESALE VP TRANSIT": "WVP",
}


# ==============================
# FILE LOADING
# ==============================

def load_rules_dataframe(filepath: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep=";")
    except Exception:
        df = pd.read_csv(filepath, sep=",")

    df.columns = [str(column).strip() for column in df.columns]

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df


# ==============================
# SMALL HELPERS
# ==============================

def split_items(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def get_short_name(product_name: str) -> str:
    return SHORT_NAME_MAP.get(product_name, product_name[:5].upper())


# ==============================
# GRAPH CONSTRUCTION
# ==============================

def build_full_graph_from_rules(filepath: str | Path) -> nx.Graph:
    df = load_rules_dataframe(filepath)
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        premises = split_items(row.Premises)
        conclusions = split_items(row.Conclusion)
        lift = float(row.Lift)
        confidence = float(row.Confidence)
        weight = lift * confidence

        for premise in premises:
            for conclusion in conclusions:
                if premise == conclusion:
                    continue

                if graph.has_edge(premise, conclusion):
                    if graph[premise][conclusion]["weight"] < weight:
                        graph[premise][conclusion].update(
                            weight=weight,
                            lift=lift,
                            confidence=confidence,
                        )
                    continue

                graph.add_edge(
                    premise,
                    conclusion,
                    weight=weight,
                    lift=lift,
                    confidence=confidence,
                )

    return graph
