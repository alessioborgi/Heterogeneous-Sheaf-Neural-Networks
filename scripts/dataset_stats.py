"""
dataset_stats.py
================
Prints statistics for every dataset in the project:
  - number of nodes (total + per type)
  - node types
  - number of edges (total + per type)
  - edge types

Run from the project root:
    python scripts/dataset_stats.py

Pass --dataset <name> to run a single dataset.
"""

import sys
import os
import argparse
import traceback

# Ensure the project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)



# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 70
SUBSEPARATOR = "-" * 50


def _print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _fmt_node_types(node_types) -> str:
    if not node_types:
        return "(none)"
    return ", ".join(str(t) for t in node_types)


def _fmt_edge_types(edge_types) -> str:
    if not edge_types:
        return "(none)"
    parts = []
    for t in edge_types:
        if isinstance(t, (tuple, list)):
            parts.append(f"({', '.join(str(x) for x in t)})")
        else:
            parts.append(str(t))
    return "\n      ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Hetero / Homogeneous single-graph datasets (NC + LP)
# ──────────────────────────────────────────────────────────────────────────────

def stats_single_graph(name: str, dm) -> None:
    """Report stats for a single-graph (node or link prediction) datamodule."""
    _print_section(name)
    try:
        dm.prepare_data()
    except Exception:
        print(f"  [ERROR loading {name}]")
        traceback.print_exc()
        return

    from torch_geometric.data import HeteroData

    # ── collect per-type stats from every candidate data object ─────────────
    per_node_type: dict = {}
    per_edge_type: dict = {}
    edge_source_label: str = "all edges"

    def _ingest_hetero(data, label: str = "all edges"):
        nonlocal edge_source_label
        if not isinstance(data, HeteroData):
            return
        for nt in data.node_types:
            if nt not in per_node_type:
                per_node_type[nt] = data[nt].num_nodes
        for et in data.edge_types:
            ei = data[et].get("edge_index")
            if ei is not None and et not in per_edge_type:
                per_edge_type[et] = ei.size(1)
                edge_source_label = label

    # 1. split data objects (LP modules)
    for attr in ("train_data", "data"):
        obj = getattr(dm, attr, None)
        if obj is not None:
            _ingest_hetero(obj, "train split")

    # 2. HGB NC: data wrapped in dm.pyg_datamodule — full graph, prefer over split
    pyg_dm = getattr(dm, "pyg_datamodule", None)
    if pyg_dm is not None:
        for inner_attr in ("data", "_data"):
            inner = getattr(pyg_dm, inner_attr, None)
            if inner is not None:
                # Override any split-derived counts with full-graph counts
                per_edge_type.clear()
                _ingest_hetero(inner, "all edges")
                break

    # 3. edge_index_dict stored directly (HGB NC): highest priority, pre-split
    edge_index_attr = getattr(dm, "edge_index", None)
    if isinstance(edge_index_attr, dict):
        per_edge_type.clear()
        edge_source_label = "all edges"
        for et, ei in edge_index_attr.items():
            if ei is not None:
                per_edge_type[et] = ei.size(1)

    # ── gather declared types from metadata ─────────────────────────────────
    metadata = getattr(dm, "metadata", None)
    if metadata is not None:
        declared_node_types, declared_edge_types = metadata
    else:
        declared_node_types = getattr(dm, "node_type_names", None) or list(per_node_type.keys())
        declared_edge_types = getattr(dm, "edge_type_names", None) or list(per_edge_type.keys())

    # ── per-type node counts from in_channels as fallback ───────────────────
    in_channels = getattr(dm, "in_channels", None)
    if not per_node_type and isinstance(in_channels, dict):
        # in_channels has the same keys but no counts — mark unknown
        for nt in in_channels:
            per_node_type[nt] = "?"

    num_nodes = getattr(dm, "num_nodes", None)
    if num_nodes is None and per_node_type:
        num_counts = [c for c in per_node_type.values() if isinstance(c, int)]
        if num_counts:
            num_nodes = sum(num_counts)

    print(f"  Nodes total     : {num_nodes if num_nodes is None else f'{num_nodes:,}'}")
    print(f"  Node types      : {_fmt_node_types(declared_node_types)}")
    if per_node_type:
        print("  Nodes per type  :")
        for nt, cnt in per_node_type.items():
            print(f"      {str(nt):40s}  {cnt if cnt == '?' else f'{cnt:,}'}")

    num_edges = getattr(dm, "num_edges", None)
    if per_edge_type:
        counts = [c for c in per_edge_type.values() if isinstance(c, int)]
        if counts and edge_source_label == "all edges":
            # override with exact sum from edge_index (more precise)
            num_edges = sum(counts)
    elif num_edges is None:
        pass  # leave as None

    num_et = len(declared_edge_types) if declared_edge_types else 0
    print(f"  Edges total     : {num_edges if num_edges is None else f'{num_edges:,}'}")
    print(f"  Edge types ({num_et}, {edge_source_label})  :")
    if declared_edge_types:
        for et in declared_edge_types:
            cnt = per_edge_type.get(et, "?")
            label = str(et) if not isinstance(et, (tuple, list)) else f"({', '.join(et)})"
            cnt_str = cnt if cnt == "?" else f"{cnt:,}"
            print(f"      {label:60s}  {cnt_str}")


# ──────────────────────────────────────────────────────────────────────────────
# Graph-classification (TU) datasets
# ──────────────────────────────────────────────────────────────────────────────

def stats_graph_classification(name: str, dm) -> None:
    """Report aggregate stats across all graphs in a graph-classification dataset."""
    _print_section(name)
    try:
        dataset = dm.prepare_data()
        dm.setup()
    except Exception:
        print(f"  [ERROR loading {name}]")
        traceback.print_exc()
        return


    total_nodes = 0
    total_edges = 0
    node_type_set: set = set()
    edge_type_set: set = set()

    for data in dataset:
        total_nodes += int(data.num_nodes)
        total_edges += int(data.edge_index.size(1)) if data.edge_index is not None else 0
        if hasattr(data, "node_type") and data.node_type is not None:
            node_type_set.update(data.node_type.unique().tolist())
        if hasattr(data, "edge_type") and data.edge_type is not None and data.edge_type.numel() > 0:
            edge_type_set.update(data.edge_type.unique().tolist())

    num_graphs = len(dataset)
    avg_nodes = total_nodes / num_graphs if num_graphs else 0
    avg_edges = total_edges / num_graphs if num_graphs else 0
    num_node_types = dm.num_node_types
    num_edge_types = dm.num_edge_types

    print(f"  Graphs          : {num_graphs:,}")
    print(f"  Nodes total     : {total_nodes:,}  (avg {avg_nodes:.1f} per graph)")
    print(f"  Num. node types : {num_node_types}")
    if node_type_set:
        print(f"  Node type IDs   : {sorted(node_type_set)}")
    print(f"  Edges total     : {total_edges:,}  (avg {avg_edges:.1f} per graph)")
    print(f"  Num. edge types : {num_edge_types}")
    if edge_type_set:
        print(f"  Edge type IDs   : {sorted(edge_type_set)}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset catalogue
# ──────────────────────────────────────────────────────────────────────────────

def build_catalogue():
    """Return an ordered dict: {display_name: (kind, factory_fn)}."""
    from hetsheaf.datasets import (
        IMDBDataModule,
        DBLPDataModule,
        ACMDataModule,
        HGBAmazonLPDataModule,
        HGBPubMedLPDataModule,
        HGBYouTubeLPDataModule,
        LastFMDataModule,
        MovieLensDataModule,
        CTD_DDADataModule,
        NDFRT_DDADataModule,
        DrugBankDDIDataModule,
        STRINGPPIDataModule,
        AS733LPDataModule,
        RouteNetNCDataModule,
        MUTAGDataModule,
        PROTEINSDataModule,
        ENZYMESDataModule,
        NCI1DataModule,
    )

    data_dir = os.path.join(ROOT, "data")

    catalogue = {}

    # ── Node classification ──────────────────────────────────────────────────
    catalogue["HGB-IMDB (NC)"] = ("single", lambda: IMDBDataModule(data_dir=data_dir))
    catalogue["HGB-DBLP (NC)"] = ("single", lambda: DBLPDataModule(data_dir=data_dir))
    catalogue["HGB-ACM (NC)"]  = ("single", lambda: ACMDataModule(data_dir=data_dir))
    catalogue["RouteNet-Traffic (NC)"] = (
        "single",
        lambda: RouteNetNCDataModule(data_dir=f"{data_dir}/routenet/traffic_models"),
    )

    # ── Link prediction ──────────────────────────────────────────────────────
    catalogue["LastFM (LP)"]      = ("single", lambda: LastFMDataModule(data_dir=data_dir))
    catalogue["MovieLens (LP)"]   = ("single", lambda: MovieLensDataModule(data_dir=f"{data_dir}/movie_lens"))
    catalogue["HGB-PubMed (LP)"]  = ("single", lambda: HGBPubMedLPDataModule(data_dir=data_dir))
    catalogue["HGB-Amazon (LP)"]  = ("single", lambda: HGBAmazonLPDataModule(data_dir=data_dir))
    catalogue["HGB-YouTube (LP)"] = ("single", lambda: HGBYouTubeLPDataModule(data_dir=data_dir))
    catalogue["AS-733 (LP)"]      = ("single", lambda: AS733LPDataModule(data_dir=data_dir))

    # ── Biomedical link prediction ───────────────────────────────────────────
    catalogue["CTD_DDA (Bio)"]     = ("single", lambda: CTD_DDADataModule(data_dir=data_dir))
    catalogue["NDFRT_DDA (Bio)"]   = ("single", lambda: NDFRT_DDADataModule(data_dir=data_dir))
    catalogue["DrugBankDDI (Bio)"] = ("single", lambda: DrugBankDDIDataModule(data_dir=data_dir))
    catalogue["STRING_PPI (Bio)"]  = ("single", lambda: STRINGPPIDataModule(data_dir=data_dir))

    # ── Graph classification ─────────────────────────────────────────────────
    catalogue["MUTAG (GC)"]    = ("gc", lambda: MUTAGDataModule(root=f"{data_dir}/MUTAG"))
    catalogue["PROTEINS (GC)"] = ("gc", lambda: PROTEINSDataModule(root=f"{data_dir}/PROTEINS"))
    catalogue["ENZYMES (GC)"]  = ("gc", lambda: ENZYMESDataModule(root=f"{data_dir}/ENZYMES"))
    catalogue["NCI1 (GC)"]     = ("gc", lambda: NCI1DataModule(root=f"{data_dir}/NCI1"))

    return catalogue


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run only this dataset (substring match on display name). "
             "Omit to run all datasets.",
    )
    args = parser.parse_args()

    catalogue = build_catalogue()

    for display_name, (kind, factory) in catalogue.items():
        if args.dataset and args.dataset.lower() not in display_name.lower():
            continue

        dm = factory()

        if kind == "gc":
            stats_graph_classification(display_name, dm)
        else:
            stats_single_graph(display_name, dm)

    print(f"\n{SEPARATOR}")
    print("  Done.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
