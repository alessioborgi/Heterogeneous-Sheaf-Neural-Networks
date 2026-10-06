#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

"""
Utilities for running heterogeneous GNNs (HAN, HGT) on homogeneous GC Data
objects that carry integer node_type / edge_type attributes.

These helpers are internal to the gnn_baselines package and not part of any
public API.
"""

import torch
from torch_geometric.data import Data, HeteroData


def data_to_heterodata(data: Data):
    """
    Convert a homogeneous Data object (with .node_type / .edge_type integer
    tensors) into a HeteroData with per-type node features and per-relation
    edge indices that are local within each (src_type, dst_type) pair.

    Returns
    -------
    hetero : HeteroData
    node_type_order : list[int]
        Sorted unique integer node-type values that are actually present in
        this graph (may be a subset of the dataset-level types).
    """
    node_type: torch.Tensor = data.node_type   # [N]
    edge_type: torch.Tensor = data.edge_type   # [E]
    edge_index: torch.Tensor = data.edge_index  # [2, E]
    x: torch.Tensor = data.x                   # [N, F]

    unique_nt = node_type.unique().sort().values
    node_type_order: list[int] = unique_nt.tolist()

    hetero = HeteroData()

    # --- node features -------------------------------------------------------
    # type_global_idx[t] is a sorted 1-D tensor of global node indices whose
    # node_type == t; used later for searchsorted remapping.
    type_global_idx: dict[int, torch.Tensor] = {}
    for nt in node_type_order:
        mask = (node_type == nt).nonzero(as_tuple=True)[0]  # sorted
        type_global_idx[nt] = mask
        hetero[f"t{nt}"].x = x[mask]

    # --- edge indices --------------------------------------------------------
    src_g = edge_index[0]
    dst_g = edge_index[1]
    src_nt = node_type[src_g]
    dst_nt = node_type[dst_g]

    for et in edge_type.unique().sort().values.tolist():
        et_mask = edge_type == et
        s_g = src_g[et_mask]
        d_g = dst_g[et_mask]
        s_nt = src_nt[et_mask]
        d_nt = dst_nt[et_mask]

        # Find unique (src_type, dst_type) pairs for this edge-type value.
        pairs = torch.stack([s_nt, d_nt], dim=1).unique(dim=0)
        for pair in pairs:
            st, dt = pair[0].item(), pair[1].item()
            pm = (s_nt == st) & (d_nt == dt)

            # Remap global → local indices via searchsorted on the sorted
            # global-index arrays built above.
            local_src = torch.searchsorted(type_global_idx[st], s_g[pm])
            local_dst = torch.searchsorted(type_global_idx[dt], d_g[pm])

            hetero[f"t{st}", f"e{et}", f"t{dt}"].edge_index = torch.stack(
                [local_src, local_dst]
            )

    return hetero, node_type_order


def flatten_xdict(
    x_dict: dict,
    node_type: torch.Tensor,
    node_type_order: list,
) -> torch.Tensor:
    """
    Reassemble x_dict (type_name → [N_t, D]) back into a flat [N, D] tensor
    in the original global node ordering, so that standard GC pooling works
    unchanged downstream.
    """
    sample = next(iter(x_dict.values()))
    out = torch.zeros(
        node_type.shape[0], sample.shape[1],
        device=sample.device, dtype=sample.dtype,
    )
    for nt in node_type_order:
        name = f"t{nt}"
        if name in x_dict:
            mask = (node_type == nt).nonzero(as_tuple=True)[0]
            out[mask] = x_dict[name]
    return out


