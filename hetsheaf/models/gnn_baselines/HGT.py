#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HGTConv

from hetsheaf.models.gnn_baselines._gc_hetero_utils import (
    data_to_heterodata,
    flatten_xdict,
)


class HGT(nn.Module):
    def __init__(
        self,
        # metadata=None triggers GC mode (homogeneous Data input/output)
        metadata: Optional[tuple] = None,
        hidden_channels: int = 256,
        in_channels: Optional[Union[int, dict, DictConfig]] = None,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initial_dropout: float = 0.0,
        output_dim: int = 256,
        # Used only in GC mode (passed from run_gnn_gc via model_kwargs)
        num_node_types: int = 1,
        num_relations: int = 1,
        **_kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.initial_dropout = initial_dropout
        self.output_dim = output_dim

        if isinstance(in_channels, DictConfig):
            in_channels = dict(in_channels) if len(in_channels) > 0 else -1
        if hasattr(metadata, "__iter__") and not isinstance(metadata, tuple):
            metadata = (list(metadata[0]), [tuple(et) for et in metadata[1]])
        in_channels_val = in_channels if isinstance(in_channels, (int, dict)) else -1

        self.node_types = list(metadata[0]) if metadata is not None else []

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels_val if i == 0 else hidden_channels,
                hidden_channels,
                heads=heads,
                metadata=metadata,
            )
            for i in range(layers)
        ])

        if int(self.output_dim) != int(hidden_channels):
            self.lin = nn.Linear(hidden_channels, self.output_dim)
        else:
            self.lin = None

    # ------------------------------------------------------------------
    # Shared layer processing (same for GC and hetero modes)
    # HGTConv uses pyg_lib.segment_matmul which requires Float32; the
    # autocast context prevents Lightning's BF16 policy from conflicting.
    # ------------------------------------------------------------------

    def _forward_xdict(self, x_dict, edge_index_dict):
        with torch.autocast(device_type="cuda", enabled=False):
            x_dict = {k: v.float() for k, v in x_dict.items()}

            if self.input_dropout > 0:
                x_dict = {
                    k: F.dropout(v, p=self.input_dropout, training=self.training)
                    for k, v in x_dict.items()
                }

            for i, layer in enumerate(self.convs):
                # Re-filter at each layer: node types drop out of x_dict after
                # each layer (HGTConv only outputs dst node types).
                cur_edge_index_dict = {
                    k: v for k, v in edge_index_dict.items()
                    if k[0] in x_dict and k[2] in x_dict
                }
                prev_x = x_dict
                if cur_edge_index_dict:
                    x_dict = layer(x_dict, cur_edge_index_dict)
                else:
                    x_dict = {}
                # Restore source-only types missing from the output by running
                # them through the layer's kqv projection (Q part) + out_lin —
                # the same path HGTConv uses for dst nodes, just without messages.
                missing = {nt: v for nt, v in prev_x.items() if nt not in x_dict}
                if missing:
                    C = layer.out_channels
                    kqv = layer.kqv_lin(missing)
                    fallback = layer.out_lin({
                        nt: F.gelu(v[:, C:2 * C]) for nt, v in kqv.items()
                    })
                    x_dict.update(fallback)
                if i == 0 and self.initial_dropout > 0:
                    x_dict = {
                        k: F.dropout(v, p=self.initial_dropout, training=self.training)
                        for k, v in x_dict.items()
                    }
                elif i > 0 and self.dropout > 0:
                    x_dict = {
                        k: F.dropout(v, p=self.dropout, training=self.training)
                        for k, v in x_dict.items()
                    }

        if self.lin is not None:
            x_dict = {k: self.lin(v) for k, v in x_dict.items()}

        return x_dict

    # ------------------------------------------------------------------
    # Forward: route between GC (homogeneous Data) and hetero paths
    # ------------------------------------------------------------------

    def forward(self, data):
        if isinstance(data, Data):
            # GC path: Data with .node_type/.edge_type → HeteroData →
            # x_dict → flat [N, D] tensor in original node order.
            hetero, nt_order = data_to_heterodata(data)
            x_dict = self._forward_xdict(hetero.x_dict, hetero.edge_index_dict)
            return flatten_xdict(x_dict, data.node_type, nt_order)

        # Hetero path (NC / LP): unchanged behaviour.
        return self._forward_xdict(data.x_dict, data.edge_index_dict)

    def __repr__(self):
        return "HGT"
