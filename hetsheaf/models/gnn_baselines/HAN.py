#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HANConv

from hetsheaf.models.gnn_baselines._gc_hetero_utils import (
    data_to_heterodata,
    flatten_xdict,
)


class HAN(nn.Module):
    def __init__(
        self,
        # metadata=None triggers GC mode (homogeneous Data input/output)
        metadata: Optional[tuple] = None,
        hidden_channels: int = 256,
        in_channels: Optional[Union[int, dict, DictConfig]] = None,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
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
            in_channels = dict(in_channels)
        in_channels_val = in_channels if isinstance(in_channels, (int, dict)) else -1

        self.node_types = list(metadata[0]) if metadata is not None else []

        self.convs = nn.ModuleList([
            HANConv(
                in_channels_val if i == 0 else hidden_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
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
    # ------------------------------------------------------------------

    def _forward_xdict(self, x_dict, edge_index_dict):
        if self.input_dropout > 0:
            x_dict = {
                k: F.dropout(v, p=self.input_dropout, training=self.training)
                for k, v in x_dict.items()
            }

        for i, layer in enumerate(self.convs):
            # Re-filter at each layer using the current x_dict: node types
            # may drop out between layers due to no incoming messages.
            cur_edge_index_dict = {
                k: v for k, v in edge_index_dict.items()
                if k[0] in x_dict and k[2] in x_dict
            }
            prev_x = x_dict
            new_x_dict = layer(x_dict, cur_edge_index_dict)
            # HANConv returns None for node types that receive no messages.
            # Fall back to the layer's own projection of the input so that
            # source-only types (e.g. 'drug') still get a valid embedding.
            x_dict = {
                k: (v if v is not None else F.relu(layer.proj[k](prev_x[k])))
                for k, v in new_x_dict.items()
                if v is not None or k in layer.proj
            }
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
        return "HAN"
