#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional, Union

import lightning as L
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import MovieLens, LastFM

from .utils import RemoveSelfLoops

DATA_DIR = "data"


class LinkPredBase(L.LightningDataModule):
    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        num_classes: int = 1,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super(LinkPredBase, self).__init__()
        self.target: tuple[str, str, str] = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None
        self.num_nodes = None
        self.train_data: Optional[Union[Data, HeteroData]] = None
        self.val_data: Optional[Union[Data, HeteroData]] = None
        self.test_data: Optional[Union[Data, HeteroData]] = None
        self.transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
            ]
        )
        self.rev_target = rev_target
        self.is_homogeneous = is_homogeneous
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.graph_size = 0
        self.task = "binary"
        self.num_node_types: int = 0
        self.num_edge_types: int = 0
        self.num_edges: int = 0
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None

    def download_data(self) -> HeteroData: ...

    def prepare_data(self) -> None:
        data = self.download_data()

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.num_edges = data.num_edges

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.num_nodes
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            # split_labels=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=self.rev_target,
        )

        self.train_data, self.val_data, self.test_data = split(data)

        # Recalculate graph_size after split (negative sampling may create new node IDs)
        if self.is_homogeneous:
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0
                else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0
                else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0
                else -1
            )
            self.graph_size = max(
                self.graph_size, max_train + 1, max_val + 1, max_test + 1
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(LastFMDataModule, self).__init__(
            data_dir=f"{data_dir}/lastfm",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = LastFM(self.data_dir, transform=self.transform)[0]

        del data[self.target]["train_neg_edge_index"]
        del data[self.target]["val_pos_edge_index"]
        del data[self.target]["val_neg_edge_index"]
        del data[self.target]["test_pos_edge_index"]
        del data[self.target]["test_neg_edge_index"]

        return data

    def __repr__(self):
        return "LastFM"


class MovieLensDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(MovieLensDataModule, self).__init__(
            data_dir=f"{data_dir}/movie_lens",
            target=("user", "rates", "movie"),
            rev_target=("movie", "rev_rates", "user"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = MovieLens(self.data_dir, transform=self.transform)[0]
        del data[self.target]["edge_label"]
        del data[self.rev_target]["edge_label"]
        # new_edge_labels = torch.ones_like(data[self.target].edge_label)
        # data[self.target].edge_label = new_edge_labels
        # data[self.rev_target].edge_label = new_edge_labels

        # print(data[self.target].edge_label)
        return data

    def __repr__(self):
        return "MovieLens"
