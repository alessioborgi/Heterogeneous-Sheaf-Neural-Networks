#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os
import shutil
from collections import Counter
from typing import Literal, Optional, Union

import lightning as L
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.lightning import LightningNodeData, LightningLinkData
from torch_geometric.datasets import HGBDataset

from hetsheaf.datasets.utils import RemoveSelfLoops, SubsetTrainSplit

DATA_DIR = "data"


class HGBBaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        target: str = "author",
        num_classes: int = 4,
        data_dir: str = DATA_DIR,
        task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
        dataset: Literal["IMDB", "DBLP", "ACM"] = "DBLP",
        homogeneous: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.num_classes: int = num_classes
        self.task = task
        self.pyg_datamodule: Optional[LightningNodeData] = None
        self.metadata = None
        self.dataset = dataset
        self.num_nodes = None
        self.in_channels: Union[Optional[dict[str, int]], int] = None
        self.homogeneous = homogeneous
        self.edge_index: Optional[Union[dict[str, torch.Tensor], torch.Tensor]] = None
        self.graph_size: Optional[int] = None
        self.num_node_types: Optional[int] = None
        self.num_edge_types: Optional[int] = None
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None

    def prepare_data(self) -> None:
        transform = T.Compose(
            [
                T.Constant(node_types=None),
                T.ToUndirected(),
                SubsetTrainSplit(val_ratio=0.1), #seed: int = None
                RemoveSelfLoops(),
                T.NormalizeFeatures(),
                T.RemoveDuplicatedEdges(),
            ]
        )
        dataset = HGBDataset(root=self.data_dir, name=self.dataset, transform=transform)

        data: Union[HeteroData, Data] = dataset[0].coalesce()
        input_nodes = data[self.target]

        self.edge_index = data.edge_index_dict
        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }
        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes

        if self.homogeneous:
            data = data.to_homogeneous()
            self.edge_index = data.edge_index
            input_nodes = data
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        self.pyg_datamodule = LightningNodeData(
            data,
            input_train_nodes=(self.target, input_nodes.train_mask),
            input_val_nodes=(self.target, input_nodes.val_mask),
            input_test_nodes=(self.target, input_nodes.test_mask),
            loader="full",
            batch_size=1,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.test_dataloader()


class IMDBDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=data_dir,
            task="multilabel",
            num_classes=5,
            dataset="IMDB",
            target="movie",
            homogeneous=homogeneous,
        )

    def __str__(self):
        return "IMDB"


class DBLPDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            dataset="DBLP",
            num_classes=4,
            target="author",
            task="multiclass",
            data_dir=data_dir,
            homogeneous=homogeneous,
        )

    def __str__(self):
        return "DBLP"


class ACMDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=data_dir,
            dataset="ACM",
            num_classes=3,
            target="paper",
            task="multiclass",
            homogeneous=homogeneous,
        )

    def __str__(self):
        return "ACM"


class HGBLinkPredBase(L.LightningDataModule):
    """
    Base class for HGB link-prediction datasets (Amazon, PubMed).

    Downloads the dataset from Tsinghua Cloud (zip), parses ``node.dat``
    (node types / optional raw features) and ``link.dat`` (positive training
    edges), then uses :class:`~torch_geometric.transforms.RandomLinkSplit` to
    produce train / val / test splits, exactly as the other LP data-modules in
    this project do.  The optional ``link.dat.test`` file is loaded when
    present so that the *target* relation keeps the HGB-provided test positives
    and negatives rather than randomly sampled ones.
    """

    # Subclasses must override these attributes
    gdrive_file_id: str = ""
    zip_name: str = ""
    # node type index → human-readable name used in HeteroData
    node_type_map: dict[int, str] = {}
    # edge type index → (src_node_type, relation_name, dst_node_type)
    edge_type_map: dict[int, tuple[str, str, str]] = {}
    # the target edge type triple for link prediction
    target: tuple[str, str, str] = ("", "", "")
    rev_target: tuple[str, str, str] = ("", "", "")
    # edge type index (in link.dat.test) for the evaluation relation
    test_edge_type_id: int = 0

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.homogeneous = homogeneous
        self.split_ratio = split_ratio

        self.metadata = None
        self.in_channels: Optional[Union[dict[str, int], int]] = None
        self.num_nodes: Optional[int] = None
        self.num_node_types: Optional[int] = None
        self.num_edge_types: Optional[int] = None
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[tuple[str, str, str]]] = None
        self.graph_size: Optional[int] = None
        self.task = "binary"
        self.num_classes = 1

        self.train_data: Optional[Union[Data, HeteroData]] = None
        self.val_data: Optional[Union[Data, HeteroData]] = None
        self.test_data: Optional[Union[Data, HeteroData]] = None

    # ------------------------------------------------------------------
    # Download & extraction helpers
    # ------------------------------------------------------------------

    def _download_and_extract(self) -> None:
        """Download the HGB LP zip archive from Google Drive and extract it."""
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if not os.path.exists(os.path.join(self.data_dir, "node.dat")):
            if not os.path.exists(zip_path):
                print(f"Downloading {self.zip_name} from Google Drive...")
                try:
                    import gdown
                    url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                    gdown.download(url, zip_path, quiet=False)
                except ImportError:
                    raise ImportError(
                        "Please install gdown to download HGB datasets: "
                        "pip install gdown"
                    )

            if os.path.exists(zip_path):
                import zipfile
                print(f"Extracting {self.zip_name}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)

                # Flatten one level of subdirectory if the zip extracted into a
                # folder named after the dataset (e.g. amazon_ini/, PubMed_ini/)
                for entry in os.listdir(self.data_dir):
                    entry_path = os.path.join(self.data_dir, entry)
                    if os.path.isdir(entry_path) and entry.endswith("_ini"):
                        for fname in os.listdir(entry_path):
                            shutil.move(
                                os.path.join(entry_path, fname),
                                os.path.join(self.data_dir, fname),
                            )
                        os.rmdir(entry_path)
                        break

    # ------------------------------------------------------------------
    # Node / edge parsing helpers
    # ------------------------------------------------------------------

    def _parse_nodes(self) -> tuple[dict[int, tuple[int, int]], dict[int, int], Counter]:
        """
        Parse ``node.dat``.

        Returns
        -------
        global_to_local : dict mapping global_node_id → (node_type, local_idx)
        node_features   : dict mapping global_node_id → feature vector list
                          (empty dict when features are absent)
        counts          : Counter mapping node_type_id → number of nodes
        """
        node_file = os.path.join(self.data_dir, "node.dat")
        global_to_local: dict[int, tuple[int, int]] = {}
        node_features: dict[int, list] = {}
        counts: Counter = Counter()
        type_counters: Counter = Counter()

        with open(node_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                node_id = int(parts[0])
                node_type = int(parts[2])
                global_to_local[node_id] = (node_type, type_counters[node_type])
                type_counters[node_type] += 1
                counts[node_type] += 1
                if len(parts) == 4:  # has comma-separated feature string
                    node_features[node_id] = [float(x) for x in parts[3].split(",")]

        return global_to_local, node_features, counts

    def _parse_links(
        self,
        filename: str,
        global_to_local: dict[int, tuple[int, int]],
        has_label: bool = False,
    ) -> dict[int, dict]:
        """
        Parse a ``link.dat`` or ``link.dat.test`` file.

        Returns a dict keyed by relation type id, each value being a dict with
        ``src``, ``dst``, and (when *has_label* is True) ``label`` lists of
        *local* node indices.
        """
        link_file = os.path.join(self.data_dir, filename)
        if not os.path.exists(link_file):
            return {}

        edge_data: dict[int, dict] = {}
        with open(link_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])
                weight_or_label = float(parts[3])
                if h_id not in global_to_local or t_id not in global_to_local:
                    continue
                if r_id not in edge_data:
                    edge_data[r_id] = {"src": [], "dst": [], "label": []}
                h_local = global_to_local[h_id][1]
                t_local = global_to_local[t_id][1]
                edge_data[r_id]["src"].append(h_local)
                edge_data[r_id]["dst"].append(t_local)
                if has_label:
                    edge_data[r_id]["label"].append(int(weight_or_label))

        return edge_data

    # ------------------------------------------------------------------
    # Build HeteroData
    # ------------------------------------------------------------------

    def _build_hetero_data(
        self,
        global_to_local: dict,
        node_features: dict,
        counts: Counter,
        train_edges: dict[int, dict],
        test_edge_data: Optional[dict[int, dict]] = None,
    ) -> HeteroData:
        """
        Construct a :class:`~torch_geometric.data.HeteroData` object from
        parsed HGB LP files.

        If *test_edge_data* is provided, the target edge type is augmented
        with ``edge_label`` / ``edge_label_index`` for the test split so that
        the HGB-defined evaluation set is preserved.
        """
        data = HeteroData()

        # ---- node features ----
        feat_dim = 128  # fallback when no raw features exist
        for type_id, type_name in self.node_type_map.items():
            n = counts[type_id]
            if any(global_to_local[g][0] == type_id for g in node_features):
                # gather feature rows in local-index order
                local_feats = {}
                for g_id, feat in node_features.items():
                    if global_to_local[g_id][0] == type_id:
                        local_feats[global_to_local[g_id][1]] = feat
                mat = [local_feats[i] for i in range(n)]
                data[type_name].x = torch.tensor(mat, dtype=torch.float)
            else:
                data[type_name].x = torch.ones(n, feat_dim)
            data[type_name].num_nodes = n

        # ---- training edges ----
        for r_id, edict in train_edges.items():
            if r_id not in self.edge_type_map:
                continue
            src_type, rel_name, dst_type = self.edge_type_map[r_id]
            ei = torch.tensor([edict["src"], edict["dst"]], dtype=torch.long)
            data[src_type, rel_name, dst_type].edge_index = ei
            # Add reverse
            rev_src, rev_rel, rev_dst = self.rev_target if (
                (src_type, rel_name, dst_type) == self.target
            ) else (dst_type, f"rev_{rel_name}", src_type)
            data[rev_src, rev_rel, rev_dst].edge_index = ei.flip(0)

        return data

    # ------------------------------------------------------------------
    # prepare_data / dataloaders
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        self._download_and_extract()

        global_to_local, node_features, counts = self._parse_nodes()
        train_edges = self._parse_links("link.dat", global_to_local, has_label=False)
        test_edge_data = self._parse_links(
            "link.dat.test", global_to_local, has_label=True
        )

        data = self._build_hetero_data(
            global_to_local, node_features, counts, train_edges, test_edge_data
        )

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.in_channels = {
            nt: data[nt].num_features for nt in data.node_types
        }

        if self.homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names
        else:
            self.node_type_names = list(data.node_types)
            self.edge_type_names = list(data.edge_types)

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.homogeneous else self.target,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=None if self.homogeneous else self.rev_target,
        )
        self.train_data, self.val_data, self.test_data = split(data)

        if self.homogeneous:
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0 else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0 else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0 else -1
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


class HGBAmazonLPDataModule(HGBLinkPredBase):
    """
    HGB-Amazon link-prediction dataset.

    - **Nodes**: items (single node type)
    - **Edge types** (4): ``also_viewed`` (target), ``also_bought``, ``brand``,
      ``category``
    - **Download**: Google Drive (``amazon_ini.zip``)

    Reference: `HGB benchmark <https://github.com/THUDM/HGB>`_
    """

    gdrive_file_id = "1XD5lXC2NOuu3KLEyYRWad4mqBxZXT468"
    zip_name = "amazon_ini.zip"
    node_type_map = {0: "item"}
    edge_type_map = {
        0: ("item", "also_viewed", "item"),
        1: ("item", "also_bought", "item"),
        2: ("item", "brand", "item"),
        3: ("item", "category", "item"),
    }
    target = ("item", "also_viewed", "item")
    rev_target = ("item", "rev_also_viewed", "item")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=f"{data_dir}/hgb_amazon",
            homogeneous=homogeneous,
        )

    def __repr__(self) -> str:
        return "HGBAmazon"


class HGBPubMedLPDataModule(HGBLinkPredBase):
    """
    HGB-PubMed link-prediction dataset.

    - **Node types** (4): ``gene``, ``disease``, ``chemical``, ``species``
    - **Target edge**: Gene–Gene interaction (``gene_gene``)
    - **Download**: Google Drive (``PubMed_ini.zip``)

    Reference: `HGB benchmark <https://github.com/THUDM/HGB>`_
    """

    gdrive_file_id = "1hJHKN5SW8Fv61GRF3Adr6MPOpNA5j0YI"
    zip_name = "PubMed_ini.zip"
    node_type_map = {
        0: "gene",
        1: "disease",
        2: "chemical",
        3: "species",
    }
    # Edge-type IDs verified from link.dat node types:
    # 0: G→G, 1: G→D, 2: D→D, 3: C→G, 4: C→D,
    # 5: C→C, 6: C→S, 7: S→G, 8: S→D, 9: S→S
    edge_type_map = {
        0: ("gene", "gene_gene", "gene"),
        1: ("gene", "gene_disease", "disease"),
        2: ("disease", "disease_disease", "disease"),
        3: ("chemical", "chemical_gene", "gene"),
        4: ("chemical", "chemical_disease", "disease"),
        5: ("chemical", "chemical_chemical", "chemical"),
        6: ("chemical", "chemical_species", "species"),
        7: ("species", "species_gene", "gene"),
        8: ("species", "species_disease", "disease"),
        9: ("species", "species_species", "species"),
    }
    target = ("gene", "gene_gene", "gene")
    rev_target = ("gene", "rev_gene_gene", "gene")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=f"{data_dir}/hgb_pubmed",
            homogeneous=homogeneous,
        )

    def __repr__(self) -> str:
        return "HGBPubMed"


class HGBYouTubeLPDataModule(HGBLinkPredBase):
    """
    HGB-YouTube link-prediction dataset (HGB benchmark format).

    - **Nodes**: users (single node type)
    - **Edge types** (5): ``interaction_0`` … ``interaction_4``;
      ``interaction_0`` is used as the LP target.
    - **Download**: Google Drive (``youtube.zip``)

    Reference: `HGB benchmark <https://github.com/THUDM/HGB>`_
    """

    gdrive_file_id = "1JZfztaVNjsBwPkJC2CfaESv-ua78zf-K"
    zip_name = "youtube.zip"
    node_type_map = {0: "user"}
    edge_type_map = {
        0: ("user", "interaction_0", "user"),
        1: ("user", "interaction_1", "user"),
        2: ("user", "interaction_2", "user"),
        3: ("user", "interaction_3", "user"),
        4: ("user", "interaction_4", "user"),
    }
    target = ("user", "interaction_0", "user")
    rev_target = ("user", "rev_interaction_0", "user")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=f"{data_dir}/hgb_youtube",
            homogeneous=homogeneous,
        )

    def _download_and_extract(self) -> None:
        """Override to handle youtube.zip which extracts without a trailing _ini dir."""
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if not os.path.exists(os.path.join(self.data_dir, "node.dat")):
            if not os.path.exists(zip_path):
                print(f"Downloading {self.zip_name} from Google Drive...")
                try:
                    import gdown
                    url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                    gdown.download(url, zip_path, quiet=False)
                except ImportError:
                    raise ImportError(
                        "Please install gdown to download HGB datasets: "
                        "pip install gdown"
                    )

            if os.path.exists(zip_path):
                import zipfile
                print(f"Extracting {self.zip_name}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)

                # Flatten any single subdirectory the zip may have created
                for entry in os.listdir(self.data_dir):
                    entry_path = os.path.join(self.data_dir, entry)
                    if os.path.isdir(entry_path) and entry not in (".", ".."):
                        candidate = os.path.join(entry_path, "node.dat")
                        if os.path.exists(candidate):
                            for fname in os.listdir(entry_path):
                                shutil.move(
                                    os.path.join(entry_path, fname),
                                    os.path.join(self.data_dir, fname),
                                )
                            os.rmdir(entry_path)
                            break

    def __repr__(self) -> str:
        return "HGBYouTube"
