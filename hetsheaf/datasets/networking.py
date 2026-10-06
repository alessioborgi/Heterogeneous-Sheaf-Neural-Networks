"""Networking datasets for heterogeneous graph learning.

Datasets:
- AS-733: BGP Autonomous Systems topology from SNAP
- RouteNet: SDN traffic engineering (link/path bipartite graphs)

Each dataset supports node classification (NC) and/or link prediction (LP).
"""

import math
import os
import tarfile
from collections import Counter, defaultdict
from typing import Literal, Optional, Union

import lightning as L
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning import LightningNodeData

from .link_pred import DATA_DIR, LinkPredBase
from .utils import RemoveSelfLoops


# =============================================================================
# Shared base: NC DataModule for networking datasets
# =============================================================================


class NetworkingNCBaseDataModule(L.LightningDataModule):
    """Base class for networking node classification datasets.

    Subclasses must implement ``_load_data()`` which returns a
    :class:`~torch_geometric.data.HeteroData` object with labels on the
    target node type.
    """

    def __init__(
        self,
        target: str,
        num_classes: int,
        task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: Literal["constant", "degree"] = "degree",
    ):
        super().__init__()
        self.target = target
        self.num_classes = num_classes
        self.task = task
        self.data_dir = data_dir
        self.homogeneous = homogeneous
        self.node_feature_type = node_feature_type

        self.metadata = None
        self.in_channels: Union[Optional[dict[str, int]], int] = None
        self.num_nodes: Optional[int] = None
        self.edge_index = None
        self.graph_size: Optional[int] = None
        self.num_node_types: Optional[int] = None
        self.num_edge_types: Optional[int] = None
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None
        self.pyg_datamodule: Optional[LightningNodeData] = None

    def _load_data(self) -> HeteroData:
        """Build and return the HeteroData object. Must be implemented by subclasses."""
        raise NotImplementedError

    def _create_masks(
        self, data: HeteroData, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42
    ) -> HeteroData:
        """Create stratified train/val/test masks on the target node type."""
        store = data[self.target]
        n = store.num_nodes
        y = store.y

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        g = torch.Generator().manual_seed(seed)

        for c in y.unique():
            class_idx = (y == c).nonzero(as_tuple=True)[0]
            nc = class_idx.size(0)
            perm = class_idx[torch.randperm(nc, generator=g)]

            n_train = max(1, int(nc * train_ratio))
            n_val = max(1, int(nc * val_ratio))
            if n_train + n_val >= nc:
                n_val = max(1, nc - n_train)
                if n_train + n_val >= nc:
                    n_train = max(1, nc - n_val - 1)

            train_mask[perm[:n_train]] = True
            val_mask[perm[n_train : n_train + n_val]] = True
            test_mask[perm[n_train + n_val :]] = True

        store.train_mask = train_mask
        store.val_mask = val_mask
        store.test_mask = test_mask
        return data

    def prepare_data(self) -> None:
        transform = T.Compose(
            [
                T.Constant(node_types=None),
                T.ToUndirected(),
                RemoveSelfLoops(),
                T.NormalizeFeatures(),
                T.RemoveDuplicatedEdges(),
            ]
        )

        data = self._load_data()
        data = self._create_masks(data)
        data = transform(data)
        data = data.coalesce()

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

    def train_dataloader(self):
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self):
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self):
        return self.pyg_datamodule.test_dataloader()


# =============================================================================
# Dataset 1: AS-733 (BGP Autonomous Systems Topology)
# =============================================================================

_AS_TIER1_THRESHOLD = 100
_AS_TIER2_THRESHOLD = 10


def _classify_as_tier(deg: int) -> int:
    if deg >= _AS_TIER1_THRESHOLD:
        return 0  # tier1
    elif deg >= _AS_TIER2_THRESHOLD:
        return 1  # tier2
    else:
        return 2  # stub


def _build_as733_hetero_data(
    edge_list: list[tuple[int, int]],
    node_feature_type: str = "degree",
) -> HeteroData:
    """Build HeteroData with single node type "as" and multiple edge types."""
    nodes = set()
    for u, v in edge_list:
        nodes.add(u)
        nodes.add(v)
    node_list = sorted(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)

    deg_count = Counter()
    for u, v in edge_list:
        deg_count[u] += 1
        deg_count[v] += 1

    node_tiers = {n: _classify_as_tier(deg_count[n]) for n in node_list}

    data = HeteroData()

    if node_feature_type == "degree":
        feats = [[math.log(deg_count[n] + 1), float(deg_count[n])] for n in node_list]
        data["as"].x = torch.tensor(feats, dtype=torch.float)
    else:
        data["as"].x = torch.ones(n_nodes, 1, dtype=torch.float)
    data["as"].num_nodes = n_nodes
    data["as"].y = torch.tensor([node_tiers[n] for n in node_list], dtype=torch.long)

    edge_groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for u, v in edge_list:
        src_tier = node_tiers[u]
        dst_tier = node_tiers[v]
        if src_tier > dst_tier:
            rel = "customer_provider"
        elif src_tier < dst_tier:
            rel = "provider_customer"
        else:
            rel = "peer_to_peer"
        edge_groups[rel][0].append(node_to_idx[u])
        edge_groups[rel][1].append(node_to_idx[v])

    for rel, (srcs, dsts) in edge_groups.items():
        if srcs:
            data["as", rel, "as"].edge_index = torch.tensor(
                [srcs, dsts], dtype=torch.long
            )

    return data


class AS733LPDataModule(LinkPredBase):
    """AS-733 BGP topology for link prediction (peering prediction)."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: Literal["constant", "degree"] = "degree",
        snapshot: str = "as20000102",
    ):
        self.snapshot = snapshot
        self._node_feature_type = node_feature_type
        super().__init__(
            data_dir=f"{data_dir}/as733",
            target=("as", "customer_provider", "as"),
            rev_target=("as", "rev_customer_provider", "as"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        os.makedirs(self.data_dir, exist_ok=True)
        tar_path = os.path.join(self.data_dir, "as-733.tar.gz")
        snapshot_path = os.path.join(self.data_dir, f"{self.snapshot}.txt")

        if not os.path.exists(snapshot_path):
            if not os.path.exists(tar_path):
                import urllib.request

                url = "https://snap.stanford.edu/data/as-733.tar.gz"
                print(f"Downloading AS-733 dataset from {url}...")
                urllib.request.urlretrieve(url, tar_path)

            print("Extracting AS-733 dataset...")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(self.data_dir)

        edges = []
        with open(snapshot_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))

        data = _build_as733_hetero_data(edges, self._node_feature_type)

        transform = T.Compose(
            [T.Constant(), T.ToUndirected(), T.NormalizeFeatures(), RemoveSelfLoops()]
        )
        return transform(data)

    def __repr__(self):
        return "AS733-LP"


# =============================================================================
# Dataset 2: RouteNet (SDN Traffic Engineering)
# =============================================================================


def _parse_routenet_sample(
    topology_graph,
    routing_matrix: np.ndarray,
    traffic_matrix: np.ndarray,
    performance_matrix: np.ndarray,
    port_stats=None,
    congestion_threshold: float = 0.3,
) -> HeteroData:
    """Build a HeteroData from a single RouteNet simulation sample."""
    data = HeteroData()
    n_nodes = len(topology_graph.nodes)

    # Extract links (force 2-tuples, deduplicate)
    seen = set()
    links = []
    for u, v, *_ in topology_graph.edges:
        if (u, v) not in seen:
            seen.add((u, v))
            links.append((u, v))

    link_idx = {(u, v): i for i, (u, v) in enumerate(links)}
    for (u, v), i in list(link_idx.items()):
        if (v, u) not in link_idx:
            link_idx[(v, u)] = i
    n_links = len(links)

    # Link features
    link_feats = []
    for u, v in links:
        edge_data = topology_graph[u][v]
        # MultiDiGraph returns AtlasView {0: {attrs}}; unwrap to the actual attrs dict
        if not isinstance(edge_data, dict) and hasattr(edge_data, '__getitem__'):
            try:
                edge_data = edge_data[0]
            except (KeyError, IndexError):
                pass
        elif isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]
        capacity = float(edge_data.get("bandwidth", edge_data.get("weight", 1.0)))
        link_feats.append([capacity])

    data["link"].x = torch.tensor(link_feats, dtype=torch.float) if link_feats else torch.zeros(0, 1)
    data["link"].num_nodes = n_links

    # Extract paths from routing matrix
    paths = []
    path_feats = []
    path_link_src = []
    path_link_dst = []

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue
            route = routing_matrix[src][dst]
            if route is None or (isinstance(route, (list, np.ndarray)) and len(route) < 2):
                continue

            route = list(route) if isinstance(route, np.ndarray) else route
            path_idx = len(paths)
            paths.append((src, dst, route))

            # Traffic
            traffic_val = 0.0
            if traffic_matrix is not None:
                t_entry = traffic_matrix[src, dst]
                if isinstance(t_entry, dict):
                    traffic_val = float(t_entry.get("AggInfo", {}).get("AvgBw", 0.0))
                else:
                    traffic_val = float(t_entry)

            # Delay
            delay_val = 0.0
            if performance_matrix is not None:
                p_entry = performance_matrix[src, dst]
                if isinstance(p_entry, dict):
                    delay_val = float(p_entry.get("AggInfo", {}).get("AvgDelay", 0.0))
                else:
                    delay_val = float(p_entry)

            path_feats.append([traffic_val, float(len(route) - 1), delay_val])

            for i in range(len(route) - 1):
                lk = (route[i], route[i + 1])
                if lk in link_idx:
                    path_link_src.append(path_idx)
                    path_link_dst.append(link_idx[lk])

    data["path"].x = torch.tensor(path_feats, dtype=torch.float) if path_feats else torch.zeros(0, 3)
    data["path"].num_nodes = len(paths)

    if path_link_src:
        data["path", "traverses", "link"].edge_index = torch.tensor(
            [path_link_src, path_link_dst], dtype=torch.long
        )
        data["link", "belongs_to", "path"].edge_index = torch.tensor(
            [path_link_dst, path_link_src], dtype=torch.long
        )

    # Link congestion labels
    if port_stats is not None and isinstance(port_stats, list) and len(port_stats) > 0:
        link_labels = []
        for u, v in links:
            util = 0.0
            if u < len(port_stats) and isinstance(port_stats[u], dict) and v in port_stats[u]:
                util = float(port_stats[u][v].get("utilization", 0.0))
            link_labels.append(1 if util > congestion_threshold else 0)
        data["link"].y = torch.tensor(link_labels, dtype=torch.long)
    else:
        link_load = torch.zeros(n_links)
        for path_i, (src, dst, route) in enumerate(paths):
            traffic = path_feats[path_i][0] if path_feats else 0.0
            for j in range(len(route) - 1):
                lk = (route[j], route[j + 1])
                if lk in link_idx:
                    link_load[link_idx[lk]] += traffic
        capacities = torch.tensor([f[0] for f in link_feats], dtype=torch.float).clamp(min=1e-6)
        utilization = link_load / capacities
        data["link"].y = (utilization > congestion_threshold).long()

    return data


def _load_datanet_api(data_dir: str, experiment: str):
    """Load the datanetAPI module. Returns (DatanetAPI class, train_dir)."""
    # datanetAPI.py may be in data_dir or one level up
    api_path = os.path.join(data_dir, "datanetAPI.py")
    if not os.path.exists(api_path):
        parent = os.path.dirname(data_dir)
        api_path_parent = os.path.join(parent, "datanetAPI.py")
        if os.path.exists(api_path_parent):
            api_path = api_path_parent
    train_dir = os.path.join(data_dir, "train")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"RouteNet data not found at {data_dir}. "
            f"Please download the RouteNet-Erlang dataset:\n"
            f"  1. Clone https://github.com/BNN-UPC/RouteNet-Erlang\n"
            f"  2. Download data: wget -O {experiment}.zip "
            f"https://bnn.upc.edu/download/dataset-v5-{experiment}/\n"
            f"  3. Extract to {data_dir}/\n"
            f"  4. Copy datanetAPI.py from the RouteNet-Erlang repo"
        )

    import importlib.util

    if not os.path.exists(api_path):
        raise FileNotFoundError(
            f"datanetAPI.py not found at {api_path}. "
            f"Copy it from the RouteNet-Erlang repository."
        )

    spec = importlib.util.spec_from_file_location("datanetAPI", api_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    return api_module.DatanetAPI, train_dir


def _load_routenet_hetero_data(
    data_dir: str, experiment: str, congestion_threshold: float, num_samples: int = 10
) -> HeteroData:
    """Load RouteNet data, averaging utilization across multiple samples.

    Handles both datasets with port_stats (scalability) and without (scheduling,
    traffic_models) by falling back to traffic/capacity-based utilization estimation.
    """
    DatanetAPI, train_dir = _load_datanet_api(data_dir, experiment)

    api = DatanetAPI(train_dir, shuffle=False)
    sample = next(iter(api))

    topo = sample.get_topology_object()
    routing = sample.get_routing_matrix()
    traffic = sample.get_traffic_matrix()
    perf = sample.get_performance_matrix()

    has_port_stats = hasattr(sample, "get_port_stats")

    if has_port_stats:
        # Accumulate real utilization across multiple samples
        link_util_accum: dict[tuple[int, int], list[float]] = defaultdict(list)
        n_nodes = len(topo.nodes)

        def _accum_port_stats(ps, nn):
            if ps is not None and isinstance(ps, list):
                for node_idx in range(min(nn, len(ps))):
                    if isinstance(ps[node_idx], dict):
                        for neigh, stats in ps[node_idx].items():
                            link_util_accum[(node_idx, neigh)].append(
                                float(stats.get("utilization", 0.0))
                            )

        _accum_port_stats(sample.get_port_stats(), n_nodes)

        for i, extra_sample in enumerate(api):
            if i >= num_samples - 1:
                break
            extra_ps = extra_sample.get_port_stats() if hasattr(extra_sample, "get_port_stats") else None
            extra_nn = len(extra_sample.get_topology_object().nodes)
            _accum_port_stats(extra_ps, extra_nn)

        avg_port_stats: list[dict] = [{} for _ in range(n_nodes)]
        for (u, v), utils_list in link_util_accum.items():
            if u < n_nodes:
                avg_port_stats[u][v] = {"utilization": sum(utils_list) / len(utils_list)}

        return _parse_routenet_sample(
            topo, routing, traffic, perf, avg_port_stats, congestion_threshold
        )
    else:
        # No port_stats (scheduling, some traffic_models): pass None and let
        # _parse_routenet_sample estimate utilization from traffic/capacity
        return _parse_routenet_sample(
            topo, routing, traffic, perf, None, congestion_threshold
        )


class RouteNetNCDataModule(NetworkingNCBaseDataModule):
    """RouteNet dataset for node classification (link congestion detection).

    Data source: RouteNet-Erlang (https://github.com/BNN-UPC/RouteNet-Erlang)
    """

    def __init__(
        self,
        data_dir: str = "data/routenet",
        homogeneous: bool = False,
        node_feature_type: Literal["constant", "degree"] = "degree",
        experiment: str = "scalability",
        congestion_threshold: float = 0.3,
        num_samples: int = 10,
    ):
        self.experiment = experiment
        self.congestion_threshold = congestion_threshold
        self.num_samples = num_samples
        super().__init__(
            target="link",
            num_classes=2,
            task="multiclass",
            data_dir=os.path.join(data_dir, experiment),
            homogeneous=homogeneous,
            node_feature_type=node_feature_type,
        )

    def _load_data(self) -> HeteroData:
        return _load_routenet_hetero_data(
            self.data_dir, self.experiment, self.congestion_threshold, self.num_samples
        )

    def __repr__(self):
        return f"RouteNet-{self.experiment}-NC"
