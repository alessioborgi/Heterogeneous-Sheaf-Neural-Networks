from typing import Optional, Callable, List, Literal
import torch
import lightning as L
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree

class ExtractTypes:
    """
    Transform that extracts node_type and edge_type from graph data.
    Replaces extracted type columns with a single feature only if removing types leaves no features.
    
    Args:
        node_type_cols: [start, end] range of columns containing one-hot node types.
                       If None, no node types are extracted.
        edge_type_cols: [start, end] range of columns containing one-hot edge types.
                       If None, no edge types are extracted.
        node_feature_type: Type of features to create - "constant" (ones) or "degree" (node degrees).
    """
    
    def __init__(
        self, 
        node_type_cols: Optional[List[int]] = None,
        edge_type_cols: Optional[List[int]] = None,
        node_feature_type: Literal["constant", "degree"] = "constant",
    ):
        self.node_type_cols = node_type_cols
        self.edge_type_cols = edge_type_cols
        self.node_feature_type = node_feature_type
    
    def __call__(self, data: Data) -> Data:
        # Handle case where data.x is None - create features based on node_feature_type
        if data.x is None:
            if self.node_feature_type == "degree":
                # Compute node degrees and use as features
                deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
                data.x = deg.view(-1, 1)
            else:  # constant
                # Create constant features (ones) with dimension 1
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
        
        # Extract node types and replace with a single feature only if needed
        node_type = None
        if self.node_type_cols is not None and data.x is not None:
            start, end = self.node_type_cols
            if data.x.size(1) >= end:
                # Extract types
                node_type = data.x[:, start:end].argmax(dim=1)
                
                # Remove type columns
                if start == 0 and end >= data.x.size(1):
                    # All columns are types - create single replacement feature
                    if self.node_feature_type == "degree":
                        deg = degree(data.edge_index[0], num_nodes=data.x.size(0), dtype=data.x.dtype)
                        data.x = deg.view(-1, 1)
                    else:  # constant
                        data.x = torch.ones(data.x.size(0), 1, dtype=data.x.dtype)
                elif start == 0:
                    # Types at the beginning
                    data.x = data.x[:, end:]
                elif end >= data.x.size(1):
                    # Types at the end
                    data.x = data.x[:, :start]
                else:
                    # Types in the middle
                    data.x = torch.cat([data.x[:, :start], data.x[:, end:]], dim=1)
        else:
            # No node types extracted - create homogeneous (all zeros)
            node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        
        # Extract edge types and replace with a single feature only if needed
        edge_type = None
        if self.edge_type_cols is not None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            start, end = self.edge_type_cols
            if data.edge_attr.dim() == 1:
                # Already integer types
                edge_type = data.edge_attr.long()
            elif data.edge_attr.dim() == 2 and data.edge_attr.size(1) >= end:
                # Extract types
                edge_type = data.edge_attr[:, start:end].argmax(dim=1)

                # Remove type columns only if needed
                if start == 0 and end >= data.edge_attr.size(1):
                    # All columns are types - create single replacement feature
                    data.edge_attr = torch.ones(data.edge_attr.size(0), 1, dtype=data.edge_attr.dtype)
                elif start == 0:
                    data.edge_attr = data.edge_attr[:, end:]
                elif end >= data.edge_attr.size(1):
                    data.edge_attr = data.edge_attr[:, :start]
                else:
                    data.edge_attr = torch.cat([data.edge_attr[:, :start], data.edge_attr[:, end:]], dim=1)
        else:
            # No edge types extracted - create homogeneous (all zeros)
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                num_edges = data.edge_index.size(1)
                edge_type = torch.zeros(num_edges, dtype=torch.long)

        # Store types
        data.node_type = node_type
        data.edge_type = edge_type
        
        # Batch vector for pooling (required since batch_size=1)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        
        return data
    
    def __repr__(self) -> str:
        return f"ExtractTypes(node_type_cols={self.node_type_cols}, edge_type_cols={self.edge_type_cols})"


class SetHomogeneous:
    """Transform that forces homogeneous mode: sets all types to 0 (single type) and ensures features exist."""
    
    def __init__(self, node_feature_type: Literal["constant", "degree"] = "constant"):
        self.node_feature_type = node_feature_type
    
    def __call__(self, data: Data) -> Data:
        # Ensure data.x exists - create features based on node_feature_type
        if data.x is None:
            if self.node_feature_type == "degree":
                deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
                data.x = deg.view(-1, 1)
            else:  # constant
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
        
        # Set dummy types (all nodes/edges are type 0 in homogeneous graphs)
        data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        
        # Set edge types if edge_index exists
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            num_edges = data.edge_index.size(1)
            data.edge_type = torch.zeros(num_edges, dtype=torch.long)
        else:
            data._store.__dict__["edge_type"] = None
            data.edge_type = None
        
        # Batch vector for pooling (required since batch_size=1)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
    
    def __repr__(self) -> str:
        return "SetHomogeneous()"


class GraphClassificationDataModule(L.LightningDataModule):
    """
    Unified graph classification datamodule.
    
    Supports both homogeneous and heterogeneous graphs via config:
    - homogeneous=True: forces all types to None
    - homogeneous=False: extracts types from specified columns
    
    Args:
        root: Path to dataset root directory
        name: Name of TU dataset
        homogeneous: If True, ignore type information
        node_type_cols: [start, end] columns for one-hot node types
        edge_type_cols: [start, end] columns for one-hot edge types
        node_feature_type: Type of node features - "constant" (ones) or "degree" (node degrees)
        num_workers: DataLoader workers
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed for splitting
    """
    
    def __init__(
        self,
        root: str,
        name: str,
        homogeneous: bool = False,
        node_type_cols: Optional[List[int]] = None,
        edge_type_cols: Optional[List[int]] = None,
        node_feature_type: Literal["constant", "degree"] = "constant",
        num_workers: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        # Allow configs to pass an explicit task without polluting logged hparams.
        self._task_override = kwargs.pop("task", None)
        self.root = root
        self.name = name
        self.homogeneous = homogeneous
        self.node_type_cols = node_type_cols
        self.edge_type_cols = edge_type_cols
        self.node_feature_type = node_feature_type
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        
        # Type extraction at runtime (not cached) to respect homogeneous flag
        if homogeneous:
            type_transform = SetHomogeneous(node_feature_type=node_feature_type)
        else:
            type_transform = ExtractTypes(
                node_type_cols=node_type_cols,
                edge_type_cols=edge_type_cols,
                node_feature_type=node_feature_type,
            )
        
        # Compose with user transform
        if transform is not None:
            self.transform = Compose([transform, type_transform])
        else:
            self.transform = type_transform
        
        self.pre_transform = pre_transform

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Computed after setup
        self._num_node_types = None
        self._num_edge_types = None
        self._is_heterogeneous = None
        self._gc_metadata = None
        
        self.save_hyperparameters()

    def prepare_data(self) -> Dataset:
        return TUDataset(
            root=self.root, 
            name=self.name, 
            transform=self.transform, 
            pre_transform=self.pre_transform,
            force_reload=True
        )

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = self.prepare_data()
                        
            n = len(self.dataset)
            n_train = int(n * self.train_split)
            n_val = int(n * self.val_split)

            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=generator)

            self.train_dataset = self.dataset[perm[:n_train]]
            self.val_dataset = self.dataset[perm[n_train:n_train + n_val]]
            self.test_dataset = self.dataset[perm[n_train + n_val:]]
            
            # Compute type counts
            self._compute_type_info()

    def _compute_type_info(self):
        """Compute the number of unique node and edge types across the dataset."""
        if self.homogeneous:
            self._num_node_types = 1
            self._num_edge_types = 1
            self._is_heterogeneous = False
            self._gc_metadata = (["t0"], [("t0", "e0", "t0")])
            return

        max_node_type = -1
        max_edge_type = -1
        has_node_types = False
        has_edge_types = False
        node_type_set: set = set()
        edge_triple_set: set = set()

        for data in self.dataset:
            if hasattr(data, 'node_type') and data.node_type is not None:
                has_node_types = True
                max_node_type = max(max_node_type, data.node_type.max().item())
                node_type_set.update(data.node_type.unique().tolist())
            if hasattr(data, 'edge_type') and data.edge_type is not None and data.edge_type.numel() > 0:
                has_edge_types = True
                max_edge_type = max(max_edge_type, data.edge_type.max().item())
                src_nt = data.node_type[data.edge_index[0]]
                dst_nt = data.node_type[data.edge_index[1]]
                for et in data.edge_type.unique().tolist():
                    mask = data.edge_type == et
                    pairs = torch.stack([src_nt[mask], dst_nt[mask]], dim=1).unique(dim=0)
                    for pair in pairs:
                        edge_triple_set.add(
                            (f"t{pair[0].item()}", f"e{et}", f"t{pair[1].item()}")
                        )

        self._num_node_types = max_node_type + 1 if has_node_types else 1
        self._num_edge_types = max_edge_type + 1 if has_edge_types else 1
        self._is_heterogeneous = has_node_types or has_edge_types

        node_types = [f"t{nt}" for nt in sorted(node_type_set)] if node_type_set else ["t0"]
        edge_types = sorted(edge_triple_set) if edge_triple_set else [("t0", "e0", "t0")]
        self._gc_metadata = (node_types, edge_types)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=self.num_workers, 
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=self.num_workers > 0
        )

    @property
    def num_node_features(self) -> int:
        if self.dataset is None: 
            self.setup()
        return self.dataset.num_node_features

    @property
    def num_classes(self) -> int:
        if self.dataset is None: 
            self.setup()
        return self.dataset.num_classes

    @property
    def num_edge_types(self) -> int:
        if self.dataset is None:
            self.setup()
        return self._num_edge_types

    @property
    def num_node_types(self) -> int:
        if self.dataset is None:
            self.setup()
        return self._num_node_types

    @property
    def is_heterogeneous(self) -> bool:
        if self.dataset is None:
            self.setup()
        return self._is_heterogeneous

    @property
    def gc_metadata(self):
        """HeteroData metadata (node types, edge type triples) for this dataset.
        Only the combinations that actually appear across all graphs are included,
        keeping HAN/HGT parameter counts small."""
        if self.dataset is None:
            self.setup()
        return self._gc_metadata

    @property
    def task(self) -> str:
        if self._task_override is not None:
            return self._task_override
        return "binary" if self.num_classes == 2 else "multiclass"
    
    def __repr__(self) -> str:
        if self._is_heterogeneous is None:
            return f"{self.name}(not setup)"
        hetero_str = "Hetero" if self._is_heterogeneous else "Homo"
        return f"{self.name}({hetero_str}, {self._num_node_types} node types, {self._num_edge_types} edge types)"


# =============================================================================
# Dataset-specific datamodules
# =============================================================================

class MUTAGDataModule(GraphClassificationDataModule):
    """
    MUTAG: 188 mutagenic compounds.
    - 7 atom types (C, N, O, F, I, Cl, Br) in columns [0, 7]
    - 4 bond types (aromatic, single, double, triple) in columns [0, 4]
    """
    def __init__(self, root: str = "data/MUTAG", **kwargs):
        kwargs.setdefault("node_type_cols", [0, 7])
        kwargs.setdefault("edge_type_cols", [0, 4])
        super().__init__(root=root, name="MUTAG", **kwargs)


class PROTEINSDataModule(GraphClassificationDataModule):
    """
    PROTEINS: 1113 proteins.
    - 3 node types (secondary structure) in columns [0, 3]
    - No edge types
    """
    def __init__(self, root: str = "data/PROTEINS", **kwargs):
        kwargs.setdefault("node_type_cols", [0, 3])
        kwargs.setdefault("edge_type_cols", None)
        super().__init__(root=root, name="PROTEINS", **kwargs)


class ENZYMESDataModule(GraphClassificationDataModule):
    """
    ENZYMES: 600 enzymes, 6 classes.
    - 3 node types in columns [0, 3]
    - No edge types
    """
    def __init__(self, root: str = "data/ENZYMES", **kwargs):
        kwargs.setdefault("node_type_cols", [0, 3])
        kwargs.setdefault("edge_type_cols", None)
        super().__init__(root=root, name="ENZYMES", **kwargs)


class NCI1DataModule(GraphClassificationDataModule):
    """
    NCI1: 4110 compounds.
    - 37 atom types in columns [0, 37]
    - No edge types
    """
    def __init__(self, root: str = "data/NCI1", **kwargs):
        kwargs.setdefault("node_type_cols", [0, 37])
        kwargs.setdefault("edge_type_cols", None)
        super().__init__(root=root, name="NCI1", **kwargs)
