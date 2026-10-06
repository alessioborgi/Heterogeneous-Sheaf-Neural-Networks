#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

# HGB node classification + link prediction
from .hgb import (
    HGBBaseDataModule,
    IMDBDataModule,
    DBLPDataModule,
    ACMDataModule,
    HGBLinkPredBase,
    HGBAmazonLPDataModule,
    HGBPubMedLPDataModule,
    HGBYouTubeLPDataModule,
)

# Link prediction
from .link_pred import (
    LinkPredBase,
    LastFMDataModule,
    MovieLensDataModule,
)

# Biomedical link prediction
from .biomedical_dataset import (
    BiomedicalDataset,
    CTD_DDADataModule,
    NDFRT_DDADataModule,
    DrugBankDDIDataModule,
    STRINGPPIDataModule,
)

# Networking
from .networking import (
    NetworkingNCBaseDataModule,
    AS733LPDataModule,
    RouteNetNCDataModule,
)

# Graph classification (TU datasets)
from .graph_classification import (
    GraphClassificationDataModule,
    MUTAGDataModule,
    PROTEINSDataModule,
    ENZYMESDataModule,
    NCI1DataModule,
)

__all__ = [
    "HGBBaseDataModule",
    "IMDBDataModule",
    "DBLPDataModule",
    "ACMDataModule",
    "HGBLinkPredBase",
    "HGBAmazonLPDataModule",
    "HGBPubMedLPDataModule",
    "HGBYouTubeLPDataModule",
    "LinkPredBase",
    "LastFMDataModule",
    "MovieLensDataModule",
    "BiomedicalDataset",
    "CTD_DDADataModule",
    "NDFRT_DDADataModule",
    "DrugBankDDIDataModule",
    "STRINGPPIDataModule",
    "NetworkingNCBaseDataModule",
    "AS733LPDataModule",
    "RouteNetNCDataModule",
    "GraphClassificationDataModule",
    "MUTAGDataModule",
    "PROTEINSDataModule",
    "ENZYMESDataModule",
    "NCI1DataModule",
]
