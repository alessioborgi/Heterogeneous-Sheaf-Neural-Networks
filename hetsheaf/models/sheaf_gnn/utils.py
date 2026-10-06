#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from hetsheaf.models.sheaf_gnn.config import SheafLearners
from hetsheaf.models.sheaf_gnn.sheaf_models import (
    # Homogeneous Learners
    LocalConcatSheafLearner,
    AttentionSheafLearner,
    # Heterogeneous Learners
    TypeConcatSheafLearner,
    TypeEnsembleSheafLearner,
    EdgeTypeConcatSheafLearner,
    NodeTypeConcatSheafLearner,
    TypesOnlySheafLearner,
    NodeTypeSheafLearner,
    EdgeTypeSheafLearner,
    # Attention-based Heterogeneous Learners
    AttentionTypeConcatSheafLearner,
    AttentionTypeEnsembleSheafLearner,
    AttentionEdgeEncodingSheafLearner,
    AttentionNodeEncodingSheafLearner,
    AttentionTypesOnlySheafLearner,
    AttentionNodeTypeSheafLearner,
    AttentionEdgeTypeSheafLearner,
)


def init_sheaf_learner(sheaf_type):
    """Initialize a sheaf learner based on the given type.

    Homogeneous Learners (feature-only, no type information):
    - local_concat: Concatenates node features [x_u || x_v]
    - attention: Learns attention-based restriction maps

    Heterogeneous Learners (incorporate node types and/or edge types):
    - type_concat (TE): Features + both node types + edge type
    - type_ensemble (ensemble): Different MLP per edge type
    - edge_type_concat (EE): Features + edge type only
    - node_type_concat (NE): Features + both node types only
    - types_only: All type information, NO features
    - node_type (NT): Node types only, NO features or edge type
    - edge_type (ET): Edge type only, NO features or node types

    Attention-based Heterogeneous Learners (produce row-stochastic attention matrices):
    - attention_type_concat: Attention + Features + both node types + edge type
    - attention_type_ensemble: Attention + Different MLP per edge type
    - attention_edge_encoding: Attention + Features + edge type only
    - attention_node_encoding: Attention + Features + both node types only
    - attention_types_only: Attention + All type information, NO features
    - attention_node_type (Attention-NT): Attention + Node types only, NO features or edge type
    - attention_edge_type (Attention-ET): Attention + Edge type only, NO features or node types
    """
    # HOMOGENEOUS LEARNERS
    if sheaf_type == SheafLearners.local_concat:
        sheaf_learner = LocalConcatSheafLearner
    elif sheaf_type == SheafLearners.attention:
        sheaf_learner = AttentionSheafLearner

    # HETEROGENEOUS LEARNERS
    elif sheaf_type == SheafLearners.type_concat:
        sheaf_learner = TypeConcatSheafLearner
    elif sheaf_type == SheafLearners.type_ensemble:
        sheaf_learner = TypeEnsembleSheafLearner
    elif sheaf_type == SheafLearners.edge_type_concat:
        sheaf_learner = EdgeTypeConcatSheafLearner
    elif sheaf_type == SheafLearners.node_type_concat:
        sheaf_learner = NodeTypeConcatSheafLearner
    elif sheaf_type == SheafLearners.types_only:
        sheaf_learner = TypesOnlySheafLearner
    elif sheaf_type == SheafLearners.node_type:
        sheaf_learner = NodeTypeSheafLearner
    elif sheaf_type == SheafLearners.edge_type:
        sheaf_learner = EdgeTypeSheafLearner

    # ATTENTION-BASED HETEROGENEOUS LEARNERS
    elif sheaf_type == SheafLearners.attention_type_concat:
        sheaf_learner = AttentionTypeConcatSheafLearner
    elif sheaf_type == SheafLearners.attention_type_ensemble:
        sheaf_learner = AttentionTypeEnsembleSheafLearner
    elif sheaf_type == SheafLearners.attention_edge_encoding:
        sheaf_learner = AttentionEdgeEncodingSheafLearner
    elif sheaf_type == SheafLearners.attention_node_encoding:
        sheaf_learner = AttentionNodeEncodingSheafLearner
    elif sheaf_type == SheafLearners.attention_types_only:
        sheaf_learner = AttentionTypesOnlySheafLearner
    elif sheaf_type == SheafLearners.attention_node_type:
        sheaf_learner = AttentionNodeTypeSheafLearner
    elif sheaf_type == SheafLearners.attention_edge_type:
        sheaf_learner = AttentionEdgeTypeSheafLearner
    else:
        raise ValueError(f"Unknown sheaf learner type: {sheaf_type}")

    return sheaf_learner
