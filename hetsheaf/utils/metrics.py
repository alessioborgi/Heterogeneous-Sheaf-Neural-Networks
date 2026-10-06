#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, F1Score
from torchmetrics.collections import MetricCollection


def get_link_prediction_metrics(prefix: str) -> MetricCollection:
    """Returns the binary link-prediction metrics reported in the paper.

    Args:
        prefix: Prefix to prepend to each metric name (e.g. "valid").

    Returns:
        MetricCollection: accuracy, AUROC, AUPR and macro/micro F1.
    """
    metrics = {
        "accuracy": Accuracy(task="binary"),
        "AUROC": AUROC(task="binary"),
        "AUPR": AveragePrecision(task="binary"),
        "F1_macro": F1Score(task="binary", average="macro"),
        "F1_micro": F1Score(task="binary", average="micro"),
    }

    return MetricCollection(metrics, prefix=f"{prefix}/")
