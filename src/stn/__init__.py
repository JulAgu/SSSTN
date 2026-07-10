"""
Search Trajectory Networks (STN) for NN training multiruns.

Python port of the euroGP R pipeline (Gabriela Ochoa et al.) applied to neural
network training trajectories in semantic (logit) space.  Each ``results/*.pkl``
multirun file is one training run (one seed/init); a condition is one
(dataset, label-type) pair whose seeds play the role of euroGP's "runs".

Modules:
    io        -- discover/group multiruns, build the long trajectory table
    location  -- location functions (logits snapshot -> node id)
    build     -- construct the igraph STN from a located trajectory table
    metrics   -- network metrics per STN (port of metrics.R)
    plot      -- render STNs to PNG (port of plot.R)
"""

from .io import discover_conditions, load_condition, save_edgelist, save_membership
from .location import hamming_label, regression_hamming
from .build import build_stn, STN
from .metrics import stn_metrics

__all__ = [
    "discover_conditions",
    "load_condition",
    "save_edgelist",
    "save_membership",
    "hamming_label",
    "regression_hamming",
    "build_stn",
    "STN",
    "stn_metrics",
]
