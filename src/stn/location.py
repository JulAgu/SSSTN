"""Location functions: map a logits snapshot to an STN node id.

The location function is the key STN design choice.  Two are provided:

* :func:`hamming_label` -- classification-native: a snapshot is reduced to its
  predicted-label vector (``argmax`` over categories) and snapshots are grouped
  by (normalised) Hamming distance.  It traces the trajectory through
  *decision-function* space rather than logit space.
* :func:`regression_hamming` -- regression twin: predicted values are quantized
  into global quantile bins, then snapshots are grouped by Hamming distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def prediction_vectors(descriptors: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Categorical predicted-label vectors (the classification phenotype).

    Reshape each flattened logits row to ``(n_train, n_classes)`` and argmax over
    the class axis.

    Parameters
    ----------
    descriptors : numpy.array
        (n_rows, n_train * n_classes) flattened logits
    n_classes : int

    Returns
    -------
    preds : numpy.array
        (n_rows, n_train) predicted class per training sample
    """
    n_rows = descriptors.shape[0]
    return descriptors.reshape(n_rows, -1, n_classes).argmax(axis=2)


def quantized_vectors(descriptors: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Global-quantile-binned prediction vectors (the regression phenotype).

    Each predicted value is binned into one of ``n_bins`` global quantile
    categories, so a bin denotes the same value range across every snapshot.
    Tied edges are dropped, degrading gracefully to fewer bins.

    Parameters
    ----------
    descriptors : numpy.array
    n_bins : int

    Returns
    -------
    codes : numpy.array
        (n_rows, n_train) integer bin code per predicted value
    """
    n_rows = descriptors.shape[0]
    values = descriptors.reshape(n_rows, -1)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.digitize(values, edges).astype(np.int64)


def hamming_label(
    descriptors: np.ndarray,
    n_classes: int,
    threshold: float = 0.1,
    max_unique: int = 16000,
) -> np.ndarray:
    """
    Node ids from predicted-label vectors, grouped by Hamming distance.

    Each snapshot's prediction vector (argmax over classes) is grouped by
    normalised Hamming distance with complete-linkage agglomerative clustering,
    so snapshots sharing a node disagree on at most ``threshold`` of the samples.
    Clustering runs on the distinct vectors only, then broadcasts back.

    Parameters
    ----------
    descriptors : numpy.array
    n_classes : int
    threshold : float
        max normalised Hamming distance (fraction of samples allowed to
        disagree), in (0, 1]
    max_unique : int
        raise ValueError if the distinct-vector count exceeds this

    Returns
    -------
    names : numpy.array
        string node ids ("pred_<k>"), one per row
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a normalised Hamming distance in (0, 1].")

    preds = prediction_vectors(descriptors, n_classes)  # (n_rows, N)

    # Collapse identical prediction vectors so clustering runs on the (far
    # smaller) set of distinct categorical phenotypes; `inverse` maps each row
    # back to its unique vector.
    uniq, inverse = np.unique(preds, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # numpy can return a (n_rows, 1) inverse

    if len(uniq) > max_unique:
        raise ValueError(
            f"{len(uniq)} distinct prediction vectors exceed max_unique={max_unique}; "
            "the O(U^2) Hamming clustering would be too large. Coarsen the trace "
            "(larger log_every) or raise max_unique deliberately."
        )

    if len(uniq) == 1:
        cluster_of_uniq = np.zeros(1, dtype=np.int64)
    else:
        cluster_of_uniq = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="hamming",
            linkage="complete",
        ).fit_predict(uniq)

    node_of_row = cluster_of_uniq[inverse]
    return np.array([f"pred_{c}" for c in node_of_row], dtype=object)


def regression_hamming(
    descriptors: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.1,
    max_unique: int = 16000,
) -> np.ndarray:
    """
    Regression twin of :func:`hamming_label`.

    Predicted values are quantized into ``n_bins`` global quantile bins, then
    snapshots are grouped by normalised Hamming distance over the binned vectors
    (complete-linkage agglomerative, on the distinct vectors only). Blind to
    within-bin differences.

    Parameters
    ----------
    descriptors : numpy.array
    n_bins : int
        number of global quantile bins (10 = deciles)
    threshold : float
        max normalised Hamming distance, in (0, 1]
    max_unique : int
        raise ValueError if the distinct-vector count exceeds this

    Returns
    -------
    names : numpy.array
        string node ids ("qbin_<k>"), one per row
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a normalised Hamming distance in (0, 1].")

    codes = quantized_vectors(descriptors, n_bins=n_bins)  # (n_rows, N)

    # Collapse identical quantized vectors so clustering runs on the (far smaller)
    # set of distinct categorical phenotypes; `inverse` maps each row back.
    uniq, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # numpy can return a (n_rows, 1) inverse

    if len(uniq) > max_unique:
        raise ValueError(
            f"{len(uniq)} distinct quantized vectors exceed max_unique={max_unique}; "
            "the O(U^2) Hamming clustering would be too large. Coarsen the trace "
            "(larger log_every), use fewer n_bins, or raise max_unique deliberately."
        )

    if len(uniq) == 1:
        cluster_of_uniq = np.zeros(1, dtype=np.int64)
    else:
        cluster_of_uniq = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="hamming",
            linkage="complete",
        ).fit_predict(uniq)

    node_of_row = cluster_of_uniq[inverse]
    return np.array([f"qbin_{c}" for c in node_of_row], dtype=object)
