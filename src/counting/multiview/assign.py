"""Grid-cell row and tree assignment of world points.

A point is in the target row when `|y - row_y| < row_spacing / 2`. Inside the
row, tree cells are split at the midpoints between neighbouring tree bases;
a point exactly on a midpoint goes to the right-hand tree, and points past
either row end go to the end tree.
"""

from __future__ import annotations

import numpy as np

from counting.data.formats.orchard_synth import OrchardLayout


def in_target_row(X: np.ndarray, layout: OrchardLayout) -> np.ndarray:
    """Which points lie in the target row: `|y - row_y| < row_spacing / 2`; non-finite points are False."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    return np.isfinite(X).all(axis=1) & (np.abs(X[:, 1] - layout.row_y) < layout.row_spacing / 2)


def assign_trees(X: np.ndarray, layout: OrchardLayout) -> np.ndarray:
    """Tree id of each point's grid cell; -1 outside the target row and for non-finite points."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    tree_x = np.asarray(layout.tree_x, dtype=float)
    mids = (tree_x[:-1] + tree_x[1:]) / 2
    row = in_target_row(X, layout)
    cell = np.searchsorted(mids, X[:, 0], side="right")   # NaN sorts last: a valid index, masked below
    return np.where(row, np.asarray(layout.tree_ids, dtype=int)[cell], -1)
