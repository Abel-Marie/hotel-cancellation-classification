"""Apply the row-level features to DataFrames and build the fitted preprocessors.

Everything learned from data (category grouping, encoders, scaling) lives in
the scikit-learn preprocessors returned here, so it is refitted inside every
cross-validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .rowlevel import CATEGORICAL_FEATURES, derive_features


def build_feature_frame(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """Run ``derive_features`` on every row; the index is kept so labels stay aligned."""
    rows = [derive_features(r, sources) for r in df.to_dict("records")]
    return pd.DataFrame(rows, index=df.index)


def split_feature_types(columns: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in columns if c in CATEGORICAL_FEATURES]
    nums = [c for c in columns if c not in CATEGORICAL_FEATURES]
    return cats, nums


def to_float32(X) -> np.ndarray:
    """Trees are trained on float32 so the ONNX copy, which runs in float32, matches them."""
    return np.asarray(X, dtype=np.float32)


def make_tree_preprocessor(cats: list[str], nums: list[str], min_frequency: int) -> list:
    """Integer codes for categories (rare ones grouped, unseen ones -1), numbers unchanged."""
    columns = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                min_frequency=min_frequency), cats),
         ("num", "passthrough", nums)],
        verbose_feature_names_out=False,
    )
    return [("prep", columns), ("float32", FunctionTransformer(to_float32))]


def make_linear_preprocessor(cats: list[str], nums: list[str], min_frequency: int) -> list:
    """One-hot categories (rare and unseen ones share a column), standardised numbers."""
    columns = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                               sparse_output=False), cats),
         ("num", StandardScaler(), nums)],
        verbose_feature_names_out=False,
    )
    return [("prep", columns)]
