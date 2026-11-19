"""Model explanations, always computed on validation data, never on training data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .config import Config


def permutation_importance_frame(model, X_val: pd.DataFrame, y_val, cfg: Config) -> pd.DataFrame:
    """Drop in validation ROC-AUC when each feature is shuffled."""
    result = permutation_importance(model, X_val, y_val, scoring="roc_auc",
                                    n_repeats=cfg.explain.permutation_repeats, random_state=cfg.seed,
                                    n_jobs=1)
    return (pd.DataFrame({"feature": X_val.columns, "importance_mean": result.importances_mean,
                          "importance_std": result.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def shap_explanation(model, X_val: pd.DataFrame, cfg: Config) -> shap.Explanation:
    """SHAP values (log-odds) for a sample of validation rows, for tree models."""
    n = min(cfg.explain.shap_rows, len(X_val))
    sample = X_val.sample(n=n, random_state=cfg.seed)
    transformed = model[:-1].transform(sample)
    names = list(model.named_steps["prep"].get_feature_names_out())
    explainer = shap.TreeExplainer(model.named_steps["model"])
    values = explainer(pd.DataFrame(transformed, columns=names, index=sample.index))
    if values.values.ndim == 3:  # some versions return one column per class
        values = values[:, :, 1]
    return values


def shap_importance_frame(values: shap.Explanation) -> pd.DataFrame:
    return (pd.DataFrame({"feature": values.feature_names,
                          "mean_abs_shap": np.abs(values.values).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
