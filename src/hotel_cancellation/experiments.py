"""Two design experiments: how to encode categories, and how to handle class imbalance.

Both use the same LightGBM settings so only the choice under test changes.
Encodings are compared by cross-validation on training data; imbalance
methods are fitted on a training subsample and compared on validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder

from .config import Config, deployed_sources
from .features import build_feature_frame, make_tree_preprocessor, split_feature_types, to_float32
from .models import base_estimator
from .tuning import cv_splitter, stratified_subsample


def seeded_target_encoder(seed: int) -> TargetEncoder:
    """Deterministic target encoder; scikit-learn 1.9 moved the seed into the ``cv`` argument."""
    import sklearn

    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", random_state=seed)


def encoding_pipelines(cats: list[str], nums: list[str], cfg: Config) -> dict[str, Pipeline]:
    minf = cfg.features.min_category_frequency

    def wrap(encoder) -> Pipeline:
        columns = ColumnTransformer([("cat", encoder, cats), ("num", "passthrough", nums)],
                                    verbose_feature_names_out=False)
        return Pipeline([("prep", columns), ("float32", FunctionTransformer(to_float32)),
                         ("model", base_estimator("lgbm", cfg))])

    return {
        "ordinal (deployed)": wrap(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                  min_frequency=minf)),
        "one-hot": wrap(OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=minf,
                                      sparse_output=False)),
        # TargetEncoder cross-fits internally, and here it is refitted inside every CV fold.
        "target encoding": wrap(seeded_target_encoder(cfg.seed)),
    }


def compare_encodings(X, y, cats, nums, cfg: Config) -> pd.DataFrame:
    Xs, ys = stratified_subsample(X, y, cfg.tuning.subsample_rows, cfg.seed)
    rows = []
    for name, pipe in encoding_pipelines(cats, nums, cfg).items():
        cv = cross_validate(pipe, Xs, ys, cv=cv_splitter(cfg), scoring=["roc_auc", "average_precision"],
                            return_estimator=True)
        width = cv["estimator"][0].named_steps["prep"].transform(Xs.head(5)).shape[1]
        rows.append({"encoding": name, "cv_roc_auc_mean": float(np.mean(cv["test_roc_auc"])),
                     "cv_roc_auc_std": float(np.std(cv["test_roc_auc"])),
                     "cv_pr_auc_mean": float(np.mean(cv["test_average_precision"])), "model_columns": int(width),
                     "fit_seconds_per_fold": float(np.mean(cv["fit_time"]))})
    return pd.DataFrame(rows)


def compare_feature_sets(train_rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Would more feature engineering help? Deployed features against deployed plus the candidate group.

    Cross-validated on all training rows with the same folds and the same LightGBM settings,
    so the per-fold differences are paired.
    """
    y = train_rows[cfg.target]
    variants = {"deployed features": deployed_sources(cfg),
                "deployed + candidate features": deployed_sources(cfg) + list(cfg.features.candidate_sources)}
    rows, folds = [], {}
    for name, sources in variants.items():
        X = build_feature_frame(train_rows, sources)
        cats, nums = split_feature_types(list(X.columns))
        pipe = Pipeline(make_tree_preprocessor(cats, nums, cfg.features.min_category_frequency)
                        + [("model", base_estimator("lgbm", cfg))])
        folds[name] = cross_validate(pipe, X, y, cv=cv_splitter(cfg), scoring="roc_auc")["test_score"]
        rows.append({"features": name, "columns": int(X.shape[1]), "cv_roc_auc_mean": float(np.mean(folds[name])),
                     "cv_roc_auc_std": float(np.std(folds[name]))})
    gain = folds["deployed + candidate features"] - folds["deployed features"]
    rows[1].update({"gain_mean": float(np.mean(gain)), "gain_min": float(np.min(gain)), "gain_max": float(np.max(gain))})
    rows[0].update({"gain_mean": 0.0, "gain_min": 0.0, "gain_max": 0.0})
    return pd.DataFrame(rows)


def compare_imbalance(X_train, y_train, X_val, y_val, cats, nums, cfg: Config) -> pd.DataFrame:
    """No correction, class weights, and SMOTENC oversampling (training rows only)."""
    Xs, ys = stratified_subsample(X_train, y_train, cfg.tuning.subsample_rows, cfg.seed)
    prep = make_tree_preprocessor(cats, nums, cfg.features.min_category_frequency)
    ratio = float((ys == 0).sum() / (ys == 1).sum())
    variants = {
        "none": Pipeline(prep + [("model", base_estimator("lgbm", cfg))]),
        "class weights": Pipeline(prep + [("model", base_estimator("lgbm", cfg, scale_pos_weight=ratio))]),
        "SMOTENC oversampling": ImbPipeline(prep + [
            ("smote", SMOTENC(categorical_features=list(range(len(cats))), random_state=cfg.seed)),
            ("model", base_estimator("lgbm", cfg))]),
    }
    rows = []
    for name, pipe in variants.items():
        pipe.fit(Xs, ys)
        p = pipe.predict_proba(X_val)[:, 1]
        rows.append({"method": name, "val_roc_auc": float(roc_auc_score(y_val, p)),
                     "val_pr_auc": float(average_precision_score(y_val, p)),
                     "val_brier": float(brier_score_loss(y_val, p)),
                     "mean_predicted_probability": float(np.mean(p)), "actual_rate": float(np.mean(y_val))})
    return pd.DataFrame(rows)
