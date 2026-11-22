"""Every figure in the project, drawn in one house style.

Each function returns a matplotlib figure; ``finish_figure`` saves it at
200 dpi and either shows it (notebook) or closes it (pipeline).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.container import BarContainer
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from .artifacts import RunContext

PALETTE = {"ink": "#1d2b36", "blue": "#2b6cb0", "red": "#c0392b", "teal": "#16867a",
           "amber": "#d68910", "grey": "#8a96a3", "light": "#d9dee3"}


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelcolor": PALETTE["ink"], "axes.edgecolor": PALETTE["light"],
        "text.color": PALETTE["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })


def finish_figure(fig, name: str, ctx: RunContext) -> None:
    """Save the figure as ``<name>.png`` in the run's figure folder, then show it (notebook) or close it.

    Returns nothing on purpose: as the last line of a notebook cell, a returned path would be
    echoed into the saved output, and that path is specific to the machine.
    """
    path = ctx.fig_dir / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if ctx.show:
        plt.show()
    else:
        plt.close(fig)


def plot_missingness(raw: pd.DataFrame):
    miss = raw.isna().sum()
    miss = (miss[miss > 0] / len(raw)).sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(miss.index, miss.values * 100, color=PALETTE["blue"])
    for i, v in enumerate(miss.values):
        ax.text(v * 100 + 1, i, f"{v:.1%}", va="center", fontsize=9)
    ax.set(xlabel="Share of rows missing (%)", title="Missing values in the raw data", xlim=(0, 110))
    return fig


def plot_cleaning_log(log: list[dict]):
    frame = pd.DataFrame(log)
    frame["affected"] = frame["rows_removed"] + frame["values_changed"].fillna(0)
    frame = frame[frame["affected"] > 0]
    colors = [PALETTE["red"] if r > 0 else PALETTE["amber"] for r in frame["rows_removed"]]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(frame["rule"], frame["affected"], color=colors)
    ax.set_xscale("log")
    for i, v in enumerate(frame["affected"]):
        ax.text(v * 1.1, i, f"{int(v):,}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Rows removed (red) or values changed (amber), log scale", title="Cleaning rules")
    return fig


def plot_duplicate_effect(removed_rate: float, kept_rate: float):
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Removed duplicate rows", "Rows kept (train + validation)"], [removed_rate, kept_rate],
                  color=[PALETTE["red"], PALETTE["blue"]])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.1%}", ha="center")
    ax.set(ylabel="Cancellation rate", ylim=(0, 1), title="Duplicates are mostly cancellations")
    return fig


def plot_rate_by_category(trval: pd.DataFrame, columns: list[str], target: str):
    rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.6 * rows), sharey=True)
    base = trval[target].mean()
    for ax, col in zip(np.ravel(axes), columns):
        stats = trval.groupby(col)[target].agg(["mean", "size"]).sort_values("size", ascending=False).head(8)
        stats = stats[stats["size"] >= 100]  # a rate from a handful of rows says nothing
        bars = ax.bar(stats.index.astype(str), stats["mean"], color=PALETTE["blue"])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in stats["mean"]], fontsize=9, padding=2)
        ax.axhline(base, color=PALETTE["red"], ls="--", lw=1)
        ax.set_title(col.replace("_", " ").capitalize())
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    for ax in np.ravel(axes)[len(columns):]:
        ax.set_visible(False)
    for ax in np.atleast_2d(axes)[:, 0]:
        ax.set_ylabel("Cancellation rate")
    np.ravel(axes)[0].set_ylim(0, 1.12)
    fig.suptitle("Cancellation rate by category (groups of at least 100 bookings; dashed line = overall rate)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_numeric_by_outcome(trval: pd.DataFrame, target: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col, clip in zip(axes, ["lead_time", "adr"], [500, 350]):
        for label, color, name in [(0, PALETTE["blue"], "Kept"), (1, PALETTE["red"], "Cancelled")]:
            vals = trval.loc[trval[target] == label, col].clip(upper=clip)
            ax.hist(vals, bins=50, density=True, alpha=0.55, color=color, label=name)
        ax.set(title=f"{col} by outcome", xlabel=col, ylabel="Density")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_monthly_rate(trval: pd.DataFrame, target: str):
    months = pd.to_datetime(trval["arrival_date_year"].astype(str) + "-" + trval["arrival_date_month"] + "-01",
                            format="%Y-%B-%d")
    series = trval.assign(month=months).groupby("month")[target].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(series.index, series["size"], width=20, color=PALETTE["light"], label="Bookings")
    ax.set_ylabel("Bookings per arrival month")
    ax2 = ax.twinx()
    ax2.plot(series.index, series["mean"], color=PALETTE["red"], marker="o", ms=3, label="Cancellation rate")
    ax2.set_ylabel("Cancellation rate")
    ax2.grid(False)
    handles = [*ax.get_legend_handles_labels()[0], *ax2.get_legend_handles_labels()[0]]
    ax.legend(handles, ["Bookings (left axis)", "Cancellation rate (right axis)"], loc="upper left", fontsize=8)
    ax.set_title("Seasonality and drift: bookings and cancellation rate by arrival month")
    return fig


def plot_correlation(features: pd.DataFrame, y: pd.Series):
    num = features.select_dtypes("number").assign(cancelled=y.to_numpy())
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.7},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=7)
    ax.set_title("Correlation between numeric features and the target")
    return fig


def plot_pca(features: pd.DataFrame, y: pd.Series, seed: int, n: int = 5000):
    num = features.select_dtypes("number")
    idx = np.random.default_rng(seed).choice(len(num), size=min(n, len(num)), replace=False)
    pca = PCA(n_components=2, random_state=seed)
    pts = pca.fit_transform(StandardScaler().fit_transform(num.iloc[idx]))
    lab = y.to_numpy()[idx]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color, name in [(0, PALETTE["blue"], "Kept"), (1, PALETTE["red"], "Cancelled")]:
        m = lab == label
        ax.scatter(pts[m, 0], pts[m, 1], s=5, alpha=0.35, color=color, label=name)
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.0%} of variance)", ylabel=f"PC2 ({ev[1]:.0%})",
           title="PCA of numeric features (unsupervised view)")
    ax.legend(markerscale=3)
    return fig


def plot_tier3_evidence(evidence: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    titles = {"required_car_parking_spaces": "Parking spaces requested", "total_of_special_requests": "Special requests",
              "country": "Country (10 largest)", "adr": "Price per night (ten equal groups)"}
    for ax, (key, title) in zip(np.ravel(axes), titles.items()):
        table = pd.DataFrame(evidence[key])
        bars = ax.bar(table["value"], table["cancel_rate"], color=PALETTE["amber"])
        small = len(table) > 6
        for b, rate, n in zip(bars, table["cancel_rate"], table["rows"]):
            ax.text(b.get_x() + b.get_width() / 2, rate + 0.015, f"{rate:.0%}" if small else f"{rate:.0%}\nn={n:,}",
                    ha="center", va="bottom", fontsize=9)
        ax.set(title=title, ylim=(0, max(0.6, float(table["cancel_rate"].max()) + 0.18)))
        ax.tick_params(axis="x", rotation=30 if small else 0, labelsize=9)
        if small:
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cancellation rate")
    fig.suptitle("Evidence for the timing-uncertain (Tier 3) columns", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_leakage_ablation(ablation: dict, dedup: dict):
    ladder = pd.DataFrame(ablation["ladder"])
    rows = list(ladder["step"]) + ["Deployed features, duplicates kept"]
    vals = list(ladder["val_roc_auc"]) + [dedup["val_roc_auc_duplicates_kept"]]
    colors = [PALETTE["teal"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"], PALETTE["red"], PALETTE["grey"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(rows, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0.5, 1.05), xlabel="Validation ROC-AUC",
           title="How much apparent performance leakage and duplicates buy")
    return fig


def plot_candidates(table: pd.DataFrame):
    t = table.sort_values("val_roc_auc")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(t["model"], t["val_roc_auc"], color=PALETTE["blue"], label="Validation (full-train fit)")
    ax.errorbar(t["cv_roc_auc_mean"], np.arange(len(t)), xerr=t["cv_roc_auc_std"], fmt="o",
                color=PALETTE["ink"], ms=4, label="CV mean ± std (subsample)")
    ax.set(xlim=(0.45, 0.85), xlabel="ROC-AUC", title="Candidate models before tuning")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_search_comparison(tuning: dict):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, color, name in [("lgbm_grid", PALETTE["grey"], "Grid"), ("lgbm_random", PALETTE["blue"], "Random"),
                             ("lgbm_bayesian", PALETTE["red"], "Bayesian (Optuna TPE)")]:
        r = tuning[key]
        ax.plot(np.arange(1, len(r["best_so_far"]) + 1), r["best_so_far"], color=color, lw=2, label=name)
        ax.scatter(np.arange(1, len(r["history"]) + 1), r["history"], color=color, s=10, alpha=0.4)
    ax.set(xlabel="Configurations tried (same budget for each strategy)", ylabel="CV ROC-AUC",
           title="LightGBM: best score found vs. search effort")
    ax.legend()
    return fig


def plot_learning_curve(lc: dict):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sizes = np.array(lc["train_sizes"])
    for key, color, name in [("train", PALETTE["blue"], "Training folds"), ("cv", PALETTE["red"], "Held-out folds")]:
        mean, std = np.array(lc[f"{key}_mean"]), np.array(lc[f"{key}_std"])
        ax.plot(sizes, mean, marker="o", color=color, label=name)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(xlabel="Training rows", ylabel="ROC-AUC", title="Learning curve (champion settings)")
    ax.legend()
    return fig


def plot_validation_curve(vc: dict):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(vc["values"], vc["train_mean"], marker="o", color=PALETTE["blue"], label="Training folds")
    ax.errorbar(vc["values"], vc["cv_mean"], yerr=vc["cv_std"], marker="o", color=PALETTE["red"],
                label="Held-out folds")
    ax.set_xscale("log", base=2)
    if vc.get("chosen"):
        ax.axvline(vc["chosen"], color=PALETTE["grey"], ls=":", label=f"Champion: {vc['chosen']}")
    ax.set(xlabel=vc["param"] + " (model capacity)", ylabel="ROC-AUC", title="Validation curve")
    ax.legend()
    return fig


def plot_roc_pr(curves: dict, title: str):
    """ROC and precision-recall curves, in the colours of the test bar chart (champion red, logreg blue).

    A model that gives every booking the same score (the majority-class baseline) has no curve: it is
    the no-skill diagonal in the ROC panel and the base-rate line in the precision-recall panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"champion": PALETTE["red"], "logreg": PALETTE["blue"]}
    base_rate = None
    for name, (y, p) in curves.items():
        base_rate = float(np.mean(y))
        if np.unique(p).size == 1:
            continue
        color = colors.get(name, PALETTE["teal"])
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=color, label=name)
        axes[1].plot(rec, prec, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", color=PALETTE["grey"], label="no skill (majority class)")
    axes[1].axhline(base_rate, ls="--", color=PALETTE["grey"], label=f"no skill: base rate {base_rate:.1%}")
    axes[1].set_ylim(0, 1.02)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration(y, raw_p, cal_p):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if np.allclose(raw_p, cal_p):  # calibration was tested and not adopted: there is only one set of probabilities
        lines = [(raw_p, PALETTE["red"], "Model probabilities (no calibration step)")]
    else:
        lines = [(raw_p, PALETTE["grey"], "Uncalibrated"), (cal_p, PALETTE["red"], "Calibrated")]
    for p, color, name in lines:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", color=color, label=name)
    ax.plot([0, 1], [0, 1], ls=":", color=PALETTE["ink"])
    ax.set(xlabel="Predicted probability", ylabel="Observed cancellation rate", title="Calibration (validation)")
    ax.legend()
    return fig


def plot_cost_curve(curve: pd.DataFrame, threshold: float, fn_cost: float, fp_cost: float):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(curve["threshold"], curve["cost_per_booking"], color=PALETTE["blue"], lw=2)
    ax.axvline(threshold, color=PALETTE["red"], ls="--", label=f"Chosen threshold {threshold:.2f}")
    ax.axvline(0.5, color=PALETTE["grey"], ls=":", label="Default 0.50")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per booking",
           title=f"Threshold from costs (missed cancellation = {fn_cost:g}, false alarm = {fp_cost:g})")
    ax.legend()
    return fig


def plot_confusion(cm: list[list[int]], title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(np.array(cm), annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Kept", "Cancelled"], yticklabels=["Kept", "Cancelled"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    return fig


def plot_importance(frame: pd.DataFrame, value: str, error: str | None, title: str, top: int = 15):
    t = frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t["feature"], t[value], xerr=t[error] if error else None, color=PALETTE["teal"])
    ax.set(title=title, xlabel=value.replace("_", " "))
    return fig


def plot_shap_summary(values):
    shap.plots.beeswarm(values, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    plt.title("SHAP: how each feature pushes the prediction (validation sample)")
    return fig


def plot_shap_waterfall(values, row: int):
    shap.plots.waterfall(values[row], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.title("Why the model scored this booking as it did (log-odds)")
    return fig


def plot_segments(report: pd.DataFrame):
    groups = list(report["by"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.2), sharey=True)
    for ax, by in zip(np.atleast_1d(axes), groups):
        part = report[report["by"] == by]
        if by == "lead_time_band":
            order = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
            part = part.set_index("segment").reindex([b for b in order if b in set(part["segment"])]).reset_index()
        x = np.arange(len(part))
        ax.bar(x - 0.2, part["actual_cancel_rate"], 0.4, color=PALETTE["grey"], label="Actual cancellation rate")
        ax.bar(x + 0.2, part["error_rate"], 0.4, color=PALETTE["red"], label="Error rate at the threshold")
        ax.set_xticks(x, part["segment"], rotation=40, ha="right", fontsize=9)
        ax.set_title(by.replace("_", " ").capitalize())
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle("Error analysis by segment (validation)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_temporal(temporal: dict):
    t = pd.DataFrame(temporal["by_month"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t["month"].str[:3], t["roc_auc"], marker="o", color=PALETTE["red"], label="ROC-AUC")
    ax.axhline(temporal["eval_roc_auc"], color=PALETTE["grey"], ls="--", label="Whole year")
    ax.set(ylabel="ROC-AUC", title=f"Trained on arrivals {temporal['fit_years']}, scored on {temporal['eval_year']}")
    ax.legend()
    return fig


def plot_test_results(test: dict):
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    names = list(test["models"].keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = [PALETTE["grey"], PALETTE["blue"], PALETTE["red"]]
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [test["models"][name]["metrics"][m] for m in metrics]
        ax.bar(x + i * width, vals, width, color=color, label=name)
    ci = test["models"]["champion"]["ci"]
    j = names.index("champion")
    lows = [test["models"]["champion"]["metrics"][m] - ci[m]["low"] for m in metrics]
    highs = [ci[m]["high"] - test["models"]["champion"]["metrics"][m] for m in metrics]
    ax.errorbar(x + j * width, [test["models"]["champion"]["metrics"][m] for m in metrics],
                yerr=[lows, highs], fmt="none", color=PALETTE["ink"], capsize=3)
    ax.set_xticks(x + width * (len(names) - 1) / 2, [m.replace("_", "-") for m in metrics])
    ax.set(ylim=(0, 1.12), title="Test set, final evaluation (champion with 95% bootstrap CI)")
    ax.legend(ncol=len(names), loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    return fig


def plot_experiments(encodings: pd.DataFrame, imbalance: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].barh(encodings["encoding"], encodings["cv_roc_auc_mean"], xerr=encodings["cv_roc_auc_std"],
                 color=PALETTE["blue"])
    lo = min(encodings["cv_roc_auc_mean"].min(), imbalance["val_roc_auc"].min()) - 0.02
    axes[0].set(xlim=(lo, None), xlabel="CV ROC-AUC (training data)", title="Encoding the categories")
    axes[1].barh(imbalance["method"], imbalance["val_pr_auc"], color=PALETTE["teal"])
    axes[1].set(xlim=(imbalance["val_pr_auc"].min() - 0.03, None), xlabel="Validation PR-AUC",
                title="Handling class imbalance")
    for ax in axes:
        for c in ax.containers:
            if isinstance(c, BarContainer):
                ax.bar_label(c, fmt="%.3f", fontsize=8, padding=3)
    fig.tight_layout()
    return fig


def plot_outliers(raw: pd.DataFrame):
    """Box plots of the skewed numeric columns (log scale), with the largest value labelled."""
    columns = {"lead_time": "Lead time (days)", "adr": "Price per night",
               "stays_in_week_nights": "Week nights", "adults": "Adults"}
    fig, axes = plt.subplots(1, len(columns), figsize=(10, 3.4))
    for ax, (col, label) in zip(axes, columns.items()):
        values = raw[col].dropna()
        values = values[values >= 0]
        ax.boxplot(values, widths=0.5, patch_artist=True, showfliers=True,
                   boxprops={"facecolor": PALETTE["light"], "edgecolor": PALETTE["ink"]},
                   medianprops={"color": PALETTE["red"], "linewidth": 2},
                   flierprops={"marker": ".", "markersize": 3, "markerfacecolor": PALETTE["grey"],
                               "markeredgecolor": PALETTE["grey"], "alpha": 0.4})
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks([])
        ax.set_title(label)
        ax.annotate(f"max {values.max():,.0f}", (1, values.max()), xytext=(1.32, values.max()), fontsize=9,
                    va="center", color=PALETTE["red"])
    axes[0].set_ylabel("Value (log scale)")
    fig.suptitle("Outliers: rare but mostly real values (red line = median)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_duplicate_decomposition(dup: dict):
    """Three scores with duplicates kept: what is memorisation and what is a different population."""
    labels = ["Random split\n(copies on both sides)", "Copies kept together,\nevery row scored",
              "Copies kept together,\none row per booking"]
    vals = [dup["roc_auc_random_split"], dup["roc_auc_grouped_split"], dup["roc_auc_grouped_one_per_record"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=[PALETTE["grey"], PALETTE["amber"], PALETTE["teal"]], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")
    for i, (name, value) in enumerate([("memorisation", dup["memorisation"]),
                                       ("repeated rows are\nan easier population", dup["population"])]):
        ax.annotate("", xy=(i + 1, vals[i + 1] + 0.022), xytext=(i, vals[i] + 0.022),
                    arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
        ax.text(i + 0.5, max(vals[i], vals[i + 1]) + 0.034, f"-{value:.3f}\n{name}", ha="center", fontsize=9)
    ax.set(ylim=(0.7, 0.97), ylabel="Validation ROC-AUC", title="Duplicates kept: where the higher score comes from")
    return fig


def plot_operating_points(y, proba, table: pd.DataFrame, title: str):
    """Precision-recall curve with the validation-chosen thresholds marked."""
    prec, rec, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rec, prec, color=PALETTE["blue"], lw=2)
    ax.axhline(float(np.mean(y)), color=PALETTE["grey"], ls=":", label="Flag everyone (base rate)")
    colors = {"cost-based (deployed)": PALETTE["red"], "default 0.5": PALETTE["amber"], "best F1": PALETTE["teal"]}
    for row in table.to_dict("records"):
        ax.scatter(row["recall"], row["precision"], s=70, zorder=3, color=colors.get(row["operating_point"]),
                   label=f"{row['operating_point']}: threshold {row['threshold']:.2f}, precision "
                         f"{row['precision']:.2f}, recall {row['recall']:.2f}")
    ax.set(xlabel="Recall (share of cancellations caught)", ylabel="Precision (share of flags that are right)",
           title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_gains(gains: pd.DataFrame, title: str):
    """Cancellation rate per risk group, and the share of all cancellations reached."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(gains["risk_group"], gains["cancel_rate"], color=PALETTE["blue"])
    overall = float((gains["cancel_rate"] * gains["bookings"]).sum() / gains["bookings"].sum())
    ax.axhline(overall, color=PALETTE["grey"], ls="--", label=f"Overall rate {overall:.0%}")
    for x, v in zip(gains["risk_group"], gains["cancel_rate"]):
        ax.text(x, v + 0.012, f"{v:.0%}", ha="center", fontsize=8)
    ax.set(xlabel="Bookings grouped by predicted risk (1 = riskiest tenth)", ylabel="Actual cancellation rate",
           title=title, xticks=gains["risk_group"])
    ax2 = ax.twinx()
    ax2.plot(gains["risk_group"], gains["cumulative_share_of_cancellations"], color=PALETTE["red"], marker="o", ms=4,
             label="Share of all cancellations reached")
    ax2.set(ylim=(0, 1.05), ylabel="Cumulative share of cancellations")
    ax2.grid(False)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="center right")
    return fig


def plot_capacity(capacity: dict, title: str):
    """Predicted against actual cancellations per hotel and arrival week."""
    weekly = pd.DataFrame(capacity["by_week"])
    top = float(max(weekly["actual"].max(), weekly["predicted"].max())) * 1.05
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(weekly["base_rate_forecast"], weekly["actual"], s=14, color=PALETTE["grey"], alpha=0.5,
               label=f"No model: bookings x overall rate (error {capacity['wape_base_rate']:.0%})")
    ax.scatter(weekly["predicted"], weekly["actual"], s=16, color=PALETTE["red"], alpha=0.75,
               label=f"Model: sum of probabilities (error {capacity['wape_model']:.0%})")
    ax.plot([0, top], [0, top], ls=":", color=PALETTE["ink"])
    ax.set(xlabel="Forecast cancellations in the week", ylabel="Actual cancellations in the week", title=title,
           xlim=(0, top), ylim=(0, top))
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_drift(report: pd.DataFrame, title: str, top: int = 15):
    t = report.head(top).iloc[::-1]
    colors = [PALETTE["red"] if v > 0.25 else PALETTE["amber"] if v > 0.1 else PALETTE["teal"] for v in t["psi"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t["feature"], t["psi"], color=colors)
    ax.axvline(0.1, color=PALETTE["grey"], ls=":", label="0.1 moderate shift")
    ax.axvline(0.25, color=PALETTE["ink"], ls="--", label="0.25 large shift")
    ax.set(xlabel="Population stability index", title=title)
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_latency(result: dict):
    lat = np.asarray(result["client_latencies_ms"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(lat[1:] if len(lat) > 1 else lat, bins=40, color=PALETTE["blue"])
    ax.axvline(result["warm_p50_ms"], color=PALETTE["teal"], ls="--", label=f"p50 {result['warm_p50_ms']:.0f} ms")
    ax.axvline(result["warm_p95_ms"], color=PALETTE["red"], ls="--", label=f"p95 {result['warm_p95_ms']:.0f} ms")
    ax.set(xlabel="Round-trip latency per request (ms)", ylabel="Requests",
           title=f"Latency of {result['requests']} requests to the running service "
                 f"(first request: {result['first_request_ms']:.0f} ms)")
    ax.legend()
    return fig
