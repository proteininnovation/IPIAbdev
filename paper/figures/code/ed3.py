"""Regenerate Extended Data Figure 3 with the manuscript's canonical model set."""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okabe_style as ok


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--ipi-xlsx", required=True)
parser.add_argument("--ds1-xlsx", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

IPI_XLSX = os.path.abspath(args.ipi_xlsx)
DS1_XLSX = os.path.abspath(args.ds1_xlsx)
OUT = os.path.abspath(args.output_dir)
os.makedirs(OUT, exist_ok=True)

ok.set_style(base_pt=6)
ARCH = {"Trans": 0, "CNN": 1, "XGB": 2, "RF": 3}
EMB = {
    "AbLang2": 0,
    "IgBert": 1,
    "AntiBERTy": 2,
    "AntiBERTa2": 3,
    "AntiBERTa2-CSSP": 4,
    "OneHot": 5,
    "k-mer": 6,
    "Biophys": 7,
}


def shorten(column: str) -> str:
    name = re.sub(r"_ipi_psr_trainset(?:_train)?_score$", "", column)
    architecture = {
        "transformer_lm": "Trans",
        "transformer_onehot": "Trans",
        "cnn": "CNN",
        "xgboost": "XGB",
        "rf": "RF",
    }
    encoding = {
        "ablang": "AbLang2",
        "igbert": "IgBert",
        "antiberty": "AntiBERTy",
        "antiberta2": "AntiBERTa2",
        "antiberta2-cssp": "AntiBERTa2-CSSP",
        "onehot": "OneHot",
        "kmer": "k-mer",
        "biophysical": "Biophys",
    }
    for key in sorted(architecture, key=len, reverse=True):
        if name.startswith(key + "_"):
            suffix = name[len(key) + 1 :]
            return f"{architecture[key]}_{encoding.get(suffix, suffix)}"
    return name


def load(path: str, sheet_name: str, expected_models: int):
    frame = pd.read_excel(path, sheet_name=sheet_name)
    score_columns = [column for column in frame.columns if str(column).endswith("_score")]
    if len(score_columns) != expected_models:
        raise ValueError(
            f"Expected {expected_models} score columns in {sheet_name}, found {len(score_columns)}"
        )
    scores = frame[score_columns].copy()
    scores.columns = [shorten(column) for column in score_columns]
    if len(set(scores.columns)) != expected_models:
        raise ValueError("Model display names are not unique after shortening.")
    return scores


def order(columns):
    def key(column):
        parts = column.split("_", 1)
        return (
            ARCH.get(parts[0], 99),
            EMB.get(parts[1] if len(parts) > 1 else "", 99),
            column,
        )

    return sorted(columns, key=key)


def corr_panel(axis, scores, title, annotate):
    columns = order(list(scores.columns))
    correlation = scores[columns].corr(method="spearman")
    minimum = float(correlation.values.min())
    sns.heatmap(
        correlation,
        ax=axis,
        cmap=ok.SEQ,
        vmin=minimum,
        vmax=1.0,
        annot=annotate,
        fmt=".2f",
        annot_kws={"fontsize": 4.3},
        square=True,
        cbar=False,
        linewidths=0.3,
        linecolor="white",
    )
    axis.set_title(title, pad=4, fontsize=6.3)
    axis.set_xticks(np.arange(len(columns)) + 0.5)
    axis.set_yticks(np.arange(len(columns)) + 0.5)
    axis.set_xticklabels(columns, rotation=90, ha="center", fontsize=4.4)
    axis.set_yticklabels(columns, rotation=0, va="center", fontsize=4.4)
    axis.tick_params(length=0)
    box = axis.get_position()
    core = min(box.width, box.height) / box.height
    color_axis = axis.inset_axes([1.04, (1 - core) / 2, 0.03, core])
    scalar = mpl.cm.ScalarMappable(
        cmap=ok.SEQ, norm=mpl.colors.Normalize(vmin=minimum, vmax=1.0)
    )
    color_bar = axis.figure.colorbar(scalar, cax=color_axis)
    color_bar.set_label("Spearman rho", fontsize=5.6)
    color_axis.tick_params(labelsize=4.6)


def kde_panel(axis, scores, title, colors, columns_per_legend):
    columns = order(list(scores.columns))
    grid = np.linspace(0, 1, 400)
    for index, model in enumerate(columns):
        values = np.clip(scores[model].dropna().values, 0, 1)
        if len(values) < 2 or np.std(values) < 1e-9:
            continue
        axis.plot(
            grid,
            gaussian_kde(values, bw_method=0.15)(grid),
            lw=0.8,
            color=colors[index % len(colors)],
            label=model,
            alpha=0.85,
        )
    axis.set_xlabel("Predicted P(Pass)", fontsize=6.3)
    axis.set_ylabel("Density", fontsize=6.3)
    axis.set_xlim(0, 1)
    axis.set_title(title, pad=4, fontsize=6.3)
    axis.grid(alpha=0.3, lw=0.3)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=columns_per_legend,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.7,
        labelspacing=0.3,
        fontsize=4.4,
    )


ipi = load(IPI_XLSX, "ipi_psr_trainset_val", 25)
ds1 = load(DS1_XLSX, "DS1", 9)
print(f"IPI {ipi.shape}; DS1 {ds1.shape}")
print("IPI models:", ", ".join(order(ipi.columns)))

tab20 = [mpl.colors.to_hex(color) for color in plt.get_cmap("tab20").colors]
figure = plt.figure(figsize=(ok.DOUBLE, 200 * ok.MM))
grid = GridSpec(
    2,
    2,
    figure=figure,
    width_ratios=[1.0, 0.8],
    height_ratios=[1.5, 1.0],
    hspace=0.55,
    wspace=0.6,
    left=0.11,
    right=0.95,
    top=0.93,
    bottom=0.16,
)
axes = [
    figure.add_subplot(grid[0, 0]),
    figure.add_subplot(grid[0, 1]),
    figure.add_subplot(grid[1, 0]),
    figure.add_subplot(grid[1, 1]),
]
corr_panel(axes[0], ipi, f"IPI PSR validation (n={len(ipi):,})\n25 canonical models", False)
corr_panel(axes[1], ds1, f"DS1 cross-library (n={len(ds1):,})\n9 models", True)
kde_panel(axes[2], ipi, "IPI PSR validation - score distributions (25 canonical models)", tab20, 5)
kde_panel(axes[3], ds1, "DS1 - score distributions (9 models)", ok.QUALITATIVE, 3)
for axis, label in zip(axes, "abcd"):
    ok.panel_label(figure, axis, label, dx=-0.06, dy=0.02, size=9)

ok.save_fig(figure, "ED_Fig3", OUT)
print("Extended Data Figure 3 completed.")
