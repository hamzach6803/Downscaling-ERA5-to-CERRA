from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "outputs" / "masks" / "extreme_batch" / "scores_extreme_summary_t2m_f1.csv"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_PATH = OUT_DIR / "metrics_acc_pod_far_45cold_45hot.png"
OUT_PATH_WITH_F1 = OUT_DIR / "metrics_acc_pod_far_f1_45cold_45hot.png"

MODELS = ["Interpolation", "PyESD", "DeepSD", "ESRGAN"]
METRICS = ["ACC", "POD", "FAR", "F1"]
EVENTS = [
    ("heat", "45 vagues de chaleur", "#e6263f"),
    ("cold", "45 vagues de froid", "#4c91c4"),
]


def percent_label(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.0), sharey=False)
    x = np.arange(len(MODELS))
    width = 0.36

    for ax, metric in zip(axes, METRICS):
        for index, (event, label, color) in enumerate(EVENTS):
            values = []
            for model in MODELS:
                row = df[(df["type_event"] == event) & (df["model"] == model)]
                if row.empty:
                    raise ValueError(f"Missing value for {event}/{model}/{metric}")
                values.append(float(row.iloc[0][f"{metric}_mean"]) * 100.0)

            offset = (index - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1.2 if metric != "FAR" else 0.18),
                    percent_label(value),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        if metric == "FAR":
            ax.set_ylim(0, 10)
            ax.set_yticks(np.arange(0, 11, 2))
        else:
            ax.set_ylim(0, 108)
            ax.set_yticks(np.arange(0, 101, 20))

    axes[0].set_ylabel("Score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.27, top=0.72, wspace=0.45)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PATH_WITH_F1, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {OUT_PATH}")
    print(f"Figure saved: {OUT_PATH_WITH_F1}")


if __name__ == "__main__":
    main()
