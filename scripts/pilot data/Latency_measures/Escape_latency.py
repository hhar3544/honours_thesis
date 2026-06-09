"""
Graph-only escape latency analysis.

This version does NOT do:
- no ANOVA
- no Tukey tests
- no p-values
- no significance brackets
- no stats CSV outputs
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_CSV = "/Users/yourname/Desktop/project_folder/Roi.csv"
OUTPUT_DIR = "."

TREATMENT_ORDER = ["S", "M", "PG", "C1"]
INJURY_ORDER = ["N", "SH", "CCI"]
PROBE_ORDER = [0, 5]

TREATMENT_LABELS = {
    "S": "Saline",
    "M": "Morphine",
    "PG": "Pregabalin",
    "C1": "Compound 1",
}

INJURY_LABELS = {
    "N": "Naive",
    "SH": "Sham",
    "CCI": "CCI",
}

COLOR_0MM = "#2C7FB8"
COLOR_5MM = "#F28E2B"

PROBE_COLORS = {
    0: COLOR_0MM,
    5: COLOR_5MM,
}

POINT_COLORS = {
    0: "#084081",
    5: "#A64B00",
}

POINT_SIZE = 70

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

POINT_SIZE = 115

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_video_name(video_name: str) -> pd.Series:
    """
    Expected examples:
        M-CCI_8A4-0
        PG-N_8C2-5
        S-N_8D2-0
        C1-CCI_8B4-5
    """
    name = Path(str(video_name).strip()).stem
    pattern = r"^([A-Z0-9]+)-(CCI|N|SH)_(.+)-([05])$"
    match = re.match(pattern, name)

    if not match:
        raise ValueError(f"Could not parse VIDEO name: {video_name}")

    treatment, injury, animal_id, probe = match.groups()

    return pd.Series({
        "Treatment": treatment,
        "Injury": injury,
        "AnimalID": animal_id,
        "AnimalUID": f"{treatment}-{injury}_{animal_id}",
        "Probe": int(probe),
    })


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def load_and_process_data(input_csv):
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_csv}. Update the path or pass the correct Roi.csv path when running the script."
        )

    raw = pd.read_csv(input_path)

    required_cols = ["VIDEO", "MEASUREMENT", "SHAPE", "VALUE"]
    missing_cols = [c for c in required_cols if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Escape latency = first ROI entry time into Chamber3
    df = raw[
        (raw["MEASUREMENT"].astype(str) == "FIRST ROI ENTRY TIME (S)") &
        (raw["SHAPE"].astype(str) == "Chamber3")
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows found for MEASUREMENT == 'FIRST ROI ENTRY TIME (S)' and SHAPE == 'Chamber3'."
        )

    parsed = df["VIDEO"].apply(parse_video_name)
    df = pd.concat([df, parsed], axis=1)

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE", "Treatment", "Injury", "Probe"]).copy()

    df = df[df["Treatment"].isin(TREATMENT_ORDER)].copy()
    df = df[df["Injury"].isin(INJURY_ORDER)].copy()
    df = df[df["Probe"].isin(PROBE_ORDER)].copy()

    df["Treatment"] = pd.Categorical(df["Treatment"], categories=TREATMENT_ORDER, ordered=True)
    df["Injury"] = pd.Categorical(df["Injury"], categories=INJURY_ORDER, ordered=True)
    df["Probe"] = pd.Categorical(df["Probe"], categories=PROBE_ORDER, ordered=True)

    return df

# ============================================================
# PLOTTING
# ============================================================
def make_plot(df, output_dir):
    treatments_present = [t for t in TREATMENT_ORDER if t in df["Treatment"].astype(str).unique()]
    injuries_present = [i for i in INJURY_ORDER if i in df["Injury"].astype(str).unique()]

    x_base = {inj: i for i, inj in enumerate(injuries_present)}
    offset = {0: -0.20, 5: 0.20}
    bar_width = 0.34

    # 2 × 2 layout is much more readable in a thesis than 1 × 4
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13.5, 10.5),
        sharey=True,
        constrained_layout=False,
    )

    axes = axes.flatten()

    ymax_global = df["VALUE"].max()
    if pd.isna(ymax_global) or ymax_global <= 0:
        ymax_global = 1

    for col_idx, treatment in enumerate(treatments_present):
        ax = axes[col_idx]
        sub = df[df["Treatment"].astype(str) == treatment].copy()
        label = TREATMENT_LABELS.get(treatment, treatment)

        for injury in injuries_present:
            for probe in PROBE_ORDER:
                grp = sub[
                    (sub["Injury"].astype(str) == injury) &
                    (sub["Probe"].astype(int) == probe)
                ].copy()

                if grp.empty:
                    continue

                x = x_base[injury] + offset[int(probe)]
                y = grp["VALUE"].dropna().values

                mean_y = np.nanmean(y) if len(y) else np.nan
                sem_y = sem(y)

                ax.bar(
                    x,
                    mean_y,
                    width=bar_width,
                    color=PROBE_COLORS[int(probe)],
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=2,
                )

                if np.isfinite(sem_y):
                    ax.errorbar(
                        x,
                        mean_y,
                        yerr=sem_y,
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.5,
                        capsize=5,
                        capthick=1.5,
                        zorder=3,
                    )

                if len(y):
                    jitter = np.linspace(-0.06, 0.06, len(y)) if len(y) > 1 else np.array([0.0])
                    ax.scatter(
                        np.full(len(y), x) + jitter,
                        y,
                        s=POINT_SIZE,
                        color=POINT_COLORS[int(probe)],
                        edgecolor="black",
                        linewidth=0.9,
                        alpha=0.95,
                        zorder=4,
                    )

        ax.set_ylim(0, ymax_global + max(1.0, ymax_global * 0.30))

        # Keep treatment as panel heading; remove this line if you want no panel labels either
        ax.set_title(label, fontsize=23, pad=14, weight="bold")

        ax.set_xlabel("Injury group", fontsize=21, weight="bold")
        ax.set_xticks([x_base[i] for i in injuries_present])
        ax.set_xticklabels([INJURY_LABELS[i] for i in injuries_present], fontsize=19)

        if col_idx in [0, 2]:
            ax.set_ylabel("Escape latency (s)", fontsize=21, weight="bold")
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="both", labelsize=18, width=1.5, length=6, direction="out")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)

    # Hide unused axes if fewer than 4 treatments are present
    for j in range(len(treatments_present), len(axes)):
        axes[j].set_visible(False)

    # Central key only: no figure title
    legend_handles = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="0 mm",
            markerfacecolor=PROBE_COLORS[0],
            markeredgecolor="black",
            markersize=16,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="5 mm",
            markerfacecolor=PROBE_COLORS[5],
            markeredgecolor="black",
            markersize=16,
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    # Space for central legend, without adding a title
    fig.subplots_adjust(
        top=0.89,
        bottom=0.08,
        left=0.08,
        right=0.98,
        wspace=0.16,
        hspace=0.42,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "escape_latency_graphs_large_2x2.png"
    pdf_path = output_dir / "escape_latency_graphs_large_2x2.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure outputs:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
# ============================================================
# MAIN
# ============================================================
def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

    df = load_and_process_data(input_csv)
    make_plot(df, output_dir)


if __name__ == "__main__":
    main()
