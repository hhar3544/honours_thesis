"""
Graph-only Chamber 2 strict entry count analysis.

What this version does:
- Reads cleaned_roi_summary.csv
- Filters to ROI == Chamber2
- Parses Treatment, Injury, AnimalID and Probe from VIDEO names
- Creates a 1-row, 4-panel BAR GRAPH (one panel per treatment)
- Saves only the output graphs (PNG and PDF)

What this version does NOT do:
- No ANOVA
- No Tukey
- No paired tests
- No significance brackets
- No stats CSV outputs
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# User settings
# =========================
INPUT_CSV = ""/Users/yourname/Desktop/project_folder/cleaned_Roi.csv""
ROI_TO_PLOT = "Chamber2"
METRIC = "STRICT_ENTRY_COUNT_OUTSIDE_GAP"

OUT_PREFIX = "chamber2_total_entries_bargraph_1row4panels"

TREATMENT_ORDER = ["S", "M", "PG", "C1"]
TREATMENT_LABELS = {
    "S": "Saline",
    "M": "Morphine",
    "PG": "Pregabalin",
    "C1": "Compound 1",
}

INJURY_ORDER = ["N", "SH", "CCI"]
INJURY_LABELS = {
    "N": "Naive",
    "SH": "Sham",
    "CCI": "CCI",
}

PROBE_ORDER = [0, 5]
PROBE_LABELS = {0: "0 mm", 5: "5 mm"}

COLOR_0MM = "#2C7FB8"
COLOR_5MM = "#F28E2B"
POINT_0MM = "#084081"
POINT_5MM = "#A64B00"

DPI = 600

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

def parse_video_name(video: str):
    pattern = r"^(?P<Treatment>.+?)-(?P<Injury>CCI|SH|N)_(?P<AnimalID>.+)-(?P<Probe>[05])$"
    match = re.match(pattern, str(video))
    if not match:
        raise ValueError(
            f"Could not parse VIDEO name: {video}. Expected format like S-CCI_8A1-0 or C1-N_8B2-5."
        )
    out = match.groupdict()
    out["Probe"] = int(out["Probe"])
    out["AnimalUID"] = f"{out['Treatment']}-{out['Injury']}_{out['AnimalID']}"
    return out


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def load_data(input_csv):
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input CSV: {input_csv}\n"
            "Either place the CSV in the expected folder or pass the full CSV path when running the script."
        )

    df = pd.read_csv(input_path)

    required_cols = {"VIDEO", "ROI", METRIC}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["ROI"].astype(str).str.lower() == ROI_TO_PLOT.lower()].copy()
    parsed = df["VIDEO"].apply(parse_video_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    df = df[df["Treatment"].isin(TREATMENT_ORDER)].copy()
    df = df[df["Injury"].isin(INJURY_ORDER)].copy()
    df = df[df["Probe"].isin(PROBE_ORDER)].copy()

    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[METRIC]).copy()

    df["Treatment"] = pd.Categorical(df["Treatment"], categories=TREATMENT_ORDER, ordered=True)
    df["Injury"] = pd.Categorical(df["Injury"], categories=INJURY_ORDER, ordered=True)
    df["Probe"] = pd.Categorical(df["Probe"], categories=PROBE_ORDER, ordered=True)

    return df


def make_bar_plot(df, out_prefix):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13.5, 10.5),
        sharey=True,
        constrained_layout=False,
    )

    axes = axes.flatten()

    bar_width = 0.34
    offset = 0.20
    jitter_width = 0.06

    ymax = df[METRIC].max()
    if pd.isna(ymax) or ymax == 0:
        ymax = 1
    upper_ylim = ymax + max(1.0, ymax * 0.30)

    for idx, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[idx]
        d_treat = df[df["Treatment"].astype(str) == treatment].copy()

        ax.set_title(
            TREATMENT_LABELS.get(treatment, treatment),
            fontsize=23,
            fontweight="bold",
            pad=14,
        )

        x = np.arange(len(INJURY_ORDER))

        for i, injury in enumerate(INJURY_ORDER):
            for probe, colour, point_colour, xpos in [
                (0, COLOR_0MM, POINT_0MM, x[i] - offset),
                (5, COLOR_5MM, POINT_5MM, x[i] + offset),
            ]:
                vals = d_treat[
                    (d_treat["Injury"].astype(str) == injury) &
                    (d_treat["Probe"].astype(int) == probe)
                ][METRIC].dropna()

                if vals.empty:
                    continue

                mean_val = vals.mean()
                sem_val = sem(vals)

                ax.bar(
                    xpos,
                    mean_val,
                    width=bar_width,
                    color=colour,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=2,
                )

                if pd.notna(sem_val):
                    ax.errorbar(
                        xpos,
                        mean_val,
                        yerr=sem_val,
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.5,
                        capsize=5,
                        capthick=1.5,
                        zorder=3,
                    )

                rng = np.random.default_rng(
                    seed=3000 + i * 10 + int(probe) + TREATMENT_ORDER.index(treatment) * 100
                )
                jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals))

                ax.scatter(
                    np.full(len(vals), xpos) + jitter,
                    vals,
                    s=POINT_SIZE,
                    color=point_colour,
                    edgecolor="black",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=4,
                )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)

        ax.tick_params(axis="both", labelsize=18, width=1.5, length=6, direction="out")
        ax.set_xticks(x)
        ax.set_xticklabels([INJURY_LABELS[i] for i in INJURY_ORDER], fontsize=19)
        ax.set_xlabel("Injury group", fontsize=21, weight="bold")

        if idx in [0, 2]:
            ax.set_ylabel("Total Chamber 2 entries", fontsize=21, weight="bold")
        else:
            ax.set_ylabel("")

        ax.set_xlim(-0.65, len(INJURY_ORDER) - 0.35)
        ax.set_ylim(0, upper_ylim)

    handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=COLOR_0MM,
            markeredgecolor="black",
            markersize=16,
            label="0 mm",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=COLOR_5MM,
            markeredgecolor="black",
            markersize=16,
            label="5 mm",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    fig.subplots_adjust(
        top=0.89,
        bottom=0.08,
        left=0.08,
        right=0.98,
        wspace=0.16,
        hspace=0.42,
    )

    fig.savefig(f"{out_prefix}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    
def main():
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = INPUT_CSV

    df = load_data(input_csv)
    make_bar_plot(df, OUT_PREFIX)

    print("Done. Files saved:")
    print(f"  {OUT_PREFIX}.png")
    print(f"  {OUT_PREFIX}.pdf")


if __name__ == "__main__":
    main()
