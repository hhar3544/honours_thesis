"""
MCA Chamber 2 crossing and turn-back graph generation from SimBA Detailed ROI data.

This version is for visual output only.
It does NOT run ANOVA, Tukey tests, post hoc tests, significance brackets, p-values, or stats output files.

Input:
    Detailed ROI CSV exported from SimBA.

Outputs:
    chamber2_crossings_bar.png
    chamber2_turnbacks_bar.png

Definitions:
    Full crossing:
        Chamber1 -> Chamber2 -> Chamber3
        Chamber3 -> Chamber2 -> Chamber1

    Turn-back / aborted entry:
        Chamber1 -> Chamber2 -> Chamber1
        Chamber3 -> Chamber2 -> Chamber3
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "/Users/yourname/Desktop/project_folder/cleaned_Roi.csv"
OUTPUT_DIR = SCRIPT_DIR

CROSSINGS_PNG = "chamber2_crossings_bar.png"
TURNBACKS_PNG = "chamber2_turnbacks_bar.png"

# Colours
COLOR_0MM = "#2C7FB8"   # blue
COLOR_5MM = "#F28E2B"   # orange

# Darker overlaid individual points
POINT_0MM = "#084081"   # darker blue
POINT_5MM = "#A64B00"   # darker orange

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


# =========================
# STYLE
# =========================
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

def sem(x):
    """Used only for the graph error bars, not inferential statistics."""
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def parse_video_name(video_name):
    name = Path(str(video_name)).stem

    probe_match = re.search(r"-(0|5)$", name)
    if not probe_match:
        raise ValueError(f"Could not parse probe height from VIDEO name: {video_name}")
    probe = int(probe_match.group(1))

    if "_" not in name:
        raise ValueError(f"Could not parse condition/animal from VIDEO name: {video_name}")

    condition_part, animal_probe_part = name.split("_", 1)
    treatment, injury = condition_part.rsplit("-", 1)
    animal_code = re.sub(r"-(0|5)$", "", animal_probe_part)

    return treatment, injury, animal_code, probe


def standardise_roi_name(x):
    x = str(x).strip().lower()

    if x in ["chamber1", "chamber 1", "chamber_1"]:
        return "Chamber1"
    if x in ["chamber2", "chamber 2", "chamber_2"]:
        return "Chamber2"
    if x in ["chamber3", "chamber 3", "chamber_3"]:
        return "Chamber3"

    return np.nan


def collapse_consecutive_rois(roi_sequence):
    collapsed = []

    for roi in roi_sequence:
        if pd.isna(roi):
            continue
        if not collapsed or roi != collapsed[-1]:
            collapsed.append(roi)

    return collapsed


def count_transitions_for_sequence(seq):
    crossing_1_to_3 = 0
    crossing_3_to_1 = 0
    turnback_to_1 = 0
    turnback_to_3 = 0

    for i in range(len(seq) - 2):
        triplet = tuple(seq[i:i + 3])

        if triplet == ("Chamber1", "Chamber2", "Chamber3"):
            crossing_1_to_3 += 1

        elif triplet == ("Chamber3", "Chamber2", "Chamber1"):
            crossing_3_to_1 += 1

        elif triplet == ("Chamber1", "Chamber2", "Chamber1"):
            turnback_to_1 += 1

        elif triplet == ("Chamber3", "Chamber2", "Chamber3"):
            turnback_to_3 += 1

    return {
        "Crossing_1_to_3": crossing_1_to_3,
        "Crossing_3_to_1": crossing_3_to_1,
        "Total_crossings": crossing_1_to_3 + crossing_3_to_1,
        "Turnback_1_to_2_to_1": turnback_to_1,
        "Turnback_3_to_2_to_3": turnback_to_3,
        "Total_turnbacks": turnback_to_1 + turnback_to_3,
    }


def load_and_count_transitions():
    input_csv = Path(INPUT_CSV).expanduser()

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find input CSV:\n  {input_csv}\n\n"
            "Fix this by either:\n"
            "  1. putting crossing_turn_ROI.csv in the same folder as this script, OR\n"
            "  2. running the script with the CSV path, for example:\n"
            "     python3 chamber2_crossing_turnback_graphs_only.py /Users/harvinharae/Desktop/honours_python/ch2_crossing_turn/crossing_turn_ROI.csv"
        )

    df = pd.read_csv(input_csv)

    required = {"VIDEO", "BODY-PART", "SHAPE NAME", "START FRAME", "END FRAME"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns are: {list(df.columns)}"
        )

    df = df[df["BODY-PART"].astype(str).str.lower().eq("nose")].copy()

    df["ROI"] = df["SHAPE NAME"].apply(standardise_roi_name)
    df = df.dropna(subset=["ROI"]).copy()

    df["START FRAME"] = pd.to_numeric(df["START FRAME"], errors="coerce")
    df["END FRAME"] = pd.to_numeric(df["END FRAME"], errors="coerce")
    df = df.dropna(subset=["START FRAME", "END FRAME"]).copy()
    df = df.sort_values(["VIDEO", "START FRAME", "END FRAME"])

    rows = []

    for video, d_video in df.groupby("VIDEO", sort=False):
        treatment, injury, animal_code, probe = parse_video_name(video)
        animal_uid = f"{treatment}-{injury}_{animal_code}"

        roi_seq = d_video["ROI"].tolist()
        collapsed_seq = collapse_consecutive_rois(roi_seq)
        counts = count_transitions_for_sequence(collapsed_seq)

        rows.append({
            "VIDEO": video,
            "Treatment": treatment,
            "Treatment_label": TREATMENT_LABELS.get(treatment, treatment),
            "Injury": injury,
            "Injury_label": INJURY_LABELS.get(injury, injury),
            "AnimalCode": animal_code,
            "AnimalUID": animal_uid,
            "Probe": probe,
            "Probe_label": PROBE_LABELS.get(probe, str(probe)),
            "n_roi_bouts_raw": len(roi_seq),
            "n_roi_bouts_collapsed": len(collapsed_seq),
            **counts,
        })

    summary = pd.DataFrame(rows)

    summary["Treatment"] = pd.Categorical(summary["Treatment"], categories=TREATMENT_ORDER, ordered=True)
    summary["Injury"] = pd.Categorical(summary["Injury"], categories=INJURY_ORDER, ordered=True)
    summary["Probe"] = pd.Categorical(summary["Probe"], categories=PROBE_ORDER, ordered=True)

    return summary


def make_bar_plot(summary, outcome_col, output_png, y_label, title):
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

    ymax = summary[outcome_col].max()
    if pd.isna(ymax) or ymax == 0:
        ymax = 1

    upper_ylim = ymax + max(1.0, ymax * 0.30)

    for idx, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[idx]
        d_treat = summary[summary["Treatment"].astype(str) == treatment].copy()

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
                ][outcome_col].dropna()

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

                rng = np.random.default_rng(seed=2000 + idx * 100 + i * 10 + int(probe))
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
            ax.set_ylabel(y_label, fontsize=21, weight="bold")
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

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path(OUTPUT_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_and_count_transitions()

    crossings_path = output_dir / CROSSINGS_PNG
    turnbacks_path = output_dir / TURNBACKS_PNG

    make_bar_plot(
        summary=summary,
        outcome_col="Total_crossings",
        output_png=crossings_path,
        y_label="Number of full crossings",
        title="Full crossings through Chamber 2",
    )

    make_bar_plot(
        summary=summary,
        outcome_col="Total_turnbacks",
        output_png=turnbacks_path,
        y_label="Number of turn-backs",
        title="Turn-back entries into Chamber 2",
    )

    print("Saved figures:")
    print(f"  {crossings_path}")
    print(f"  {turnbacks_path}")

    print("\nDefinitions:")
    print("  Full crossing = Chamber1 → Chamber2 → Chamber3 OR Chamber3 → Chamber2 → Chamber1")
    print("  Turn-back = Chamber1 → Chamber2 → Chamber1 OR Chamber3 → Chamber2 → Chamber3")

    print("\nGraph annotations:")
    print("  No ANOVA, Tukey tests, p-values, stars, or significance brackets are generated.")


if __name__ == "__main__":
    main()
