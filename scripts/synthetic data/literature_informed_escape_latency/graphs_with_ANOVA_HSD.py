"""
Synthetic escape latency graph with planned Tukey HSD annotations.

- Reads synthetic_escape_latency_n12.csv
- Runs three-way ANOVA: Treatment × Injury × Probe
- Runs Tukey HSD on combined groups
- Annotates significant planned comparisons only
- Saves PNG, PDF, ANOVA CSV and planned Tukey CSV
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = "/Users/yourname/Desktop/project_folder/Synthetic_escape.csv"
OUTPUT_DIR = SCRIPT_DIR

OUT_PREFIX = "synthetic_escape_latency"

OUTCOME_COL = "Escape_latency_s"

TREATMENT_ORDER = ["Vehicle", "Pregabalin", "Morphine", "Compound 1"]
INJURY_ORDER = ["Naive", "CCI"]
PROBE_ORDER = ["0mm", "5mm"]

TREATMENT_LABELS = {
    "Vehicle": "Vehicle",
    "Pregabalin": "Pregabalin",
    "Morphine": "Morphine",
    "Compound 1": "Compound 1",
}

INJURY_LABELS = {
    "Naive": "Naïve",
    "CCI": "CCI",
}

PROBE_LABELS = {
    "0mm": "0 mm",
    "5mm": "5 mm",
}

COLOR_0MM = "#2C7FB8"
COLOR_5MM = "#F28E2B"

PROBE_COLORS = {
    "0mm": COLOR_0MM,
    "5mm": COLOR_5MM,
}

POINT_COLORS = {
    "0mm": "#084081",
    "5mm": "#A64B00",
}

POINT_SIZE = 115
DPI = 600

# Which Tukey comparisons to annotate
ANNOTATE_PROBE_WITHIN_TREATMENT_INJURY = True
ANNOTATE_INJURY_WITHIN_TREATMENT_PROBE = True
ANNOTATE_TREATMENT_WITHIN_INJURY_PROBE = False  # usually too crowded for this graph

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


# ============================================================
# FUNCTIONS
# ============================================================
def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def p_to_stars(p):
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def clean_probe(x):
    x = str(x).strip()
    if x in ["0", "0.0", "0 mm", "0mm"]:
        return "0mm"
    if x in ["5", "5.0", "5 mm", "5mm"]:
        return "5mm"
    return x


def make_group(row):
    return f"{row['Treatment']}|{row['Injury']}|{row['Probe']}"


def parse_group(group_name):
    treatment, injury, probe = group_name.split("|")
    return treatment, injury, probe


def load_data(input_csv):
    df = pd.read_csv(input_csv)

    required = {"Treatment", "Injury", "Probe", OUTCOME_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Probe"] = df["Probe"].apply(clean_probe)

    df = df[df["Treatment"].isin(TREATMENT_ORDER)].copy()
    df = df[df["Injury"].isin(INJURY_ORDER)].copy()
    df = df[df["Probe"].isin(PROBE_ORDER)].copy()

    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    df = df.dropna(subset=[OUTCOME_COL, "Treatment", "Injury", "Probe"]).copy()

    df["Treatment"] = pd.Categorical(df["Treatment"], categories=TREATMENT_ORDER, ordered=True)
    df["Injury"] = pd.Categorical(df["Injury"], categories=INJURY_ORDER, ordered=True)
    df["Probe"] = pd.Categorical(df["Probe"], categories=PROBE_ORDER, ordered=True)

    df["Group"] = df.apply(make_group, axis=1)

    return df


def run_anova_and_tukey(df, output_dir):
    model = ols(
        f"{OUTCOME_COL} ~ C(Treatment) * C(Injury) * C(Probe)",
        data=df
    ).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(output_dir / f"{OUT_PREFIX}_anova.csv")

    tukey = pairwise_tukeyhsd(
        endog=df[OUTCOME_COL],
        groups=df["Group"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"], errors="coerce")
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    planned_rows = []

    for _, row in tukey_df.iterrows():
        g1 = row["group1"]
        g2 = row["group2"]

        t1, i1, p1 = parse_group(g1)
        t2, i2, p2 = parse_group(g2)

        comparison_type = None

        # Probe effect within matched Treatment × Injury
        if t1 == t2 and i1 == i2 and p1 != p2:
            comparison_type = "Probe within Treatment × Injury"

        # Injury effect within matched Treatment × Probe
        elif t1 == t2 and p1 == p2 and i1 != i2:
            comparison_type = "Injury within Treatment × Probe"

        # Treatment effect within matched Injury × Probe
        elif i1 == i2 and p1 == p2 and t1 != t2:
            comparison_type = "Treatment within Injury × Probe"

        if comparison_type is not None:
            planned_rows.append({
                "comparison_type": comparison_type,
                "group1": g1,
                "group2": g2,
                "meandiff": row["meandiff"],
                "p_adj": row["p-adj"],
                "lower": row["lower"],
                "upper": row["upper"],
                "reject": row["reject"],
                "stars": p_to_stars(row["p-adj"]),
            })

    planned_df = pd.DataFrame(planned_rows)
    planned_df.to_csv(output_dir / f"{OUT_PREFIX}_planned_tukey.csv", index=False)

    return planned_df


def add_sig_bar(ax, x1, x2, y, h, stars, fontsize=20):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=1.8,
        color="black",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        stars,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        weight="bold",
        clip_on=False,
    )


def make_plot(df, planned_df, output_dir):
    treatments_present = [t for t in TREATMENT_ORDER if t in df["Treatment"].astype(str).unique()]
    injuries_present = [i for i in INJURY_ORDER if i in df["Injury"].astype(str).unique()]

    x_base = {inj: i for i, inj in enumerate(injuries_present)}
    offset = {"0mm": -0.20, "5mm": 0.20}
    bar_width = 0.34

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13.5, 10.5),
        sharey=True,
        constrained_layout=False,
    )

    axes = axes.flatten()

    ymax_global = df[OUTCOME_COL].max()
    if pd.isna(ymax_global) or ymax_global <= 0:
        ymax_global = 1

    # Add extra room for significance brackets
    upper_ylim = ymax_global + max(3.0, ymax_global * 0.45)

    x_lookup = {}

    for idx, treatment in enumerate(treatments_present):
        ax = axes[idx]
        sub = df[df["Treatment"].astype(str) == treatment].copy()
        label = TREATMENT_LABELS.get(treatment, treatment)

        for injury in injuries_present:
            for probe in PROBE_ORDER:
                grp = sub[
                    (sub["Injury"].astype(str) == injury) &
                    (sub["Probe"].astype(str) == probe)
                ].copy()

                if grp.empty:
                    continue

                x = x_base[injury] + offset[probe]
                y = grp[OUTCOME_COL].dropna().values

                mean_y = np.nanmean(y)
                sem_y = sem(y)

                x_lookup[(treatment, injury, probe)] = x

                ax.bar(
                    x,
                    mean_y,
                    width=bar_width,
                    color=PROBE_COLORS[probe],
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

                jitter = np.linspace(-0.06, 0.06, len(y)) if len(y) > 1 else np.array([0.0])

                ax.scatter(
                    np.full(len(y), x) + jitter,
                    y,
                    s=POINT_SIZE,
                    color=POINT_COLORS[probe],
                    edgecolor="black",
                    linewidth=0.9,
                    alpha=0.95,
                    zorder=4,
                )

        # Significance brackets within this treatment panel
        sigs = planned_df[
            (planned_df["reject"] == True) &
            (planned_df["stars"] != "")
        ].copy()

        sigs_panel = []

        for _, row in sigs.iterrows():
            t1, i1, p1 = parse_group(row["group1"])
            t2, i2, p2 = parse_group(row["group2"])

            if t1 != treatment or t2 != treatment:
                continue

            if row["comparison_type"] == "Probe within Treatment × Injury" and ANNOTATE_PROBE_WITHIN_TREATMENT_INJURY:
                sigs_panel.append(row)

            elif row["comparison_type"] == "Injury within Treatment × Probe" and ANNOTATE_INJURY_WITHIN_TREATMENT_PROBE:
                sigs_panel.append(row)

            elif row["comparison_type"] == "Treatment within Injury × Probe" and ANNOTATE_TREATMENT_WITHIN_INJURY_PROBE:
                sigs_panel.append(row)

        # Draw brackets, stacked upwards
        y_start = ymax_global + max(0.8, ymax_global * 0.05)
        y_step = max(1.5, ymax_global * 0.08)
        h = max(0.5, ymax_global * 0.025)

        for b_idx, row in enumerate(sigs_panel):
            t1, i1, p1 = parse_group(row["group1"])
            t2, i2, p2 = parse_group(row["group2"])

            key1 = (t1, i1, p1)
            key2 = (t2, i2, p2)

            if key1 not in x_lookup or key2 not in x_lookup:
                continue

            x1 = x_lookup[key1]
            x2 = x_lookup[key2]

            if abs(x1 - x2) < 0.01:
                continue

            y = y_start + b_idx * y_step
            add_sig_bar(ax, x1, x2, y, h, row["stars"])

        ax.set_ylim(0, upper_ylim)

        ax.set_title(label, fontsize=23, pad=14, weight="bold")
        ax.set_xlabel("Injury group", fontsize=21, weight="bold")
        ax.set_xticks([x_base[i] for i in injuries_present])
        ax.set_xticklabels([INJURY_LABELS[i] for i in injuries_present], fontsize=19)

        if idx in [0, 2]:
            ax.set_ylabel("Escape latency (s)", fontsize=21, weight="bold")
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="both", labelsize=18, width=1.5, length=6, direction="out")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)

    for j in range(len(treatments_present), len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="0 mm",
            markerfacecolor=PROBE_COLORS["0mm"],
            markeredgecolor="black",
            markersize=16,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="5 mm",
            markerfacecolor=PROBE_COLORS["5mm"],
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

    fig.subplots_adjust(
        top=0.89,
        bottom=0.08,
        left=0.08,
        right=0.98,
        wspace=0.16,
        hspace=0.42,
    )

    png_path = output_dir / f"{OUT_PREFIX}.png"
    pdf_path = output_dir / f"{OUT_PREFIX}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure outputs:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    output_dir = Path(OUTPUT_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_CSV)

    planned_df = run_anova_and_tukey(df, output_dir)
    make_plot(df, planned_df, output_dir)

    print("\nSaved stats outputs:")
    print(f"  {output_dir / f'{OUT_PREFIX}_anova.csv'}")
    print(f"  {output_dir / f'{OUT_PREFIX}_planned_tukey.csv'}")


if __name__ == "__main__":
    main()
