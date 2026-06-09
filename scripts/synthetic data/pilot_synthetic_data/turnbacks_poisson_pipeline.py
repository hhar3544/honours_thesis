# -*- coding: utf-8 -*-
"""
# Why use the Poisson Distribution for Count data
When you compare means using a t‑test or ANOVA, you assume:



1.   Data are approximately normal
2.   Variance is unrelated to the mean
3.   Differences act additively


For count data:

1. The distribution is skewed
2. Variance grows with the mean
3. Effects act multiplicatively

Bernouli data has:

$SD= \sqrt{ p(1−p)}$


Explanation:

Although mean event counts can be calculated for each group, count data arise from discrete stochastic processes in which the variance depends on the mean. As a result, comparing means using methods that assume normally distributed errors can be misleading. Instead, generalized linear models account for the mean–variance relationship and allow effects to be interpreted in terms of rate ratios, which are more appropriate for event count data.

# Create Synthetic Dataset

---



Here we create a synthetic dataset based on the Poisson distribution for count data with multiple treatment groups and conditions.

Here we have 4 treatment groups, 2 injury states, and 2 experimental cage conditions.

Note that this is simplified because the time of exposure is the same for all experiments so that part of the formula is not needed and not shown here.

This example includes creating a synthetic dataset, analysing results, and plotting the effects. You will need to change the name for events and treatments and injury in the dataset to you can show how it relates to real treatments and surgeries.
"""

# FULL CROSSING
# These effects are the multipliers for your synthetic data.
# I do the same thing in Datamanagement class in
#Excel where I multiply =rand() by different numbers.
# So change these depending on whether you expect an increase
# or decrease depending on Rx, injury, probe.
import numpy as np
import pandas as pd
baseline_rate = 2.5  # events per experiment
treatment_effect = {
    "Saline": 1.0,
    "Pregabalin": 0.85,
    "Compound 1": 0.75,
    "Morphine": 0.65
}

injury_effect = {
    "No": 1.0,
    "Yes": 1.8
}

cage_effect = {
    "Probe": 2.5,
    "No-probe": 1.0
}

treatment_injury_rescue = {
    ("Saline", "Yes"): 1.0,
    ("Pregabalin", "Yes"): 0.75,
    ("Compound 1", "Yes"): 0.60,
    ("Morphine", "Yes"): 0.45
}

cage_injury_effect = {
    ("Probe", "Yes"): 1.6,
    ("Probe", "No"): 1.1
}

# Had to add in interaction between Rx and injury effect so some Rx can rescue.
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

treatments = ["Saline", "Pregabalin", "Compound 1", "Morphine"]
injuries = ["No", "Yes"]
cages = ["No-probe", "Probe"]

rows = []
replicates_per_cell = 12

for trt in treatments:
    for inj in injuries:
        for cage in cages:

            lam = (
                baseline_rate
                * treatment_effect[trt]
                * injury_effect[inj]
                * cage_effect[cage]
                * treatment_injury_rescue.get((trt, inj), 1.0)
                * cage_injury_effect.get((cage, inj), 1.0)
            )

            counts = rng.poisson(lam=lam, size=replicates_per_cell)

            for c in counts:
                rows.append({
                    "count": c,
                    "treatment": trt,
                    "injury": inj,
                    "cage": cage
                })

df = pd.DataFrame(rows)

for col in ["treatment", "injury", "cage"]:
    df[col] = df[col].astype("category")

# Create dataset
'''
You should see:

Lower counts for injured animals
Higher counts for treatments with larger multipliers
Slight reduction for probe-containing cages
'''

df = pd.DataFrame(rows)

df["treatment"] = df["treatment"].astype("category")
df["injury"] = df["injury"].astype("category")
df["cage"] = df["cage"].astype("category")

df.head(2)

"""## Poisson Model is:


$λ = λ_{0} \times Treatment×Injury×Cage$

**Taking logs:**

$log(λ)=log(λ0)+log(Treatment)+log⁡(Injury)+log⁡(Cage)$
"""

# Observe group means
df.groupby(["treatment", "injury", "cage"])["count"].mean()

# Observe effect of treatments where counts return for T1.
df.groupby(["treatment", "injury"])["count"].mean().unstack()

"""# Analysis using generalized linear model with a Poisson distribution and log link

---



See below we use the negative binomial GLM.'

We use a negative binomial GLM because the event count data are overdispersed: the variability in counts is larger than expected under a Poisson model. The negative binomial distribution generalizes the Poisson by allowing the variance to exceed the mean. This is very common in real data so the negative binomial is more useful than the straight Poisson distribution. You can imagine it as Poisson plus a little extra...

"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

df = pd.DataFrame(rows)
df["treatment"] = pd.Categorical(
    df["treatment"],
    categories=["Saline", "Morphine", "Pregabalin", "Compound 1"],
    ordered=True
)

df["injury"] = pd.Categorical(
    df["injury"],
    categories=["No", "Yes"],
    ordered=True
)

df["cage"] = pd.Categorical(
    df["cage"],
    categories=["No-probe", "Probe"],
    ordered=True
)

model = smf.glm(
    formula="count ~ treatment * injury * cage",
    data=df,
    family=sm.families.NegativeBinomial()
).fit()

print(model.summary())

"""The model shows that overdispersion was not really a problem here but still the negative binomial is a conservative choice and we will stick with it because it is quite likely with real data.

$Dispersion \space ratio = \frac{Pearson \space χ^2}{Df_{residual}}$


"""

DR = 41.5/176
print(DR)

model.params # this is in the log dimension...

# Convert to rate ratios
np.exp(model.params)

"""Turn-back counts were analysed using a generalized linear model with a log link, including treatment, injury state, probe condition and all interaction terms. Results are reported as rate ratios relative to the reference condition: Saline-treated, uninjured animals in the no-probe condition.

The predicted reference count was 2.67 turn-backs. Relative to Saline, turn-back rates were lower under Morphine (RR = 0.66), Pregabalin (RR = 0.59) and Compound 1 (RR = 0.78). Injury increased turn-backs by 81% in Saline-treated animals (RR = 1.81), while probe exposure increased turn-backs by 150% (RR = 2.50), consistent with turn-backs reflecting hesitation or avoidance-like behaviour.

Treatment modified the injury effect. Under no-probe conditions, Morphine reduced injury-associated turn-backs (combined injury effect: RR = 0.86; predicted count = 1.50), while Compound 1 showed an intermediate reduction (RR = 1.28; 2.67 counts). Pregabalin showed less rescue, with injured animals retaining a higher turn-back rate (RR = 1.89; 3.00 counts), while injured Saline-treated animals showed 4.83 predicted turn-backs.

Under injury + probe conditions, Saline produced the highest predicted turn-back count (17.42), followed by Pregabalin (13.50), Compound 1 (8.33) and Morphine (4.83). Overall, the model suggests that injury and probe exposure increase turn-backs, while Morphine most strongly reduces this avoidance-like response, with Compound 1 showing an intermediate effect.

# Plotting the Results

---
"""

# ============================================================
# PLOTTING THE RESULTS: THESIS-STYLE BAR GRAPH
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# Colours matched to thesis figures
COLOR_0MM = "#2C7FB8"
COLOR_5MM = "#F28E2B"
POINT_0MM = "#084081"
POINT_5MM = "#A64B00"

PROBE_COLORS = {
    "No-probe": COLOR_0MM,
    "Probe": COLOR_5MM,
}

POINT_COLORS = {
    "No-probe": POINT_0MM,
    "Probe": POINT_5MM,
}

INJURY_ORDER = ["No", "Yes"]
INJURY_LABELS = {
    "No": "Naïve",
    "Yes": "CCI",
}

PROBE_ORDER = ["No-probe", "Probe"]

TREATMENT_ORDER = ["Saline", "Morphine", "Pregabalin", "Compound 1"]

POINT_SIZE = 115
bar_width = 0.34
offset = {"No-probe": -0.20, "Probe": 0.20}


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


# Use individual synthetic values, not pre-averaged means
plot_df = df.copy()

plot_df["treatment"] = pd.Categorical(
    plot_df["treatment"],
    categories=TREATMENT_ORDER,
    ordered=True
)

plot_df["injury"] = pd.Categorical(
    plot_df["injury"],
    categories=INJURY_ORDER,
    ordered=True
)

plot_df["cage"] = pd.Categorical(
    plot_df["cage"],
    categories=PROBE_ORDER,
    ordered=True
)

treatments_present = [
    t for t in TREATMENT_ORDER
    if t in plot_df["treatment"].astype(str).unique()
]

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(13.5, 10.5),
    sharey=True,
    constrained_layout=False,
)

axes = axes.flatten()

ymax = plot_df["count"].max()
if np.isnan(ymax) or ymax <= 0:
    ymax = 1

upper_ylim = ymax + max(1.0, ymax * 0.30)

for idx, trt in enumerate(treatments_present):
    ax = axes[idx]
    sub = plot_df[plot_df["treatment"].astype(str) == trt].copy()

    x_base = {inj: i for i, inj in enumerate(INJURY_ORDER)}

    for inj in INJURY_ORDER:
        for probe in PROBE_ORDER:
            grp = sub[
                (sub["injury"].astype(str) == inj) &
                (sub["cage"].astype(str) == probe)
            ].copy()

            if grp.empty:
                continue

            x = x_base[inj] + offset[probe]
            y = grp["count"].dropna().values

            if len(y) == 0:
                continue

            mean_y = np.nanmean(y)
            sem_y = sem(y)

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

    ax.set_title(str(trt), fontsize=23, pad=14, weight="bold")
    ax.set_xlabel("Injury group", fontsize=21, weight="bold")
    ax.set_xticks([x_base[i] for i in INJURY_ORDER])
    ax.set_xticklabels([INJURY_LABELS[i] for i in INJURY_ORDER], fontsize=19)

    if idx in [0, 2]:
        ax.set_ylabel("Chamber 2 turn-backs", fontsize=21, weight="bold")
    else:
        ax.set_ylabel("")

    ax.set_ylim(0, upper_ylim)
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
        markerfacecolor=COLOR_0MM,
        markeredgecolor="black",
        markersize=16,
    ),
    Line2D(
        [0], [0],
        marker="s",
        color="w",
        label="5 mm",
        markerfacecolor=COLOR_5MM,
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

plt.savefig("synthetic_chamber2_turnbacks_bar_2x2.png", dpi=600, bbox_inches="tight")
plt.savefig("synthetic_chamber2_turnbacks_bar_2x2.pdf", dpi=600, bbox_inches="tight")
plt.show()
"""# Comparing the effects of treatment with each other"""

#Step 1: set condition to compare
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# Define condition
injury_level = "Yes"
cage_level = "Probe"

#Step 2: build design rows for each treatment
treatments = df["treatment"].cat.categories

design_rows = []

for t in treatments:
    design_rows.append({
        "treatment": t,
        "injury": injury_level,
        "cage": cage_level
    })

design_df = pd.DataFrame(design_rows)

#Step 3: get model design matrix
import patsy

X = patsy.dmatrix(model.model.data.design_info, design_df)

#Step 4: compute pairwise contrasts
params = model.params.values
cov = model.cov_params().values

results = []

for (i, t1), (j, t2) in combinations(enumerate(treatments), 2):

    L = X[i] - X[j]   # contrast vector

    log_diff = L @ params
    se = np.sqrt(L @ cov @ L)

    z = log_diff / se
    p = 2 * (1 - norm.cdf(abs(z)))

    rr = np.exp(log_diff)

    results.append({
        "comparison": f"{t1} vs {t2}",
        "log_diff": log_diff,
        "rate_ratio": rr,
        "p_raw": p
    })

results_df = pd.DataFrame(results)

# Step 5: apply Holm correction
pvals = results_df["p_raw"].values

reject, p_holm, _, _ = multipletests(pvals, method="holm")

results_df["p_holm"] = p_holm
results_df["significant"] = reject

results_df.sort_values("p_holm")

"""We need to restrict comparisons because the Holm is making everything non-significant even though we encoded a strong biological effect in the synthetic data."""
