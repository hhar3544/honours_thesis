import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Synthetic escape latency dataset generator
# ============================================================
# Generates n = 12 synthetic values per group.
# Values are generated from a normal distribution and rescaled
# so each group preserves the target mean and estimated SD.
#
# Output CSV contains only:
# Mouse_ID, Treatment, Injury, Probe, Escape_latency_s
# ============================================================

np.random.seed(42)

n_literature = 6
n_synthetic = 12

output_path = (".")

# Mean and SEM values used to estimate SD
# Estimated SD = SEM * sqrt(6)
params = [
    # Vehicle
    {"Treatment": "Vehicle", "Injury": "Naive", "Probe": "0mm", "Mean": 1.50, "SEM": 0.45},
    {"Treatment": "Vehicle", "Injury": "Naive", "Probe": "5mm", "Mean": 10.18, "SEM": 2.65},
    {"Treatment": "Vehicle", "Injury": "CCI", "Probe": "0mm", "Mean": 4.40, "SEM": 1.10},
    {"Treatment": "Vehicle", "Injury": "CCI", "Probe": "5mm", "Mean": 16.85, "SEM": 0.49},

    # Pregabalin
    {"Treatment": "Pregabalin", "Injury": "Naive", "Probe": "0mm", "Mean": 1.50, "SEM": 0.45},
    {"Treatment": "Pregabalin", "Injury": "Naive", "Probe": "5mm", "Mean": 10.18, "SEM": 2.65},
    {"Treatment": "Pregabalin", "Injury": "CCI", "Probe": "0mm", "Mean": 4.40, "SEM": 1.10},
    {"Treatment": "Pregabalin", "Injury": "CCI", "Probe": "5mm", "Mean": 8.94, "SEM": 1.78},

    # Morphine
    {"Treatment": "Morphine", "Injury": "Naive", "Probe": "0mm", "Mean": 4.59, "SEM": 1.57},
    {"Treatment": "Morphine", "Injury": "Naive", "Probe": "5mm", "Mean": 16.80, "SEM": 3.39},
    {"Treatment": "Morphine", "Injury": "CCI", "Probe": "0mm", "Mean": 4.40, "SEM": 1.10},
    {"Treatment": "Morphine", "Injury": "CCI", "Probe": "5mm", "Mean": 11.27, "SEM": 2.38},
]


def generate_exact_normal_values(target_mean, target_sd, n, min_value=0):
    """
    Generate n normally distributed synthetic values.
    Values are rescaled to preserve the exact target mean and sample SD.
    Negative escape latency values are rejected and regenerated.
    """
    for _ in range(10000):
        raw = np.random.normal(loc=0, scale=1, size=n)

        values = target_mean + ((raw - raw.mean()) / raw.std(ddof=1)) * target_sd

        if np.all(values >= min_value):
            return values

    raise ValueError(
        f"Could not generate positive values for mean={target_mean}, SD={target_sd}"
    )


rows = []

for group in params:
    estimated_sd = group["SEM"] * np.sqrt(n_literature)

    values = generate_exact_normal_values(
        target_mean=group["Mean"],
        target_sd=estimated_sd,
        n=n_synthetic,
        min_value=0,
    )

    for i, value in enumerate(values, start=1):
        rows.append({
            "Mouse_ID": f"{group['Treatment']}_{group['Injury']}_{group['Probe']}_{i:02d}",
            "Treatment": group["Treatment"],
            "Injury": group["Injury"],
            "Probe": group["Probe"],
            "Escape_latency_s": round(value, 3),
        })

synthetic_df = pd.DataFrame(rows)

synthetic_df.to_csv(output_path, index=False)

print(f"Saved synthetic dataset to: {output_path.resolve()}")
print(synthetic_df.head())
