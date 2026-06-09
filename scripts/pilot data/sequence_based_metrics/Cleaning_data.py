"""
Clean SimBA ROI entry events by merging fragmented ROI bouts and removing very short events.

Input expected: SimBA Detailed ROI data CSV with columns:
VIDEO, ANIMAL, BODY-PART, SHAPE NAME, START TIME, END TIME, START FRAME, END FRAME, DURATION (S)

Outputs:
1) cleaned_roi_events.csv   = cleaned/merged bout-level ROI events
2) cleaned_roi_summary.csv  = summary metrics per video/animal/body-part/ROI

Recommended starting thresholds for MCA:
- merge_gap_s = 0.30: merge events separated by <0.3 s
- min_duration_s = 0.20: remove bouts shorter than 0.2 s after merging
- new_entry_gap_s = 1.00: optional stricter count; only count a new visit after animal has been outside ROI for >=1 s
"""

from pathlib import Path
import pandas as pd


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = Path("/Users/yourname/Desktop/project_folder/Raw_ROI.csv")
OUTPUT_DIR = Path("cleaned_roi_outputs")

# Main cleaning thresholds
MERGE_GAP_S = 0.30       # merge fragmented events separated by less than this
MIN_DURATION_S = 0.20    # remove events shorter than this after merging
NEW_ENTRY_GAP_S = 1.00   # stricter visit count: requires this much time outside ROI before counting a new visit

# Optional filters
BODY_PARTS_TO_KEEP = ["Nose"]   # use ["Nose"] for chamber entry; set to None to keep all body-parts
ROIS_TO_KEEP = None             # e.g. ["Chamber2", "Chamber3"]; set to None to keep all ROIs


# =========================
# CLEANING FUNCTIONS
# =========================
def merge_roi_events(group: pd.DataFrame, merge_gap_s: float, min_duration_s: float) -> pd.DataFrame:
    """Merge close ROI events within one VIDEO/ANIMAL/BODY-PART/SHAPE NAME group."""
    group = group.sort_values(["START TIME", "END TIME"]).reset_index(drop=True)

    merged_rows = []
    current = None

    for _, row in group.iterrows():
        if current is None:
            current = row.copy()
            continue

        gap = row["START TIME"] - current["END TIME"]

        if gap <= merge_gap_s:
            # Same visit/bout: extend the current event
            current["END TIME"] = max(current["END TIME"], row["END TIME"])
            current["END FRAME"] = max(current["END FRAME"], row["END FRAME"])
        else:
            # Finish previous event and start a new one
            current["DURATION (S)"] = current["END TIME"] - current["START TIME"]
            merged_rows.append(current)
            current = row.copy()

    if current is not None:
        current["DURATION (S)"] = current["END TIME"] - current["START TIME"]
        merged_rows.append(current)

    merged = pd.DataFrame(merged_rows)
    merged = merged[merged["DURATION (S)"] >= min_duration_s].copy()
    return merged


def count_strict_entries(events: pd.DataFrame, new_entry_gap_s: float) -> int:
    """
    Counts entries only when the animal has been outside the ROI for at least new_entry_gap_s.
    This is stricter than simple bout count and helps avoid boundary re-entry inflation.
    """
    if events.empty:
        return 0

    events = events.sort_values("START TIME").reset_index(drop=True)
    count = 1
    previous_end = events.loc[0, "END TIME"]

    for i in range(1, len(events)):
        gap = events.loc[i, "START TIME"] - previous_end
        if gap >= new_entry_gap_s:
            count += 1
        previous_end = max(previous_end, events.loc[i, "END TIME"])

    return count


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Remove SimBA's unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    required_cols = [
        "VIDEO", "ANIMAL", "BODY-PART", "SHAPE NAME",
        "START TIME", "END TIME", "START FRAME", "END FRAME", "DURATION (S)"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optional body-part and ROI filtering
    if BODY_PARTS_TO_KEEP is not None:
        df = df[df["BODY-PART"].isin(BODY_PARTS_TO_KEEP)].copy()

    if ROIS_TO_KEEP is not None:
        df = df[df["SHAPE NAME"].isin(ROIS_TO_KEEP)].copy()

    # Merge close events and remove short events per video/animal/body-part/ROI
    group_cols = ["VIDEO", "ANIMAL", "BODY-PART", "SHAPE NAME"]
    cleaned_parts = []

    for _, group in df.groupby(group_cols, dropna=False):
        cleaned_group = merge_roi_events(group, MERGE_GAP_S, MIN_DURATION_S)
        if not cleaned_group.empty:
            cleaned_parts.append(cleaned_group)

    cleaned = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else pd.DataFrame(columns=df.columns)
    cleaned = cleaned.sort_values(group_cols + ["START TIME"]).reset_index(drop=True)

    # Build summary metrics
    summary_rows = []
    for keys, group in cleaned.groupby(group_cols, dropna=False):
        video, animal, bodypart, roi = keys
        group = group.sort_values("START TIME")

        summary_rows.append({
            "VIDEO": video,
            "ANIMAL": animal,
            "BODY-PART": bodypart,
            "ROI": roi,
            "CLEANED_VISIT_COUNT": len(group),
            "STRICT_ENTRY_COUNT_OUTSIDE_GAP": count_strict_entries(group, NEW_ENTRY_GAP_S),
            "FIRST_ROI_ENTRY_TIME_S": group["START TIME"].min(),
            "TOTAL_ROI_TIME_S": group["DURATION (S)"].sum(),
            "MEAN_VISIT_DURATION_S": group["DURATION (S)"].mean(),
            "MEDIAN_VISIT_DURATION_S": group["DURATION (S)"].median(),
            "MAX_VISIT_DURATION_S": group["DURATION (S)"].max(),
            "MERGE_GAP_S_USED": MERGE_GAP_S,
            "MIN_DURATION_S_USED": MIN_DURATION_S,
            "NEW_ENTRY_GAP_S_USED": NEW_ENTRY_GAP_S,
        })

    summary = pd.DataFrame(summary_rows)

    # Add raw counts for comparison
    raw_counts = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="RAW_SIMBA_EVENT_COUNT")
        .rename(columns={"SHAPE NAME": "ROI"})
    )

    summary = summary.merge(
        raw_counts,
        on=["VIDEO", "ANIMAL", "BODY-PART", "ROI"],
        how="left"
    )

    summary["EVENTS_REMOVED_OR_MERGED"] = (
        summary["RAW_SIMBA_EVENT_COUNT"] - summary["CLEANED_VISIT_COUNT"]
    )

    # Save outputs
    cleaned_out = OUTPUT_DIR / "cleaned_roi_events.csv"
    summary_out = OUTPUT_DIR / "cleaned_roi_summary.csv"

    cleaned.to_csv(cleaned_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("Done.")
    print(f"Raw events: {len(df)}")
    print(f"Cleaned events: {len(cleaned)}")
    print(f"Saved cleaned events to: {cleaned_out}")
    print(f"Saved summary to: {summary_out}")


if __name__ == "__main__":
    main()

