#!/usr/bin/env python3
"""
TRIC-seq — Analyze chimeras

Maps per-chimera genomic positions to annotated features and produces
pairwise interaction counts, unique interaction counts (io), and normalized
scores suitable for downstream odds-ratio / null-model analysis.

Input
-----
1) BED-like chimera file with at least 3 columns:
   chrom  start  end  [optional extra columns]
   By default 'start' and 'end' are treated as the two ligation loci.
   Positions can be given as 0-based (default) or 1-based.

2) Feature annotation table (CSV) with at least these columns (names configurable):
   range,startpos,endpos,Type
   - 'range' is the feature ID (e.g., "gene|5pUTR|CDS|3pUTR|sRNA|tRNA|rRNA").
   - positions can be 1-based (default) or 0-based; specify via flags.
   - end positions are treated as inclusive.

Output
------
CSV with columns:
  ref,target,counts,io,totals,total_ref,score,adjusted_score,
  ref_type,target_type,self_interactions_ref,self_interactions_target,self_interaction_score

Plus a JSON sidecar with run metadata (versions, seeds, checksums, params).

Usage
-----
python tricseq_analyze_chimeras.py \
  --chimera-bed data/EC/RX22GD7_chim.bed \
  --annotations data/MX/annotations_MX.csv \
  --out data/EC/analysis_RX22GD7.csv \
  --ann-cols range startpos endpos Type \
  --bed-one-based 0 \
  --ann-one-based 1 \
  --require-both-mapped 0

Notes
-----
- For bacterial genomes (~<15 Mb) the default array-based mapper is very fast.
- Falls back to interval-tree mapping if array allocation fails.
- If some features overlap, the last-written will win in array mode; for strict
  overlap control, pre-resolve overlaps in the annotation step.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import hashlib
import logging
from typing import Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from intervaltree import Interval, IntervalTree  # optional fallback
    _HAS_INTERVALTREE = True
except Exception:
    _HAS_INTERVALTREE = False


def md5sum(path: str) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_analyze_chimeras",
        description="Map chimera ends to features and compute pairwise interaction statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--chimera-bed", required=True, help="BED-like chimera file (chrom start end ...).")
    ap.add_argument("--annotations", required=True, help="CSV with feature ranges.")
    ap.add_argument("--out", required=True, help="Output CSV path for pairwise analysis.")
    ap.add_argument("--manifest", default=None, help="Optional JSON sidecar path. Defaults to <out>.manifest.json")

    # Column names for the annotations CSV
    ap.add_argument("--ann-cols", nargs=4, metavar=("range", "startpos", "endpos", "Type"),
                    default=("range", "startpos", "endpos", "Type"),
                    help="Column names in the annotations CSV (feature id, start, end, type).")
    ap.add_argument("--chrom-col", default=None,
                    help="Optional chromosome column name in annotations; if absent, first BED chrom is used.")
    ap.add_argument("--bed-one-based", type=int, choices=(0, 1), default=0,
                    help="Set to 1 if BED start/end are 1-based inclusive.")
    ap.add_argument("--ann-one-based", type=int, choices=(0, 1), default=1,
                    help="Set to 1 if annotations start/end are 1-based inclusive.")
    ap.add_argument("--require-both-mapped", type=int, choices=(0, 1), default=0,
                    help="Drop rows unless both ends map to features (default keeps rows where at least one maps).")
    ap.add_argument("--drop-unmapped-label", default="Unmapped",
                    help="Label to use for unmapped ends if not dropped.")
    ap.add_argument("--fallback-intervaltree", type=int, choices=(0, 1), default=1,
                    help="If array mapping fails (memory), fall back to interval-tree (if installed).")
    ap.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    return ap.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_bed(path: str, one_based: bool) -> pd.DataFrame:
    # Read first three columns; tolerate extra columns
    df = pd.read_csv(path, sep=r"\s+|\t", header=None, engine="python")
    if df.shape[1] < 3:
        raise ValueError("BED file must have at least 3 columns: chrom start end")
    df = df.iloc[:, :3]
    df.columns = ["Chromosome", "Start", "End"]
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df.dropna(subset=["Start", "End"]).astype({"Start": int, "End": int})

    # Normalize to 0-based half-open internal convention
    if one_based:
        # Treat provided start/end as 1-based inclusive single-base coordinates for chimera ends
        df["Start"] = df["Start"] - 1
        df["End"] = df["End"] - 1
    return df


def load_annotations(path: str, cols: Tuple[str, str, str, str], chrom_col: Optional[str], one_based: bool) -> pd.DataFrame:
    rcol, scol, ecol, tcol = cols
    usecols = [rcol, scol, ecol, tcol] + ([chrom_col] if chrom_col else [])
    df = pd.read_csv(path, usecols=usecols)
    if chrom_col and chrom_col not in df.columns:
        raise ValueError(f"--chrom-col '{chrom_col}' not found in annotations.")
    for c in (scol, ecol):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[rcol, scol, ecol, tcol]).copy()
    df[[scol, ecol]] = df[[scol, ecol]].astype(int)
    if one_based:
        # convert to 0-based inclusive for internal use
        df[scol] = df[scol] - 1
        df[ecol] = df[ecol] - 1
    # sanity
    bad = (df[ecol] < df[scol]).sum()
    if bad:
        logging.warning("Found %d rows where end < start; swapping.", bad)
        swap_idx = df[ecol] < df[scol]
        tmp = df.loc[swap_idx, scol].copy()
        df.loc[swap_idx, scol] = df.loc[swap_idx, ecol]
        df.loc[swap_idx, ecol] = tmp
    return df.rename(columns={rcol: "range", scol: "startpos", ecol: "endpos", tcol: "Type"})


def build_array_mapper(ann: pd.DataFrame, genome_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
    # Determine genome length
    max_end = int(ann["endpos"].max())
    if genome_len is None:
        genome_len = max_end + 1
    mapper = np.empty(int(genome_len) + 1, dtype=object)  # indexable by 0-based position
    mapper[:] = None
    # Fill; later rows overwrite earlier ones in overlaps
    for _, row in ann.iterrows():
        s = int(row["startpos"])
        e = int(row["endpos"])
        # inclusive endpoints -> slice [s:e+1]
        if s <= e:
            mapper[s : e + 1] = row["range"]
    return mapper, int(genome_len)


def build_interval_tree(ann: pd.DataFrame) -> IntervalTree:
    if not _HAS_INTERVALTREE:
        raise RuntimeError("intervaltree not installed; cannot fall back.")
    it = IntervalTree()
    for _, row in ann.iterrows():
        # IntervalTree uses half-open intervals [start, end)
        it.add(Interval(int(row["startpos"]), int(row["endpos"]) + 1, row["range"]))
    it.merge_overlaps(strict=False)
    return it


def map_positions_to_ranges(
    chim: pd.DataFrame,
    ann: pd.DataFrame,
    array_mapper: Optional[np.ndarray],
    genome_len: Optional[int],
    itree: Optional[IntervalTree],
) -> pd.DataFrame:
    df = chim.copy()
    # clip within genome bounds if array-based
    if array_mapper is not None and genome_len is not None:
        df["Start"] = df["Start"].clip(0, genome_len)
        df["End"] = df["End"].clip(0, genome_len)
        df["Start_Range"] = array_mapper[df["Start"].values]
        df["End_Range"] = array_mapper[df["End"].values]
    else:
        # interval tree mapping: choose the first (or shortest) overlapping feature if multiple
        def _map_pos(p: int) -> Optional[str]:
            hits = itree.overlap(p, p + 1)
            if not hits:
                return None
            # choose smallest interval to reduce ambiguity
            iv = min(hits, key=lambda x: (x.end - x.start))
            return iv.data
        df["Start_Range"] = df["Start"].apply(_map_pos)
        df["End_Range"] = df["End"].apply(_map_pos)
    return df


def symmetric_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def main():
    args = parse_args()
    setup_logging(args.log)

    t0 = time.time()
    logging.info("Loading BED: %s", args.chimera_bed)
    chim = load_bed(args.chimera_bed, one_based=bool(args.bed_one_based))

    logging.info("Loading annotations: %s", args.annotations)
    ann = load_annotations(args.annotations, tuple(args.ann_cols), args.chrom_col, one_based=bool(args.ann_one_based))

    # Build mapper
    array_mapper, genome_len = None, None
    itree = None
    try:
        array_mapper, genome_len = build_array_mapper(ann)
        logging.info("Array mapper built (genome_len=%d).", genome_len)
    except MemoryError:
        logging.warning("Array mapper allocation failed; attempting interval-tree fallback.")
        if args.fallback_intervaltree and _HAS_INTERVALTREE:
            itree = build_interval_tree(ann)
            logging.info("Interval tree built (n=%d intervals).", len(itree))
        else:
            raise

    # Map positions
    logging.info("Mapping chimera ends to features…")
    mapped = map_positions_to_ranges(chim, ann, array_mapper, genome_len, itree)

    # Drop based on mapping policy
    before = mapped.shape[0]
    if args.require_both_mapped:
        mapped = mapped.dropna(subset=["Start_Range", "End_Range"], how="any")
    else:
        mapped = mapped.dropna(subset=["Start_Range", "End_Range"], how="all")
        mapped["Start_Range"] = mapped["Start_Range"].fillna(args.drop_unmapped_label)
        mapped["End_Range"] = mapped["End_Range"].fillna(args.drop_unmapped_label)
    logging.info("Kept %d/%d interactions after mapping policy.", mapped.shape[0], before)

    # Assign an interaction index (for io)
    mapped = mapped.reset_index(drop=True)
    mapped["Interaction_Index"] = mapped.index

    # Totals per feature (counts-based; include both ends)
    feat_series = pd.concat([
        mapped["Start_Range"].rename("Feature"),
        mapped["End_Range"].rename("Feature")
    ], ignore_index=True)
    feature_totals = feat_series.value_counts().rename_axis("Feature").reset_index(name="total_interactions")
    feature_totals_dict = dict(zip(feature_totals["Feature"], feature_totals["total_interactions"]))

    # Self-interactions
    self_df = mapped[mapped["Start_Range"] == mapped["End_Range"]]
    self_counts = self_df["Start_Range"].value_counts().rename_axis("Feature").reset_index(name="self_interactions")
    self_interactions_dict = dict(zip(self_counts["Feature"], self_counts["self_interactions"]))

    total_interactions_in_dataset = mapped.shape[0]

    # Symmetric pair counts (read-level counts)
    pairs = mapped.apply(lambda r: symmetric_pair(r["Start_Range"], r["End_Range"]), axis=1)
    pair_counts = pairs.value_counts().rename_axis(["sorted_ref", "sorted_target"]).reset_index(name="counts")

    # Unique coordinate pairs ("io") ignoring order (based on genomic coords)
    uniq_pairs = (
        mapped.assign(coord_pair=mapped.apply(lambda r: tuple(sorted([(r["Chromosome"], int(r["Start"])),
                                                                      (r["Chromosome"], int(r["End"]))])),
                                              axis=1),
                      feat_pair=mapped.apply(lambda r: symmetric_pair(r["Start_Range"], r["End_Range"]), axis=1))
        .drop_duplicates(subset=["coord_pair", "feat_pair"])
    )
    io_counts = uniq_pairs["feat_pair"].value_counts().rename_axis(["sorted_ref", "sorted_target"]).reset_index(name="io")

    # Merge counts and io
    counts = pd.merge(pair_counts, io_counts, on=["sorted_ref", "sorted_target"], how="left").fillna({"io": 0}).astype({"io": int})

    # Duplicate rows so both orders appear (ref,target) and (target,ref)
    rows = []
    for _, r in counts.iterrows():
        a, b, cts, io = r["sorted_ref"], r["sorted_target"], int(r["counts"]), int(r["io"])
        rows.append({"ref": a, "target": b, "counts": cts, "io": io})
        if a != b:
            rows.append({"ref": b, "target": a, "counts": cts, "io": io})
    interaction_counts_df = pd.DataFrame(rows)

    # Map totals and self-interactions
    interaction_counts_df["totals"] = interaction_counts_df["target"].map(feature_totals_dict)
    interaction_counts_df["total_ref"] = interaction_counts_df["ref"].map(feature_totals_dict)
    interaction_counts_df["self_interactions_ref"] = interaction_counts_df["ref"].map(self_interactions_dict).fillna(0).astype(int)
    interaction_counts_df["self_interactions_target"] = interaction_counts_df["target"].map(self_interactions_dict).fillna(0).astype(int)

    # Raw normalized score
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (interaction_counts_df["counts"] * float(total_interactions_in_dataset)) / (
            interaction_counts_df["totals"] * interaction_counts_df["total_ref"]
        )
    interaction_counts_df["score"] = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjust totals by removing self-interactions
    adj_tot = interaction_counts_df["totals"] - interaction_counts_df["self_interactions_target"]
    adj_ref = interaction_counts_df["total_ref"] - interaction_counts_df["self_interactions_ref"]
    adj_tot = adj_tot.mask(adj_tot <= 0)
    adj_ref = adj_ref.mask(adj_ref <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_score = (interaction_counts_df["counts"] * float(total_interactions_in_dataset)) / (adj_tot * adj_ref)
    interaction_counts_df["adjusted_score"] = np.nan_to_num(adj_score, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature types
    ftype = dict(zip(ann["range"], ann["Type"]))
    interaction_counts_df["ref_type"] = interaction_counts_df["ref"].map(ftype)
    interaction_counts_df["target_type"] = interaction_counts_df["target"].map(ftype)

    # Self-interaction score for self pairs
    self_rows = interaction_counts_df["ref"] == interaction_counts_df["target"]
    denom = (interaction_counts_df["totals"] - interaction_counts_df["self_interactions_ref"]).where(self_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        sis = interaction_counts_df["self_interactions_ref"].where(self_rows) / denom
    interaction_counts_df["self_interaction_score"] = np.nan_to_num(sis, nan=0.0, posinf=0.0, neginf=0.0)

    # Order columns
    cols = [
        "ref", "target", "counts", "io",
        "totals", "total_ref", "score", "adjusted_score",
        "ref_type", "target_type",
        "self_interactions_ref", "self_interactions_target", "self_interaction_score",
    ]
    interaction_counts_df = interaction_counts_df.loc[:, cols]

    # Sort for readability
    interaction_counts_df = interaction_counts_df.sort_values(
        by=["adjusted_score", "score", "counts", "io"], ascending=False
    ).reset_index(drop=True)

    # Save results
    out_csv = args.out
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    interaction_counts_df.to_csv(out_csv, index=False)
    logging.info("Wrote: %s  (n=%d)", out_csv, interaction_counts_df.shape[0])

    # Manifest
    manifest_path = args.manifest or (out_csv + ".manifest.json")
    manifest = {
        "script": os.path.basename(__file__),
        "version_info": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "intervaltree": "present" if _HAS_INTERVALTREE else "absent",
        },
        "inputs": {
            "chimera_bed": {"path": args.chimera_bed, "md5": md5sum(args.chimera_bed)},
            "annotations": {"path": args.annotations, "md5": md5sum(args.annotations)},
        },
        "outputs": {"pairwise_csv": out_csv},
        "params": {
            "ann_cols": list(args.ann_cols),
            "chrom_col": args.chrom_col,
            "bed_one_based": bool(args.bed_one_based),
            "ann_one_based": bool(args.ann_one_based),
            "require_both_mapped": bool(args.require_both_mapped),
            "drop_unmapped_label": args.drop_unmapped_label,
            "fallback_intervaltree": bool(args.fallback_intervaltree),
        },
        "counts": {
            "n_interactions_input": int(chim.shape[0]),
            "n_interactions_mapped": int(mapped.shape[0]),
            "n_pairs_out": int(interaction_counts_df.shape[0]),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logging.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
