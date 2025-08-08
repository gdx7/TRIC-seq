#!/usr/bin/env python3
"""
Genome-wide Gini of interaction positions across CDSs

Computes, for each CDS feature, the inequality (Gini coefficient) of where
cross-flank interaction positions fall along the CDS. A "cross-flank" event
is counted when exactly one chimera end lands inside the CDS and the other end
lies outside the CDS ±FLANK bp window. Counts are tallied per base and Gini is
computed from that distribution.

Inputs
------
1) Annotations CSV with columns (configurable via --ann-cols):
      RNA, Start_bp, End_bp, Feature[, Strand]
   Genomic coordinates should be 1-based inclusive if --ann-one-based=1.

2) One or more contact files (BED or 2-col CSV) providing genomic positions
   of ligation ends. For BED, the script uses columns 2 (start) and 3 (end) as
   the two chimera ends (0-based by default). For CSV, use two columns C1,C2
   (or the first two columns).

Outputs
-------
- CSV with per-CDS rows:
    RNA, Start_bp, End_bp, Strand, Length_bp, N_interactions, Gini
- Optional PNG histograms if --plot-prefix is provided.
- A JSON manifest with provenance (<out>.manifest.json).

Example
-------
python tricseq_gini_by_cds.py \
  --annotations annotations.csv \
  --contacts path/to/*.bed path/to/*.csv \
  --out gini_by_cds.csv \
  --feature-class CDS \
  --flank 5000 \
  --ann-one-based 1 \
  --bed-one-based 0 \
  --csv-one-based 0
"""

from __future__ import annotations
import argparse, os, sys, json, time, hashlib, glob, logging
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------- utils --------------------

def md5sum(path: str) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_gini_by_cds",
        description="Compute per-CDS Gini of cross-flank interaction positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--annotations", required=True, help="Annotations CSV.")
    ap.add_argument("--ann-cols", nargs="+",
                    default=["RNA","Start_bp","End_bp","Feature","Strand"],
                    help="Column names in annotations CSV (order: RNA Start End Feature [Strand]).")
    ap.add_argument("--feature-class", default="CDS", help="Feature class to analyze (e.g., CDS).")
    ap.add_argument("--contacts", nargs="+", required=True,
                    help="One or more contact files (glob patterns ok). BED or 2-col CSV supported.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--plot-prefix", default=None, help="If set, write Gini hist PNGs with this prefix.")
    ap.add_argument("--flank", type=int, default=5000, help="Flank (bp) excluded around CDS when counting.")
    ap.add_argument("--ann-one-based", type=int, choices=(0,1), default=1,
                    help="Set 1 if annotation Start/End are 1-based inclusive.")
    ap.add_argument("--bed-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if BED start/end are 1-based inclusive.")
    ap.add_argument("--csv-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if CSV C1/C2 positions are 1-based.")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per chunk when streaming contacts.")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()


# -------------------- I/O helpers --------------------

def load_annotations(path: str, cols: List[str], feature_class: str, one_based: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    # rename to canonical
    rename = {cols[0]:"RNA", cols[1]:"Start_bp", cols[2]:"End_bp"}
    if len(cols) >= 4 and cols[3] in df.columns: rename[cols[3]] = "Feature"
    if len(cols) >= 5 and cols[4] in df.columns: rename[cols[4]] = "Strand"
    df = df.rename(columns=rename)
    if "Feature" not in df.columns:
        raise ValueError("Annotations must include a 'Feature' column (provide via --ann-cols).")

    df = df[df["Feature"] == feature_class].copy()
    if df.empty:
        raise ValueError(f"No rows with Feature == '{feature_class}' in annotations.")

    for c in ("Start_bp","End_bp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["RNA","Start_bp","End_bp"])
    df[["Start_bp","End_bp"]] = df[["Start_bp","End_bp"]].astype(int)

    if one_based:
        df["Start_bp"] -= 1
        df["End_bp"]   -= 1

    swap = df["End_bp"] < df["Start_bp"]
    if swap.any():
        tmp = df.loc[swap, "Start_bp"].copy()
        df.loc[swap, "Start_bp"] = df.loc[swap, "End_bp"]
        df.loc[swap, "End_bp"]   = tmp

    # collapse duplicates to min/max span per RNA
    agg = {"Start_bp":"min","End_bp":"max","Feature":"first"}
    if "Strand" in df.columns: agg["Strand"] = "first"
    df = df.groupby("RNA", as_index=False).agg(**{k:(k,v) for k,v in agg.items()})
    df["Length_bp"] = (df["End_bp"] - df["Start_bp"] + 1).astype(int)
    df = df.sort_values(["Start_bp","End_bp"]).reset_index(drop=True)
    return df

def expand_paths(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        expanded = glob.glob(p)
        if not expanded and os.path.exists(p):
            expanded = [p]
        out.extend(expanded)
    # de-dup keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def stream_contacts(paths: list[str], chunksize: int,
                    bed_one_based: bool, csv_one_based: bool) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (C1, C2) arrays per chunk from BED or 2-col CSV files."""
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bed":
            # skip UCSC headers if present
            with open(path, "r") as fh:
                first = fh.readline()
                skip = 1 if first.startswith(("track","browser")) else 0
            for chunk in pd.read_csv(path, sep="\t", header=None, skiprows=skip,
                                     usecols=[1,2], chunksize=chunksize, dtype={1:int,2:int},
                                     on_bad_lines="skip"):
                c1 = chunk.iloc[:,0].to_numpy(np.int64, copy=False)
                c2 = chunk.iloc[:,1].to_numpy(np.int64, copy=False)
                if bed_one_based:
                    c1 -= 1; c2 -= 1
                yield c1, c2
        else:
            # CSV: try named C1,C2; else first two cols
            try:
                for chunk in pd.read_csv(path, chunksize=chunksize):
                    if {"C1","C2"}.issubset(chunk.columns):
                        c1 = chunk["C1"].astype(np.int64).to_numpy()
                        c2 = chunk["C2"].astype(np.int64).to_numpy()
                    else:
                        c1 = chunk.iloc[:,0].astype(np.int64).to_numpy()
                        c2 = chunk.iloc[:,1].astype(np.int64).to_numpy()
                    if csv_one_based:
                        c1 -= 1; c2 -= 1
                    yield c1, c2
            except Exception:
                for chunk in pd.read_csv(path, header=None, chunksize=chunksize):
                    c1 = chunk.iloc[:,0].astype(np.int64).to_numpy()
                    c2 = chunk.iloc[:,1].astype(np.int64).to_numpy()
                    if csv_one_based:
                        c1 -= 1; c2 -= 1
                    yield c1, c2


# -------------------- core logic --------------------

def build_position_maps(genes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    genome_len = int(genes["End_bp"].max()) + 1
    n = len(genes)
    gene_idx_at_pos = np.full(genome_len + 1, -1, dtype=np.int32)
    offset_at_pos   = np.full(genome_len + 1, -1, dtype=np.int32)

    lengths = genes["Length_bp"].to_numpy(np.int32)
    base_offset_by_gene = np.zeros(n, dtype=np.int64)
    if n > 0:
        base_offset_by_gene[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    starts = genes["Start_bp"].to_numpy(np.int64)
    ends   = genes["End_bp"].to_numpy(np.int64)
    for i in range(n):
        s = int(starts[i]); e = int(ends[i])
        if s <= e:
            gene_idx_at_pos[s:e+1] = i
            offset_at_pos[s:e+1] = np.arange(0, e - s + 1, dtype=np.int32)
    return gene_idx_at_pos, offset_at_pos, base_offset_by_gene, genome_len

def scatter_counts_for_chunk(
    c1: np.ndarray, c2: np.ndarray, flank: int,
    gene_idx_at_pos: np.ndarray, offset_at_pos: np.ndarray,
    starts: np.ndarray, ends: np.ndarray,
    base_offset_by_gene: np.ndarray,
    flat_counts: np.ndarray
) -> None:
    # end1 inside gene?
    g1 = gene_idx_at_pos[c1]
    mask1 = g1 != -1
    if mask1.any():
        g1v = g1[mask1]
        s1 = starts[g1v]
        e1 = ends[g1v]
        other = c2[mask1]
        outside = (other < (s1 - flank)) | (other > (e1 + flank))
        if outside.any():
            pos = offset_at_pos[c1[mask1][outside]]
            gi = g1v[outside]
            glob = base_offset_by_gene[gi] + pos
            np.add.at(flat_counts, glob, 1)

    # end2 inside gene?
    g2 = gene_idx_at_pos[c2]
    mask2 = g2 != -1
    if mask2.any():
        g2v = g2[mask2]
        s2 = starts[g2v]
        e2 = ends[g2v]
        other = c1[mask2]
        outside = (other < (s2 - flank)) | (other > (e2 + flank))
        if outside.any():
            pos = offset_at_pos[c2[mask2][outside]]
            gi = g2v[outside]
            glob = base_offset_by_gene[gi] + pos
            np.add.at(flat_counts, glob, 1)

def gini_from_counts(x: np.ndarray) -> float:
    s = x.sum()
    if s <= 0:
        return 0.0
    xv = np.sort(x.astype(np.float64), kind="mergesort")
    n = xv.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    g = (n + 1.0 - 2.0 * np.sum((n + 1.0 - ranks) * xv) / s) / n
    return float(max(0.0, min(1.0, g)))


# -------------------- main --------------------

def main():
    args = parse_args()
    setup_logging(args.log)
    t0 = time.time()

    genes = load_annotations(args.annotations, args.ann_cols, args.feature_class, one_based=bool(args.ann_one_based))
    n_genes = len(genes)
    logging.info("Loaded %d %s features.", n_genes, args.feature_class)

    gene_idx_at_pos, offset_at_pos, base_offset_by_gene, genome_len = build_position_maps(genes)
    logging.info("Genome length inferred: %d bp", genome_len)

    total_bases = int(base_offset_by_gene[-1] + genes["Length_bp"].iloc[-1]) if n_genes > 0 else 0
    flat_counts = np.zeros(total_bases, dtype=np.int32)

    starts = genes["Start_bp"].to_numpy(np.int64)
    ends   = genes["End_bp"].to_numpy(np.int64)

    contact_paths = expand_paths(args.contacts)
    if not contact_paths:
        raise ValueError("No contact files found for the given patterns.")
    logging.info("Processing %d contact file(s).", len(contact_paths))

    total_read = 0
    for c1, c2 in stream_contacts(contact_paths, args.chunksize,
                                  bed_one_based=bool(args.bed_one_based),
                                  csv_one_based=bool(args.csv_one_based)):
        total_read += c1.size
        scatter_counts_for_chunk(c1, c2, int(args.flank),
                                 gene_idx_at_pos, offset_at_pos,
                                 starts, ends, base_offset_by_gene,
                                 flat_counts)
    logging.info("Processed %d contacts.", total_read)

    lengths = genes["Length_bp"].to_numpy(np.int64)
    starts_flat = base_offset_by_gene
    ends_flat = starts_flat + lengths

    N_interactions = np.empty(n_genes, dtype=np.int64)
    Gini = np.empty(n_genes, dtype=np.float64)

    for i in range(n_genes):
        seg = flat_counts[starts_flat[i]:ends_flat[i]]
        N_interactions[i] = int(seg.sum())
        Gini[i] = gini_from_counts(seg)

    out_df = genes[["RNA","Start_bp","End_bp"] + (["Strand"] if "Strand" in genes.columns else [])].copy()
    out_df["Length_bp"] = lengths
    out_df["N_interactions"] = N_interactions
    out_df["Gini"] = Gini

    out_csv = args.out
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logging.info("Wrote: %s (rows=%d)", out_csv, out_df.shape[0])

    if args.plot_prefix:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4), dpi=150)
            plt.hist(out_df["Gini"].to_numpy(), bins=40)
            plt.xlabel("Gini (per-base interaction inequality)")
            plt.ylabel("Number of CDS")
            plt.title("Gini across CDS")
            plt.tight_layout()
            plt.savefig(f"{args.plot_prefix}_gini_all.png"); plt.close()

            mask = out_df["N_interactions"] >= 200
            if mask.any():
                plt.figure(figsize=(6,4), dpi=150)
                plt.hist(out_df.loc[mask, "Gini"].to_numpy(), bins=40)
                plt.xlabel("Gini (per-base interaction inequality)")
                plt.ylabel("Number of CDS (N_interactions ≥ 200)")
                plt.title("Gini across CDS (≥200 interactions)")
                plt.tight_layout()
                plt.savefig(f"{args.plot_prefix}_gini_ge200.png"); plt.close()
        except Exception as e:
            logging.warning("Plotting failed: %s", e)

    manifest = {
        "script": os.path.basename(__file__),
        "version_info": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {
            "annotations": {"path": args.annotations, "md5": md5sum(args.annotations)},
            "contacts": [{"path": p, "md5": md5sum(p)} for p in contact_paths[:50]],
        },
        "outputs": {"csv": out_csv, "plots_prefix": args.plot_prefix},
        "params": {
            "feature_class": args.feature_class,
            "flank": int(args.flank),
            "ann_one_based": bool(args.ann_one_based),
            "bed_one_based": bool(args.bed_one_based),
            "csv_one_based": bool(args.csv_one_based),
            "chunksize": int(args.chunksize),
        },
        "stats": {
            "n_genes": int(n_genes),
            "genome_len": int(genome_len),
            "total_bases_in_genes": int(total_bases),
            "total_contacts_streamed": int(total_read),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    with open(out_csv + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
