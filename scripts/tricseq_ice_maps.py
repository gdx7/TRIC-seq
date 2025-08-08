#!/usr/bin/env python3
"""
TRIC-seq — Raw, Coverage, and ICE-normalized contact maps for selected RNAs

Reads chimera/contact files (BED or 2-column CSV), bins contacts within each
target RNA’s genomic span, and writes:
  - raw contact matrix
  - coverage-normalized matrix
  - ICE-normalized matrix
plus heatmap figures for each.

Inputs
------
1) Annotation CSV with at least: RNA, Start_bp, End_bp
   (column names configurable via --ann-cols)

2) One or more contact files (glob patterns allowed), each either:
   - BED: uses columns 2 and 3 (0-based start positions) as two chimera ends
   - CSV: first two columns (or named C1,C2) as chimera ends (0- or 1-based via flags)

Usage
-----
python tricseq_ice_maps.py \
  --annotations annotations.csv \
  --contacts data/*.bed data/*.csv \
  --genes clpB dnaX \
  --outdir out/maps \
  --bin-size 30 \
  --remove-diagonal 1 \
  --bed-one-based 0 \
  --csv-one-based 0

Notes
-----
- ICE here is the iterative correction procedure that balances row/column sums.
- If an RNA has no intra-region contacts after filtering, a note is logged and it is skipped.
"""

from __future__ import annotations
import argparse, os, sys, glob, json, logging, math
from typing import Iterable, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- CLI & logging -----------------------------

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_ice_maps",
        description="Build raw/coverage/ICE-normalized contact maps for selected RNAs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--annotations", required=True, help="Annotation CSV with RNA spans.")
    ap.add_argument("--ann-cols", nargs="+",
                    default=["RNA","Start_bp","End_bp"],
                    help="Columns in annotation CSV (order: RNA, Start, End).")
    ap.add_argument("--contacts", nargs="+", required=True,
                    help="Contact files (BED or 2-col CSV). Glob patterns allowed.")
    ap.add_argument("--genes", nargs="*", default=[],
                    help="Gene names to process (match by prefix or exact).")
    ap.add_argument("--genes-file", default=None,
                    help="Optional text file with one gene name per line.")
    ap.add_argument("--bin-size", type=int, default=30, help="Bin size in nucleotides.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--remove-diagonal", type=int, choices=(0,1), default=0,
                    help="Zero the diagonal before normalization.")
    ap.add_argument("--bed-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if BED coords are 1-based inclusive.")
    ap.add_argument("--csv-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if CSV coords are 1-based.")
    ap.add_argument("--fig-format", default="png", help="Figure format (png, pdf, svg).")
    ap.add_argument("--ice-max-iters", type=int, default=500, help="Max ICE iterations.")
    ap.add_argument("--ice-tol", type=float, default=1e-5, help="ICE convergence tolerance (L2 norm of bias change).")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()


# ----------------------------- I/O helpers -----------------------------

def load_annotations(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if len(cols) >= 1 and cols[0] in df.columns: rename[cols[0]] = "RNA"
    if len(cols) >= 2 and cols[1] in df.columns: rename[cols[1]] = "Start_bp"
    if len(cols) >= 3 and cols[2] in df.columns: rename[cols[2]] = "End_bp"
    df = df.rename(columns=rename)

    for c in ("RNA","Start_bp","End_bp"):
        if c not in df.columns:
            raise ValueError(f"Annotation CSV must include '{c}' (configure via --ann-cols).")

    df["Start_bp"] = pd.to_numeric(df["Start_bp"], errors="coerce").astype(int)
    df["End_bp"]   = pd.to_numeric(df["End_bp"],   errors="coerce").astype(int)
    swap = df["End_bp"] < df["Start_bp"]
    if swap.any():
        tmp = df.loc[swap,"Start_bp"].copy()
        df.loc[swap,"Start_bp"] = df.loc[swap,"End_bp"]
        df.loc[swap,"End_bp"]   = tmp

    # Keep unique RNA spans (min/max if duplicated)
    df = df.groupby("RNA", as_index=False).agg(Start_bp=("Start_bp","min"), End_bp=("End_bp","max"))
    return df

def expand_paths(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        expanded = glob.glob(p)
        if not expanded and os.path.exists(p):
            expanded = [p]
        out.extend(expanded)
    # dedupe while preserving order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def stream_contacts(paths: List[str], bed_one_based: bool, csv_one_based: bool, chunksize: int=2_000_000
                   ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (Coord1, Coord2) arrays from each file chunk."""
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bed":
            # skip UCSC headers if present
            try:
                with open(path, "r") as fh:
                    first = fh.readline()
                skip = 1 if first.startswith(("track","browser")) else 0
            except Exception:
                skip = 0
            for chunk in pd.read_csv(path, sep="\t", header=None, skiprows=skip,
                                     usecols=[1,2], chunksize=chunksize, dtype={1:int,2:int},
                                     on_bad_lines="skip"):
                c1 = chunk.iloc[:,0].to_numpy(np.int64, copy=False)
                c2 = chunk.iloc[:,1].to_numpy(np.int64, copy=False)
                if bed_one_based:
                    c1 -= 1; c2 -= 1
                yield c1, c2
        else:
            # CSV: prefer named C1,C2 if present
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


# ----------------------------- math helpers -----------------------------

def coverage_normalize(mat: np.ndarray) -> np.ndarray:
    total = float(mat.sum())
    if total <= 0:
        return mat.astype(float, copy=True)
    return mat.astype(float) / total

def ice_normalize(mat: np.ndarray, max_iters: int=500, tol: float=1e-5) -> np.ndarray:
    """
    Simple ICE (iterative correction) balancing to equalize row/col sums.
    Returns mat / (b_i * b_j), where b is the learned bias vector.
    Diagonal can be zeroed before calling this if desired.
    """
    m = mat.astype(float).copy()
    n = m.shape[0]
    # Avoid divide-by-zero; keep a mask of valid rows/cols
    valid = (m.sum(axis=0) > 0) & (m.sum(axis=1) > 0)
    if not np.any(valid):
        return m
    b = np.ones(n, dtype=float)
    b[~valid] = 1.0  # inert for invalid rows/cols

    for _ in range(max_iters):
        # current balanced matrix rowsums = sum_j m_ij / (b_i b_j) = (1/b_i) * sum_j m_ij / b_j
        denom = b.copy()
        denom[denom == 0] = 1.0
        inv_b = 1.0 / denom
        mb = m * inv_b  # divide columns by b_j
        rowsums = mb.sum(axis=1) * inv_b  # then divide by b_i
        target = rowsums[valid].mean() if np.any(valid) else 1.0
        target = target if target > 0 else 1.0
        scale = np.ones_like(rowsums)
        scale[valid] = rowsums[valid] / target
        new_b = b * scale
        # normalize biases to mean 1 over valid entries to stabilize
        mean_b = new_b[valid].mean()
        if mean_b <= 0 or not np.isfinite(mean_b):
            break
        new_b[valid] /= mean_b
        delta = np.linalg.norm(new_b - b)
        b = new_b
        if delta < tol:
            break

    # Final corrected matrix
    denom = np.outer(b, b)
    denom[denom == 0] = 1.0
    return m / denom


# ----------------------------- core logic -----------------------------

def build_matrix_for_gene(
    contacts: Iterable[Tuple[np.ndarray, np.ndarray]],
    start: int,
    end: int,
    bin_size: int,
    remove_diag: bool,
) -> Tuple[np.ndarray, int]:
    """
    Build a symmetric contact matrix for contacts whose both ends fall inside [start, end] (inclusive).
    Returns (matrix, n_bins)
    """
    length = int(end - start + 1)
    n_bins = int(math.ceil(length / float(bin_size)))
    mat = np.zeros((n_bins, n_bins), dtype=np.int64)

    s = int(start)
    e = int(end)

    for c1, c2 in contacts:
        # mask both in region
        mask = (c1 >= s) & (c1 <= e) & (c2 >= s) & (c2 <= e)
        if not np.any(mask):
            continue
        b1 = ((c1[mask] - s) // bin_size).astype(np.int64)
        b2 = ((c2[mask] - s) // bin_size).astype(np.int64)
        # accumulate symmetric counts
        for i, j in zip(b1, b2):
            mat[i, j] += 1
            if i != j:
                mat[j, i] += 1

    if remove_diag:
        np.fill_diagonal(mat, 0)
    return mat, n_bins

def save_matrix(path_csv: str, mat: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
    pd.DataFrame(mat).to_csv(path_csv, index=False)

def plot_heatmap(path_fig: str, mat: np.ndarray, title: str) -> None:
    os.makedirs(os.path.dirname(path_fig) or ".", exist_ok=True)
    data = mat.astype(float).copy()
    data[data == 0] = np.nan
    # robust min/max
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanpercentile(finite_vals, 5)
        vmax = np.nanpercentile(finite_vals, 95)
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = np.nanmax(finite_vals)
        if vmax <= vmin:
            vmax = vmin + 1.0
    plt.figure(figsize=(6, 5))
    plt.imshow(data, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="Reds")
    plt.colorbar(shrink=0.8, label="Contact weight")
    plt.title(title)
    plt.xlabel("Bin")
    plt.ylabel("Bin")
    plt.tight_layout()
    plt.savefig(path_fig, dpi=300)
    plt.close()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    setup_logging(args.log)
    os.makedirs(args.outdir, exist_ok=True)

    # Load annotations
    ann = load_annotations(args.annotations, args.ann_cols)

    # Resolve genes input
    genes = list(args.genes or [])
    if args.genes_file:
        with open(args.genes_file, "r") as fh:
            genes += [ln.strip() for ln in fh if ln.strip()]
    genes = [g for g in genes if g]
    if not genes:
        logging.error("No genes provided. Use --genes and/or --genes-file.")
        sys.exit(2)

    # Expand and stream contacts
    paths = expand_paths(args.contacts)
    if not paths:
        logging.error("No contact files found for given patterns.")
        sys.exit(2)

    # For streaming multiple times (per gene), it is simpler to read once into memory.
    # If files are huge, consider re-streaming here per gene.
    coord_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    total_rows = 0
    for c1, c2 in stream_contacts(paths, bed_one_based=bool(args.bed_one_based), csv_one_based=bool(args.csv_one_based)):
        coord_pairs.append((c1, c2))
        total_rows += c1.size
    logging.info("Loaded %d total chimera records across %d chunks.", total_rows, len(coord_pairs))

    # Process each requested gene
    for gene in genes:
        # find RNA by prefix or exact match (prefer exact)
        row = ann[ann["RNA"] == gene]
        if row.empty:
            candidates = ann[ann["RNA"].astype(str).str.startswith(gene)]
            if candidates.empty:
                logging.warning("Gene '%s' not found in annotations. Skipping.", gene)
                continue
            row = candidates.iloc[[0]]
            logging.info("Using annotation match '%s' for requested gene '%s'.", row["RNA"].iloc[0], gene)

        start = int(row["Start_bp"].iloc[0])
        end   = int(row["End_bp"].iloc[0])
        length = end - start + 1
        n_bins = int(math.ceil(length / float(args.bin_size)))
        logging.info("Gene %s: %d..%d (len=%d bp) -> %d bins @ %d nt.", row["RNA"].iloc[0], start, end, length, n_bins, args.bin_size)

        # Build raw matrix (restrict both ends to region)
        raw, _ = build_matrix_for_gene(
            contacts=coord_pairs,
            start=start,
            end=end,
            bin_size=int(args.bin_size),
            remove_diag=bool(args.remove_diagonal),
        )

        if raw.sum() == 0:
            logging.warning("No intra-region contacts for %s. Skipping.", gene)
            continue

        cov = coverage_normalize(raw)
        ice = ice_normalize(raw if not args.remove_diagonal else raw.copy(), max_iters=int(args.ice_max_iters), tol=float(args.ice_tol))

        # Save matrices
        base = os.path.join(args.outdir, f"{gene}_bin{int(args.bin_size)}")
        save_matrix(base + "_raw.csv", raw)
        save_matrix(base + "_cov.csv", cov)
        save_matrix(base + "_ice.csv", ice)

        # Save plots
        plot_heatmap(base + "_raw." + args.fig_format, raw, f"{gene} — raw")
        plot_heatmap(base + "_cov." + args.fig_format, cov, f"{gene} — coverage-normalized")
        plot_heatmap(base + "_ice." + args.fig_format, ice, f"{gene} — ICE-normalized")

        # Manifest
        manifest = {
            "gene": gene,
            "span": {"start_bp": start, "end_bp": end, "length_bp": length},
            "bin_size": int(args.bin_size),
            "remove_diagonal": bool(args.remove_diagonal),
            "matrices": {
                "raw_csv": base + "_raw.csv",
                "cov_csv": base + "_cov.csv",
                "ice_csv": base + "_ice.csv",
            },
            "figures": {
                "raw": base + f"_raw.{args.fig_format}",
                "cov": base + f"_cov.{args.fig_format}",
                "ice": base + f"_ice.{args.fig_format}",
            },
            "params": {
                "ice_max_iters": int(args.ice_max_iters),
                "ice_tol": float(args.ice_tol),
                "bed_one_based": bool(args.bed_one_based),
                "csv_one_based": bool(args.csv_one_based),
            }
        }
        with open(base + "_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

        logging.info("Wrote outputs for %s under: %s*", gene, base)

    logging.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
