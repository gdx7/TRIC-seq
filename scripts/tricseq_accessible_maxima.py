#!/usr/bin/env python3
"""
TRIC-seq — Accessible-region finder from LONG-RANGE intermolecular contacts (maxima only)

Purpose
-------
For a chosen RNA feature, tally ligation events from TRIC-seq chimeras where
one end falls inside the feature and the other end lies *outside* a ±FLANK bp
window (i.e., long-range > FLANK and *intermolecular* by distance). Smooth the
per-base profile, call local maxima, and export peak contexts. Optionally fold
the full sequence (ViennaRNA) and plot a 1D profile with peak markers.

Inputs
------
1) Annotations CSV with at least: gene name, start, end. Strand is optional.
   Configure column names via --ann-cols (default: RNA,Start,End,Strand).

2) Reference genome FASTA (single sequence or the relevant chromosome).

3) One or more contact files (BED or 2-col CSV) giving the two genomic
   coordinates per chimera row. For BED, columns 2 and 3 are used (0-based).
   For CSV, either name columns C1,C2 or the first two columns are used.

Outputs
-------
- <out_prefix>_features.csv   : table of peak maxima with base/context
- <out_prefix>_profile.png    : plot of raw/smoothed counts + peaks
- <out_prefix>.manifest.json  : provenance (params, versions, checksums)
- (stdout) sequence + dot-bracket MFE (if ViennaRNA installed)

Example
-------
python tricseq_accessible_maxima.py \
  --annotations annotations.csv \
  --fasta genome.fasta \
  --contacts data/*.bed data/*.csv \
  --gene CsrB \
  --out-prefix out/CsrB_ssrna \
  --flank 5000 --window 5 --prominence-factor 0.25 --peak-distance 3 --context 5
"""

from __future__ import annotations
import argparse, json, os, sys, glob, time, hashlib, logging
from typing import Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Bio import SeqIO
from Bio.Seq import Seq

# Optional ViennaRNA
try:
    import RNA  # type: ignore
    _HAS_VIENNA = True
except Exception:
    _HAS_VIENNA = False


# ------------------------- utils -------------------------

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
        prog="tricseq_accessible_maxima",
        description="Call accessible ssRNA maxima from long-range intermolecular TRIC-seq contacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--annotations", required=True, help="Annotations CSV.")
    ap.add_argument("--ann-cols", nargs="+",
                    default=["RNA","Start","End","Strand"],
                    help="Annotation column names (order: gene,start,end[,strand]).")
    ap.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    ap.add_argument("--contacts", nargs="+", required=True,
                    help="One or more contact files (glob OK), BED or 2-col CSV.")
    ap.add_argument("--gene", required=True, help="Gene/RNA name to analyze.")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (directory will be created).")

    # Parameters
    ap.add_argument("--flank", type=int, default=5000, help="Min distance (bp) considered long-range.")
    ap.add_argument("--window", type=int, default=5, help="Smoothing window (boxcar, nt).")
    ap.add_argument("--prominence-factor", type=float, default=0.25,
                    help="Peak prominence = factor * std(smoothed).")
    ap.add_argument("--peak-distance", type=int, default=3, help="Min distance between peaks (nt).")
    ap.add_argument("--context", type=int, default=5, help="Flank for sequence context around each maximum (nt).")

    ap.add_argument("--bed-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if BED positions are 1-based inclusive.")
    ap.add_argument("--csv-one-based", type=int, choices=(0,1), default=0,
                    help="Set 1 if CSV C1/C2 are 1-based.")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()


# ------------------------- I/O -------------------------

def load_annotations(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if len(cols) >= 1 and cols[0] in df.columns: rename[cols[0]] = "RNA"
    if len(cols) >= 2 and cols[1] in df.columns: rename[cols[1]] = "Start"
    if len(cols) >= 3 and cols[2] in df.columns: rename[cols[2]] = "End"
    if len(cols) >= 4 and cols[3] in df.columns: rename[cols[3]] = "Strand"
    df = df.rename(columns=rename)

    for c in ("RNA","Start","End"):
        if c not in df.columns:
            raise ValueError(f"Annotations must include '{c}' (configure via --ann-cols).")
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce").astype(int)
    df["End"]   = pd.to_numeric(df["End"],   errors="coerce").astype(int)
    if "Strand" not in df.columns:
        df["Strand"] = "+"

    # sanitize
    swap = df["End"] < df["Start"]
    if swap.any():
        tmp = df.loc[swap,"Start"].copy()
        df.loc[swap,"Start"] = df.loc[swap,"End"]
        df.loc[swap,"End"]   = tmp

    # collapse duplicates (min start, max end; first strand)
    df = df.groupby("RNA", as_index=False).agg(Start=("Start","min"), End=("End","max"), Strand=("Strand","first"))
    return df

def load_fasta_sequence(path: str) -> str:
    recs = list(SeqIO.parse(path, "fasta"))
    if not recs:
        raise ValueError("No records found in FASTA.")
    if len(recs) > 1:
        # Concatenate sequences in order (assumes shared coordinate system across contacts; adjust if needed)
        logging.warning("FASTA has multiple records; concatenating sequences in given order.")
        return "".join(str(r.seq) for r in recs)
    return str(recs[0].seq)

def expand_paths(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        out.extend(glob.glob(p) or ([p] if os.path.exists(p) else []))
    # de-dup keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def stream_contacts(paths: List[str], bed_one_based: bool, csv_one_based: bool,
                    chunksize: int = 1_000_000) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (C1, C2) integer arrays from each file chunk."""
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bed":
            with open(path, "r") as fh:
                first = fh.readline()
                skip = 1 if first.startswith(("track","browser")) else 0
            for chunk in pd.read_csv(path, sep=r"\s+|\t", header=None, engine="python",
                                     skiprows=skip, usecols=[1,2], chunksize=chunksize,
                                     dtype={1:int, 2:int}):
                c1 = chunk.iloc[:,0].to_numpy(np.int64, copy=False)
                c2 = chunk.iloc[:,1].to_numpy(np.int64, copy=False)
                if bed_one_based:
                    c1 -= 1; c2 -= 1
                yield c1, c2
        else:
            # CSV: prefer named columns C1,C2; else first two columns
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
                # Fallback: no header
                for chunk in pd.read_csv(path, header=None, chunksize=chunksize):
                    c1 = chunk.iloc[:,0].astype(np.int64).to_numpy()
                    c2 = chunk.iloc[:,1].astype(np.int64).to_numpy()
                    if csv_one_based:
                        c1 -= 1; c2 -= 1
                    yield c1, c2


# ------------------------- core logic -------------------------

def extract_gene(ann: pd.DataFrame, gene: str) -> Tuple[int,int,str,int,str]:
    row = ann.loc[ann["RNA"] == gene]
    if row.empty:
        raise ValueError(f"Gene '{gene}' not found in annotations.")
    start = int(row["Start"].iloc[0])
    end   = int(row["End"].iloc[0])
    strand = str(row["Strand"].iloc[0]) if "Strand" in row.columns else "+"
    length = end - start + 1
    if length <= 0:
        raise ValueError(f"Non-positive length for {gene}: start={start}, end={end}")
    return start, end, strand, length, gene

def subseq(genome: str, start_1based: int, end_1based: int, strand: str) -> str:
    s = genome[start_1based - 1 : end_1based]
    if strand == "-":
        return str(Seq(s).reverse_complement())
    return s

def count_long_range_profile(
    gene_start: int, gene_end: int, gene_strand: str, gene_len: int,
    contacts: Iterable[Tuple[np.ndarray,np.ndarray]],
    flank: int
) -> np.ndarray:
    prof = np.zeros(gene_len, dtype=np.int32)

    def in_gene(x: np.ndarray) -> np.ndarray:
        return (x >= gene_start) & (x <= gene_end)

    for c1, c2 in contacts:
        # ends inside gene?
        in1 = in_gene(c1); in2 = in_gene(c2)
        if not (in1.any() or in2.any()):
            continue

        # long-range: partner outside ±flank
        outside1 = (~((c2 >= (gene_start - flank)) & (c2 <= (gene_end + flank))))
        outside2 = (~((c1 >= (gene_start - flank)) & (c1 <= (gene_end + flank))))

        mask1 = in1 & outside1
        mask2 = in2 & outside2

        # map genomic coord -> gene index (0-based, 5'→3')
        if mask1.any():
            pos = c1[mask1]
            idx = (pos - gene_start) if gene_strand == "+" else (gene_end - pos)
            idx = idx[(idx >= 0) & (idx < gene_len)]
            np.add.at(prof, idx, 1)
        if mask2.any():
            pos = c2[mask2]
            idx = (pos - gene_start) if gene_strand == "+" else (gene_end - pos)
            idx = idx[(idx >= 0) & (idx < gene_len)]
            np.add.at(prof, idx, 1)

    return prof

def smooth_boxcar(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w == 1: return x.astype(float)
    kern = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kern, mode="same")

def call_maxima(smoothed: np.ndarray, peak_distance: int, prom_factor: float) -> np.ndarray:
    prom = float(np.std(smoothed)) * float(prom_factor)
    distance = max(1, int(peak_distance))
    peaks, _ = find_peaks(smoothed, distance=distance, prominence=prom)
    return peaks


# ------------------------- plotting -------------------------

def plot_profile(out_png: str, gene: str, prof: np.ndarray, smoothed: np.ndarray, peaks: np.ndarray) -> None:
    x = np.arange(1, len(prof) + 1)
    plt.figure(figsize=(12, 5))
    plt.bar(x, prof, width=1.0, color="lightgrey", label="Raw long-range ligations")
    plt.plot(x, smoothed, lw=2, label="Smoothed profile")
    if peaks.size:
        plt.scatter(x[peaks], smoothed[peaks], s=40, color="red", zorder=5, label="Maxima")
    plt.title(f"{gene} — long-range intermolecular profile (TRIC-seq)")
    plt.xlabel("Nucleotide position (5'→3')")
    plt.ylabel("Ligation events (smoothed)")
    plt.xlim(0, len(prof) + 1)
    # y-limit auto; keep clean grid
    plt.grid(axis="y", linestyle="--", alpha=0.2)
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ------------------------- main -------------------------

def main():
    args = parse_args()
    setup_logging(args.log)

    t0 = time.time()
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # Load inputs
    ann = load_annotations(args.annotations, args.ann_cols)
    genome = load_fasta_sequence(args.fasta)
    paths = expand_paths(args.contacts)
    if not paths:
        raise ValueError("No contact files found for given patterns.")

    g_start, g_end, g_strand, g_len, gene = extract_gene(ann, args.gene)
    logging.info("Gene %s: start=%d end=%d strand=%s len=%d", gene, g_start, g_end, g_strand, g_len)

    # Build profile by streaming contacts
    prof = count_long_range_profile(
        g_start, g_end, g_strand, g_len,
        contacts=stream_contacts(paths, bool(args.bed_one_based), bool(args.csv_one_based)),
        flank=int(args.flank),
    )
    total_lr = int(prof.sum())
    logging.info("Total long-range ligation events tallied: %d", total_lr)
    if total_lr == 0:
        logging.warning("No long-range events for %s; exiting.", gene)
        # still write an empty features CSV + manifest
        feats_path = f"{args.out_prefix}_features.csv"
        pd.DataFrame(columns=["Feature_Type","Gene_Pos_1based","Genomic_Pos","Central_Base","Sequence_Context"]).to_csv(feats_path, index=False)
        # manifest at end
        write_manifest(args, feats_path, f"{args.out_prefix}_profile.png", total_lr, time.time() - t0, ann_rows=len(ann), contacts_files=paths)
        return

    # Smooth & peaks
    sm = smooth_boxcar(prof, int(args.window))
    peaks = call_maxima(sm, int(args.peak_distance), float(args.prominence_factor))
    logging.info("Maxima called: %d", int(peaks.size))

    # Export features
    seq = subseq(genome, g_start, g_end, g_strand)
    ctx = int(args.context)
    rows = []
    for p in peaks:
        pos_1b = int(p + 1)
        if g_strand == "+":
            gpos = g_start + p
        else:
            gpos = g_end - p
        s = max(0, p - ctx)
        e = min(len(seq), p + ctx + 1)
        central = seq[p] if 0 <= p < len(seq) else "N"
        rows.append({
            "Feature_Type": "Maxima (ssRNA)",
            "Gene_Pos_1based": pos_1b,
            "Genomic_Pos": int(gpos),
            "Central_Base": central,
            "Sequence_Context": seq[s:e],
        })
    feats = pd.DataFrame(rows)
    feats_path = f"{args.out_prefix}_features.csv"
    feats.to_csv(feats_path, index=False)

    # Optional folding printout
    if _HAS_VIENNA:
        ss, mfe = RNA.fold(seq)
        print(f"\n> {gene}  (MFE: {mfe:.2f} kcal/mol)")
        print("Sequence:")
        print(seq)
        print("\nDot-Bracket:")
        print(ss)
    else:
        logging.info("ViennaRNA not available; skipping secondary structure output.")

    # Plot
    out_png = f"{args.out_prefix}_profile.png"
    plot_profile(out_png, gene, prof, sm, peaks)

    # Manifest
    write_manifest(args, feats_path, out_png, total_lr, time.time() - t0, ann_rows=len(ann), contacts_files=paths)
    logging.info("Done.")

def write_manifest(args, feats_csv, plot_png, total_lr, runtime, ann_rows, contacts_files):
    manifest = {
        "script": os.path.basename(__file__),
        "version_info": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "viennarna": ("present" if _HAS_VIENNA else "absent"),
        },
        "inputs": {
            "annotations": {"path": args.annotations, "md5": md5sum(args.annotations)},
            "fasta": {"path": args.fasta, "md5": md5sum(args.fasta)},
            "contacts": [{"path": p, "md5": md5sum(p)} for p in contacts_files[:100]],
        },
        "outputs": {"features_csv": feats_csv, "profile_png": plot_png},
        "params": {
            "gene": args.gene,
            "flank": int(args.flank),
            "window": int(args.window),
            "prominence_factor": float(args.prominence_factor),
            "peak_distance": int(args.peak_distance),
            "context": int(args.context),
            "bed_one_based": bool(args.bed_one_based),
            "csv_one_based": bool(args.csv_one_based),
            "ann_cols": args.ann_cols,
        },
        "stats": {
            "annotation_rows": int(ann_rows),
            "total_long_range_events": int(total_lr),
        },
        "runtime_sec": round(float(runtime), 3),
    }
    with open(f"{args.out_prefix}.manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
