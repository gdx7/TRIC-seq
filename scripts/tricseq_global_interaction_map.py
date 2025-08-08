#!/usr/bin/env python3
"""
TRIC-seq — Global interaction map (gene-centric scatter)

Build a genome-wide scatter plot of interaction partners for a given RNA.
Each point is an interacting partner positioned by its genomic coordinate
(x = partner start), sized by interaction support (counts), and colored by
the partner's feature type. The y-axis shows an interaction strength
(e.g., odds_ratio or adjusted_score) with optional caps and symlog scaling.

Two pages are saved into a single PDF:
  1) Labeled plot (auto-labels high-scoring, well-separated partners)
  2) Unlabeled plot (clean version for figures)

Also saves a plain-text list of labeled partner names.

Inputs
------
1) Pair table CSV with at least: ref, target, counts
   And one of the following weight columns (choose via --weight-col):
     - odds_ratio (recommended if available from configuration-model step)
     - adjusted_score (fallback)
     - score (fallback)

   If available, the script will also use:
     - totals / total_ref  (to print a per-gene interaction load)
     - ref_type / target_type (ignored if --feature-col in annotations is provided)

2) Annotation CSV mapping gene names to coordinates (and optionally feature type).
   Provide column names via --ann-cols (default: gene_name,start,end,feature_type,strand,chromosome).

Usage
-----
python tricseq_global_interaction_map.py \
  --pairs pairs_with_mc.csv \
  --annotations annotations.csv \
  --gene GcvB \
  --out-prefix out/GcvB_interactome \
  --weight-col odds_ratio \
  --min-counts 5 \
  --min-distance 5000 \
  --y-cap 5000 \
  --annotate-threshold 5 \
  --exclude-feature-types rRNA tRNA \
  --highlight-basenames basenames.txt

Notes
-----
- Feature type coloring prefers the annotation's feature column if available;
  otherwise falls back to ref_type/target_type columns in the pairs CSV.
- For multi-chromosome genomes, x uses the partner gene's 'start' coordinate as-is.
- “Basenames” let you highlight the best-scoring partner among sets that share
  prefixes like "5'" / "3'" (e.g., treat "5'oppA" and "oppA" as the same base).
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------- CLI & logging -----------------------------

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_global_interaction_map",
        description="Gene-centric global interaction scatter for TRIC-seq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs", required=True, help="Pairs CSV (ref,target,counts,weight columns, …).")
    ap.add_argument("--annotations", required=True, help="Annotations CSV (gene_name,start,end[,feature_type…]).")
    ap.add_argument("--ann-cols", nargs="+",
                    default=["gene_name","start","end","feature_type","strand","chromosome"],
                    help="Column names in annotations CSV (order: gene, start, end, [feature_type] [strand] [chromosome]).")
    ap.add_argument("--gene", required=True, help="Gene name to center the map on (must match 'ref'/'target').")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for PDF and label list.")
    ap.add_argument("--weight-col", default="odds_ratio",
                    help="Edge weight column to plot on y-axis (e.g., odds_ratio, adjusted_score, score).")
    ap.add_argument("--min-counts", type=int, default=5, help="Minimum read-level counts per pair to include.")
    ap.add_argument("--min-distance", type=int, default=5000, help="Minimum genomic distance (bp) from the focal gene.")
    ap.add_argument("--y-cap", type=float, default=5000.0, help="Cap for y-axis values.")
    ap.add_argument("--symlog-linthresh", type=float, default=10.0, help="Symlog linthresh for y-axis.")
    ap.add_argument("--annotate-threshold", type=float, default=5.0, help="Annotate partners with y >= this value.")
    ap.add_argument("--annotate-x-sep", type=int, default=2000, help="Minimum x-separation (bp) to avoid label crowding.")
    ap.add_argument("--exclude-feature-types", nargs="*", default=["rRNA","tRNA"],
                    help="Feature types to exclude from plotting (if available).")
    ap.add_argument("--highlight-basenames", default=None,
                    help="Optional text file with one basename per line to highlight best-scoring partner (facecolor fill).")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()

# ----------------------------- I/O helpers -----------------------------

def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ref","target","counts"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Pairs CSV missing required columns: {missing}")
    return df

def load_annotations(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if len(cols) >= 1 and cols[0] in df.columns: rename[cols[0]] = "gene_name"
    if len(cols) >= 2 and cols[1] in df.columns: rename[cols[1]] = "start"
    if len(cols) >= 3 and cols[2] in df.columns: rename[cols[2]] = "end"
    if len(cols) >= 4 and cols[3] in df.columns: rename[cols[3]] = "feature_type"
    if len(cols) >= 5 and cols[4] in df.columns: rename[cols[4]] = "strand"
    if len(cols) >= 6 and cols[5] in df.columns: rename[cols[5]] = "chromosome"
    df = df.rename(columns=rename)

    for c in ("gene_name","start","end"):
        if c not in df.columns:
            raise ValueError(f"Annotations must include '{c}' (configure via --ann-cols).")

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    df = df.dropna(subset=["gene_name","start","end"])
    df[["start","end"]] = df[["start","end"]].astype(int)

    swap = df["end"] < df["start"]
    if swap.any():
        tmp = df.loc[swap,"start"].copy()
        df.loc[swap,"start"] = df.loc[swap,"end"]
        df.loc[swap,"end"]   = tmp

    # collapse duplicate gene rows to min/max span
    agg = {"start":"min","end":"max"}
    if "feature_type" in df.columns: agg["feature_type"] = "first"
    if "strand" in df.columns:       agg["strand"] = "first"
    if "chromosome" in df.columns:   agg["chromosome"] = "first"
    df = df.groupby("gene_name", as_index=False).agg(agg)
    return df

def load_basenames(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    names = set()
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if s:
                names.add(s)
    return names or None

# ----------------------------- core helpers -----------------------------

def gene_coords(ann: pd.DataFrame) -> Dict[str,Tuple[int,int]]:
    return dict(zip(ann["gene_name"], zip(ann["start"], ann["end"])))

def gene_ftypes(ann: pd.DataFrame) -> Dict[str,str]:
    return dict(zip(ann["gene_name"], ann["feature_type"])) if "feature_type" in ann.columns else {}

def interval_distance(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    s1,e1 = a; s2,e2 = b
    if e1 < s2: return s2 - e1
    if e2 < s1: return s1 - e2
    return 0  # overlap/adjacent

def basename(name: str) -> str:
    return name[2:] if name.startswith(("5'","3'")) else name

def total_for_gene(pairs: pd.DataFrame, gene: str) -> int:
    # Prefer per-gene totals if present; else sum read-level counts over incident edges.
    if "totals" in pairs.columns and "total_ref" in pairs.columns:
        # If gene appears in 'ref', show total_ref; if in 'target', show totals.
        row_ref = pairs.loc[pairs["ref"] == gene]
        row_tar = pairs.loc[pairs["target"] == gene]
        vals = []
        if not row_ref.empty and "total_ref" in row_ref:
            vals.append(int(pd.to_numeric(row_ref["total_ref"], errors="coerce").max()))
        if not row_tar.empty and "totals" in row_tar:
            vals.append(int(pd.to_numeric(row_tar["totals"], errors="coerce").max()))
        if vals:
            return max(vals)
    # fallback
    return int(pairs.loc[(pairs["ref"] == gene) | (pairs["target"] == gene), "counts"].sum())

# ----------------------------- plotting -----------------------------

FEATURE_EDGE_COLORS = {
    "ncRNA":  "#A40194",  # magenta
    "sponge": "#F12C2C",  # red
    "tRNA":   "#82F778",  # light green
    "hkRNA":  "#C4C5C5",  # grey
    "CDS":    "#F78208",  # orange
    "5'UTR":  "#76AAD7",  # blue
    "3'UTR":  "#0C0C0C",  # black
}

def feature_color(ftype: Optional[str]) -> str:
    if pd.isna(ftype) or ftype is None:
        return "orange"
    return FEATURE_EDGE_COLORS.get(str(ftype), "orange")

def build_partner_table(
    pairs: pd.DataFrame,
    ann: pd.DataFrame,
    gene: str,
    weight_col: str,
    min_counts: int,
    min_distance: int,
    exclude_ftypes: List[str],
) -> pd.DataFrame:
    coords = gene_coords(ann)
    ftypes = gene_ftypes(ann)

    if gene not in coords:
        raise ValueError(f"Gene '{gene}' not found in annotations.")

    # keep rows involving the gene (either side)
    df = pairs[(pairs["ref"] == gene) | (pairs["target"] == gene)].copy()
    if df.empty:
        return df

    # drop missing weights / counts
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in pairs CSV.")
    df = df.dropna(subset=[weight_col, "counts"])
    df = df[df["counts"] >= int(min_counts)]
    df = df[df[weight_col] > 0]

    # attach partner column and partner feature type
    df["partner"] = np.where(df["ref"] == gene, df["target"], df["ref"])

    # attach feature types (prefer annotation feature_type; otherwise fallback to existing columns)
    if "feature_type" in ann.columns:
        df["partner_type"] = df["partner"].map(ftypes)
    else:
        # fallback to ref_type/target_type if present
        if {"ref_type","target_type"}.issubset(df.columns):
            df["partner_type"] = np.where(df["ref"] == gene, df["target_type"], df["ref_type"])
        else:
            df["partner_type"] = np.nan

    # exclude requested feature types
    if exclude_ftypes:
        df = df[~df["partner_type"].isin(exclude_ftypes)]

    # distance filter (interval distance)
    gspan = coords[gene]
    dists = df["partner"].map(lambda p: interval_distance(gspan, coords.get(p, gspan)))
    df = df.assign(distance_bp=dists)
    df = df.dropna(subset=["distance_bp"])
    df = df[df["distance_bp"] >= int(min_distance)]

    # add plot x (partner start) and y (weight)
    df["x"] = df["partner"].map(lambda p: coords.get(p, (np.nan, np.nan))[0])
    df = df.dropna(subset=["x"])
    df["y"] = df[weight_col].astype(float)

    return df

def autolabel(ax, x, y, names, annotate_threshold: float, x_sep: int) -> List[str]:
    """Annotate points with y >= threshold if not within x_sep of a higher/equal neighbor."""
    labeled = []
    idx_sorted = np.argsort(-np.array(y))  # descending by y
    placed = []

    for idx in idx_sorted:
        xi, yi, name = x[idx], y[idx], names[idx]
        if yi < annotate_threshold:
            break
        # suppress if too close in x to an already placed higher/equal point
        too_close = any(abs(xi - x[j]) < x_sep and y[j] >= yi for j in placed)
        if too_close:
            continue
        ax.annotate(
            name, xy=(xi, yi), xycoords="data",
            xytext=(2, 5), textcoords="offset points",
            fontsize=14, rotation=60, alpha=0.8
        )
        ax.plot([xi, xi], [0, yi], ":", linewidth=1.0, alpha=0.15, color="grey")
        labeled.append(name)
        placed.append(idx)
    return labeled

def make_plots(
    df: pd.DataFrame,
    ann: pd.DataFrame,
    gene: str,
    out_prefix: str,
    y_cap: float,
    symlog_linthresh: float,
    annotate_threshold: float,
    annotate_x_sep: int,
    basenames_highlight: Optional[set],
):
    coords = gene_coords(ann)
    gstart, gend = coords[gene]
    genome_max = int(ann["end"].max())

    # Build arrays for plotting
    x = df["x"].to_numpy(dtype=float)
    y = np.minimum(df["y"].to_numpy(dtype=float), float(y_cap))
    sizes = (df["counts"].to_numpy(dtype=float)) * 50.0
    names = df["partner"].tolist()
    types = df["partner_type"].tolist()

    # Sort by size descending so small points sit on top
    order = np.argsort(-sizes)
    x, y, sizes = x[order], y[order], sizes[order]
    names = [names[i] for i in order]
    types = [types[i] for i in order]

    edge_colors = [feature_color(t) for t in types]

    # Highlight logic by basename: fill the best-scoring representative
    if basenames_highlight:
        best_idx_for_base: Dict[str,int] = {}
        for i, nm in enumerate(names):
            base = basename(nm)
            if base in basenames_highlight:
                if base not in best_idx_for_base or y[i] > y[best_idx_for_base[base]]:
                    best_idx_for_base[base] = i
        face_colors = ["C2" if best_idx_for_base.get(basename(nm)) == i else "white"
                       for i, nm in enumerate(names)]
    else:
        face_colors = ["white"] * len(names)

    # Legend handles
    feat_handles = [Patch(facecolor="none", edgecolor=feature_color(k), linewidth=2) for k in FEATURE_EDGE_COLORS]
    feat_labels  = list(FEATURE_EDGE_COLORS.keys())
    size_handles = [plt.scatter([], [], s=s*50, facecolors="none", edgecolors="black", linewidths=2)
                    for s in [5, 50, 500]]
    size_labels  = [f"{s} interactions" for s in [5, 50, 500]]

    # Compute total interactions to annotate focal gene (for info)
    # We'll recompute from df (incident edges counts sum)
    total_interactions = int(df["counts"].sum())

    pdf_path = f"{out_prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: labeled
        fig, ax = plt.subplots(figsize=(18, 10))
        plt.subplots_adjust(right=0.8)
        ax.scatter(x, y, s=sizes, facecolors=face_colors, alpha=0.7,
                   edgecolors=edge_colors, linewidths=1.5)

        # Mark focal gene on x-axis
        ax.scatter(gstart, 0, color="black", marker="^", s=100, zorder=5)
        ax.text(gstart, -0.002 * y_cap, str(total_interactions), color="grey",
                ha="center", va="top", alpha=1)

        # Axes
        ax.set_xlabel("Genomic coordinate (start of partner gene)")
        ax.set_ylabel("Interaction strength")
        ax.set_title(f"{gene}: global interaction map")
        ax.set_xlim(-0.04 * genome_max, genome_max * 1.05)
        ax.set_yscale("symlog", linthresh=symlog_linthresh, linscale=1, base=10)
        ax.set_ylim(0, y_cap)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1))
        ax.yaxis.set_major_formatter(ScalarFormatter())

        # Legends
        fig.legend(handles=feat_handles, labels=feat_labels, title="Feature type",
                   loc="upper left", bbox_to_anchor=(0.82, 0.95))
        fig.legend(handles=size_handles, labels=size_labels, title="Circle size",
                   loc="upper left", bbox_to_anchor=(0.82, 0.75))

        # Auto labels
        labeled_names = autolabel(ax, x, y, names, annotate_threshold, annotate_x_sep)

        plt.tight_layout(rect=[0, 0, 0.8, 1])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: unlabeled
        fig2, ax2 = plt.subplots(figsize=(18, 10))
        plt.subplots_adjust(right=0.8)
        ax2.scatter(x, y, s=sizes, facecolors=face_colors, alpha=0.7,
                    edgecolors=edge_colors, linewidths=1.5)
        ax2.scatter(gstart, 0, color="black", marker="^", s=100, zorder=5)
        ax2.text(gstart, -0.002 * y_cap, str(total_interactions), color="grey",
                 ha="center", va="top", alpha=1)
        ax2.set_xlabel("Genomic coordinate (start of partner gene)")
        ax2.set_ylabel("Interaction strength")
        ax2.set_title(f"{gene}: global interaction map (unlabeled)")
        ax2.set_xlim(-0.04 * genome_max, genome_max * 1.05)
        ax2.set_yscale("symlog", linthresh=symlog_linthresh, linscale=1, base=10)
        ax2.set_ylim(0, y_cap)
        ax2.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0]))
        ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1))
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        fig2.legend(handles=feat_handles, labels=feat_labels, title="Feature type",
                    loc="upper left", bbox_to_anchor=(0.82, 0.95))
        fig2.legend(handles=size_handles, labels=size_labels, title="Circle size",
                    loc="upper left", bbox_to_anchor=(0.82, 0.75))
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        pdf.savefig(fig2)
        plt.close(fig2)

    # Save labeled partner list
    labels_txt = f"{out_prefix}_labeled_partners.txt"
    with open(labels_txt, "w") as fh:
        fh.write("\n".join(labeled_names) + ("\n" if labeled_names else ""))

    logging.info("Wrote PDF: %s", pdf_path)
    logging.info("Wrote labels: %s", labels_txt)

# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    setup_logging(args.log)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    pairs = load_pairs(args.pairs)
    ann = load_annotations(args.annotations, args.ann_cols)

    # Build partner table for the requested gene
    df = build_partner_table(
        pairs=pairs,
        ann=ann,
        gene=args.gene,
        weight_col=args.weight_col,
        min_counts=int(args.min_counts),
        min_distance=int(args.min_distance),
        exclude_ftypes=args.exclude_feature_types or [],
    )
    if df.empty:
        logging.warning("No partners passed filters for gene '%s'. Nothing to plot.", args.gene)
        sys.exit(0)

    # Optional basename highlights
    basenames_set = load_basenames(args.highlight_basenames)

    # Compose plots
    make_plots(
        df=df,
        ann=ann,
        gene=args.gene,
        out_prefix=args.out_prefix,
        y_cap=float(args.y_cap),
        symlog_linthresh=float(args.symlog_linthresh),
        annotate_threshold=float(args.annotate_threshold),
        annotate_x_sep=int(args.annotate_x_sep),
        basenames_highlight=basenames_set,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
