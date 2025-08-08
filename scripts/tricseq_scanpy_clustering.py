#!/usr/bin/env python3
"""
TRIC-seq — Unsupervised clustering of the RNA–RNA interactome (Scanpy)

Builds a weighted RNA×RNA matrix from a pair table (e.g., from
tricseq_analyze_chimeras + tricseq_configuration_model), applies
filters, and performs PCA → kNN graph → Leiden clustering, with
optional UMAP/t-SNE embeddings and marker visualizations.

Inputs
------
1) Pair table CSV with at least:
      ref, target, counts, adjusted_score
   (additional columns like totals, total_ref are used if present)

2) Annotation CSV to provide genomic spans and (optionally) feature types.
   Column names are configurable via --ann-cols (default: gene_name,start,end,feature_type,strand,chromosome).

Outputs
-------
- Cluster assignments CSV (one row per RNA)
- AnnData .h5ad file with embeddings and clustering
- Optional plots (UMAP/t-SNE) saved to disk
- JSON manifest with parameters and environment info

Example
-------
python tricseq_scanpy_clustering.py \
  --pairs pairs_mc.csv \
  --annotations annotations.csv \
  --out-prefix out/cluster \
  --min-counts 5 \
  --min-distance 3000 \
  --weight-col adjusted_score \
  --score-cap 500 \
  --exclude-feature-types rRNA tRNA \
  --resolution 1.0 \
  --neighbors 15 \
  --n-pcs 15 \
  --make-umap 1 \
  --make-tsne 1 \
  --seed 42
"""

from __future__ import annotations
import argparse, json, logging, os, sys, time, hashlib, re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# scanpy/anndata for clustering/plots
import scanpy as sc
import anndata


# ------------------------- utils -------------------------

def md5sum(path: str) -> Optional[str]:
    try:
        import hashlib
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
        prog="tricseq_scanpy_clustering",
        description="Leiden clustering of RNA–RNA interactome using Scanpy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs", required=True, help="Pair table CSV (ref,target,counts,adjusted_score, ...).")
    ap.add_argument("--annotations", required=True, help="Annotation CSV for coordinates and (optional) feature types.")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for CSV/H5AD/plots (directory will be created).")

    # Columns in annotations
    ap.add_argument("--ann-cols", nargs="+",
                    default=["gene_name","start","end","feature_type","strand","chromosome"],
                    help="Columns in annotations CSV (order: gene, start, end, [feature_type] [strand] [chromosome]).")

    # Filtering thresholds
    ap.add_argument("--min-counts", type=int, default=5, help="Minimum read-level counts per pair to keep.")
    ap.add_argument("--min-totals", type=int, default=10, help="Minimum 'totals' per target if column exists.")
    ap.add_argument("--min-total-ref", type=int, default=10, help="Minimum 'total_ref' per ref if column exists.")
    ap.add_argument("--min-distance", type=int, default=3000, help="Minimum genomic distance between partners (bp).")

    # Edge weights
    ap.add_argument("--weight-col", default="adjusted_score", help="Column to use as edge weight.")
    ap.add_argument("--score-cap", type=float, default=500.0, help="Cap for the weight column.")
    ap.add_argument("--drop-zero-weight", type=int, choices=(0,1), default=1, help="Drop pairs with weight <= 0.")

    # Feature/gene exclusions
    ap.add_argument("--exclude-feature-types", nargs="*", default=["rRNA","tRNA"],
                    help="Feature types to exclude if annotation provides 'feature_type'.")
    ap.add_argument("--exclude-genes", nargs="*", default=[], help="Exact gene names to drop.")
    ap.add_argument("--exclude-rrna-prefix", default="rr", help="Regex prefix (with optional 5'/3') to drop (e.g., rr).")
    ap.add_argument("--case-insensitive", type=int, choices=(0,1), default=1, help="Case-insensitive regex matching.")

    # Scanpy/Leiden params
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    ap.add_argument("--neighbors", type=int, default=15, help="kNN neighbors.")
    ap.add_argument("--n-pcs", type=int, default=15, help="Number of PCs to use in neighbors/embeddings.")
    ap.add_argument("--normalize", type=int, choices=(0,1), default=1, help="Apply per-observation normalization.")
    ap.add_argument("--log1p", type=int, choices=(0,1), default=1, help="Apply log1p transform.")
    ap.add_argument("--make-umap", type=int, choices=(0,1), default=1, help="Compute UMAP and save plot.")
    ap.add_argument("--make-tsne", type=int, choices=(0,1), default=1, help="Compute t-SNE and save plot.")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")

    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()


# ------------------------- I/O -------------------------

def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ref","target","counts"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Pair table missing required columns: {missing}")
    return df

def load_annotations(path: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # map to canonical names if present
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
            raise ValueError(f"Annotation CSV must include column '{c}' (configure via --ann-cols).")

    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype(int)
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce").astype(int)
    swap = df["end"] < df["start"]
    if swap.any():
        tmp = df.loc[swap,"start"].copy()
        df.loc[swap,"start"] = df.loc[swap,"end"]
        df.loc[swap,"end"]   = tmp

    # collapse duplicates (same gene name) to min/max span and first feature_type
    agg = {"start":"min","end":"max"}
    if "feature_type" in df.columns:
        agg["feature_type"] = "first"
    if "strand" in df.columns:
        agg["strand"] = "first"
    if "chromosome" in df.columns:
        agg["chromosome"] = "first"

    df = df.groupby("gene_name", as_index=False).agg(agg)
    return df


# ------------------------- filters -------------------------

def build_gene_maps(ann: pd.DataFrame) -> Tuple[Dict[str,Tuple[int,int]], Dict[str,str]]:
    coords = dict(zip(ann["gene_name"], zip(ann["start"], ann["end"])))
    ftypes = dict(zip(ann["gene_name"], ann["feature_type"])) if "feature_type" in ann.columns else {}
    return coords, ftypes

def distance_bp(a: str, b: str, coords: Dict[str,Tuple[int,int]]) -> Optional[int]:
    aco = coords.get(a); bco = coords.get(b)
    if aco is None or bco is None: return None
    s1,e1 = aco; s2,e2 = bco
    if e1 < s2: return s2 - e1
    if e2 < s1: return s1 - e2
    return 0  # overlap/adjacent

def build_rrna_regex(prefix: str, case_insensitive: bool) -> re.Pattern:
    # matches optional 5' or 3' prefix, then prefix (e.g., 'rr'), e.g. "5'rrf", "3'rrl", "rrs"
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(rf"^(5'|3')?{re.escape(prefix)}", flags)


# ------------------------- matrix build -------------------------

def build_weighted_matrix(df: pd.DataFrame, genes: List[str], weight_col: str) -> csr_matrix:
    idx_map = {g:i for i,g in enumerate(genes)}
    r = df["ref"].map(idx_map).to_numpy()
    c = df["target"].map(idx_map).to_numpy()
    w = df[weight_col].astype(float).to_numpy()

    # symmetric: add both (i,j) and (j,i)
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([w, w])

    n = len(genes)
    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    return mat


# ------------------------- main -------------------------

def main():
    args = parse_args()
    setup_logging(args.log)
    t0 = time.time()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    pairs = load_pairs(args.pairs)
    ann = load_annotations(args.annotations, args.ann_cols)
    coords, ftypes = build_gene_maps(ann)

    # Basic filters
    keep = pairs["counts"] >= int(args.min_counts)
    if "totals" in pairs.columns:
        keep &= pairs["totals"] >= int(args.min_totals)
    if "total_ref" in pairs.columns:
        keep &= pairs["total_ref"] >= int(args.min_total_ref)
    pairs = pairs.loc[keep].copy()

    # rRNA-like name filter (regex)
    rrna_pat = build_rrna_regex(args.exclude_rrna_prefix, bool(args.case_insensitive))
    pairs = pairs[~(pairs["ref"].astype(str).str.match(rrna_pat) |
                    pairs["target"].astype(str).str.match(rrna_pat))].copy()

    # Feature-type filter (if available)
    if args.exclude_feature_types and "feature_type" in ann.columns:
        ref_ft = pairs["ref"].map(ftypes)
        tar_ft = pairs["target"].map(ftypes)
        mask_ft = (~ref_ft.isin(args.exclude_feature_types)) & (~tar_ft.isin(args.exclude_feature_types))
        pairs = pairs.loc[mask_ft].copy()

    # Drop explicit genes
    if args.exclude_genes:
        pairs = pairs[~pairs["ref"].isin(args.exclude_genes)]
        pairs = pairs[~pairs["target"].isin(args.exclude_genes)]

    # Distance filter
    if args.min_distance and args.min_distance > 0:
        dists = pairs.apply(lambda r: distance_bp(r["ref"], r["target"], coords), axis=1)
        pairs = pairs.assign(distance_bp=dists).dropna(subset=["distance_bp"])
        pairs = pairs[pairs["distance_bp"] >= int(args.min_distance)].copy()

    # Weight column
    if args.weight_col not in pairs.columns:
        raise ValueError(f"--weight-col '{args.weight_col}' not found in pair table.")
    if int(args.drop_zero_weight):
        pairs = pairs[pairs[args.weight_col] > 0].copy()
    if args.score_cap is not None:
        pairs.loc[:, args.weight_col] = pairs[args.weight_col].clip(upper=float(args.score_cap))

    # Keep only genes that appear after filtering
    genes = sorted(set(pairs["ref"]).union(set(pairs["target"])))
    logging.info("Genes after filtering: %d", len(genes))
    if len(genes) < 3:
        raise ValueError("Too few genes after filtering for clustering.")

    # Build weighted adjacency (RNA×RNA)
    X = build_weighted_matrix(pairs, genes, args.weight_col)
    logging.info("Matrix shape: %s, nnz=%d", X.shape, X.nnz)

    # AnnData: observations = RNAs; variables = RNAs (square matrix)
    adata = sc.AnnData(X=X, obs=pd.DataFrame(index=genes), var=pd.DataFrame(index=genes))
    adata.uns["params"] = {
        "min_counts": int(args.min_counts),
        "min_totals": int(args.min_totals),
        "min_total_ref": int(args.min_total_ref),
        "min_distance": int(args.min_distance),
        "weight_col": args.weight_col,
        "score_cap": float(args.score_cap),
        "exclude_feature_types": args.exclude_feature_types,
        "exclude_genes": args.exclude_genes,
        "exclude_rrna_prefix": args.exclude_rrna_prefix,
    }

    # Preprocessing
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=150, facecolor="white")
    np.random.seed(int(args.random_state))

    if int(args.normalize):
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    if int(args.log1p):
        sc.pp.log1p(adata)

    # PCA
    n_pcs = int(args.n_pcs)
    max_pcs = max(1, min(adata.n_obs - 1, adata.n_vars - 1, n_pcs))
    sc.tl.pca(adata, n_comps=max_pcs, svd_solver="arpack", random_state=int(args.random_state))

    # Neighbors + Leiden
    sc.pp.neighbors(adata, n_neighbors=int(args.neighbors), n_pcs=max_pcs, metric="cosine", random_state=int(args.random_state))
    sc.tl.leiden(adata, resolution=float(args.resolution), random_state=int(args.random_state), key_added="leiden")

    # Embeddings & plots
    plot_dir = os.path.dirname(args.out_prefix) or "."
    if int(args.make_umap):
        sc.tl.umap(adata, random_state=int(args.random_state))
        sc.pl.umap(adata, color="leiden", s=10, frameon=False, save=False, show=False)
        import matplotlib.pyplot as plt
        plt.title("UMAP — Leiden clusters"); plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_umap_leiden.png", dpi=300); plt.close()

    if int(args.make_tsne):
        sc.tl.tsne(adata, n_pcs=max_pcs, random_state=int(args.random_state))
        sc.pl.tsne(adata, color="leiden", s=10, frameon=False, save=False, show=False)
        import matplotlib.pyplot as plt
        plt.title("t-SNE — Leiden clusters"); plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_tsne_leiden.png", dpi=300); plt.close()

    # Save outputs
    clusters_csv = f"{args.out_prefix}_clusters.csv"
    adata.obs[["leiden"]].to_csv(clusters_csv)
    h5ad_path = f"{args.out_prefix}.h5ad"
    adata.write(h5ad_path)

    logging.info("Wrote clusters: %s", clusters_csv)
    logging.info("Wrote AnnData: %s", h5ad_path)

    # Manifest
    manifest = {
        "script": os.path.basename(__file__),
        "version_info": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scanpy": sc.__version__,
            "anndata": anndata.__version__,
        },
        "inputs": {
            "pairs": {"path": args.pairs, "md5": md5sum(args.pairs)},
            "annotations": {"path": args.annotations, "md5": md5sum(args.annotations)},
        },
        "outputs": {
            "clusters_csv": clusters_csv,
            "h5ad": h5ad_path,
            "umap_png": f"{args.out_prefix}_umap_leiden.png" if int(args.make_umap) else None,
            "tsne_png": f"{args.out_prefix}_tsne_leiden.png" if int(args.make_tsne) else None,
        },
        "params": {
            "min_counts": int(args.min_counts),
            "min_totals": int(args.min_totals),
            "min_total_ref": int(args.min_total_ref),
            "min_distance": int(args.min_distance),
            "weight_col": args.weight_col,
            "score_cap": float(args.score_cap),
            "drop_zero_weight": bool(args.drop_zero_weight),
            "exclude_feature_types": args.exclude_feature_types,
            "exclude_genes": args.exclude_genes,
            "exclude_rrna_prefix": args.exclude_rrna_prefix,
            "resolution": float(args.resolution),
            "neighbors": int(args.neighbors),
            "n_pcs": max_pcs,
            "normalize": bool(args.normalize),
            "log1p": bool(args.log1p),
            "make_umap": bool(args.make_umap),
            "make_tsne": bool(args.make_tsne),
            "random_state": int(args.random_state),
            "ann_cols": args.ann_cols,
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    with open(f"{args.out_prefix}.manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    logging.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger().exception("Fatal error: %s", e)
        sys.exit(1)
