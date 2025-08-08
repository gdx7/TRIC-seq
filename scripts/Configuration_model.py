#!/usr/bin/env python3
"""
TRIC-seq — Degree-preserving configuration-model null

Given pairwise interaction counts (ref, target, counts, io, …),
simulate a degree-preserving random multigraph (configuration model)
to obtain expected counts and empirical p-values for each observed pair.

Key features
------------
- Uses 'counts' (default) or 'io' as degrees (configurable).
- Optional minimum genomic distance filter via annotations.
- GPU acceleration with CuPy (optional), deterministic with --seed.
- Batched Monte Carlo without allocating N^2 histograms.
- Computes enrichment, depletion, and two-sided p-values + BH-FDR.
- Merges results back into the input table and appends feature coordinates.

Inputs
------
1) interaction_counts CSV: must contain columns at least:
   ref,target,counts,io  (others are carried through)
2) annotations CSV: at least columns (rna,start,end) [names configurable]

Output
------
CSV with added columns:
  expected_count, p_enrich, p_deplete, p_twosided,
  q_enrich, q_deplete, q_twosided,
  odds_ratio, log2_or, start_ref, end_ref, start_target, end_target
and a JSON <out>.manifest.json with provenance.

Usage
-----
python tricseq_configuration_model.py \
  --pairs data/EC/Nova/analysis_RX22GD7.csv \
  --annotations data/MX/annotations_MX.csv \
  --out data/EC/Nova/analysis_RX22GD7_MC.csv \
  --deg-metric counts \
  --iters 200000 \
  --batch 50 \
  --min-dist 5000 \
  --seed 42 \
  --gpu 1

Notes
-----
- For large degrees (sum_k), MC can be heavy. Increase --batch, reduce --iters,
  or use GPU. The script adapts to CPU if CuPy is unavailable or --gpu=0.
- Distance filtering is optional; set --min-dist 0 to disable.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import hashlib
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional GPU
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# ----------------------------- utils -----------------------------

def md5sum(path: str) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR on 1D array (nan-safe)."""
    x = np.asarray(pvals, dtype=float)
    n = np.sum(~np.isnan(x))
    order = np.argsort(np.where(np.isnan(x), np.inf, x))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(x) + 1)
    q = x * (n / np.maximum(ranks, 1))
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(np.where(np.isnan(x[order]), np.nan, q[order][::-1]))[::-1]
    out = np.empty_like(x)
    out[order] = q_sorted
    # cap at 1
    out = np.where(np.isnan(out), np.nan, np.minimum(out, 1.0))
    return out

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_configuration_model",
        description="Monte Carlo configuration-model null for TRIC-seq pair counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs", required=True, help="Input CSV from analyze_chimeras (ref,target,counts,io,...)")
    ap.add_argument("--annotations", required=True, help="CSV with feature coords; default cols rna,start,end (rename via --ann-cols)")
    ap.add_argument("--out", required=True, help="Output CSV with MC results merged.")
    ap.add_argument("--manifest", default=None, help="Optional JSON manifest path (default: <out>.manifest.json)")

    ap.add_argument("--ann-cols", nargs=3, metavar=("rna","start","end"),
                    default=("rna","start","end"),
                    help="Column names in annotations CSV for feature ID and coordinates.")
    ap.add_argument("--deg-metric", choices=("counts","io"), default="counts",
                    help="Feature degree metric for stub list.")
    ap.add_argument("--min-dist", type=int, default=5000,
                    help="Minimum genomic distance between features to keep (0 disables).")
    ap.add_argument("--iters", type=int, default=200000, help="Total Monte Carlo iterations.")
    ap.add_argument("--batch", type=int, default=50, help="Iterations per batch.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = nondeterministic).")
    ap.add_argument("--gpu", type=int, choices=(0,1), default=1, help="Use GPU if CuPy available.")
    ap.add_argument("--pseudocount", type=float, default=1e-6, help="Pseudocount for odds_ratio/log2_or.")
    ap.add_argument("--drop-self", type=int, choices=(0,1), default=1, help="Drop self-pairs (ref==target).")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()

# ----------------------------- core -----------------------------

def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ref","target","counts","io"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in --pairs: {missing}")
    return df

def load_annotations(path: str, cols: Tuple[str,str,str]) -> pd.DataFrame:
    rna, start, end = cols
    usecols = [rna,start,end]
    ann = pd.read_csv(path, usecols=usecols).dropna()
    ann = ann.rename(columns={rna:"rna", start:"start", end:"end"})
    # coerce numeric
    ann["start"] = pd.to_numeric(ann["start"], errors="coerce").astype(int)
    ann["end"]   = pd.to_numeric(ann["end"],   errors="coerce").astype(int)
    # normalize start<=end
    swap = ann["end"] < ann["start"]
    if swap.any():
        tmp = ann.loc[swap, "start"].copy()
        ann.loc[swap, "start"] = ann.loc[swap, "end"]
        ann.loc[swap, "end"]   = tmp
    # collapse duplicates to min/max
    ann = ann.groupby("rna", as_index=False).agg(start=("start","min"), end=("end","max"))
    return ann

def add_distance_filter(pairs: pd.DataFrame, ann: pd.DataFrame, min_dist: int) -> pd.DataFrame:
    if min_dist <= 0:
        return pairs
    coords = dict(zip(ann["rna"], zip(ann["start"], ann["end"])))
    def _gap(a: str, b: str) -> int:
        if a not in coords or b not in coords:
            return 0
        s1, e1 = coords[a]; s2, e2 = coords[b]
        if e1 < s2:  # a before b
            return s2 - e1
        if e2 < s1:  # b before a
            return s1 - e2
        return 0  # overlapping or adjacent
    keep = pairs.apply(lambda r: _gap(r["ref"], r["target"]) >= min_dist, axis=1)
    return pairs.loc[keep].copy()

def build_stub_array(pairs: pd.DataFrame, metric: str) -> np.ndarray:
    deg = {}
    for _, r in pairs.iterrows():
        w = int(r[metric])
        if w <= 0: 
            continue
        deg[r["ref"]]    = deg.get(r["ref"], 0) + w
        deg[r["target"]] = deg.get(r["target"], 0) + w
    # build stubs
    labels = list(deg.keys())
    degrees = np.array([deg[x] for x in labels], dtype=int)
    total = int(degrees.sum())
    if total % 2 == 1:
        # drop one stub to make even
        idx = np.argmax(degrees)
        degrees[idx] -= 1
        total -= 1
    # repeat via numpy for speed
    stubs = np.repeat(np.arange(len(labels), dtype=np.int32), degrees)
    return stubs, labels, degrees

def make_observed_index(pairs: pd.DataFrame, labels: list[str]) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int,int],int]]:
    # map feature->int
    fmap = {f:i for i,f in enumerate(labels)}
    # undirected pair key
    obs = {}
    for _, r in pairs.iterrows():
        a = fmap.get(r["ref"]); b = fmap.get(r["target"])
        if a is None or b is None: 
            continue
        if a == b: 
            continue
        u, v = (a,b) if a < b else (b,a)
        obs[(u,v)] = int(r["counts"])
    # encode pairs as u*N + v
    N = len(labels)
    keys = np.array(sorted(obs.keys()), dtype=np.int64)
    codes = keys[:,0] * N + keys[:,1]
    counts = np.array([obs[tuple(k)] for k in keys], dtype=np.int64)
    return codes, counts, obs

def rng_seed(seed: int, use_gpu: bool):
    if seed and seed > 0:
        np.random.seed(seed)
        if use_gpu and _HAS_CUPY:
            cp.random.seed(seed)

def simulate_mc(
    stubs: np.ndarray,
    N: int,
    obs_codes: np.ndarray,
    iters: int,
    batch: int,
    seed: int,
    use_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sim_sum[M], ge_obs[M], le_obs[M]
    """
    rng_seed(seed, use_gpu)
    M = len(obs_codes)
    sim_sum = np.zeros(M, dtype=np.float64)
    ge_obs = np.zeros(M, dtype=np.int64)
    le_obs = np.zeros(M, dtype=np.int64)

    # pre-sort obs codes for fast mapping
    obs_sorted_idx = np.argsort(obs_codes)
    obs_sorted = obs_codes[obs_sorted_idx]

    # Helper: accumulate counts from one permutation batch
    def accumulate_from_codes(codes_vec):
        # unique + counts
        uniq, cts = np.unique(codes_vec, return_counts=True)
        # map uniq -> observed via searchsorted
        pos = np.searchsorted(obs_sorted, uniq)
        mask = (pos < len(obs_sorted)) & (obs_sorted[pos] == uniq)
        if mask.any():
            # indices in observed order
            obs_pos = obs_sorted_idx[pos[mask]]
            # add counts
            add = np.zeros(M, dtype=np.int64)
            add[obs_pos] = cts[mask]
        else:
            add = np.zeros(M, dtype=np.int64)
        return add

    # GPU or CPU buffers
    if use_gpu and _HAS_CUPY:
        stubs_gpu = cp.asarray(stubs, dtype=cp.int32)
        obs_sorted_gpu = cp.asarray(obs_sorted, dtype=cp.int64)
        obs_sorted_idx_gpu = cp.asarray(obs_sorted_idx, dtype=cp.int64)

    done = 0
    while done < iters:
        b = min(batch, iters - done)
        # For each iteration, permute stubs and form undirected pairs
        for _ in range(b):
            if use_gpu and _HAS_CUPY:
                perm = cp.random.permutation(stubs_gpu)
                pairs = perm.reshape((-1, 2))
                lo = cp.minimum(pairs[:,0], pairs[:,1])
                hi = cp.maximum(pairs[:,0], pairs[:,1])
                codes = (lo.astype(cp.int64) * N + hi.astype(cp.int64))
                uniq, cts = cp.unique(codes, return_counts=True)
                # map uniq -> observed via searchsorted on GPU
                pos = cp.searchsorted(obs_sorted_gpu, uniq)
                mask = (pos < obs_sorted_gpu.size) & (obs_sorted_gpu[pos] == uniq)
                if int(mask.sum()) > 0:
                    obs_pos = obs_sorted_idx_gpu[pos[mask]]
                    add_gpu = cp.zeros(M, dtype=cp.int64)
                    add_gpu[obs_pos] = cts[mask]
                    add = cp.asnumpy(add_gpu)
                else:
                    add = np.zeros(M, dtype=np.int64)
            else:
                perm = np.random.permutation(stubs)
                pairs = perm.reshape((-1, 2))
                lo = np.minimum(pairs[:,0], pairs[:,1]).astype(np.int64)
                hi = np.maximum(pairs[:,0], pairs[:,1]).astype(np.int64)
                codes = lo * N + hi
                add = accumulate_from_codes(codes)

            sim_sum += add
            # ge/le tallies handled later in one pass for efficiency
            # But we can accumulate per-iter as: ge_obs += (add >= obs_counts)
            # That requires obs_counts in scope; we handle after loop by storing per-iter
        # To compute ge/le without storing every iteration’s vector, approximate by
        # comparing codes hist against observed via repetition.
        # Simpler: maintain rolling tallies by drawing another pass comparing add>=obs.
        # Here we approximate via Poisson with mean add (per iter), but better to
        # compute exactly: store batch adds in a list (OK for moderate M * batch).
        done += b

    return sim_sum, ge_obs, le_obs  # ge/le will be filled below with a second pass

def main():
    args = parse_args()
    setup_logging(args.log)
    t0 = time.time()

    if args.gpu and not _HAS_CUPY:
        logging.warning("CuPy not available; falling back to CPU.")

    pairs = load_pairs(args.pairs)
    ann = load_annotations(args.annotations, tuple(args.ann_cols))

    # Drop self pairs if requested
    if args.drop_self:
        pairs = pairs[pairs["ref"] != pairs["target"]].copy()

    # Distance filter (if enabled)
    if args.min_dist > 0:
        before = len(pairs)
        pairs = add_distance_filter(pairs, ann, args.min_dist)
        logging.info("Distance filter kept %d/%d pairs (min-dist=%d).", len(pairs), before, args.min_dist)

    # Build stubs from chosen degree metric
    stubs, labels, degrees = build_stub_array(pairs, args.deg_metric)
    N = len(labels)
    if stubs.size == 0:
        raise ValueError("Empty stub list; check --deg-metric and input counts.")

    # Observed undirected pair index
    obs_codes, obs_counts_vec, obs_dict = make_observed_index(pairs, labels)
    M = len(obs_codes)
    logging.info("Features: %d | Observed undirected pairs: %d | Total stubs: %d (edges=%d)",
                 N, M, stubs.size, stubs.size // 2)

    # Monte Carlo simulation
    iters = int(args.iters)
    batch = max(1, int(args.batch))
    use_gpu = bool(args.gpu and _HAS_CUPY)
    logging.info("Running MC: iters=%d, batch=%d, gpu=%s, seed=%d", iters, batch, use_gpu, args.seed)

    # We need both sum of counts and tail probabilities (>=, <=).
    # To avoid storing per-iteration vectors for all pairs (M large),
    # we do two passes: (1) accumulate total counts to get mean,
    # (2) re-run with tally of >= and <= relative to observed.
    sim_sum, _, _ = simulate_mc(stubs, N, obs_codes, iters, batch, args.seed or 0, use_gpu)

    # Second pass for ge/le (use different seed stream if seed given to decorrelate)
    ge_obs = np.zeros(M, dtype=np.int64)
    le_obs = np.zeros(M, dtype=np.int64)

    # Precompute observed for comparisons
    obs_counts = obs_counts_vec.copy()

    # helper reused from simulate_mc (duplicated small inner for clarity/perf)
    def compare_batch(counts_add: np.ndarray):
        # counts_add is counts vector for observed M pairs in ONE iteration
        ge = (counts_add >= obs_counts).astype(np.int64)
        le = (counts_add <= obs_counts).astype(np.int64)
        return ge, le

    # Iterate again, but this time accumulate per-iter vector (M) and compare
    # For memory, keep batch loops on CPU/GPU as above
    # Reuse code structure with slight modification to produce one-iter add
    rng_seed((args.seed or 0) + 1337, use_gpu)

    done = 0
    while done < iters:
        b = min(batch, iters - done)
        for _ in range(b):
            if use_gpu:
                stubs_gpu = cp.asarray(stubs, dtype=cp.int32)
                perm = cp.random.permutation(stubs_gpu)
                pairs_gpu = perm.reshape((-1, 2))
                lo = cp.minimum(pairs_gpu[:,0], pairs_gpu[:,1])
                hi = cp.maximum(pairs_gpu[:,0], pairs_gpu[:,1])
                codes = (lo.astype(cp.int64) * N + hi.astype(cp.int64))
                uniq, cts = cp.unique(codes, return_counts=True)
                # map uniq -> observed
                obs_sorted_idx = np.argsort(obs_codes)
                obs_sorted = obs_codes[obs_sorted_idx]
                obs_sorted_gpu = cp.asarray(obs_sorted, dtype=cp.int64)
                pos = cp.searchsorted(obs_sorted_gpu, uniq)
                mask = (pos < obs_sorted_gpu.size) & (obs_sorted_gpu[pos] == uniq)
                add_gpu = cp.zeros(M, dtype=cp.int64)
                if int(mask.sum()) > 0:
                    obs_pos_sorted = pos[mask]
                    # map back to original order
                    back_idx = cp.asarray(obs_sorted_idx, dtype=cp.int64)[obs_pos_sorted]
                    add_gpu[back_idx] = cts[mask]
                add = cp.asnumpy(add_gpu)
            else:
                perm = np.random.permutation(stubs)
                pairs_np = perm.reshape((-1, 2))
                lo = np.minimum(pairs_np[:,0], pairs_np[:,1]).astype(np.int64)
                hi = np.maximum(pairs_np[:,0], pairs_np[:,1]).astype(np.int64)
                codes = lo * N + hi
                uniq, cts = np.unique(codes, return_counts=True)
                # map uniq -> observed
                obs_sorted_idx = np.argsort(obs_codes)
                obs_sorted = obs_codes[obs_sorted_idx]
                pos = np.searchsorted(obs_sorted, uniq)
                mask = (pos < len(obs_sorted)) & (obs_sorted[pos] == uniq)
                add = np.zeros(M, dtype=np.int64)
                if np.any(mask):
                    back_idx = obs_sorted_idx[pos[mask]]
                    add[back_idx] = cts[mask]

            ge, le = compare_batch(add)
            ge_obs += ge
            le_obs += le
        done += b

    # Expectations and p-values
    mean_sim = sim_sum / float(iters)
    p_enrich = ge_obs / float(iters)
    p_deplete = le_obs / float(iters)
    p_two = np.minimum(2.0 * np.minimum(p_enrich, p_deplete), 1.0)

    # Odds ratio vs expected (with pseudocount)
    pc = float(args.pseudocount)
    odds_ratio = (obs_counts + pc) / (mean_sim + pc)
    log2_or = np.log2(odds_ratio)

    # FDR
    q_enrich = bh_fdr(p_enrich)
    q_deplete = bh_fdr(p_deplete)
    q_two = bh_fdr(p_two)

    # Build results table
    # map back int labels -> names
    labels_arr = np.array(labels, dtype=object)
    u = (obs_codes // N).astype(int)
    v = (obs_codes %  N).astype(int)
    res = pd.DataFrame({
        "ref": labels_arr[u],
        "target": labels_arr[v],
        "observed_counts": obs_counts,
        "expected_count": mean_sim,
        "p_enrich": p_enrich,
        "p_deplete": p_deplete,
        "p_twosided": p_two,
        "q_enrich": q_enrich,
        "q_deplete": q_deplete,
        "q_twosided": q_two,
        "odds_ratio": odds_ratio,
        "log2_or": log2_or,
    })

    # Merge back to full directed table (ref,target) from input
    df_in = pd.read_csv(args.pairs)
    df_in["pair_key"] = df_in.apply(lambda r: "_".join(sorted([str(r["ref"]), str(r["target"])])), axis=1)
    res["pair_key"] = res.apply(lambda r: "_".join(sorted([str(r["ref"]), str(r["target"])])), axis=1)
    merged = df_in.merge(
        res.drop(columns=["ref","target"]), on="pair_key", how="left"
    ).drop(columns=["pair_key"])

    # Append coordinates of both partners
    ann_ref = ann.rename(columns={"rna":"ref","start":"start_ref","end":"end_ref"})
    ann_tar = ann.rename(columns={"rna":"target","start":"start_target","end":"end_target"})
    merged = merged.merge(ann_ref, on="ref", how="left").merge(ann_tar, on="target", how="left")

    # Save
    out_csv = args.out
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    logging.info("Wrote: %s  (rows=%d)", out_csv, merged.shape[0])

    # Manifest
    manifest_path = args.manifest or (out_csv + ".manifest.json")
    manifest = {
        "script": os.path.basename(__file__),
        "version_info": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cupy": (cp.__version__ if _HAS_CUPY else "absent"),
        },
        "inputs": {
            "pairs_csv": {"path": args.pairs, "md5": md5sum(args.pairs)},
            "annotations": {"path": args.annotations, "md5": md5sum(args.annotations)},
        },
        "outputs": {"csv": out_csv},
        "params": {
            "deg_metric": args.deg_metric,
            "min_dist": args.min_dist,
            "iters": args.iters,
            "batch": args.batch,
            "seed": args.seed,
            "gpu": bool(args.gpu and _HAS_CUPY),
            "pseudocount": args.pseudocount,
            "drop_self": bool(args.drop_self),
        },
        "stats": {
            "features": int(N),
            "observed_pairs": int(M),
            "total_stubs": int(stubs.size),
            "edges": int(stubs.size // 2),
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
