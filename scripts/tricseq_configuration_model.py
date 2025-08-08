#!/usr/bin/env python3
"""
TRIC-seq — Degree-preserving configuration-model null

Simulates a degree-preserving random multigraph (configuration model) over
RNA features to estimate the expected interaction counts between pairs,
and computes empirical enrichment/depletion p-values and BH-FDR.

Inputs
------
1) Pair table (CSV) from the chimera analysis step with at least:
      ref,target,counts,io
   (Other columns are carried through to the output.)

2) Annotations CSV providing feature spans to enable optional distance filtering.
   Provide column names via --ann-cols (default: rna,start,end). Spans for the
   same feature ID are collapsed to min(start), max(end).

Outputs
-------
- The input pair table augmented with:
    expected_count, p_enrich, p_deplete, p_twosided,
    q_enrich, q_deplete, q_twosided,
    odds_ratio, log2_or, start_ref, end_ref, start_target, end_target
- A JSON manifest (<out>.manifest.json) with provenance (params, versions, checksums).

Example
-------
python tricseq_configuration_model.py \
  --pairs path/to/pairs.csv \
  --annotations path/to/annotations.csv \
  --out path/to/pairs_with_mc.csv \
  --deg-metric counts \
  --iters 200000 \
  --batch 50 \
  --min-dist 5000 \
  --seed 42 \
  --gpu 1
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional GPU (CuPy)
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


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (nan-safe)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return out
    x = p[mask]
    n = x.size
    order = np.argsort(x)
    ranks = np.arange(1, n + 1, dtype=float)
    x_sorted = x[order]
    q_sorted = np.minimum.accumulate((x_sorted * n / ranks)[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    out_vals = np.empty_like(x_sorted)
    out_vals[order] = q_sorted
    out[mask] = out_vals
    return out


def rng_seed(seed: int, use_gpu: bool) -> None:
    if seed and seed > 0:
        np.random.seed(seed)
        if use_gpu and _HAS_CUPY:
            cp.random.seed(seed)


# ----------------------------- args -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tricseq_configuration_model",
        description="Monte Carlo configuration-model null for TRIC-seq pair counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs", required=True, help="Input CSV from analyze_chimeras (ref,target,counts,io,...).")
    ap.add_argument("--annotations", required=True, help="CSV with feature coordinates.")
    ap.add_argument("--out", required=True, help="Output CSV with MC results merged.")
    ap.add_argument("--manifest", default=None, help="Optional JSON manifest path (default: <out>.manifest.json).")

    ap.add_argument("--ann-cols", nargs=3, metavar=("rna", "start", "end"),
                    default=("rna", "start", "end"),
                    help="Column names in annotations CSV for feature ID and coordinates.")
    ap.add_argument("--deg-metric", choices=("counts", "io"), default="counts",
                    help="Feature degree metric for stub list.")
    ap.add_argument("--min-dist", type=int, default=5000,
                    help="Minimum genomic distance between features to keep (0 disables).")
    ap.add_argument("--iters", type=int, default=200000, help="Total Monte Carlo iterations.")
    ap.add_argument("--batch", type=int, default=50, help="Iterations per batch.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = nondeterministic).")
    ap.add_argument("--gpu", type=int, choices=(0, 1), default=1, help="Use GPU if CuPy available.")
    ap.add_argument("--pseudocount", type=float, default=1e-6, help="Pseudocount for odds_ratio/log2_or.")
    ap.add_argument("--drop-self", type=int, choices=(0, 1), default=1, help="Drop self-pairs (ref==target).")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    return ap.parse_args()


# ----------------------------- I/O -----------------------------

def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ref", "target", "counts", "io"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in --pairs: {missing}")
    return df


def load_annotations(path: str, cols: Tuple[str, str, str]) -> pd.DataFrame:
    rna, start, end = cols
    usecols = [rna, start, end]
    ann = pd.read_csv(path, usecols=usecols).dropna()
    ann = ann.rename(columns={rna: "rna", start: "start", end: "end"})
    ann["start"] = pd.to_numeric(ann["start"], errors="coerce").astype(int)
    ann["end"] = pd.to_numeric(ann["end"], errors="coerce").astype(int)
    swap = ann["end"] < ann["start"]
    if swap.any():
        tmp = ann.loc[swap, "start"].copy()
        ann.loc[swap, "start"] = ann.loc[swap, "end"]
        ann.loc[swap, "end"] = tmp
    # collapse duplicates to min/max span per feature
    ann = ann.groupby("rna", as_index=False).agg(start=("start", "min"), end=("end", "max"))
    return ann


# ----------------------------- core helpers -----------------------------

def add_distance_filter(pairs: pd.DataFrame, ann: pd.DataFrame, min_dist: int) -> pd.DataFrame:
    if min_dist <= 0:
        return pairs
    coords = dict(zip(ann["rna"], zip(ann["start"], ann["end"])))

    def _gap(a: str, b: str) -> int:
        if a not in coords or b not in coords:
            return 0
        s1, e1 = coords[a]
        s2, e2 = coords[b]
        if e1 < s2:
            return s2 - e1
        if e2 < s1:
            return s1 - e2
        return 0  # overlapping or adjacent

    keep = pairs.apply(lambda r: _gap(r["ref"], r["target"]) >= min_dist, axis=1)
    return pairs.loc[keep].copy()


def build_stub_array(pairs: pd.DataFrame, metric: str) -> Tuple[np.ndarray, list[str], np.ndarray]:
    deg: Dict[str, int] = {}
    for _, r in pairs.iterrows():
        w = int(r[metric])
        if w <= 0:
            continue
        deg[r["ref"]] = deg.get(r["ref"], 0) + w
        deg[r["target"]] = deg.get(r["target"], 0) + w
    labels = list(deg.keys())
    degrees = np.array([deg[x] for x in labels], dtype=np.int64)
    total = int(degrees.sum())
    if total % 2 == 1:
        # make even by removing one stub from the max-degree node
        i = int(np.argmax(degrees))
        degrees[i] -= 1
        total -= 1
    stubs = np.repeat(np.arange(len(labels), dtype=np.int32), degrees)
    return stubs, labels, degrees


def make_observed_index(pairs: pd.DataFrame, labels: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    fmap = {f: i for i, f in enumerate(labels)}
    N = len(labels)
    obs_codes = {}
    for _, r in pairs.iterrows():
        a = fmap.get(r["ref"])
        b = fmap.get(r["target"])
        if a is None or b is None or a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        code = np.int64(u) * N + np.int64(v)
        obs_codes[code] = obs_codes.get(code, 0) + int(r["counts"])
    codes = np.array(sorted(obs_codes.keys()), dtype=np.int64)
    counts = np.array([obs_codes[c] for c in codes], dtype=np.int64)
    return codes, counts


# ----------------------------- simulation -----------------------------

def simulate_means(
    stubs: np.ndarray,
    N: int,
    obs_codes: np.ndarray,
    iters: int,
    batch: int,
    seed: int,
    use_gpu: bool,
) -> np.ndarray:
    """Return sim_sum[M] of counts across iterations for each observed undirected pair."""
    rng_seed(seed, use_gpu)
    M = len(obs_codes)
    sim_sum = np.zeros(M, dtype=np.float64)

    # precompute for mapping uniq->observed
    obs_sorted_idx = np.argsort(obs_codes)
    obs_sorted = obs_codes[obs_sorted_idx]

    if use_gpu and _HAS_CUPY:
        stubs_gpu = cp.asarray(stubs, dtype=cp.int32)
        obs_sorted_gpu = cp.asarray(obs_sorted, dtype=cp.int64)
        back_idx_gpu = cp.asarray(obs_sorted_idx, dtype=cp.int64)

    done = 0
    while done < iters:
        b = min(batch, iters - done)
        for _ in range(b):
            if use_gpu and _HAS_CUPY:
                perm = cp.random.permutation(stubs_gpu)
                pairs = perm.reshape((-1, 2))
                lo = cp.minimum(pairs[:, 0], pairs[:, 1])
                hi = cp.maximum(pairs[:, 0], pairs[:, 1])
                codes = (lo.astype(cp.int64) * N + hi.astype(cp.int64))
                uniq, cts = cp.unique(codes, return_counts=True)
                pos = cp.searchsorted(obs_sorted_gpu, uniq)
                mask = (pos < obs_sorted_gpu.size) & (obs_sorted_gpu[pos] == uniq)
                if int(mask.sum()) > 0:
                    add = np.zeros(M, dtype=np.int64)
                    obs_pos = back_idx_gpu[pos[mask]]
                    add_gpu = cp.zeros(M, dtype=cp.int64)
                    add_gpu[obs_pos] = cts[mask]
                    add = cp.asnumpy(add_gpu)
                    sim_sum += add
            else:
                perm = np.random.permutation(stubs)
                pairs = perm.reshape((-1, 2))
                lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                codes = lo * N + hi
                uniq, cts = np.unique(codes, return_counts=True)
                pos = np.searchsorted(obs_sorted, uniq)
                mask = (pos < len(obs_sorted)) & (obs_sorted[pos] == uniq)
                if np.any(mask):
                    add = np.zeros(M, dtype=np.int64)
                    add[np.take(obs_sorted_idx, pos[mask])] = cts[mask]
                    sim_sum += add
        done += b
    return sim_sum


def simulate_tails(
    stubs: np.ndarray,
    N: int,
    obs_codes: np.ndarray,
    obs_counts: np.ndarray,
    iters: int,
    batch: int,
    seed: int,
    use_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ge_obs[M], le_obs[M]) tail tallies across iterations."""
    rng_seed(seed, use_gpu)
    M = len(obs_codes)
    ge_obs = np.zeros(M, dtype=np.int64)
    le_obs = np.zeros(M, dtype=np.int64)

    obs_sorted_idx = np.argsort(obs_codes)
    obs_sorted = obs_codes[obs_sorted_idx]

    if use_gpu and _HAS_CUPY:
        stubs_gpu = cp.asarray(stubs, dtype=cp.int32)
        obs_sorted_gpu = cp.asarray(obs_sorted, dtype=cp.int64)
        back_idx_gpu = cp.asarray(obs_sorted_idx, dtype=cp.int64)
        obs_counts_gpu = cp.asarray(obs_counts, dtype=cp.int64)

    done = 0
    while done < iters:
        b = min(batch, iters - done)
        for _ in range(b):
            if use_gpu and _HAS_CUPY:
                perm = cp.random.permutation(stubs_gpu)
                pairs = perm.reshape((-1, 2))
                lo = cp.minimum(pairs[:, 0], pairs[:, 1])
                hi = cp.maximum(pairs[:, 0], pairs[:, 1])
                codes = (lo.astype(cp.int64) * N + hi.astype(cp.int64))
                uniq, cts = cp.unique(codes, return_counts=True)
                pos = cp.searchsorted(obs_sorted_gpu, uniq)
                mask = (pos < obs_sorted_gpu.size) & (obs_sorted_gpu[pos] == uniq)
                add = cp.zeros(len(obs_codes), dtype=cp.int64)
                if int(mask.sum()) > 0:
                    obs_pos = back_idx_gpu[pos[mask]]
                    add[obs_pos] = cts[mask]
                ge_obs += cp.asnumpy(add >= obs_counts_gpu).astype(np.int64)
                le_obs += cp.asnumpy(add <= obs_counts_gpu).astype(np.int64)
            else:
                perm = np.random.permutation(stubs)
                pairs = perm.reshape((-1, 2))
                lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                codes = lo * N + hi
                uniq, cts = np.unique(codes, return_counts=True)
                pos = np.searchsorted(obs_sorted, uniq)
                mask = (pos < len(obs_sorted)) & (obs_sorted[pos] == uniq)
                add = np.zeros(len(obs_codes), dtype=np.int64)
                if np.any(mask):
                    add[np.take(obs_sorted_idx, pos[mask])] = cts[mask]
                ge_obs += (add >= obs_counts).astype(np.int64)
                le_obs += (add <= obs_counts).astype(np.int64)
        done += b
    return ge_obs, le_obs


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    setup_logging(args.log)
    t0 = time.time()

    if args.gpu and not _HAS_CUPY:
        logging.warning("CuPy not available; falling back to CPU.")

    pairs_all = load_pairs(args.pairs)
    ann = load_annotations(args.annotations, tuple(args.ann_cols))

    # Drop self pairs if requested
    if args.drop_self:
        pairs_all = pairs_all[pairs_all["ref"] != pairs_all["target"]].copy()

    # Distance filter (optional)
    if args.min_dist > 0:
        before = len(pairs_all)
        pairs_all = add_distance_filter(pairs_all, ann, args.min_dist)
        logging.info("Distance filter kept %d/%d rows (min-dist=%d).", len(pairs_all), before, args.min_dist)

    # Build stubs
    stubs, labels, degrees = build_stub_array(pairs_all, args.deg_metric)
    N = len(labels)
    if stubs.size == 0:
        raise ValueError("Empty stub list; check --deg-metric and input counts.")
    logging.info("Features: %d | Total stubs: %d (edges=%d)", N, stubs.size, stubs.size // 2)

    # Observed undirected index (codes + counts)
    obs_codes, obs_counts = make_observed_index(pairs_all, labels)
    M = len(obs_codes)
    logging.info("Observed undirected pairs (unique): %d", M)

    # Monte Carlo passes
    use_gpu = bool(args.gpu and _HAS_CUPY)
    iters = int(args.iters)
    batch = max(1, int(args.batch))

    logging.info("MC means pass: iters=%d, batch=%d, gpu=%s", iters, batch, use_gpu)
    sim_sum = simulate_means(stubs, N, obs_codes, iters, batch, args.seed or 0, use_gpu)

    logging.info("MC tails pass: iters=%d, batch=%d, gpu=%s", iters, batch, use_gpu)
    ge_obs, le_obs = simulate_tails(stubs, N, obs_codes, obs_counts, iters, batch, (args.seed or 0) + 1337, use_gpu)

    # Expectations & p-values
    expected = sim_sum / float(iters)
    p_enrich = ge_obs / float(iters)
    p_deplete = le_obs / float(iters)
    p_two = np.minimum(2.0 * np.minimum(p_enrich, p_deplete), 1.0)

    # Odds ratio vs expected (with pseudocount)
    pc = float(args.pseudocount)
    odds_ratio = (obs_counts + pc) / (expected + pc)
    log2_or = np.log2(odds_ratio)

    # FDR
    q_enrich = bh_fdr(p_enrich)
    q_deplete = bh_fdr(p_deplete)
    q_two = bh_fdr(p_two)

    # Build undirected results as DataFrame
    labels_arr = np.array(labels, dtype=object)
    u = (obs_codes // N).astype(int)
    v = (obs_codes % N).astype(int)
    undirected = pd.DataFrame({
        "ref_u": labels_arr[u],
        "target_v": labels_arr[v],
        "observed_counts": obs_counts,
        "expected_count": expected,
        "p_enrich": p_enrich,
        "p_deplete": p_deplete,
        "p_twosided": p_two,
        "q_enrich": q_enrich,
        "q_deplete": q_deplete,
        "q_twosided": q_two,
        "odds_ratio": odds_ratio,
        "log2_or": log2_or,
    })
    # canonical key for merge (sorted pair)
    undirected["pair_key"] = undirected.apply(
        lambda r: "_".join(sorted([str(r["ref_u"]), str(r["target_v"])])), axis=1
    )
    undirected = undirected.drop(columns=["ref_u", "target_v"])

    # Merge back into the original (directed) pair table
    df_in = pd.read_csv(args.pairs)
    df_in["pair_key"] = df_in.apply(lambda r: "_".join(sorted([str(r["ref"]), str(r["target"])])), axis=1)
    merged = df_in.merge(undirected, on="pair_key", how="left").drop(columns=["pair_key"])

    # Append coordinates for both partners
    ann_ref = ann.rename(columns={"rna": "ref", "start": "start_ref", "end": "end_ref"})
    ann_tar = ann.rename(columns={"rna": "target", "start": "start_target", "end": "end_target"})
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
            "ann_cols": list(args.ann_cols),
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
