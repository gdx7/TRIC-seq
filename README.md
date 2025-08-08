# TRIC-seq: Mapping the Bacterial RNA–RNA Interactome

TRIC-seq is an **in situ** proximity-ligation approach that captures both intramolecular RNA structure and **trans** RNA–RNA interactions. This repository provides a reproducible, scriptable toolkit to go from raw chimera coordinates to high-confidence interaction maps, null-model enrichment, unsupervised clustering, per-gene inequality/structure summaries, and figure-ready plots.

> **Highlights**
> - Fast chimera→feature mapping with per-pair counts and unique events (`io`)
> - Degree-preserving configuration-model null (CPU/GPU) with **expected counts**, **odds ratios**, **empirical p-values**, **BH-FDR**
> - Leiden clustering (Scanpy) + UMAP/t-SNE
> - Gene-centric global scatter plots (odds ratio vs genomic coordinate)
> - Genome-wide **Gini** inequality of interaction density
> - Long-range maxima profiling (putative **ssRNA accessibility**)
> - Raw / coverage / **ICE**-normalized intra-gene contact maps
> - Every script emits a JSON **manifest** (versions, params, checksums)

---

## Repository Layout

All tools expose a CLI and write a `*.manifest.json` for reproducibility.

| Script | Purpose | Key Outputs |
|---|---|---|
| `tricseq_expand_annotations_and_mask_rrna.py` | Build/clean a compact feature table from GFF3/CSV; optional masking of duplicate rRNA copies in FASTA | `annotations.csv`, masked FASTA, manifest |
| `tricseq_analyze_chimeras.py` | Map chimera ends to features; compute symmetric **pair counts**, **io**, normalized scores, self-interaction metrics | `pairs_counts.csv`, manifest |
| `tricseq_configuration_model.py` | Degree-preserving configuration model with empirical tails; merges into pair table | `pairs_with_mc.csv`, manifest |
| `tricseq_scanpy_clustering.py` | Build weighted RNA×RNA matrix → PCA → kNN → **Leiden**; optional UMAP/t-SNE | `*_clusters.csv`, `.h5ad`, `*_umap_leiden.png`, `*_tsne_leiden.png`, manifest |
| `tricseq_global_interaction_map.py` | Gene-centric **global scatter**: x = partner start, y = interaction strength, size = counts, color = feature type | `<gene>_interactome.pdf` (labeled + unlabeled), `<gene>_labeled_partners.txt`, manifest |
| `tricseq_gini_by_feature.py` | Per-gene interaction inequality (**Gini**) and counts, excluding ±flanks; stratify by feature class | `annotations_with_gini_counts.csv`, manifest |
| `tricseq_ssrna_maxima_longrange.py` | Long-range (>5 kb) **inter-molecular** contacts → 1D profile → local **maxima** (ssRNA proxies); prints sequence + dot-bracket (ViennaRNA) | `<gene>_long_range_structural_features.csv`, `<gene>_1D_long_range_profile.png`, manifest |
| `tricseq_ice_maps.py` | Intra-gene binned matrices: raw, coverage-normalized, **ICE**-normalized; heatmaps for each | `<gene>_bin<B>_{raw,cov,ice}.csv`, corresponding figures, manifest |

---

## Installation

### Conda (recommended)

```bash
conda create -n tricseq python=3.10 -y
conda activate tricseq

# Core
pip install numpy pandas matplotlib seaborn tqdm scikit-learn

# Feature mapping fallback
pip install intervaltree

# Clustering stack
pip install scanpy anndata umap-learn leidenalg

# Bio I/O
pip install biopython

# Optional GPU acceleration for configuration model
# Install a CuPy wheel matching your CUDA, e.g.:
# pip install cupy-cuda12x

# RNA folding (for maxima script)
conda install -c bioconda viennarna -y
# (or system packages / pip wheel where available)
```

Alternatively, if you maintain `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Input Formats

**Chimera interactions**
- **BED**: columns 2 and 3 are the chimera ends (0-based).
- **CSV**: first two columns (or `C1`,`C2`) are the ends.  
Scripts expose `--bed-one-based` / `--csv-one-based` if needed.

**Annotations** (CSV)
```
gene_name,start,end[,feature_type][,strand][,chromosome]
```

**Genome FASTA (optional)**  
Used by the annotation expander/masker and the maxima/structure script.

---

## Quickstart (end-to-end)

1) **Expand annotations (and optionally mask extra rRNA copies)**
```bash
python tricseq_expand_annotations_and_mask_rrna.py   --gff3 path/to/genome.gff3   --fasta path/to/genome.fasta   --out-annotations annotations.csv   --out-masked-fasta genome_masked.fasta
```

2) **Map chimeras → feature pairs & count edges**
```bash
python tricseq_analyze_chimeras.py   --chimera-bed path/to/chimeras.bed   --annotations annotations.csv   --out pairs_counts.csv
```

3) **Configuration-model null → expected counts & odds ratios**
```bash
python tricseq_configuration_model.py   --pairs pairs_counts.csv   --annotations annotations.csv   --out pairs_with_mc.csv   --deg-metric counts   --iters 200000 --batch 50 --min-dist 5000 --seed 42 --gpu 1
```

4) **Unsupervised clustering (Leiden)**
```bash
python tricseq_scanpy_clustering.py   --pairs pairs_with_mc.csv   --annotations annotations.csv   --out-prefix out/cluster   --min-counts 5 --min-distance 3000   --weight-col adjusted_score --score-cap 500   --resolution 1.0 --neighbors 15 --n-pcs 15   --make-umap 1 --make-tsne 1 --random-state 42
```

5) **Figure-ready global map for a focal gene**
```bash
python tricseq_global_interaction_map.py   --pairs pairs_with_mc.csv   --annotations annotations.csv   --gene GcvB   --out-prefix out/GcvB_interactome   --weight-col odds_ratio   --min-counts 5 --min-distance 5000 --y-cap 5000
```

6) **Per-gene inequality (Gini)**
```bash
python tricseq_gini_by_feature.py   --annotations annotations.csv   --contacts data/*.bed   --out annotations_with_gini_counts.csv   --flank 5000 --min-interactions 200 --features CDS ncRNA tRNA
```

7) **Long-range maxima / ssRNA accessibility**
```bash
python tricseq_ssrna_maxima_longrange.py   --annotations annotations.csv   --contacts data/*.bed   --fasta genome.fasta   --gene CsrB   --outdir out/maxima   --flank 5000 --window 3
```

8) **Raw / coverage / ICE-normalized intra-gene maps**
```bash
python tricseq_ice_maps.py   --annotations annotations.csv   --contacts data/*.bed data/*.csv   --genes clpB dnaX   --outdir out/maps   --bin-size 30 --remove-diagonal 1
```

---

## Expected Columns (pair table)

After steps 2–3, the merged pair table typically includes:

- `ref`, `target` — feature IDs (e.g., `gene`, `5'UTR`, `CDS`, `sRNA`)
- `counts` — read-level interaction counts (symmetric; both directions present)
- `io` — unique interaction events (unique coordinate pairs per feature pair)
- `totals`, `total_ref` — incident reads per feature (target/ref perspective)
- `score`, `adjusted_score` — normalized scores (see mapper docstring)
- `ref_type`, `target_type` — from annotation when available
- `self_interactions_ref`, `self_interactions_target`, `self_interaction_score`
- **Configuration-model outputs**: `expected_count`, `odds_ratio`, `log2_or`, `p_enrich`, `p_deplete`, `p_twosided`, `q_*`

---

## Tips & Best Practices

- **Coordinate bases**: BED is usually 0-based; many annotations are 1-based inclusive. Use `--*-one-based` flags where applicable.
- **Distance filters**: `--min-dist` (null) and `--min-distance` (clustering/plots) help remove near-neighbor inflation.
- **GPU acceleration**: `tricseq_configuration_model.py` uses CuPy if available; otherwise run with `--gpu 0`.
- **Reproducibility**: set `--seed` where offered; keep `*.manifest.json` with outputs.
- **Feature overlaps**: pre-resolve overlaps in the expanded annotation step or rely on the mapper’s interval-tree fallback.

---

## Troubleshooting

- **“Weight column not found”** → pass the correct `--weight-col` present in your CSV (e.g., `odds_ratio` after the null model or `adjusted_score`).
- **Too few nodes for clustering** → relax `--min-counts` / `--min-distance` or reduce exclusions (include more feature types).
- **CuPy import error** → install a CUDA-matching wheel (e.g., `cupy-cuda12x`) or run with `--gpu 0`.
- **ViennaRNA not found** → `conda install -c bioconda viennarna` or use your system package manager.

---

## Citation

Please cite the TRIC-seq manuscript/preprint
