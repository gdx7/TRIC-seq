TRIC-seq: Mapping RNA–RNA Interactomes in Bacteria
TRIC-seq is an in situ proximity ligation approach for charting the bacterial RNA–RNA interactome. This repository provides a lightweight, reproducible analysis toolkit to go from raw chimera BED/CSV files to high-confidence interaction maps, null-model enrichment, clustering, and publication-ready figures.

Highlights

Degree-preserving configuration-model null (GPU-accelerated option)

Per-pair odds ratios, empirical p-values, BH-FDR

Unsupervised Leiden clustering of the interactome

Global, gene-centric interaction maps

Per-gene inequality (Gini) metrics

Long-range maxima profiling (ssRNA accessibility proxy)

Raw / coverage / ICE-normalized contact maps

Repository layout
Scripts are CLI tools; each writes a JSON manifest with run metadata for reproducibility.

tricseq_expand_annotations_and_mask_rrna.py
Expand/clean annotations from GFF3/CSV and output a compact feature table; optionally mask duplicate rRNA copies in the FASTA.

tricseq_analyze_chimeras.py
Map chimera ends to features; compute pairwise counts, unique interaction counts (io), normalized scores, and self-interaction metrics.

tricseq_configuration_model.py
Degree-preserving configuration-model simulation to get expected counts, odds ratios, empirical p-values, and BH-FDR; merges into the pair table.

tricseq_scanpy_clustering.py
Build a weighted RNA×RNA matrix from filtered pairs; PCA → kNN → Leiden clustering; outputs .h5ad and cluster CSV plus optional UMAP/t-SNE PNGs.

tricseq_global_interaction_map.py
Gene-centric global scatter: x = partner genomic start, y = interaction strength (e.g., odds_ratio), size = counts, color = feature type. Saves labeled + unlabeled PDF pages.

tricseq_gini_by_feature.py
Genome-wide Gini inequality of per-nucleotide interaction density and interaction counts for CDS/ncRNA/tRNA, excluding ±flanks.

tricseq_ssrna_maxima_longrange.py
Long-range (>5 kb) inter-molecular contacts → 1D profile of ligation events within a target RNA → local maxima as putative ssRNA sites; exports CSV and a profile plot; prints MFE structure (ViennaRNA).

tricseq_ice_maps.py
Intra-gene contact matrices binned along a target RNA: raw, coverage-normalized, and ICE-normalized (+ heatmaps).

If your filenames differ slightly, update the examples below accordingly.


Expand annotations (and optional rRNA masking)
python tricseq_expand_annotations_and_mask_rrna.py \
  --gff3 path/to/genome.gff3 \
  --fasta path/to/genome.fasta \
  --out-annotations annotations.csv \
  --out-masked-fasta genome_masked.fasta

Map chimera ends to features & count pairs
python tricseq_analyze_chimeras.py \
  --chimera-bed path/to/chimeras.bed \
  --annotations annotations.csv \
  --out pairs_counts.csv

Null model (configuration) → expected counts & odds ratios
python tricseq_configuration_model.py \
  --pairs pairs_counts.csv \
  --annotations annotations.csv \
  --out pairs_with_mc.csv \
  --deg-metric counts \
  --iters 200000 --batch 50 --min-dist 5000 --seed 42 --gpu 1

Unsupervised clustering (Leiden)
python tricseq_scanpy_clustering.py \
  --pairs pairs_with_mc.csv \
  --annotations annotations.csv \
  --out-prefix out/cluster \
  --min-counts 5 --min-distance 3000 \
  --weight-col adjusted_score --score-cap 500 \
  --resolution 1.0 --neighbors 15 --n-pcs 15 \
  --make-umap 1 --make-tsne 1 --random-state 42

Global map for a focal gene (publication-style scatter)
python tricseq_global_interaction_map.py \
  --pairs pairs_with_mc.csv \
  --annotations annotations.csv \
  --gene GcvB \
  --out-prefix out/GcvB_interactome \
  --weight-col odds_ratio \
  --min-counts 5 --min-distance 5000 --y-cap 5000

Per-gene inequality (Gini)
python tricseq_gini_by_feature.py \
  --annotations annotations.csv \
  --contacts path/to/*.bed \
  --out annotations_with_gini_counts.csv \
  --flank 5000 --min-interactions 200 --features CDS ncRNA tRNA

Long-range maxima / accessibility (optional)
python tricseq_ssrna_maxima_longrange.py \
  --annotations annotations.csv \
  --contacts path/to/*.bed \
  --fasta genome.fasta \
  --gene CsrB \
  --outdir out/maxima \
  --flank 5000 --window 3

Raw / coverage / ICE-normalized contact maps (per gene)
python tricseq_ice_maps.py \
  --annotations annotations.csv \
  --contacts path/to/*.bed \
  --genes clpB dnaX \
  --outdir out/maps \
  --bin-size 30 --remove-diagonal 1
All scripts emit a *.manifest.json with versions, params, and checksums.

Columns & outputs (key tables)
Pair table (pairs_counts.csv → pairs_with_mc.csv)
ref, target: feature IDs (e.g., gene, 5'UTR, CDS, sRNA)

counts: read-level interaction counts (symmetric, duplicated both directions)

io: unique interaction events (unique coordinate pairs per feature pair)

totals, total_ref: total incident reads per feature

score, adjusted_score: normalized scores (see script docstring)

ref_type, target_type: from annotation feature_type if available

self_interactions_ref, self_interactions_target, self_interaction_score

(after configuration model) expected_count, odds_ratio, log2_or, p_*, q_*

Clustering outputs
*_clusters.csv: feature → Leiden cluster id

*.h5ad: Scanpy object with PCA/neighbor graph/embeddings

*_umap_leiden.png, *_tsne_leiden.png: overview figures

Global scatter
*_interactome.pdf: two pages (labeled & unlabeled)

*_labeled_partners.txt: names auto-labeled in the figure

Gini
annotations_with_gini_counts.csv: per-feature N_interactions, Gini

Maxima
<gene>_long_range_structural_features.csv: maxima positions & sequence context

<gene>_1D_long_range_profile.png: bar+smoothed profile with peak markers

ICE maps
<gene>_bin<bin>_raw.csv, _cov.csv, _ice.csv + corresponding figures
