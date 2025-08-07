# TRIC-seq Analysis Pipeline

This repository contains the Python scripts used to process and analyze the data from the TRIC-seq manuscript, "Comprehensive architecture of the bacterial RNA Interactome." The pipeline allows users to go from raw genomic feature files and interaction data to statistically significant interaction lists, structural analyses, and network visualizations.

## Requirements

The scripts are written in Python 3. The following libraries are required and can be installed via pip:

```bash
pip install pandas numpy biopython intervaltree scipy scanpy viennarna seaborn matplotlib
```

Alternatively, you can create an environment using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Analysis Workflow

The analysis is performed in a sequential manner. The output of one script often serves as the input for the next.

### 1. Annotation Expansion (`Expand_annotations.py`)

This script takes a standard GFF3 annotation file and a genome FASTA file and generates a comprehensive, gap-less annotation file. This is a critical first step for the unbiased, *de novo* discovery of RNA features.

**Usage:**
```bash
python Expand_annotations.py \
  --gff3 <path/to/genome.gff3> \
  --fasta <path/to/genome.fasta> \
  --output_csv <path/to/output_annotations.csv> \
  --output_fasta <path/to/output_modified.fasta> \
  --gap_cutoff 40
```

---

### 2. Chimera Analysis (`Analyze_chimeras.py`)

This script takes the raw chimeric read file (in BED format) and the comprehensive annotation file generated in Step 1. It maps the ends of each chimera to their corresponding genomic features and calculates raw interaction counts and normalized enrichment scores.

**Usage:**
```bash
python Analyze_chimeras.py \
  --interactions <path/to/chimeric_reads.bed> \
  --annotations <path/to/output_annotations.csv> \
  --output <path/to/interaction_analysis.csv>
```

---

### 3. Configuration Model Simulation (`Configuration_model.py`)

This script takes the interaction analysis file from Step 2 and performs a Monte Carlo simulation based on a configuration model to determine the statistical significance of long-range (*trans*) interactions.

**Usage:**
```bash
python Configuration_model.py \
  --input_file <path/to/interaction_analysis.csv> \
  --annotations <path/to/output_annotations.csv> \
  --output_file <path/to/analysis_with_MC.csv> \
  --simulations 1000000
```

*Note: This script can be computationally intensive.*

---

### 4. Gini Coefficient Calculation (`Gini coefficient.py`)

This script calculates the Gini coefficient for all annotated features based on their long-range interaction profiles, providing a measure of structural heterogeneity.

**Usage:**
```bash
python Gini coefficient.py \
  --annotations <path/to/output_annotations.csv> \
  --interactions_dir <path/to/interaction_files/> \
  --output_file <path/to/annotations_with_gini.csv> \
  --flank 5000
```

---

### 5. Network Clustering and Visualization (`Scanpy_clustering.py`)

This script takes the final, statistically processed interaction file from Step 3 and performs unsupervised clustering to identify modules within the RNA interactome.

**Usage:**
```bash
python Scanpy_clustering.py \
  --interaction_file <path/to/analysis_with_MC.csv> \
  --output_prefix <your_prefix>
```

This will generate several output files, including `rna_clusters.csv` (cluster assignments) and various PDF plots.
