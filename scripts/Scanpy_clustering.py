import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Perform unsupervised clustering on the RNA interactome.")
    parser.add_argument('--interaction_file', required=True, help="Final interaction file with MC stats.")
    parser.add_argument('--annotation_file', required=True, help="Comprehensive annotation file.")
    parser.add_argument('--output_prefix', required=True, help="Prefix for output files.")
    args = parser.parse_args()

    print("Reading and filtering interaction data...")
    interactions_df = pd.read_csv(args.interaction_file)
    annotation_df = pd.read_csv(args.annotation_file)
    
    filtered_df = interactions_df[interactions_df['counts'] >= 5]

    rna_list = sorted(list(set(filtered_df['ref']).union(set(filtered_df['target']))))
    rna_to_idx = {rna: idx for idx, rna in enumerate(rna_list)}

    print("Creating sparse interaction matrix...")
    row_indices = filtered_df['ref'].map(rna_to_idx)
    col_indices = filtered_df['target'].map(rna_to_idx)
    data = filtered_df['adjusted_score'].astype(float)
    
    interaction_matrix_sparse = csr_matrix(
        (np.concatenate([data, data]), (np.concatenate([row_indices, col_indices]), np.concatenate([col_indices, row_indices]))),
        shape=(len(rna_list), len(rna_list))
    )

    print("Creating AnnData object and running clustering pipeline...")
    adata = sc.AnnData(X=interaction_matrix_sparse, obs=pd.DataFrame(index=rna_list))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)
    sc.tl.leiden(adata, resolution=1.0)
    sc.tl.tsne(adata, n_pcs=15)
    
    print("Saving cluster assignments and plots...")
    clusters_df = adata.obs[['leiden']]
    clusters_df.to_csv(f"{args.output_prefix}_rna_clusters.csv")
    adata.write(f"{args.output_prefix}_adata.h5ad")
    
    sc.pl.tsne(adata, color='leiden', save=f"_{args.output_prefix}_leiden.pdf", show=False)

if __name__ == "__main__":
    main()

