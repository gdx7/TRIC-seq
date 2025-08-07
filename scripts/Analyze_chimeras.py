import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Map chimeric reads to genomic features and calculate interaction scores.")
    parser.add_argument('--interactions', required=True, help="Path to the input chimeric reads BED file.")
    parser.add_argument('--annotations', required=True, help="Path to the comprehensive annotation CSV file.")
    parser.add_argument('--output', required=True, help="Path for the output interaction analysis CSV file.")
    args = parser.parse_args()

    print("Reading RNA-RNA interaction data...")
    df_interactions = pd.read_csv(args.interactions, sep='\t', header=None, names=['Chromosome', 'Start', 'End'])
    
    print("Reading annotations data...")
    df_annotations = pd.read_csv(args.annotations)
    
    print("Creating position-to-range mapping...")
    genome_length = int(df_annotations['End'].max())
    position_to_range = np.empty(genome_length + 2, dtype=object)
    
    for _, row in df_annotations.iterrows():
        start, end, range_name = int(row['Start']), int(row['End']), row['RNA']
        if start < end:
            position_to_range[start:end+1] = range_name

    print("Mapping interaction positions to ranges...")
    df_interactions['Start'] = df_interactions['Start'].astype(int).clip(0, genome_length)
    df_interactions['End'] = df_interactions['End'].astype(int).clip(0, genome_length)
    df_interactions['Start_Range'] = position_to_range[df_interactions['Start'].values]
    df_interactions['End_Range'] = position_to_range[df_interactions['End'].values]
    
    df_interactions.dropna(subset=['Start_Range', 'End_Range'], how='all', inplace=True)
    df_interactions['Interaction_Index'] = df_interactions.index

    print("Calculating total interactions involving each feature...")
    feature_interactions = pd.concat([
        df_interactions[['Start_Range', 'Interaction_Index']].rename(columns={'Start_Range': 'Feature'}),
        df_interactions[['End_Range', 'Interaction_Index']].rename(columns={'End_Range': 'Feature'})
    ])
    feature_totals = feature_interactions.groupby('Feature').size().reset_index(name='total_interactions')
    feature_totals_dict = dict(zip(feature_totals['Feature'], feature_totals['total_interactions']))

    print("Calculating self-interactions per feature...")
    self_interactions_df = df_interactions[df_interactions['Start_Range'] == df_interactions['End_Range']]
    self_interactions_counts = self_interactions_df.groupby('Start_Range').size().reset_index(name='self_interactions')
    self_interactions_dict = dict(zip(self_interactions_counts['Start_Range'], self_interactions_counts['self_interactions']))

    total_interactions_in_dataset = df_interactions.shape[0]
    print(f"Total interactions in the dataset: {total_interactions_in_dataset}")

    print("Counting interactions between features...")
    df_interactions['pair'] = df_interactions.apply(lambda row: tuple(sorted([row['Start_Range'], row['End_Range']])), axis=1)
    symmetric_counts = df_interactions.groupby('pair').size().reset_index(name='counts')
    symmetric_counts[['sorted_ref', 'sorted_target']] = pd.DataFrame(symmetric_counts['pair'].tolist(), index=symmetric_counts.index)
    symmetric_counts.drop(columns='pair', inplace=True)

    final_rows = []
    for _, row in symmetric_counts.iterrows():
        if row['sorted_ref'] == row['sorted_target']:
            final_rows.append({'ref': row['sorted_ref'], 'target': row['sorted_target'], 'counts': row['counts']})
        else:
            final_rows.append({'ref': row['sorted_ref'], 'target': row['sorted_target'], 'counts': row['counts']})
            final_rows.append({'ref': row['sorted_target'], 'target': row['sorted_ref'], 'counts': row['counts']})
    interaction_counts_df = pd.DataFrame(final_rows)

    print("Mapping totals and self-interactions...")
    interaction_counts_df['totals'] = interaction_counts_df['target'].map(feature_totals_dict)
    interaction_counts_df['total_ref'] = interaction_counts_df['ref'].map(feature_totals_dict)
    interaction_counts_df['self_interactions_ref'] = interaction_counts_df['ref'].map(self_interactions_dict).fillna(0)
    interaction_counts_df['self_interactions_target'] = interaction_counts_df['target'].map(self_interactions_dict).fillna(0)

    print("Calculating enrichment scores...")
    interaction_counts_df['score'] = (interaction_counts_df['counts'] * total_interactions_in_dataset) / \
                                     (interaction_counts_df['totals'] * interaction_counts_df['total_ref'])
    interaction_counts_df['score'] = interaction_counts_df['score'].replace([np.inf, -np.inf], np.nan).fillna(0)

    interaction_counts_df['adjusted_totals'] = interaction_counts_df['totals'] - interaction_counts_df['self_interactions_target']
    interaction_counts_df['adjusted_total_ref'] = interaction_counts_df['total_ref'] - interaction_counts_df['self_interactions_ref']
    interaction_counts_df['adjusted_totals'] = interaction_counts_df['adjusted_totals'].apply(lambda x: x if x > 0 else np.nan)
    interaction_counts_df['adjusted_total_ref'] = interaction_counts_df['adjusted_total_ref'].apply(lambda x: x if x > 0 else np.nan)
    
    interaction_counts_df['adjusted_score'] = (interaction_counts_df['counts'] * total_interactions_in_dataset) / \
                                              (interaction_counts_df['adjusted_totals'] * interaction_counts_df['adjusted_total_ref'])
    interaction_counts_df['adjusted_score'] = interaction_counts_df['adjusted_score'].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("Adding feature types...")
    feature_type_dict = dict(zip(df_annotations['RNA'], df_annotations['Type']))
    interaction_counts_df['ref_type'] = interaction_counts_df['ref'].map(feature_type_dict)
    interaction_counts_df['target_type'] = interaction_counts_df['target'].map(feature_type_dict)

    self_interaction_rows = interaction_counts_df['ref'] == interaction_counts_df['target']
    interaction_counts_df['self_interaction_score'] = np.nan
    interaction_counts_df.loc[self_interaction_rows, 'self_interaction_score'] = \
        interaction_counts_df.loc[self_interaction_rows, 'self_interactions_ref'] / \
        (interaction_counts_df.loc[self_interaction_rows, 'totals'] - interaction_counts_df.loc[self_interaction_rows, 'self_interactions_ref'])
    interaction_counts_df['self_interaction_score'] = interaction_counts_df['self_interaction_score'].replace([np.inf, -np.inf], np.nan).fillna(0)

    final_df = interaction_counts_df[['ref', 'target', 'counts', 'total_ref', 'totals', 'score',
                                      'adjusted_score', 'ref_type', 'target_type', 'self_interaction_score']]
    
    final_df = final_df.sort_values(by=['adjusted_score', 'score'], ascending=False)
    
    print(f"Interaction analysis saved to {args.output}")
    final_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
