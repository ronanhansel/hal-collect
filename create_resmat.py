import pandas as pd
import pickle
import os
from pathlib import Path

# Read the merged results
merged_df = pd.read_csv('hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv')

# Read the inputs for text_input
inputs_df = pickle.load(open('data/all_benchmarks_inputs.pkl', 'rb'))

# Create a mapping from (task_id, benchmark_id) to text_input (taking the first occurrence)
# Since the same task can have different text_inputs for different models, we pick one representative
inputs_mapping = {}
for _, row in inputs_df.iterrows():
    key = (row['task_id'], row['benchmark_id'])
    if key not in inputs_mapping:  # Take the first one
        inputs_mapping[key] = row['task_input']

# Normalize scaffold names (remove extra spaces, standardize)
def normalize_scaffold(scaffold):
    if pd.isna(scaffold):
        return 'unknown'
    return scaffold.strip().lower().replace(' ', '_')

merged_df['scaffold_normalized'] = merged_df['scaffold'].apply(normalize_scaffold)

# Create row index: scaffold_normalized + model
merged_df['row_index'] = merged_df['scaffold_normalized'] + '_' + merged_df['model']

# Define the columns to create resmats for
columns_to_process = [
    'binary_success_rate',
    'environmentalbarrier.label',
    'instructionfollowing.label',
    'selfcorrection.label',
    'tooluse.label',
    'verification.label'
]

# Function to convert values to binary
def to_binary(value):
    if pd.isna(value) or value == '':
        return None
    if value == 'match' or value == True or value == 'True':
        return 1
    if value == 'no match' or value == False or value == 'False':
        return 0
    # For binary_success_rate which is already numeric
    try:
        return int(float(value))
    except:
        return None

# Create output directory
output_dir = Path('data')
output_dir.mkdir(exist_ok=True)

# Process each column
for col in columns_to_process:
    print(f"Processing {col}...")
    
    # Filter out rows where the column value exists
    df_filtered = merged_df[merged_df[col].notna()].copy()
    
    if col == 'binary_success_rate':
        # Add text_input for binary_success_rate (one per task_id, benchmark_id)
        df_filtered['text_input'] = df_filtered.apply(
            lambda row: inputs_mapping.get(
                (row['task_id'], row['benchmark_id']), 
                ''
            ),
            axis=1
        )
        
        # Create MultiIndex columns: (task_id, text_input, benchmark)
        df_filtered['col_tuple'] = df_filtered.apply(
            lambda row: (row['task_id'], row['text_input'], row['benchmark_id']),
            axis=1
        )
    else:
        # Create MultiIndex columns: (task_id, benchmark)
        df_filtered['col_tuple'] = df_filtered.apply(
            lambda row: (row['task_id'], row['benchmark_id']),
            axis=1
        )
    
    # Convert values to binary
    df_filtered['binary_value'] = df_filtered[col].apply(to_binary)
    
    # Pivot to create the resmat
    if col == 'binary_success_rate':
        # For binary_success_rate, we need to handle the 3-level MultiIndex
        pivot_data = []
        for row_idx in df_filtered['row_index'].unique():
            row_data = {'row_index': row_idx}
            subset = df_filtered[df_filtered['row_index'] == row_idx]
            for _, row in subset.iterrows():
                row_data[row['col_tuple']] = row['binary_value']
            pivot_data.append(row_data)
        
        resmat = pd.DataFrame(pivot_data).set_index('row_index')
        
        # Convert columns to MultiIndex
        if len(resmat.columns) > 0:
            resmat.columns = pd.MultiIndex.from_tuples(
                resmat.columns,
                names=['task_id', 'text_input', 'benchmark']
            )
    else:
        # For other columns, use regular pivot
        pivot_data = []
        for row_idx in df_filtered['row_index'].unique():
            row_data = {'row_index': row_idx}
            subset = df_filtered[df_filtered['row_index'] == row_idx]
            for _, row in subset.iterrows():
                row_data[row['col_tuple']] = row['binary_value']
            pivot_data.append(row_data)
        
        resmat = pd.DataFrame(pivot_data).set_index('row_index')
        
        # Convert columns to MultiIndex
        if len(resmat.columns) > 0:
            resmat.columns = pd.MultiIndex.from_tuples(
                resmat.columns,
                names=['task_id', 'benchmark']
            )
    
    # Sort rows and columns
    resmat = resmat.sort_index()
    resmat = resmat.reindex(sorted(resmat.columns), axis=1)
    
    # Save to pickle
    output_path = output_dir / f'resmat_{col}.pkl'
    resmat.to_pickle(output_path)
    print(f"Saved {output_path} with shape {resmat.shape}")
    print(f"  Rows: {len(resmat)}, Columns: {len(resmat.columns)}")
    print()

print("All resmats created successfully!")
