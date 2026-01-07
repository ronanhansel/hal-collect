import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add tools to path for shared naming utilities
sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from naming import generate_test_taker_id

# Load the dataset
# Ensure the directory 'result' exists or update path as needed
file_path = 'result/result_matrix.csv' 
df = pd.read_csv(file_path)

def process_row(row):
    """
    Generates the unique test_taker_id using format: scaffold:model_effort
    """
    return generate_test_taker_id(row['agent_name'], row['model_name'])

# --- Execution ---

df['test_taker_id'] = df.apply(process_row, axis=1)

# Select columns to keep (ID + numeric benchmarks)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_keep = ['test_taker_id'] + numeric_cols

# Filter and Merge
df_subset = df[cols_to_keep]
merged_df = df_subset.groupby('test_taker_id').max().reset_index()

# Save
output_file = 'result/result_matrix_merged.csv'
merged_df.to_csv(output_file, index=False)

print(f"Processing complete. Data saved to {output_file}")
print(f"Total unique IDs: {len(merged_df)}")
print("IDs", merged_df['test_taker_id'].tolist())