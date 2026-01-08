import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# 1. Load the Normalized Data
# Ensure this file exists from the previous step
input_file = 'result/result_matrix_merged.csv'
df = pd.read_csv(input_file)
df = df.set_index('test_taker_id')

def visualize_response_matrix_clustered(df, filename='output/response_matrix_visualization.pdf'):
    """
    Visualizes the response matrix with fixed row and column order if provided.
    """
    df_viz = df.copy()

    # If columns are not MultiIndex, create them
    if not isinstance(df_viz.columns, pd.MultiIndex):
        new_columns = []
        for col in df_viz.columns:
            if '.' in col:
                parts = col.split('.', 1)
                bench, task = parts[0], parts[1]
            else:
                bench = col.split('_')[0] if '_' in col else col
                task = col
            new_columns.append((bench, task))
        df_viz.columns = pd.MultiIndex.from_tuples(new_columns, names=['benchmark', 'task_id'])

    # If provided, use global row/col order
    global fixed_row_order, fixed_col_order, fixed_col_display
    if 'fixed_row_order' in globals() and fixed_row_order is not None:
        df_viz = df_viz.iloc[fixed_row_order]
    if 'fixed_col_order' in globals() and fixed_col_order is not None:
        df_viz = df_viz.iloc[:, fixed_col_order]
        # Update display names for columns if provided
        if 'fixed_col_display' in globals() and fixed_col_display is not None:
            df_viz.columns = pd.MultiIndex.from_arrays([fixed_col_display, df_viz.columns.get_level_values('task_id')], names=['benchmark', 'task_id'])

    # --- Visualization ---
    plot_data = df_viz.fillna(-1).values
    cmap = mcolors.ListedColormap(['white', '#FF4444', '#4444FF'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    n_rows, n_cols = plot_data.shape
    fig, ax = plt.subplots(figsize=(28, 14))
    cax = ax.pcolormesh(range(n_cols + 1), range(n_rows + 1), plot_data,
                        cmap=cmap, norm=norm, edgecolors='none', linewidth=0)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xticks([])

    # Add Vertical Dividers for Benchmarks
    benchmarks = df_viz.columns.get_level_values('benchmark')
    unique_benchs = []
    boundaries = []
    curr_b = benchmarks[0]
    start_idx = 0
    unique_benchs.append(curr_b)
    for i, b in enumerate(benchmarks):
        if b != curr_b:
            boundaries.append(i)
            mid = (start_idx + i) / 2
            ax.text(mid, -1.0, curr_b, ha='center', va='bottom',
                    rotation=90, fontsize=12, fontweight='bold')
            curr_b = b
            start_idx = i
            unique_benchs.append(b)
    mid = (start_idx + len(benchmarks)) / 2
    ax.text(mid, -1.0, curr_b, ha='center', va='bottom',
            rotation=90, fontsize=12, fontweight='bold')
    for b in boundaries:
        ax.axvline(x=b, color='black', linewidth=0.5, linestyle='--')

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(df_viz.index, fontsize=2, va='center')
    cbar = plt.colorbar(cax, ax=ax, fraction=0.01, pad=0.01)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Not Attempted', 'Incorrect', 'Correct'])
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved clustered visualization to {filename}")

# Visualize the result matrix
print("\n" + "="*60)
print("VISUALIZING RESULT MATRIX")
print("="*60)

# --- Compute fixed row/col order from result matrix ---
df_viz = df.copy()
# Columns: MultiIndex
new_columns = []
for col in df_viz.columns:
    if '.' in col:
        parts = col.split('.', 1)
        bench, task = parts[0], parts[1]
    else:
        bench = col.split('_')[0] if '_' in col else col
        task = col
    new_columns.append((bench, task))
df_viz.columns = pd.MultiIndex.from_tuples(new_columns, names=['benchmark', 'task_id'])

benchmark_order = ['assistantbench', 'taubench', 'corebench',  'swebench', 'gaia', 'scienceagentbench', 'scicode',
                   'usaco', 'colbench']
benchmark_name_map = {
    'assistantbench': 'AssistantBench', 'corebench': 'CORE', 'gaia': 'GAIA',
    'scicode': 'Scicode', 'swebench': 'SWE', 'taubench': 'TAU',
    'usaco': 'USACO', 'colbench': 'COL', 'online': 'Mind2Web',
    'scienceagentbench': 'ScienceAgentBench'
}
benchmark_order_map = {b: i for i, b in enumerate(benchmark_order)}
def get_bench_sort_key(bench_name):
    base = bench_name.split('_')[0]
    return benchmark_order_map.get(base, 999)
task_success_rate = df_viz.mean()
col_meta = pd.DataFrame({
    'bench': df_viz.columns.get_level_values('benchmark'),
    'task': df_viz.columns.get_level_values('task_id'),
    'success': task_success_rate.values
})
col_meta['bench_sort_key'] = col_meta['bench'].apply(get_bench_sort_key)
col_meta['bench_display'] = col_meta['bench'].map(lambda x: benchmark_name_map.get(x, benchmark_name_map.get(x.split('_')[0], x)))
sorted_cols_idx = col_meta.sort_values(['bench_sort_key', 'success']).index
fixed_col_order = sorted_cols_idx
fixed_col_display = col_meta.iloc[sorted_cols_idx]['bench_display'].values
df_viz = df_viz.iloc[:, fixed_col_order]
df_viz.columns = pd.MultiIndex.from_arrays([fixed_col_display, df_viz.columns.get_level_values('task_id')], names=['benchmark', 'task_id'])

# Rows: clustering
cluster_data = df_viz.fillna(0.5)
if len(df_viz) > 1:
    Z = hierarchy.linkage(cluster_data, method='ward', metric='euclidean', optimal_ordering=True)
    leaves_order = hierarchy.leaves_list(Z)[::-1]
    fixed_row_order = leaves_order
    df_viz = df_viz.iloc[fixed_row_order]
else:
    fixed_row_order = None

# Save result matrix visualization
visualize_response_matrix_clustered(df, filename='result/response_matrix_visualization.pdf')

# Visualize all rubric matrices with fixed order
print("\n" + "="*60)
print("VISUALIZING RUBRIC MATRICES")
print("="*60)
rubric_types = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
for rubric_type in rubric_types:
    print(f"\nProcessing {rubric_type}...")
    try:
        rubric_file = f'rubrics/rubrics_matrix_{rubric_type}.csv'
        rubric_df = pd.read_csv(rubric_file)
        rubric_df = rubric_df.set_index('test_taker_id')
        output_file = f'result/rubrics_{rubric_type}_visualization.pdf'
        visualize_response_matrix_clustered(rubric_df, filename=output_file)
    except Exception as e:
        print(f"  Error processing {rubric_type}: {e}")
print("\n" + "="*60)
print("✓ ALL VISUALIZATIONS COMPLETED")
print("="*60)