import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# 1. Load the Normalized Data
# Ensure this file exists from the previous step
input_file = 'result/result_matrix_normalized.csv'
df = pd.read_csv(input_file)
df = df.set_index('test_taker_id')

def visualize_response_matrix_clustered(df, filename='output/response_matrix_visualization.pdf'):
    """
    Visualizes the response matrix with:
    - Columns: Grouped by benchmark (sorted by difficulty within benchmark).
    - Rows: Clustered by similarity (Hierarchical Clustering) to group similar agents.
    - Output: High-res PDF.
    """
    df_viz = df.copy()

    # --- PART 1: Column Processing (Benchmarks) ---
    # Create cleaner column MultiIndex (Benchmark, TaskID)
    new_columns = []
    for col in df_viz.columns:
        # Heuristic to split "benchmark.taskid" or "benchmark_taskid"
        if '.' in col:
            parts = col.split('.', 1)
            bench, task = parts[0], parts[1]
        else:
            # Fallback for simple names
            bench = col.split('_')[0] if '_' in col else col
            task = col
        new_columns.append((bench, task))
    
    df_viz.columns = pd.MultiIndex.from_tuples(new_columns, names=['benchmark', 'task_id'])

    # Define Benchmark Order & Mapping
    benchmark_order = ['assistantbench', 'taubench', 'corebench',  'swebench', 'gaia', 'scienceagentbench', 'scicode',
                       'usaco', 'colbench']
    
    benchmark_name_map = {
        'assistantbench': 'AssistantBench', 'corebench': 'CORE', 'gaia': 'GAIA',
        'scicode': 'Scicode', 'swebench': 'SWE', 'taubench': 'TAU',
        'usaco': 'USACO', 'colbench': 'COL', 'online': 'Mind2Web',
        'scienceagentbench': 'ScienceAgentBench'
    }

    # Assign Sort Keys for Columns
    benchmark_order_map = {b: i for i, b in enumerate(benchmark_order)}
    
    # Helper to get sort key for a benchmark prefix
    def get_bench_sort_key(bench_name):
        # Remove potential suffixes like "_text"
        base = bench_name.split('_')[0]
        return benchmark_order_map.get(base, 999)

    # Sort Columns: Primary=Benchmark Order, Secondary=Difficulty (Success Rate)
    # We calculate success rate per task first
    task_success_rate = df_viz.mean()
    
    # Create a temporary sorting DataFrame for columns
    col_meta = pd.DataFrame({
        'bench': df_viz.columns.get_level_values('benchmark'),
        'task': df_viz.columns.get_level_values('task_id'),
        'success': task_success_rate.values
    })
    col_meta['bench_sort_key'] = col_meta['bench'].apply(get_bench_sort_key)
    col_meta['bench_display'] = col_meta['bench'].map(lambda x: benchmark_name_map.get(x, benchmark_name_map.get(x.split('_')[0], x)))
    
    # Sort columns by Benchmark Order -> Success Rate (Hard to Easy or Easy to Hard)
    # Ascending success rate = Hardest tasks first (usually looks better)
    sorted_cols_idx = col_meta.sort_values(['bench_sort_key', 'success']).index
    
    # Reorder DataFrame Columns
    df_viz = df_viz.iloc[:, sorted_cols_idx]
    
    # Update Column Index to use Display Names for visualization logic
    new_level = col_meta.iloc[sorted_cols_idx]['bench_display'].values
    df_viz.columns = pd.MultiIndex.from_arrays([new_level, df_viz.columns.get_level_values('task_id')], names=['benchmark', 'task_id'])

    # --- PART 2: Row Processing (Clustering) ---
    print("Performing hierarchical clustering on agents...")
    
    # Prepare data for clustering: 
    # Replace NaN with a distinct value (e.g., 0.5 or -1) so "not attempted" is a feature.
    # Metric: 'euclidean' or 'cityblock' usually works well for 0/1 data.
    cluster_data = df_viz.fillna(0.5) 
    
    # Calculate Linkage Matrix (Ward's method usually gives distinct, tight clusters)
    # Check if we have enough rows
    if len(df_viz) > 1:
        # [MODIFIED] Added optimal_ordering=True
        # This reorganizes the leaves to maximize similarity between adjacent rows
        Z = hierarchy.linkage(cluster_data, method='ward', metric='euclidean', optimal_ordering=True)
        
        # Get the sorted order of leaves
        leaves_order = hierarchy.leaves_list(Z)
        
        # [Optional] Keep your reverse if you prefer top-down ordering
        leaves_order = leaves_order[::-1]
        
        # Reorder Rows
        df_viz = df_viz.iloc[leaves_order]
    else:
        print("Not enough rows to cluster.")

    # --- PART 3: Visualization ---
    
    # Prepare Plot Data (NaN -> -1 for coloring)
    plot_data = df_viz.fillna(-1).values
    
    # Define Colors: -1: White, 0: Red, 1: Blue
    cmap = mcolors.ListedColormap(['white', '#FF4444', '#4444FF']) # Slightly softer Red/Blue
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Setup Figure
    # Height adjusted dynamically based on number of rows to ensure labels fit
    n_rows, n_cols = plot_data.shape 
    fig, ax = plt.subplots(figsize=(28, 14))
    
    # Plot Heatmap
    cax = ax.pcolormesh(range(n_cols + 1), range(n_rows + 1), plot_data, 
                        cmap=cmap, norm=norm, edgecolors='none', linewidth=0)
    
    ax.invert_yaxis() # Top-down
    ax.set_aspect('auto')
    ax.set_xticks([]) # Hide x ticks

    # Add Vertical Dividers for Benchmarks
    benchmarks = df_viz.columns.get_level_values('benchmark')
    unique_benchs = []
    boundaries = []
    
    # Iterate to find boundaries and label positions
    curr_b = benchmarks[0]
    start_idx = 0
    unique_benchs.append(curr_b)
    
    for i, b in enumerate(benchmarks):
        if b != curr_b:
            boundaries.append(i)
            # Add label for previous block
            mid = (start_idx + i) / 2
            ax.text(mid, -1.0, curr_b, ha='center', va='bottom', 
                    rotation=90, fontsize=12, fontweight='bold')
            
            curr_b = b
            start_idx = i
            unique_benchs.append(b)
            
    # Add last label
    mid = (start_idx + len(benchmarks)) / 2
    ax.text(mid, -1.0, curr_b, ha='center', va='bottom', 
            rotation=90, fontsize=12, fontweight='bold')

    # Draw Lines
    for b in boundaries:
        ax.axvline(x=b, color='black', linewidth=0.5, linestyle='--')

    # --- Row Labels (Model Names) ---
    ax.set_yticks(np.arange(n_rows) + 0.5)
    # DECREASED FONT SIZE HERE
    ax.set_yticklabels(df_viz.index, fontsize=2, va='center') 
    
    # Colorbar
    cbar = plt.colorbar(cax, ax=ax, fraction=0.01, pad=0.01)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Not Attempted', 'Incorrect', 'Correct'])

    plt.tight_layout()
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved clustered visualization to {filename}")

# Visualize the result matrix
print("\n" + "="*60)
print("VISUALIZING RESULT MATRIX")
print("="*60)
visualize_response_matrix_clustered(df, filename='result/response_matrix_visualization.pdf')

# Visualize all rubric matrices
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