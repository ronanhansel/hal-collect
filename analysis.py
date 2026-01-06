import pandas as pd
df = pd.read_pickle('resmat/response_matrix_merged.pkl')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

def visualize_response_matrix(df, filename='response_matrix_visualization.png'):
    """
    Visualize response matrix with:
    - Red: incorrect (0)
    - Blue: correct (1)  
    - White: NaN
    - Grouped by benchmark with vertical lines
    """
    # Create a copy of the dataframe and group benchmarks by prefix
    df_viz = df.copy()
    
    # Extract benchmark prefixes (before first underscore)
    benchmark_names = df_viz.columns.get_level_values("benchmark")
    benchmark_prefixes = [b.split('_')[0] if '_' in b else b for b in benchmark_names]
    
    # Define the desired order for benchmarks
    benchmark_order = ['assistantbench', 'corebench', 'gaia', 'scicode', 'swebench', 
                       'taubench', 'usaco', 'colbench', 'online']
    
    # Map benchmark names to more legible versions
    benchmark_name_map = {
        'assistantbench': 'AssistantBench',
        'corebench': 'CORE',
        'gaia': 'GAIA',
        'scicode': 'Scicode',
        'swebench': 'SWE',
        'taubench': 'TAU',
        'usaco': 'USACO',
        'colbench': 'COL',
        'online': 'Mind2Web'
    }
    
    # Create sort keys for each column based on the desired order
    benchmark_order_map = {bench: idx for idx, bench in enumerate(benchmark_order)}
    sort_keys = [benchmark_order_map.get(prefix, 999) for prefix in benchmark_prefixes]
    
    # Create new MultiIndex with sort keys and mapped names
    task_ids = df_viz.columns.get_level_values("task_id")
    benchmark_prefixes_mapped = [benchmark_name_map.get(b, b) for b in benchmark_prefixes]
    
    new_columns = pd.MultiIndex.from_tuples(
        list(zip(task_ids, sort_keys, benchmark_prefixes_mapped)), 
        names=['task_id', 'sort_key', 'benchmark']
    )
    df_viz.columns = new_columns
    
    # Sort by the sort_key to get the desired order
    df_viz = df_viz.sort_index(axis=1, level=['sort_key', 'task_id'])
    
    # Drop the sort_key level, keeping only task_id and benchmark (display name)
    df_viz.columns = df_viz.columns.droplevel('sort_key')
    
    # Balanced seriation: Sort to reduce sparsity while maintaining structure
    
    # For columns: Sort within each benchmark by difficulty (success rate)
    # This creates red (hard) to blue (easy) gradient
    benchmark_values = df_viz.columns.get_level_values("benchmark")
    col_order = []
    
    for bench in pd.unique(benchmark_values):
        bench_mask = benchmark_values == bench
        bench_cols = df_viz.loc[:, bench_mask]
        
        # Calculate success rate for each task (ignoring NaN)
        success_rates = bench_cols.sum() / bench_cols.notna().sum()
        # Sort by success rate ascending (hardest tasks first)
        sorted_cols = success_rates.sort_values().index
        col_order.extend(sorted_cols)
    
    df_viz = df_viz[col_order]
    
    # For rows: Hybrid approach for balanced density
    # Primary sort: number of tasks attempted (coverage) - descending to put dense rows on top
    # Secondary sort: success rate among attempted tasks - ascending for weak to strong
    tasks_attempted = df_viz.notna().sum(axis=1)
    success_rate = df_viz.sum(axis=1) / df_viz.notna().sum(axis=1)
    
    # Create sorting dataframe
    sort_df = pd.DataFrame({
        'coverage': -tasks_attempted,  # Negative for descending (dense first)
        'success_rate': success_rate
    })
    
    # Sort: dense rows (high coverage) first, then by success rate (weak to strong)
    df_viz = df_viz.loc[sort_df.sort_values(['coverage', 'success_rate']).index]
    
    # Extract the benchmark labels from MultiIndex columns
    benchmark_values = df_viz.columns.get_level_values("benchmark")
    
    # Identify the boundaries where the benchmark changes
    boundaries = []
    for i in range(1, len(benchmark_values)):
        if benchmark_values[i] != benchmark_values[i - 1]:
            boundaries.append(i - 0.5)  # Place line between columns
    
    # Create colormap: white is NaN, red is 0, blue is 1
    # We'll use -1 for NaN in visualization
    value = df_viz.copy()
    value = value.fillna(-1)  # Replace NaN with -1 for visualization
    
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Calculate midpoints for each benchmark group label
    benchmarks_list = list(benchmark_values)
    benchmark_names_list = []
    benchmark_midpoints = []
    current_benchmark = benchmarks_list[0]
    start_index = 0
    
    for i, bench in enumerate(benchmarks_list):
        if bench != current_benchmark:
            midpoint = (start_index + i - 1) / 2.0
            benchmark_names_list.append(current_benchmark)
            benchmark_midpoints.append(midpoint)
            current_benchmark = bench
            start_index = i
    
    # Add the last benchmark group
    midpoint = (start_index + len(benchmarks_list) - 1) / 2.0
    benchmark_names_list.append(current_benchmark)
    benchmark_midpoints.append(midpoint)
    
    # Plot the matrix with increased height for better readability
    fig, ax = plt.subplots(figsize=(28, 14))
    
    # Use pcolormesh instead of matshow for vector output (sharp when zoomed)
    num_rows, num_cols = value.shape
    cax = ax.pcolormesh(range(num_cols + 1), range(num_rows + 1), value.values, 
                        cmap=cmap, norm=norm, edgecolors='none', linewidth=0)
    
    # Invert y-axis to match matshow behavior (0 at top)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    
    # Add vertical dotted lines at each boundary
    for b in boundaries:
        ax.axvline(x=b, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    
    # Add benchmark labels above the matrix at midpoints
    for name, pos in zip(benchmark_names_list, benchmark_midpoints):
        ax.text(pos + 0.5, -1.5, name, ha='center', va='bottom', rotation=90, fontsize=10, fontweight='bold')
    
    # Add full agent labels on the y-axis (shift by 0.5 for pcolormesh centering)
    agent_labels = df_viz.index.tolist()
    ax.set_yticks([i + 0.5 for i in range(len(df_viz.index))])
    ax.set_yticklabels(agent_labels, fontsize=2)
    
    # Remove x-axis tick labels (we only show benchmark names)
    ax.set_xticks([])
    
    # Add a colorbar
    cbar = plt.colorbar(cax, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Not Attempted", "Incorrect", "Correct"])
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved visualization to {filename}")
    print(f"Total benchmarks: {len(benchmark_names_list)}")
    print(f"Benchmark groups: {benchmark_names_list}")
    plt.show()

# Visualize the merged response matrix
visualize_response_matrix(df, 'output/response_matrix_visualization.pdf')
# Diagnostic: Check the structure of the dataframe
print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.index[:5])
print("\nFirst few columns:")
print(df.columns[:10])
print("\nUnique benchmarks:")
print(df.columns.get_level_values("benchmark").unique())
print("\nBenchmark value counts:")
print(df.columns.get_level_values("benchmark").value_counts())
# Check benchmark grouping by prefix
benchmark_names = df.columns.get_level_values("benchmark")
print("Sample benchmark names:", benchmark_names[:20].tolist())

# Extract benchmark prefixes (before first underscore)
benchmark_prefixes = [b.split('_')[0] if '_' in b else b for b in benchmark_names]
print("\nUnique benchmark prefixes:")
print(set(benchmark_prefixes))

# Check how many of each prefix
from collections import Counter
print("\nPrefix counts:")
for prefix, count in Counter(benchmark_prefixes).items():
    print(f"  {prefix}: {count}")