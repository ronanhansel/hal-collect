import pandas as pd
import json
import os
from tqdm import tqdm
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def trace_paths_from_dir(dir: str, benchmark_filter: str = None) -> list[str]:
    '''
    Returns a list of JSON files in a directory.
    Generally, dir = "/traces", which contains multiple JSON trace files.
    If benchmark_filter is provided, only returns files matching that benchmark pattern.
    '''
    trace_files = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if benchmark_filter is None or file.startswith(benchmark_filter):
                trace_files.append(os.path.join(dir, file))
    return trace_files

def clean_model_name(name: str) -> str:
    '''
    Cleans the model name by removing unwanted characters.
    '''
    name = name.replace('"', '').strip()
    # Note: Do NOT do blanket replacements of 'o' - this corrupts model names like "sonnet"
    # If specific O3/O4 fixes are needed, handle them explicitly with full context
    name = name.split("/")[-1]
    return name

def trace_config_summary(dir: str, output_dir: str, benchmark_filter: str = None) -> pd.DataFrame:
    '''
    Compiles all the configs from each trace into a single DataFrame.
    '''
    configs = []
    json_files = trace_paths_from_dir(dir, benchmark_filter)
    for file_name in tqdm(json_files, desc="Processing files"):
        tqdm.write(f"Processing: {os.path.basename(file_name)}")
        with open(file_name, "r") as f:
            data = json.load(f)
            model_name = data['config']['agent_args'].get('model_name', '')
            successful_tasks = data['results'].get('successful_tasks', [])
            failed_tasks = data['results'].get('failed_tasks', [])
            data['config']['model_name'] = clean_model_name(model_name)
            data['config']['successful_tasks'] = len(successful_tasks)
            data['config']['failed_tasks'] = len(failed_tasks)
            data['config']['total_tasks'] = len(successful_tasks) + len(failed_tasks)
            configs.append(data['config'])

    df = pd.DataFrame(configs)
    df = df[['run_id', 'benchmark_name', 'agent_name' , 'model_name', 'successful_tasks', 'failed_tasks', 'total_tasks']]

    # save to csv
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "trace_summary.csv"), index=False)
    return df

def build_matrix(dir: str, output_dir: str, benchmark_filter: str = None):
    json_files = trace_paths_from_dir(dir, benchmark_filter)
    rows = []  # Collect all rows
    
    for file_name in tqdm(json_files, desc="Processing files"):
        with open(file_name, "r") as f:
            data = json.load(f)
            # Update progress bar with current filename
            tqdm.write(f"Processing: {os.path.basename(file_name)}")
            
            benchmark_name = data["config"]["benchmark_name"]
            raw_model_name = data["config"]["agent_args"].get("model_name", "")

            # Start with base row data
            row = {
                "benchmark_name": benchmark_name,
                "agent_name": data["config"]["agent_name"],
                "model_name": clean_model_name(raw_model_name),
            }

            # Handle scienceagentbench differently - use raw_eval_results.eval_result
            if benchmark_name == "scienceagentbench":
                eval_results = data.get("raw_eval_results", {}).get("eval_result", {})
                for task_id, task_data in eval_results.items():
                    clean_task_name = f"{benchmark_name}.{task_id}"
                    # Use success_rate as binary success (1 if success_rate > 0, else 0)
                    success_rate = task_data.get("success_rate", 0)
                    row[clean_task_name] = 1 if success_rate > 0 else 0
            else:
                # For other benchmarks, use the original logic
                successful = data.get("results", {}).get("successful_tasks", [])
                failed = data.get("results", {}).get("failed_tasks", [])
                all_tasks = successful + failed

                for task in all_tasks:
                    clean_task_name = f"{benchmark_name}.{task}"
                    row[clean_task_name] = 1 if task in successful else 0
            rows.append(row)
    
    df_new = pd.DataFrame(rows)
    
    # Check if result_matrix.csv already exists and merge if it does
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result_matrix.csv")
    
    if os.path.isfile(result_path):
        print(f"Found existing result_matrix.csv. Merging with new results...")
        df_existing = pd.read_csv(result_path)
        
        # Merge dataframes - combine on agent_name, model_name, benchmark_name
        # For overlapping tasks, prefer new data
        df = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Remove duplicates based on agent_name, model_name, and benchmark_name
        # Keep last (newest) entry
        key_cols = ['agent_name', 'model_name', 'benchmark_name']
        df = df.drop_duplicates(subset=key_cols, keep='last')
        
        print(f"Merged {len(df_existing)} existing rows with {len(df_new)} new rows. Final: {len(df)} rows.")
    else:
        df = df_new
    
    # save to csv
    df.to_csv(result_path, index=False)
    return df

def plot_matrix_single_benchmark(output_dir: str, benchmark_name: str):
    matrix_path = os.path.join(output_dir, "result_matrix.csv")
    if not os.path.isfile(matrix_path):
        print(f"Error: {matrix_path} not found. Run --build_matrix first.")
        return
    df = pd.read_csv(matrix_path)

    # Filter for the specific benchmark
    df_benchmark = df[df['benchmark_name'] == benchmark_name]
    
    # Get task columns (all columns that start with the benchmark name)
    task_columns = [col for col in df_benchmark.columns if col.startswith(f"{benchmark_name}.")]
    
    # Create a matrix with agent_name as index (rows) and tasks as columns
    matrix_data = df_benchmark[['agent_name'] + task_columns].set_index('agent_name')
    
    # Sort rows (agent names) alphabetically
    matrix_data = matrix_data.sort_index()
    
    # Remove the benchmark prefix from task column names (just use task IDs)
    matrix_data.columns = [col.replace(f"{benchmark_name}.", "") for col in matrix_data.columns]
    
    plt.figure(figsize=(20, 8))
    # plt.rcParams.update({'font.size': 12}) 

    # Replace NaN with -1 for visualization purposes
    plot_data = matrix_data.fillna(-1)
    
    # Custom colormap: red (-1/NaN), white (0/failure), green (1/success)
    colors = ['red', 'white', 'dodgerblue']
    cmap = ListedColormap(colors)
    
    sns.heatmap(plot_data, annot=False, cmap=cmap, cbar=False,
                linewidths=0.5, 
                linecolor='gray',
                xticklabels=False,
                vmin=-1, vmax=1)  # Set range to include all three values
    
    plt.title(f'Task Performance: {benchmark_name}')
    plt.xlabel('Task ID')
    plt.ylabel('Agent Name')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{benchmark_name}_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser(description='Process trace files from a directory')
    parser.add_argument('directory', type=str, help='Directory containing trace JSON files')
    parser.add_argument('--output', type=str, default='./result', help='Output directory for results (default: ./result)')
    parser.add_argument('--summarize', action='store_true', help='Generate config summary')
    parser.add_argument('--build_matrix', action='store_true', help='Build task success/failure matrix')
    parser.add_argument('--plot_matrix', action='store_true', help='Plot matrix for a specific benchmark', default=None)
    parser.add_argument('--benchmark', type=str, help='Benchmark name for filtering/plotting (e.g., "scienceagentbench" to match scienceagentbench* files)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    print(f"Processing directory: {args.directory}")
    print(f"Output directory: {args.output}")
    if args.benchmark:
        print(f"Filtering for benchmark: {args.benchmark}")
    
    if args.summarize:
        print(f"Generating summary for: {args.directory}")
        trace_config_summary(args.directory, args.output, args.benchmark)
    
    if args.build_matrix:
        print(f"Building task matrix for: {args.directory}")
        build_matrix(args.directory, args.output, args.benchmark)
    
    if args.plot_matrix:
        if not args.benchmark:
            print("Error: --benchmark must be specified when using --plot_matrix")
            return
        benchmark_name = args.benchmark
        print(f"Plotting matrix for benchmark: {benchmark_name}")
        plot_matrix_single_benchmark(args.output, benchmark_name)
    
    # Let me see some stuff
    result_matrix_path = os.path.join(args.output, "result_matrix.csv")
    if os.path.isfile(result_matrix_path):
        df = pd.read_csv(result_matrix_path)
        unique_models = sorted(df['model_name'].dropna().unique())
        print(unique_models)
        print(len(unique_models))


if __name__ == "__main__":
    main() 