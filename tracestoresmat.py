#!/usr/bin/env python3
"""
Script to compile a response matrix from trace files.
Creates a DataFrame with:
- Rows: agent_name__model_name
- Columns: MultiIndex (task_id, benchmark)
- Values: 1 (success), 0 (failure), NaN (not attempted)
"""

import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
from collections import defaultdict
import sys

# Add tools directory to path to import clean_name
sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from clean_name import normalize_model_name, get_clean_scaffold_and_model

# Optional: ijson for streaming large files (not currently used)
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


def extract_trace_info_efficient(file_path: Path) -> Dict:
    """
    Efficiently extract only config and results from potentially large JSON files.
    Uses streaming parser for large files.
    """
    try:
        # For files under 100MB, use regular json.load (faster)
        file_size = file_path.stat().st_size
        
        if file_size < 100 * 1024 * 1024:  # 100MB threshold
            with open(file_path, 'r') as f:
                data = json.load(f)
                return {
                    'config': data.get('config', {}),
                    'results': data.get('results', {})
                }
        else:
            # For large files, use streaming parser
            print(f"  Large file detected ({file_size / (1024*1024):.1f}MB), using streaming parser...")
            config = {}
            results = {}
            
            with open(file_path, 'rb') as f:
                parser = ijson.parse(f)
                current_section = None
                current_key = None
                
                for prefix, event, value in parser:
                    if prefix == 'config':
                        current_section = 'config'
                    elif prefix == 'results':
                        current_section = 'results'
                    elif prefix.startswith('config.') and current_section == 'config':
                        key = prefix[7:]  # Remove 'config.'
                        if '.' not in key:  # Top-level config keys only
                            config[key] = value
                    elif prefix.startswith('results.') and current_section == 'results':
                        key = prefix[8:]  # Remove 'results.'
                        if '.' not in key:  # Top-level results keys only
                            results[key] = value
                    
                    # Stop early if we have what we need
                    if config and results and 'successful_tasks' in results and 'failed_tasks' in results:
                        break
            
            return {'config': config, 'results': results}
    
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return None


def extract_trace_info_simple(file_path: Path) -> Dict:
    """
    Simple extraction that just loads the JSON and extracts what we need.
    Works better for most files.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return {
                'config': data.get('config', {}),
                'results': data.get('results', {})
            }
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return None


def compile_response_matrix(traces_dir: Path, benchmark_filter: str = None) -> pd.DataFrame:
    """
    Compile response matrix from trace files.
    
    Args:
        traces_dir: Path to directory containing trace JSON files
        benchmark_filter: Optional benchmark name to filter traces (e.g., 'usaco')
    
    Returns:
        DataFrame with agent_name__model_name as rows, (task_id, benchmark) as columns
    """
    # Storage for all data
    agent_task_results = defaultdict(dict)  # {agent_key: {(task_id, benchmark): result}}
    all_tasks = set()  # Set of all (task_id, benchmark) tuples
    
    # Find all trace files
    trace_files = list(traces_dir.glob("*_UPLOAD.json"))
    
    # Filter by benchmark if specified
    if benchmark_filter:
        trace_files = [f for f in trace_files if f.name.startswith(f"{benchmark_filter}_")]
        print(f"Filtering to {benchmark_filter} benchmark: found {len(trace_files)} files")
    else:
        print(f"Processing all benchmarks: found {len(trace_files)} files")
    
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return None
    
    # Process each trace file
    for i, trace_file in enumerate(trace_files, 1):
        print(f"[{i}/{len(trace_files)}] Processing {trace_file.name}...")
        
        # Extract info
        info = extract_trace_info_simple(trace_file)
        if not info:
            continue
        
        config = info['config']
        results = info['results']
        
        # Extract agent and model names - skip if missing
        agent_name = config.get('agent_name')
        model_name = config.get('agent_args', {}).get('model_name')
        benchmark = config.get('benchmark_name')
        
        # Skip entries with missing required fields
        if not agent_name or not model_name or not benchmark:
            print(f"  Skipping: missing required fields (agent={agent_name}, model={model_name}, benchmark={benchmark})")
            continue
        
        # Create agent key using consistent normalization
        raw_key = f"{agent_name}__{model_name}"
        scaffold, model = get_clean_scaffold_and_model(raw_key)
        
        # Skip if normalization failed
        if not scaffold or not model:
            print(f"  Skipping: failed to normalize '{raw_key}'")
            continue
        
        agent_key = f"{scaffold}_{model}"
        
        # Process successful tasks
        successful_tasks = results.get('successful_tasks', [])
        for task_id in successful_tasks:
            task_key = (task_id, benchmark)
            agent_task_results[agent_key][task_key] = 1
            all_tasks.add(task_key)
        
        # Process failed tasks
        failed_tasks = results.get('failed_tasks', [])
        for task_id in failed_tasks:
            task_key = (task_id, benchmark)
            agent_task_results[agent_key][task_key] = 0
            all_tasks.add(task_key)
        
        print(f"  Agent: {agent_name}")
        print(f"  Model: {model_name}")
        print(f"  Benchmark: {benchmark}")
        print(f"  Success: {len(successful_tasks)}, Failed: {len(failed_tasks)}")
    
    # Convert to DataFrame
    print(f"\nBuilding response matrix...")
    print(f"  Agents: {len(agent_task_results)}")
    print(f"  Tasks: {len(all_tasks)}")
    
    # Sort tasks for consistent ordering
    sorted_tasks = sorted(all_tasks)
    
    # Build the matrix
    data = []
    agents = sorted(agent_task_results.keys())
    
    for agent in agents:
        row = []
        for task in sorted_tasks:
            # Get result: 1 (success), 0 (failure), or NaN (not attempted)
            result = agent_task_results[agent].get(task, float('nan'))
            row.append(result)
        data.append(row)
    
    # Create MultiIndex for columns
    multi_index = pd.MultiIndex.from_tuples(sorted_tasks, names=['task_id', 'benchmark'])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=agents, columns=multi_index)
    
    return df


def merge_response_matrices(resmat_dir: Path) -> pd.DataFrame:
    """
    Merge all response matrix pickle files with advanced normalization.
    """
    pkl_files = list(resmat_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in {resmat_dir}")
        return None
    
    print(f"Found {len(pkl_files)} pickle files to merge.")
    
    all_dfs = []
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
                
                # Create a new index based on normalized Scaffold__Model
                new_index = []
                for idx in df.index:
                    scaffold, model = get_clean_scaffold_and_model(str(idx))
                    
                    # Skip entries that failed normalization
                    if not scaffold or not model:
                        continue
                    
                    new_idx = f"{scaffold}_{model}"
                    new_index.append(new_idx)
                
                df.index = new_index
                
                # Filter out any rows with empty string indices
                df = df[df.index != "_"]
                df = df[df.index != ""]
                
                # Deduplicate WITHIN file immediately
                # If "opus 4.1" and "opus_4_1" existed in the same file, they are now identical keys
                # We merge them by taking the max (preserving 1s/successes)
                df = df.groupby(df.index).max()
                
                all_dfs.append(df)
                
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")

    if not all_dfs:
        return None

    print("\nMerging dataframes...")
    
    # Concatenate all dataframes
    # combine_first is great but slow for many dfs. 
    # pd.concat is faster, but we need to handle the duplicates after.
    merged_df = pd.concat(all_dfs, axis=0, sort=True)
    
    print(f"  Rows before final deduplication: {len(merged_df)}")
    
    # Final Deduplication
    # We group by index (Scaffold__Model) and maximize to keep successes
    # This merges "HAL-Generalist__opus-4-1" from file A and file B
    merged_df = merged_df.groupby(merged_df.index).max()
    
    print(f"  Rows after final deduplication: {len(merged_df)}")
    
    # Sort for readability
    merged_df = merged_df.sort_index()
    merged_df = merged_df.sort_index(axis=1)
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description='Compile response matrix from trace files'
    )
    parser.add_argument(
        '--traces_dir',
        type=Path,
        default=Path(__file__).parent / 'traces',
        help='Path to traces directory (default: ./traces)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        nargs='+',
        default=None,
        help='Filter to specific benchmark(s) (e.g., usaco, assistantbench, or multiple: scicode swebench)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output pickle file path (default: resmat/response_matrix_<benchmark>.pkl)'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge all pickle files in resmat directory'
    )
    parser.add_argument(
        '--resmat_dir',
        type=Path,
        default=Path(__file__).parent / 'resmat',
        help='Path to resmat directory (default: ./resmat)'
    )
    
    args = parser.parse_args()
    
    # Handle merge mode
    if args.merge:
        print(f"Merging all pickle files in {args.resmat_dir}...\n")
        df = merge_response_matrices(args.resmat_dir)
        
        if df is None or df.empty:
            print("No data to save.")
            return
        
        # Set output path for merged file
        if args.output is None:
            args.output = args.resmat_dir / "response_matrix_merged.pkl"
        
        print(f"\nSaving merged response matrix to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(df, f)
        
        print(f"✓ Saved merged response matrix with shape {df.shape}")
        print(f"  Rows (agents): {len(df)}")
        print(f"  Columns (tasks): {len(df.columns)}")
        print(f"  Non-NaN values: {df.notna().sum().sum()}")
        print(f"  Success rate: {df[df.notna()].mean().mean():.2%}")
        
        # Show sample
        print("\nSample of merged response matrix:")
        print(df.iloc[:10, :10])
        print("\nMerged rows")
        print(df.index.tolist())
        return
    
    # Normal compilation mode - handle multiple benchmarks
    benchmarks_to_process = args.benchmark if args.benchmark else [None]
    
    # Ensure resmat directory exists
    args.resmat_dir.mkdir(exist_ok=True)
    
    for idx, benchmark in enumerate(benchmarks_to_process, 1):
        if len(benchmarks_to_process) > 1:
            print(f"\n{'='*60}")
            print(f"Processing benchmark {idx}/{len(benchmarks_to_process)}: {benchmark}")
            print(f"{'='*60}\n")
        
        # Set default output path based on benchmark filter
        if args.output is None:
            if benchmark:
                output_path = args.resmat_dir / f"response_matrix_{benchmark}.pkl"
            else:
                output_path = args.resmat_dir / "response_matrix_all.pkl"
        else:
            # If custom output specified and multiple benchmarks, add suffix
            if len(benchmarks_to_process) > 1:
                output_path = args.output.parent / f"{args.output.stem}_{benchmark}{args.output.suffix}"
            else:
                output_path = args.output
        
        print(f"Traces directory: {args.traces_dir}")
        print(f"Output file: {output_path}")
        print()
        
        # Compile response matrix
        df = compile_response_matrix(args.traces_dir, benchmark)
        
        if df is None or df.empty:
            print("No data to save.")
            continue
        
        # Save to pickle
        print(f"\nSaving response matrix to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)
        
        print(f"✓ Saved response matrix with shape {df.shape}")
        print(f"  Rows (agents): {len(df)}")
        print(f"  Columns (tasks): {len(df.columns)}")
        print(f"  Non-NaN values: {df.notna().sum().sum()}")
        print(f"  Success rate: {df[df.notna()].mean().mean():.2%}")
        
        # Show sample
        print("\nSample of response matrix:")
        print(df.iloc[:5, :5])


if __name__ == "__main__":
    main()
