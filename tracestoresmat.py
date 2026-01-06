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
import re

# Optional: ijson for streaming large files (not currently used)
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

def normalize_model_name(model_name: str) -> str:
    """
    Normalizes model name to snake_case with YYYY_MM_DD date format.
    """
    if not isinstance(model_name, str):
        return "unknown"

    # 1. Lowercase
    name = model_name.lower().strip()

    # 2. Strip common provider prefixes
    prefixes = [
        r'openrouter/anthropic/', r'openrouter/deepseek/', r'openrouter/openai/',
        r'openrouter/google/', r'openrouter/', r'together_ai/deepseek-ai/',
        r'together_ai/', r'anthropic/', r'openai/', r'google/', r'gemini/',
        r'deepseek-ai/', r'deepseek/', r'azure/', r'aws/', r'vertex/'
    ]
    for prefix in prefixes:
        name = re.sub(f'^{prefix}', '', name)

    # 3. Extract and normalize Date (using the helper from previous steps)
    # Ensure normalize_date_string returns (clean_name, "YYYY_MM_DD")
    name, date_str = normalize_date_string(name)

    # 4. Handle specific model aliases BEFORE strict cleaning
    if 'deepseek-chat' in name and 'v3' not in name:
        name = name.replace('deepseek-chat', 'deepseek-v3')

    # 5. Apply Strict Snake Case
    name = to_strict_snake_case(name)

    # 6. Reattach Date (which is already in YYYY_MM_DD format)
    if date_str:
        name = f"{name}_{date_str}"

    return name

def normalize_date_string(text: str) -> Tuple[str, str]:
    """
    Extracts a date from a string and converts it to YYYY_MM_DD format.
    Returns: (text_without_date, formatted_date_string)
    """
    # Patterns for YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD (years 2020-2029)
    # We look for word boundaries or underscores around the date
    date_patterns = [
        r'(202[0-9])[-_]?([0-1][0-9])[-_]?([0-3][0-9])', # Matches 20250219, 2025-02-19, 2025_02_19
    ]
    
    found_date = ""
    clean_text = text
    
    for pattern in date_patterns:
        match = re.search(pattern, clean_text)
        if match:
            # Extract Y, M, D
            y, m, d = match.groups()
            # Create standard format
            found_date = f"{y}_{m}_{d}"
            # Remove the original date from the text (replace with placeholder to avoid merging issues)
            # We strip it out completely to clean the base name, then append it back later
            clean_text = re.sub(pattern, '', clean_text)
            break
            
    return clean_text, found_date
def to_strict_snake_case(text: str) -> str:
    """
    Converts text to strict snake_case:
    1. Lowercase
    2. Replaces ALL non-alphanumeric chars (hyphens, dots, spaces) with '_'
    3. Collapses multiple '__' into single '_'
    4. Strips leading/trailing '_'
    """
    if not isinstance(text, str):
        return "unknown"
    
    # Lowercase
    text = text.lower()
    
    # Replace anything that is NOT a letter or number with _
    text = re.sub(r'[^a-z0-9]', '_', text)
    
    # Collapse multiple underscores to single
    text = re.sub(r'_+', '_', text)
    
    return text.strip('_')

def get_clean_scaffold_and_model(raw_index_str: str) -> Tuple[str, str]:
    """
    Parses a raw index string into (Scaffold, Model) with strict snake_case.
    Input: "AssistantBench-Agent__claude-3-7-sonnet"
    Output: ("assistantbench_agent", "claude_3_7_sonnet")
    """
    # Lowercase immediately
    raw_lower = raw_index_str.lower().strip()
    
    # --- Step 1: Split Scaffold and Model ---
    # We still need to split by '__' if it exists in the raw data to separate Agent from Model
    if '__' in raw_lower:
        parts = raw_lower.split('__')
        raw_scaffold = parts[0]
        raw_model = parts[-1]
    else:
        raw_scaffold = raw_lower
        raw_model = "unknown"

    # --- Step 2: Clean the Scaffold Name ---
    # Normalize to known scaffolds, then apply strict snake_case
    
    if 'hal' in raw_scaffold and 'general' in raw_scaffold:
        clean_scaffold = "hal_generalist"
    elif 'browser' in raw_scaffold and 'use' in raw_scaffold:
        clean_scaffold = "browser_use"
    elif 'core' in raw_scaffold:
        clean_scaffold = "core_agent"
    elif 'scicode' in raw_scaffold:
        clean_scaffold = "scicode_agent"
    elif 'assistant' in raw_scaffold:
        clean_scaffold = "assistantbench_agent"
    elif 'taubench' in raw_scaffold:
        if 'few' in raw_scaffold:
            clean_scaffold = "taubench_fewshot"
        elif 'tool' in raw_scaffold:
            clean_scaffold = "taubench_toolcalling"
        else:
            clean_scaffold = "taubench"
    else:
        # Remove parens and clean
        clean_scaffold = re.sub(r'\s*\(.*?\)', '', raw_scaffold)
        clean_scaffold = to_strict_snake_case(clean_scaffold)

    # --- Step 3: Normalize the Model Name ---
    clean_model = normalize_model_name(raw_model)

    return clean_scaffold, clean_model


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
        
        # Extract agent and model names
        agent_name = config.get('agent_name', 'unknown')
        model_name = config.get('agent_args', {}).get('model_name', 'unknown')
        benchmark = config.get('benchmark_name', 'unknown')
        
        # Create agent key with special handling for HAL Generalist
        agent_name_lower = agent_name.lower()
        if any(x in agent_name_lower for x in ['hal generalist agent', 'hal generalist', 'hal_generalist_agent', 'hal_generalist']):
            # Normalize model name for HAL Generalist to merge across providers
            normalized_model = normalize_model_name(model_name)
            agent_key = f"hal_generalist_{normalized_model}"
        else:
            agent_key = f"{agent_name}__{model_name}"
        
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
                    # This will now return ("assistantbench-agent", "claude-3-7-sonnet_2025_02_19")
                    scaffold, model = get_clean_scaffold_and_model(str(idx))
                    
                    # The final key will look like: assistantbench-agent__claude-3-7-sonnet_2025_02_19
                    new_idx = f"{scaffold}_{model}"
                    new_index.append(new_idx)
                
                df.index = new_index
                
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
