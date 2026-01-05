"""
Extract task inputs from taubench trace files and align them with all_benchmarks_merged.csv

This script:
1. Reads all_benchmarks_merged.csv to get (benchmark_id, model, task_id, agent_run_id) mappings
2. For each taubench entry, finds the corresponding trace file
3. Extracts the task input (user instruction) from the trace_id
4. Creates a new CSV with task_id and the full task input text

Usage:
    python extract_taubench_inputs.py
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def find_trace_file_for_model(model: str, task_id: int, traces_dir: Path) -> Optional[Path]:
    """
    Find the trace file that contains data for a specific model and task_id.
    
    For taubench, we need to match:
    - benchmark name (taubench_airline or taubench_retail)
    - model name (partial match in filename)
    """
    # Normalize model name for filename matching
    # Handle special cases
    model_base = model.lower().split('_')[0]  # Get base model name (e.g., "gpt-5" from "gpt-5_high")
    model_normalized = model_base.replace('-', '').replace('.', '')
    
    # Search for matching trace files
    trace_files = list(traces_dir.glob('taubench_*_UPLOAD.json'))
    
    # Try to find files that match the model
    candidates = []
    for trace_file in trace_files:
        filename = trace_file.name.lower()
        filename_normalized = filename.replace('-', '').replace('.', '').replace('_', '')
        
        # Check if normalized model name is in filename
        if model_normalized in filename_normalized:
            # For models with suffixes like _high, _low, check if they match
            if '_' in model:
                suffix = model.split('_')[1]  # e.g., "high", "low"
                if suffix in filename:
                    candidates.append(trace_file)
            else:
                candidates.append(trace_file)
    
    return candidates


def extract_task_input_from_trace(trace_file: Path, task_id: str, agent_run_id: str) -> Optional[str]:
    """
    Extract the task input (user instruction) from a trace file.
    
    Args:
        trace_file: Path to the trace JSON file
        task_id: The task ID (e.g., "0", "1", "2")
        agent_run_id: The agent run ID (trace_id in the trace file)
    
    Returns:
        The task input text, or None if not found
    """
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Find entries matching this task_id and agent_run_id (trace_id)
        matching_entries = []
        for entry in data['raw_logging_results']:
            weave_task_id = entry.get('attributes', {}).get('weave_task_id')
            trace_id = entry.get('trace_id')
            
            if str(weave_task_id) == str(task_id) and trace_id == agent_run_id:
                matching_entries.append(entry)
        
        if not matching_entries:
            # If no exact match, try to find any entry for this task_id
            for entry in data['raw_logging_results']:
                weave_task_id = entry.get('attributes', {}).get('weave_task_id')
                if str(weave_task_id) == str(task_id):
                    matching_entries.append(entry)
        
        if not matching_entries:
            return None
        
        # Get the first LLM call for this task (should have the user instruction)
        for entry in sorted(matching_entries, key=lambda x: x.get('started_at', '')):
            inputs = entry.get('inputs', {})
            messages = inputs.get('messages', [])
            
            if messages:
                # Extract all user messages
                task_input_parts = []
                for msg in messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if content and content != "Hi! How can I help you today?":
                            task_input_parts.append(content)
                
                if task_input_parts:
                    return "\n\n".join(task_input_parts)
                
                # If no user messages found, return the system message
                for msg in messages:
                    if msg.get('role') == 'system':
                        return msg.get('content', '')
        
        return None
        
    except Exception as e:
        print(f"Error processing {trace_file}: {e}")
        return None


def extract_all_taubench_inputs(
    csv_path: Path,
    traces_dir: Path,
    output_path: Path,
    benchmark_filter: str = 'taubench'
):
    """
    Extract all taubench inputs and create a new CSV.
    
    Args:
        csv_path: Path to all_benchmarks_merged.csv
        traces_dir: Directory containing trace JSON files
        output_path: Path to save the output CSV
        benchmark_filter: Filter for specific benchmark (default: 'taubench')
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows from CSV")
    
    # Filter for taubench
    df_filtered = df[df['benchmark_id'] == benchmark_filter].copy()
    print(f"Found {len(df_filtered)} {benchmark_filter} entries")
    
    # Group by model to find trace files more efficiently
    results = []
    found_agent_run_ids = set()
    
    for model in df_filtered['model'].unique():
        print(f"\nProcessing model: {model}")
        model_df = df_filtered[df_filtered['model'] == model]
        
        # Find candidate trace files for this model
        trace_candidates = find_trace_file_for_model(model, 0, traces_dir)
        print(f"  Found {len(trace_candidates)} candidate trace files")
        
        for trace_file in trace_candidates:
            print(f"  Checking trace file: {trace_file.name}")
            
            # Try to extract inputs for each task
            for _, row in model_df.iterrows():
                task_id = str(row['task_id'])
                agent_run_id = row['agent_run_id']
                
                # Skip if we already found this agent_run_id
                if agent_run_id in found_agent_run_ids:
                    continue
                
                task_input = extract_task_input_from_trace(trace_file, task_id, agent_run_id)
                
                if task_input:
                    results.append({
                        'benchmark_id': row['benchmark_id'],
                        'model': row['model'],
                        'task_id': task_id,
                        'agent_run_id': agent_run_id,
                        'task_input': task_input,
                        'trace_file': trace_file.name
                    })
                    found_agent_run_ids.add(agent_run_id)
            
            # Check how many we found so far
            found_count = len([r for r in results if r['model'] == model])
            print(f"    Found {found_count}/{len(model_df)} task inputs for {model}")
    
    # Create output DataFrame
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(output_df)} entries to {output_path}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"  Total entries: {len(output_df)}")
        print(f"  Unique models: {output_df['model'].nunique()}")
        print(f"  Unique tasks: {output_df['task_id'].nunique()}")
        print(f"\n  Coverage by model:")
        for model in output_df['model'].unique():
            count = len(output_df[output_df['model'] == model])
            print(f"    {model}: {count} tasks")
    else:
        print("\nNo task inputs found!")


def main():
    # Set up paths
    base_dir = Path('/home/azureuser/cloudfiles/code/hal-collect')
    csv_path = base_dir / 'hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv'
    traces_dir = base_dir / 'traces'
    output_path = base_dir / 'taubench_inputs_aligned.csv'
    
    # Extract inputs
    extract_all_taubench_inputs(csv_path, traces_dir, output_path)


if __name__ == '__main__':
    main()
