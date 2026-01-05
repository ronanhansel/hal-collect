"""
Extract task inputs from ALL benchmark trace files and align them with all_benchmarks_merged.csv

This script handles:
- assistantbench: Hash-based task IDs
- corebench: Capsule IDs (e.g., capsule-0504157)
- scicode: Numeric task IDs
- taubench: Numeric task IDs (0-49 for airline)

Usage:
    python extract_all_inputs.py
    
    # Or for specific benchmark:
    python extract_all_inputs.py --benchmark taubench
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse


# Benchmark-specific model name patterns for finding trace files
BENCHMARK_PATTERNS = {
    'assistantbench': {
        'prefix': 'assistantbench_',
        'models': {
            'claude-3-7-sonnet-20250219': ['claude37sonnet20250219'],
            'claude-3-7-sonnet-20250219_high': ['claude37sonnet20250219_high'],
            'claude-3-7-sonnet-20250219_low': ['claude37sonnet20250219_low'],
            'claude-opus-4.1': ['claudeopus41'],
            'claude-opus-4.1-20250514': ['claudeopus4120250514'],
            'claude-opus-4.1_high': ['claudeopus41_high'],
            'claude-sonnet-4.5': ['claudesonnet45'],
            'claude-sonnet-4.5-20250929': ['claudesonnet4520250929'],
            'DeepSeek-AI-DeepSeek-R1': ['deepseekaideepseekr1'],
            'DeepSeek-AI-DeepSeek-V3': ['deepseekaideepseekv3'],
            'DeepSeek-R1': ['deepseekr1'],
            'gemini-2.0-flash': ['gemini20flash'],
        }
    },
    'corebench': {
        'prefix': 'core',
        'models': {}  # Will use generic matching
    },
    'scicode': {
        'prefix': 'scicode_',
        'models': {}  # Will use generic matching
    },
    'taubench': {
        'prefix': 'taubench_',
        'models': {}  # Will use generic matching
    }
}


def normalize_model_name(model: str) -> List[str]:
    """
    Normalize model name for filename matching.
    Returns a list of possible filename patterns.
    """
    patterns = []
    
    # Base normalization
    base = model.lower().replace('-', '').replace('.', '').replace('_', '')
    patterns.append(base)
    
    # Handle models with suffixes (_high, _low, etc.)
    if '_' in model:
        parts = model.split('_')
        base_model = parts[0].lower().replace('-', '').replace('.', '')
        suffix = parts[1]
        patterns.append(f"{base_model}.*{suffix}")
    
    return patterns


def find_trace_files_for_benchmark_model(
    benchmark_id: str, 
    model: str, 
    traces_dir: Path
) -> List[Path]:
    """
    Find trace files for a specific benchmark and model.
    """
    if benchmark_id not in BENCHMARK_PATTERNS:
        return []
    
    pattern_info = BENCHMARK_PATTERNS[benchmark_id]
    prefix = pattern_info['prefix']
    
    # Get all trace files for this benchmark
    trace_files = list(traces_dir.glob(f'{prefix}*_UPLOAD.json'))
    
    # Check if we have a specific pattern for this model
    if model in pattern_info['models']:
        model_patterns = pattern_info['models'][model]
    else:
        # Use generic normalization
        model_patterns = normalize_model_name(model)
    
    # Find matching files
    candidates = []
    for trace_file in trace_files:
        filename = trace_file.name.lower()
        for pattern in model_patterns:
            # Simple substring match
            pattern_norm = pattern.replace('.*', '')
            if pattern_norm in filename.replace('-', '').replace('.', '').replace('_', ''):
                # If there's a suffix requirement, check it's in the filename
                if '_' in model:
                    suffix = model.split('_')[1]
                    if suffix in filename:
                        candidates.append(trace_file)
                        break
                else:
                    candidates.append(trace_file)
                    break
    
    return candidates


def extract_task_input_from_trace(
    trace_file: Path, 
    task_id: str, 
    agent_run_id: str,
    benchmark_id: str
) -> Optional[str]:
    """
    Extract the task input from a trace file.
    
    Works for all benchmarks - uses weave_task_id and trace_id matching.
    """
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        if 'raw_logging_results' not in data:
            return None
        
        # Find entries matching this task_id and agent_run_id (trace_id)
        matching_entries = []
        for entry in data['raw_logging_results']:
            weave_task_id = entry.get('attributes', {}).get('weave_task_id')
            trace_id = entry.get('trace_id')
            
            # Match task_id (weave_task_id should match CSV task_id)
            if str(weave_task_id) == str(task_id) and trace_id == agent_run_id:
                matching_entries.append(entry)
        
        # If no exact match on trace_id, try just matching task_id
        if not matching_entries:
            for entry in data['raw_logging_results']:
                weave_task_id = entry.get('attributes', {}).get('weave_task_id')
                if str(weave_task_id) == str(task_id):
                    matching_entries.append(entry)
        
        if not matching_entries:
            return None
        
        # Get the first LLM call for this task
        for entry in sorted(matching_entries, key=lambda x: x.get('started_at', '')):
            inputs = entry.get('inputs', {})
            
            # Handle different message formats
            if not isinstance(inputs, dict):
                continue
                
            messages = inputs.get('messages', [])
            if not isinstance(messages, list):
                continue
            
            if messages:
                # Extract user messages (skip greetings)
                task_input_parts = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        
                        # Handle structured content (list of dicts with type/text)
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                            content = '\n'.join(text_parts)
                        
                        # Skip common greetings
                        if content and content not in ["Hi! How can I help you today?", "Hello", "Hi"]:
                            task_input_parts.append(str(content))
                
                if task_input_parts:
                    return "\n\n".join(task_input_parts)
                
                # If no user messages, try system message
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get('role') == 'system':
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                            content = '\n'.join(text_parts)
                        return str(content)
        
        return None
        
    except Exception as e:
        print(f"    Error processing {trace_file.name}: {e}")
        return None


def extract_benchmark_inputs(
    csv_df: pd.DataFrame,
    benchmark_id: str,
    traces_dir: Path
) -> List[Dict]:
    """
    Extract all inputs for a specific benchmark.
    """
    results = []
    found_agent_run_ids = set()
    
    benchmark_df = csv_df[csv_df['benchmark_id'] == benchmark_id].copy()
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {benchmark_id.upper()}")
    print(f"{'='*60}")
    print(f"Total entries: {len(benchmark_df)}")
    print(f"Unique models: {benchmark_df['model'].nunique()}")
    print(f"Unique tasks: {benchmark_df['task_id'].nunique()}")
    
    for model in sorted(benchmark_df['model'].unique()):
        print(f"\n  Processing model: {model}")
        model_df = benchmark_df[benchmark_df['model'] == model]
        
        # Find trace files for this benchmark and model
        trace_files = find_trace_files_for_benchmark_model(benchmark_id, model, traces_dir)
        print(f"    Found {len(trace_files)} candidate trace files")
        
        if not trace_files:
            print(f"    ⚠️  No trace files found for {model}")
            continue
        
        # Build set of needed agent_run_ids for this model
        needed_ids = set(model_df['agent_run_id'].values) - found_agent_run_ids
        
        for trace_file in trace_files:
            if not needed_ids:  # All found
                break
                
            print(f"    Checking: {trace_file.name[:60]}...")
            
            # Load trace file once
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
            except Exception as e:
                print(f"      Error loading file: {e}")
                continue
            
            if 'raw_logging_results' not in trace_data:
                continue
            
            # Build index of task_id to agent_run_id for this file
            task_traces = defaultdict(set)
            for entry in trace_data['raw_logging_results']:
                weave_task_id = entry.get('attributes', {}).get('weave_task_id')
                trace_id = entry.get('trace_id')
                if weave_task_id and trace_id:
                    task_traces[str(weave_task_id)].add(trace_id)
            
            # Now process each needed row
            for _, row in model_df.iterrows():
                task_id = str(row['task_id'])
                agent_run_id = row['agent_run_id']
                
                # Skip if already found or not in this file
                if agent_run_id in found_agent_run_ids:
                    continue
                if task_id not in task_traces or agent_run_id not in task_traces[task_id]:
                    continue
                
                task_input = extract_task_input_from_trace_data(
                    trace_data, task_id, agent_run_id
                )
                
                if task_input:
                    results.append({
                        'benchmark_id': benchmark_id,
                        'model': model,
                        'task_id': task_id,
                        'agent_run_id': agent_run_id,
                        'task_input': task_input,
                        'trace_file': trace_file.name
                    })
                    found_agent_run_ids.add(agent_run_id)
                    needed_ids.discard(agent_run_id)
            
            found_count = len([r for r in results if r['model'] == model])
            print(f"      Found {found_count}/{len(model_df)} inputs")
    
    return results


def extract_task_input_from_trace_data(
    trace_data: Dict,
    task_id: str,
    agent_run_id: str
) -> Optional[str]:
    """
    Extract task input from already-loaded trace data.
    """
    if 'raw_logging_results' not in trace_data:
        return None
    
    # Find entries matching this task_id and agent_run_id
    matching_entries = []
    for entry in trace_data['raw_logging_results']:
        weave_task_id = entry.get('attributes', {}).get('weave_task_id')
        trace_id = entry.get('trace_id')
        
        if str(weave_task_id) == str(task_id) and trace_id == agent_run_id:
            matching_entries.append(entry)
    
    # If no exact match, try just task_id
    if not matching_entries:
        for entry in trace_data['raw_logging_results']:
            weave_task_id = entry.get('attributes', {}).get('weave_task_id')
            if str(weave_task_id) == str(task_id):
                matching_entries.append(entry)
                break  # Just get first one
    
    if not matching_entries:
        return None
    
    # Get the first LLM call for this task
    for entry in sorted(matching_entries, key=lambda x: x.get('started_at', ''))[:5]:  # Only check first 5
        inputs = entry.get('inputs', {})
        
        # Handle different message formats
        if not isinstance(inputs, dict):
            continue
            
        messages = inputs.get('messages', [])
        if not isinstance(messages, list):
            continue
        
        if messages:
            # Extract user messages (skip greetings)
            task_input_parts = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    
                    # Handle structured content (list of dicts with type/text)
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                        content = '\n'.join(text_parts)
                    
                    # Skip common greetings
                    if content and content not in ["Hi! How can I help you today?", "Hello", "Hi"]:
                        task_input_parts.append(str(content))
            
            if task_input_parts:
                return "\n\n".join(task_input_parts)
            
            # If no user messages, try system message
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') == 'system':
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                        content = '\n'.join(text_parts)
                    if content:
                        return str(content)
    
    return None


def extract_all_inputs(
    csv_path: Path,
    traces_dir: Path,
    output_path: Path,
    benchmark_filter: Optional[str] = None
):
    """
    Extract inputs for all benchmarks (or a specific one).
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows from CSV")
    
    # Get benchmarks to process
    if benchmark_filter:
        benchmarks = [benchmark_filter]
    else:
        benchmarks = sorted(df['benchmark_id'].unique())
    
    print(f"\nProcessing {len(benchmarks)} benchmark(s): {', '.join(benchmarks)}")
    
    # Extract inputs for each benchmark
    all_results = []
    for benchmark_id in benchmarks:
        results = extract_benchmark_inputs(df, benchmark_id, traces_dir)
        all_results.extend(results)
    
    # Save results
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total entries extracted: {len(output_df)}")
        print(f"Output saved to: {output_path}")
        
        # Show coverage by benchmark
        print(f"\nCoverage by benchmark:")
        for benchmark in sorted(output_df['benchmark_id'].unique()):
            bench_df = output_df[output_df['benchmark_id'] == benchmark]
            orig_df = df[df['benchmark_id'] == benchmark]
            coverage = len(bench_df) / len(orig_df) * 100 if len(orig_df) > 0 else 0
            print(f"  {benchmark}: {len(bench_df)}/{len(orig_df)} ({coverage:.1f}%)")
        
        # Show coverage by model within each benchmark
        print(f"\nDetailed coverage by benchmark and model:")
        for benchmark in sorted(output_df['benchmark_id'].unique()):
            print(f"\n  {benchmark}:")
            bench_df = output_df[output_df['benchmark_id'] == benchmark]
            orig_bench_df = df[df['benchmark_id'] == benchmark]
            for model in sorted(bench_df['model'].unique()):
                count = len(bench_df[bench_df['model'] == model])
                orig_count = len(orig_bench_df[orig_bench_df['model'] == model])
                print(f"    {model}: {count}/{orig_count}")
    else:
        print("\n⚠️  No inputs extracted!")


def main():
    parser = argparse.ArgumentParser(description='Extract task inputs from benchmark traces')
    parser.add_argument(
        '--benchmark', 
        type=str, 
        default=None,
        help='Specific benchmark to process (default: all)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv',
        help='Path to input CSV'
    )
    parser.add_argument(
        '--traces',
        type=str,
        default='traces',
        help='Directory containing trace files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='all_benchmarks_inputs_aligned.csv',
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path('/home/azureuser/cloudfiles/code/hal-collect')
    csv_path = base_dir / args.csv
    traces_dir = base_dir / args.traces
    output_path = base_dir / args.output
    
    # Extract inputs
    extract_all_inputs(csv_path, traces_dir, output_path, args.benchmark)


if __name__ == '__main__':
    main()
