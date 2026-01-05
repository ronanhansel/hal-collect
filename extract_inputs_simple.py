"""
Simplified extraction script that matches by (benchmark, model, task_id) only.

Since agent_run_id doesn't reliably map to trace_id across all benchmarks,
we extract one instance per (benchmark, model, task_id) combination.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def find_trace_files(benchmark_id: str, model: str, traces_dir: Path):
    """Find trace files for a benchmark and model."""
    # Normalize model for matching
    model_parts = model.lower().replace('-', '').replace('.', '').replace('_', '')
    
    # Get prefix for benchmark
    prefix_map = {
        'assistantbench': 'assistantbench_',
        'corebench': 'core',
        'scicode': 'scicode_',
        'taubench': 'taubench_'
    }
    prefix = prefix_map.get(benchmark_id, benchmark_id + '_')
    
    # Find all trace files for this benchmark
    all_files = list(traces_dir.glob(f'{prefix}*_UPLOAD.json'))
    
    # Filter by model
    candidates = []
    for f in all_files:
        fname_norm = f.name.lower().replace('-', '').replace('.', '').replace('_', '')
        if model_parts in fname_norm:
            candidates.append(f)
    
    return candidates


def extract_text_from_content(content):
    """Extract text from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
            elif isinstance(item, str):
                parts.append(item)
        return '\n'.join(parts)
    return str(content)


def extract_from_trace_file(trace_file: Path, needed_task_ids: set):
    """Extract task inputs from a trace file for specific task IDs."""
    results = {}
    
    try:
        with open(trace_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"      Error loading {trace_file.name}: {e}")
        return results
    
    if 'raw_logging_results' not in data:
        return results
    
    # Group entries by weave_task_id
    task_entries = defaultdict(list)
    for entry in data['raw_logging_results']:
        task_id = entry.get('attributes', {}).get('weave_task_id')
        if task_id and str(task_id) in needed_task_ids:
            task_entries[str(task_id)].append(entry)
    
    # Extract input from first LLM call for each task
    for task_id, entries in task_entries.items():
        for entry in sorted(entries, key=lambda x: x.get('started_at', ''))[:3]:
            inputs = entry.get('inputs', {})
            if not isinstance(inputs, dict):
                continue
            
            messages = inputs.get('messages', [])
            if not isinstance(messages, list) or not messages:
                continue
            
            # Try to extract user message
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') == 'user':
                    content = extract_text_from_content(msg.get('content', ''))
                    if content and content not in ['Hi! How can I help you today?', 'Hello', 'Hi']:
                        results[task_id] = content
                        break
            
            if task_id in results:
                break
            
            # Fall back to system message
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') == 'system':
                    content = extract_text_from_content(msg.get('content', ''))
                    if content:
                        results[task_id] = content
                        break
            
            if task_id in results:
                break
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv')
    parser.add_argument('--traces', default='traces')
    parser.add_argument('--output', default='all_benchmarks_inputs.csv')
    parser.add_argument('--benchmark', default=None, help='Specific benchmark to process')
    args = parser.parse_args()
    
    base_dir = Path('/home/azureuser/cloudfiles/code/hal-collect')
    
    # Load CSV
    df = pd.read_csv(base_dir / args.csv)
    print(f"Loaded {len(df)} rows")
    
    # Filter benchmarks
    if args.benchmark:
        df = df[df['benchmark_id'] == args.benchmark]
    
    benchmarks = sorted(df['benchmark_id'].unique())
    print(f"Processing {len(benchmarks)} benchmarks: {benchmarks}")
    
    all_results = []
    
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {benchmark}")
        print(f"{'='*60}")
        
        bench_df = df[df['benchmark_id'] == benchmark]
        
        for model in sorted(bench_df['model'].unique()):
            print(f"\n  Model: {model}")
            model_df = bench_df[bench_df['model'] == model]
            task_ids = set(model_df['task_id'].astype(str).values)
            
            print(f"    Need {len(task_ids)} tasks")
            
            # Find trace files
            trace_files = find_trace_files(benchmark, model, base_dir / args.traces)
            print(f"    Found {len(trace_files)} trace files")
            
            if not trace_files:
                continue
            
            # Extract from each trace file
            found_inputs = {}
            for trace_file in trace_files:
                if len(found_inputs) >= len(task_ids):
                    break
                
                print(f"      Checking: {trace_file.name[:55]}...")
                needed = task_ids - set(found_inputs.keys())
                batch_results = extract_from_trace_file(trace_file, needed)
                found_inputs.update(batch_results)
                print(f"        Found {len(found_inputs)}/{len(task_ids)}")
            
            # Create result rows (one per task_id, using first agent_run_id we find)
            for task_id, task_input in found_inputs.items():
                # Get first matching row from CSV
                task_rows = model_df[model_df['task_id'].astype(str) == task_id]
                if not task_rows.empty:
                    row = task_rows.iloc[0]
                    all_results.append({
                        'benchmark_id': benchmark,
                        'model': model,
                        'task_id': task_id,
                        'agent_run_id': row['agent_run_id'],  # Keep one agent_run_id for reference
                        'task_input': task_input
                    })
    
    # Save
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_path = base_dir / args.output
        output_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Saved {len(output_df)} entries to {output_path}")
        
        # Show coverage
        for benchmark in sorted(output_df['benchmark_id'].unique()):
            bench_out = output_df[output_df['benchmark_id'] == benchmark]
            bench_orig = df[df['benchmark_id'] == benchmark]
            unique_tasks_found = bench_out.groupby(['model', 'task_id']).size().count()
            unique_tasks_needed = bench_orig.groupby(['model', 'task_id']).size().count()
            print(f"\n  {benchmark}: {unique_tasks_found}/{unique_tasks_needed} unique (model, task) pairs")
            
            for model in sorted(bench_out['model'].unique()):
                found = len(bench_out[bench_out['model'] == model])
                needed = bench_orig[bench_orig['model'] == model]['task_id'].nunique()
                print(f"    {model}: {found}/{needed}")
    else:
        print("\n⚠️  No results extracted")


if __name__ == '__main__':
    main()
