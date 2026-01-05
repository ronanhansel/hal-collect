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


def extract_assistantbench_messages(messages_wrapper):
    """Extract messages from assistantbench's special serialized format."""
    if not messages_wrapper or not isinstance(messages_wrapper, list):
        return []
    
    # Unwrap: messages is [[{msg1}, {msg2}, ...]]
    if isinstance(messages_wrapper[0], list):
        messages_list = messages_wrapper[0]
    else:
        messages_list = messages_wrapper
    
    result = []
    for msg in messages_list:
        if isinstance(msg, dict) and 'kwargs' in msg:
            kwargs = msg.get('kwargs', {})
            msg_type = kwargs.get('type', 'unknown')
            content = kwargs.get('content', '')
            
            if isinstance(content, str) and content:
                result.append({
                    'role': msg_type,
                    'content': content
                })
    
    return result


def normalize_whitespace(text):
    """Normalize excessive whitespace while preserving structure."""
    import re
    # Collapse multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)
    # Collapse multiple newlines to max 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove spaces at line starts/ends
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text


def extract_from_trace_file(trace_file: Path, needed_task_ids: set):
    """Extract ONLY the first input from a trace file for specific task IDs."""
    results = {}
    is_assistantbench = 'assistantbench' in str(trace_file)
    
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
    
    # Extract ONLY the first input message for each task
    for task_id, entries in task_entries.items():
        best_content = None
        
        # For assistantbench: find LLM call with system prompt (not the short warmup call)
        # For others: find entry with "Instruction:" (taubench) or first valid entry
        sorted_entries = sorted(entries, key=lambda x: x.get('started_at', ''))
        
        for entry in sorted_entries[:10]:
            inputs = entry.get('inputs', {})
            if not isinstance(inputs, dict):
                continue
            
            messages = inputs.get('messages', [])
            if not messages:
                continue
            
            # Handle assistantbench's special format
            if is_assistantbench:
                parsed_messages = extract_assistantbench_messages(messages)
                if not parsed_messages:
                    continue
                
                # Look for LLM call with system message (length > 1000)
                has_substantial_system = any(
                    msg['role'] in ['system'] and len(msg['content']) > 1000
                    for msg in parsed_messages
                )
                
                if not has_substantial_system:
                    continue
                
                # Extract system + first 2 human messages
                all_parts = []
                for msg in parsed_messages:
                    if msg['role'] == 'system':
                        all_parts.append(msg['content'])
                    elif msg['role'] == 'human':
                        all_parts.append(msg['content'])
                        if len(all_parts) >= 3:  # system + 2 human messages
                            break
                
                if all_parts:
                    best_content = '\n\n---\n\n'.join(all_parts)
                    break
            
            # Handle other benchmarks
            else:
                if not isinstance(messages, list):
                    continue
                
                # Extract ONLY the initial task description
                all_parts = []
                seen_user = False
                
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    
                    role = msg.get('role')
                    
                    # Take first system/developer message
                    if role in ['system', 'developer'] and not all_parts:
                        content = extract_text_from_content(msg.get('content', ''))
                        if content:
                            all_parts.append(content)
                    
                    # Take ONLY the first user message (the actual task)
                    elif role == 'user' and not seen_user:
                        content = extract_text_from_content(msg.get('content', ''))
                        # Skip generic greetings
                        if content and content not in ['Hi! How can I help you today?', 'Hello', 'Hi']:
                            all_parts.append(content)
                            seen_user = True
                            break  # Stop after first user message
                
                if all_parts:
                    combined = '\n\n---\n\n'.join(all_parts)
                    # Normalize whitespace to fix LaTeX/Wikipedia formatting issues
                    combined = normalize_whitespace(combined)
                    
                    # For taubench: only keep if it has "Instruction:" (task-specific)
                    # For others: take first valid entry
                    if 'Instruction:' in combined or 'taubench' not in str(trace_file):
                        best_content = combined
                        break  # Found the right entry
        
        if best_content:
            results[task_id] = best_content
    
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
        # Save as pickle to avoid CSV formatting issues with multi-line content
        output_path_pkl = output_path.with_suffix('.pkl')
        output_df.to_pickle(output_path_pkl)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Saved {len(output_df)} entries to {output_path_pkl}")
        
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
