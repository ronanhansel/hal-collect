import pandas as pd
import json
import os
from pathlib import Path

# TEST MODE: Set to a number to limit tasks for testing, or None to process all
TEST_LIMIT = None  # Process all tasks

# Read the result matrix CSV
result_csv = Path(__file__).parent.parent / "result" / "result_matrix_merged.csv"
df = pd.read_csv(result_csv)

# Filter columns that start with 'corebench_hard.'
corebench_columns = [col for col in df.columns if col.startswith('corebench_hard.')]

# Extract task IDs (everything after 'corebench_hard.')
task_ids = [col.split('.', 1)[1] for col in corebench_columns]

# Apply test limit if set
if TEST_LIMIT is not None:
    task_ids = task_ids[:TEST_LIMIT]
    print(f"TEST MODE: Processing only first {TEST_LIMIT} tasks")

print(f"Found {len(task_ids)} corebench_hard tasks to process")

# Directory containing trace files
traces_dir = Path(__file__).parent.parent / "traces"

# Get all corebench_hard trace files and sort by size (smallest to biggest)
trace_files = list(traces_dir.glob("corebench_hard_*_UPLOAD.json"))
trace_files.sort(key=lambda f: f.stat().st_size)

print(f"Found {len(trace_files)} trace files to process")

# Store results and track which tasks we've found
results = {}
tasks_to_find = set(task_ids)

# Process files from smallest to biggest
for trace_file in trace_files:
    if not tasks_to_find:
        print("All tasks found!")
        break
    
    print(f"Processing {trace_file.name} (size: {trace_file.stat().st_size / 1024 / 1024:.2f} MB, {len(tasks_to_find)} tasks remaining)")
    
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Check if this file has raw_logging_results
        if 'raw_logging_results' in data:
            for result in data['raw_logging_results']:
                # Extract task_id from attributes.weave_task_id
                task_id = result.get('attributes', {}).get('weave_task_id', '')
                
                # Check if this is a task we're looking for
                if task_id in tasks_to_find:
                    # Extract the first user message from inputs.messages
                    inputs = result.get('inputs', {})
                    messages = inputs.get('messages', [])
                    
                    for message in messages:
                        if message.get('role') == 'user':
                            # Get the text content
                            content = message.get('content', [])
                            query = None
                            
                            if isinstance(content, list):
                                for item in content:
                                    if item.get('type') == 'text':
                                        text = item.get('text', '')
                                        # Extract just the query part (after "New task:\n")
                                        if 'New task:' in text:
                                            query = text.split('New task:', 1)[1].strip()
                                            break
                            elif isinstance(content, str):
                                if 'New task:' in content:
                                    query = content.split('New task:', 1)[1].strip()
                            
                            if query:
                                results[task_id] = query
                                tasks_to_find.remove(task_id)
                                print(f"  Found task: {task_id}")
                            break
    except Exception as e:
        print(f"Error processing {trace_file.name}: {e}")
        continue

# Report any missing tasks
if tasks_to_find:
    print(f"\nWARNING: Could not find {len(tasks_to_find)} tasks:")
    for task_id in sorted(tasks_to_find):
        print(f"  - {task_id}")

# Convert results dict to list format for DataFrame
results_list = [{'task_id': task_id, 'text_input': query} for task_id, query in results.items()]

# Create DataFrame and save to CSV
output_df = pd.DataFrame(results_list)
output_csv = Path(__file__).parent.parent / "output" / "corebench_hard_inputs.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"\nExtracted {len(results_list)} task queries out of {len(task_ids)} total tasks")
print(f"Saved to: {output_csv}")
