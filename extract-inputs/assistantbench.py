import pandas as pd
import json
import os
from pathlib import Path

# TEST MODE: Set to a number to limit tasks for testing, or None to process all
TEST_LIMIT = None  # Process all tasks

# Read the result matrix CSV
result_csv = Path(__file__).parent.parent / "result" / "result_matrix_merged.csv"
df = pd.read_csv(result_csv)

# Filter columns that start with 'assistantbench.'
assistantbench_columns = [col for col in df.columns if col.startswith('assistantbench.')]

# Extract task IDs (everything after 'assistantbench.')
task_ids = [col.split('.', 1)[1] for col in assistantbench_columns]

# Apply test limit if set
if TEST_LIMIT is not None:
    task_ids = task_ids[:TEST_LIMIT]
    print(f"TEST MODE: Processing only first {TEST_LIMIT} tasks")

print(f"Found {len(task_ids)} assistantbench tasks to process")

# Directory containing trace files
traces_dir = Path(__file__).parent.parent / "traces"

# Get all assistantbench trace files and sort by size (smallest to biggest)
trace_files = list(traces_dir.glob("assistantbench_*_UPLOAD.json"))
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
                    # Extract the user message that contains "Your ultimate task is:"
                    inputs = result.get('inputs', {})
                    messages = inputs.get('messages', [])
                    
                    # Handle nested messages structure (messages can be a list of lists)
                    if messages and isinstance(messages[0], list):
                        messages = messages[0]
                    
                    query = None
                    for message in messages:
                        # Handle different message formats (dict or object with kwargs)
                        if isinstance(message, dict):
                            role = message.get('role')
                            content = message.get('content', '')
                            
                            # Also check kwargs structure (LangChain format)
                            if 'kwargs' in message:
                                role = message['kwargs'].get('type')
                                content = message['kwargs'].get('content', '')
                                # Map LangChain types to standard roles
                                if role == 'human':
                                    role = 'user'
                            
                            if role == 'user' and isinstance(content, str) and 'Your ultimate task is:' in content:
                                # Extract the task after "Your ultimate task is:"
                                query = content.split('Your ultimate task is:', 1)[1].strip()
                                # Remove trailing instruction about "done action"
                                if '. If you achieved your ultimate task' in query:
                                    query = query.split('. If you achieved your ultimate task')[0].strip()
                                # Remove all surrounding quotes (triple, double, single)
                                query = query.strip().strip('"').strip("'").strip()
                                break
                    
                    if query:
                        results[task_id] = query
                        tasks_to_find.remove(task_id)
                        print(f"  Found task: {task_id}")
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
output_csv = Path(__file__).parent.parent / "output" / "assistantbench_inputs.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"\nExtracted {len(results_list)} task queries out of {len(task_ids)} total tasks")
print(f"Saved to: {output_csv}")
