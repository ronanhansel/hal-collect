import pandas as pd
import json
from pathlib import Path

# Set to None to process all tasks, or set to a small number for testing
TEST_LIMIT = None

# Read the result matrix to get task IDs
result_matrix = pd.read_csv('../result/result_matrix_merged.csv')

# Get all taubench_airline task columns
taubench_airline_cols = [col for col in result_matrix.columns if col.startswith('taubench_airline.')]
print(f"Found {len(taubench_airline_cols)} taubench_airline tasks")

# Extract task IDs (remove the 'taubench_airline.' prefix)
task_ids = set([col.replace('taubench_airline.', '') for col in taubench_airline_cols])
print(f"Total tasks to extract: {len(task_ids)}")

# Get all trace files and sort by size (smallest to largest)
trace_dir = Path('../traces')
trace_files = [(f, f.stat().st_size) for f in trace_dir.glob('*taubench_airline*.json')]
trace_files.sort(key=lambda x: x[1])  # Sort by size
print(f"Found {len(trace_files)} trace files")

# Dictionary to store extracted queries
extracted_queries = {}

# Process files from smallest to largest
for file_path, file_size in trace_files:
    if TEST_LIMIT and len(extracted_queries) >= TEST_LIMIT:
        print(f"Reached test limit of {TEST_LIMIT}")
        break
        
    print(f"Processing {file_path.name} ({file_size / (1024*1024):.1f} MB)...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check raw_eval_results for task data
        if 'raw_eval_results' in data and isinstance(data['raw_eval_results'], dict):
            for task_id, task_data in data['raw_eval_results'].items():
                # Check if this is one of our target tasks
                if task_id not in task_ids:
                    continue
                
                if task_id in extracted_queries:
                    continue
                
                # Extract the instruction from the task
                if 'task' in task_data and 'instruction' in task_data['task']:
                    instruction = task_data['task']['instruction']
                    
                    # Clean up the instruction - remove extra whitespace and quotes
                    instruction = instruction.strip().strip('"').strip("'").strip()
                    
                    extracted_queries[task_id] = instruction
                    print(f"Found task: {task_id} (length: {len(instruction)})")
        
        # Check if we've found all tasks
        if len(extracted_queries) == len(task_ids):
            print("All tasks found!")
            break
            
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        continue

print(f"\nExtracted {len(extracted_queries)} task queries out of {len(task_ids)} total tasks")

# Create DataFrame and save to CSV
df = pd.DataFrame(list(extracted_queries.items()), columns=['task_id', 'text_input'])
df = df.sort_values('task_id')

output_path = '../output/taubench_airline_inputs.csv'
df.to_csv(output_path, index=False)
print(f"Saved to: {Path(output_path).resolve()}")
