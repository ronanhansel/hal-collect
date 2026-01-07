import pandas as pd
import json
from pathlib import Path

# Set to None to process all tasks, or set to a small number for testing
TEST_LIMIT = None

# Read the result matrix to get task IDs
result_matrix = pd.read_csv('../result/result_matrix_merged.csv')

# Get all scienceagentbench task columns
scienceagentbench_cols = [col for col in result_matrix.columns if col.startswith('scienceagentbench.')]
print(f"Found {len(scienceagentbench_cols)} scienceagentbench tasks")

# Extract task IDs (remove the 'scienceagentbench.' prefix)
task_ids = set([col.replace('scienceagentbench.', '') for col in scienceagentbench_cols])
print(f"Total tasks to extract: {len(task_ids)}")

# Get all trace files and sort by size (smallest to largest)
trace_dir = Path('../traces')
trace_files = [(f, f.stat().st_size) for f in trace_dir.glob('*scienceagentbench*.json')]
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
        
        # Check raw_logging_results for task data
        if 'raw_logging_results' in data and isinstance(data['raw_logging_results'], list):
            for result in data['raw_logging_results']:
                # Get task_id from weave_task_id field
                task_id = str(result.get('weave_task_id', ''))
                
                if not task_id or task_id in extracted_queries:
                    continue
                
                # Check if this is one of our target tasks
                if task_id not in task_ids:
                    continue
                
                print(f"Found task: {task_id}")
                
                # Extract the query from inputs.messages
                if 'inputs' in result and 'messages' in result['inputs']:
                    messages = result['inputs']['messages']
                    
                    # Find the user message
                    for msg in messages:
                        # Handle both dict format and potential other formats
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                        else:
                            continue
                        
                        if role == 'user' and content:
                            # Extract the actual task query
                            # Look for "Here's the user request you need to work on:"
                            if "Here's the user request you need to work on:" in content:
                                task_start = content.find("Here's the user request you need to work on:") + len("Here's the user request you need to work on:")
                                task_query = content[task_start:].strip()
                                
                                # Strip out repetitive patterns - keep only the core task description
                                # Remove everything after "You can access the dataset" or similar phrases
                                markers = [
                                    "\nYou can access the dataset",
                                    "\nThe dataset is located",
                                    "\nHere is the directory structure",
                                    "\nDataset location:",
                                ]
                                for marker in markers:
                                    if marker in task_query:
                                        task_query = task_query[:task_query.find(marker)].strip()
                                        break
                                
                                # Clean up the query - remove quotes and extra whitespace
                                task_query = task_query.strip().strip('"').strip("'").strip()
                                
                                extracted_queries[task_id] = task_query
                                print(f"  Extracted query (length: {len(task_query)})")
                                break
        
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

output_path = '../output/scienceagentbench_inputs.csv'
df.to_csv(output_path, index=False)
print(f"Saved to: {Path(output_path).resolve()}")
