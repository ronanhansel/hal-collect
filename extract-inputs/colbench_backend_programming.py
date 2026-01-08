import pandas as pd
import json
from pathlib import Path

# Set to None to process all tasks, or set to a small number for testing
TEST_LIMIT = None

# Read the result matrix to get task IDs
result_matrix = pd.read_csv('../result/result_matrix_merged.csv')

# Get all colbench_backend_programming task columns
colbench_cols = [col for col in result_matrix.columns if col.startswith('colbench_backend_programming.')]
print(f"Found {len(colbench_cols)} colbench_backend_programming tasks")

# Extract task IDs (remove the 'colbench_backend_programming.' prefix)
task_ids = set([col.replace('colbench_backend_programming.', '') for col in colbench_cols])
print(f"Total tasks to extract: {len(task_ids)}")

# Get all trace files and sort by size (smallest to largest)
trace_dir = Path('../traces')
trace_files = [(f, f.stat().st_size) for f in trace_dir.glob('*colbench_backend_programming*.json')]
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
        if 'raw_logging_results' in data:
            for result in data['raw_logging_results']:
                # Get task_id from attributes.weave_task_id
                if 'attributes' not in result or 'weave_task_id' not in result['attributes']:
                    continue
                
                task_id = result['attributes']['weave_task_id']
                
                # Check if this is one of our target tasks
                if task_id not in task_ids:
                    continue
                
                if task_id in extracted_queries:
                    continue
                
                # Extract messages from inputs
                if 'inputs' in result and 'messages' in result['inputs']:
                    messages = result['inputs']['messages']
                    
                    # Concatenate user messages, skipping system prompt (index 0) and assistant messages
                    # The pattern is: 0=system, 1=user, 2=assistant, 3=user, 4=assistant, etc.
                    # So we want indices 1, 3, 5, 7, ...
                    user_messages = []
                    for i, msg in enumerate(messages):
                        # Skip index 0 (system prompt) and only take odd indices (user messages)
                        if i > 0 and i % 2 == 1 and msg.get('role') == 'user':
                            content = msg.get('content', '')
                            
                            # For the first user message (index 1), remove simulation boilerplate
                            if i == 1:
                                # Remove "Your task is to simulate..." section
                                if "You would like the LLM agent to help you with the following problem:" in content:
                                    # Extract content after the problem statement intro
                                    parts = content.split("You would like the LLM agent to help you with the following problem:")
                                    if len(parts) > 1:
                                        problem_section = parts[1]
                                        # Remove everything after "Your goal is to engage" if present
                                        if "Your goal is to engage" in problem_section:
                                            problem_section = problem_section.split("Your goal is to engage")[0]
                                        content = problem_section.strip()
                            
                            user_messages.append(content)
                    
                    # Join all user messages
                    query = ' '.join(user_messages).strip()
                    
                    if query:
                        extracted_queries[task_id] = query
                        print(f"  Extracted task {task_id} ({len(extracted_queries)}/{len(task_ids)})")
        
        # Stop if we've found all tasks
        if len(extracted_queries) >= len(task_ids):
            print(f"All tasks found! Extracted {len(extracted_queries)} task queries out of {len(task_ids)} total tasks.")
            break
            
    except Exception as e:
        print(f"  Error processing file: {e}")
        continue

print(f"\nExtracted {len(extracted_queries)} out of {len(task_ids)} tasks")

# Create output DataFrame
output_data = []
for task_id in sorted(extracted_queries.keys()):
    output_data.append({
        'task_id': task_id,
        'text_input': extracted_queries[task_id]
    })

output_df = pd.DataFrame(output_data)

# Save to CSV
output_file = '../output/colbench_backend_programming_inputs.csv'
output_df.to_csv(output_file, index=False)
print(f"\nSaved {len(output_df)} tasks to {output_file}")
