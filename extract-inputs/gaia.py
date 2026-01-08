import pandas as pd
import json
from pathlib import Path

# Set to None to process all tasks, or set to a small number for testing
TEST_LIMIT = None

# Read the result matrix to get task IDs
result_matrix = pd.read_csv('../result/result_matrix_merged.csv')

# Get all gaia task columns
gaia_cols = [col for col in result_matrix.columns if col.startswith('gaia.')]
print(f"Found {len(gaia_cols)} gaia tasks")

# Extract task IDs (remove the 'gaia.' prefix)
task_ids = set([col.replace('gaia.', '') for col in gaia_cols])
print(f"Total tasks to extract: {len(task_ids)}")

# Get all trace files and sort by size (smallest to largest)
trace_dir = Path('../traces')
trace_files = [(f, f.stat().st_size) for f in trace_dir.glob('*gaia*.json')]
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
                
                # Extract the query from inputs.messages
                if 'inputs' in result and 'messages' in result['inputs']:
                    messages = result['inputs']['messages']
                    
                    # Find the user message with "New task:" or "Please answer the question below"
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', [])
                            
                            if role == 'user' and isinstance(content, list) and len(content) > 0:
                                # Extract text from content
                                text_item = content[0]
                                if isinstance(text_item, dict):
                                    text = text_item.get('text', '')
                                else:
                                    continue
                                
                                # Check if this contains the task
                                if 'New task:' in text or 'Please answer the question below' in text:
                                    # Extract only the actual question after the repetitive instructions
                                    if "Here is the question and attached files are stored in your current directory:" in text:
                                        question = text.split("Here is the question and attached files are stored in your current directory:")[1].strip()
                                    elif "Here is the question:" in text:
                                        question = text.split("Here is the question:")[1].strip()
                                    else:
                                        # If we can't find the marker, skip the boilerplate
                                        question = text
                                        # Remove "New task:" prefix if present
                                        if question.startswith("New task:"):
                                            question = question[len("New task:"):].strip()
                                        # Try to remove the repetitive instructions
                                        if "Please answer the question below" in question:
                                            # Skip to after the instructions
                                            lines = question.split('\n')
                                            # Find where the actual question starts (after empty lines following instructions)
                                            start_idx = 0
                                            for i, line in enumerate(lines):
                                                if 'comma separated list' in line:
                                                    start_idx = i + 1
                                                    break
                                            # Skip empty lines after instructions
                                            while start_idx < len(lines) and not lines[start_idx].strip():
                                                start_idx += 1
                                            question = '\n'.join(lines[start_idx:]).strip()
                                    
                                    # Clean up the question - remove extra whitespace and quotes
                                    question = question.strip().strip('"').strip("'").strip()
                                    
                                    extracted_queries[task_id] = question
                                    print(f"Found task: {task_id} (length: {len(question)})")
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

output_path = '../output/gaia_inputs.csv'
df.to_csv(output_path, index=False)
print(f"Saved to: {Path(output_path).resolve()}")
