import pandas as pd
import json
import re
from pathlib import Path

# Set to None to process all tasks, or set to a small number for testing
TEST_LIMIT = None

# Read the result matrix to get task IDs
result_matrix = pd.read_csv('../result/result_matrix_merged.csv')

# Get all scicode task columns
scicode_cols = [col for col in result_matrix.columns if col.startswith('scicode.')]
print(f"Found {len(scicode_cols)} scicode tasks")

# Extract task IDs (remove the 'scicode.' prefix)
task_ids = set([col.replace('scicode.', '') for col in scicode_cols])
print(f"Total tasks to extract: {len(task_ids)}")

# Get all trace files and sort by size (smallest to largest)
trace_dir = Path('../traces')
trace_files = [(f, f.stat().st_size) for f in trace_dir.glob('*scicode*.json')]
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
                    
                    # Find the user message with problem steps
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            
                            if role == 'user' and content:
                                # Check if this contains the problem steps
                                if 'PROBLEM STEPS AND FUNCTION CODE:' in content:
                                    # Extract only the specified sections with their headers
                                    # We want:
                                    # - PROBLEM STEPS AND FUNCTION CODE:
                                    # - NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
                                    # - DEPENDENCIES:
                                    
                                    # Find the starting position of PROBLEM STEPS section
                                    steps_start = content.find('PROBLEM STEPS AND FUNCTION CODE:')
                                    if steps_start == -1:
                                        continue
                                    
                                    # Find the ending position (before RESPONSE GUIDELINES or end of content)
                                    # We want to end after DEPENDENCIES section
                                    deps_match = re.search(r'DEPENDENCIES:(.*?)(?=\n\n[A-Z][A-Z\s]+:|$)', content[steps_start:], re.DOTALL)
                                    
                                    if deps_match:
                                        # Calculate the end position after DEPENDENCIES section
                                        deps_end = steps_start + deps_match.end()
                                    else:
                                        # If we can't find DEPENDENCIES, try to find where to end
                                        response_guidelines_pos = content.find('RESPONSE GUIDELINES:', steps_start)
                                        if response_guidelines_pos != -1:
                                            deps_end = response_guidelines_pos
                                        else:
                                            deps_end = len(content)
                                    
                                    # Extract the content from PROBLEM STEPS to end of DEPENDENCIES
                                    extracted_content = content[steps_start:deps_end].strip()
                                    
                                    # Remove PROBLEM DESCRIPTION section if it appears before PROBLEM STEPS
                                    # (it shouldn't, but just in case)
                                    extracted_content = re.sub(r'PROBLEM DESCRIPTION:.*?(?=PROBLEM STEPS)', '', extracted_content, flags=re.DOTALL)
                                    
                                    # Remove APPROACH GUIDELINES section if it appears
                                    extracted_content = re.sub(r'APPROACH GUIDELINES:.*?(?=PROBLEM STEPS|NEXT STEP|DEPENDENCIES|$)', '', extracted_content, flags=re.DOTALL)
                                    
                                    # Remove RESPONSE GUIDELINES section if it appears
                                    extracted_content = re.sub(r'RESPONSE GUIDELINES:.*$', '', extracted_content, flags=re.DOTALL)
                                    
                                    # Clean up extra whitespace
                                    extracted_content = extracted_content.strip()
                                    
                                    extracted_queries[task_id] = extracted_content
                                    print(f"Found task: {task_id} (length: {len(extracted_content)})")
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

output_path = '../output/scicode_inputs.csv'
df.to_csv(output_path, index=False)
print(f"Saved to: {Path(output_path).resolve()}")
