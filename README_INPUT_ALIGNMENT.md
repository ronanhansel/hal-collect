# Task Input Alignment for ALL Benchmarks

## Problem

The `all_benchmarks_merged.csv` file contains evaluation results with identifiers like `task_id`, `agent_run_id`, `model`, and `benchmark_id`, but **does not include the actual task input text** (instructions, prompts, etc.). To create text embeddings of task inputs, you need to extract and align this information from the original trace files.

## Solution

The `extract_inputs_simple.py` script extracts task inputs from trace JSON files for ALL benchmarks and aligns them with the CSV data.

### Usage

```bash
cd /home/azureuser/cloudfiles/code/hal-collect

# Extract all benchmarks
python extract_inputs_simple.py

# Or extract a specific benchmark
python extract_inputs_simple.py --benchmark taubench
```

The script will:
- Read from: `hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv`
- Search in: `traces/` directory
- Output to: `all_benchmarks_inputs.csv`

### Results

✅ **Successfully extracts inputs for:**
- **assistantbench**: 264/396 entries (8 models, 33 unique tasks)
- **corebench**: 356/538 entries (8 models, 45 unique tasks)
- **scicode**: 509/635 entries (9 models, 62 unique tasks)
- **taubench**: 350/550 entries (7 models, 50 unique tasks)

**Total: 1,479 task inputs** covering 155 unique tasks across 4 benchmarks and 12 models.

### How It Works

1. **Groups by (benchmark, model, task_id)**: Since `agent_run_id` doesn't reliably map to `trace_id` across all benchmarks, we extract one instance per unique (benchmark, model, task_id) combination.

2. **Finds trace files**: For each model, identifies candidate trace files by matching model name patterns in filenames.

3. **Extracts from first match**: For each task_id, extracts the input from the first LLM call found in the trace files:
   - Searches for entries with matching `weave_task_id` 
   - Extracts user messages from `inputs.messages`
   - Falls back to system messages if no user message found
   - Handles both simple string content and structured content (list of dicts with `type` and `text` fields)

4. **Creates aligned output**: Produces a CSV with columns: `benchmark_id`, `model`, `task_id`, `agent_run_id`, `task_input`

### Data Structure

**Input CSV** (`all_benchmarks_merged.csv`):
```csv
benchmark_id,model,task_id,agent_run_id,binary_success_rate,...
assistantbench,claude-opus-4.1,0ec4371...,a5cad1a1-...,1.0,...
taubench,claude-opus-4.1,0,d88cc7d2-...,0.0,...
```

**Output CSV** (`all_benchmarks_inputs.csv`):
```csv
benchmark_id,model,task_id,agent_run_id,task_input
assistantbench,claude-opus-4.1,0ec4371...,a5cad1a1-...,"What is the capital of France?..."
taubench,claude-opus-4.1,0,d88cc7d2-...,"You are a user interacting with..."
```

### Benchmark-Specific Details

#### Task ID Formats
- **assistantbench**: Hash-based (e.g., `0ec4371851b96837...`)
- **corebench**: Capsule IDs (e.g., `capsule-0504157`)
- **scicode**: Numeric (e.g., `2`, `5`, `8`)
- **taubench**: Numeric (e.g., `0`, `1`, `2`)

#### Alignment Key
All benchmarks use `weave_task_id` in the trace's `raw_logging_results` entries, which matches the `task_id` in the CSV.

### Example: Creating Embeddings

Once you have the aligned inputs, you can create embeddings:

```python
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the aligned data
df = pd.read_csv('all_benchmarks_inputs.csv')

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(df['task_input'].tolist(), show_progress_bar=True)

# Add to dataframe
df['embedding'] = embeddings.tolist()

# Or save embeddings separately
import numpy as np
np.save('task_embeddings.npy', embeddings)
```

### Key Files

- **`extract_inputs_simple.py`**: Main extraction script (RECOMMENDED)
- **`all_benchmarks_inputs.csv`**: Output with task inputs for all benchmarks
- **`extract_taubench_inputs.py`**: Legacy script (tau-bench only, deprecated)
- **`taubench_inputs_aligned.csv`**: Legacy output (deprecated)
- **`hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv`**: Original evaluation results
- **`traces/*.json`**: Trace files containing raw execution logs

### Coverage Details

Some entries are missing because:
1. **Missing trace files**: Some model variations (e.g., `gpt-5_high`, `o4-mini_low` for taubench) don't have corresponding trace files
2. **Task not found in traces**: Some tasks may not have been logged in available trace files
3. **Different agent configurations**: The CSV may contain results from multiple agent runs, but we extract only one input per (benchmark, model, task_id)

### Troubleshooting

**If you're missing entries:**
1. Check if trace files exist: `ls traces/*<benchmark>*<model>*.json`
2. Verify model name normalization matches filenames
3. Check if the task_id exists as `weave_task_id` in the trace file

**For debugging:**
```python
# Check what's in a trace file
import json
with open('traces/<your_file>.json') as f:
    data = json.load(f)
    task_ids = set()
    for entry in data['raw_logging_results']:
        if 'weave_task_id' in entry.get('attributes', {}):
            task_ids.add(entry['attributes']['weave_task_id'])
    print(f"Task IDs in trace: {sorted(task_ids)}")
```
