# Rubric Regeneration Workflow

This guide explains how to regenerate rubric evaluations from your existing trace files.

## Overview

The rubric generation process involves:
1. **Upload**: Send trace files to Docent platform
2. **Evaluate**: Run rubric evaluations in Docent UI
3. **Download**: Export evaluation results as CSV files
4. **Compile**: Merge CSVs into consolidated rubric files

## Prerequisites

- Trace files already downloaded and decrypted in `traces/` directory
- Docent API key (get from https://docent.ai)
- Python environment with required packages

### Setup

```bash
# Install required packages
pip install docent python-dotenv pydantic

# Set up environment variables
echo "DOCENT_API_KEY=your_key_here" >> .env
```

## Step-by-Step Process

### Step 1: Upload Traces to Docent

Use the `regenerate_rubrics.py` script to upload your local traces:

```bash
# Upload all benchmarks
python regenerate_rubrics.py --verbose

# Upload specific benchmark only
python regenerate_rubrics.py --benchmark assistantbench --verbose

# Preview what would be uploaded (dry run)
python regenerate_rubrics.py --dry-run
```

This script:
- Scans your `traces/` directory for `*_UPLOAD.json` files
- Groups files by benchmark (assistantbench, corebench, scicode, etc.)
- Detects agent types (generalist vs specialist) from filenames
- Calls the appropriate v1_integration scripts to upload to Docent
- Creates collections named like `hal_<benchmark>_<agent_type>`

**Note**: The upload uses the existing integration scripts in `hal-paper-analysis/docent/v1_integration/`, which properly parse and format the trace data for Docent.

### Step 2: Run Evaluations in Docent

After upload completes:

1. Go to https://docent.ai
2. Navigate to your Collections
3. Find collections named like `hal_assistantbench_generalist`, `hal_assistantbench_specialist`, etc.
4. For each collection, run the following rubric evaluations:
   - **tooluse**: Evaluates tool usage patterns
   - **instructionfollowing**: Assesses instruction adherence
   - **selfcorrection**: Measures self-correction capabilities
   - **verification**: Checks verification behaviors
   - **environmentalbarrier**: Analyzes environmental constraint handling

5. Wait for evaluations to complete (this may take time depending on collection size)

### Step 3: Download CSV Results

For each benchmark and rubric combination:

1. In Docent UI, navigate to the evaluation results
2. Click "Export" or "Download CSV"
3. Save files with naming pattern: `<benchmark>_<rubric>.csv`
   - Example: `assistantbench_tooluse.csv`
   - Example: `corebench_instructionfollowing.csv`
4. Save all CSVs to: `hal-paper-analysis/qualitative/results/rubrics/`

**Expected CSV format:**
```csv
benchmark_id,model,task_id,agent_run_id,label,<other columns>
assistantbench,claude-3.7-sonnet,task_1,run_abc123,0.85,...
```

### Step 4: Compile Rubrics

Once all CSVs are downloaded, merge them:

```bash
python compile_rubric_results.py
```

This script:
- Reads all individual rubric CSVs from `hal-paper-analysis/qualitative/results/rubrics/`
- Merges rubrics for each benchmark (creates `<benchmark>_merged.csv`)
- Combines all benchmarks (creates `all_benchmarks_merged.csv`)
- Applies model name cleaning and standardization

**Output files:**
- `assistantbench_merged.csv` - All rubrics for AssistantBench
- `corebench_merged.csv` - All rubrics for CoreBench
- `all_benchmarks_merged.csv` - All rubrics for all benchmarks

## Troubleshooting

### Upload Issues

**Problem**: Script can't find v1_integration scripts
```
⚠️  Warning: Script not found: hal-paper-analysis/docent/v1_integration/assistant_bench.py
```
**Solution**: Make sure the `hal-paper-analysis` repository is cloned and up to date

**Problem**: Missing DOCENT_API_KEY
```
⚠️  Warning: DOCENT_API_KEY not found in environment
```
**Solution**: Add your API key to `.env` file or export it: `export DOCENT_API_KEY=your_key`

### Upload Script Failures

If a specific integration script fails, you can run it manually:

```bash
cd hal-paper-analysis/docent/v1_integration

# For AssistantBench
python assistant_bench.py --agent-type generalist --traces-dir ../../../traces --verbose
python assistant_bench.py --agent-type specialist --traces-dir ../../../traces --verbose

# For other benchmarks, use corresponding scripts:
# core_bench.py, sci_code.py, tau_bench.py, gaia.py, etc.
```

### Compilation Issues

**Problem**: Missing CSV files
```
No files found for benchmark assistantbench
```
**Solution**: Download all required rubric CSVs from Docent first

**Problem**: Column mismatch errors
**Solution**: Ensure CSV files match the expected format with `benchmark_id`, `model`, `task_id`, `agent_run_id`, and `label` columns

## File Structure

```
hal-collect/
├── traces/                              # Your decrypted trace files
│   ├── assistantbench_*_UPLOAD.json
│   ├── corebench_*_UPLOAD.json
│   └── ...
├── hal-paper-analysis/
│   ├── docent/v1_integration/          # Upload scripts
│   │   ├── assistant_bench.py
│   │   ├── core_bench.py
│   │   └── ...
│   └── qualitative/results/rubrics/    # CSV results
│       ├── assistantbench_tooluse.csv
│       ├── assistantbench_instructionfollowing.csv
│       ├── assistantbench_merged.csv
│       └── all_benchmarks_merged.csv
├── regenerate_rubrics.py               # Step 1: Upload script
└── compile_rubric_results.py           # Step 4: Compilation script
```

## Advanced Usage

### Process Specific Benchmarks

```bash
# Only upload AssistantBench
python regenerate_rubrics.py --benchmark assistantbench

# Only upload CoreBench
python regenerate_rubrics.py --benchmark corebench
```

### Custom Trace Directory

```bash
python regenerate_rubrics.py --traces-dir /path/to/custom/traces
```

### Verbose Output

```bash
python regenerate_rubrics.py --verbose
```

This shows detailed output from the v1_integration upload scripts.

## Notes

- **Upload is one-way**: Traces are uploaded to Docent but not automatically evaluated
- **Manual evaluation required**: You must trigger evaluations in the Docent UI
- **CSV download is manual**: Results must be downloaded from Docent UI
- **Existing rubrics**: If you already have rubric CSVs and just want to recompile, skip steps 1-3 and run `compile_rubric_results.py` directly

## Support

For issues with:
- **Trace upload**: Check v1_integration scripts in `hal-paper-analysis/docent/v1_integration/`
- **Docent evaluations**: Contact Docent support or check their documentation
- **CSV compilation**: See `compile_rubric_results.py` and `util/rename_helper.py`
