import os
import glob
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from util.rename_helper import standardize_task_success_column, clean_model_name, clean_rubric_name

def compile_rubric_results_by_benchmark(directory, benchmark_name, save=True):
    rubric_files = glob.glob(os.path.join(directory, f"{benchmark_name}_*.csv"))
    if not rubric_files:
        print(f"No files found for benchmark {benchmark_name}")
        return pd.DataFrame()  # Return an empty DataFrame if no files found
    merged = compile_file_list(rubric_files)
    if save:
        merged.to_csv(os.path.join(directory, f"{benchmark_name}_merged.csv"), index=False)
    return merged  

def get_rubric_type(file_name):
    return file_name.split('_')[-1].replace('.csv', '')

def prettify_scaffold_name(scaffold_slug):
    """
    Convert scaffold slug to pretty name.
    Example: "usaco_episodic__semantic" -> "USACO Episodic + Semantic"
    """
    # Handle double underscores (replace with space)
    scaffold = scaffold_slug.replace('__', '_').replace('_', ' ')
    
    # Split into words and title case each
    words = scaffold.split()
    pretty_words = []
    
    for word in words:
        # Keep acronyms uppercase
        if word.upper() in ['USACO', 'HAL', 'TAU', 'SAB', 'SWE', 'HF', 'CORE']:
            pretty_words.append(word.upper())
        elif word.lower() in ['agent', 'bench', 'generalist', 'calling', 'shot', 'debug', 'research', 'use', 'open', 'deep', 'tool', 'zero', 'few', 'episodic', 'semantic']:
            pretty_words.append(word.capitalize())
        else:
            pretty_words.append(word.capitalize())
    
    # Join with ' + ' for better readability
    return ' '.join(pretty_words)

def extract_scaffold_from_filename(filename):
    """
    Extract scaffold from trace filename.
    Format: {benchmark}_{scaffold_parts}_{model}_{timestamp}_UPLOAD.json
    
    Examples:
    - usaco_usaco_episodic__semantic_claude37sonnet20250219_1744648510_UPLOAD.json
      -> scaffold: "usaco_episodic__semantic"
    - corebench_hard_core_agent_opus_45_1764027531_UPLOAD.json
      -> scaffold: "core_agent"
    - assistantbench_assistantbench_browser_agent_claude37sonnet20250219_1746383806_UPLOAD.json
      -> scaffold: "assistantbench_browser_agent"
    """
    basename = os.path.basename(filename).replace('_UPLOAD.json', '').replace('_UPLOAD.zip', '')
    
    # Split by underscore
    parts = basename.split('_')
    
    if len(parts) < 4:
        return None
    
    # Handle compound benchmarks (corebench_hard)
    benchmark_prefix_len = 1
    if len(parts) >= 2 and parts[0] == 'corebench' and parts[1] == 'hard':
        benchmark_prefix_len = 2
    
    # Last part is always timestamp (numeric), second to last might be 'high'/'low'/'medium'
    # Work backwards to find where the model ends
    timestamp_idx = len(parts) - 1
    
    # Find where model starts by looking for parts that look like model names
    # Model patterns: claude37sonnet20250219, gpt4o, deepseekr1, o3, opus, sonnet, etc.
    model_start_idx = None
    
    # Common model name patterns
    model_indicators = {
        'claude', 'gpt', 'deepseek', 'gemini', 'opus', 'sonnet', 'haiku', 
        'o3', 'o4', 'deepseekai', 'minimal'
    }
    
    for i in range(benchmark_prefix_len, timestamp_idx):
        part = parts[i].lower()
        
        # Check if this part is a known model indicator
        if any(indicator in part for indicator in model_indicators):
            model_start_idx = i
            break
        
        # Check if next part has a date pattern (8 digits)
        if i + 1 < timestamp_idx and re.match(r'^\d{8}', parts[i + 1]):
            model_start_idx = i
            break
    
    if model_start_idx is None or model_start_idx <= benchmark_prefix_len:
        return None
    
    # Everything between benchmark prefix and model is the scaffold
    scaffold_parts = parts[benchmark_prefix_len:model_start_idx]
    
    if not scaffold_parts:
        return None
    
    scaffold_slug = '_'.join(scaffold_parts)
    
    # Prettify the scaffold name
    scaffold = prettify_scaffold_name(scaffold_slug)
    
    return scaffold

def normalize_model_name(model):
    """Normalize model name for matching against filenames."""
    # Remove common suffixes like _high, _low, _medium before normalizing
    model = re.sub(r'_(high|low|medium)$', '', model)
    
    # Remove common suffixes and normalize
    normalized = model.lower().replace('-', '').replace('.', '').replace('_', '').replace(' ', '')
    # Remove parentheses and dates
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    return normalized

def build_scaffold_mapping_from_filenames(traces_dir="traces"):
    """
    Build a mapping from (benchmark, model) to scaffold by parsing trace filenames.
    This matches the approach used in extract_inputs_simple.py.
    """
    print(f"Building scaffold mapping from trace filenames...")
    trace_files = glob.glob(os.path.join(traces_dir, "*_UPLOAD.json"))
    
    # Build mapping: (benchmark, normalized_model) -> scaffold
    mapping = {}
    
    for trace_file in trace_files:
        basename = os.path.basename(trace_file)
        
        # Extract benchmark (first part before first underscore, or first two parts for corebench_hard)
        parts = basename.split('_')
        if len(parts) >= 2 and parts[0] == 'corebench' and parts[1] == 'hard':
            benchmark = 'corebench_hard'
        else:
            benchmark = parts[0]
        
        # Extract scaffold
        scaffold = extract_scaffold_from_filename(trace_file)
        if not scaffold:
            continue
        
        # Normalize the entire filename for model matching
        fname_norm = basename.lower().replace('-', '').replace('.', '').replace('_', '')
        
        # Store with the normalized filename portion for later matching
        mapping[(benchmark, fname_norm)] = scaffold
    
    print(f"  Built mapping for {len(mapping)} (benchmark, model) combinations")
    return mapping

def find_scaffold_for_row(benchmark_id, model, filename_mapping):
    """Find the scaffold for a given benchmark and model by matching against filenames."""
    # Normalize the model for matching
    model_norm = normalize_model_name(model)
    
    # Try exact benchmark match first
    for (bench, fname_norm), scaffold in filename_mapping.items():
        if bench == benchmark_id and model_norm in fname_norm:
            return scaffold
    
    # Try with variations (corebench vs corebench_hard)
    benchmark_variations = [benchmark_id]
    if benchmark_id == 'corebench':
        benchmark_variations.append('corebench_hard')
    
    for bench_var in benchmark_variations:
        for (bench, fname_norm), scaffold in filename_mapping.items():
            if bench == bench_var and model_norm in fname_norm:
                return scaffold
    
    return None


def compile_file_list(rubric_files):
    join_keys = ['benchmark_id', 'model', 'task_id', 'agent_run_id', "binary_success_rate"]

    dfs = []
    for rubric_file in rubric_files:
        df = pd.read_csv(rubric_file)
        
        # Standardize the success rate column name (currently different across benchmarks)
        df = standardize_task_success_column(df, benchmark_name=df['benchmark_id'].iloc[0])

        # Filter to keep relevant columns (only if they exist)
        filtered_columns = join_keys.copy()
        if 'label' in df.columns:
            filtered_columns.append('label')
        
        df = df[filtered_columns] # Note: A significant number of columns are dropped here
        
        # Rename rubric-specific columns to keep data organized
        rubric_type = get_rubric_type(rubric_file)
        rename_dict = {col: f"{rubric_type}.{col}" for col in df.columns if col not in join_keys}
        df = df.rename(columns=rename_dict)        
        dfs.append(df)

    # Outer join all dataframes
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=join_keys, how='outer')
    return merged

def main():
    benchmark_names = ["assistantbench", "corebench", "scicode", "taubench"]
    directory = "hal-paper-analysis/qualitative/results/rubrics"

    # Build scaffold mapping from trace filenames
    scaffold_map = build_scaffold_mapping_from_filenames()

    # # compile each benchmark into a single rubric file
    for benchmark_name in benchmark_names:
        benchmark_df = compile_rubric_results_by_benchmark(directory, benchmark_name)

    # Merge all benchmarks into a single file
    all_benchmark_files = glob.glob(os.path.join(directory, "*_merged.csv"))
    df_list = [pd.read_csv(rubric_file) for rubric_file in all_benchmark_files]
    all_merged = pd.concat(df_list, ignore_index=True)
    all_merged['model'] = all_merged['model'].apply(clean_model_name)
    
    # Add scaffold column using the mapping
    print("Adding scaffold column...")
    all_merged['scaffold'] = all_merged.apply(
        lambda row: find_scaffold_for_row(row['benchmark_id'], row['model'], scaffold_map),
        axis=1
    )
    
    # Reorder columns to put scaffold near the beginning
    columns = list(all_merged.columns)
    columns.remove('scaffold')
    columns.insert(2, 'scaffold')  # Insert after benchmark_id and model
    all_merged = all_merged[columns]
    
    # Report any missing scaffolds
    missing_scaffolds = all_merged['scaffold'].isna().sum()
    if missing_scaffolds > 0:
        print(f"  Warning: {missing_scaffolds} rows missing scaffold information")
        # Show which (benchmark, model) combos are missing
        missing_combos = all_merged[all_merged['scaffold'].isna()][['benchmark_id', 'model']].drop_duplicates()
        print(f"  Missing scaffolds for {len(missing_combos)} (benchmark, model) combinations:")
        for _, row in missing_combos.head(10).iterrows():
            print(f"    - {row['benchmark_id']}, {row['model']}")
    
    all_merged.to_csv(os.path.join(directory, "all_benchmarks_merged.csv"), index=False)
    print(f"Saved all_benchmarks_merged.csv with {len(all_merged)} rows and scaffold column")
    return 


if __name__ == "__main__":
    main()