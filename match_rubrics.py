#!/usr/bin/env python3
"""
Match rubrics with response matrix.
Aligns rubric labels from all_benchmarks_merged.csv with the normalized response matrix.
Exports 5 rubric matrices (one for each subtask) as CSV files.

Output format:
- Each rubric matrix has shape (test_takers x tasks) matching result_matrix_normalized.csv
- Rows: test_taker_id (e.g., 'assistantbench_browser_agent:claude_3_7_sonnet_20250219')
- Columns: benchmark.task_id (e.g., 'assistantbench.ccec2229ced...')
- Values: rubric scores (0 or 1) for each (test_taker, task) combination, NaN if no rubric data

The 5 rubric types are:
1. environmentalbarrier - ability to overcome environmental constraints
2. instructionfollowing - adherence to task instructions
3. selfcorrection - ability to detect and fix errors
4. tooluse - effective use of available tools
5. verification - validation of outputs
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add tools to path for shared naming utilities
sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from naming import generate_test_taker_id


def load_data():
    """Load all necessary data files."""
    print("Loading data files...")
    
    # Load normalized result matrix (our target)
    result_matrix = pd.read_csv('result/result_matrix_normalized.csv')
    result_matrix = result_matrix.set_index('test_taker_id')
    print(f"✓ Result matrix (normalized) loaded: {result_matrix.shape}")
    
    # Load original result matrix to get agent_name and model_name mapping
    result_matrix_orig = pd.read_csv('result/result_matrix.csv')
    print(f"✓ Result matrix (original) loaded: {len(result_matrix_orig)} rows")
    
    # Load transcript
    transcript = pd.read_csv('output/transcripts.csv')
    print(f"✓ Transcript loaded: {len(transcript)} rows")
    
    # Load rubrics
    rubrics = pd.read_csv('hal-paper-analysis/qualitative/results/rubrics/all_benchmarks_merged.csv')
    print(f"✓ Rubrics loaded: {len(rubrics)} rows")
    
    return result_matrix, result_matrix_orig, transcript, rubrics


def match_rubrics_to_result_matrix(rubrics, transcript, result_matrix_orig, result_matrix):
    """
    Match rubrics to result matrix index using shared naming logic.
    
    Strategy:
    1. Join rubrics with transcript on agent_run_id to get task_id and benchmark_id
    2. Join with result_matrix_orig on (benchmark, task_id) to get actual agent_name and model_name
    3. Generate test_taker_id using the agent_name and model_name from result_matrix_orig
    4. This ensures perfect alignment since we use the same values that created result_matrix
    
    Returns:
        Dict of rubric matrices (one per rubric type), and matched rubrics DataFrame
    """
    print("\nMatching rubrics to result matrix...")
    
    # Rubric columns (excluding 'match' which is for verification)
    rubric_cols = [
        'environmentalbarrier.label',
        'instructionfollowing.label', 
        'selfcorrection.label',
        'tooluse.label',
        'verification.label'
    ]
    
    # Step 1: Join rubrics with transcript to get task_id, benchmark_id, and run_id
    # The run_id links to the specific trace file that produced this rubric
    print(f"  Joining rubrics with transcript on agent_run_id...")
    rubrics_with_transcript = rubrics.merge(
        transcript[['agent_run_id', 'benchmark_id', 'task_id', 'run_id', 'model']],
        on='agent_run_id',
        how='left',
        suffixes=('_rubric', '_transcript')
    )
    
    matched_count = rubrics_with_transcript['benchmark_id_transcript'].notna().sum()
    print(f"    Matched {matched_count}/{len(rubrics)} rubrics with transcript")
    
    # Use model from transcript as authoritative source (it's cleaned in compile_traces)
    rubrics_with_transcript['model_final'] = rubrics_with_transcript['model_transcript'].fillna(
        rubrics_with_transcript['model_rubric']
    )
    
    
    # Use model from transcript as authoritative source (it's cleaned in compile_traces)
    rubrics_with_transcript['model_final'] = rubrics_with_transcript['model_transcript'].fillna(
        rubrics_with_transcript['model_rubric']
    )
    rubrics_with_transcript['benchmark_final'] = rubrics_with_transcript['benchmark_id_transcript'].fillna(
        rubrics_with_transcript['benchmark_id_rubric']
    )
    rubrics_with_transcript['task_id_final'] = rubrics_with_transcript['task_id_transcript'].fillna(
        rubrics_with_transcript['task_id_rubric']
    )
    
    # Step 2: Join with result_matrix_orig to get agent_name for generating test_taker_id
    # The key insight: we match on run_id (from transcript) to find the exact row in result_matrix
    # that corresponds to this rubric annotation
    print(f"  Joining with result_matrix to match rubrics to specific test_takers...")
    
    # First, add run_id to result_matrix_orig (extract from benchmark_name or another source)
    # Actually, result_matrix doesn't have run_id. We need a different strategy.
    # 
    # Strategy: For each rubric, we have:
    #   - agent_run_id (unique identifier for the run)
    #   - model from transcript (authoritative)
    #   - benchmark and task_id
    # 
    # For result_matrix_orig, we have:
    #   - agent_name and model_name
    #   - task columns
    #
    # The issue is that multiple rows in result_matrix can have the same (agent_name, model_name, benchmark)
    # because different "efforts" (high/low) create separate rows.
    #
    # Solution: We need to extract the effort from the rubric's model name and match exactly.
    
    print(f"  Generating test_taker_id for result_matrix rows...")
    result_matrix_orig['test_taker_id'] = result_matrix_orig.apply(
        lambda row: generate_test_taker_id(row['agent_name'], row['model_name']),
        axis=1
    )
    
    # Step 3: For each rubric, find the matching row in result_matrix_orig
    # Match on: model_name AND task column existence
    print(f"  Matching rubrics to specific test_takers...")
    rubrics_matched_list = []
    
    for _, rubric_row in rubrics_with_transcript.iterrows():
        benchmark = rubric_row['benchmark_final']
        task_id = str(rubric_row['task_id_final'])
        model = rubric_row['model_final']  # This is from transcript, already cleaned
        
        if pd.isna(benchmark) or pd.isna(task_id) or pd.isna(model) or task_id == 'nan':
            continue
        
        # Find task column in result_matrix
        task_col = f"{benchmark}.{task_id}"
        matching_task_cols = []
        
        # Get all task columns (exclude metadata)
        task_columns = [col for col in result_matrix_orig.columns if col not in ['benchmark_name', 'agent_name', 'model_name', 'test_taker_id']]
        
        if task_col in task_columns:
            matching_task_cols.append(task_col)
        else:
            # Try partial benchmark match
            for col in task_columns:
                if '.' in col:
                    col_benchmark, col_task = col.split('.', 1)
                    if col_benchmark.startswith(benchmark) and col_task == task_id:
                        matching_task_cols.append(col)
                        break  # Take first match
        
        if not matching_task_cols:
            continue
        
        task_col = matching_task_cols[0]
        
        # Now find rows in result_matrix_orig that:
        # 1. Have this task column with data
        # 2. Have model_name matching the rubric's model
        rows_with_data = result_matrix_orig[
            (result_matrix_orig[task_col].notna()) &
            (result_matrix_orig['model_name'] == model)
        ]
        
        # Add rubric data to each matching row
        for _, row in rows_with_data.iterrows():
            rubric_row_copy = rubric_row.copy()
            rubric_row_copy['agent_name'] = row['agent_name']
            rubric_row_copy['model_name'] = row['model_name']
            rubric_row_copy['test_taker_id'] = row['test_taker_id']
            rubric_row_copy['task_column'] = task_col
            rubrics_matched_list.append(rubric_row_copy)
    
    rubrics_matched = pd.DataFrame(rubrics_matched_list)
    
    print(f"  Matched {len(rubrics_matched)}/{len(rubrics)} rubric entries to result matrix")
    
    if len(rubrics_matched) > 0:
        print(f"  Unique test_taker_id values: {rubrics_matched['test_taker_id'].nunique()}")
        print(f"  Benchmarks matched: {sorted(rubrics_matched['benchmark_final'].unique())}")
        print(f"  Rubrics by benchmark:")
        for benchmark, count in rubrics_matched['benchmark_final'].value_counts().items():
            print(f"    {benchmark}: {count}")
    else:
        print("  WARNING: No rubrics matched!")
        return {}, pd.DataFrame()
    
    print(f"  Created test_taker_id for {len(rubrics_matched)} rubric entries")
    print(f"  Unique test_taker_id values: {rubrics_matched['test_taker_id'].nunique()}")
    print(f"  Sample test_taker_id values:")
    for idx in rubrics_matched['test_taker_id'].unique()[:5]:
        print(f"    {idx}")
    
    # Check which indices from rubrics are in result_matrix
    rubrics_indices = set(rubrics_matched['test_taker_id'].unique())
    result_indices = set(result_matrix.index)
    matched_indices = rubrics_indices & result_indices
    unmatched_indices = rubrics_indices - result_indices
    
    print(f"\n  Index matching analysis:")
    print(f"    Indices from rubrics: {len(rubrics_indices)}")
    print(f"    Indices in result matrix: {len(result_indices)}")
    print(f"    Matched: {len(matched_indices)}")
    print(f"    Unmatched (in rubrics but not in result matrix): {len(unmatched_indices)}")
    
    if len(unmatched_indices) > 0:
        print(f"\n  ⚠ Unmatched indices (in rubrics but NOT in result matrix): {len(unmatched_indices)}")
        for idx in sorted(list(unmatched_indices))[:10]:
            print(f"    {idx}")
    
    # Also check which result_matrix indices don't have rubric data
    missing_rubrics = result_indices - rubrics_indices
    if len(missing_rubrics) > 0:
        print(f"\n  ℹ Indices in result matrix but NOT in rubrics: {len(missing_rubrics)}")
        for idx in sorted(list(missing_rubrics))[:10]:
            print(f"    {idx}")
    
    # Convert rubric labels to binary values
    for col in rubric_cols:
        rubrics_matched[col] = rubrics_matched[col].apply(lambda x: 
            1 if x in ['match', True, 'True', 1] 
            else 0 if x in ['no match', False, 'False', 0]
            else float('nan')
        )
    
    # task_column is already set during matching
    # Create aligned dataframes for each rubric type
    # Now we create matrices with same shape as result_matrix (test_takers x tasks)
    rubric_matrices = {}
    
    print(f"\n  Creating rubric matrices (test_takers x tasks)...")
    for col in rubric_cols:
        rubric_name = col.replace('.label', '')
        
        # Create empty DataFrame with same shape as result_matrix
        rubric_matrix = pd.DataFrame(
            index=result_matrix.index,
            columns=result_matrix.columns,
            dtype=float
        )
        
        # Fill in rubric scores for each (test_taker_id, task) combination
        matched_count = 0
        for _, row in rubrics_matched.iterrows():
            test_taker_id = row['test_taker_id']
            task_column = row['task_column']
            rubric_value = row[col]
            
            # Check if this test_taker and task exist in result_matrix
            if test_taker_id in rubric_matrix.index and task_column in rubric_matrix.columns:
                if pd.notna(rubric_value):
                    rubric_matrix.loc[test_taker_id, task_column] = rubric_value
                    matched_count += 1
        
        rubric_matrices[rubric_name] = rubric_matrix
        
        non_nan_count = rubric_matrix.notna().sum().sum()
        print(f"    {rubric_name}: {non_nan_count} task-level rubric scores matched")
        print(f"      Test takers with data: {rubric_matrix.notna().any(axis=1).sum()}/{len(result_matrix)}")
        print(f"      Tasks with data: {rubric_matrix.notna().any(axis=0).sum()}/{len(result_matrix.columns)}")
    
    return rubric_matrices, rubrics_matched


def export_rubric_matrices(rubric_matrices, result_matrix):
    """
    Export rubric matrices as CSV files with same dimension and order as result_matrix.
    Each matrix has test_taker_id rows and task columns.
    """
    print("\nExporting rubric matrices...")
    
    # Create rubrics directory if it doesn't exist
    os.makedirs('rubrics', exist_ok=True)
    
    for rubric_name, rubric_df in rubric_matrices.items():
        # Verify dimensions match
        assert rubric_df.shape == result_matrix.shape, f"Shape mismatch for {rubric_name}"
        assert all(rubric_df.index == result_matrix.index), f"Index mismatch for {rubric_name}"
        assert all(rubric_df.columns == result_matrix.columns), f"Columns mismatch for {rubric_name}"
        
        # Reset index to include test_taker_id as a column
        rubric_df_export = rubric_df.copy()
        rubric_df_export.insert(0, 'test_taker_id', rubric_df_export.index)
        
        output_path = f'rubrics/rubrics_matrix_{rubric_name}.csv'
        rubric_df_export.to_csv(output_path, index=False)
        
        non_nan = rubric_df.notna().sum().sum()
        print(f"  ✓ Exported {output_path}")
        print(f"    Shape: {rubric_df.shape}, Non-NaN values: {non_nan}")


def verify_alignment(rubric_matrices, result_matrix):
    """
    Verify that rubric matrices are properly aligned with result matrix.
    """
    print("\n" + "="*60)
    print("VERIFICATION: Checking alignment")
    print("="*60)
    
    print(f"\nResult matrix:")
    print(f"  Shape: {result_matrix.shape}")
    print(f"  Index length: {len(result_matrix.index)}")
    print(f"  Columns: {len(result_matrix.columns)}")
    print(f"  First 3 indices: {result_matrix.index[:3].tolist()}")
    print(f"  Last 3 indices: {result_matrix.index[-3:].tolist()}")
    
    all_aligned = True
    for rubric_name, rubric_df in rubric_matrices.items():
        print(f"\nRubric matrix '{rubric_name}':")
        print(f"  Shape: {rubric_df.shape}")
        print(f"  Index length: {len(rubric_df.index)}")
        print(f"  Columns: {len(rubric_df.columns)}")
        
        # Check exact alignment
        if rubric_df.shape != result_matrix.shape:
            print(f"  ✗ SHAPE MISMATCH: {rubric_df.shape} != {result_matrix.shape}")
            all_aligned = False
        elif not all(rubric_df.index == result_matrix.index):
            print(f"  ✗ INDEX ORDER MISMATCH")
            all_aligned = False
        elif not all(rubric_df.columns == result_matrix.columns):
            print(f"  ✗ COLUMNS ORDER MISMATCH")
            all_aligned = False
        else:
            print(f"  ✓ Perfectly aligned with result matrix")
            non_nan = rubric_df.notna().sum().sum()
            print(f"  Non-NaN values: {non_nan} ({non_nan / rubric_df.size * 100:.2f}% coverage)")
            print(f"  Test takers with data: {rubric_df.notna().any(axis=1).sum()}/{len(rubric_df)}")
            print(f"  Tasks with data: {rubric_df.notna().any(axis=0).sum()}/{len(rubric_df.columns)}")
    
    if all_aligned:
        print("\n" + "="*60)
        print("✓ ALL RUBRIC MATRICES ARE PROPERLY ALIGNED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ ALIGNMENT ISSUES DETECTED")
        print("="*60)
    
    return all_aligned


def main():
    """Main execution function."""
    print("="*60)
    print("RUBRIC MATCHING SCRIPT")
    print("="*60)
    
    # Load data
    result_matrix, result_matrix_orig, transcript, rubrics = load_data()
    
    # Match rubrics to result matrix using shared naming logic
    rubric_matrices, rubrics_matched = match_rubrics_to_result_matrix(rubrics, transcript, result_matrix_orig, result_matrix)
    
    # Export rubric matrices as CSV
    export_rubric_matrices(rubric_matrices, result_matrix)
    
    # Verify alignment
    verify_alignment(rubric_matrices, result_matrix)
    
    print("\n" + "="*60)
    print("✓ SCRIPT COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
