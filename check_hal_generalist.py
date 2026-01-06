#!/usr/bin/env python3
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load the merged matrix
p = Path('resmat/response_matrix_merged.pkl')
with open(p, 'rb') as f:
    df = pickle.load(f)

print("="*80)
print("ALL MODEL NAMES AFTER DEDUPLICATION")
print("="*80)
print(f"\nTotal models: {len(df.index)}\n")

# Group by type
hal_generalist = []
other_models = []

for idx in sorted(df.index):
    if idx.startswith('hal_generalist_'):
        hal_generalist.append(idx)
    else:
        other_models.append(idx)

print(f"\nHAL GENERALIST MODELS ({len(hal_generalist)}):")
print("-"*80)
for model in hal_generalist:
    print(model)

print(f"\n\nOTHER MODELS ({len(other_models)}):")
print("-"*80)
for model in other_models:
    print(model)

print("\n" + "="*80)
print("BEFORE DEDUPLICATION - Loading individual benchmark files")
print("="*80)

# Load all individual benchmark files
resmat_dir = Path('resmat')
pkl_files = [f for f in resmat_dir.glob("*.pkl") if f.name != "response_matrix_merged.pkl"]

all_models_before = []
models_by_benchmark = defaultdict(list)

for pkl_file in sorted(pkl_files):
    print(f"\nLoading {pkl_file.name}...")
    with open(pkl_file, 'rb') as f:
        df_bench = pickle.load(f)
        benchmark_name = pkl_file.stem.replace('response_matrix_', '')
        for idx in df_bench.index:
            all_models_before.append((idx, benchmark_name))
            models_by_benchmark[benchmark_name].append(idx)

print(f"\n\nTotal models before deduplication: {len(all_models_before)}")
print(f"Total models after deduplication: {len(df.index)}")
print(f"Models merged: {len(all_models_before) - len(df.index)}")

print("\n" + "="*80)
print("MODELS BY BENCHMARK (BEFORE DEDUPLICATION)")
print("="*80)

for benchmark in sorted(models_by_benchmark.keys()):
    print(f"\n{benchmark.upper()} ({len(models_by_benchmark[benchmark])} models):")
    print("-"*80)
    # Filter for hal_generalist
    hal_gen = [m for m in sorted(models_by_benchmark[benchmark]) if 'hal' in m.lower() and 'generalist' in m.lower()]
    others = [m for m in sorted(models_by_benchmark[benchmark]) if m not in hal_gen]
    
    if hal_gen:
        print("  HAL Generalist:")
        for model in hal_gen:
            print(f"    {model}")
    
    if others and len(others) <= 20:  # Only show others if list is reasonable
        print("  Others:")
        for model in others[:20]:
            print(f"    {model}")
        if len(others) > 20:
            print(f"    ... and {len(others) - 20} more")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Before deduplication: {len(all_models_before)} total entries")
print(f"After deduplication: {len(df.index)} unique models")
print(f"Reduction: {len(all_models_before) - len(df.index)} entries merged")
