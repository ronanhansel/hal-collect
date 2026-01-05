#!/usr/bin/env python3
"""
Upload Traces to Docent for Rubric Evaluation

This script uses the existing v1_integration scripts to upload your local traces
to Docent for evaluation. After upload, you need to run evaluations in Docent,
then download the CSV results.

Workflow:
1. Run this script to upload traces to Docent
2. Go to Docent UI and run rubric evaluations on the collections
3. Download CSV results from Docent
4. Run compile_rubric_results.py to merge the CSVs

Usage:
    python regenerate_rubrics.py --traces-dir traces/ --benchmark assistantbench
"""

import argparse
import glob
import os
import sys
import subprocess
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv



def extract_benchmark_from_filename(filename: str) -> str:
    """Extract benchmark name from filename (e.g., 'assistantbench' from 'assistantbench_..._UPLOAD.json')."""
    parts = filename.split('_')
    # Map filenames to benchmark script names
    benchmark_map = {
        'assistantbench': 'assistant_bench',
        'corebench': 'core_bench',
        'scicode': 'sci_code',
        'taubench': 'tau_bench',
        'gaia': 'gaia',
        'scienceagentbench': 'scienceagent',
        'swebench': 'swe_bench_mini',
        'usaco': 'usaco',
    }
    benchmark_key = parts[0] if parts else "unknown"
    return benchmark_map.get(benchmark_key.lower(), benchmark_key)


def find_trace_files(traces_dir: str, benchmark: Optional[str] = None) -> dict:
    """Find all trace files, grouped by benchmark."""
    pattern = f"{traces_dir}/*_UPLOAD.json"
    all_files = glob.glob(pattern)
    
    # Group by benchmark
    benchmarks = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        benchmark_key = filename.split('_')[0].lower()
        
        # Filter if specified
        if benchmark and benchmark_key != benchmark.lower():
            continue
        
        if benchmark_key not in benchmarks:
            benchmarks[benchmark_key] = []
        benchmarks[benchmark_key].append(file_path)
    
    return benchmarks


def get_agent_type(filename: str) -> str:
    """Determine if this is a generalist or specialist agent from filename."""
    # generalist: *_hal_generalist_agent_*
    # specialist: *_<benchmark>_*_agent_* or *_browser_agent_*
    if '_hal_generalist_agent_' in filename:
        return 'generalist'
    else:
        return 'specialist'


def upload_benchmark_to_docent(benchmark: str, trace_files: List[str], verbose: bool = False) -> None:
    """Upload a benchmark's traces to Docent using the v1_integration scripts."""
    
    script_name = extract_benchmark_from_filename(benchmark)
    script_path = f"hal-paper-analysis/docent/v1_integration/{script_name}.py"
    
    if not os.path.exists(script_path):
        print(f"⚠️  Warning: Script not found: {script_path}")
        print(f"   Skipping {benchmark}")
        return
    
    # Determine agent types present in files
    agent_types = set()
    for trace_file in trace_files:
        agent_types.add(get_agent_type(os.path.basename(trace_file)))
    
    print(f"\n{'='*80}")
    print(f"Uploading {benchmark.upper()}")
    print(f"  Traces: {len(trace_files)} files")
    print(f"  Agent types: {', '.join(sorted(agent_types))}")
    print(f"{'='*80}\n")
    
    # Run upload for each agent type
    for agent_type in sorted(agent_types):
        print(f"\n📤 Uploading {agent_type} agent traces...")
        
        cmd = [
            "python",
            script_path,
            "--agent-type", agent_type,
            "--traces-dir", os.path.dirname(trace_files[0])
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"✅ {agent_type} upload successful")
                if verbose and result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ {agent_type} upload failed")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Error running {script_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload traces to Docent for rubric evaluation",
        epilog="""
Workflow:
  1. Run this script to upload traces to Docent collections
  2. Go to Docent UI (https://docent.ai) and run rubric evaluations
  3. Download CSV results from Docent to hal-paper-analysis/qualitative/results/rubrics/
  4. Run compile_rubric_results.py to merge the CSVs

Example:
  python regenerate_rubrics.py --benchmark assistantbench --verbose
        """
    )
    parser.add_argument("--traces-dir", default="traces", 
                       help="Directory containing trace JSON files (default: traces)")
    parser.add_argument("--benchmark", 
                       help="Process specific benchmark only (e.g., 'assistantbench', 'corebench')")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output from upload scripts")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be uploaded without actually uploading")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for required env vars
    if not os.getenv("DOCENT_API_KEY"):
        print("⚠️  Warning: DOCENT_API_KEY not found in environment")
        print("   Make sure it's set in your .env file or environment variables")
        print("   The v1_integration scripts will need it.")
    
    # Find trace files grouped by benchmark
    benchmarks = find_trace_files(args.traces_dir, args.benchmark)
    
    if not benchmarks:
        print(f"❌ No trace files found in {args.traces_dir}")
        if args.benchmark:
            print(f"   (filtered for benchmark: {args.benchmark})")
        sys.exit(1)
    
    print(f"\n📊 Found {len(benchmarks)} benchmark(s) to process:")
    total_files = 0
    for benchmark, files in sorted(benchmarks.items()):
        agent_types = set(get_agent_type(os.path.basename(f)) for f in files)
        print(f"  - {benchmark:20s}: {len(files):3d} files ({', '.join(sorted(agent_types))})")
        total_files += len(files)
    print(f"\nTotal: {total_files} trace files")
    
    if args.dry_run:
        print("\n🏃 Dry run mode - no uploads will be performed")
        return
    
    # Confirm before proceeding
    print("\n" + "="*80)
    response = input("Proceed with upload to Docent? (y/N): ")
    if response.lower() not in ('y', 'yes'):
        print("❌ Cancelled")
        sys.exit(0)
    
    # Process each benchmark
    for benchmark, files in sorted(benchmarks.items()):
        upload_benchmark_to_docent(benchmark, files, verbose=args.verbose)
    
    print(f"\n{'='*80}")
    print("✅ Upload complete!")
    print(f"{'='*80}")
    print("\n📋 Next steps:")
    print("  1. Go to Docent UI: https://docent.ai")
    print("  2. Find your collections (named like 'hal_<benchmark>_<agent_type>')")
    print("  3. Run rubric evaluations on each collection:")
    print("     - tooluse")
    print("     - instructionfollowing")
    print("     - selfcorrection")
    print("     - verification")
    print("     - environmentalbarrier")
    print("  4. Download evaluation results as CSV files")
    print("  5. Save CSVs to: hal-paper-analysis/qualitative/results/rubrics/")
    print("  6. Run: python compile_rubric_results.py")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
