import json
from pathlib import Path
from collections import defaultdict

def extract_scaffold_from_agent_name(agent_name):
    """
    Extract scaffold from agent_name by splitting on the first '('.
    Example: "HAL Generalist Agent (GPT-4.1)" -> "HAL Generalist Agent"
    """
    if not agent_name or '(' not in agent_name:
        return agent_name
    return agent_name.split('(', 1)[0].strip()

def list_scaffolds_from_traces():
    """List all unique scaffolds from trace files."""
    
    # Path to the traces folder
    traces_dir = Path(__file__).parent / "traces"
    
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        return
    
    # Get all JSON trace files
    trace_files = sorted(traces_dir.glob("*_UPLOAD.json"))
    
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return
    
    print(f"Found {len(trace_files)} trace files\n")
    print("=" * 80)
    
    # Store scaffold information
    scaffold_to_benchmarks = defaultdict(set)
    scaffold_to_models = defaultdict(set)
    all_scaffolds = set()
    benchmark_counts = defaultdict(int)
    
    for trace_file in trace_files:
        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
            
            # Extract agent_name and benchmark_name from config
            if 'config' in data:
                agent_name = data['config'].get('agent_name', '')
                benchmark_name = data['config'].get('benchmark_name', '')
                model_name = data['config'].get('agent_args', {}).get('model_name', '')
                
                # Extract scaffold
                scaffold = extract_scaffold_from_agent_name(agent_name)
                
                if scaffold:
                    all_scaffolds.add(scaffold)
                    scaffold_to_benchmarks[scaffold].add(benchmark_name)
                    if model_name:
                        scaffold_to_models[scaffold].add(model_name)
                    benchmark_counts[benchmark_name] += 1
        
        except Exception as e:
            print(f"  ⚠️  Error reading {trace_file.name}: {e}")
    
    # Print results organized by scaffold
    print("\nUNIQUE SCAFFOLDS FOUND:")
    print("-" * 80)
    
    for scaffold in sorted(all_scaffolds):
        print(f"\n📋 {scaffold}")
        print(f"   Benchmarks: {', '.join(sorted(scaffold_to_benchmarks[scaffold]))}")
        if scaffold_to_models[scaffold]:
            print(f"   Models: {len(scaffold_to_models[scaffold])} unique model(s)")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"\nSUMMARY")
    print("-" * 80)
    print(f"Total unique scaffolds: {len(all_scaffolds)}")
    print(f"\nAll scaffolds:")
    for scaffold in sorted(all_scaffolds):
        print(f"  - {scaffold}")
    
    print(f"\nBenchmarks covered:")
    for benchmark, count in sorted(benchmark_counts.items()):
        print(f"  - {benchmark}: {count} trace file(s)")

if __name__ == "__main__":
    list_scaffolds_from_traces()
