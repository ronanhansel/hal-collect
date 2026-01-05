import pandas as pd
from pathlib import Path

def list_unique_testtakers_from_leaderboard():
    """List all unique test-takers (scaffold + model) from each CSV file in the leaderboard folder."""
    
    # Path to the leaderboard folder
    leaderboard_dir = Path(__file__).parent
    
    # Get all CSV files
    csv_files = sorted(leaderboard_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {leaderboard_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    print("=" * 80)
    
    # Store all test-takers across all files
    all_testtakers = set()
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}")
        print("-" * 80)
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if len(df.columns) >= 3:
                # Get the 2nd column (scaffold, index 1) and 3rd column (model, index 2)
                scaffold_column = df.iloc[:, 1]
                model_column = df.iloc[:, 2]
                
                # Create unique test-taker identifiers by combining scaffold + model
                testtakers = []
                for scaffold, model in zip(scaffold_column, model_column):
                    if pd.notna(scaffold) and pd.notna(model):
                        testtaker = f"{scaffold} + {model}"
                        testtakers.append(testtaker)
                        all_testtakers.add(testtaker)
                
                unique_testtakers = sorted(set(testtakers))
                
                print(f"Number of unique test-takers: {len(unique_testtakers)}")
                for testtaker in unique_testtakers:
                    print(f"  - {testtaker}")
            else:
                print(f"  ⚠️  File has fewer than 3 columns ({len(df.columns)} columns found)")
        
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"\nSUMMARY")
    print("-" * 80)
    print(f"Total unique test-takers across all files: {len(all_testtakers)}")
    print("\nAll unique test-takers:")
    for testtaker in sorted(all_testtakers):
        print(f"  - {testtaker}")

if __name__ == "__main__":
    list_unique_testtakers_from_leaderboard()
