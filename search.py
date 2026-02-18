from pathlib import Path

def scan_files():
    print("--- SYSTEM FILE SCAN ---")
    current_dir = Path.cwd()
    print(f"Scanning from: {current_dir}\n")
    
    # List all files recursively
    all_files = list(current_dir.rglob("*"))
    
    if not all_files:
        print("No files found at all. Are you in the right directory?")
        return

    # Filter for CSVs and print them clearly
    print("Found the following CSV files:")
    csv_found = False
    for f in all_files:
        if f.suffix.lower() == ".csv":
            # Highlight files that might be House-related
            tag = "[MATCH?]" if "house" in f.name.lower() or "rep" in f.name.lower() else ""
            print(f" - {f.relative_to(current_dir)} {tag}")
            csv_found = True
    
    if not csv_found:
        print("No CSV files found in the entire directory tree.")
    
    print("\n--- SCAN COMPLETE ---")

if __name__ == "__main__":
    scan_files()
