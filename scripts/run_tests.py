# scripts/run_tests.py
import subprocess
import json
import re
import sys
from pathlib import Path

def run_tests():
    print("Running Test Suite...")
    
    # 1. Run pytest and generate both terminal and JSON coverage reports
    result = subprocess.run(
        [
            "uv", "run", "pytest", 
            "--cov=src/mi_datasets", 
            "--cov-report=term", 
            "--cov-report=json"
        ],
        check=False # We handle the exit code manually
    )
    
    if result.returncode != 0:
        print("\nERROR: Tests failed. Aborting pipeline. Fix the code before updating coverage.")
        sys.exit(result.returncode)

    print("\nParsing Coverage Data...")
    
    # 2. Extract the exact percentage from the JSON report
    cov_file = Path("coverage.json")
    if not cov_file.exists():
        print("ERROR: coverage.json was not generated.")
        sys.exit(1)

    with open(cov_file, "r") as f:
        data = json.load(f)
    
    # Extract percentage and round to 1 decimal place
    percent = round(float(data["totals"]["percent_covered_display"]), 1)
    
    # 3. Determine the badge color based on strict thresholds
    if percent >= 90:
        color = "brightgreen"
    elif percent >= 75:
        color = "yellow"
    elif percent >= 60:
        color = "orange"
    else:
        color = "red"

    # shields.io requires % to be url-encoded as %25
    badge_url = f"https://img.shields.io/badge/coverage-{percent}%25-{color}"

    # 4. Inject the new badge URL into the README.md
    readme = Path("README.md")
    if not readme.exists():
        print("ERROR: README.md not found in the root directory.")
        sys.exit(1)

    content = readme.read_text(encoding="utf-8")
    
    # Regex searches for the specific Coverage badge pattern and replaces the URL
    new_content, num_replacements = re.subn(
        r"!\[Coverage\]\(https://img\.shields\.io/badge/coverage-[^)]+\)",
        f"![Coverage]({badge_url})",
        content
    )

    if num_replacements == 0:
        print("WARNING: Could not find the Coverage badge placeholder in README.md.")
    else:
        readme.write_text(new_content, encoding="utf-8")
        print(f"SUCCESS: README.md successfully updated to: {percent}% coverage.")

    # 5. Clean up the JSON file so it doesn't clutter the workspace
    cov_file.unlink()

if __name__ == "__main__":
    run_tests()