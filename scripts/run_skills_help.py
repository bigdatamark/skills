#!/usr/bin/env python3
# [MODULE DOC] Smoke test: run every Python script under skills directories with --help via uv.
# Behavior: finds Python files under "skills" directories, runs each as "uv run <file> --help"
# with a 30 second timeout, prints each result and a final summary.
# CLI arguments: none. Env vars: none. Constants: none.
# Top-level code: runs main when executed as a script.
# Side effects: spawns uv subprocesses (which may install dependencies); prints to stdout.
# Outputs: always exits 0, even when scripts fail.
"""
Script to execute all Python programs under "skills" directories with `uv run` and --help flag.
"""

import subprocess
from pathlib import Path

def find_python_files():
    # [FUNCTION DOC] Find all Python files under skills directories.
    # Behavior: walks from the current working directory matching the pattern "../skills"; for each
    # match that is a directory, collects all .py files recursively; deduplicates and sorts.
    # Inputs: none (depends on the current working directory).
    # Outputs: sorted list of file paths.
    # Exceptions: none caught; the path library may reject the pattern (see known issue).
    # Side effects: filesystem traversal.
    # KNOWN ISSUE: a recursive glob for "../skills" is not a sensible pattern - depending on the
    # Python version it matches nothing or raises an invalid-pattern error, so the intended skills
    # directories are likely never found; a pattern like "skills" was probably intended.
    """Find all Python files under skills directories."""
    python_files = []
    
    # Search for skills directories and find Python files
    for skills_dir in Path('.').rglob('../skills'):
        if skills_dir.is_dir():
            python_files.extend(skills_dir.rglob('*.py'))
    
    return sorted(set(python_files))

def run_with_help(python_file):
    # [FUNCTION DOC] Run one Python file with "uv run <file> --help" and print the outcome.
    # Behavior: prints a banner and the file name, runs the command capturing output (30 s timeout);
    # on exit code 0 prints SUCCESS and stdout, otherwise prints FAILED, the code, stderr and stdout.
    # Inputs: python_file - path of the script to run.
    # Outputs: False only when the uv executable is not found; True in every other case.
    # Exceptions: catches timeout (prints TIMEOUT), FileNotFoundError (prints uv-missing error),
    # and any other exception (prints ERROR with the message); nothing propagates.
    # Side effects: spawns a subprocess; prints to stdout.
    # KNOWN ISSUE: returns True for non-zero exits, timeouts and other errors, so the caller counts
    # real failures as successes.
    """Run a Python file with uv run --help."""
    try:
        print(f"\n{'='*60}")
        print(f"Running: {python_file}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            ['uv', 'run', str(python_file), '--help'],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            print("SUCCESS - Output:")
            print(result.stdout)
        else:
            print(f"FAILED - Return code: {result.returncode}")
            if result.stderr:
                print("Stderr:")
                print(result.stderr)
            if result.stdout:
                print("Stdout:")
                print(result.stdout)
                
    except subprocess.TimeoutExpired:
        print("TIMEOUT - Command took too long")
    except FileNotFoundError:
        print("ERROR - uv command not found. Please install uv.")
        return False
    except Exception as e:
        print(f"ERROR - {str(e)}")
    
    return True

def main():
    # [FUNCTION DOC] Entry point: find skill scripts, run each with --help and print a summary.
    # Behavior: prints a progress line, finds files; if none, prints a message and returns. Otherwise
    # runs each file, counting True results as successes and False as failures, then prints totals.
    # Inputs: none.
    # Outputs: none returned; summary printed to stdout.
    # Exceptions: none caught here.
    # Side effects: subprocesses and printing via helpers.
    """Main function to find and run all Python files."""
    print("Finding Python files under skills directories...")
    python_files = find_python_files()
    
    if not python_files:
        print("No Python files found under skills directories.")
        return
    
    print(f"Found {len(python_files)} Python files")
    
    success_count = 0
    failed_count = 0
    
    for python_file in python_files:
        if run_with_help(python_file):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Total files: {len(python_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
