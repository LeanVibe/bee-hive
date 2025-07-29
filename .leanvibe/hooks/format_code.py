#!/usr/bin/env python3
"""Code formatting hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys
from pathlib import Path

def format_code():
    """Auto-format code according to project standards."""
    try:
        project_root = Path(os.environ.get("LEANVIBE_PROJECT_ROOT", "."))
        
        # Format Python files with black
        python_files = list(project_root.rglob("*.py"))
        if python_files:
            try:
                subprocess.run(
                    ["black", "--line-length", "88"] + [str(f) for f in python_files],
                    check=True,
                    timeout=120
                )
                print(f"Formatted {len(python_files)} Python files")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Black formatter not available, skipping Python formatting")
        
        # Format TypeScript/JavaScript files with prettier (if available)
        ts_files = list(project_root.rglob("*.ts")) + list(project_root.rglob("*.js"))
        if ts_files:
            try:
                subprocess.run(
                    ["prettier", "--write"] + [str(f) for f in ts_files],
                    check=True,
                    timeout=120
                )
                print(f"Formatted {len(ts_files)} TypeScript/JavaScript files")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Prettier formatter not available, skipping TypeScript/JavaScript formatting")
        
        print("Code formatting completed")
        sys.exit(0)
        
    except Exception as e:
        print(f"Code formatting error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    format_code()
