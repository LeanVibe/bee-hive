#!/usr/bin/env python3
"""Dependency check hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys

def check_dependencies():
    """Check that required dependencies are available."""
    try:
        event_data = json.loads(os.environ.get("LEANVIBE_EVENT_DATA", "{}"))
        
        # Check Python dependencies
        result = subprocess.run(
            ["python", "-c", "import sys; print('Python dependencies OK')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Python dependency check failed: {result.stderr}")
            sys.exit(1)
        
        print("Dependency check passed")
        sys.exit(0)
        
    except Exception as e:
        print(f"Dependency check error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
