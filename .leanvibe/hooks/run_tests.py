#!/usr/bin/env python3
"""Test execution hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run relevant tests for changed code."""
    try:
        project_root = Path(os.environ.get("LEANVIBE_PROJECT_ROOT", "."))
        
        # Try different test runners
        test_runners = [
            (["python", "-m", "pytest", "-v"], "pytest"),
            (["python", "-m", "unittest", "discover"], "unittest"),
            (["npm", "test"], "npm")
        ]
        
        test_passed = False
        
        for command, runner_name in test_runners:
            try:
                result = subprocess.run(
                    command,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"Tests passed using {runner_name}")
                    print(result.stdout)
                    test_passed = True
                    break
                else:
                    print(f"Tests failed using {runner_name}: {result.stderr}")
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if test_passed:
            sys.exit(0)
        else:
            print("No test runner succeeded")
            sys.exit(1)
        
    except Exception as e:
        print(f"Test execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
