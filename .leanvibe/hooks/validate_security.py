#!/usr/bin/env python3
"""Security validation hook for LeanVibe Agent Hive."""

import json
import os
import sys


def validate_security() -> None:
    """Validate that agent task doesn't access sensitive files."""
    try:
        event_data = json.loads(os.environ.get("LEANVIBE_EVENT_DATA", "{}"))

        # Check for sensitive file access
        sensitive_patterns = [
            ".env", "secrets", "private_key", "password",
            ".ssh", "credentials", "token", "api_key"
        ]

        task_data = event_data.get("parameters", {})
        file_paths = []

        # Extract file paths from various task parameters
        for _key, value in task_data.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                file_paths.append(value)

        # Check for sensitive patterns
        for file_path in file_paths:
            for pattern in sensitive_patterns:
                if pattern.lower() in file_path.lower():
                    print(f"SECURITY VIOLATION: Attempting to access sensitive file: {file_path}")
                    sys.exit(1)

        print("Security validation passed")
        sys.exit(0)

    except Exception as e:
        print(f"Security validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_security()
