#!/bin/bash
# LeanVibe Agent Hive 2.0 - Quality Gate Hook
# Automatically triggered after file edits to maintain system quality

set -euo pipefail

# Read hook input from stdin
input_data=$(cat)

# Extract tool information
tool_name=$(echo "$input_data" | jq -r '.tool_name // "unknown"')
file_path=$(echo "$input_data" | jq -r '.tool_input.file_path // ""')

# Only process relevant file types
if [[ ! "$file_path" =~ \.(py|ts|js|md|json|yaml|yml)$ ]]; then
    exit 0
fi

echo "🔍 LeanVibe Quality Gate: Processing $file_path"

# Quality gate checks
quality_issues=0

# Check 1: Python files
if [[ "$file_path" =~ \.py$ ]]; then
    # Basic syntax check
    if ! python3 -m py_compile "$file_path" 2>/dev/null; then
        echo "❌ Python syntax error in $file_path"
        quality_issues=$((quality_issues + 1))
    fi
    
    # Check for common issues
    if grep -q "print(" "$file_path" && [[ "$file_path" != *test* ]] && [[ "$file_path" != *demo* ]]; then
        echo "⚠️  Debug print() found in production code: $file_path"
    fi
fi

# Check 2: TypeScript/JavaScript files  
if [[ "$file_path" =~ \.(ts|js)$ ]]; then
    # Check for console.log in production files
    if grep -q "console.log" "$file_path" && [[ "$file_path" != *test* ]] && [[ "$file_path" != *demo* ]]; then
        echo "⚠️  Console.log found in production code: $file_path"
    fi
fi

# Check 3: Configuration files
if [[ "$file_path" =~ \.(json|yaml|yml)$ ]]; then
    # Basic JSON syntax check
    if [[ "$file_path" =~ \.json$ ]]; then
        if ! jq empty "$file_path" 2>/dev/null; then
            echo "❌ Invalid JSON syntax in $file_path"
            quality_issues=$((quality_issues + 1))
        fi
    fi
fi

# Check 4: Security concerns
if grep -qi "api.key\|secret\|password" "$file_path" && [[ "$file_path" != *.md ]]; then
    if grep -qi "sk-\|ghp_\|gho_" "$file_path"; then
        echo "🚨 SECURITY: Potential API key found in $file_path"
        quality_issues=$((quality_issues + 1))
    fi
fi

# Check 5: LeanVibe-specific quality
if [[ "$file_path" =~ bee-hive ]]; then
    # Check for proper error handling in Python
    if [[ "$file_path" =~ \.py$ ]] && [[ "$file_path" =~ (app/|scripts/) ]]; then
        if ! grep -q "try:\|except\|raise" "$file_path" && [[ $(wc -l < "$file_path") -gt 20 ]]; then
            echo "⚠️  Large Python file without error handling: $file_path"
        fi
    fi
    
    # Check for proper logging
    if [[ "$file_path" =~ (app/|scripts/) ]] && [[ $(wc -l < "$file_path") -gt 50 ]]; then
        if ! grep -qi "log\|print" "$file_path"; then
            echo "⚠️  Large file without logging: $file_path"
        fi
    fi
fi

# Trigger system health check for critical files
if [[ "$file_path" =~ (main.py|app.py|__init__.py|config) ]]; then
    echo "🔄 Critical file modified, triggering health check..."
    if command -v python3 >/dev/null 2>&1; then
        if [[ -f "/Users/bogdan/work/leanvibe-dev/bee-hive/scripts/intelligent_recovery.py" ]]; then
            python3 /Users/bogdan/work/leanvibe-dev/bee-hive/scripts/intelligent_recovery.py status >/dev/null 2>&1 || echo "⚠️  System health check recommended"
        fi
    fi
fi

# Summary
if [[ $quality_issues -eq 0 ]]; then
    echo "✅ Quality gate passed for $file_path"
else
    echo "❌ Quality gate found $quality_issues issues in $file_path"
fi

exit 0