#!/usr/bin/env python3
"""
Validation Script for Autonomous Development Demo

This script validates that the autonomous development demo is correctly set up
and can run successfully. It performs dry-run testing without requiring API keys.
"""

import sys
import os
import tempfile
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def validate_imports():
    """Validate that all required imports work."""
    print("🔍 Validating imports...")
    
    try:
        # Test core engine import
        from app.core.autonomous_development_engine import (
            AutonomousDevelopmentEngine,
            DevelopmentTask,
            TaskComplexity,
            DevelopmentPhase,
            DevelopmentArtifact,
            DevelopmentResult
        )
        print("✅ Core engine imports successful")
        
        # Test that classes can be instantiated (without API key)
        task = DevelopmentTask(
            id="test_task",
            description="Test task",
            requirements=["Test requirement"],
            complexity=TaskComplexity.SIMPLE
        )
        print("✅ Task creation successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def validate_demo_scripts():
    """Validate that demo scripts are properly structured."""
    print("\n🔍 Validating demo scripts...")
    
    demo_scripts = [
        "scripts/demos/autonomous_development_demo.py",
        "scripts/demos/standalone_autonomous_demo.py"
    ]
    
    for script_path in demo_scripts:
        full_path = project_root / script_path
        
        if not full_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        # Check if script is executable
        if not os.access(full_path, os.X_OK):
            print(f"⚠️  Script not executable: {script_path}")
        
        # Validate Python syntax
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {script_path} - syntax valid")
        except SyntaxError as e:
            print(f"❌ {script_path} - syntax error: {e}")
            return False
    
    return True


def validate_file_structure():
    """Validate that all necessary files exist."""
    print("\n🔍 Validating file structure...")
    
    required_files = [
        "app/core/autonomous_development_engine.py",
        "docs/AUTONOMOUS_DEVELOPMENT_DEMO.md",
        "scripts/demos/autonomous_development_demo.py",
        "scripts/demos/standalone_autonomous_demo.py"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        print(f"✅ {file_path} exists")
    
    return True


def simulate_development_workflow():
    """Simulate the development workflow without API calls."""
    print("\n🔍 Simulating development workflow...")
    
    try:
        # Create a temporary workspace
        workspace = Path(tempfile.mkdtemp(prefix="demo_validation_"))
        print(f"✅ Workspace created: {workspace}")
        
        # Simulate file generation
        test_files = {
            "solution.py": '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
            "test_solution.py": '''import unittest
from solution import fibonacci

class TestFibonacci(unittest.TestCase):
    def test_fibonacci(self):
        self.assertEqual(fibonacci(5), 5)

if __name__ == '__main__':
    unittest.main()
''',
            "README.md": '''# Fibonacci Calculator

Simple fibonacci number calculator.

## Usage

```python
from solution import fibonacci
result = fibonacci(5)
```
'''
        }
        
        # Write test files
        for filename, content in test_files.items():
            file_path = workspace / filename
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Generated {filename}")
        
        # Validate syntax
        for filename in ["solution.py", "test_solution.py"]:
            file_path = workspace / filename
            with open(file_path, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"✅ {filename} syntax valid")
        
        # Clean up
        import shutil
        shutil.rmtree(workspace)
        print("✅ Workspace cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow simulation error: {e}")
        return False


def check_dependencies():
    """Check for required dependencies."""
    print("\n🔍 Checking dependencies...")
    
    # Check for anthropic package (optional for validation)
    try:
        import anthropic
        print("✅ anthropic package available")
    except ImportError:
        print("⚠️  anthropic package not installed (required for actual demo)")
        print("   Install with: pip install anthropic")
    
    # Check Python version
    if sys.version_info >= (3, 8):
        print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    else:
        print(f"❌ Python version {sys.version_info.major}.{sys.version_info.minor} is too old (need 3.8+)")
        return False
    
    return True


def main():
    """Run all validation checks."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          🔍 Autonomous Development Demo Validation                           ║
║                                                                              ║
║  This script validates that the autonomous development demo is properly      ║
║  set up and ready to run.                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    all_checks_passed = True
    
    # Run validation checks
    checks = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Demo Scripts", validate_demo_scripts),
        ("Dependencies", check_dependencies),
        ("Development Workflow", simulate_development_workflow)
    ]
    
    for check_name, check_function in checks:
        print(f"\n{'='*60}")
        print(f"Running {check_name} validation...")
        print('='*60)
        
        if not check_function():
            all_checks_passed = False
            print(f"❌ {check_name} validation failed")
        else:
            print(f"✅ {check_name} validation passed")
    
    # Summary
    print(f"\n{'='*80}")
    if all_checks_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nThe autonomous development demo is ready to run:")
        print("1. Set your API key: export ANTHROPIC_API_KEY='your_key'")
        print("2. Run: python scripts/demos/standalone_autonomous_demo.py")
        print("\nOr run the full demo:")
        print("   python scripts/demos/autonomous_development_demo.py")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please fix the issues above before running the demo.")
    
    print('='*80)
    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)