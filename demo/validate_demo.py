#!/usr/bin/env python3
"""
Demo Validation Script
Validates that all demo components are working correctly
"""

import os
import sys
import json
from pathlib import Path

def validate_files():
    """Validate all required demo files exist."""
    print("🔍 Validating demo files...")
    
    demo_dir = Path(__file__).parent
    required_files = [
        "index.html",
        "assets/styles.css", 
        "assets/demo.js",
        "assets/syntax-highlighter.js",
        "api/demo_endpoint.py",
        "api/__init__.py",
        "fallback/autonomous_engine.py",
        "fallback/__init__.py",
        "demo_server.py",
        "manifest.json",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = demo_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def validate_html():
    """Validate HTML structure."""
    print("🔍 Validating HTML structure...")
    
    html_file = Path(__file__).parent / "index.html"
    if not html_file.exists():
        print("❌ index.html not found")
        return False
    
    content = html_file.read_text()
    
    # Check for required sections
    required_sections = [
        'class="hero"',
        'class="demo-section"',
        'class="task-selection"',
        'development-progress',
        'results-section',
        'id="startDemoBtn"',
        'id="liveCodeDisplay"'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print("❌ Missing HTML sections:")
        for section in missing_sections:
            print(f"   - {section}")
        return False
    
    print("✅ HTML structure valid")
    return True

def validate_css():
    """Validate CSS file."""
    print("🔍 Validating CSS...")
    
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if not css_file.exists():
        print("❌ styles.css not found")
        return False
    
    content = css_file.read_text()
    
    # Check for key CSS classes
    required_classes = [
        ".hero",
        ".demo-interface",
        ".task-card",
        ".development-progress",
        ".phase",
        ".code-display",
        ".results-section"
    ]
    
    missing_classes = []
    for css_class in required_classes:
        if css_class not in content:
            missing_classes.append(css_class)
    
    if missing_classes:
        print("❌ Missing CSS classes:")
        for css_class in missing_classes:
            print(f"   - {css_class}")
        return False
    
    print("✅ CSS structure valid")
    return True

def validate_javascript():
    """Validate JavaScript file."""
    print("🔍 Validating JavaScript...")
    
    js_file = Path(__file__).parent / "assets" / "demo.js"
    if not js_file.exists():
        print("❌ demo.js not found")
        return False
    
    content = js_file.read_text()
    
    # Check for key JavaScript components
    required_components = [
        "class AutonomousDevelopmentDemo",
        "startDemo",
        "connectEventSource",
        "handleProgressUpdate",
        "generateMockCode"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print("❌ Missing JavaScript components:")
        for component in missing_components:
            print(f"   - {component}")
        return False
    
    print("✅ JavaScript structure valid")
    return True

def validate_python_syntax():
    """Validate Python file syntax."""
    print("🔍 Validating Python syntax...")
    
    python_files = [
        "demo_server.py",
        "api/demo_endpoint.py", 
        "fallback/autonomous_engine.py"
    ]
    
    demo_dir = Path(__file__).parent
    
    for file_path in python_files:
        full_path = demo_dir / file_path
        if not full_path.exists():
            print(f"❌ {file_path} not found")
            return False
        
        try:
            import ast
            content = full_path.read_text()
            ast.parse(content)
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return False
    
    print("✅ Python syntax valid")
    return True

def validate_manifest():
    """Validate PWA manifest."""
    print("🔍 Validating PWA manifest...")
    
    manifest_file = Path(__file__).parent / "manifest.json"
    if not manifest_file.exists():
        print("❌ manifest.json not found")
        return False
    
    try:
        content = manifest_file.read_text()
        manifest = json.loads(content)
        
        required_fields = ["name", "short_name", "start_url", "display", "theme_color"]
        missing_fields = []
        
        for field in required_fields:
            if field not in manifest:
                missing_fields.append(field)
        
        if missing_fields:
            print("❌ Missing manifest fields:")
            for field in missing_fields:
                print(f"   - {field}")
            return False
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in manifest: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating manifest: {e}")
        return False
    
    print("✅ PWA manifest valid")
    return True

def validate_dependencies():
    """Check if required Python dependencies are available."""
    print("🔍 Checking Python dependencies...")
    
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),  
        ("anthropic", "Anthropic API client (optional)")
    ]
    
    missing_packages = []
    optional_missing = []
    
    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == "anthropic":
                optional_missing.append((package, description))
            else:
                missing_packages.append((package, description))
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package, description in missing_packages:
            print(f"   - {package}: {description}")
        print("\n   Install with: pip install fastapi uvicorn")
        return False
    
    if optional_missing:
        print("⚠️  Optional packages missing:")
        for package, description in optional_missing:
            print(f"   - {package}: {description}")
        print("   Demo will work with fallback templates")
    
    print("✅ Dependencies satisfied")
    return True

def main():
    """Run all validation checks."""
    print("🚀 LeanVibe Agent Hive 2.0 - Demo Validation")
    print("=" * 60)
    
    validators = [
        validate_files,
        validate_html,
        validate_css,
        validate_javascript,
        validate_python_syntax,
        validate_manifest,
        validate_dependencies
    ]
    
    all_passed = True
    
    for validator in validators:
        try:
            if not validator():
                all_passed = False
        except Exception as e:
            print(f"❌ Validation error: {e}")
            all_passed = False
        print()
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\n🎉 Demo is ready to run!")
        print("   Start with: python demo_server.py")
        print("   Or: ./demo_server.py")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\n🔧 Please fix the issues above before running the demo")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)