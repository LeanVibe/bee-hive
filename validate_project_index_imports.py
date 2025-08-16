#!/usr/bin/env python3
"""
Project Index Import Validation Script

This script validates all Project Index dependencies and imports to ensure the system
can function properly. It tests both core dependencies and module imports.
"""

import sys
import traceback
from typing import List, Dict, Any, Tuple

print("=== Project Index Import Validation ===")
print()

# Test core dependencies first
dependencies_to_test = [
    "apscheduler",
    "tree_sitter", 
    "networkx",
    "git",  # gitpython
    "watchdog",
    # Optional language parsers
    "tree_sitter_python",
    "tree_sitter_javascript", 
    "tree_sitter_typescript"
]

print("📦 Testing Core Dependencies:")
dependency_results = {}

for dep in dependencies_to_test:
    try:
        __import__(dep)
        print(f"✅ {dep}")
        dependency_results[dep] = True
    except ImportError as e:
        print(f"❌ {dep}: {e}")
        dependency_results[dep] = False
    except Exception as e:
        print(f"⚠️  {dep}: {e}")
        dependency_results[dep] = False

print()

# Test Project Index imports
print("🔌 Testing Project Index Module Imports:")

imports_to_test = [
    # Core Project Index modules
    ("app.project_index.core", "ProjectIndexer"),
    ("app.project_index.analyzer", "CodeAnalyzer"),
    ("app.project_index.models", "ProjectIndexConfig"),
    ("app.project_index.websocket_events", "ProjectIndexEventPublisher"),  # Fixed!
    ("app.project_index.context_assembler", "ContextAssembler"),  # Fixed module path
    ("app.project_index.cache", "AdvancedCacheManager"),  # Fixed class name
    ("app.project_index.graph", "DependencyGraph"),
    ("app.project_index.events", "EventPublisher"),  # Fixed class name
    ("app.project_index.file_monitor", "EnhancedFileMonitor"),  # Fixed class name
    
    # API modules  
    ("app.api.project_index", "router"),
    # Note: project_index_optimization and project_index_websocket are utility modules, not routers
    
    # Model modules
    ("app.models.project_index", "ProjectIndex"),
    ("app.schemas.project_index", "ProjectIndexResponse"),  # Fixed: use actual schema class
    
    # Utility modules (just test import, no specific class)
    ("app.api.project_index_optimization", None),  # Utility module
    ("app.api.project_index_websocket", None),  # Utility module
]

import_results = {}

for module_path, class_name in imports_to_test:
    try:
        if class_name is None:
            # Just test module import, no specific class
            __import__(module_path)
            print(f"✅ {module_path} (module import)")
            import_results[f"{module_path}"] = True
        elif class_name == "router":
            # Special case for router imports
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
            import_results[f"{module_path}.{class_name}"] = True
        else:
            # Regular class imports
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
            import_results[f"{module_path}.{class_name}"] = True
    except ImportError as e:
        test_name = f"{module_path}.{class_name}" if class_name else module_path
        print(f"❌ {test_name}: ImportError - {e}")
        import_results[test_name] = False
    except AttributeError as e:
        test_name = f"{module_path}.{class_name}" if class_name else module_path
        print(f"❌ {test_name}: AttributeError - {e}")
        import_results[test_name] = False
    except Exception as e:
        test_name = f"{module_path}.{class_name}" if class_name else module_path
        print(f"⚠️  {test_name}: {type(e).__name__} - {e}")
        import_results[test_name] = False

print()

# Test infrastructure imports
print("🏗️  Testing Infrastructure Imports:")

infrastructure_imports = [
    ("app.core.database", "get_session"),  # Fixed function name
    ("app.core.redis", "get_redis_client"), 
    ("app.core.config", "settings"),
]

infrastructure_results = {}

for module_path, class_name in infrastructure_imports:
    try:
        module = __import__(module_path, fromlist=[class_name])
        getattr(module, class_name)
        print(f"✅ {module_path}.{class_name}")
        infrastructure_results[f"{module_path}.{class_name}"] = True
    except Exception as e:
        print(f"❌ {module_path}.{class_name}: {e}")
        infrastructure_results[f"{module_path}.{class_name}"] = False

print()

# Summary
print("📊 VALIDATION SUMMARY:")
print("=" * 50)

total_deps = len(dependencies_to_test)
passed_deps = sum(dependency_results.values())
print(f"Dependencies: {passed_deps}/{total_deps} ({'✅' if passed_deps == total_deps else '❌'})")

total_imports = len(imports_to_test)
passed_imports = sum(import_results.values())
print(f"Project Index: {passed_imports}/{total_imports} ({'✅' if passed_imports == total_imports else '❌'})")

total_infra = len(infrastructure_imports)
passed_infra = sum(infrastructure_results.values())
print(f"Infrastructure: {passed_infra}/{total_infra} ({'✅' if passed_infra == total_infra else '❌'})")

total_tests = total_deps + total_imports + total_infra
total_passed = passed_deps + passed_imports + passed_infra

print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")

if total_passed == total_tests:
    print("\n🎉 ALL IMPORTS SUCCESSFUL - Project Index system ready!")
    sys.exit(0)
else:
    print(f"\n⚠️  {total_tests - total_passed} imports failed - fixes needed")
    
    # Show failed imports for debugging
    print("\nFAILED IMPORTS:")
    
    for dep, success in dependency_results.items():
        if not success:
            print(f"  - Dependency: {dep}")
    
    for import_name, success in import_results.items():
        if not success:
            print(f"  - Import: {import_name}")
            
    for infra_name, success in infrastructure_results.items():
        if not success:
            print(f"  - Infrastructure: {infra_name}")
    
    sys.exit(1)