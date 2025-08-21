#!/usr/bin/env python3
"""
Dependency Analysis Script
Maps import dependencies across app/core/ to identify consolidation safety
"""

import ast
import os
from collections import defaultdict, deque
from pathlib import Path
import re

def analyze_dependencies():
    """Analyze import dependencies across core modules"""
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    py_files = [f for f in core_path.glob('*.py') if f.name not in ['__init__.py']]
    
    # Track dependencies
    internal_imports = defaultdict(set)  # Maps file -> set of internal files it imports
    external_imports = defaultdict(set)  # Maps file -> set of external modules
    import_count = defaultdict(int)      # Maps file -> how many times it's imported
    
    print("🔗 DEPENDENCY ANALYSIS")
    print("=" * 25)
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            current_module = file_path.stem
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if 'app.core' in module_name or module_name.startswith('.'):
                            # Internal import
                            imported_module = module_name.split('.')[-1]
                            internal_imports[current_module].add(imported_module)
                            import_count[imported_module] += 1
                        else:
                            # External import
                            external_imports[current_module].add(module_name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        if 'app.core' in module_name or module_name.startswith('.'):
                            # Internal import
                            if module_name.startswith('.'):
                                imported_module = module_name.lstrip('.')
                            else:
                                imported_module = module_name.split('.')[-1]
                            internal_imports[current_module].add(imported_module)
                            import_count[imported_module] += 1
                        else:
                            # External import
                            external_imports[current_module].add(module_name.split('.')[0])
        
        except Exception as e:
            print(f"  Error analyzing {file_path.name}: {e}")
    
    # Find most imported modules (core dependencies)
    print("📈 MOST IMPORTED MODULES (dependency hubs):")
    most_imported = sorted(import_count.items(), key=lambda x: x[1], reverse=True)[:15]
    for module, count in most_imported:
        print(f"  {module:30} : imported {count:2d} times")
    
    # Find modules with most dependencies
    print("\n🕸️  MODULES WITH MOST DEPENDENCIES:")
    most_dependencies = sorted(internal_imports.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    for module, deps in most_dependencies:
        if len(deps) > 0:
            print(f"  {module:30} : imports {len(deps):2d} internal modules")
    
    # Detect circular dependencies
    print("\n🔄 CIRCULAR DEPENDENCY DETECTION:")
    circular_deps = find_circular_dependencies(internal_imports)
    if circular_deps:
        for cycle in circular_deps[:10]:  # Show first 10 cycles
            print(f"  {' -> '.join(cycle)} -> {cycle[0]}")
    else:
        print("  No circular dependencies detected!")
    
    # Find orphaned modules (no internal imports/exports)
    print("\n🏝️  POTENTIAL ORPHANED MODULES:")
    all_modules = set(f.stem for f in py_files)
    imported_modules = set(import_count.keys())
    importing_modules = set(internal_imports.keys())
    
    orphaned = all_modules - imported_modules - importing_modules
    for module in sorted(orphaned)[:15]:
        print(f"  {module}")
    
    # External dependencies analysis
    print("\n🌐 EXTERNAL DEPENDENCIES (consolidation complexity):")
    external_deps = defaultdict(int)
    for module, deps in external_imports.items():
        for dep in deps:
            external_deps[dep] += 1
    
    for dep, count in sorted(external_deps.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {dep:20} : used by {count:2d} modules")
    
    return {
        'internal_imports': internal_imports,
        'import_count': import_count,
        'external_imports': external_imports,
        'circular_deps': circular_deps
    }

def find_circular_dependencies(imports_graph):
    """Find circular dependencies using DFS"""
    circular_deps = []
    visited = set()
    rec_stack = set()
    
    def dfs(node, path):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            circular_deps.append(cycle)
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in imports_graph.get(node, []):
            dfs(neighbor, path.copy())
        
        rec_stack.remove(node)
    
    for module in imports_graph:
        if module not in visited:
            dfs(module, [])
    
    return circular_deps

if __name__ == "__main__":
    analyze_dependencies()