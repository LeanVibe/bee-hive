#!/usr/bin/env python3
"""
Duplication Detection Script
Identifies duplicate functions, classes, and patterns for consolidation planning
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
import re

def analyze_duplication():
    """Analyze function and class duplication across core modules"""
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    py_files = [f for f in core_path.glob('*.py') if f.name not in ['__init__.py']]
    
    # Track functions and classes
    functions = defaultdict(list)  # function_name -> [(file, line_no)]
    classes = defaultdict(list)    # class_name -> [(file, line_no)]
    constants = defaultdict(list)  # constant_name -> [(file, line_no)]
    
    print("🔍 DUPLICATION ANALYSIS")
    print("=" * 25)
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            current_file = file_path.name
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions[node.name].append((current_file, node.lineno))
                elif isinstance(node, ast.ClassDef):
                    classes[node.name].append((current_file, node.lineno))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            constants[target.id].append((current_file, node.lineno))
        
        except Exception as e:
            print(f"  Error analyzing {file_path.name}: {e}")
    
    # Find duplicate functions
    print("🔄 DUPLICATE FUNCTIONS (same name across files):")
    duplicate_funcs = {name: locations for name, locations in functions.items() if len(locations) > 1}
    for func_name, locations in sorted(duplicate_funcs.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
        print(f"  {func_name:25} : {len(locations)} files")
        for file, line in locations[:3]:
            print(f"    - {file}:{line}")
        if len(locations) > 3:
            print(f"    ... and {len(locations) - 3} more")
    
    # Find duplicate classes
    print("\n🏗️  DUPLICATE CLASSES (same name across files):")
    duplicate_classes = {name: locations for name, locations in classes.items() if len(locations) > 1}
    for class_name, locations in sorted(duplicate_classes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {class_name:25} : {len(locations)} files")
        for file, line in locations:
            print(f"    - {file}:{line}")
    
    # Find duplicate constants
    print("\n📊 DUPLICATE CONSTANTS:")
    duplicate_constants = {name: locations for name, locations in constants.items() if len(locations) > 1}
    for const_name, locations in sorted(duplicate_constants.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {const_name:25} : {len(locations)} files")
        for file, line in locations[:3]:
            print(f"    - {file}:{line}")
        if len(locations) > 3:
            print(f"    ... and {len(locations) - 3} more")
    
    # Analyze similar function patterns
    print("\n🎯 SIMILAR FUNCTION PATTERNS:")
    function_patterns = defaultdict(list)
    for func_name in functions.keys():
        # Group by similar patterns (after removing numbers/versions)
        pattern = re.sub(r'[_\d]+$', '', func_name.lower())
        pattern = re.sub(r'(v\d+|version\d+)', '', pattern)
        function_patterns[pattern].append(func_name)
    
    for pattern, func_names in sorted(function_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        if len(func_names) > 2:
            print(f"  Pattern '{pattern}': {len(func_names)} functions")
            for func in func_names[:5]:
                print(f"    - {func}")
            if len(func_names) > 5:
                print(f"    ... and {len(func_names) - 5} more")
    
    return {
        'duplicate_functions': duplicate_funcs,
        'duplicate_classes': duplicate_classes,
        'duplicate_constants': duplicate_constants,
        'total_functions': len(functions),
        'total_classes': len(classes)
    }

def analyze_configuration_duplication():
    """Analyze configuration and setup code duplication"""
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    py_files = [f for f in core_path.glob('*.py') if f.name not in ['__init__.py']]
    
    print("\n⚙️  CONFIGURATION DUPLICATION:")
    
    # Common configuration patterns
    config_patterns = [
        'get_redis', 'get_database', 'get_session', 'AsyncSession',
        'structlog.get_logger', 'settings.', 'logger = ',
        'REDIS_URL', 'DATABASE_URL', 'API_KEY'
    ]
    
    pattern_counts = defaultdict(list)
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in config_patterns:
                if pattern in content:
                    pattern_counts[pattern].append(file_path.name)
        
        except Exception as e:
            continue
    
    for pattern, files in sorted(pattern_counts.items(), key=lambda x: len(x[1]), reverse=True):
        if len(files) > 5:  # Show patterns used in 5+ files
            print(f"  {pattern:20} : {len(files):3d} files")

if __name__ == "__main__":
    duplication_results = analyze_duplication()
    analyze_configuration_duplication()
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total functions analyzed: {duplication_results['total_functions']}")
    print(f"  Total classes analyzed: {duplication_results['total_classes']}")
    print(f"  Functions with duplicates: {len(duplication_results['duplicate_functions'])}")
    print(f"  Classes with duplicates: {len(duplication_results['duplicate_classes'])}")