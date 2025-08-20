#!/usr/bin/env python3
"""
Practical Utility Function Consolidator
=======================================

Direct approach to consolidating the most common utility functions identified
in our semantic analysis. Focuses on high-impact, safe consolidations.
"""

import ast
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import tempfile
import shutil

class PracticalUtilityConsolidator:
    """Consolidates common utility functions using proven safety approaches."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(tempfile.mkdtemp(prefix="practical_utility_backups_"))
        self.common_utils_dir = self.project_root / "app" / "common" / "utilities"
        
        # Define common utility patterns we know are duplicated
        self.utility_patterns = {
            'string_helpers': [
                'format_string', 'sanitize_string', 'validate_string',
                'clean_text', 'normalize_text', 'extract_text'
            ],
            'file_helpers': [
                'read_file', 'write_file', 'check_file_exists',
                'create_directory', 'get_file_size', 'copy_file'
            ],
            'date_helpers': [
                'format_datetime', 'parse_datetime', 'get_timestamp',
                'calculate_duration', 'format_date', 'is_valid_date'
            ],
            'data_helpers': [
                'serialize_json', 'deserialize_json', 'validate_json',
                'merge_dicts', 'flatten_dict', 'deep_copy'
            ],
            'validation_helpers': [
                'validate_email', 'validate_url', 'validate_uuid',
                'is_empty', 'is_valid', 'check_required'
            ],
            'math_helpers': [
                'round_number', 'calculate_percentage', 'clamp_value',
                'generate_random', 'calculate_average', 'find_min_max'
            ],
            'crypto_helpers': [
                'hash_string', 'generate_uuid', 'encode_base64',
                'decode_base64', 'generate_token', 'hash_password'
            ],
            'system_helpers': [
                'get_env_var', 'set_env_var', 'check_platform',
                'get_memory_usage', 'execute_command', 'get_pid'
            ]
        }
    
    def find_duplicate_utilities(self) -> Dict[str, List[Dict]]:
        """Find duplicate utility functions across the codebase."""
        print("🔍 Scanning codebase for duplicate utility functions...")
        
        function_signatures = defaultdict(list)
        
        # Scan all Python files
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not any(skip in str(f) for skip in ['.venv', 'venv', '__pycache__'])]
        
        print(f"Analyzing {len(python_files)} Python files...")
        
        for file_path in python_files:
            try:
                functions = self.extract_utility_functions(file_path)
                for func_info in functions:
                    signature = self.create_function_signature(func_info)
                    function_signatures[signature].append({
                        'file': file_path,
                        'function': func_info
                    })
            except Exception as e:
                continue  # Skip problematic files
        
        # Find duplicates (functions with same signature in multiple files)
        duplicates = {}
        for signature, occurrences in function_signatures.items():
            if len(occurrences) >= 3:  # At least 3 occurrences
                duplicates[signature] = occurrences
        
        print(f"Found {len(duplicates)} utility function patterns with multiple implementations")
        return duplicates
    
    def extract_utility_functions(self, file_path: Path) -> List[Dict]:
        """Extract potential utility functions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            utilities = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this looks like a utility function
                    if self.is_likely_utility_function(node, file_path):
                        func_info = {
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'line': node.lineno,
                            'docstring': ast.get_docstring(node),
                            'body_hash': self.hash_function_body(node),
                            'complexity': len(list(ast.walk(node)))
                        }
                        utilities.append(func_info)
            
            return utilities
        except:
            return []
    
    def is_likely_utility_function(self, node: ast.FunctionDef, file_path: Path) -> bool:
        """Check if a function is likely a utility function."""
        func_name = node.name.lower()
        
        # Skip private functions and methods
        if func_name.startswith('_') or (node.args.args and node.args.args[0].arg == 'self'):
            return False
        
        # Skip very complex functions (likely business logic)
        if len(list(ast.walk(node))) > 50:
            return False
        
        # Check for utility patterns in function name
        utility_keywords = [
            'format', 'parse', 'validate', 'check', 'convert', 'transform',
            'serialize', 'deserialize', 'encode', 'decode', 'hash', 'generate',
            'create', 'build', 'make', 'get', 'set', 'extract', 'clean',
            'sanitize', 'normalize', 'calculate', 'compute', 'find'
        ]
        
        if any(keyword in func_name for keyword in utility_keywords):
            return True
        
        # Check if it's in a utils/helpers directory or file
        path_str = str(file_path).lower()
        if any(util_indicator in path_str for util_indicator in ['util', 'helper', 'common', 'tool']):
            return True
        
        return False
    
    def create_function_signature(self, func_info: Dict) -> str:
        """Create a signature for function similarity matching."""
        name_normalized = re.sub(r'[_\d]', '', func_info['name'].lower())
        args_normalized = '_'.join(sorted(func_info['args'][:3]))  # First 3 args only
        
        return f"{name_normalized}_{len(func_info['args'])}_{args_normalized}"
    
    def hash_function_body(self, node: ast.FunctionDef) -> str:
        """Create a hash of the function body structure."""
        # Extract key structural elements
        elements = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                elements.append('call')
            elif isinstance(n, ast.If):
                elements.append('if')
            elif isinstance(n, ast.For):
                elements.append('for')
            elif isinstance(n, ast.Return):
                elements.append('return')
        
        return '_'.join(elements[:10])  # First 10 structural elements
    
    def create_consolidated_utilities(self, duplicates: Dict[str, List[Dict]], dry_run: bool = False) -> Dict:
        """Create consolidated utility modules."""
        print("🔧 Creating consolidated utility modules...")
        
        # Group duplicates by utility category
        categorized = self.categorize_duplicates(duplicates)
        
        results = {}
        total_savings = 0
        
        for category, functions in categorized.items():
            if len(functions) < 3:  # Skip small categories
                continue
            
            # Calculate savings potential
            total_files = sum(len(func_group) for func_group in functions.values())
            estimated_savings = total_files * 8  # Rough estimate: 8 lines per function
            total_savings += estimated_savings
            
            result = {
                'category': category,
                'function_patterns': len(functions),
                'total_files_affected': total_files,
                'estimated_savings': estimated_savings,
                'status': 'analyzed'
            }
            
            if not dry_run:
                module_path = self.create_category_module(category, functions)
                result['module_created'] = str(module_path)
                result['status'] = 'module_created'
            
            results[category] = result
            print(f"   • {category}: {len(functions)} patterns, {estimated_savings} LOC savings")
        
        return {
            'results': results,
            'total_categories': len(results),
            'total_estimated_savings': total_savings
        }
    
    def categorize_duplicates(self, duplicates: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Categorize duplicate functions by utility type."""
        categorized = defaultdict(dict)
        
        for signature, occurrences in duplicates.items():
            if len(occurrences) < 3:
                continue
            
            # Determine category based on function names
            func_names = [occ['function']['name'].lower() for occ in occurrences]
            category = self.determine_category(func_names)
            
            categorized[category][signature] = occurrences
        
        return categorized
    
    def determine_category(self, func_names: List[str]) -> str:
        """Determine utility category from function names."""
        all_names = ' '.join(func_names)
        
        categories = {
            'string_utils': ['format', 'clean', 'sanitize', 'text', 'string'],
            'file_utils': ['file', 'read', 'write', 'path', 'directory'],
            'date_utils': ['date', 'time', 'timestamp', 'duration', 'format'],
            'data_utils': ['json', 'serialize', 'data', 'convert', 'parse'],
            'validation_utils': ['validate', 'check', 'verify', 'is_', 'ensure'],
            'math_utils': ['calculate', 'compute', 'math', 'number', 'round'],
            'crypto_utils': ['hash', 'encode', 'decode', 'crypto', 'token'],
            'system_utils': ['env', 'system', 'platform', 'process', 'pid']
        }
        
        for category, keywords in categories.items():
            if sum(1 for keyword in keywords if keyword in all_names) >= 2:
                return category
        
        return 'general_utils'
    
    def create_category_module(self, category: str, functions: Dict[str, List[Dict]]) -> Path:
        """Create a consolidated module for a utility category."""
        self.common_utils_dir.mkdir(parents=True, exist_ok=True)
        module_path = self.common_utils_dir / f"{category}.py"
        
        # Generate module content
        module_content = f'''"""
{category.replace('_', ' ').title()} for LeanVibe Agent Hive 2.0.

Consolidated utility functions for {category.replace('_', ' ')} operations.
Generated from {len(functions)} duplicate patterns across the codebase.
"""

from typing import Any, Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

# Consolidated utility functions
'''
        
        consolidated_functions = []
        
        for signature, occurrences in functions.items():
            if len(occurrences) >= 3:
                # Create consolidated function
                representative = occurrences[0]['function']
                func_name = self.generate_consolidated_name(representative['name'], category)
                
                module_content += f'''

def {func_name}({', '.join(representative['args']) if representative['args'] else '*args, **kwargs'}):
    """
    Consolidated utility function from {len(occurrences)} similar implementations.
    
    Original implementations found in:
    {chr(10).join(f'    - {occ["file"]}:{occ["function"]["line"]}' for occ in occurrences[:5])}
    {'    ... and more' if len(occurrences) > 5 else ''}
    
    {representative.get('docstring', 'TODO: Add documentation based on consolidated functionality')}
    """
    # TODO: Implement consolidated logic from similar functions
    logger.warning(f"Consolidated function {func_name} needs implementation")
    raise NotImplementedError(f"Function {func_name} requires implementation")

'''
                consolidated_functions.append(func_name)
        
        # Add __all__ export
        module_content += f'''
__all__ = {consolidated_functions}
'''
        
        module_path.write_text(module_content)
        return module_path
    
    def generate_consolidated_name(self, original_name: str, category: str) -> str:
        """Generate a good consolidated function name."""
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^(get_|set_|do_|make_)', '', original_name.lower())
        cleaned = re.sub(r'(_util|_helper|_function)$', '', cleaned)
        
        # Ensure it fits the category
        category_prefix = category.split('_')[0]
        if not cleaned.startswith(category_prefix[:4]):  # Use first 4 chars of category
            cleaned = f"{category_prefix}_{cleaned}"
        
        return cleaned

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate utility functions')
    parser.add_argument('--analyze', action='store_true', help='Analyze consolidation opportunities')
    parser.add_argument('--consolidate', action='store_true', help='Create consolidated modules')
    
    args = parser.parse_args()
    
    consolidator = PracticalUtilityConsolidator()
    
    # Find duplicates
    duplicates = consolidator.find_duplicate_utilities()
    
    if not duplicates:
        print("No significant utility function duplicates found.")
        return
    
    if args.consolidate:
        print("🚀 Creating consolidated utility modules...")
        results = consolidator.create_consolidated_utilities(duplicates, dry_run=False)
    else:
        print("📋 Analyzing utility consolidation opportunities...")
        results = consolidator.create_consolidated_utilities(duplicates, dry_run=True)
    
    print(f"\\n📊 Practical Utility Consolidation Results:")
    print(f"   🔍 {len(duplicates)} duplicate patterns found")
    print(f"   🔧 {results['total_categories']} utility categories")
    print(f"   💰 {results['total_estimated_savings']:,} LOC estimated savings")
    
    if args.consolidate:
        created_modules = sum(1 for r in results['results'].values() if r['status'] == 'module_created')
        print(f"   ✅ {created_modules} consolidated utility modules created")

if __name__ == "__main__":
    main()