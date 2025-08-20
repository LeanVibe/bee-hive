#!/usr/bin/env python3
"""
Utility Function Consolidator
=============================

Systematic consolidation of the 178 utility function patterns identified
by semantic analysis. Creates unified utility libraries and updates references.
"""

import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import shutil
import tempfile
from datetime import datetime

class UtilityFunctionConsolidator:
    """Consolidates duplicate utility functions into common libraries."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(tempfile.mkdtemp(prefix="utility_backups_"))
        self.common_utils_dir = self.project_root / "app" / "common" / "utilities"
        self.results = []
        
        # Load semantic analysis results
        self.semantic_report = self.load_semantic_report()
        
    def load_semantic_report(self) -> Dict:
        """Load the semantic analysis report."""
        report_path = self.project_root / "advanced_semantic_analysis_report.json"
        if not report_path.exists():
            raise FileNotFoundError("Semantic analysis report not found. Run advanced_semantic_analyzer.py first.")
        
        with open(report_path, 'r') as f:
            return json.load(f)
    
    def extract_utility_patterns(self) -> Dict[str, List]:
        """Extract utility function patterns from semantic analysis."""
        utility_patterns = []
        
        # Get utility opportunities from detailed report
        for group_key, patterns in self.semantic_report.get('detailed_opportunities', {}).items():
            if 'utility' in group_key.lower():
                utility_patterns.extend(patterns)
        
        print(f"📋 Found {len(utility_patterns)} utility consolidation patterns")
        return self.group_utility_patterns(utility_patterns)
    
    def group_utility_patterns(self, patterns: List) -> Dict[str, List]:
        """Group utility patterns by specific functionality."""
        grouped = defaultdict(list)
        
        for pattern in patterns:
            # Analyze pattern to determine utility category
            category = self.categorize_utility_pattern(pattern)
            grouped[category].append(pattern)
        
        return dict(grouped)
    
    def categorize_utility_pattern(self, pattern: Dict) -> str:
        """Categorize utility pattern into specific functionality."""
        semantic_sig = pattern.get('semantic_signature', '').lower()
        
        categories = {
            'string_utils': ['string', 'text', 'format', 'parse'],
            'file_utils': ['file', 'path', 'directory', 'io'],
            'data_utils': ['data', 'serialize', 'convert', 'transform'],
            'time_utils': ['time', 'date', 'timestamp', 'duration'],
            'math_utils': ['math', 'calculate', 'compute', 'number'],
            'collection_utils': ['list', 'dict', 'set', 'array', 'collection'],
            'validation_utils': ['validate', 'check', 'verify', 'ensure'],
            'crypto_utils': ['crypto', 'hash', 'encrypt', 'decode'],
            'network_utils': ['network', 'url', 'request', 'response'],
            'system_utils': ['system', 'process', 'env', 'platform']
        }
        
        for category, keywords in categories.items():
            if any(keyword in semantic_sig for keyword in keywords):
                return category
        
        return 'general_utils'
    
    def create_consolidated_utilities(self, utility_groups: Dict[str, List], dry_run: bool = False) -> Dict:
        """Create consolidated utility libraries for each category."""
        consolidation_results = {}
        
        for category, patterns in utility_groups.items():
            if len(patterns) < 2:  # Need at least 2 patterns to consolidate
                continue
            
            print(f"\\n🔧 Consolidating {category}: {len(patterns)} patterns")
            
            # Analyze functions in this category
            functions = self.extract_functions_from_patterns(patterns)
            
            if len(functions) >= 3:  # Minimum threshold for consolidation
                result = self.consolidate_category(category, functions, dry_run)
                consolidation_results[category] = result
        
        return consolidation_results
    
    def extract_functions_from_patterns(self, patterns: List) -> List[Dict]:
        """Extract individual functions from semantic patterns."""
        functions = []
        
        for pattern in patterns:
            files = pattern.get('files', [])
            for file_path in files[:5]:  # Limit to 5 files per pattern for analysis
                try:
                    file_functions = self.extract_functions_from_file(Path(file_path))
                    functions.extend(file_functions)
                except Exception as e:
                    print(f"Warning: Could not analyze {file_path}: {e}")
        
        return functions
    
    def extract_functions_from_file(self, file_path: Path) -> List[Dict]:
        """Extract function definitions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions and methods
                    if node.name.startswith('_') or len(node.args.args) > 0 and node.args.args[0].arg == 'self':
                        continue
                    
                    func_info = {
                        'name': node.name,
                        'file': file_path,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'source': self.extract_function_source(content, node),
                        'docstring': ast.get_docstring(node),
                        'complexity': self.estimate_complexity(node)
                    }
                    functions.append(func_info)
            
            return functions
            
        except Exception as e:
            print(f"Error extracting functions from {file_path}: {e}")
            return []
    
    def extract_function_source(self, content: str, node: ast.FunctionDef) -> str:
        """Extract the source code of a function."""
        lines = content.split('\\n')
        start_line = node.lineno - 1
        
        # Find the end of the function by looking for the next def/class at the same indentation
        end_line = len(lines)
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and (line.startswith('def ') or line.startswith('class ')) and len(line) - len(line.lstrip()) <= base_indent:
                end_line = i
                break
        
        return '\\n'.join(lines[start_line:end_line])
    
    def estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate function complexity by counting nodes."""
        return len(list(ast.walk(node)))
    
    def consolidate_category(self, category: str, functions: List[Dict], dry_run: bool = False) -> Dict:
        """Consolidate functions in a specific category."""
        # Group similar functions
        similar_groups = self.group_similar_functions(functions)
        
        consolidation_opportunities = []
        total_savings = 0
        
        for group in similar_groups:
            if len(group) >= 3:  # Minimum group size for consolidation
                opportunity = {
                    'function_count': len(group),
                    'files_affected': len(set(f['file'] for f in group)),
                    'estimated_loc_savings': sum(f['complexity'] for f in group) * 2,  # Rough estimate
                    'consolidated_function_name': self.suggest_unified_name(group),
                    'functions': group
                }
                consolidation_opportunities.append(opportunity)
                total_savings += opportunity['estimated_loc_savings']
        
        result = {
            'category': category,
            'consolidation_opportunities': consolidation_opportunities,
            'total_functions_analyzed': len(functions),
            'total_estimated_savings': total_savings,
            'status': 'analyzed' if dry_run else 'ready_for_consolidation'
        }
        
        if not dry_run and consolidation_opportunities:
            # Create consolidated utility module
            module_path = self.create_utility_module(category, consolidation_opportunities)
            result['module_created'] = str(module_path)
        
        return result
    
    def group_similar_functions(self, functions: List[Dict]) -> List[List[Dict]]:
        """Group functions by similarity."""
        groups = []
        processed = set()
        
        for i, func1 in enumerate(functions):
            if i in processed:
                continue
            
            group = [func1]
            processed.add(i)
            
            for j, func2 in enumerate(functions[i+1:], i+1):
                if j in processed:
                    continue
                
                if self.are_functions_similar(func1, func2):
                    group.append(func2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def are_functions_similar(self, func1: Dict, func2: Dict) -> bool:
        """Check if two functions are similar enough to consolidate."""
        # Name similarity
        name_similarity = self.calculate_name_similarity(func1['name'], func2['name'])
        
        # Argument similarity
        args_similarity = len(set(func1['args']) & set(func2['args'])) / max(len(set(func1['args']) | set(func2['args'])), 1)
        
        # Complexity similarity (should be similar complexity)
        complexity_ratio = min(func1['complexity'], func2['complexity']) / max(func1['complexity'], func2['complexity'])
        
        # Overall similarity score
        similarity = (name_similarity * 0.4 + args_similarity * 0.3 + complexity_ratio * 0.3)
        
        return similarity > 0.7  # High similarity threshold
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between function names."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def suggest_unified_name(self, functions: List[Dict]) -> str:
        """Suggest a unified name for consolidated functions."""
        names = [f['name'] for f in functions]
        
        # Find common words
        words = []
        for name in names:
            words.extend(re.findall(r'[a-zA-Z]+', name))
        
        word_counts = Counter(w.lower() for w in words)
        most_common_words = [word for word, count in word_counts.most_common(3) if count > 1]
        
        if most_common_words:
            return '_'.join(most_common_words)
        else:
            return f"unified_{names[0].lower()}"
    
    def create_utility_module(self, category: str, opportunities: List[Dict]) -> Path:
        """Create a consolidated utility module."""
        module_path = self.common_utils_dir / f"{category}.py"
        
        module_content = f'''"""
{category.replace('_', ' ').title()} for LeanVibe Agent Hive 2.0.

Consolidated utility functions for {category.replace('_', ' ')} operations.
Auto-generated from {len(opportunities)} consolidation opportunities.
"""

from typing import Any, Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

'''
        
        for i, opportunity in enumerate(opportunities):
            if opportunity['function_count'] >= 3:
                # Create a template consolidated function
                func_name = opportunity['consolidated_function_name']
                module_content += f'''

def {func_name}(*args, **kwargs) -> Any:
    """
    Consolidated function from {opportunity['function_count']} similar implementations.
    
    This function consolidates similar logic from:
    {chr(10).join(f"    - {func['file']}:{func['line']}" for func in opportunity['functions'][:3])}
    {'    ... and more' if len(opportunity['functions']) > 3 else ''}
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Any: Result based on the consolidated logic
    """
    # TODO: Implement consolidated logic from similar functions
    logger.info(f"Executing consolidated {func_name}")
    raise NotImplementedError(f"Consolidated function {func_name} needs implementation")

'''
        
        # Add __all__ export list
        exported_functions = [opp['consolidated_function_name'] for opp in opportunities if opp['function_count'] >= 3]
        module_content += f'''
__all__ = {exported_functions}
'''
        
        # Write the module
        self.common_utils_dir.mkdir(parents=True, exist_ok=True)
        module_path.write_text(module_content)
        
        return module_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate utility functions')
    parser.add_argument('--analyze', action='store_true', help='Analyze utility consolidation opportunities')
    parser.add_argument('--consolidate', action='store_true', help='Create consolidated utility modules')
    
    args = parser.parse_args()
    
    consolidator = UtilityFunctionConsolidator()
    
    try:
        utility_groups = consolidator.extract_utility_patterns()
        
        if args.consolidate:
            print("🚀 Creating consolidated utility modules...")
            results = consolidator.create_consolidated_utilities(utility_groups, dry_run=False)
        else:
            print("📋 Analyzing utility consolidation opportunities...")
            results = consolidator.create_consolidated_utilities(utility_groups, dry_run=True)
        
        # Print summary
        total_savings = sum(result['total_estimated_savings'] for result in results.values())
        total_functions = sum(result['total_functions_analyzed'] for result in results.values())
        
        print(f"\\n📊 Utility Consolidation Results:")
        print(f"   🔧 {len(results)} utility categories analyzed")
        print(f"   📁 {total_functions} functions analyzed")
        print(f"   💰 {total_savings:,} LOC estimated savings")
        
        for category, result in results.items():
            opps = result['consolidation_opportunities']
            if opps:
                print(f"   • {category}: {len(opps)} consolidation opportunities, {result['total_estimated_savings']:,} LOC savings")
        
        if args.consolidate:
            created_modules = [result.get('module_created') for result in results.values() if 'module_created' in result]
            print(f"\\n✅ Created {len(created_modules)} consolidated utility modules")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Run 'python advanced_semantic_analyzer.py --analyze' first to generate the semantic analysis report.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")