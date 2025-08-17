#!/usr/bin/env python3
"""
Project Index-based Manager Analysis
Uses the project index analyzer to understand manager class structure and dependencies.
"""

import sys
import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json

# Add app directory to path
sys.path.append('.')

from app.project_index.analyzer import CodeAnalyzer
from app.project_index.models import AnalysisConfiguration, FileAnalysisResult

class ProjectIndexManagerAnalyzer:
    """Uses project index analyzer to analyze manager classes."""
    
    def __init__(self, core_path: str):
        self.core_path = Path(core_path)
        self.analyzer = CodeAnalyzer()
        self.analysis_config = AnalysisConfiguration(
            extract_dependencies=True,
            analyze_complexity=True,
            extract_docstrings=True,
            calculate_metrics=True
        )
        
    def find_manager_files(self) -> List[Path]:
        """Find all manager files in the core directory."""
        manager_files = []
        
        for file_path in self.core_path.rglob("*.py"):
            if 'manager' in file_path.name.lower() and '__pycache__' not in str(file_path):
                manager_files.append(file_path)
        
        return manager_files
    
    def analyze_manager_structure(self) -> Dict[str, any]:
        """Analyze manager files using project index analyzer."""
        manager_files = self.find_manager_files()
        analysis_results = {}
        
        print(f"Found {len(manager_files)} manager files to analyze...")
        
        for file_path in manager_files:
            try:
                print(f"Analyzing {file_path.name}...")
                
                # Use project index analyzer
                result = self.analyzer.analyze_file(file_path, self.analysis_config)
                
                if result and result.success:
                    analysis_results[file_path.name] = {
                        'file_path': str(file_path),
                        'classes': result.classes,
                        'functions': result.functions,
                        'dependencies': result.dependencies,
                        'complexity': result.complexity_metrics,
                        'lines_of_code': result.lines_of_code,
                        'imports': result.imports,
                        'docstring': result.docstring
                    }
                else:
                    print(f"Failed to analyze {file_path}: {result.error if result else 'Unknown error'}")
                    
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        return analysis_results
    
    def extract_manager_relationships(self, analysis_results: Dict) -> Dict[str, Set[str]]:
        """Extract relationships between managers."""
        relationships = defaultdict(set)
        
        # Map class names to files
        class_to_file = {}
        for file_name, analysis in analysis_results.items():
            for class_info in analysis.get('classes', []):
                class_to_file[class_info['name']] = file_name
        
        # Find relationships through imports and dependencies
        for file_name, analysis in analysis_results.items():
            current_managers = {cls['name'] for cls in analysis.get('classes', []) if 'Manager' in cls['name']}
            
            # Check dependencies
            for dep in analysis.get('dependencies', []):
                dep_name = dep.get('name', '')
                if 'Manager' in dep_name and dep_name in class_to_file:
                    for current_manager in current_managers:
                        relationships[current_manager].add(dep_name)
            
            # Check imports
            for imp in analysis.get('imports', []):
                imp_name = imp.get('name', '')
                if 'manager' in imp_name.lower():
                    # Try to map to manager classes
                    for class_name, class_file in class_to_file.items():
                        if imp_name.lower() in class_file.lower():
                            for current_manager in current_managers:
                                relationships[current_manager].add(class_name)
        
        return dict(relationships)
    
    def categorize_by_functionality(self, analysis_results: Dict) -> Dict[str, List[str]]:
        """Categorize managers by functionality based on analysis."""
        categories = defaultdict(list)
        
        # Keyword mapping for categorization
        category_keywords = {
            'lifecycle': ['lifecycle', 'spawn', 'register', 'activate', 'deactivate'],
            'state': ['state', 'checkpoint', 'recovery', 'persistence', 'snapshot', 'backup'],
            'communication': ['message', 'pubsub', 'stream', 'redis', 'coordination', 'event'],
            'resource': ['resource', 'capacity', 'allocation', 'monitoring', 'performance', 'cpu', 'memory'],
            'security': ['auth', 'jwt', 'security', 'permission', 'access', 'encryption', 'secret', 'key'],
            'storage': ['database', 'vector', 'embedding', 'pgvector', 'persistence', 'cache'],
            'workflow': ['workflow', 'task', 'orchestration', 'execution', 'batch', 'pipeline'],
            'context': ['context', 'memory', 'knowledge', 'compression', 'semantic']
        }
        
        for file_name, analysis in analysis_results.items():
            file_content = []
            
            # Collect content for analysis
            if analysis.get('docstring'):
                file_content.append(analysis['docstring'].lower())
            
            for cls in analysis.get('classes', []):
                if cls.get('docstring'):
                    file_content.append(cls['docstring'].lower())
                file_content.append(cls['name'].lower())
                
                for method in cls.get('methods', []):
                    file_content.append(method['name'].lower())
            
            for func in analysis.get('functions', []):
                file_content.append(func['name'].lower())
            
            content_text = ' '.join(file_content)
            
            # Score each category
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content_text)
                if score > 0:
                    category_scores[category] = score
            
            # Assign to best category
            if category_scores:
                best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
                categories[best_category].append(file_name)
            else:
                categories['miscellaneous'].append(file_name)
        
        return dict(categories)
    
    def calculate_consolidation_metrics(self, analysis_results: Dict) -> Dict[str, any]:
        """Calculate metrics for consolidation planning."""
        metrics = {
            'total_files': len(analysis_results),
            'total_classes': 0,
            'total_methods': 0,
            'total_lines': 0,
            'complexity_distribution': defaultdict(int),
            'common_patterns': defaultdict(int)
        }
        
        all_method_names = []
        all_class_names = []
        
        for file_name, analysis in analysis_results.items():
            metrics['total_lines'] += analysis.get('lines_of_code', 0)
            
            for cls in analysis.get('classes', []):
                metrics['total_classes'] += 1
                all_class_names.append(cls['name'])
                
                for method in cls.get('methods', []):
                    metrics['total_methods'] += 1
                    all_method_names.append(method['name'])
                    
                    # Complexity distribution
                    complexity = method.get('complexity', 0)
                    if complexity < 5:
                        metrics['complexity_distribution']['low'] += 1
                    elif complexity < 10:
                        metrics['complexity_distribution']['medium'] += 1
                    else:
                        metrics['complexity_distribution']['high'] += 1
        
        # Find common patterns
        from collections import Counter
        method_counts = Counter(all_method_names)
        metrics['common_methods'] = {name: count for name, count in method_counts.items() if count > 1}
        
        class_name_parts = []
        for name in all_class_names:
            # Split camelCase names
            parts = []
            current = ""
            for char in name:
                if char.isupper() and current:
                    parts.append(current.lower())
                    current = char
                else:
                    current += char
            if current:
                parts.append(current.lower())
            class_name_parts.extend(parts)
        
        part_counts = Counter(class_name_parts)
        metrics['common_name_parts'] = {part: count for part, count in part_counts.items() if count > 2}
        
        return metrics
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        print("Starting Project Index-based manager analysis...")
        
        analysis_results = self.analyze_manager_structure()
        relationships = self.extract_manager_relationships(analysis_results)
        categories = self.categorize_by_functionality(analysis_results)
        metrics = self.calculate_consolidation_metrics(analysis_results)
        
        report = []
        report.append("# Project Index Manager Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        report.append("## Analysis Summary")
        report.append(f"- **Manager Files Analyzed**: {metrics['total_files']}")
        report.append(f"- **Total Manager Classes**: {metrics['total_classes']}")
        report.append(f"- **Total Methods**: {metrics['total_methods']}")
        report.append(f"- **Total Lines of Code**: {metrics['total_lines']:,}")
        report.append("")
        
        # Complexity Analysis
        report.append("## Complexity Distribution")
        complexity_dist = metrics['complexity_distribution']
        total_methods = sum(complexity_dist.values())
        
        if total_methods > 0:
            report.append(f"- **Low Complexity** (<5): {complexity_dist['low']} ({complexity_dist['low']/total_methods:.1%})")
            report.append(f"- **Medium Complexity** (5-10): {complexity_dist['medium']} ({complexity_dist['medium']/total_methods:.1%})")
            report.append(f"- **High Complexity** (>10): {complexity_dist['high']} ({complexity_dist['high']/total_methods:.1%})")
        report.append("")
        
        # Functional Categories
        report.append("## Functional Categories")
        for category, files in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            total_lines = sum(analysis_results[f].get('lines_of_code', 0) for f in files if f in analysis_results)
            report.append(f"### {category.upper()} ({len(files)} files, {total_lines:,} lines)")
            
            for file_name in sorted(files):
                if file_name in analysis_results:
                    lines = analysis_results[file_name].get('lines_of_code', 0)
                    classes = len(analysis_results[file_name].get('classes', []))
                    report.append(f"- `{file_name}` - {lines} lines, {classes} classes")
            report.append("")
        
        # Common Patterns
        if metrics['common_methods']:
            report.append("## Most Duplicated Methods")
            for method, count in sorted(metrics['common_methods'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"- `{method}()`: {count} implementations")
            report.append("")
        
        if metrics['common_name_parts']:
            report.append("## Common Class Name Patterns")
            for part, count in sorted(metrics['common_name_parts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"- `{part}`: {count} occurrences")
            report.append("")
        
        # Relationship Analysis
        if relationships:
            report.append("## Manager Relationships")
            report.append("*Dependencies between manager classes*")
            report.append("")
            
            for manager, deps in sorted(relationships.items()):
                if deps:
                    report.append(f"### {manager}")
                    report.append(f"**Dependencies**: {', '.join(sorted(deps))}")
                    report.append("")
        
        # Consolidation Recommendations
        report.append("## Project Index Consolidation Recommendations")
        report.append("")
        
        # Group by category for consolidation
        large_categories = {cat: files for cat, files in categories.items() if len(files) >= 3}
        
        for category, files in large_categories.items():
            total_lines = sum(analysis_results[f].get('lines_of_code', 0) for f in files if f in analysis_results)
            total_classes = sum(len(analysis_results[f].get('classes', [])) for f in files if f in analysis_results)
            
            estimated_consolidated = total_lines * 0.6  # Assume 40% reduction
            
            report.append(f"### Consolidate {category.upper()} Domain")
            report.append(f"- **Current**: {len(files)} files, {total_lines:,} lines, {total_classes} classes")
            report.append(f"- **Target**: 1-2 unified managers, ~{estimated_consolidated:.0f} lines")
            report.append(f"- **Reduction**: {(total_lines - estimated_consolidated) / total_lines:.0%}")
            report.append(f"- **Files**: {', '.join(files)}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = ProjectIndexManagerAnalyzer("./app/core")
    report = analyzer.generate_analysis_report()
    print(report)