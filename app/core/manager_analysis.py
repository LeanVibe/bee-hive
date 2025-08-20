import asyncio
#!/usr/bin/env python3
"""
Manager Analysis Tool for Epic 1.5 Consolidation
Analyzes all manager classes to understand their functionality and grouping.
"""

import os
import re
import ast
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ManagerInfo:
    """Information about a manager class."""
    file_path: str
    class_name: str
    line_count: int
    methods: List[str]
    imports: List[str]
    dependencies: List[str]
    docstring: str
    functionality_keywords: Set[str]

@dataclass
class FunctionalDomain:
    """Functional domain grouping for managers."""
    name: str
    description: str
    managers: List[ManagerInfo]
    total_lines: int
    redundancy_score: float
    core_capabilities: Set[str]

class ManagerAnalyzer:
    """Analyzes manager classes for consolidation planning."""
    
    def __init__(self, core_path: str):
        self.core_path = core_path
        self.managers: List[ManagerInfo] = []
        self.domains: Dict[str, FunctionalDomain] = {}
        
        # Functionality keywords for domain classification
        self.domain_keywords = {
            'workflow': {'workflow', 'task', 'orchestration', 'execution', 'pipeline', 'state', 'batch'},
            'agent': {'agent', 'lifecycle', 'registration', 'spawning', 'coordination', 'persona'},
            'resource': {'resource', 'capacity', 'allocation', 'monitoring', 'performance', 'memory', 'cpu'},
            'communication': {'redis', 'pubsub', 'streams', 'messaging', 'coordination', 'event', 'notification'},
            'configuration': {'config', 'settings', 'feature_flag', 'secret', 'api_key', 'environment'},
            'storage': {'database', 'persistence', 'checkpoint', 'recovery', 'backup', 'pgvector', 'vector'},
            'context': {'context', 'memory', 'knowledge', 'embedding', 'semantic', 'compression'},
            'security': {'auth', 'jwt', 'security', 'encryption', 'permission', 'access'},
            'infrastructure': {'tmux', 'session', 'branch', 'git', 'workspace', 'work_tree', 'deployment'},
            'monitoring': {'monitoring', 'observability', 'metrics', 'logging', 'health', 'status'}
        }
    
    def analyze_manager_file(self, file_path: str) -> List[ManagerInfo]:
        """Analyze a single manager file and extract manager class information."""
        managers = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Count lines
            line_count = len(content.splitlines())
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Find manager classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and 'Manager' in node.name:
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    
                    # Extract docstring
                    docstring = ""
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant)):
                        docstring = node.body[0].value.value or ""
                    
                    # Extract functionality keywords from content
                    content_lower = content.lower()
                    functionality_keywords = set()
                    for domain, keywords in self.domain_keywords.items():
                        for keyword in keywords:
                            if keyword in content_lower:
                                functionality_keywords.add(keyword)
                    
                    # Extract dependencies from imports and content
                    dependencies = []
                    for imp in imports:
                        if any(keyword in imp.lower() for keywords in self.domain_keywords.values() for keyword in keywords):
                            dependencies.append(imp)
                    
                    manager_info = ManagerInfo(
                        file_path=file_path,
                        class_name=node.name,
                        line_count=line_count,
                        methods=methods,
                        imports=imports,
                        dependencies=dependencies,
                        docstring=docstring,
                        functionality_keywords=functionality_keywords
                    )
                    managers.append(manager_info)
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return managers
    
    def find_all_managers(self) -> None:
        """Find and analyze all manager files."""
        manager_patterns = ['*manager*.py', '*management*.py', '*coordinator*.py']
        
        for pattern in manager_patterns:
            for root, dirs, files in os.walk(self.core_path):
                # Skip __pycache__ directories
                if '__pycache__' in root:
                    continue
                    
                for file in files:
                    if (('manager' in file.lower() or 'management' in file.lower() or 'coordinator' in file.lower()) 
                        and file.endswith('.py')):
                        file_path = os.path.join(root, file)
                        managers = self.analyze_manager_file(file_path)
                        self.managers.extend(managers)
    
    def classify_managers_by_domain(self) -> None:
        """Classify managers into functional domains."""
        domain_managers = defaultdict(list)
        
        for manager in self.managers:
            # Score manager for each domain based on keywords
            domain_scores = {}
            for domain, keywords in self.domain_keywords.items():
                score = len(manager.functionality_keywords & keywords)
                if score > 0:
                    domain_scores[domain] = score
            
            # Assign to highest scoring domain(s)
            if domain_scores:
                max_score = max(domain_scores.values())
                best_domains = [d for d, s in domain_scores.items() if s == max_score]
                
                # For now, assign to first best domain (could be improved)
                primary_domain = best_domains[0]
                domain_managers[primary_domain].append(manager)
            else:
                # Default domain for unclassified managers
                domain_managers['misc'].append(manager)
        
        # Create domain objects
        for domain_name, managers in domain_managers.items():
            total_lines = sum(m.line_count for m in managers)
            
            # Calculate redundancy score (simplified)
            all_methods = []
            for m in managers:
                all_methods.extend(m.methods)
            unique_methods = set(all_methods)
            redundancy_score = 1.0 - (len(unique_methods) / len(all_methods)) if all_methods else 0.0
            
            # Extract core capabilities
            core_capabilities = set()
            for m in managers:
                core_capabilities.update(m.functionality_keywords)
            
            domain_description = {
                'workflow': 'Task orchestration, execution, and state management',
                'agent': 'Agent lifecycle, coordination, and persona management',
                'resource': 'System resource monitoring and allocation',
                'communication': 'Inter-agent messaging and event coordination',
                'configuration': 'System configuration and feature management',
                'storage': 'Data persistence, checkpointing, and recovery',
                'context': 'Context management, memory, and knowledge sharing',
                'security': 'Authentication, authorization, and security',
                'infrastructure': 'Development infrastructure and workspace management',
                'monitoring': 'System monitoring, observability, and health checks',
                'misc': 'Miscellaneous managers not fitting other categories'
            }.get(domain_name, f'Managers related to {domain_name}')
            
            self.domains[domain_name] = FunctionalDomain(
                name=domain_name,
                description=domain_description,
                managers=managers,
                total_lines=total_lines,
                redundancy_score=redundancy_score,
                core_capabilities=core_capabilities
            )
    
    def generate_consolidation_report(self) -> str:
        """Generate comprehensive consolidation analysis report."""
        total_managers = len(self.managers)
        total_lines = sum(m.line_count for m in self.managers)
        
        report = []
        report.append("# Manager Class Consolidation Analysis for Epic 1.5")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- **Total Manager Classes Found**: {total_managers}")
        report.append(f"- **Total Lines of Code**: {total_lines:,}")
        report.append(f"- **Functional Domains Identified**: {len(self.domains)}")
        report.append("")
        
        # Domain Analysis
        report.append("## Functional Domain Analysis")
        report.append("")
        
        for domain_name, domain in sorted(self.domains.items(), key=lambda x: x[1].total_lines, reverse=True):
            report.append(f"### {domain_name.upper()} Domain")
            report.append(f"**Description**: {domain.description}")
            report.append(f"**Manager Count**: {len(domain.managers)}")
            report.append(f"**Total Lines**: {domain.total_lines:,}")
            report.append(f"**Redundancy Score**: {domain.redundancy_score:.2%}")
            report.append(f"**Core Capabilities**: {', '.join(sorted(domain.core_capabilities))}")
            report.append("")
            
            report.append("**Managers in Domain**:")
            for manager in sorted(domain.managers, key=lambda x: x.line_count, reverse=True):
                filename = os.path.basename(manager.file_path)
                report.append(f"- `{manager.class_name}` ({filename}) - {manager.line_count} lines")
            report.append("")
        
        # Detailed Manager Inventory
        report.append("## Detailed Manager Inventory")
        report.append("")
        
        for domain_name, domain in sorted(self.domains.items()):
            report.append(f"### {domain_name.upper()} Domain Managers")
            
            for manager in sorted(domain.managers, key=lambda x: x.line_count, reverse=True):
                report.append(f"#### {manager.class_name}")
                report.append(f"- **File**: `{os.path.relpath(manager.file_path, self.core_path)}`")
                report.append(f"- **Lines of Code**: {manager.line_count}")
                report.append(f"- **Method Count**: {len(manager.methods)}")
                report.append(f"- **Key Methods**: {', '.join(manager.methods[:5])}")
                if manager.docstring:
                    report.append(f"- **Purpose**: {manager.docstring.split('.')[0]}")
                report.append("")
        
        return "\n".join(report)
    
    def analyze(self) -> str:
        """Run complete analysis and return report."""
        self.find_all_managers()
        self.classify_managers_by_domain()
        return self.generate_consolidation_report()

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ManagerAnalysisScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            analyzer = ManagerAnalyzer("/Users/bogdan/work/leanvibe-dev/bee-hive/app/core")
            report = analyzer.analyze()
            self.logger.info(report)
            
            return {"status": "completed"}
    
    script_main(ManagerAnalysisScript)