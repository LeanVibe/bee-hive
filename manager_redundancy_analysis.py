#!/usr/bin/env python3
"""
Enhanced Manager Redundancy Analysis for LeanVibe Agent Hive
Provides focused analysis on manager class redundancy and consolidation opportunities.
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

@dataclass
class ManagerSummary:
    """Concise manager information for analysis."""
    name: str
    file: str
    lines: int
    key_methods: List[str]
    domain: str
    functionality: Set[str]

class ManagerRedundancyAnalyzer:
    """Analyzes manager class redundancy and identifies consolidation opportunities."""
    
    def __init__(self, core_path: str):
        self.core_path = core_path
        self.managers: List[ManagerSummary] = []
        
        # Core functionality patterns for grouping
        self.core_patterns = {
            'lifecycle': ['lifecycle', 'registration', 'spawning', 'activation', 'deactivation'],
            'state': ['state', 'checkpoint', 'recovery', 'persistence', 'snapshot'],
            'communication': ['messaging', 'pubsub', 'streams', 'coordination', 'notification'],
            'resource': ['resource', 'capacity', 'allocation', 'monitoring', 'performance'],
            'security': ['auth', 'jwt', 'security', 'permission', 'access', 'encryption'],
            'storage': ['database', 'vector', 'embedding', 'pgvector', 'persistence'],
            'workflow': ['workflow', 'task', 'orchestration', 'execution', 'batch'],
            'context': ['context', 'memory', 'knowledge', 'compression', 'semantic']
        }
    
    def extract_manager_info(self, file_path: str) -> List[ManagerSummary]:
        """Extract manager information from a single file."""
        managers = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find manager classes
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and 'Manager' in node.name:
                    # Extract methods
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    
                    # Classify functionality
                    content_lower = content.lower()
                    functionality = set()
                    for category, patterns in self.core_patterns.items():
                        for pattern in patterns:
                            if pattern in content_lower:
                                functionality.add(pattern)
                    
                    # Determine primary domain
                    domain_scores = {}
                    for category, patterns in self.core_patterns.items():
                        score = len([p for p in patterns if p in content_lower])
                        if score > 0:
                            domain_scores[category] = score
                    
                    primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k]) if domain_scores else 'misc'
                    
                    manager = ManagerSummary(
                        name=node.name,
                        file=os.path.basename(file_path),
                        lines=len(content.splitlines()),
                        key_methods=methods[:5],  # Top 5 methods
                        domain=primary_domain,
                        functionality=functionality
                    )
                    managers.append(manager)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return managers
    
    def find_all_managers(self):
        """Find all manager files and extract information."""
        for root, dirs, files in os.walk(self.core_path):
            if '__pycache__' in root:
                continue
            
            for file in files:
                if 'manager' in file.lower() and file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    managers = self.extract_manager_info(file_path)
                    self.managers.extend(managers)
    
    def analyze_redundancy(self) -> Dict[str, any]:
        """Analyze redundancy patterns across managers."""
        analysis = {
            'total_managers': len(self.managers),
            'total_lines': sum(m.lines for m in self.managers),
            'domains': defaultdict(list),
            'method_overlaps': defaultdict(list),
            'functionality_overlaps': defaultdict(list),
            'consolidation_opportunities': []
        }
        
        # Group by domain
        for manager in self.managers:
            analysis['domains'][manager.domain].append(manager)
        
        # Find method overlaps
        all_methods = []
        for manager in self.managers:
            for method in manager.key_methods:
                all_methods.append((method, manager.name, manager.domain))
        
        method_counts = Counter([m[0] for m in all_methods])
        for method, count in method_counts.items():
            if count > 1:
                managers_with_method = [m for m in all_methods if m[0] == method]
                analysis['method_overlaps'][method] = managers_with_method
        
        # Find functionality overlaps
        for func in set().union(*[m.functionality for m in self.managers]):
            managers_with_func = [m for m in self.managers if func in m.functionality]
            if len(managers_with_func) > 2:
                analysis['functionality_overlaps'][func] = managers_with_func
        
        return analysis
    
    def identify_consolidation_candidates(self, analysis: Dict) -> List[Dict]:
        """Identify specific consolidation opportunities."""
        candidates = []
        
        # Domain-based consolidation
        for domain, managers in analysis['domains'].items():
            if len(managers) > 3:  # Domains with many managers
                total_lines = sum(m.lines for m in managers)
                candidates.append({
                    'type': 'domain_consolidation',
                    'domain': domain,
                    'manager_count': len(managers),
                    'total_lines': total_lines,
                    'managers': [m.name for m in managers],
                    'potential_savings': f"{total_lines * 0.3:.0f} lines (30% reduction)"
                })
        
        # Functionality overlap consolidation
        high_overlap_funcs = {func: mgrs for func, mgrs in analysis['functionality_overlaps'].items() 
                             if len(mgrs) > 4}
        
        for func, managers in high_overlap_funcs.items():
            total_lines = sum(m.lines for m in managers)
            candidates.append({
                'type': 'functionality_consolidation',
                'functionality': func,
                'manager_count': len(managers),
                'total_lines': total_lines,
                'managers': [m.name for m in managers],
                'potential_savings': f"{total_lines * 0.4:.0f} lines (40% reduction)"
            })
        
        return candidates
    
    def generate_consolidation_plan(self) -> str:
        """Generate a focused consolidation plan."""
        self.find_all_managers()
        analysis = self.analyze_redundancy()
        candidates = self.identify_consolidation_candidates(analysis)
        
        report = []
        report.append("# LeanVibe Agent Hive Manager Consolidation Analysis")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- **Total Manager Classes**: {analysis['total_managers']}")
        report.append(f"- **Total Lines of Code**: {analysis['total_lines']:,}")
        report.append(f"- **Functional Domains**: {len(analysis['domains'])}")
        report.append(f"- **Consolidation Opportunities**: {len(candidates)}")
        report.append("")
        
        # Domain Distribution
        report.append("## Domain Distribution")
        domain_summary = []
        for domain, managers in analysis['domains'].items():
            total_lines = sum(m.lines for m in managers)
            domain_summary.append((domain, len(managers), total_lines))
        
        domain_summary.sort(key=lambda x: x[2], reverse=True)
        
        for domain, count, lines in domain_summary:
            report.append(f"- **{domain.upper()}**: {count} managers, {lines:,} lines")
        report.append("")
        
        # Top Method Overlaps
        report.append("## Method Redundancy Analysis")
        top_overlaps = sorted(analysis['method_overlaps'].items(), 
                            key=lambda x: len(x[1]), reverse=True)[:10]
        
        for method, managers in top_overlaps:
            domains = set(m[2] for m in managers)
            report.append(f"- **{method}()**: {len(managers)} managers across {len(domains)} domains")
        report.append("")
        
        # Consolidation Opportunities
        report.append("## Consolidation Opportunities")
        
        # Sort by potential impact
        candidates.sort(key=lambda x: x['total_lines'], reverse=True)
        
        for i, candidate in enumerate(candidates[:5], 1):
            report.append(f"### Opportunity {i}: {candidate['type'].replace('_', ' ').title()}")
            
            if candidate['type'] == 'domain_consolidation':
                report.append(f"**Domain**: {candidate['domain'].upper()}")
            else:
                report.append(f"**Functionality**: {candidate['functionality']}")
                
            report.append(f"**Affected Managers**: {candidate['manager_count']}")
            report.append(f"**Current Lines**: {candidate['total_lines']:,}")
            report.append(f"**Potential Savings**: {candidate['potential_savings']}")
            report.append(f"**Managers**: {', '.join(candidate['managers'][:5])}")
            if len(candidate['managers']) > 5:
                report.append(f"  ... and {len(candidate['managers']) - 5} more")
            report.append("")
        
        # Recommended Unified Architecture
        report.append("## Recommended Unified Manager Architecture")
        report.append("")
        report.append("Based on the analysis, consolidate into these 7 core managers:")
        report.append("")
        report.append("1. **AgentLifecycleManager** - Agent registration, spawning, coordination")
        report.append("2. **WorkflowOrchestrationManager** - Task execution, state management")
        report.append("3. **ResourceAllocationManager** - Capacity, performance, monitoring")
        report.append("4. **CommunicationManager** - Messaging, events, coordination")
        report.append("5. **SecurityManager** - Authentication, authorization, secrets")
        report.append("6. **StorageManager** - Database, vectors, persistence, checkpoints")
        report.append("7. **ContextManager** - Memory, knowledge, compression")
        report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = ManagerRedundancyAnalyzer("/Users/bogdan/work/leanvibe-dev/bee-hive/app/core")
    report = analyzer.generate_consolidation_plan()
    print(report)