#!/usr/bin/env python3
"""
Comprehensive Technical Debt Analysis for Multi-CLI Agent Coordination System
Using First Principles Thinking to identify consolidation opportunities.
"""

import os
import ast
import re
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    path: str
    name: str
    lines_of_code: int
    classes: List[str]
    functions: List[str]
    imports: List[str]
    purpose: str  # extracted from docstring
    functionality_keywords: Set[str]
    dependencies: List[str]
    complexity_score: float = 0.0

@dataclass
class RedundancyPattern:
    """Identifies redundant functionality patterns."""
    pattern_name: str
    files: List[str]
    total_lines: int
    redundancy_score: float  # 0-1 scale
    consolidation_target: str
    effort_estimate: str

@dataclass
class TechnicalDebt:
    """Technical debt analysis results."""
    category: str
    severity: str  # Critical, High, Medium, Low  
    description: str
    affected_files: List[str]
    total_lines: int
    impact: str
    recommendation: str
    effort_estimate: str

class TechnicalDebtAnalyzer:
    """Comprehensive technical debt analyzer."""
    
    def __init__(self, app_path: str, core_path: str):
        self.app_path = app_path
        self.core_path = core_path
        self.file_analyses: List[FileAnalysis] = []
        self.redundancy_patterns: List[RedundancyPattern] = []
        self.technical_debts: List[TechnicalDebt] = []
        
        # Pattern keywords for classification
        self.functionality_patterns = {
            'orchestrator': {
                'keywords': {'orchestrator', 'orchestration', 'coordinate', 'delegate', 'manage', 'schedule'},
                'class_patterns': [r'.*Orchestrator.*', r'.*Manager.*', r'.*Coordinator.*']
            },
            'manager': {
                'keywords': {'manager', 'management', 'lifecycle', 'resource', 'memory', 'context'},
                'class_patterns': [r'.*Manager.*', r'.*Management.*']
            },
            'engine': {
                'keywords': {'engine', 'processor', 'executor', 'handler', 'worker'},
                'class_patterns': [r'.*Engine.*', r'.*Processor.*', r'.*Executor.*']
            },
            'communication': {
                'keywords': {'message', 'communication', 'websocket', 'redis', 'event', 'pubsub'},
                'class_patterns': [r'.*Communication.*', r'.*Message.*', r'.*Event.*']
            },
            'context': {
                'keywords': {'context', 'memory', 'compression', 'consolidation', 'semantic'},
                'class_patterns': [r'.*Context.*', r'.*Memory.*', r'.*Semantic.*']
            },
            'security': {
                'keywords': {'auth', 'security', 'permission', 'access', 'rbac', 'jwt'},
                'class_patterns': [r'.*Auth.*', r'.*Security.*', r'.*Access.*']
            },
            'workflow': {
                'keywords': {'workflow', 'task', 'execution', 'pipeline', 'job'},
                'class_patterns': [r'.*Workflow.*', r'.*Task.*', r'.*Job.*']
            }
        }
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return None
            
            # Basic metrics
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            # Extract classes and functions
            classes = []
            functions = []
            docstring = ""
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            # Extract module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                docstring = tree.body[0].value.value
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Extract functionality keywords
            content_lower = content.lower()
            functionality_keywords = set()
            for pattern_name, pattern_info in self.functionality_patterns.items():
                for keyword in pattern_info['keywords']:
                    if keyword in content_lower:
                        functionality_keywords.add(keyword)
            
            # Calculate complexity score (simplified)
            complexity_score = len(classes) * 0.3 + len(functions) * 0.1 + lines_of_code * 0.001
            
            return FileAnalysis(
                path=file_path,
                name=os.path.basename(file_path),
                lines_of_code=lines_of_code,
                classes=classes,
                functions=functions,
                imports=imports,
                purpose=docstring.split('\n')[0] if docstring else "",
                functionality_keywords=functionality_keywords,
                dependencies=[imp for imp in imports if any(kw in imp.lower() for pattern in self.functionality_patterns.values() for kw in pattern['keywords'])],
                complexity_score=complexity_score
            )
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def scan_directory(self, directory: str) -> None:
        """Scan directory for Python files and analyze them."""
        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'node_modules', '.pytest_cache'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        self.file_analyses.append(analysis)
    
    def identify_orchestrator_redundancy(self) -> None:
        """Identify orchestrator redundancy patterns."""
        orchestrator_files = [f for f in self.file_analyses if 'orchestrator' in f.name.lower() or 
                             any('orchestrator' in cls.lower() for cls in f.classes)]
        
        if len(orchestrator_files) > 5:  # Threshold for redundancy
            total_lines = sum(f.lines_of_code for f in orchestrator_files)
            
            # Calculate redundancy score based on similar functionality
            functionality_overlap = 0
            all_keywords = set()
            for f in orchestrator_files:
                all_keywords.update(f.functionality_keywords)
            
            if all_keywords:
                keyword_counts = Counter()
                for f in orchestrator_files:
                    for keyword in f.functionality_keywords:
                        keyword_counts[keyword] += 1
                
                # Redundancy score: how many keywords appear in multiple files
                overlapping_keywords = [k for k, v in keyword_counts.items() if v > 1]
                redundancy_score = len(overlapping_keywords) / len(all_keywords) if all_keywords else 0
            else:
                redundancy_score = 0
            
            pattern = RedundancyPattern(
                pattern_name="Orchestrator Redundancy",
                files=[f.name for f in orchestrator_files],
                total_lines=total_lines,
                redundancy_score=redundancy_score,
                consolidation_target="unified_orchestrator.py",
                effort_estimate="High (4-6 weeks)"
            )
            self.redundancy_patterns.append(pattern)
    
    def identify_manager_redundancy(self) -> None:
        """Identify manager class redundancy."""
        manager_files = [f for f in self.file_analyses if 'manager' in f.name.lower() or 
                        any('manager' in cls.lower() for cls in f.classes)]
        
        # Group by functionality
        functionality_groups = defaultdict(list)
        for f in manager_files:
            primary_function = self.classify_primary_function(f)
            functionality_groups[primary_function].append(f)
        
        for function_name, files in functionality_groups.items():
            if len(files) > 3:  # Threshold for redundancy
                total_lines = sum(f.lines_of_code for f in files)
                redundancy_score = self.calculate_redundancy_score(files)
                
                pattern = RedundancyPattern(
                    pattern_name=f"{function_name.title()} Manager Redundancy",
                    files=[f.name for f in files],
                    total_lines=total_lines,
                    redundancy_score=redundancy_score,
                    consolidation_target=f"unified_{function_name}_manager.py",
                    effort_estimate="Medium (2-4 weeks)" if len(files) < 8 else "High (4-6 weeks)"
                )
                self.redundancy_patterns.append(pattern)
    
    def identify_communication_redundancy(self) -> None:
        """Identify communication protocol redundancy."""
        comm_files = [f for f in self.file_analyses if 
                     any(keyword in f.functionality_keywords for keyword in ['message', 'communication', 'websocket', 'redis', 'event', 'pubsub'])]
        
        if len(comm_files) > 5:
            total_lines = sum(f.lines_of_code for f in comm_files)
            redundancy_score = self.calculate_redundancy_score(comm_files)
            
            pattern = RedundancyPattern(
                pattern_name="Communication Protocol Redundancy",
                files=[f.name for f in comm_files],
                total_lines=total_lines,
                redundancy_score=redundancy_score,
                consolidation_target="unified_communication_service.py",
                effort_estimate="High (3-5 weeks)"
            )
            self.redundancy_patterns.append(pattern)
    
    def classify_primary_function(self, file_analysis: FileAnalysis) -> str:
        """Classify file's primary function."""
        keywords = file_analysis.functionality_keywords
        
        for function_name, pattern_info in self.functionality_patterns.items():
            overlap = keywords & pattern_info['keywords']
            if overlap:
                return function_name
        
        return 'misc'
    
    def calculate_redundancy_score(self, files: List[FileAnalysis]) -> float:
        """Calculate redundancy score for a group of files."""
        all_functions = []
        all_keywords = set()
        
        for f in files:
            all_functions.extend(f.functions)
            all_keywords.update(f.functionality_keywords)
        
        if not all_functions and not all_keywords:
            return 0.0
        
        # Function name similarity
        function_counter = Counter(all_functions)
        duplicate_functions = sum(count - 1 for count in function_counter.values() if count > 1)
        function_redundancy = duplicate_functions / len(all_functions) if all_functions else 0
        
        # Keyword overlap
        keyword_counter = Counter()
        for f in files:
            for keyword in f.functionality_keywords:
                keyword_counter[keyword] += 1
        
        overlapping_keywords = sum(count - 1 for count in keyword_counter.values() if count > 1)
        keyword_redundancy = overlapping_keywords / len(all_keywords) if all_keywords else 0
        
        return (function_redundancy + keyword_redundancy) / 2
    
    def identify_technical_debts(self) -> None:
        """Identify various forms of technical debt."""
        
        # Critical: Multiple orchestrator implementations
        orchestrator_files = [f for f in self.file_analyses if 'orchestrator' in f.name.lower()]
        if len(orchestrator_files) > 10:
            self.technical_debts.append(TechnicalDebt(
                category="Architecture",
                severity="Critical",
                description=f"Multiple orchestrator implementations ({len(orchestrator_files)} files)",
                affected_files=[f.name for f in orchestrator_files],
                total_lines=sum(f.lines_of_code for f in orchestrator_files),
                impact="High maintenance overhead, inconsistent behavior, performance bottlenecks",
                recommendation="Consolidate into unified orchestrator with plugin architecture",
                effort_estimate="6-8 weeks"
            ))
        
        # High: Manager class proliferation
        manager_files = [f for f in self.file_analyses if 'manager' in f.name.lower()]
        if len(manager_files) > 30:
            self.technical_debts.append(TechnicalDebt(
                category="Code Organization",
                severity="High",
                description=f"Excessive manager classes ({len(manager_files)} files)",
                affected_files=[f.name for f in manager_files[:10]],  # Top 10
                total_lines=sum(f.lines_of_code for f in manager_files),
                impact="Code duplication, testing complexity, maintenance burden",
                recommendation="Consolidate by functional domain into unified managers",
                effort_estimate="4-6 weeks"
            ))
        
        # High: Engine redundancy
        engine_files = [f for f in self.file_analyses if 'engine' in f.name.lower()]
        if len(engine_files) > 20:
            self.technical_debts.append(TechnicalDebt(
                category="Performance",
                severity="High",
                description=f"Multiple engine implementations ({len(engine_files)} files)",
                affected_files=[f.name for f in engine_files],
                total_lines=sum(f.lines_of_code for f in engine_files),
                impact="Resource inefficiency, performance degradation, complexity",
                recommendation="Consolidate into specialized high-performance engines",
                effort_estimate="5-7 weeks"
            ))
        
        # Medium: Communication protocol duplication
        comm_keywords = {'message', 'communication', 'websocket', 'redis', 'event', 'pubsub'}
        comm_files = [f for f in self.file_analyses if f.functionality_keywords & comm_keywords]
        if len(comm_files) > 15:
            self.technical_debts.append(TechnicalDebt(
                category="Integration",
                severity="Medium",
                description=f"Communication protocol duplication ({len(comm_files)} files)",
                affected_files=[f.name for f in comm_files[:5]],
                total_lines=sum(f.lines_of_code for f in comm_files),
                impact="Inconsistent messaging, integration issues, debugging complexity",
                recommendation="Standardize on unified communication layer",
                effort_estimate="3-4 weeks"
            ))
        
        # Medium: Configuration management sprawl
        config_files = [f for f in self.file_analyses if any(keyword in f.functionality_keywords 
                       for keyword in ['config', 'settings', 'feature_flag', 'secret'])]
        if len(config_files) > 10:
            self.technical_debts.append(TechnicalDebt(
                category="Configuration",
                severity="Medium",
                description=f"Configuration management sprawl ({len(config_files)} files)",
                affected_files=[f.name for f in config_files],
                total_lines=sum(f.lines_of_code for f in config_files),
                impact="Configuration inconsistency, deployment complexity",
                recommendation="Centralize configuration management",
                effort_estimate="2-3 weeks"
            ))
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive technical debt analysis report."""
        total_files = len(self.file_analyses)
        total_lines = sum(f.lines_of_code for f in self.file_analyses)
        
        report = []
        report.append("# COMPREHENSIVE TECHNICAL DEBT ANALYSIS")
        report.append("## Multi-CLI Agent Coordination System")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("## EXECUTIVE SUMMARY")
        report.append("")
        report.append("### First Principles Analysis")
        report.append("**Core Purpose**: Multi-agent coordination and task delegation system")
        report.append("**Essential Components**: Agent management, task orchestration, communication")
        report.append("**Current State**: Significant architectural redundancy and over-engineering")
        report.append("")
        report.append("### Key Findings")
        report.append(f"- **Total Files Analyzed**: {total_files:,}")
        report.append(f"- **Total Lines of Code**: {total_lines:,}")
        report.append(f"- **Redundancy Patterns Found**: {len(self.redundancy_patterns)}")
        report.append(f"- **Technical Debt Issues**: {len(self.technical_debts)}")
        report.append("")
        
        critical_debt = [d for d in self.technical_debts if d.severity == "Critical"]
        high_debt = [d for d in self.technical_debts if d.severity == "High"]
        
        report.append("### Impact Assessment")
        report.append(f"- **Critical Issues**: {len(critical_debt)}")
        report.append(f"- **High Priority Issues**: {len(high_debt)}")
        
        # Calculate potential savings
        total_redundant_lines = sum(p.total_lines for p in self.redundancy_patterns)
        potential_reduction = (total_redundant_lines / total_lines) * 100 if total_lines > 0 else 0
        
        report.append(f"- **Estimated Code Reduction**: {potential_reduction:.1f}% ({total_redundant_lines:,} lines)")
        report.append("")
        
        # Technical Debt Analysis
        report.append("## TECHNICAL DEBT ANALYSIS")
        report.append("")
        
        for debt in sorted(self.technical_debts, key=lambda d: ['Critical', 'High', 'Medium', 'Low'].index(d.severity)):
            severity_emoji = {'Critical': '🚨', 'High': '⚠️', 'Medium': '⚡', 'Low': 'ℹ️'}[debt.severity]
            report.append(f"### {severity_emoji} {debt.severity}: {debt.description}")
            report.append(f"**Category**: {debt.category}")
            report.append(f"**Total Lines**: {debt.total_lines:,}")
            report.append(f"**Impact**: {debt.impact}")
            report.append(f"**Recommendation**: {debt.recommendation}")
            report.append(f"**Effort Estimate**: {debt.effort_estimate}")
            report.append("")
            report.append("**Affected Files**:")
            for file in debt.affected_files[:10]:  # Limit to top 10
                report.append(f"  - {file}")
            if len(debt.affected_files) > 10:
                report.append(f"  - ... and {len(debt.affected_files) - 10} more files")
            report.append("")
        
        # Redundancy Patterns
        report.append("## REDUNDANCY PATTERNS")
        report.append("")
        
        for pattern in sorted(self.redundancy_patterns, key=lambda p: p.total_lines, reverse=True):
            report.append(f"### {pattern.pattern_name}")
            report.append(f"**Total Lines**: {pattern.total_lines:,}")
            report.append(f"**Redundancy Score**: {pattern.redundancy_score:.1%}")
            report.append(f"**Files Affected**: {len(pattern.files)}")
            report.append(f"**Consolidation Target**: {pattern.consolidation_target}")
            report.append(f"**Effort Estimate**: {pattern.effort_estimate}")
            report.append("")
        
        # Consolidation Roadmap
        report.append("## CONSOLIDATION ROADMAP")
        report.append("")
        
        report.append("### Phase 1: Critical Consolidation (Weeks 1-8)")
        for debt in [d for d in self.technical_debts if d.severity == "Critical"]:
            report.append(f"- **{debt.category}**: {debt.recommendation} ({debt.effort_estimate})")
        
        report.append("")
        report.append("### Phase 2: High Priority Consolidation (Weeks 9-16)")
        for debt in [d for d in self.technical_debts if d.severity == "High"]:
            report.append(f"- **{debt.category}**: {debt.recommendation} ({debt.effort_estimate})")
        
        report.append("")
        report.append("### Phase 3: Medium Priority Optimization (Weeks 17-24)")
        for debt in [d for d in self.technical_debts if d.severity == "Medium"]:
            report.append(f"- **{debt.category}**: {debt.recommendation} ({debt.effort_estimate})")
        
        # Implementation Priority
        report.append("")
        report.append("## IMPLEMENTATION PRIORITY")
        report.append("")
        
        report.append("### Immediate Action Required (Next Sprint)")
        critical_patterns = [p for p in self.redundancy_patterns if p.redundancy_score > 0.7]
        for pattern in critical_patterns[:3]:
            report.append(f"1. **{pattern.pattern_name}**: {pattern.total_lines:,} lines → {pattern.consolidation_target}")
        
        report.append("")
        report.append("### Quick Wins (2-4 weeks)")
        medium_patterns = [p for p in self.redundancy_patterns if 0.4 <= p.redundancy_score <= 0.7]
        for pattern in medium_patterns[:5]:
            report.append(f"- {pattern.pattern_name}: {len(pattern.files)} files → 1 unified file")
        
        # Business Impact
        report.append("")
        report.append("## BUSINESS IMPACT")
        report.append("")
        report.append("### Development Velocity Impact")
        report.append(f"- **Current Maintenance Overhead**: {len(self.file_analyses)} files to maintain")
        report.append(f"- **Testing Complexity**: {len(critical_debt + high_debt)} high-impact debt items")
        report.append(f"- **Onboarding Difficulty**: {total_lines:,} lines across {total_files} files")
        report.append("")
        
        report.append("### Post-Consolidation Benefits")
        estimated_final_files = total_files - sum(len(p.files) - 1 for p in self.redundancy_patterns)
        estimated_final_lines = total_lines - int(total_redundant_lines * 0.8)  # Conservative estimate
        
        report.append(f"- **Projected Files**: {estimated_final_files} ({((total_files - estimated_final_files)/total_files*100):.1f}% reduction)")
        report.append(f"- **Projected Lines**: {estimated_final_lines:,} ({((total_lines - estimated_final_lines)/total_lines*100):.1f}% reduction)")
        report.append(f"- **Maintenance Reduction**: ~{len(critical_debt + high_debt) * 2}x easier maintenance")
        report.append(f"- **Testing Simplification**: ~{len(self.redundancy_patterns) * 3}x fewer test scenarios")
        
        return "\n".join(report)
    
    def analyze(self) -> str:
        """Run comprehensive analysis."""
        print("Scanning directories...")
        self.scan_directory(self.app_path)
        
        print("Identifying redundancy patterns...")
        self.identify_orchestrator_redundancy()
        self.identify_manager_redundancy()
        self.identify_communication_redundancy()
        
        print("Analyzing technical debt...")
        self.identify_technical_debts()
        
        print("Generating report...")
        return self.generate_comprehensive_report()

if __name__ == "__main__":
    analyzer = TechnicalDebtAnalyzer(
        app_path="/Users/bogdan/work/leanvibe-dev/bee-hive/app",
        core_path="/Users/bogdan/work/leanvibe-dev/bee-hive/app/core"
    )
    report = analyzer.analyze()
    print(report)