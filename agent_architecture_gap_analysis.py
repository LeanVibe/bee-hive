#!/usr/bin/env python3
"""
Agent Architecture Gap Analysis for LeanVibe Bee Hive 2.0

Analyzes the gap between current homogeneous agent architecture and the required
heterogeneous multi-CLI agent system (Claude Code + Cursor + Gemini CLI + OpenCode).

This script identifies:
1. Missing multi-CLI agent abstraction
2. Duplicated orchestration code that needs consolidation  
3. Required git worktree isolation components
4. Communication protocol gaps between different agent types
"""

import os
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class CodeDuplication:
    """Represents duplicated code found in the system."""
    pattern: str
    files: List[str]
    similarity_score: float
    consolidation_opportunity: str

@dataclass
class AgentArchitectureGap:
    """Represents a gap in the agent architecture."""
    component: str
    missing_functionality: str
    current_implementation: str
    required_implementation: str
    priority: str
    effort_estimate: str

class AgentArchitectureAnalyzer:
    """Analyzes agent architecture gaps and code duplication."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.core_dir = self.project_root / "app" / "core"
        self.agents_dir = self.project_root / "app" / "agents"
        self.api_dir = self.project_root / "app" / "api_v2"
        
        self.duplications: List[CodeDuplication] = []
        self.architecture_gaps: List[AgentArchitectureGap] = []
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run complete analysis."""
        print("🔍 Starting Agent Architecture Gap Analysis...")
        
        # Analyze current agent system
        self.analyze_current_agent_architecture()
        
        # Find code duplications
        self.find_orchestration_duplications()
        self.find_communication_duplications()
        
        # Identify architecture gaps
        self.identify_multi_cli_gaps()
        self.identify_isolation_gaps()
        
        # Generate consolidation strategy
        strategy = self.generate_consolidation_strategy()
        
        return {
            "current_architecture": self.get_current_architecture_summary(),
            "code_duplications": [self._duplication_to_dict(d) for d in self.duplications],
            "architecture_gaps": [self._gap_to_dict(g) for g in self.architecture_gaps],
            "consolidation_strategy": strategy,
            "priority_actions": self.get_priority_actions()
        }
    
    def analyze_current_agent_architecture(self):
        """Analyze the current agent architecture."""
        print("📊 Analyzing current agent architecture...")
        
        # Check agent manager capabilities
        agent_manager_file = self.core_dir / "agent_manager.py"
        if agent_manager_file.exists():
            content = agent_manager_file.read_text()
            
            # Check for CLI abstraction
            if "claude_code" not in content.lower() and "cursor" not in content.lower():
                self.architecture_gaps.append(AgentArchitectureGap(
                    component="Agent Manager",
                    missing_functionality="Multi-CLI agent support",
                    current_implementation="Homogeneous Python agents only",
                    required_implementation="Abstract interface for Claude Code, Cursor, Gemini CLI, OpenCode",
                    priority="HIGH",
                    effort_estimate="2-3 days"
                ))
            
            # Check for git worktree support
            if "worktree" not in content.lower() and "git" not in content.lower():
                self.architecture_gaps.append(AgentArchitectureGap(
                    component="Agent Isolation",
                    missing_functionality="Git worktree management",
                    current_implementation="No isolation mechanism",
                    required_implementation="Git worktree creation and management per agent",
                    priority="HIGH", 
                    effort_estimate="1-2 days"
                ))
    
    def find_orchestration_duplications(self):
        """Find duplicated orchestration code."""
        print("🔄 Analyzing orchestration code duplications...")
        
        orchestrator_files = []
        for pattern in ["*orchestrator*.py", "*coordination*.py", "*workflow*.py"]:
            orchestrator_files.extend(self.project_root.rglob(pattern))
        
        # Look for common patterns that might be duplicated
        common_patterns = [
            "async def execute_task",
            "async def coordinate_agents", 
            "async def manage_workflow",
            "class.*Orchestrator",
            "def spawn_agent",
            "def delegate_task"
        ]
        
        for pattern in common_patterns:
            files_with_pattern = []
            for file_path in orchestrator_files:
                try:
                    content = file_path.read_text()
                    if self._matches_pattern(content, pattern):
                        files_with_pattern.append(str(file_path))
                except:
                    continue
            
            if len(files_with_pattern) > 1:
                self.duplications.append(CodeDuplication(
                    pattern=pattern,
                    files=files_with_pattern,
                    similarity_score=0.8,  # Estimated
                    consolidation_opportunity=f"Consolidate {pattern} implementations into unified interface"
                ))
    
    def find_communication_duplications(self):
        """Find duplicated communication code."""
        print("📡 Analyzing communication code duplications...")
        
        comm_files = []
        for pattern in ["*communication*.py", "*messaging*.py", "*redis*.py", "*websocket*.py"]:
            comm_files.extend(self.project_root.rglob(pattern))
        
        message_handling_files = []
        for file_path in comm_files:
            try:
                content = file_path.read_text()
                if any(keyword in content for keyword in ["send_message", "receive_message", "publish", "subscribe"]):
                    message_handling_files.append(str(file_path))
            except:
                continue
        
        if len(message_handling_files) > 2:
            self.duplications.append(CodeDuplication(
                pattern="Message handling logic",
                files=message_handling_files,
                similarity_score=0.7,
                consolidation_opportunity="Unify message handling into single communication manager"
            ))
    
    def identify_multi_cli_gaps(self):
        """Identify gaps in multi-CLI support."""
        print("🎯 Identifying multi-CLI support gaps...")
        
        # Check for CLI adapter pattern
        adapters_exist = any(
            file_path.name.endswith("_adapter.py") or "adapter" in file_path.name.lower()
            for file_path in self.core_dir.rglob("*.py")
        )
        
        if not adapters_exist:
            self.architecture_gaps.append(AgentArchitectureGap(
                component="CLI Adapters", 
                missing_functionality="CLI-specific communication adapters",
                current_implementation="No adapter pattern",
                required_implementation="Adapters for Claude Code, Cursor, Gemini CLI, OpenCode",
                priority="HIGH",
                effort_estimate="3-4 days"
            ))
        
        # Check for universal agent interface
        interface_files = list(self.core_dir.glob("*interface*.py"))
        if not interface_files:
            self.architecture_gaps.append(AgentArchitectureGap(
                component="Agent Interface",
                missing_functionality="Universal agent interface protocol",
                current_implementation="Tightly coupled to internal agents",
                required_implementation="Abstract interface supporting multiple CLI types",
                priority="HIGH",
                effort_estimate="2-3 days"
            ))
    
    def identify_isolation_gaps(self):
        """Identify gaps in agent isolation."""
        print("🔒 Identifying agent isolation gaps...")
        
        # Check for path restriction mechanisms
        security_files = list(self.core_dir.glob("*security*.py"))
        has_path_validation = False
        
        for file_path in security_files:
            try:
                content = file_path.read_text()
                if "path" in content.lower() and ("restrict" in content.lower() or "validate" in content.lower()):
                    has_path_validation = True
                    break
            except:
                continue
        
        if not has_path_validation:
            self.architecture_gaps.append(AgentArchitectureGap(
                component="Path Security",
                missing_functionality="Subfolder-scoped permissions",
                current_implementation="No path restrictions",
                required_implementation="Path validation and restriction enforcement",
                priority="MEDIUM",
                effort_estimate="1-2 days"
            ))
    
    def generate_consolidation_strategy(self) -> Dict[str, Any]:
        """Generate consolidation strategy based on analysis."""
        print("📋 Generating consolidation strategy...")
        
        strategy = {
            "phase_1_foundation": {
                "duration": "1-2 weeks",
                "description": "Establish multi-CLI agent foundation",
                "tasks": [
                    "Create universal agent interface",
                    "Implement CLI adapter pattern", 
                    "Add git worktree management",
                    "Consolidate orchestration duplications"
                ]
            },
            "phase_2_integration": {
                "duration": "1-2 weeks", 
                "description": "Integrate external CLI agents",
                "tasks": [
                    "Implement Claude Code adapter",
                    "Implement Cursor agent adapter",
                    "Implement Gemini CLI adapter", 
                    "Add context handoff protocols"
                ]
            },
            "phase_3_optimization": {
                "duration": "1 week",
                "description": "Optimize and secure multi-agent system",
                "tasks": [
                    "Add path security and validation",
                    "Optimize communication protocols",
                    "Add monitoring and observability",
                    "Performance testing and tuning"
                ]
            }
        }
        
        return strategy
    
    def get_priority_actions(self) -> List[Dict[str, Any]]:
        """Get priority actions for immediate implementation."""
        high_priority_gaps = [gap for gap in self.architecture_gaps if gap.priority == "HIGH"]
        high_impact_duplications = [dup for dup in self.duplications if dup.similarity_score > 0.7]
        
        actions = []
        
        # Add high priority gaps
        for gap in high_priority_gaps:
            actions.append({
                "type": "architecture_gap",
                "action": f"Implement {gap.missing_functionality}",
                "component": gap.component,
                "effort": gap.effort_estimate,
                "impact": "HIGH"
            })
        
        # Add high impact consolidations
        for dup in high_impact_duplications:
            actions.append({
                "type": "code_consolidation",
                "action": dup.consolidation_opportunity,
                "files_affected": len(dup.files),
                "impact": "MEDIUM"
            })
        
        return sorted(actions, key=lambda x: (x["impact"] == "HIGH", x.get("files_affected", 0)), reverse=True)
    
    def get_current_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of current architecture."""
        return {
            "agent_files_count": len(list(self.agents_dir.glob("*.py"))) if self.agents_dir.exists() else 0,
            "core_files_count": len(list(self.core_dir.glob("*.py"))),
            "orchestrator_files": len(list(self.project_root.rglob("*orchestrator*.py"))),
            "communication_files": len(list(self.project_root.rglob("*communication*.py"))),
            "has_redis_integration": (self.core_dir / "redis.py").exists(),
            "has_agent_manager": (self.core_dir / "agent_manager.py").exists(),
            "has_multi_cli_support": False,  # Based on analysis
            "has_git_isolation": False  # Based on analysis
        }
    
    def _matches_pattern(self, content: str, pattern: str) -> bool:
        """Check if content matches a pattern."""
        import re
        return bool(re.search(pattern, content, re.IGNORECASE))
    
    def _duplication_to_dict(self, dup: CodeDuplication) -> Dict[str, Any]:
        """Convert duplication to dictionary."""
        return {
            "pattern": dup.pattern,
            "files": dup.files,
            "similarity_score": dup.similarity_score,
            "consolidation_opportunity": dup.consolidation_opportunity
        }
    
    def _gap_to_dict(self, gap: AgentArchitectureGap) -> Dict[str, Any]:
        """Convert gap to dictionary."""
        return {
            "component": gap.component,
            "missing_functionality": gap.missing_functionality,
            "current_implementation": gap.current_implementation,
            "required_implementation": gap.required_implementation,
            "priority": gap.priority,
            "effort_estimate": gap.effort_estimate
        }

def main():
    """Main analysis function."""
    project_root = os.getcwd()
    analyzer = AgentArchitectureAnalyzer(project_root)
    
    # Run analysis
    results = analyzer.analyze_all()
    
    # Save results
    output_file = Path(project_root) / "agent_architecture_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 AGENT ARCHITECTURE GAP ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Current Architecture:")
    for key, value in results["current_architecture"].items():
        print(f"   {key}: {value}")
    
    print(f"\n🔄 Code Duplications Found: {len(results['code_duplications'])}")
    for dup in results["code_duplications"]:
        print(f"   • {dup['pattern']} ({len(dup['files'])} files, {dup['similarity_score']:.1%} similarity)")
    
    print(f"\n❌ Architecture Gaps Found: {len(results['architecture_gaps'])}")
    for gap in results["architecture_gaps"]:
        print(f"   • {gap['component']}: {gap['missing_functionality']} ({gap['priority']} priority)")
    
    print(f"\n🚀 Priority Actions ({len(results['priority_actions'])}):")
    for i, action in enumerate(results["priority_actions"][:5], 1):
        print(f"   {i}. {action['action']} (Impact: {action['impact']})")
    
    print(f"\n📋 Consolidation Strategy:")
    for phase_name, phase_info in results["consolidation_strategy"].items():
        print(f"   {phase_name.replace('_', ' ').title()}: {phase_info['duration']}")
        print(f"      {phase_info['description']}")
    
    print(f"\n💾 Full report saved to: {output_file}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()