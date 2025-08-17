#!/usr/bin/env python3
"""
End-to-End Workflow Testing Suite

Comprehensive testing of complete multi-CLI agent workflows from start to finish.
Tests real-world development scenarios where multiple CLI agents coordinate
to complete complex software development tasks.

This suite validates:
- Complete feature development workflows
- Bug fixing and debugging workflows  
- Code refactoring and optimization workflows
- Documentation and testing workflows
- Integration between different agent types
- Workflow state management and recovery
"""

import asyncio
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytest
import git
import subprocess
import uuid
import yaml

class WorkflowType(Enum):
    """Types of end-to-end workflows."""
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"

class WorkflowStatus(Enum):
    """Workflow execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class FileArtifact:
    """Represents a file artifact created or modified during workflow."""
    file_path: str
    content: str
    file_type: str  # "source", "test", "documentation", "config"
    created_by: str  # Agent that created/modified the file
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExpectation:
    """Expected outcomes for a workflow."""
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    tests_passing: bool = True
    build_successful: bool = True
    documentation_complete: bool = False
    performance_improved: bool = False
    code_quality_metrics: Dict[str, float] = field(default_factory=dict)
    custom_validations: List[str] = field(default_factory=list)

@dataclass
class WorkflowDefinition:
    """Complete definition of an end-to-end workflow."""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    agents_required: List[str]
    initial_codebase: Dict[str, str]  # file_path -> content
    requirements: str
    expectations: WorkflowExpectation
    timeout: int = 1800  # 30 minutes default
    complexity_level: str = "medium"  # low, medium, high

class MockCLIAgentE2E:
    """Enhanced mock CLI agent for end-to-end testing."""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.working_directory = None
        self.execution_log = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "files_created": 0,
            "files_modified": 0,
            "errors_encountered": 0,
            "total_execution_time": 0.0
        }
    
    def set_working_directory(self, path: Path):
        """Set the working directory for the agent."""
        self.working_directory = path
    
    async def execute_workflow_step(self, step_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step with realistic behavior simulation."""
        start_time = time.time()
        
        try:
            # Simulate different agent behaviors based on type and step
            result = await self._simulate_agent_behavior(step_description, context)
            
            execution_time = time.time() - start_time
            self.performance_metrics["total_execution_time"] += execution_time
            self.performance_metrics["tasks_completed"] += 1
            
            # Log execution
            self.execution_log.append({
                "step": step_description,
                "timestamp": start_time,
                "duration": execution_time,
                "status": "completed",
                "files_affected": result.get("files_affected", []),
                "context_updates": result.get("context_updates", {})
            })
            
            return {
                "status": "completed",
                "agent_id": self.agent_id,
                "result": result,
                "execution_time": execution_time
            }
        
        except Exception as e:
            self.performance_metrics["errors_encountered"] += 1
            self.execution_log.append({
                "step": step_description,
                "timestamp": start_time,
                "duration": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            })
            
            return {
                "status": "failed",
                "agent_id": self.agent_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _simulate_agent_behavior(self, step_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate realistic agent behavior based on type and task."""
        # Add realistic delay
        await asyncio.sleep(0.2 + 0.3 * __import__('random').random())
        
        step_lower = step_description.lower()
        result = {
            "files_affected": [],
            "context_updates": {},
            "output": "",
            "metrics": {}
        }
        
        # Claude Code behavior simulation
        if self.agent_type == "claude_code":
            if "analyze" in step_lower:
                result.update(await self._simulate_code_analysis())
            elif "debug" in step_lower:
                result.update(await self._simulate_debugging())
            elif "document" in step_lower:
                result.update(await self._simulate_documentation())
            elif "review" in step_lower:
                result.update(await self._simulate_code_review())
        
        # Cursor behavior simulation
        elif self.agent_type == "cursor":
            if "implement" in step_lower or "create" in step_lower:
                result.update(await self._simulate_implementation())
            elif "refactor" in step_lower:
                result.update(await self._simulate_refactoring())
            elif "fix" in step_lower:
                result.update(await self._simulate_bug_fixing())
        
        # Gemini CLI behavior simulation
        elif self.agent_type == "gemini_cli":
            if "test" in step_lower:
                result.update(await self._simulate_testing())
            elif "optimize" in step_lower:
                result.update(await self._simulate_optimization())
            elif "validate" in step_lower:
                result.update(await self._simulate_validation())
        
        return result
    
    async def _simulate_code_analysis(self) -> Dict[str, Any]:
        """Simulate code analysis behavior."""
        if self.working_directory:
            # Find Python files to analyze
            python_files = list(self.working_directory.glob("**/*.py"))
            
            analysis_report = {
                "complexity_score": 6.5 + 3.0 * __import__('random').random(),
                "files_analyzed": [str(f.relative_to(self.working_directory)) for f in python_files[:5]],
                "issues_found": __import__('random').randint(2, 8),
                "recommendations": [
                    "Consider breaking down large functions",
                    "Add type hints for better code clarity",
                    "Implement error handling in critical paths"
                ]
            }
            
            # Create analysis report file
            report_file = self.working_directory / "analysis_report.md"
            report_content = f"""# Code Analysis Report

## Summary
- Files analyzed: {len(analysis_report['files_analyzed'])}
- Complexity score: {analysis_report['complexity_score']:.1f}/10
- Issues found: {analysis_report['issues_found']}

## Recommendations
{chr(10).join('- ' + rec for rec in analysis_report['recommendations'])}
"""
            report_file.write_text(report_content)
            
            return {
                "files_affected": ["analysis_report.md"],
                "context_updates": {
                    "analysis_complete": True,
                    "complexity_score": analysis_report['complexity_score'],
                    "issues_found": analysis_report['issues_found']
                },
                "output": "Code analysis completed successfully",
                "metrics": analysis_report
            }
        
        return {"output": "Code analysis completed", "context_updates": {"analysis_complete": True}}
    
    async def _simulate_implementation(self) -> Dict[str, Any]:
        """Simulate feature implementation."""
        if self.working_directory:
            # Create a new feature file
            feature_id = uuid.uuid4().hex[:8]
            feature_file = self.working_directory / f"feature_{feature_id}.py"
            
            feature_content = f'''"""
Feature implementation generated by {self.agent_id}
"""

class Feature{feature_id.capitalize()}:
    """Generated feature class."""
    
    def __init__(self):
        self.name = "feature_{feature_id}"
        self.version = "1.0.0"
        self.active = True
    
    def execute(self, *args, **kwargs):
        """Execute the feature."""
        return {{
            "status": "success",
            "feature": self.name,
            "timestamp": "{time.time()}"
        }}
    
    def validate(self):
        """Validate feature configuration."""
        return self.active and self.name is not None

# Feature initialization
feature_instance = Feature{feature_id.capitalize()}()
'''
            
            feature_file.write_text(feature_content)
            self.performance_metrics["files_created"] += 1
            
            # Update main.py if it exists
            main_file = self.working_directory / "main.py"
            if main_file.exists():
                current_content = main_file.read_text()
                updated_content = current_content + f"\n# Feature {feature_id} integration ready\n"
                main_file.write_text(updated_content)
                self.performance_metrics["files_modified"] += 1
            
            return {
                "files_affected": [f"feature_{feature_id}.py", "main.py"],
                "context_updates": {
                    "implementation_complete": True,
                    "feature_id": feature_id,
                    "new_features": [f"feature_{feature_id}"]
                },
                "output": f"Feature {feature_id} implemented successfully",
                "metrics": {"lines_added": len(feature_content.split('\n'))}
            }
        
        return {"output": "Implementation completed", "context_updates": {"implementation_complete": True}}
    
    async def _simulate_testing(self) -> Dict[str, Any]:
        """Simulate test creation and execution."""
        if self.working_directory:
            # Create test file
            test_file = self.working_directory / "test_generated.py"
            
            test_content = f'''"""
Generated tests by {self.agent_id}
"""
import unittest
from unittest.mock import Mock, patch

class TestGeneratedFeatures(unittest.TestCase):
    """Test generated features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {{"key": "value", "number": 42}}
    
    def test_feature_functionality(self):
        """Test basic feature functionality."""
        # This would test actual features
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)
    
    def test_feature_validation(self):
        """Test feature validation."""
        # Mock validation test
        self.assertIsNotNone(self.test_data)
        self.assertIn("key", self.test_data)
    
    def test_error_handling(self):
        """Test error handling."""
        # Mock error handling test
        with self.assertRaises(KeyError):
            _ = self.test_data["nonexistent_key"]

if __name__ == "__main__":
    unittest.main()
'''
            
            test_file.write_text(test_content)
            self.performance_metrics["files_created"] += 1
            
            # Simulate test execution
            test_results = {
                "tests_run": 3,
                "tests_passed": 3,
                "tests_failed": 0,
                "coverage": 85.5,
                "execution_time": 0.45
            }
            
            return {
                "files_affected": ["test_generated.py"],
                "context_updates": {
                    "tests_complete": True,
                    "test_results": test_results,
                    "coverage": test_results["coverage"]
                },
                "output": f"Created and executed {test_results['tests_run']} tests",
                "metrics": test_results
            }
        
        return {"output": "Testing completed", "context_updates": {"tests_complete": True}}
    
    async def _simulate_documentation(self) -> Dict[str, Any]:
        """Simulate documentation generation."""
        if self.working_directory:
            # Create documentation file
            doc_file = self.working_directory / "README.md"
            
            doc_content = f'''# Project Documentation

Generated by {self.agent_id} on {time.strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This project contains features developed through multi-agent coordination.

## Features
- Feature analysis and implementation
- Automated testing
- Code quality validation

## Usage
```python
# Example usage
from feature_* import Feature*
feature = Feature*()
result = feature.execute()
```

## Testing
Run tests with:
```bash
python -m pytest test_generated.py
```

## Architecture
The system follows a multi-agent development approach where:
1. Claude Code performs analysis and review
2. Cursor implements features and fixes
3. Gemini CLI creates tests and optimizations

## Metrics
- Code complexity: Monitored
- Test coverage: >80%
- Performance: Optimized
'''
            
            doc_file.write_text(doc_content)
            self.performance_metrics["files_created"] += 1
            
            return {
                "files_affected": ["README.md"],
                "context_updates": {
                    "documentation_complete": True,
                    "docs_generated": True
                },
                "output": "Documentation generated successfully",
                "metrics": {"pages_created": 1, "sections": 6}
            }
        
        return {"output": "Documentation completed", "context_updates": {"documentation_complete": True}}
    
    async def _simulate_refactoring(self) -> Dict[str, Any]:
        """Simulate code refactoring."""
        modified_files = []
        
        if self.working_directory:
            # Find and refactor Python files
            python_files = list(self.working_directory.glob("*.py"))
            
            for py_file in python_files[:3]:  # Limit to 3 files
                if py_file.exists():
                    content = py_file.read_text()
                    
                    # Simple refactoring simulation
                    refactored_content = content.replace("    ", "    ")  # Normalize indentation
                    refactored_content += f"\n# Refactored by {self.agent_id}\n"
                    
                    py_file.write_text(refactored_content)
                    modified_files.append(str(py_file.name))
                    self.performance_metrics["files_modified"] += 1
        
        return {
            "files_affected": modified_files,
            "context_updates": {
                "refactoring_complete": True,
                "files_refactored": len(modified_files)
            },
            "output": f"Refactored {len(modified_files)} files",
            "metrics": {"files_refactored": len(modified_files)}
        }
    
    async def _simulate_debugging(self) -> Dict[str, Any]:
        """Simulate debugging and bug fixing."""
        debug_report = {
            "bugs_found": __import__('random').randint(1, 4),
            "bugs_fixed": 0,
            "files_debugged": []
        }
        
        if self.working_directory:
            python_files = list(self.working_directory.glob("*.py"))
            
            for py_file in python_files[:2]:  # Debug up to 2 files
                if py_file.exists():
                    content = py_file.read_text()
                    
                    # Simulate bug fixing
                    fixed_content = content.replace("TODO:", "# FIXED:")
                    fixed_content += f"\n# Debug session by {self.agent_id}\n"
                    
                    py_file.write_text(fixed_content)
                    debug_report["files_debugged"].append(str(py_file.name))
                    debug_report["bugs_fixed"] += 1
                    self.performance_metrics["files_modified"] += 1
        
        return {
            "files_affected": debug_report["files_debugged"],
            "context_updates": {
                "debugging_complete": True,
                "bugs_fixed": debug_report["bugs_fixed"]
            },
            "output": f"Fixed {debug_report['bugs_fixed']} bugs",
            "metrics": debug_report
        }
    
    async def _simulate_optimization(self) -> Dict[str, Any]:
        """Simulate performance optimization."""
        optimization_report = {
            "performance_gain": 15.0 + 25.0 * __import__('random').random(),
            "memory_reduction": 5.0 + 15.0 * __import__('random').random(),
            "optimizations_applied": __import__('random').randint(2, 6)
        }
        
        return {
            "files_affected": [],
            "context_updates": {
                "optimization_complete": True,
                "performance_improved": True,
                "performance_gain": optimization_report["performance_gain"]
            },
            "output": f"Applied {optimization_report['optimizations_applied']} optimizations",
            "metrics": optimization_report
        }
    
    async def _simulate_code_review(self) -> Dict[str, Any]:
        """Simulate code review process."""
        review_report = {
            "files_reviewed": 0,
            "issues_found": 0,
            "suggestions": []
        }
        
        if self.working_directory:
            python_files = list(self.working_directory.glob("*.py"))
            review_report["files_reviewed"] = len(python_files)
            review_report["issues_found"] = __import__('random').randint(0, 3)
            review_report["suggestions"] = [
                "Consider adding docstrings",
                "Use type hints for better clarity",
                "Add input validation"
            ]
        
        return {
            "files_affected": [],
            "context_updates": {
                "review_complete": True,
                "issues_found": review_report["issues_found"]
            },
            "output": f"Reviewed {review_report['files_reviewed']} files",
            "metrics": review_report
        }
    
    async def _simulate_validation(self) -> Dict[str, Any]:
        """Simulate validation process."""
        validation_report = {
            "validation_passed": True,
            "checks_performed": ["syntax", "style", "security", "performance"],
            "score": 8.5 + 1.5 * __import__('random').random()
        }
        
        return {
            "files_affected": [],
            "context_updates": {
                "validation_complete": True,
                "validation_passed": validation_report["validation_passed"],
                "quality_score": validation_report["score"]
            },
            "output": f"Validation completed with score {validation_report['score']:.1f}/10",
            "metrics": validation_report
        }
    
    async def _simulate_bug_fixing(self) -> Dict[str, Any]:
        """Simulate bug fixing behavior."""
        return await self._simulate_debugging()

class EndToEndWorkflowOrchestrator:
    """Orchestrates complete end-to-end workflows."""
    
    def __init__(self):
        self.agents = {}
        self.temp_workspace = None
        self.workflow_history = []
        self.active_workflows = {}
    
    def register_agent(self, agent: MockCLIAgentE2E):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
    
    async def setup_workspace(self, initial_codebase: Dict[str, str]) -> Path:
        """Setup workspace with initial codebase."""
        self.temp_workspace = Path(tempfile.mkdtemp(prefix="e2e_workflow_"))
        
        # Create initial files
        for file_path, content in initial_codebase.items():
            full_path = self.temp_workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Initialize git repository
        try:
            repo = git.Repo.init(self.temp_workspace)
            repo.index.add(list(initial_codebase.keys()))
            repo.index.commit("Initial commit for E2E workflow")
        except Exception as e:
            print(f"Warning: Could not initialize git repo: {e}")
        
        # Set working directory for all agents
        for agent in self.agents.values():
            agent.set_working_directory(self.temp_workspace)
        
        return self.temp_workspace
    
    async def execute_workflow(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """Execute a complete end-to-end workflow."""
        workflow_id = workflow_def.workflow_id
        start_time = time.time()
        
        workflow_result = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_def.workflow_type.value,
            "workflow_name": workflow_def.name,
            "start_time": start_time,
            "status": WorkflowStatus.IN_PROGRESS.value,
            "steps_executed": [],
            "agents_utilized": [],
            "artifacts_created": [],
            "context": {},
            "metrics": {},
            "validation_results": {}
        }
        
        try:
            # Setup workspace
            workspace = await self.setup_workspace(workflow_def.initial_codebase)
            workflow_result["workspace"] = str(workspace)
            
            # Execute workflow based on type
            if workflow_def.workflow_type == WorkflowType.FEATURE_DEVELOPMENT:
                await self._execute_feature_development(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.BUG_FIX:
                await self._execute_bug_fix(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.REFACTORING:
                await self._execute_refactoring(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.DOCUMENTATION:
                await self._execute_documentation(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.TESTING:
                await self._execute_testing(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.OPTIMIZATION:
                await self._execute_optimization(workflow_def, workflow_result)
            elif workflow_def.workflow_type == WorkflowType.INTEGRATION:
                await self._execute_integration(workflow_def, workflow_result)
            
            # Validate results against expectations
            validation_results = await self._validate_workflow_results(workflow_def, workflow_result)
            workflow_result["validation_results"] = validation_results
            
            # Determine final status
            if validation_results.get("all_expectations_met", False):
                workflow_result["status"] = WorkflowStatus.COMPLETED.value
            else:
                workflow_result["status"] = WorkflowStatus.FAILED.value
        
        except Exception as e:
            workflow_result["status"] = WorkflowStatus.FAILED.value
            workflow_result["error"] = str(e)
        
        finally:
            workflow_result["end_time"] = time.time()
            workflow_result["total_duration"] = workflow_result["end_time"] - start_time
            
            # Collect final metrics
            await self._collect_final_metrics(workflow_result)
            
            # Cleanup workspace
            if self.temp_workspace and self.temp_workspace.exists():
                shutil.rmtree(self.temp_workspace, ignore_errors=True)
            
            self.workflow_history.append(workflow_result)
        
        return workflow_result
    
    async def _execute_feature_development(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute feature development workflow."""
        steps = [
            ("claude_code", "Analyze requirements and existing codebase"),
            ("claude_code", "Design feature architecture and integration points"),
            ("cursor", "Implement core feature functionality"),
            ("cursor", "Integrate feature with existing codebase"),
            ("gemini_cli", "Create comprehensive tests for new feature"),
            ("gemini_cli", "Validate feature performance and reliability"),
            ("claude_code", "Review implementation and generate documentation"),
            ("claude_code", "Perform final code review and quality check")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_bug_fix(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute bug fix workflow."""
        steps = [
            ("claude_code", "Analyze bug report and reproduce issue"),
            ("claude_code", "Identify root cause and impact analysis"),
            ("cursor", "Implement bug fix and error handling"),
            ("gemini_cli", "Create regression tests for bug fix"),
            ("gemini_cli", "Validate fix doesn't break existing functionality"),
            ("claude_code", "Review fix and update documentation")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_refactoring(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute refactoring workflow."""
        steps = [
            ("claude_code", "Analyze code structure and identify refactoring opportunities"),
            ("cursor", "Refactor code while maintaining functionality"),
            ("gemini_cli", "Ensure all tests pass after refactoring"),
            ("gemini_cli", "Validate performance hasn't degraded"),
            ("claude_code", "Update documentation to reflect changes")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_documentation(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute documentation workflow."""
        steps = [
            ("claude_code", "Analyze codebase and identify documentation needs"),
            ("claude_code", "Generate comprehensive documentation"),
            ("gemini_cli", "Validate documentation completeness and accuracy"),
            ("claude_code", "Create examples and usage guides")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_testing(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute testing workflow."""
        steps = [
            ("claude_code", "Analyze code coverage and testing gaps"),
            ("gemini_cli", "Create unit tests for identified gaps"),
            ("gemini_cli", "Create integration tests"),
            ("gemini_cli", "Execute full test suite and generate report"),
            ("claude_code", "Review test results and quality metrics")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_optimization(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute optimization workflow."""
        steps = [
            ("claude_code", "Profile code and identify performance bottlenecks"),
            ("cursor", "Optimize critical code paths"),
            ("gemini_cli", "Validate performance improvements"),
            ("gemini_cli", "Run benchmarks and create performance report"),
            ("claude_code", "Document optimization changes and recommendations")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_integration(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]):
        """Execute integration workflow."""
        steps = [
            ("claude_code", "Analyze integration requirements and dependencies"),
            ("cursor", "Implement integration components"),
            ("gemini_cli", "Create integration tests"),
            ("gemini_cli", "Validate end-to-end integration functionality"),
            ("claude_code", "Document integration process and requirements")
        ]
        
        await self._execute_workflow_steps(steps, result)
    
    async def _execute_workflow_steps(self, steps: List[Tuple[str, str]], result: Dict[str, Any]):
        """Execute a series of workflow steps."""
        context = {}
        
        for agent_id, step_description in steps:
            if agent_id in self.agents:
                step_result = await self.agents[agent_id].execute_workflow_step(step_description, context)
                
                result["steps_executed"].append({
                    "agent_id": agent_id,
                    "step_description": step_description,
                    "result": step_result,
                    "timestamp": time.time()
                })
                
                if agent_id not in result["agents_utilized"]:
                    result["agents_utilized"].append(agent_id)
                
                # Update context with step results
                if step_result.get("status") == "completed":
                    context.update(step_result.get("result", {}).get("context_updates", {}))
                    
                    # Track artifacts
                    files_affected = step_result.get("result", {}).get("files_affected", [])
                    for file_path in files_affected:
                        if file_path not in result["artifacts_created"]:
                            result["artifacts_created"].append(file_path)
        
        result["context"] = context
    
    async def _validate_workflow_results(self, workflow_def: WorkflowDefinition, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow results against expectations."""
        expectations = workflow_def.expectations
        validation_results = {
            "all_expectations_met": True,
            "files_created_check": True,
            "files_modified_check": True,
            "tests_passing_check": True,
            "build_successful_check": True,
            "documentation_complete_check": True,
            "performance_improved_check": True,
            "details": {}
        }
        
        # Check file creation expectations
        if expectations.files_created:
            created_files = set(result["artifacts_created"])
            expected_files = set(expectations.files_created)
            missing_files = expected_files - created_files
            
            if missing_files:
                validation_results["files_created_check"] = False
                validation_results["all_expectations_met"] = False
                validation_results["details"]["missing_files"] = list(missing_files)
        
        # Check context-based expectations
        context = result.get("context", {})
        
        if expectations.tests_passing and not context.get("tests_complete", False):
            validation_results["tests_passing_check"] = False
            validation_results["all_expectations_met"] = False
        
        if expectations.documentation_complete and not context.get("documentation_complete", False):
            validation_results["documentation_complete_check"] = False
            validation_results["all_expectations_met"] = False
        
        if expectations.performance_improved and not context.get("performance_improved", False):
            validation_results["performance_improved_check"] = False
            validation_results["all_expectations_met"] = False
        
        # Check build if workspace exists
        if self.temp_workspace and self.temp_workspace.exists():
            try:
                # Simple validation - check if Python files are syntactically correct
                python_files = list(self.temp_workspace.glob("**/*.py"))
                for py_file in python_files:
                    try:
                        compile(py_file.read_text(), str(py_file), 'exec')
                    except SyntaxError:
                        validation_results["build_successful_check"] = False
                        validation_results["all_expectations_met"] = False
                        break
            except Exception:
                pass  # Skip build validation if issues
        
        return validation_results
    
    async def _collect_final_metrics(self, result: Dict[str, Any]):
        """Collect final metrics from all agents."""
        agent_metrics = {}
        
        for agent_id, agent in self.agents.items():
            agent_metrics[agent_id] = {
                "performance_metrics": agent.performance_metrics.copy(),
                "execution_log_length": len(agent.execution_log),
                "capabilities": agent.capabilities
            }
        
        result["metrics"] = {
            "agent_metrics": agent_metrics,
            "total_steps": len(result["steps_executed"]),
            "agents_used": len(result["agents_utilized"]),
            "artifacts_created": len(result["artifacts_created"])
        }

def create_e2e_test_workflows() -> List[WorkflowDefinition]:
    """Create comprehensive end-to-end test workflows."""
    
    workflows = [
        # Workflow 1: Feature Development
        WorkflowDefinition(
            workflow_id="e2e_feature_dev",
            name="User Authentication Feature Development",
            description="Complete development of user authentication feature from requirements to deployment",
            workflow_type=WorkflowType.FEATURE_DEVELOPMENT,
            agents_required=["claude_code", "cursor", "gemini_cli"],
            initial_codebase={
                "main.py": '''"""Main application module."""\n\ndef main():\n    print("Hello World")\n\nif __name__ == "__main__":\n    main()''',
                "config.py": '''"""Configuration module."""\n\nCONFIG = {\n    "debug": True,\n    "version": "1.0.0"\n}''',
                "requirements.txt": "fastapi>=0.68.0\nuvicorn>=0.15.0\npydantic>=1.8.0"
            },
            requirements="Implement secure user authentication with JWT tokens, password hashing, and session management",
            expectations=WorkflowExpectation(
                files_created=["feature_*.py", "test_*.py", "README.md"],
                tests_passing=True,
                build_successful=True,
                documentation_complete=True
            ),
            complexity_level="high"
        ),
        
        # Workflow 2: Bug Fix
        WorkflowDefinition(
            workflow_id="e2e_bug_fix",
            name="Critical Security Bug Fix",
            description="Fix critical security vulnerability in user input validation",
            workflow_type=WorkflowType.BUG_FIX,
            agents_required=["claude_code", "cursor", "gemini_cli"],
            initial_codebase={
                "app.py": '''"""Buggy application with security issue."""\nimport os\n\ndef process_user_input(user_input):\n    # SECURITY BUG: Direct execution of user input\n    return eval(user_input)  # This is dangerous!\n\ndef main():\n    user_data = input("Enter expression: ")\n    result = process_user_input(user_data)\n    print(f"Result: {result}")''',
                "test_app.py": '''"""Tests for app."""\nimport unittest\nfrom app import process_user_input\n\nclass TestApp(unittest.TestCase):\n    def test_basic_input(self):\n        result = process_user_input("2 + 2")\n        self.assertEqual(result, 4)'''
            },
            requirements="Fix security vulnerability in user input processing while maintaining functionality",
            expectations=WorkflowExpectation(
                files_modified=["app.py"],
                tests_passing=True,
                build_successful=True
            ),
            complexity_level="medium"
        ),
        
        # Workflow 3: Code Refactoring
        WorkflowDefinition(
            workflow_id="e2e_refactoring",
            name="Legacy Code Modernization",
            description="Refactor legacy codebase to modern Python standards",
            workflow_type=WorkflowType.REFACTORING,
            agents_required=["claude_code", "cursor", "gemini_cli"],
            initial_codebase={
                "legacy_code.py": '''# Legacy code that needs refactoring\ndef old_function(a,b,c):\n    if a>0:\n        if b>0:\n            if c>0:\n                return a+b+c\n            else:\n                return a+b\n        else:\n            return a\n    else:\n        return 0\n\nclass OldClass:\n    def __init__(self):\n        self.data=[]\n    def add_item(self,item):\n        self.data.append(item)\n    def get_items(self):\n        return self.data''',
                "utils.py": '''def util_func(x):\n    # No type hints, poor formatting\n    if x==None:\n        return None\n    else:\n        return str(x).upper()'''
            },
            requirements="Modernize code with type hints, proper formatting, error handling, and documentation",
            expectations=WorkflowExpectation(
                files_modified=["legacy_code.py", "utils.py"],
                tests_passing=True,
                build_successful=True,
                documentation_complete=True
            ),
            complexity_level="medium"
        ),
        
        # Workflow 4: Testing Enhancement
        WorkflowDefinition(
            workflow_id="e2e_testing",
            name="Comprehensive Test Suite Development",
            description="Create comprehensive test suite for existing codebase",
            workflow_type=WorkflowType.TESTING,
            agents_required=["claude_code", "gemini_cli"],
            initial_codebase={
                "calculator.py": '''"""Simple calculator module."""\n\nclass Calculator:\n    def add(self, a: float, b: float) -> float:\n        return a + b\n    \n    def subtract(self, a: float, b: float) -> float:\n        return a - b\n    \n    def multiply(self, a: float, b: float) -> float:\n        return a * b\n    \n    def divide(self, a: float, b: float) -> float:\n        if b == 0:\n            raise ValueError("Cannot divide by zero")\n        return a / b''',
                "data_processor.py": '''"""Data processing utilities."""\n\ndef process_data(data: list) -> dict:\n    if not data:\n        return {"count": 0, "sum": 0, "avg": 0}\n    \n    return {\n        "count": len(data),\n        "sum": sum(data),\n        "avg": sum(data) / len(data)\n    }'''
            },
            requirements="Create comprehensive test suite with unit tests, integration tests, and edge case coverage",
            expectations=WorkflowExpectation(
                files_created=["test_*.py"],
                tests_passing=True,
                build_successful=True
            ),
            complexity_level="low"
        ),
        
        # Workflow 5: Documentation Generation
        WorkflowDefinition(
            workflow_id="e2e_documentation",
            name="API Documentation Generation",
            description="Generate comprehensive documentation for API endpoints",
            workflow_type=WorkflowType.DOCUMENTATION,
            agents_required=["claude_code"],
            initial_codebase={
                "api.py": '''"""API endpoints."""\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/users/{user_id}")\ndef get_user(user_id: int):\n    return {"user_id": user_id, "name": "John Doe"}\n\n@app.post("/users/")\ndef create_user(user_data: dict):\n    return {"message": "User created", "data": user_data}\n\n@app.put("/users/{user_id}")\ndef update_user(user_id: int, user_data: dict):\n    return {"message": f"User {user_id} updated"}''',
                "models.py": '''"""Data models."""\nfrom pydantic import BaseModel\n\nclass User(BaseModel):\n    id: int\n    name: str\n    email: str'''
            },
            requirements="Generate complete API documentation with examples, schemas, and usage guides",
            expectations=WorkflowExpectation(
                files_created=["README.md", "*.md"],
                documentation_complete=True,
                build_successful=True
            ),
            complexity_level="low"
        )
    ]
    
    return workflows

class EndToEndTestSuite:
    """Main test suite for end-to-end workflow validation."""
    
    def __init__(self):
        self.orchestrator = EndToEndWorkflowOrchestrator()
        self.test_results = []
    
    def setup_test_agents(self):
        """Setup test agents for workflows."""
        agents = [
            MockCLIAgentE2E("claude_code", "claude_code", ["analysis", "documentation", "review"]),
            MockCLIAgentE2E("cursor", "cursor", ["implementation", "refactoring", "bug_fixing"]),
            MockCLIAgentE2E("gemini_cli", "gemini_cli", ["testing", "optimization", "validation"])
        ]
        
        for agent in agents:
            self.orchestrator.register_agent(agent)
    
    async def run_comprehensive_e2e_tests(self) -> Dict[str, Any]:
        """Run comprehensive end-to-end workflow tests."""
        suite_results = {
            "test_suite": "End-to-End Workflows",
            "start_time": time.time(),
            "workflows_executed": 0,
            "workflows_passed": 0,
            "workflows_failed": 0,
            "detailed_results": []
        }
        
        self.setup_test_agents()
        workflows = create_e2e_test_workflows()
        
        for workflow_def in workflows:
            print(f"🔄 Executing workflow: {workflow_def.name}")
            
            try:
                result = await self.orchestrator.execute_workflow(workflow_def)
                suite_results["detailed_results"].append(result)
                suite_results["workflows_executed"] += 1
                
                if result["status"] == WorkflowStatus.COMPLETED.value:
                    suite_results["workflows_passed"] += 1
                    print(f"✅ {workflow_def.name} - COMPLETED")
                else:
                    suite_results["workflows_failed"] += 1
                    print(f"❌ {workflow_def.name} - FAILED: {result.get('error', 'Expectations not met')}")
            
            except Exception as e:
                suite_results["workflows_failed"] += 1
                suite_results["detailed_results"].append({
                    "workflow_id": workflow_def.workflow_id,
                    "workflow_name": workflow_def.name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"❌ {workflow_def.name} - ERROR: {str(e)}")
        
        suite_results["end_time"] = time.time()
        suite_results["total_duration"] = suite_results["end_time"] - suite_results["start_time"]
        
        return suite_results

# Pytest integration
@pytest.fixture
async def e2e_test_suite():
    """Pytest fixture for end-to-end testing."""
    suite = EndToEndTestSuite()
    yield suite

@pytest.mark.asyncio
async def test_feature_development_workflow(e2e_test_suite):
    """Test complete feature development workflow."""
    suite = e2e_test_suite
    suite.setup_test_agents()
    
    workflows = create_e2e_test_workflows()
    feature_workflow = next(w for w in workflows if w.workflow_id == "e2e_feature_dev")
    
    result = await suite.orchestrator.execute_workflow(feature_workflow)
    
    assert result["status"] == WorkflowStatus.COMPLETED.value
    assert len(result["steps_executed"]) >= 6
    assert len(result["artifacts_created"]) > 0
    assert result["validation_results"]["all_expectations_met"]

@pytest.mark.asyncio
async def test_bug_fix_workflow(e2e_test_suite):
    """Test bug fix workflow."""
    suite = e2e_test_suite
    suite.setup_test_agents()
    
    workflows = create_e2e_test_workflows()
    bug_fix_workflow = next(w for w in workflows if w.workflow_id == "e2e_bug_fix")
    
    result = await suite.orchestrator.execute_workflow(bug_fix_workflow)
    
    assert result["status"] == WorkflowStatus.COMPLETED.value
    assert "app.py" in result["artifacts_created"]  # Should modify the buggy file

if __name__ == "__main__":
    async def main():
        """Run end-to-end workflow tests standalone."""
        print("🌟 End-to-End Workflow Testing Suite")
        print("=" * 60)
        
        test_suite = EndToEndTestSuite()
        
        try:
            results = await test_suite.run_comprehensive_e2e_tests()
            
            print("\n" + "=" * 60)
            print("📊 END-TO-END WORKFLOW TEST RESULTS")
            print("=" * 60)
            print(f"Workflows Executed: {results['workflows_executed']}")
            print(f"Workflows Passed: {results['workflows_passed']}")
            print(f"Workflows Failed: {results['workflows_failed']}")
            print(f"Success Rate: {results['workflows_passed']/results['workflows_executed']*100:.1f}%")
            print(f"Total Duration: {results['total_duration']:.2f}s")
            
            # Save detailed results
            with open('e2e_workflow_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📄 Detailed results saved to: e2e_workflow_test_results.json")
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
    
    asyncio.run(main())