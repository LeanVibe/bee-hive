"""
CLI Agent Orchestrator for LeanVibe Agent Hive 2.0

Enterprise orchestration platform for multiple CLI coding agents:
- Claude Code (claude.ai/code)
- Gemini CLI  
- OpenCode
- GitHub Copilot CLI
- Custom enterprise agents

Provides unified orchestration, security, and governance for all AI coding tools.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

import structlog

logger = structlog.get_logger()


class CLIAgentType(str, Enum):
    """Types of supported CLI coding agents."""
    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli"
    OPENCODE = "opencode"
    GITHUB_COPILOT = "github_copilot"
    CURSOR_CLI = "cursor_cli"
    CODEIUM_CLI = "codeium_cli"
    CUSTOM = "custom"


class AgentCapability(str, Enum):
    """Core capabilities that CLI agents can provide."""
    ARCHITECTURAL_DESIGN = "architectural_design"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_OPTIMIZATION = "code_optimization"
    RAPID_PROTOTYPING = "rapid_prototyping"
    OSS_INTEGRATION = "oss_integration"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    PERFORMANCE_TUNING = "performance_tuning"
    SECURITY_ANALYSIS = "security_analysis"


@dataclass
class CLIAgentInfo:
    """Information about a detected CLI agent."""
    agent_type: CLIAgentType
    name: str
    version: Optional[str]
    executable_path: str
    capabilities: List[AgentCapability]
    confidence_scores: Dict[AgentCapability, float]
    is_available: bool
    installation_status: str
    last_detected: datetime


@dataclass
class AgentTask:
    """Task to be executed by a CLI agent."""
    id: str
    description: str
    task_type: AgentCapability
    requirements: List[str]
    context: Dict[str, Any]
    priority: int = 5
    timeout_seconds: int = 60


@dataclass
class AgentResponse:
    """Response from a CLI agent execution."""
    task_id: str
    agent_type: CLIAgentType
    success: bool
    output: str
    artifacts: List[Dict[str, Any]]
    execution_time_seconds: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CLIAgentAdapter(ABC):
    """Abstract base class for CLI agent adapters."""
    
    def __init__(self, agent_info: CLIAgentInfo):
        self.agent_info = agent_info
        self.logger = structlog.get_logger().bind(agent_type=agent_info.agent_type.value)
    
    @abstractmethod
    async def detect_installation(self) -> CLIAgentInfo:
        """Detect if this CLI agent is installed and available."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a task using this CLI agent."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[AgentCapability, float]:
        """Get capability confidence scores for this agent."""
        pass
    
    async def validate_task(self, task: AgentTask) -> bool:
        """Validate if this agent can handle the given task."""
        capabilities = self.get_capabilities()
        return task.task_type in capabilities and capabilities[task.task_type] > 0.5


class ClaudeCodeAdapter(CLIAgentAdapter):
    """Adapter for Claude Code CLI agent."""
    
    async def detect_installation(self) -> CLIAgentInfo:
        """Detect Claude Code installation."""
        try:
            # Check for claude-code executable
            claude_path = shutil.which("claude-code") or shutil.which("claude")
            
            if claude_path:
                # Try to get version
                try:
                    result = subprocess.run(
                        [claude_path, "--version"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    version = result.stdout.strip() if result.returncode == 0 else "unknown"
                except Exception:
                    version = "unknown"
                
                return CLIAgentInfo(
                    agent_type=CLIAgentType.CLAUDE_CODE,
                    name="Claude Code",
                    version=version,
                    executable_path=claude_path,
                    capabilities=list(self.get_capabilities().keys()),
                    confidence_scores=self.get_capabilities(),
                    is_available=True,
                    installation_status="detected",
                    last_detected=datetime.utcnow()
                )
            else:
                return CLIAgentInfo(
                    agent_type=CLIAgentType.CLAUDE_CODE,
                    name="Claude Code",
                    version=None,
                    executable_path="",
                    capabilities=[],
                    confidence_scores={},
                    is_available=False,
                    installation_status="not_found",
                    last_detected=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error("Error detecting Claude Code", error=str(e))
            return CLIAgentInfo(
                agent_type=CLIAgentType.CLAUDE_CODE,
                name="Claude Code",
                version=None,
                executable_path="",
                capabilities=[],
                confidence_scores={},
                is_available=False,
                installation_status=f"error: {str(e)}",
                last_detected=datetime.utcnow()
            )
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute task using Claude Code."""
        start_time = datetime.utcnow()
        
        try:
            if not self.agent_info.is_available:
                raise Exception("Claude Code not available")
            
            # Construct Claude Code command based on task type
            command = self._build_claude_command(task)
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Use timeout with asyncio.wait_for
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=task.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise Exception(f"Task execution timed out after {task.timeout_seconds} seconds")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if process.returncode == 0:
                # Parse successful output
                output = stdout.decode('utf-8')
                artifacts = self._parse_claude_output(output, task)
                
                return AgentResponse(
                    task_id=task.id,
                    agent_type=CLIAgentType.CLAUDE_CODE,
                    success=True,
                    output=output,
                    artifacts=artifacts,
                    execution_time_seconds=execution_time
                )
            else:
                # Handle error
                error_msg = stderr.decode('utf-8') or "Unknown error"
                return AgentResponse(
                    task_id=task.id,
                    agent_type=CLIAgentType.CLAUDE_CODE,
                    success=False,
                    output=stdout.decode('utf-8'),
                    artifacts=[],
                    execution_time_seconds=execution_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.CLAUDE_CODE,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> Dict[AgentCapability, float]:
        """Claude Code capability confidence scores."""
        return {
            AgentCapability.ARCHITECTURAL_DESIGN: 0.95,
            AgentCapability.CODE_GENERATION: 0.90,
            AgentCapability.CODE_REVIEW: 0.88,
            AgentCapability.DOCUMENTATION: 0.92,
            AgentCapability.REFACTORING: 0.85,
            AgentCapability.DEBUGGING: 0.83,
            AgentCapability.TESTING: 0.80,
            AgentCapability.SECURITY_ANALYSIS: 0.82
        }
    
    def _build_claude_command(self, task: AgentTask) -> List[str]:
        """Build Claude Code command based on task."""
        base_cmd = [self.agent_info.executable_path, "--print"]
        
        # Build comprehensive prompt based on task type and requirements
        prompt_parts = []
        
        if task.task_type == AgentCapability.CODE_GENERATION:
            prompt_parts.append(f"Generate code for: {task.description}")
        elif task.task_type == AgentCapability.CODE_REVIEW:
            prompt_parts.append(f"Review this code and provide feedback: {task.description}")
        elif task.task_type == AgentCapability.REFACTORING:
            prompt_parts.append(f"Refactor this code: {task.description}")
        elif task.task_type == AgentCapability.ARCHITECTURAL_DESIGN:
            prompt_parts.append(f"Design the architecture for: {task.description}")
        elif task.task_type == AgentCapability.DEBUGGING:
            prompt_parts.append(f"Debug and fix this issue: {task.description}")
        else:
            prompt_parts.append(f"Help with: {task.description}")
        
        # Add requirements as context
        if task.requirements:
            prompt_parts.append("Requirements:")
            for req in task.requirements:
                prompt_parts.append(f"- {req}")
        
        # Add context information
        if task.context:
            if task.context.get("language"):
                prompt_parts.append(f"Language: {task.context['language']}")
            if task.context.get("framework"):
                prompt_parts.append(f"Framework: {task.context['framework']}")
        
        # Join all prompt parts
        full_prompt = "\n".join(prompt_parts)
        base_cmd.append(full_prompt)
        
        return base_cmd
    
    def _parse_claude_output(self, output: str, task: AgentTask) -> List[Dict[str, Any]]:
        """Parse Claude Code output into artifacts."""
        artifacts = []
        
        # Basic parsing - in real implementation, this would be more sophisticated
        if output.strip():
            artifacts.append({
                "type": "code",
                "content": output,
                "language": task.context.get("language", "python"),
                "file_name": f"claude_generated_{task.id}.py"
            })
        
        return artifacts


class GeminiCLIAdapter(CLIAgentAdapter):
    """Adapter for Gemini CLI agent."""
    
    async def detect_installation(self) -> CLIAgentInfo:
        """Detect Gemini CLI installation."""
        try:
            # Check for gemini executable
            gemini_path = shutil.which("gemini") or shutil.which("gemini-cli")
            
            if gemini_path:
                try:
                    result = subprocess.run(
                        [gemini_path, "--version"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    version = result.stdout.strip() if result.returncode == 0 else "unknown"
                except Exception:
                    version = "unknown"
                
                return CLIAgentInfo(
                    agent_type=CLIAgentType.GEMINI_CLI,
                    name="Gemini CLI",
                    version=version,
                    executable_path=gemini_path,
                    capabilities=list(self.get_capabilities().keys()),
                    confidence_scores=self.get_capabilities(),
                    is_available=True,
                    installation_status="detected",
                    last_detected=datetime.utcnow()
                )
            else:
                return CLIAgentInfo(
                    agent_type=CLIAgentType.GEMINI_CLI,
                    name="Gemini CLI",
                    version=None,
                    executable_path="",
                    capabilities=[],
                    confidence_scores={},
                    is_available=False,
                    installation_status="not_found",
                    last_detected=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error("Error detecting Gemini CLI", error=str(e))
            return CLIAgentInfo(
                agent_type=CLIAgentType.GEMINI_CLI,
                name="Gemini CLI",
                version=None,
                executable_path="",
                capabilities=[],
                confidence_scores={},
                is_available=False,
                installation_status=f"error: {str(e)}",
                last_detected=datetime.utcnow()
            )
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute task using Gemini CLI."""
        start_time = datetime.utcnow()
        
        try:
            if not self.agent_info.is_available:
                raise Exception("Gemini CLI not available")
            
            # For now, simulate Gemini CLI execution
            # In real implementation, this would call actual Gemini CLI
            await asyncio.sleep(1)  # Simulate processing time
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Simulate successful response
            simulated_output = f"Gemini CLI response for task: {task.description}"
            artifacts = [{
                "type": "code",
                "content": f"# Generated by Gemini CLI\\n# Task: {task.description}\\npass",
                "language": task.context.get("language", "python"),
                "file_name": f"gemini_generated_{task.id}.py"
            }]
            
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.GEMINI_CLI,
                success=True,
                output=simulated_output,
                artifacts=artifacts,
                execution_time_seconds=execution_time,
                metadata={"simulated": True}
            )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.GEMINI_CLI,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> Dict[AgentCapability, float]:
        """Gemini CLI capability confidence scores."""
        return {
            AgentCapability.RAPID_PROTOTYPING: 0.93,
            AgentCapability.CODE_OPTIMIZATION: 0.89,
            AgentCapability.PERFORMANCE_TUNING: 0.87,
            AgentCapability.CODE_GENERATION: 0.85,
            AgentCapability.REFACTORING: 0.83,
            AgentCapability.DEBUGGING: 0.80,
            AgentCapability.TESTING: 0.78
        }


class OpenCodeAdapter(CLIAgentAdapter):
    """Adapter for OpenCode CLI agent."""
    
    async def detect_installation(self) -> CLIAgentInfo:
        """Detect OpenCode installation."""
        try:
            # Check for opencode executable
            opencode_path = shutil.which("opencode") or shutil.which("oc")
            
            if opencode_path:
                try:
                    result = subprocess.run(
                        [opencode_path, "--version"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    version = result.stdout.strip() if result.returncode == 0 else "unknown"
                except Exception:
                    version = "unknown"
                
                return CLIAgentInfo(
                    agent_type=CLIAgentType.OPENCODE,
                    name="OpenCode",
                    version=version,
                    executable_path=opencode_path,
                    capabilities=list(self.get_capabilities().keys()),
                    confidence_scores=self.get_capabilities(),
                    is_available=True,
                    installation_status="detected",
                    last_detected=datetime.utcnow()
                )
            else:
                return CLIAgentInfo(
                    agent_type=CLIAgentType.OPENCODE,
                    name="OpenCode",
                    version=None,
                    executable_path="",
                    capabilities=[],
                    confidence_scores={},
                    is_available=False,
                    installation_status="not_found",
                    last_detected=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error("Error detecting OpenCode", error=str(e))
            return CLIAgentInfo(
                agent_type=CLIAgentType.OPENCODE,
                name="OpenCode",
                version=None,
                executable_path="",
                capabilities=[],
                confidence_scores={},
                is_available=False,
                installation_status=f"error: {str(e)}",
                last_detected=datetime.utcnow()
            )
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute task using OpenCode."""
        start_time = datetime.utcnow()
        
        try:
            if not self.agent_info.is_available:
                raise Exception("OpenCode not available")
            
            # For now, simulate OpenCode execution
            await asyncio.sleep(0.8)  # Simulate processing time
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Simulate successful response
            simulated_output = f"OpenCode response for task: {task.description}"
            artifacts = [{
                "type": "code",
                "content": f"# Generated by OpenCode\\n# Task: {task.description}\\n# Uses OSS best practices\\npass",
                "language": task.context.get("language", "python"),
                "file_name": f"opencode_generated_{task.id}.py"
            }]
            
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.OPENCODE,
                success=True,
                output=simulated_output,
                artifacts=artifacts,
                execution_time_seconds=execution_time,
                metadata={"simulated": True}
            )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.OPENCODE,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> Dict[AgentCapability, float]:
        """OpenCode capability confidence scores."""
        return {
            AgentCapability.OSS_INTEGRATION: 0.94,
            AgentCapability.CODE_GENERATION: 0.82,
            AgentCapability.TESTING: 0.85,
            AgentCapability.DOCUMENTATION: 0.79,
            AgentCapability.SECURITY_ANALYSIS: 0.77,
            AgentCapability.REFACTORING: 0.75
        }


class CLIAgentOrchestrator:
    """
    Enterprise orchestration platform for multiple CLI coding agents.
    
    Provides unified coordination, task routing, and quality assurance
    across Claude Code, Gemini CLI, OpenCode, and other CLI agents.
    """
    
    def __init__(self):
        self.agents: Dict[CLIAgentType, CLIAgentAdapter] = {}
        self.agent_info: Dict[CLIAgentType, CLIAgentInfo] = {}
        self.logger = structlog.get_logger().bind(component="cli_orchestrator")
        
        # Initialize available adapters
        self.available_adapters = {
            CLIAgentType.CLAUDE_CODE: ClaudeCodeAdapter,
            CLIAgentType.GEMINI_CLI: GeminiCLIAdapter,
            CLIAgentType.OPENCODE: OpenCodeAdapter
        }
    
    async def detect_available_agents(self) -> Dict[CLIAgentType, CLIAgentInfo]:
        """Detect all available CLI coding agents."""
        self.logger.info("Starting CLI agent detection")
        
        detection_tasks = []
        for agent_type, adapter_class in self.available_adapters.items():
            # Create temporary adapter for detection
            temp_info = CLIAgentInfo(
                agent_type=agent_type,
                name=agent_type.value,
                version=None,
                executable_path="",
                capabilities=[],
                confidence_scores={},
                is_available=False,
                installation_status="detecting",
                last_detected=datetime.utcnow()
            )
            adapter = adapter_class(temp_info)
            detection_tasks.append(adapter.detect_installation())
        
        # Run detection in parallel
        detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(detection_results):
            if isinstance(result, CLIAgentInfo):
                agent_type = result.agent_type
                self.agent_info[agent_type] = result
                
                if result.is_available:
                    # Create adapter for available agent
                    adapter_class = self.available_adapters[agent_type]
                    self.agents[agent_type] = adapter_class(result)
                    self.logger.info(
                        "CLI agent detected",
                        agent_type=agent_type.value,
                        version=result.version,
                        path=result.executable_path
                    )
                else:
                    self.logger.info(
                        "CLI agent not available",
                        agent_type=agent_type.value,
                        status=result.installation_status
                    )
            else:
                agent_type = list(self.available_adapters.keys())[i]
                self.logger.error(
                    "CLI agent detection failed",
                    agent_type=agent_type.value,
                    error=str(result)
                )
        
        available_count = len([info for info in self.agent_info.values() if info.is_available])
        self.logger.info(
            "CLI agent detection complete",
            total_agents=len(self.available_adapters),
            available_agents=available_count
        )
        
        return self.agent_info
    
    def get_available_agents(self) -> List[CLIAgentInfo]:
        """Get list of available CLI agents."""
        return [info for info in self.agent_info.values() if info.is_available]
    
    def select_optimal_agent(self, task: AgentTask) -> Optional[CLIAgentType]:
        """Select the optimal agent for a given task."""
        best_agent = None
        best_score = 0.0
        
        for agent_type, adapter in self.agents.items():
            capabilities = adapter.get_capabilities()
            if task.task_type in capabilities:
                score = capabilities[task.task_type]
                if score > best_score:
                    best_score = score
                    best_agent = agent_type
        
        if best_agent:
            self.logger.info(
                "Optimal agent selected",
                task_id=task.id,
                task_type=task.task_type.value,
                selected_agent=best_agent.value,
                confidence_score=best_score
            )
        
        return best_agent
    
    async def execute_with_agent(self, agent_type: CLIAgentType, task: AgentTask) -> AgentResponse:
        """Execute task with specific agent."""
        if agent_type not in self.agents:
            return AgentResponse(
                task_id=task.id,
                agent_type=agent_type,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=0.0,
                error_message=f"Agent {agent_type.value} not available"
            )
        
        adapter = self.agents[agent_type]
        
        # Validate task
        if not await adapter.validate_task(task):
            return AgentResponse(
                task_id=task.id,
                agent_type=agent_type,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=0.0,
                error_message=f"Agent {agent_type.value} cannot handle task type {task.task_type.value}"
            )
        
        self.logger.info(
            "Executing task with agent",
            task_id=task.id,
            agent_type=agent_type.value,
            task_type=task.task_type.value
        )
        
        return await adapter.execute_task(task)
    
    async def execute_with_optimal_agent(self, task: AgentTask) -> AgentResponse:
        """Execute task with the optimal available agent."""
        optimal_agent = self.select_optimal_agent(task)
        
        if not optimal_agent:
            return AgentResponse(
                task_id=task.id,
                agent_type=CLIAgentType.CUSTOM,
                success=False,
                output="",
                artifacts=[],
                execution_time_seconds=0.0,
                error_message=f"No available agent can handle task type {task.task_type.value}"
            )
        
        return await self.execute_with_agent(optimal_agent, task)
    
    async def execute_with_multiple_agents(self, task: AgentTask, agent_types: List[CLIAgentType]) -> List[AgentResponse]:
        """Execute task with multiple agents for cross-validation."""
        self.logger.info(
            "Executing task with multiple agents",
            task_id=task.id,
            agents=[agent.value for agent in agent_types]
        )
        
        # Execute in parallel
        execution_tasks = [
            self.execute_with_agent(agent_type, task)
            for agent_type in agent_types
            if agent_type in self.agents
        ]
        
        responses = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Filter successful responses
        valid_responses = [
            response for response in responses
            if isinstance(response, AgentResponse)
        ]
        
        return valid_responses
    
    def calculate_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Calculate consensus from multiple agent responses."""
        if not responses:
            return {"consensus": "no_responses", "confidence": 0.0}
        
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            return {"consensus": "all_failed", "confidence": 0.0}
        
        # Simple consensus based on success rate
        success_rate = len(successful_responses) / len(responses)
        
        # In a more sophisticated implementation, this would compare
        # actual code outputs for similarity and quality
        
        return {
            "consensus": "agreement" if success_rate >= 0.7 else "disagreement",
            "confidence": success_rate,
            "successful_agents": len(successful_responses),
            "total_agents": len(responses),
            "avg_execution_time": sum(r.execution_time_seconds for r in successful_responses) / len(successful_responses)
        }


# Factory function for easy instantiation
async def create_cli_agent_orchestrator() -> CLIAgentOrchestrator:
    """Create and initialize CLI agent orchestrator."""
    orchestrator = CLIAgentOrchestrator()
    await orchestrator.detect_available_agents()
    return orchestrator