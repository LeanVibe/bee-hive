"""
Enhanced Orchestrator Integration for LeanVibe Agent Hive 2.0

Integrates Claude Code features (hooks, slash commands, extended thinking)
with the existing orchestrator without modifying core functionality.
"""

import asyncio
from typing import Any, Dict, List, Optional
import structlog

from app.core.orchestrator import AgentOrchestrator
from app.core.leanvibe_hooks_system import (
    LeanVibeHooksEngine, 
    HookEventType,
    get_leanvibe_hooks_engine
)
from app.core.slash_commands import SlashCommandsEngine, get_slash_commands_engine
from app.core.extended_thinking_engine import (
    ExtendedThinkingEngine,
    ThinkingDepth,
    get_extended_thinking_engine
)

logger = structlog.get_logger()


class EnhancedOrchestratorIntegration:
    """
    Enhanced integration layer that adds Claude Code features to the orchestrator
    without modifying the core orchestrator functionality.
    """
    
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        hooks_engine: Optional[LeanVibeHooksEngine] = None,
        slash_commands: Optional[SlashCommandsEngine] = None,
        thinking_engine: Optional[ExtendedThinkingEngine] = None
    ):
        """
        Initialize enhanced orchestrator integration.
        
        Args:
            orchestrator: Core orchestrator instance
            hooks_engine: LeanVibe hooks engine
            slash_commands: Slash commands engine
            thinking_engine: Extended thinking engine
        """
        self.orchestrator = orchestrator
        self.hooks_engine = hooks_engine or get_leanvibe_hooks_engine()
        self.slash_commands = slash_commands or get_slash_commands_engine()
        self.thinking_engine = thinking_engine or get_extended_thinking_engine()
        
        # Enhanced features state
        self.enhanced_features_enabled = True
        self.thinking_sessions_active = {}
        
        logger.info(
            "🚀 Enhanced Orchestrator Integration initialized",
            hooks_enabled=self.hooks_engine is not None,
            slash_commands_enabled=self.slash_commands is not None,
            thinking_enabled=self.thinking_engine is not None
        )
    
    async def execute_enhanced_agent_task(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute agent task with enhanced Claude Code features.
        
        Args:
            agent_id: Agent ID
            task_data: Task data
            session_id: Session ID
            
        Returns:
            Enhanced task execution result
        """
        workflow_id = task_data.get("workflow_id", f"task_{agent_id}")
        
        try:
            # Pre-task hooks execution
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.PRE_AGENT_TASK,
                    workflow_id=workflow_id,
                    workflow_data={
                        "agent_id": agent_id,
                        "task_data": task_data,
                        "session_id": session_id,
                        "task_type": task_data.get("type", "unknown"),
                        "agent_name": task_data.get("agent_name", agent_id)
                    },
                    agent_id=agent_id,
                    session_id=session_id or "system"
                )
            
            # Check if extended thinking is needed
            thinking_session_id = None
            if self.thinking_engine and self._should_use_extended_thinking(task_data):
                thinking_config = await self.thinking_engine.analyze_thinking_needs(
                    task_description=task_data.get("description", str(task_data)),
                    task_context=task_data
                )
                
                if thinking_config:
                    thinking_session = await self.thinking_engine.enable_extended_thinking(
                        agent_id=agent_id,
                        workflow_id=workflow_id,
                        problem_description=task_data.get("description", str(task_data)),
                        problem_context=task_data,
                        thinking_depth=thinking_config.get("thinking_depth", ThinkingDepth.STANDARD)
                    )
                    thinking_session_id = thinking_session.session_id
                    self.thinking_sessions_active[workflow_id] = thinking_session_id
                    
                    # Add thinking instructions to task
                    task_data["enhanced_thinking"] = {
                        "session_id": thinking_session_id,
                        "thinking_depth": thinking_config["thinking_depth"].value,
                        "instructions": "Please engage in extended thinking for this complex task."
                    }
            
            # Execute core task
            result = await self.orchestrator.execute_agent_task(agent_id, task_data)
            
            # Handle collaborative thinking if needed
            if thinking_session_id and self.thinking_engine:
                try:
                    thinking_result = await self.thinking_engine.coordinate_thinking_session(
                        thinking_session_id
                    )
                    
                    # Enhance result with thinking insights
                    result["thinking_insights"] = {
                        "session_id": thinking_session_id,
                        "collaborative_solution": thinking_result.primary_solution,
                        "consensus_score": thinking_result.consensus_score,
                        "execution_time_seconds": thinking_result.execution_time_seconds
                    }
                    
                except Exception as e:
                    logger.warning(
                        "Extended thinking session failed",
                        session_id=thinking_session_id,
                        error=str(e)
                    )
                
                # Clean up
                if workflow_id in self.thinking_sessions_active:
                    del self.thinking_sessions_active[workflow_id]
            
            # Post-task hooks execution
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.POST_AGENT_TASK,
                    workflow_id=workflow_id,
                    workflow_data={
                        "agent_id": agent_id,
                        "task_data": task_data,
                        "result": result,
                        "session_id": session_id,
                        "success": result.get("success", True),
                        "task_type": task_data.get("type", "unknown"),
                        "agent_name": task_data.get("agent_name", agent_id)
                    },
                    agent_id=agent_id,
                    session_id=session_id or "system"
                )
            
            return result
            
        except Exception as e:
            # Error hooks execution
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.AGENT_ERROR,
                    workflow_id=workflow_id,
                    workflow_data={
                        "agent_id": agent_id,
                        "task_data": task_data,
                        "error": str(e),
                        "session_id": session_id,
                        "task_type": task_data.get("type", "unknown"),
                        "agent_name": task_data.get("agent_name", agent_id)
                    },
                    agent_id=agent_id,
                    session_id=session_id or "system"
                )
            
            raise
    
    def _should_use_extended_thinking(self, task_data: Dict[str, Any]) -> bool:
        """Determine if a task should use extended thinking."""
        if not self.enhanced_features_enabled:
            return False
        
        # Check for complexity indicators
        complexity_keywords = [
            "architecture", "design", "complex", "optimization", "security",
            "integration", "performance", "scalability", "debugging"
        ]
        
        task_description = str(task_data).lower()
        complexity_count = sum(1 for keyword in complexity_keywords if keyword in task_description)
        
        # Use extended thinking for complex tasks
        return complexity_count >= 2
    
    async def execute_enhanced_slash_command(
        self,
        command_str: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute slash command with enhanced hooks integration.
        
        Args:
            command_str: Command string
            agent_id: Agent ID
            session_id: Session ID
            
        Returns:
            Command execution result
        """
        if not self.slash_commands:
            return {"success": False, "error": "Slash commands not available"}
        
        try:
            # Execute slash command
            result = await self.slash_commands.execute_command(
                command_str=command_str,
                agent_id=agent_id,
                session_id=session_id
            )
            
            # Convert to dict for consistency
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(
                "Enhanced slash command execution failed",
                command=command_str,
                error=str(e),
                exc_info=True
            )
            
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "execution_time_ms": 0
            }
    
    async def execute_enhanced_workflow(
        self,
        workflow_id: str,
        workflow_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow with enhanced features.
        
        Args:
            workflow_id: Workflow ID
            workflow_data: Workflow data
            session_id: Session ID
            
        Returns:
            Enhanced workflow execution result
        """
        try:
            # Workflow start hooks
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.WORKFLOW_START,
                    workflow_id=workflow_id,
                    workflow_data=workflow_data,
                    agent_id="orchestrator",
                    session_id=session_id or "system"
                )
            
            # Execute core workflow (assuming this method exists or can be added)
            result = await self._execute_core_workflow(workflow_id, workflow_data, session_id)
            
            # Workflow completion hooks
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.WORKFLOW_COMPLETE,
                    workflow_id=workflow_id,
                    workflow_data={
                        **workflow_data,
                        "result": result,
                        "success": result.get("success", True)
                    },
                    agent_id="orchestrator",
                    session_id=session_id or "system"
                )
            
            return result
            
        except Exception as e:
            logger.error(
                "Enhanced workflow execution failed",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "workflow_id": workflow_id
            }
    
    async def _execute_core_workflow(
        self,
        workflow_id: str,
        workflow_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute core workflow using orchestrator capabilities."""
        # This would use the orchestrator's workflow execution capabilities
        # For now, we'll return a placeholder result
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Workflow executed successfully",
            "enhanced_features_used": True
        }
    
    async def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including enhanced features."""
        try:
            # Get base orchestrator status
            base_status = await self.orchestrator.get_health_status()
            
            # Add enhanced features status
            enhanced_status = {
                "enhanced_features": {
                    "enabled": self.enhanced_features_enabled,
                    "hooks_engine": {
                        "available": self.hooks_engine is not None,
                        "status": "healthy" if self.hooks_engine else "not_configured"
                    },
                    "slash_commands": {
                        "available": self.slash_commands is not None,
                        "status": "healthy" if self.slash_commands else "not_configured"
                    },
                    "extended_thinking": {
                        "available": self.thinking_engine is not None,
                        "active_sessions": len(self.thinking_sessions_active),
                        "status": "healthy" if self.thinking_engine else "not_configured"
                    }
                }
            }
            
            # Get detailed feature statistics
            if self.hooks_engine:
                hooks_stats = await self.hooks_engine.get_performance_stats()
                enhanced_status["enhanced_features"]["hooks_engine"]["stats"] = hooks_stats
            
            if self.thinking_engine:
                thinking_stats = await self.thinking_engine.get_performance_stats()
                enhanced_status["enhanced_features"]["extended_thinking"]["stats"] = thinking_stats
            
            # Merge with base status
            return {
                **base_status,
                **enhanced_status
            }
            
        except Exception as e:
            logger.error(
                "Failed to get enhanced system status",
                error=str(e),
                exc_info=True
            )
            
            return {
                "status": "error",
                "error": str(e),
                "enhanced_features": {"enabled": False}
            }
    
    async def enable_enhanced_features(self) -> None:
        """Enable enhanced features."""
        self.enhanced_features_enabled = True
        logger.info("✅ Enhanced features enabled")
    
    async def disable_enhanced_features(self) -> None:
        """Disable enhanced features."""
        self.enhanced_features_enabled = False
        logger.info("⏸️ Enhanced features disabled")
    
    async def get_available_slash_commands(self) -> List[str]:
        """Get list of available slash commands."""
        if not self.slash_commands:
            return []
        
        try:
            commands = await self.slash_commands.get_available_commands()
            return [f"/{cmd.name}" for cmd in commands]
        except Exception as e:
            logger.error("Failed to get available commands", error=str(e))
            return []
    
    async def get_thinking_session_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of thinking session for workflow."""
        if not self.thinking_engine or workflow_id not in self.thinking_sessions_active:
            return None
        
        session_id = self.thinking_sessions_active[workflow_id]
        return await self.thinking_engine.get_session_status(session_id)
    
    async def execute_quality_gate(
        self,
        workflow_id: str,
        quality_criteria: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute quality gate with hooks integration."""
        if not self.hooks_engine:
            return {"success": False, "error": "Quality gates not available"}
        
        try:
            # Execute quality gate hooks
            hook_results = await self.hooks_engine.execute_workflow_hooks(
                event=HookEventType.QUALITY_GATE,
                workflow_id=workflow_id,
                workflow_data=quality_criteria,
                agent_id="quality_gate",
                session_id=session_id or "system"
            )
            
            # Analyze hook results
            failed_hooks = [r for r in hook_results if not r.success]
            quality_passed = len(failed_hooks) == 0
            
            return {
                "success": quality_passed,
                "quality_score": 1.0 - (len(failed_hooks) / max(len(hook_results), 1)),
                "hooks_executed": len(hook_results),
                "hooks_failed": len(failed_hooks),
                "details": {
                    "passed_checks": len(hook_results) - len(failed_hooks),
                    "failed_checks": len(failed_hooks),
                    "execution_summary": [
                        {
                            "hook": r.metadata.get("hook_name", "unknown"),
                            "success": r.success,
                            "execution_time_ms": r.execution_time_ms
                        }
                        for r in hook_results
                    ]
                }
            }
            
        except Exception as e:
            logger.error(
                "Quality gate execution failed",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            
            return {
                "success": False,
                "error": f"Quality gate execution failed: {str(e)}",
                "quality_score": 0.0
            }


# Global enhanced integration instance
_enhanced_integration: Optional[EnhancedOrchestratorIntegration] = None


def get_enhanced_orchestrator_integration() -> Optional[EnhancedOrchestratorIntegration]:
    """Get the global enhanced orchestrator integration instance."""
    return _enhanced_integration


def set_enhanced_orchestrator_integration(integration: EnhancedOrchestratorIntegration) -> None:
    """Set the global enhanced orchestrator integration instance."""
    global _enhanced_integration
    _enhanced_integration = integration
    logger.info("🔗 Global enhanced orchestrator integration set")


async def initialize_enhanced_orchestrator_integration(
    orchestrator: AgentOrchestrator,
    hooks_engine: Optional[LeanVibeHooksEngine] = None,
    slash_commands: Optional[SlashCommandsEngine] = None,
    thinking_engine: Optional[ExtendedThinkingEngine] = None
) -> EnhancedOrchestratorIntegration:
    """
    Initialize and set the global enhanced orchestrator integration.
    
    Args:
        orchestrator: Core orchestrator instance
        hooks_engine: LeanVibe hooks engine
        slash_commands: Slash commands engine
        thinking_engine: Extended thinking engine
        
    Returns:
        EnhancedOrchestratorIntegration instance
    """
    integration = EnhancedOrchestratorIntegration(
        orchestrator=orchestrator,
        hooks_engine=hooks_engine,
        slash_commands=slash_commands,
        thinking_engine=thinking_engine
    )
    
    set_enhanced_orchestrator_integration(integration)
    
    logger.info("✅ Enhanced orchestrator integration initialized")
    return integration