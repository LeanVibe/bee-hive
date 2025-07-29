"""
Extended Thinking Engine for LeanVibe Agent Hive 2.0

Integrates Claude Code's extended thinking capabilities with LeanVibe's
multi-agent orchestration for enhanced problem-solving on complex tasks.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog
from pydantic import BaseModel, Field

from app.core.orchestrator import AgentOrchestrator
from app.core.communication import MessageBroker

logger = structlog.get_logger()


class ThinkingDepth(str, Enum):
    """Depth levels for extended thinking."""
    STANDARD = "standard"
    DEEP = "deep"
    COLLABORATIVE = "collaborative"
    INTENSIVE = "intensive"


class ThinkingTrigger(str, Enum):
    """Triggers for extended thinking activation."""
    ARCHITECTURAL_DECISIONS = "architectural_decisions"
    SECURITY_ANALYSIS = "security_analysis"
    COMPLEX_DEBUGGING = "complex_debugging"
    SYSTEM_OPTIMIZATION = "system_optimization"
    BUSINESS_LOGIC = "business_logic"
    INTEGRATION_DESIGN = "integration_design"
    PERFORMANCE_ANALYSIS = "performance_analysis"


class ThinkingSession(BaseModel):
    """Extended thinking session configuration."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session ID")
    workflow_id: str = Field(..., description="Associated workflow ID")
    agent_id: str = Field(..., description="Primary agent ID")
    thinking_depth: ThinkingDepth = Field(..., description="Thinking depth level")
    problem_description: str = Field(..., description="Problem description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Problem context")
    participating_agents: List[str] = Field(default_factory=list, description="Participating agent IDs")
    thinking_time_limit_seconds: int = Field(default=1800, description="Thinking time limit")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
    completed_at: Optional[datetime] = Field(None, description="Session completion time")
    status: str = Field(default="active", description="Session status")


class ThinkingResult(BaseModel):
    """Result of extended thinking session."""
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether thinking session succeeded")
    solution: str = Field(default="", description="Generated solution")
    reasoning: str = Field(default="", description="Thinking process and reasoning")
    alternatives_considered: List[str] = Field(default_factory=list, description="Alternative solutions")
    confidence_score: float = Field(default=0.0, description="Confidence in solution (0-1)")
    implementation_steps: List[str] = Field(default_factory=list, description="Implementation steps")
    risks_identified: List[str] = Field(default_factory=list, description="Identified risks")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies and prerequisites")
    execution_time_seconds: float = Field(..., description="Thinking session duration")
    participating_agents: List[str] = Field(default_factory=list, description="Agents that participated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CollaborativeThinkingResult(BaseModel):
    """Result of collaborative thinking session."""
    session_id: str = Field(..., description="Session ID")
    primary_solution: str = Field(..., description="Primary collaborative solution")
    agent_contributions: Dict[str, str] = Field(default_factory=dict, description="Individual agent contributions")
    consensus_score: float = Field(default=0.0, description="Agreement level between agents (0-1)")
    synthesis_reasoning: str = Field(..., description="How solutions were synthesized")
    conflicting_viewpoints: List[str] = Field(default_factory=list, description="Conflicting perspectives")
    unified_implementation: List[str] = Field(default_factory=list, description="Unified implementation plan")
    execution_time_seconds: float = Field(..., description="Total session duration")


class ExtendedThinkingEngine:
    """
    Extended Thinking Engine for complex problem-solving with multi-agent collaboration.
    
    Provides enhanced reasoning capabilities for architectural decisions, complex debugging,
    security analysis, and system optimization through coordinated agent thinking sessions.
    """
    
    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None,
        communication_bus: Optional[MessageBroker] = None
    ):
        """
        Initialize Extended Thinking Engine.
        
        Args:
            orchestrator: Agent orchestrator for multi-agent coordination
            communication_bus: Communication bus for real-time coordination
        """
        self.orchestrator = orchestrator
        self.communication_bus = communication_bus
        
        # Thinking session management
        self.active_sessions: Dict[str, ThinkingSession] = {}
        self.session_history: List[ThinkingSession] = []
        
        # Performance statistics
        self.performance_stats = {
            "sessions_created": 0,
            "sessions_completed": 0,
            "sessions_failed": 0,
            "total_thinking_time_seconds": 0,
            "average_confidence_score": 0.0,
            "collaborative_sessions": 0
        }
        
        # Thinking trigger patterns
        self.thinking_triggers = self._initialize_thinking_triggers()
        
        logger.info("🧠 Extended Thinking Engine initialized")
    
    def _initialize_thinking_triggers(self) -> Dict[ThinkingTrigger, Dict[str, Any]]:
        """Initialize thinking trigger patterns and configurations."""
        return {
            ThinkingTrigger.ARCHITECTURAL_DECISIONS: {
                "patterns": ["design", "architecture", "scalability", "performance", "structure"],
                "thinking_depth": ThinkingDepth.DEEP,
                "preferred_agents": ["solution_architect", "performance_specialist", "backend_specialist"],
                "thinking_time_minutes": 20,
                "collaboration_required": True
            },
            
            ThinkingTrigger.SECURITY_ANALYSIS: {
                "patterns": ["security", "vulnerability", "authentication", "authorization", "encryption"],
                "thinking_depth": ThinkingDepth.DEEP,
                "preferred_agents": ["security_specialist", "backend_specialist"],
                "thinking_time_minutes": 15,
                "collaboration_required": True
            },
            
            ThinkingTrigger.COMPLEX_DEBUGGING: {
                "patterns": ["debug", "error", "issue", "bug", "problem", "failure"],
                "thinking_depth": ThinkingDepth.COLLABORATIVE,
                "preferred_agents": ["debugger", "code_reviewer", "backend_specialist"],
                "thinking_time_minutes": 25,
                "collaboration_required": True
            },
            
            ThinkingTrigger.SYSTEM_OPTIMIZATION: {
                "patterns": ["optimize", "performance", "bottleneck", "efficiency", "speed"],
                "thinking_depth": ThinkingDepth.DEEP,
                "preferred_agents": ["performance_specialist", "devops_specialist"],
                "thinking_time_minutes": 18,
                "collaboration_required": False
            },
            
            ThinkingTrigger.BUSINESS_LOGIC: {
                "patterns": ["business", "logic", "workflow", "process", "requirements"],
                "thinking_depth": ThinkingDepth.STANDARD,
                "preferred_agents": ["business_analyst", "backend_specialist"],
                "thinking_time_minutes": 12,
                "collaboration_required": False
            },
            
            ThinkingTrigger.INTEGRATION_DESIGN: {
                "patterns": ["integration", "api", "service", "microservice", "communication"],
                "thinking_depth": ThinkingDepth.DEEP,
                "preferred_agents": ["integration_specialist", "api_specialist", "backend_specialist"],
                "thinking_time_minutes": 22,
                "collaboration_required": True
            }
        }
    
    async def analyze_thinking_needs(
        self,
        task_description: str,
        task_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze if a task requires extended thinking and determine configuration.
        
        Args:
            task_description: Description of the task
            task_context: Task context and metadata
            
        Returns:
            Thinking configuration if thinking is recommended, None otherwise
        """
        try:
            description_lower = task_description.lower()
            context_text = json.dumps(task_context, default=str).lower()
            combined_text = f"{description_lower} {context_text}"
            
            # Check for thinking triggers
            matching_triggers = []
            
            for trigger, config in self.thinking_triggers.items():
                patterns = config["patterns"]
                
                # Count pattern matches
                pattern_matches = sum(1 for pattern in patterns if pattern in combined_text)
                match_ratio = pattern_matches / len(patterns)
                
                if match_ratio > 0.3:  # At least 30% of patterns match
                    matching_triggers.append((trigger, match_ratio, config))
            
            if not matching_triggers:
                return None
            
            # Select the best matching trigger
            best_trigger, match_ratio, config = max(matching_triggers, key=lambda x: x[1])
            
            # Determine if collaborative thinking is needed
            complexity_indicators = [
                "complex", "difficult", "challenging", "multiple", "various",
                "integration", "system", "architecture", "design"
            ]
            
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in combined_text)
            needs_collaboration = config["collaboration_required"] or complexity_score > 2
            
            thinking_config = {
                "trigger": best_trigger,
                "thinking_depth": ThinkingDepth.COLLABORATIVE if needs_collaboration else config["thinking_depth"],
                "preferred_agents": config["preferred_agents"],
                "thinking_time_minutes": config["thinking_time_minutes"],
                "confidence": match_ratio,
                "complexity_score": complexity_score,
                "collaboration_recommended": needs_collaboration
            }
            
            logger.info(
                "🎯 Extended thinking recommended",
                trigger=best_trigger.value,
                thinking_depth=thinking_config["thinking_depth"].value,
                confidence=match_ratio,
                collaboration=needs_collaboration
            )
            
            return thinking_config
            
        except Exception as e:
            logger.error(
                "Error analyzing thinking needs",
                error=str(e),
                exc_info=True
            )
            return None
    
    async def enable_extended_thinking(
        self,
        agent_id: str,
        workflow_id: str,
        problem_description: str,
        problem_context: Dict[str, Any],
        thinking_depth: ThinkingDepth = ThinkingDepth.STANDARD
    ) -> ThinkingSession:
        """
        Enable extended thinking for an agent on a complex task.
        
        Args:
            agent_id: Primary agent ID
            workflow_id: Workflow ID
            problem_description: Description of the problem
            problem_context: Problem context and data
            thinking_depth: Depth of thinking required
            
        Returns:
            ThinkingSession instance
        """
        try:
            # Create thinking session
            session = ThinkingSession(
                workflow_id=workflow_id,
                agent_id=agent_id,
                thinking_depth=thinking_depth,
                problem_description=problem_description,
                context=problem_context,
                thinking_time_limit_seconds=self._get_thinking_time_limit(thinking_depth)
            )
            
            # Register session
            self.active_sessions[session.session_id] = session
            self.performance_stats["sessions_created"] += 1
            
            # Determine participating agents for collaborative thinking
            if thinking_depth in [ThinkingDepth.COLLABORATIVE, ThinkingDepth.INTENSIVE]:
                participating_agents = await self._select_participating_agents(
                    problem_description, problem_context, agent_id
                )
                session.participating_agents = participating_agents
                
                if len(participating_agents) > 1:
                    self.performance_stats["collaborative_sessions"] += 1
            
            # Notify communication bus
            if self.communication_bus:
                # Note: MessageBroker API might be different
                # This would need to be adapted based on actual implementation
                pass
            
            logger.info(
                "🧠 Extended thinking session started",
                session_id=session.session_id,
                thinking_depth=thinking_depth.value,
                participating_agents=len(session.participating_agents),
                time_limit_seconds=session.thinking_time_limit_seconds
            )
            
            return session
            
        except Exception as e:
            self.performance_stats["sessions_failed"] += 1
            logger.error(
                "Failed to enable extended thinking",
                agent_id=agent_id,
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _get_thinking_time_limit(self, thinking_depth: ThinkingDepth) -> int:
        """Get thinking time limit based on depth."""
        time_limits = {
            ThinkingDepth.STANDARD: 600,      # 10 minutes
            ThinkingDepth.DEEP: 1200,         # 20 minutes
            ThinkingDepth.COLLABORATIVE: 1800, # 30 minutes
            ThinkingDepth.INTENSIVE: 2700     # 45 minutes
        }
        return time_limits.get(thinking_depth, 600)
    
    async def _select_participating_agents(
        self,
        problem_description: str,
        problem_context: Dict[str, Any],
        primary_agent_id: str
    ) -> List[str]:
        """Select agents to participate in collaborative thinking."""
        try:
            if not self.orchestrator:
                return [primary_agent_id]
            
            # Get available agents
            available_agents = await self.orchestrator.get_all_agents()
            
            # Analyze problem to determine required expertise
            required_expertise = self._analyze_required_expertise(problem_description, problem_context)
            
            # Select agents based on expertise match
            selected_agents = [primary_agent_id]
            
            for agent in available_agents:
                agent_id = agent.get("id", "")
                agent_role = agent.get("role", "")
                agent_capabilities = agent.get("capabilities", [])
                
                if agent_id == primary_agent_id:
                    continue
                
                # Check if agent has required expertise
                expertise_match = any(
                    expertise in agent_role.lower() or 
                    any(expertise in cap.lower() for cap in agent_capabilities)
                    for expertise in required_expertise
                )
                
                if expertise_match and len(selected_agents) < 4:  # Limit to 4 agents
                    selected_agents.append(agent_id)
            
            return selected_agents
            
        except Exception as e:
            logger.warning(
                "Failed to select participating agents",
                error=str(e)
            )
            return [primary_agent_id]
    
    def _analyze_required_expertise(
        self,
        problem_description: str,
        problem_context: Dict[str, Any]
    ) -> Set[str]:
        """Analyze problem to determine required expertise areas."""
        expertise_keywords = {
            "security": ["security", "vulnerability", "authentication", "authorization", "encryption"],
            "performance": ["performance", "optimization", "speed", "bottleneck", "efficiency"],
            "architecture": ["architecture", "design", "structure", "pattern", "scalability"],
            "backend": ["api", "database", "server", "service", "backend"],
            "frontend": ["ui", "interface", "frontend", "user", "experience"],
            "devops": ["deployment", "infrastructure", "docker", "kubernetes", "ci/cd"],
            "testing": ["test", "testing", "quality", "validation", "verification"],
            "integration": ["integration", "communication", "protocol", "interface"]
        }
        
        combined_text = f"{problem_description.lower()} {json.dumps(problem_context, default=str).lower()}"
        
        required_expertise = set()
        
        for expertise, keywords in expertise_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                required_expertise.add(expertise)
        
        return required_expertise
    
    async def coordinate_thinking_session(
        self,
        session_id: str
    ) -> CollaborativeThinkingResult:
        """
        Coordinate a collaborative thinking session between multiple agents.
        
        Args:
            session_id: Thinking session ID
            
        Returns:
            CollaborativeThinkingResult with synthesized solution
        """
        start_time = time.time()
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Thinking session {session_id} not found")
            
            if len(session.participating_agents) <= 1:
                raise ValueError("Collaborative thinking requires multiple agents")
            
            logger.info(
                "🤝 Starting collaborative thinking session",
                session_id=session_id,
                participating_agents=session.participating_agents
            )
            
            # Coordinate individual thinking phases
            agent_contributions = {}
            thinking_tasks = []
            
            for agent_id in session.participating_agents:
                task = self._coordinate_agent_thinking(
                    agent_id, session, f"agent_{agent_id}"
                )
                thinking_tasks.append((agent_id, task))
            
            # Wait for all agents to complete their thinking
            for agent_id, task in thinking_tasks:
                try:
                    contribution = await asyncio.wait_for(task, timeout=session.thinking_time_limit_seconds)
                    agent_contributions[agent_id] = contribution
                except asyncio.TimeoutError:
                    logger.warning(
                        "Agent thinking timed out",
                        session_id=session_id,
                        agent_id=agent_id
                    )
                    agent_contributions[agent_id] = "Thinking session timed out"
            
            # Synthesize collaborative solution
            synthesis_result = await self._synthesize_collaborative_solution(
                session, agent_contributions
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create collaborative result
            result = CollaborativeThinkingResult(
                session_id=session_id,
                primary_solution=synthesis_result["solution"],
                agent_contributions=agent_contributions,
                consensus_score=synthesis_result["consensus_score"],
                synthesis_reasoning=synthesis_result["reasoning"],
                conflicting_viewpoints=synthesis_result["conflicts"],
                unified_implementation=synthesis_result["implementation_steps"],
                execution_time_seconds=execution_time
            )
            
            # Update session and statistics
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            self.session_history.append(session)
            del self.active_sessions[session_id]
            
            self.performance_stats["sessions_completed"] += 1
            self.performance_stats["total_thinking_time_seconds"] += execution_time
            
            # Notify completion
            if self.communication_bus:
                # Note: MessageBroker API might be different
                pass
            
            logger.info(
                "✅ Collaborative thinking session completed",
                session_id=session_id,
                consensus_score=result.consensus_score,
                execution_time_seconds=execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats["sessions_failed"] += 1
            
            logger.error(
                "❌ Collaborative thinking session failed",
                session_id=session_id,
                execution_time_seconds=execution_time,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _coordinate_agent_thinking(
        self,
        agent_id: str,
        session: ThinkingSession,
        correlation_id: str
    ) -> str:
        """Coordinate individual agent thinking within collaborative session."""
        try:
            if not self.orchestrator:
                return f"Agent {agent_id} thinking: Enhanced analysis of {session.problem_description}"
            
            # Create thinking task for agent
            thinking_task = {
                "type": "extended_thinking",
                "problem": session.problem_description,
                "context": session.context,
                "thinking_depth": session.thinking_depth.value,
                "correlation_id": correlation_id,
                "thinking_instructions": f"""
Please engage in extended thinking about this problem:

{session.problem_description}

Context: {json.dumps(session.context, indent=2, default=str)}

As an expert in your domain, provide:
1. Your analysis of the problem
2. Potential solutions from your perspective
3. Implementation considerations
4. Risks and challenges you foresee
5. Dependencies on other domains

Think deeply and consider multiple approaches before settling on your recommendation.
""",
                "session_id": session.session_id
            }
            
            # Execute thinking task
            result = await self.orchestrator.execute_agent_task(agent_id, thinking_task)
            
            return result.get("output", f"Agent {agent_id} completed thinking analysis")
            
        except Exception as e:
            logger.error(
                "Failed to coordinate agent thinking",
                agent_id=agent_id,
                session_id=session.session_id,
                error=str(e)
            )
            return f"Agent {agent_id} thinking failed: {str(e)}"
    
    async def _synthesize_collaborative_solution(
        self,
        session: ThinkingSession,
        agent_contributions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Synthesize individual agent contributions into unified solution."""
        try:
            # Analyze contributions for consensus and conflicts
            contributions_text = "\n\n".join([
                f"Agent {agent_id}: {contribution}"
                for agent_id, contribution in agent_contributions.items()
            ])
            
            # Simple synthesis algorithm (can be enhanced with AI)
            common_themes = self._extract_common_themes(agent_contributions)
            conflicting_points = self._identify_conflicts(agent_contributions)
            
            # Generate unified solution
            solution_parts = []
            
            if common_themes:
                solution_parts.append("Based on collaborative analysis, the following approach is recommended:")
                solution_parts.extend([f"- {theme}" for theme in common_themes])
            
            if conflicting_points:
                solution_parts.append("\nConflicting viewpoints to consider:")
                solution_parts.extend([f"- {conflict}" for conflict in conflicting_points])
            
            solution = "\n".join(solution_parts)
            
            # Calculate consensus score
            consensus_score = max(0.0, 1.0 - (len(conflicting_points) / max(len(common_themes), 1)))
            
            # Generate implementation steps
            implementation_steps = [
                "Review collaborative analysis and consensus points",
                "Address conflicting viewpoints through further analysis",
                "Create detailed implementation plan",
                "Identify required resources and dependencies",
                "Execute implementation with monitoring"
            ]
            
            return {
                "solution": solution,
                "reasoning": f"Synthesized from {len(agent_contributions)} agent perspectives",
                "consensus_score": consensus_score,
                "conflicts": conflicting_points,
                "implementation_steps": implementation_steps
            }
            
        except Exception as e:
            logger.error(
                "Failed to synthesize collaborative solution",
                session_id=session.session_id,
                error=str(e)
            )
            
            return {
                "solution": f"Synthesis failed: {str(e)}",
                "reasoning": "Error in synthesis process",
                "consensus_score": 0.0,
                "conflicts": ["Synthesis error"],
                "implementation_steps": ["Retry synthesis process"]
            }
    
    def _extract_common_themes(self, contributions: Dict[str, str]) -> List[str]:
        """Extract common themes from agent contributions."""
        # Simple keyword-based theme extraction
        common_keywords = ["recommend", "suggest", "propose", "implement", "design", "use"]
        themes = []
        
        for keyword in common_keywords:
            matching_contributions = [
                contrib for contrib in contributions.values()
                if keyword in contrib.lower()
            ]
            
            if len(matching_contributions) >= len(contributions) // 2:  # Majority agreement
                themes.append(f"Multiple agents {keyword} similar approaches")
        
        return themes[:5]  # Limit to top 5 themes
    
    def _identify_conflicts(self, contributions: Dict[str, str]) -> List[str]:
        """Identify conflicting viewpoints in agent contributions."""
        # Simple conflict detection based on opposing keywords
        conflict_pairs = [
            ("recommend", "avoid"),
            ("prefer", "reject"),
            ("should", "should not"),
            ("yes", "no"),
            ("use", "don't use")
        ]
        
        conflicts = []
        
        for pos_keyword, neg_keyword in conflict_pairs:
            pos_agents = [
                agent_id for agent_id, contrib in contributions.items()
                if pos_keyword in contrib.lower()
            ]
            neg_agents = [
                agent_id for agent_id, contrib in contributions.items()
                if neg_keyword in contrib.lower()
            ]
            
            if pos_agents and neg_agents:
                conflicts.append(f"Disagreement on {pos_keyword}/{neg_keyword} approach")
        
        return conflicts[:3]  # Limit to top 3 conflicts
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of thinking session."""
        session = self.active_sessions.get(session_id)
        if not session:
            # Check session history
            for historical_session in self.session_history:
                if historical_session.session_id == session_id:
                    return {
                        "session_id": session_id,
                        "status": historical_session.status,
                        "completed_at": historical_session.completed_at.isoformat() if historical_session.completed_at else None,
                        "thinking_depth": historical_session.thinking_depth.value,
                        "participating_agents": historical_session.participating_agents
                    }
            return None
        
        elapsed_time = (datetime.utcnow() - session.started_at).total_seconds()
        
        return {
            "session_id": session_id,
            "status": session.status,
            "thinking_depth": session.thinking_depth.value,
            "participating_agents": session.participating_agents,
            "elapsed_time_seconds": elapsed_time,
            "time_limit_seconds": session.thinking_time_limit_seconds,
            "progress": min(elapsed_time / session.thinking_time_limit_seconds, 1.0)
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_confidence = 0.0
        if self.performance_stats["sessions_completed"] > 0:
            # This would be calculated from actual session results
            avg_confidence = 0.85  # Placeholder
        
        return {
            "performance_stats": {
                **self.performance_stats,
                "average_confidence_score": avg_confidence,
                "success_rate": (
                    self.performance_stats["sessions_completed"] /
                    max(self.performance_stats["sessions_created"], 1) * 100
                ),
                "average_thinking_time_seconds": (
                    self.performance_stats["total_thinking_time_seconds"] /
                    max(self.performance_stats["sessions_completed"], 1)
                ) if self.performance_stats["sessions_completed"] > 0 else 0
            },
            "active_sessions": len(self.active_sessions),
            "session_history_count": len(self.session_history),
            "thinking_triggers": list(self.thinking_triggers.keys()),
            "capabilities": {
                "collaborative_thinking": True,
                "multi_agent_coordination": self.orchestrator is not None,
                "real_time_communication": self.communication_bus is not None,
                "thinking_depth_levels": list(ThinkingDepth),
                "supported_triggers": list(ThinkingTrigger)
            }
        }


# Global extended thinking engine instance
_extended_thinking_engine: Optional[ExtendedThinkingEngine] = None


def get_extended_thinking_engine() -> Optional[ExtendedThinkingEngine]:
    """Get the global extended thinking engine instance."""
    return _extended_thinking_engine


def set_extended_thinking_engine(engine: ExtendedThinkingEngine) -> None:
    """Set the global extended thinking engine instance."""
    global _extended_thinking_engine
    _extended_thinking_engine = engine
    logger.info("🔗 Global extended thinking engine set")


async def initialize_extended_thinking_engine(
    orchestrator: Optional[AgentOrchestrator] = None,
    communication_bus: Optional[MessageBroker] = None
) -> ExtendedThinkingEngine:
    """
    Initialize and set the global extended thinking engine.
    
    Args:
        orchestrator: Agent orchestrator instance
        communication_bus: Communication bus instance
        
    Returns:
        ExtendedThinkingEngine instance
    """
    engine = ExtendedThinkingEngine(
        orchestrator=orchestrator,
        communication_bus=communication_bus
    )
    
    set_extended_thinking_engine(engine)
    
    logger.info("✅ Extended thinking engine initialized")
    return engine