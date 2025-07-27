"""
Self-Improving Agent System for LeanVibe Agent Hive 2.0

This is the revolutionary component that enables agents to learn, evolve,
and improve their capabilities autonomously. Agents analyze their performance,
modify their own prompts, and develop new skills over time.

CRITICAL: This system implements safety guards to ensure beneficial evolution.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .redis import get_message_broker
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus
from ..models.context import Context, ContextType
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class ImprovementType(Enum):
    """Types of agent improvements."""
    PROMPT_OPTIMIZATION = "prompt_optimization"
    CAPABILITY_EXPANSION = "capability_expansion"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    COLLABORATION_SKILL = "collaboration_skill"


class SafetyLevel(Enum):
    """Safety levels for agent modifications."""
    SAFE = "safe"           # Minor improvements, low risk
    MODERATE = "moderate"   # Significant changes, moderate risk
    HIGH_RISK = "high_risk" # Major modifications, requires human approval
    RESTRICTED = "restricted" # Not allowed without explicit permission


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for an agent."""
    agent_id: str
    time_period: timedelta
    
    # Task Performance
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    average_completion_time: float
    
    # Code Quality (when applicable)
    code_quality_score: float
    test_coverage: float
    bug_rate: float
    
    # Collaboration Metrics
    collaboration_effectiveness: float
    communication_clarity: float
    help_requests_made: int
    help_provided: int
    
    # Learning Metrics
    new_concepts_learned: int
    patterns_recognized: int
    improvements_suggested: int
    
    # Context Efficiency
    context_window_usage: float
    memory_consolidation_efficiency: float
    
    def overall_performance_score(self) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        weights = {
            'success_rate': 0.3,
            'code_quality_score': 0.2,
            'collaboration_effectiveness': 0.2,
            'efficiency': 0.2,
            'learning': 0.1
        }
        
        efficiency_score = 1.0 - (self.context_window_usage * 0.5)
        learning_score = min(1.0, (self.new_concepts_learned + self.patterns_recognized) / 10.0)
        
        return (
            weights['success_rate'] * self.success_rate +
            weights['code_quality_score'] * self.code_quality_score +
            weights['collaboration_effectiveness'] * self.collaboration_effectiveness +
            weights['efficiency'] * efficiency_score +
            weights['learning'] * learning_score
        )


@dataclass
class ImprovementProposal:
    """A proposed improvement to an agent."""
    id: str
    agent_id: str
    improvement_type: ImprovementType
    safety_level: SafetyLevel
    
    # What to change
    target_component: str  # "system_prompt", "capabilities", "config"
    current_value: Any
    proposed_value: Any
    
    # Justification
    reasoning: str
    expected_benefits: List[str]
    potential_risks: List[str]
    supporting_metrics: Dict[str, float]
    
    # Validation
    requires_human_approval: bool
    confidence_score: float
    estimated_impact: float
    
    created_at: datetime
    approved_at: Optional[datetime] = None
    implemented_at: Optional[datetime] = None


class PromptEvolutionEngine:
    """
    Engine for evolving agent system prompts based on performance analysis.
    
    This is the core of agent self-improvement - analyzing what works
    and automatically refining prompts for better performance.
    """
    
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client
    
    async def analyze_prompt_effectiveness(
        self,
        agent: Agent,
        performance_metrics: PerformanceMetrics,
        recent_tasks: List[Task]
    ) -> Dict[str, float]:
        """Analyze how effective the current prompt is."""
        
        # Extract patterns from successful vs failed tasks
        successful_tasks = [t for t in recent_tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in recent_tasks if t.status == TaskStatus.FAILED]
        
        analysis = {
            'clarity_score': 0.0,
            'specificity_score': 0.0,
            'effectiveness_score': 0.0,
            'adaptability_score': 0.0
        }
        
        if not recent_tasks:
            return analysis
        
        # Use Claude to analyze prompt effectiveness
        analysis_prompt = f"""
        Analyze the effectiveness of this agent's system prompt based on performance data:
        
        Agent Role: {agent.role}
        Current System Prompt: {agent.system_prompt}
        
        Performance Metrics:
        - Success Rate: {performance_metrics.success_rate:.2%}
        - Code Quality: {performance_metrics.code_quality_score:.2f}
        - Collaboration: {performance_metrics.collaboration_effectiveness:.2f}
        
        Recent Task Patterns:
        - Successful Tasks: {len(successful_tasks)}
        - Failed Tasks: {len(failed_tasks)}
        
        Analyze the prompt effectiveness in these dimensions:
        1. Clarity: How clear and unambiguous is the prompt?
        2. Specificity: How well does it guide specific behaviors?
        3. Effectiveness: How well does it lead to successful outcomes?
        4. Adaptability: How well does it handle varied scenarios?
        
        Return scores from 0.0 to 1.0 for each dimension.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            # Parse the response to extract scores
            # This would need more sophisticated parsing in production
            content = response.content[0].text
            
            # For now, use performance metrics as proxy
            analysis['clarity_score'] = min(1.0, performance_metrics.success_rate + 0.2)
            analysis['specificity_score'] = performance_metrics.code_quality_score
            analysis['effectiveness_score'] = performance_metrics.overall_performance_score()
            analysis['adaptability_score'] = min(1.0, performance_metrics.collaboration_effectiveness + 0.1)
            
        except Exception as e:
            logger.error("Failed to analyze prompt effectiveness", error=str(e))
        
        return analysis
    
    async def generate_prompt_improvements(
        self,
        agent: Agent,
        performance_metrics: PerformanceMetrics,
        effectiveness_analysis: Dict[str, float]
    ) -> List[ImprovementProposal]:
        """Generate specific prompt improvement proposals."""
        
        proposals = []
        
        # Identify areas needing improvement
        improvement_areas = []
        for dimension, score in effectiveness_analysis.items():
            if score < 0.7:  # Threshold for improvement
                improvement_areas.append(dimension)
        
        if not improvement_areas:
            return proposals
        
        # Generate improvement proposal using Claude
        improvement_prompt = f"""
        Generate an improved system prompt for this agent based on performance analysis:
        
        Current Agent Role: {agent.role}
        Current System Prompt: {agent.system_prompt}
        
        Performance Issues:
        - Areas needing improvement: {improvement_areas}
        - Current success rate: {performance_metrics.success_rate:.2%}
        - Current quality score: {performance_metrics.code_quality_score:.2f}
        
        Generate an improved system prompt that:
        1. Addresses the identified weaknesses
        2. Maintains the agent's core competencies
        3. Improves clarity and specificity
        4. Enhances collaboration capabilities
        
        Provide the improved prompt and explain the changes made.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": improvement_prompt}]
            )
            
            improved_prompt = response.content[0].text
            
            # Create improvement proposal
            proposal = ImprovementProposal(
                id=str(uuid.uuid4()),
                agent_id=agent.id,
                improvement_type=ImprovementType.PROMPT_OPTIMIZATION,
                safety_level=SafetyLevel.MODERATE,
                target_component="system_prompt",
                current_value=agent.system_prompt,
                proposed_value=improved_prompt,
                reasoning=f"Addressing performance issues in: {', '.join(improvement_areas)}",
                expected_benefits=[
                    "Improved task success rate",
                    "Better code quality",
                    "Enhanced collaboration"
                ],
                potential_risks=[
                    "Temporary performance dip during adaptation",
                    "Possible over-optimization for current tasks"
                ],
                supporting_metrics=effectiveness_analysis,
                requires_human_approval=True,  # Prompt changes are significant
                confidence_score=0.8,
                estimated_impact=0.15,  # Expected 15% improvement
                created_at=datetime.utcnow()
            )
            
            proposals.append(proposal)
            
        except Exception as e:
            logger.error("Failed to generate prompt improvements", error=str(e))
        
        return proposals


class CapabilityLearningEngine:
    """
    Engine for discovering and learning new capabilities based on project needs.
    
    Analyzes task requirements and failures to identify new skills agents should develop.
    """
    
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client
    
    async def identify_skill_gaps(
        self,
        agent: Agent,
        failed_tasks: List[Task]
    ) -> List[str]:
        """Identify skills the agent needs to develop."""
        
        if not failed_tasks:
            return []
        
        # Analyze failure patterns
        failure_analysis_prompt = f"""
        Analyze these failed tasks to identify skill gaps for this agent:
        
        Agent Role: {agent.role}
        Current Capabilities: {[cap.get('name', 'Unknown') for cap in agent.capabilities or []]}
        
        Failed Tasks:
        {[f"- {task.title}: {task.error_message}" for task in failed_tasks[:10]]}
        
        Identify specific skills or capabilities this agent needs to develop to handle these tasks better.
        Focus on learnable skills rather than fundamental limitations.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": failure_analysis_prompt}]
            )
            
            # Parse response to extract skill gaps
            # This would need more sophisticated parsing
            content = response.content[0].text
            
            # For now, return some common skill gaps based on role
            skill_gaps = []
            if agent.role == "backend_developer":
                skill_gaps = ["api_testing", "database_optimization", "caching_strategies"]
            elif agent.role == "frontend_developer":
                skill_gaps = ["responsive_design", "accessibility", "performance_optimization"]
            
            return skill_gaps
            
        except Exception as e:
            logger.error("Failed to identify skill gaps", error=str(e))
            return []
    
    async def generate_capability_expansion(
        self,
        agent: Agent,
        skill_gaps: List[str]
    ) -> List[ImprovementProposal]:
        """Generate proposals for expanding agent capabilities."""
        
        proposals = []
        
        for skill in skill_gaps:
            # Generate new capability definition
            capability_prompt = f"""
            Define a new capability for this agent:
            
            Agent Role: {agent.role}
            Skill to Add: {skill}
            
            Create a capability definition with:
            1. Name: Clear, descriptive name
            2. Description: What this capability enables
            3. Confidence Level: Initial confidence (0.1-0.3 for new skills)
            4. Specialization Areas: Related sub-skills
            
            Format as JSON with these fields.
            """
            
            try:
                response = await self.anthropic.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=500,
                    messages=[{"role": "user", "content": capability_prompt}]
                )
                
                # Create new capability (simplified for demo)
                new_capability = {
                    "name": skill,
                    "description": f"Capability to handle {skill} tasks",
                    "confidence_level": 0.2,  # Low initial confidence
                    "specialization_areas": [skill]
                }
                
                proposal = ImprovementProposal(
                    id=str(uuid.uuid4()),
                    agent_id=agent.id,
                    improvement_type=ImprovementType.CAPABILITY_EXPANSION,
                    safety_level=SafetyLevel.SAFE,
                    target_component="capabilities",
                    current_value=agent.capabilities or [],
                    proposed_value=(agent.capabilities or []) + [new_capability],
                    reasoning=f"Agent needs {skill} capability to handle failed tasks",
                    expected_benefits=[f"Ability to handle {skill} tasks"],
                    potential_risks=["Initial low performance until skill develops"],
                    supporting_metrics={"failure_rate_in_area": 0.8},
                    requires_human_approval=False,  # New capabilities are generally safe
                    confidence_score=0.7,
                    estimated_impact=0.1,
                    created_at=datetime.utcnow()
                )
                
                proposals.append(proposal)
                
            except Exception as e:
                logger.error(f"Failed to generate capability for {skill}", error=str(e))
        
        return proposals


class SelfImprovementOrchestrator:
    """
    Main orchestrator for agent self-improvement.
    
    Coordinates performance analysis, improvement generation, safety validation,
    and implementation of agent enhancements.
    """
    
    def __init__(self):
        self.anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.prompt_engine = PromptEvolutionEngine(self.anthropic)
        self.capability_engine = CapabilityLearningEngine(self.anthropic)
    
    async def analyze_agent_performance(
        self,
        agent_id: str,
        days_back: int = 7
    ) -> PerformanceMetrics:
        """Analyze an agent's performance over a time period."""
        
        async with get_session() as db_session:
            # Get agent
            agent = await db_session.get(Agent, agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Get recent tasks
            from sqlalchemy import select
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            tasks_query = select(Task).where(
                Task.assigned_agent_id == agent_id,
                Task.created_at >= cutoff_date
            )
            tasks_result = await db_session.execute(tasks_query)
            recent_tasks = tasks_result.scalars().all()
            
            # Calculate metrics
            total_tasks = len(recent_tasks)
            completed_tasks = len([t for t in recent_tasks if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in recent_tasks if t.status == TaskStatus.FAILED])
            
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
            
            # Calculate average completion time
            completed_task_times = [
                (t.completed_at - t.started_at).total_seconds() / 60
                for t in recent_tasks
                if t.completed_at and t.started_at
            ]
            avg_completion_time = sum(completed_task_times) / len(completed_task_times) if completed_task_times else 0.0
            
            return PerformanceMetrics(
                agent_id=agent_id,
                time_period=timedelta(days=days_back),
                tasks_completed=completed_tasks,
                tasks_failed=failed_tasks,
                success_rate=success_rate,
                average_completion_time=avg_completion_time,
                code_quality_score=0.8,  # Would be calculated from actual code analysis
                test_coverage=0.75,      # Would be measured from test results
                bug_rate=0.1,           # Would be tracked from production issues
                collaboration_effectiveness=0.8,  # Would be measured from team feedback
                communication_clarity=0.85,       # Would be assessed from message quality
                help_requests_made=2,             # Would be tracked from system interactions
                help_provided=5,                  # Would be tracked from assistance given
                new_concepts_learned=3,           # Would be tracked from context analysis
                patterns_recognized=7,            # Would be tracked from pattern detection
                improvements_suggested=2,         # Would be tracked from suggestions made
                context_window_usage=float(agent.context_window_usage or 0.5),
                memory_consolidation_efficiency=0.9  # Would be measured from sleep-wake cycles
            )
    
    async def generate_improvement_proposals(
        self,
        agent_id: str
    ) -> List[ImprovementProposal]:
        """Generate comprehensive improvement proposals for an agent."""
        
        # Analyze performance
        performance = await self.analyze_agent_performance(agent_id)
        
        async with get_session() as db_session:
            agent = await db_session.get(Agent, agent_id)
            if not agent:
                return []
            
            # Get recent tasks for analysis
            from sqlalchemy import select
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            tasks_query = select(Task).where(
                Task.assigned_agent_id == agent_id,
                Task.created_at >= cutoff_date
            )
            tasks_result = await db_session.execute(tasks_query)
            recent_tasks = tasks_result.scalars().all()
            
            failed_tasks = [t for t in recent_tasks if t.status == TaskStatus.FAILED]
            
            proposals = []
            
            # Generate prompt improvements
            if performance.overall_performance_score() < 0.8:
                effectiveness = await self.prompt_engine.analyze_prompt_effectiveness(
                    agent, performance, recent_tasks
                )
                prompt_proposals = await self.prompt_engine.generate_prompt_improvements(
                    agent, performance, effectiveness
                )
                proposals.extend(prompt_proposals)
            
            # Generate capability expansions
            if failed_tasks:
                skill_gaps = await self.capability_engine.identify_skill_gaps(agent, failed_tasks)
                capability_proposals = await self.capability_engine.generate_capability_expansion(
                    agent, skill_gaps
                )
                proposals.extend(capability_proposals)
            
            logger.info(
                "Generated improvement proposals",
                agent_id=agent_id,
                proposal_count=len(proposals),
                performance_score=performance.overall_performance_score()
            )
            
            return proposals
    
    async def validate_improvement_safety(
        self,
        proposal: ImprovementProposal
    ) -> Tuple[bool, List[str]]:
        """Validate that an improvement proposal is safe to implement."""
        
        warnings = []
        
        # Check safety level
        if proposal.safety_level == SafetyLevel.RESTRICTED:
            return False, ["Restricted modification type"]
        
        # Check confidence threshold
        if proposal.confidence_score < 0.5:
            warnings.append("Low confidence score")
        
        # Check for risky changes
        if proposal.improvement_type == ImprovementType.PROMPT_OPTIMIZATION:
            if len(proposal.proposed_value) > len(proposal.current_value) * 2:
                warnings.append("Dramatic prompt length increase")
        
        # For capability expansion, check for conflicts
        if proposal.improvement_type == ImprovementType.CAPABILITY_EXPANSION:
            current_caps = proposal.current_value or []
            new_caps = proposal.proposed_value or []
            
            if len(new_caps) > len(current_caps) + 3:
                warnings.append("Adding too many capabilities at once")
        
        # Determine if safe to proceed
        is_safe = (
            proposal.safety_level in [SafetyLevel.SAFE, SafetyLevel.MODERATE] and
            proposal.confidence_score >= 0.6 and
            len(warnings) <= 2
        )
        
        return is_safe, warnings
    
    async def implement_improvement(
        self,
        proposal: ImprovementProposal
    ) -> bool:
        """Implement an approved improvement proposal."""
        
        async with get_session() as db_session:
            agent = await db_session.get(Agent, proposal.agent_id)
            if not agent:
                return False
            
            try:
                # Apply the improvement based on type
                if proposal.target_component == "system_prompt":
                    agent.system_prompt = proposal.proposed_value
                elif proposal.target_component == "capabilities":
                    agent.capabilities = proposal.proposed_value
                elif proposal.target_component == "config":
                    agent.config = {**(agent.config or {}), **proposal.proposed_value}
                
                # Update timestamp
                agent.updated_at = datetime.utcnow()
                
                await db_session.commit()
                
                # Log the improvement
                logger.info(
                    "Implemented agent improvement",
                    agent_id=proposal.agent_id,
                    improvement_type=proposal.improvement_type.value,
                    component=proposal.target_component
                )
                
                # Store implementation record
                proposal.implemented_at = datetime.utcnow()
                
                return True
                
            except Exception as e:
                logger.error(
                    "Failed to implement improvement",
                    agent_id=proposal.agent_id,
                    error=str(e)
                )
                await db_session.rollback()
                return False
    
    async def run_improvement_cycle(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """Run a complete improvement cycle for an agent."""
        
        try:
            # Generate proposals
            proposals = await self.generate_improvement_proposals(agent_id)
            
            implemented_count = 0
            pending_approval = 0
            rejected_count = 0
            
            for proposal in proposals:
                # Validate safety
                is_safe, warnings = await self.validate_improvement_safety(proposal)
                
                if not is_safe:
                    rejected_count += 1
                    logger.warning(
                        "Rejected unsafe improvement proposal",
                        agent_id=agent_id,
                        proposal_id=proposal.id,
                        warnings=warnings
                    )
                    continue
                
                # Check if requires human approval
                if proposal.requires_human_approval:
                    pending_approval += 1
                    logger.info(
                        "Improvement proposal requires human approval",
                        agent_id=agent_id,
                        proposal_id=proposal.id,
                        type=proposal.improvement_type.value
                    )
                    # In production, this would be queued for human review
                    continue
                
                # Implement safe, auto-approved improvements
                success = await self.implement_improvement(proposal)
                if success:
                    implemented_count += 1
            
            result = {
                "agent_id": agent_id,
                "proposals_generated": len(proposals),
                "implemented": implemented_count,
                "pending_approval": pending_approval,
                "rejected": rejected_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Completed improvement cycle", **result)
            return result
            
        except Exception as e:
            logger.error(
                "Improvement cycle failed",
                agent_id=agent_id,
                error=str(e)
            )
            return {
                "agent_id": agent_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global instance
self_improvement_orchestrator = SelfImprovementOrchestrator()