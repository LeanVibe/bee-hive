"""
Dashboard Development Context Sharing Protocol

Intelligent context sharing system for multi-agent dashboard development team
enabling shared architectural decisions, implementation progress, and knowledge transfer.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
from redis.asyncio import Redis as AsyncRedis
import hashlib


class ContextType(Enum):
    """Types of shared context."""
    ARCHITECTURAL_DECISION = "architectural_decision"
    IMPLEMENTATION_PROGRESS = "implementation_progress"
    TECHNICAL_SPECIFICATION = "technical_specification"
    QUALITY_METRIC = "quality_metric"
    INTEGRATION_REQUIREMENT = "integration_requirement"
    SECURITY_CONSTRAINT = "security_constraint"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    KNOWLEDGE_ARTIFACT = "knowledge_artifact"


class ContextScope(Enum):
    """Scope of context sharing."""
    GLOBAL = "global"              # Shared across all agents
    PHASE_SPECIFIC = "phase"       # Shared within development phase
    AGENT_PAIR = "agent_pair"      # Shared between two specific agents
    DOMAIN_SPECIFIC = "domain"     # Shared within domain (security, performance, etc.)


class ContextPriority(Enum):
    """Priority levels for context updates."""
    CRITICAL = "critical"          # Immediate attention required
    HIGH = "high"                  # Important for current work
    MEDIUM = "medium"              # Useful for coordination
    LOW = "low"                    # Background information


@dataclass
class ContextEntry:
    """Individual context entry."""
    context_id: str
    context_type: ContextType
    scope: ContextScope
    priority: ContextPriority
    title: str
    content: Dict[str, Any]
    created_by: str
    created_at: datetime
    updated_at: datetime
    version: int
    tags: List[str]
    related_contexts: List[str] = field(default_factory=list)
    access_pattern: str = "read_write"  # read_only, read_write, append_only
    expiry_at: Optional[datetime] = None
    
    def get_content_hash(self) -> str:
        """Generate hash of content for change detection."""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass 
class ArchitecturalDecision:
    """Architectural decision record."""
    decision_id: str
    title: str
    status: str  # proposed, accepted, rejected, superseded
    context: str
    decision: str
    consequences: List[str]
    alternatives_considered: List[str]
    decided_by: str
    decided_at: datetime
    affects_agents: List[str]
    implementation_notes: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationProgress:
    """Implementation progress tracking."""
    task_id: str
    agent_id: str
    component: str
    progress_percent: int
    status: str
    milestones_completed: List[str]
    current_milestone: str
    blockers: List[str]
    dependencies_status: Dict[str, str]
    integration_points: List[str]
    quality_metrics: Dict[str, Any]
    estimated_completion: datetime
    last_updated: datetime


@dataclass
class TechnicalSpecification:
    """Technical specification for components."""
    spec_id: str
    component_name: str
    specification: Dict[str, Any]
    interfaces: Dict[str, Any]
    constraints: List[str]
    quality_requirements: Dict[str, Any]
    integration_requirements: List[str]
    testing_requirements: List[str]
    created_by: str
    approved_by: List[str]
    version: str


class DashboardContextSharingProtocol:
    """
    Context sharing protocol for multi-agent dashboard development.
    
    Manages shared architectural decisions, implementation progress,
    technical specifications, and knowledge artifacts across all agents.
    """
    
    def __init__(self, redis_client: AsyncRedis, session_id: str):
        self.redis = redis_client
        self.session_id = session_id
        
        # Context storage patterns
        self.context_keys = {
            "global": f"dashboard_context:{session_id}:global",
            "phase": f"dashboard_context:{session_id}:phase:{{phase_id}}",
            "agent_pair": f"dashboard_context:{session_id}:pair:{{agent1}}:{{agent2}}",
            "domain": f"dashboard_context:{session_id}:domain:{{domain}}"
        }
        
        # Context change streams
        self.change_streams = {
            "context_updates": f"dashboard_context:{session_id}:updates",
            "decisions": f"dashboard_context:{session_id}:decisions",
            "progress": f"dashboard_context:{session_id}:progress",
            "specifications": f"dashboard_context:{session_id}:specifications"
        }
        
        # Agent specialization domains
        self.agent_domains = {
            "dashboard-architect": ["architecture", "integration", "design_patterns"],
            "frontend-developer": ["ui_components", "user_experience", "responsive_design"],
            "api-integration": ["api_design", "websocket_protocols", "data_flow"],
            "security-specialist": ["authentication", "authorization", "security_compliance"],
            "performance-engineer": ["performance_metrics", "optimization", "monitoring"],
            "qa-validator": ["testing_strategies", "quality_gates", "compliance_validation"]
        }
    
    async def initialize_context_sharing(self) -> str:
        """Initialize context sharing system."""
        initialization_context = ContextEntry(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.ARCHITECTURAL_DECISION,
            scope=ContextScope.GLOBAL,
            priority=ContextPriority.HIGH,
            title="Dashboard Development Context Sharing Initialization",
            content={
                "session_id": self.session_id,
                "initialized_at": datetime.now(timezone.utc).isoformat(),
                "agents": list(self.agent_domains.keys()),
                "context_types": [ct.value for ct in ContextType],
                "sharing_protocols": {
                    "architectural_decisions": "consensus_required",
                    "implementation_progress": "real_time_updates",
                    "technical_specifications": "version_controlled",
                    "quality_metrics": "automated_collection"
                }
            },
            created_by="orchestrator",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["initialization", "protocol", "multi_agent"]
        )
        
        await self._store_context(initialization_context)
        return initialization_context.context_id
    
    async def share_architectural_decision(self, decision: ArchitecturalDecision) -> str:
        """Share architectural decision with relevant agents."""
        context_entry = ContextEntry(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.ARCHITECTURAL_DECISION,
            scope=ContextScope.GLOBAL,
            priority=ContextPriority.CRITICAL,
            title=f"Architectural Decision: {decision.title}",
            content=asdict(decision),
            created_by=decision.decided_by,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["architecture", "decision", "critical"]
        )
        
        await self._store_context(context_entry)
        
        # Notify affected agents
        await self._notify_agents(
            agents=decision.affects_agents,
            context_id=context_entry.context_id,
            message=f"New architectural decision: {decision.title}",
            priority=ContextPriority.CRITICAL
        )
        
        return context_entry.context_id
    
    async def update_implementation_progress(self, progress: ImplementationProgress) -> str:
        """Update implementation progress and share with coordinating agents."""
        context_entry = ContextEntry(
            context_id=f"progress_{progress.task_id}_{progress.agent_id}",
            context_type=ContextType.IMPLEMENTATION_PROGRESS,
            scope=ContextScope.GLOBAL,
            priority=ContextPriority.MEDIUM,
            title=f"Progress: {progress.component} ({progress.progress_percent}%)",
            content=asdict(progress),
            created_by=progress.agent_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["progress", progress.agent_id, progress.component]
        )
        
        await self._store_context(context_entry)
        
        # Send progress update to coordination stream
        await self.redis.xadd(
            self.change_streams["progress"],
            {
                "context_id": context_entry.context_id,
                "agent_id": progress.agent_id,
                "component": progress.component,
                "progress_percent": progress.progress_percent,
                "status": progress.status,
                "blockers": json.dumps(progress.blockers),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return context_entry.context_id
    
    async def share_technical_specification(self, spec: TechnicalSpecification) -> str:
        """Share technical specification with relevant agents."""
        context_entry = ContextEntry(
            context_id=spec.spec_id,
            context_type=ContextType.TECHNICAL_SPECIFICATION,
            scope=ContextScope.GLOBAL,
            priority=ContextPriority.HIGH,
            title=f"Technical Spec: {spec.component_name}",
            content=asdict(spec),
            created_by=spec.created_by,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["specification", "technical", spec.component_name]
        )
        
        await self._store_context(context_entry)
        
        # Determine which agents need this specification
        relevant_agents = self._determine_relevant_agents(spec.component_name, spec.integration_requirements)
        
        await self._notify_agents(
            agents=relevant_agents,
            context_id=context_entry.context_id,
            message=f"New technical specification: {spec.component_name}",
            priority=ContextPriority.HIGH
        )
        
        return context_entry.context_id
    
    async def get_context_for_agent(self, agent_id: str, context_types: List[ContextType] = None) -> List[ContextEntry]:
        """Get relevant context for specific agent."""
        relevant_contexts = []
        
        # Get global contexts
        global_contexts = await self._get_contexts_by_scope(ContextScope.GLOBAL)
        relevant_contexts.extend(global_contexts)
        
        # Get domain-specific contexts
        agent_domains = self.agent_domains.get(agent_id, [])
        for domain in agent_domains:
            domain_contexts = await self._get_contexts_by_domain(domain)
            relevant_contexts.extend(domain_contexts)
        
        # Get agent-pair contexts (contexts shared with this agent specifically)
        pair_contexts = await self._get_agent_pair_contexts(agent_id)
        relevant_contexts.extend(pair_contexts)
        
        # Filter by context types if specified
        if context_types:
            relevant_contexts = [
                ctx for ctx in relevant_contexts 
                if ctx.context_type in context_types
            ]
        
        # Sort by priority and recency
        relevant_contexts.sort(
            key=lambda x: (x.priority.value, x.updated_at),
            reverse=True
        )
        
        return relevant_contexts
    
    async def create_integration_context(self, agent1: str, agent2: str, 
                                       integration_spec: Dict[str, Any]) -> str:
        """Create shared context for agent-to-agent integration."""
        context_entry = ContextEntry(
            context_id=f"integration_{agent1}_{agent2}_{uuid.uuid4().hex[:8]}",
            context_type=ContextType.INTEGRATION_REQUIREMENT,
            scope=ContextScope.AGENT_PAIR,
            priority=ContextPriority.HIGH,
            title=f"Integration: {agent1} ↔ {agent2}",
            content={
                "agent1": agent1,
                "agent2": agent2,
                "integration_spec": integration_spec,
                "status": "planned",
                "coordination_channel": f"dashboard_dev:integration:{agent1}_{agent2}",
                "dependencies": integration_spec.get("dependencies", []),
                "interfaces": integration_spec.get("interfaces", {}),
                "quality_requirements": integration_spec.get("quality_requirements", {}),
                "testing_strategy": integration_spec.get("testing_strategy", {})
            },
            created_by="orchestrator",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["integration", agent1, agent2]
        )
        
        # Store in agent-pair context
        await self._store_agent_pair_context(agent1, agent2, context_entry)
        
        # Notify both agents
        await self._notify_agents(
            agents=[agent1, agent2],
            context_id=context_entry.context_id,
            message=f"New integration requirement: {agent1} ↔ {agent2}",
            priority=ContextPriority.HIGH
        )
        
        return context_entry.context_id
    
    async def update_quality_metrics(self, agent_id: str, component: str, 
                                   metrics: Dict[str, Any]) -> str:
        """Update quality metrics and share with QA validator."""
        context_entry = ContextEntry(
            context_id=f"quality_{component}_{agent_id}_{int(datetime.now(timezone.utc).timestamp())}",
            context_type=ContextType.QUALITY_METRIC,
            scope=ContextScope.DOMAIN_SPECIFIC,
            priority=ContextPriority.MEDIUM,
            title=f"Quality Metrics: {component}",
            content={
                "agent_id": agent_id,
                "component": component,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trend_analysis": self._calculate_metric_trends(component, metrics)
            },
            created_by=agent_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            tags=["quality", "metrics", component, agent_id]
        )
        
        await self._store_context(context_entry)
        
        # Always share quality metrics with QA validator
        await self._notify_agents(
            agents=["qa-validator"],
            context_id=context_entry.context_id,
            message=f"Quality metrics update: {component}",
            priority=ContextPriority.MEDIUM
        )
        
        return context_entry.context_id
    
    async def search_context(self, query: str, agent_id: str) -> List[ContextEntry]:
        """Search context entries relevant to agent."""
        # Get all contexts for agent
        agent_contexts = await self.get_context_for_agent(agent_id)
        
        # Simple text search in titles and content
        query_lower = query.lower()
        matching_contexts = []
        
        for context in agent_contexts:
            # Search in title
            if query_lower in context.title.lower():
                matching_contexts.append(context)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in context.tags):
                matching_contexts.append(context)
                continue
            
            # Search in content (basic string search)
            content_str = json.dumps(context.content).lower()
            if query_lower in content_str:
                matching_contexts.append(context)
        
        return matching_contexts
    
    async def get_agent_knowledge_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive knowledge summary for agent."""
        contexts = await self.get_context_for_agent(agent_id)
        
        summary = {
            "agent_id": agent_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_contexts": len(contexts),
            "context_breakdown": {},
            "recent_decisions": [],
            "current_specifications": [],
            "integration_requirements": [],
            "quality_status": {},
            "knowledge_gaps": []
        }
        
        # Breakdown by context type
        for context_type in ContextType:
            type_contexts = [ctx for ctx in contexts if ctx.context_type == context_type]
            summary["context_breakdown"][context_type.value] = len(type_contexts)
        
        # Recent architectural decisions
        decision_contexts = [
            ctx for ctx in contexts 
            if ctx.context_type == ContextType.ARCHITECTURAL_DECISION
        ][:5]
        summary["recent_decisions"] = [
            {
                "title": ctx.title,
                "created_at": ctx.created_at.isoformat(),
                "priority": ctx.priority.value
            }
            for ctx in decision_contexts
        ]
        
        # Current specifications
        spec_contexts = [
            ctx for ctx in contexts 
            if ctx.context_type == ContextType.TECHNICAL_SPECIFICATION
        ]
        summary["current_specifications"] = [
            {
                "title": ctx.title,
                "version": ctx.version,
                "updated_at": ctx.updated_at.isoformat()
            }
            for ctx in spec_contexts
        ]
        
        # Integration requirements
        integration_contexts = [
            ctx for ctx in contexts 
            if ctx.context_type == ContextType.INTEGRATION_REQUIREMENT
        ]
        summary["integration_requirements"] = [
            {
                "title": ctx.title,
                "priority": ctx.priority.value,
                "created_at": ctx.created_at.isoformat()
            }
            for ctx in integration_contexts
        ]
        
        return summary
    
    async def _store_context(self, context: ContextEntry) -> None:
        """Store context entry in Redis."""
        context_key = self._get_context_key(context.scope, context.context_id)
        context_data = asdict(context)
        
        # Serialize datetime objects
        context_data["created_at"] = context.created_at.isoformat()
        context_data["updated_at"] = context.updated_at.isoformat()
        if context.expiry_at:
            context_data["expiry_at"] = context.expiry_at.isoformat()
        
        # Store as hash
        await self.redis.hset(context_key, mapping=context_data)
        
        # Set expiry if specified
        if context.expiry_at:
            await self.redis.expireat(context_key, context.expiry_at)
        
        # Add to context updates stream
        await self.redis.xadd(
            self.change_streams["context_updates"],
            {
                "context_id": context.context_id,
                "context_type": context.context_type.value,
                "scope": context.scope.value,
                "created_by": context.created_by,
                "title": context.title,
                "priority": context.priority.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _store_agent_pair_context(self, agent1: str, agent2: str, context: ContextEntry) -> None:
        """Store context specific to agent pair."""
        pair_key = self.context_keys["agent_pair"].format(agent1=agent1, agent2=agent2)
        context_data = asdict(context)
        
        # Serialize datetime objects
        context_data["created_at"] = context.created_at.isoformat()
        context_data["updated_at"] = context.updated_at.isoformat()
        
        await self.redis.hset(f"{pair_key}:{context.context_id}", mapping=context_data)
    
    async def _notify_agents(self, agents: List[str], context_id: str, 
                           message: str, priority: ContextPriority) -> None:
        """Notify agents of context updates."""
        notification = {
            "context_id": context_id,
            "message": message,
            "priority": priority.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id
        }
        
        for agent_id in agents:
            agent_channel = f"dashboard_dev:{agent_id}_notifications"
            await self.redis.xadd(agent_channel, notification)
    
    def _get_context_key(self, scope: ContextScope, context_id: str) -> str:
        """Generate Redis key for context storage."""
        base_key = self.context_keys["global"]
        if scope == ContextScope.PHASE_SPECIFIC:
            base_key = self.context_keys["phase"].format(phase_id="current")
        elif scope == ContextScope.DOMAIN_SPECIFIC:
            base_key = self.context_keys["domain"].format(domain="general")
        
        return f"{base_key}:{context_id}"
    
    async def _get_contexts_by_scope(self, scope: ContextScope) -> List[ContextEntry]:
        """Get contexts by scope."""
        # This would typically scan Redis keys and reconstruct ContextEntry objects
        # For now, return empty list
        return []
    
    async def _get_contexts_by_domain(self, domain: str) -> List[ContextEntry]:
        """Get contexts by domain."""
        # This would typically query domain-specific contexts
        return []
    
    async def _get_agent_pair_contexts(self, agent_id: str) -> List[ContextEntry]:
        """Get contexts shared with specific agent."""
        # This would typically query agent-pair contexts
        return []
    
    def _determine_relevant_agents(self, component_name: str, integration_requirements: List[str]) -> List[str]:
        """Determine which agents are relevant for a component/specification."""
        relevant_agents = []
        
        # Basic mapping based on component type
        component_lower = component_name.lower()
        
        if any(keyword in component_lower for keyword in ["security", "jwt", "auth"]):
            relevant_agents.append("security-specialist")
        
        if any(keyword in component_lower for keyword in ["ui", "frontend", "component"]):
            relevant_agents.append("frontend-developer")
        
        if any(keyword in component_lower for keyword in ["api", "backend", "websocket"]):
            relevant_agents.append("api-integration")
        
        if any(keyword in component_lower for keyword in ["performance", "monitoring", "metrics"]):
            relevant_agents.append("performance-engineer")
        
        if any(keyword in component_lower for keyword in ["architecture", "design", "pattern"]):
            relevant_agents.append("dashboard-architect")
        
        # QA validator is always relevant for quality-related specifications
        relevant_agents.append("qa-validator")
        
        return list(set(relevant_agents))  # Remove duplicates
    
    def _calculate_metric_trends(self, component: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trends for quality metrics."""
        # This would typically compare with historical metrics
        # For now, return basic trend analysis
        return {
            "trend": "stable",
            "change_percent": 0,
            "recommendation": "Continue monitoring"
        }


# Example usage and testing
async def example_context_sharing():
    """Example usage of context sharing protocol."""
    redis_client = AsyncRedis(host='localhost', port=6379, db=0, decode_responses=True)
    
    context_protocol = DashboardContextSharingProtocol(
        redis_client=redis_client,
        session_id="dashboard_dev_session_001"
    )
    
    # Initialize context sharing
    init_id = await context_protocol.initialize_context_sharing()
    print(f"Initialized context sharing: {init_id}")
    
    # Share architectural decision
    decision = ArchitecturalDecision(
        decision_id=str(uuid.uuid4()),
        title="Use Lit Web Components for Dashboard UI",
        status="accepted",
        context="Need modern, maintainable UI framework for dashboard development",
        decision="Adopt Lit Web Components with TypeScript for all UI development",
        consequences=[
            "Improved component reusability",
            "Better TypeScript integration",
            "Smaller bundle size compared to React/Vue"
        ],
        alternatives_considered=["React", "Vue", "Angular", "Vanilla JS"],
        decided_by="dashboard-architect",
        decided_at=datetime.now(timezone.utc),
        affects_agents=["frontend-developer", "qa-validator"]
    )
    
    decision_id = await context_protocol.share_architectural_decision(decision)
    print(f"Shared architectural decision: {decision_id}")
    
    # Update implementation progress
    progress = ImplementationProgress(
        task_id="jwt_001",
        agent_id="security-specialist",
        component="JWT Authentication",
        progress_percent=75,
        status="in_progress",
        milestones_completed=["JWT library integration", "Token validation logic"],
        current_milestone="Integration testing",
        blockers=["Need test users for validation"],
        dependencies_status={"database_models": "complete", "api_endpoints": "in_progress"},
        integration_points=["github_integration_api", "user_authentication"],
        quality_metrics={"test_coverage": 85, "security_score": 95},
        estimated_completion=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    progress_id = await context_protocol.update_implementation_progress(progress)
    print(f"Updated implementation progress: {progress_id}")
    
    # Get agent knowledge summary
    summary = await context_protocol.get_agent_knowledge_summary("security-specialist")
    print(f"Agent knowledge summary: {json.dumps(summary, indent=2)}")
    
    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(example_context_sharing())