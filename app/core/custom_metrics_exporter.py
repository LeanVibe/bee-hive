"""
Custom Metrics Exporter for LeanVibe Agent Hive 2.0
Exports agent-specific metrics for Kubernetes HPA scaling decisions
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import aioredis
from prometheus_client import (
    CollectorRegistry, 
    Gauge, 
    Counter, 
    Histogram, 
    generate_latest,
    CONTENT_TYPE_LATEST
)
from prometheus_client.metrics import MetricWrapperBase
from fastapi import FastAPI, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.context import Context
from app.core.database import get_session

logger = logging.getLogger(__name__)

class AgentMetricType(str, Enum):
    """Types of agent-specific metrics"""
    TASK_QUEUE_DEPTH = "task_queue_depth"
    ACTIVE_SESSIONS = "active_sessions" 
    ARCHITECTURE_REQUESTS = "architecture_requests"
    DESIGN_COMPLEXITY_SCORE = "design_complexity_score"
    TEST_QUEUE_SIZE = "test_queue_size"
    REVIEW_BACKLOG_SIZE = "review_backlog_size"
    COORDINATION_COMPLEXITY = "coordination_complexity"
    CONTEXT_ENGINE_LOAD = "context_engine_load"
    PERFORMANCE_SCORE = "performance_score"

@dataclass
class MetricConfiguration:
    """Configuration for a specific metric"""
    name: str
    description: str
    metric_type: str  # gauge, counter, histogram
    labels: List[str]
    update_interval_seconds: int = 30
    enabled: bool = True

class CustomMetricsExporter:
    """Exports custom metrics for Kubernetes HPA and monitoring"""
    
    def __init__(self, redis_client: aioredis.Redis, db_session_factory):
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.running = False
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start background collection
        self.collection_task = None
    
    def _initialize_metrics(self):
        """Initialize all custom metrics"""
        
        # Agent task queue metrics
        self.metrics['agent_task_queue_depth'] = Gauge(
            'agent_task_queue_depth',
            'Number of pending tasks per agent',
            ['agent_type', 'agent_id', 'priority'],
            registry=self.registry
        )
        
        self.metrics['agent_active_sessions'] = Gauge(
            'agent_active_sessions',
            'Number of active development sessions per agent',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        # Architecture-specific metrics
        self.metrics['agent_architecture_requests'] = Gauge(
            'agent_architecture_requests',
            'Number of architecture planning requests per agent',
            ['agent_type', 'agent_id', 'complexity_level'],
            registry=self.registry
        )
        
        self.metrics['agent_design_complexity_score'] = Gauge(
            'agent_design_complexity_score',
            'Current design complexity score (0-100)',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        # QA-specific metrics
        self.metrics['agent_test_queue_size'] = Gauge(
            'agent_test_queue_size',
            'Number of tests in queue per QA agent',
            ['agent_type', 'agent_id', 'test_type'],
            registry=self.registry
        )
        
        self.metrics['agent_review_backlog_size'] = Gauge(
            'agent_review_backlog_size',
            'Number of code reviews in backlog per agent',
            ['agent_type', 'agent_id', 'review_priority'],
            registry=self.registry
        )
        
        # Meta agent coordination metrics
        self.metrics['agent_coordination_complexity'] = Gauge(
            'agent_coordination_complexity',
            'Coordination complexity score for meta agents (0-100)',
            ['agent_id', 'coordination_type'],
            registry=self.registry
        )
        
        # Context engine metrics
        self.metrics['context_engine_processing_rate'] = Gauge(
            'context_engine_processing_rate',
            'Context engine processing rate (operations per second)',
            ['operation_type'],
            registry=self.registry
        )
        
        self.metrics['context_engine_memory_usage'] = Gauge(
            'context_engine_memory_usage',
            'Context engine memory usage in MB',
            ['memory_type'],
            registry=self.registry
        )
        
        # System-wide metrics
        self.metrics['total_active_agents'] = Gauge(
            'total_active_agents',
            'Total number of active agents in the system',
            ['agent_type'],
            registry=self.registry
        )
        
        self.metrics['cicd_pipeline_queue_depth'] = Gauge(
            'cicd_pipeline_queue_depth',
            'Number of CI/CD pipeline jobs in queue',
            ['pipeline_type', 'priority'],
            registry=self.registry
        )
        
        # Performance and health metrics
        self.metrics['agent_performance_score'] = Gauge(
            'agent_performance_score',
            'Agent performance score based on completion rate and quality (0-100)',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        self.metrics['agent_health_score'] = Gauge(
            'agent_health_score',
            'Agent health score based on responsiveness and error rates (0-100)',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        # Resource utilization metrics
        self.metrics['agent_cpu_efficiency'] = Gauge(
            'agent_cpu_efficiency',
            'Agent CPU efficiency score (task completion per CPU unit)',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        self.metrics['agent_memory_efficiency'] = Gauge(
            'agent_memory_efficiency',
            'Agent memory efficiency score (task completion per memory unit)',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
    
    async def start_collection(self):
        """Start background metric collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Custom metrics collection started")
    
    async def stop_collection(self):
        """Stop background metric collection"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Custom metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    async def _collect_all_metrics(self):
        """Collect all custom metrics"""
        try:
            # Collect metrics concurrently
            await asyncio.gather(
                self._collect_agent_metrics(),
                self._collect_task_metrics(),
                self._collect_context_metrics(),
                self._collect_system_metrics(),
                self._collect_performance_metrics(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _collect_agent_metrics(self):
        """Collect agent-specific metrics"""
        async with self.db_session_factory() as session:
            # Get all active agents
            query = select(Agent).where(Agent.status == AgentStatus.ACTIVE)
            result = await session.execute(query)
            active_agents = result.scalars().all()
            
            # Update total active agents metric
            agent_counts = {}
            for agent in active_agents:
                agent_type = agent.agent_type
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            for agent_type, count in agent_counts.items():
                self.metrics['total_active_agents'].labels(agent_type=agent_type).set(count)
            
            # Collect individual agent metrics
            for agent in active_agents:
                await self._collect_agent_specific_metrics(session, agent)
    
    async def _collect_agent_specific_metrics(self, session: AsyncSession, agent: Agent):
        """Collect metrics for a specific agent"""
        try:
            agent_id = str(agent.id)
            agent_type = agent.agent_type
            
            # Task queue depth by priority
            task_query = select(Task).where(
                and_(
                    Task.assigned_agent_id == agent.id,
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.IN_PROGRESS])
                )
            )
            tasks = await session.execute(task_query)
            
            priority_counts = {}
            for task in tasks.scalars().all():
                priority = task.priority.value if task.priority else 'normal'
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Update task queue metrics
            for priority, count in priority_counts.items():
                self.metrics['agent_task_queue_depth'].labels(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    priority=priority
                ).set(count)
            
            # Active sessions from Redis
            session_key = f"agent_sessions:{agent_id}"
            session_count = await self.redis.scard(session_key)
            self.metrics['agent_active_sessions'].labels(
                agent_type=agent_type,
                agent_id=agent_id
            ).set(session_count or 0)
            
            # Agent-type specific metrics
            if agent_type == "architect":
                await self._collect_architect_metrics(session, agent)
            elif agent_type == "qa":
                await self._collect_qa_metrics(session, agent)
            elif agent_type == "meta":
                await self._collect_meta_metrics(session, agent)
            
        except Exception as e:
            logger.error(f"Error collecting metrics for agent {agent.id}: {e}")
    
    async def _collect_architect_metrics(self, session: AsyncSession, agent: Agent):
        """Collect architect-specific metrics"""
        agent_id = str(agent.id)
        
        # Architecture requests by complexity
        arch_requests_key = f"architect_requests:{agent_id}"
        
        # Get complexity levels from Redis
        complexity_data = await self.redis.hgetall(arch_requests_key)
        for complexity_level, count in complexity_data.items():
            self.metrics['agent_architecture_requests'].labels(
                agent_type="architect",
                agent_id=agent_id,
                complexity_level=complexity_level.decode()
            ).set(int(count))
        
        # Design complexity score
        complexity_key = f"design_complexity:{agent_id}"
        complexity_score = await self.redis.get(complexity_key)
        if complexity_score:
            self.metrics['agent_design_complexity_score'].labels(
                agent_type="architect",
                agent_id=agent_id
            ).set(float(complexity_score))
    
    async def _collect_qa_metrics(self, session: AsyncSession, agent: Agent):
        """Collect QA-specific metrics"""
        agent_id = str(agent.id)
        
        # Test queue by test type
        test_queue_key = f"test_queue:{agent_id}"
        test_data = await self.redis.hgetall(test_queue_key)
        
        for test_type, count in test_data.items():
            self.metrics['agent_test_queue_size'].labels(
                agent_type="qa",
                agent_id=agent_id,
                test_type=test_type.decode()
            ).set(int(count))
        
        # Review backlog by priority
        review_backlog_key = f"review_backlog:{agent_id}"
        review_data = await self.redis.hgetall(review_backlog_key)
        
        for priority, count in review_data.items():
            self.metrics['agent_review_backlog_size'].labels(
                agent_type="qa",
                agent_id=agent_id,
                review_priority=priority.decode()
            ).set(int(count))
    
    async def _collect_meta_metrics(self, session: AsyncSession, agent: Agent):
        """Collect meta agent coordination metrics"""
        agent_id = str(agent.id)
        
        # Coordination complexity by type
        coordination_key = f"coordination:{agent_id}"
        coordination_data = await self.redis.hgetall(coordination_key)
        
        for coord_type, complexity in coordination_data.items():
            self.metrics['agent_coordination_complexity'].labels(
                agent_id=agent_id,
                coordination_type=coord_type.decode()
            ).set(float(complexity))
    
    async def _collect_task_metrics(self):
        """Collect task-related metrics"""
        try:
            # CI/CD pipeline queue depth
            pipeline_queue_key = "cicd_pipeline_queue"
            pipeline_data = await self.redis.hgetall(pipeline_queue_key)
            
            for pipeline_type, data in pipeline_data.items():
                pipeline_info = eval(data)  # Assuming JSON-like structure
                count = pipeline_info.get('count', 0)
                priority = pipeline_info.get('priority', 'normal')
                
                self.metrics['cicd_pipeline_queue_depth'].labels(
                    pipeline_type=pipeline_type.decode(),
                    priority=priority
                ).set(count)
                
        except Exception as e:
            logger.error(f"Error collecting task metrics: {e}")
    
    async def _collect_context_metrics(self):
        """Collect context engine metrics"""
        try:
            # Processing rate by operation type
            context_metrics_key = "context_engine_metrics"
            context_data = await self.redis.hgetall(context_metrics_key)
            
            for operation_type, rate in context_data.items():
                self.metrics['context_engine_processing_rate'].labels(
                    operation_type=operation_type.decode()
                ).set(float(rate))
            
            # Memory usage by type
            memory_key = "context_engine_memory"
            memory_data = await self.redis.hgetall(memory_key)
            
            for memory_type, usage in memory_data.items():
                self.metrics['context_engine_memory_usage'].labels(
                    memory_type=memory_type.decode()
                ).set(float(usage))
                
        except Exception as e:
            logger.error(f"Error collecting context metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # System health and performance metrics
            system_key = "system_metrics"
            system_data = await self.redis.hgetall(system_key)
            
            # Extract and update various system metrics
            for key, value in system_data.items():
                key_str = key.decode()
                if key_str.startswith('performance_'):
                    # Handle performance metrics
                    pass
                elif key_str.startswith('health_'):
                    # Handle health metrics
                    pass
                    
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect agent performance and efficiency metrics"""
        try:
            async with self.db_session_factory() as session:
                # Get performance data for all agents
                agents = await session.execute(
                    select(Agent).where(Agent.status == AgentStatus.ACTIVE)
                )
                
                for agent in agents.scalars().all():
                    agent_id = str(agent.id)
                    agent_type = agent.agent_type
                    
                    # Get performance metrics from Redis
                    perf_key = f"agent_performance:{agent_id}"
                    perf_data = await self.redis.hgetall(perf_key)
                    
                    if perf_data:
                        # Performance score
                        if b'performance_score' in perf_data:
                            score = float(perf_data[b'performance_score'])
                            self.metrics['agent_performance_score'].labels(
                                agent_type=agent_type,
                                agent_id=agent_id
                            ).set(score)
                        
                        # Health score
                        if b'health_score' in perf_data:
                            health = float(perf_data[b'health_score'])
                            self.metrics['agent_health_score'].labels(
                                agent_type=agent_type,
                                agent_id=agent_id
                            ).set(health)
                        
                        # CPU efficiency
                        if b'cpu_efficiency' in perf_data:
                            cpu_eff = float(perf_data[b'cpu_efficiency'])
                            self.metrics['agent_cpu_efficiency'].labels(
                                agent_type=agent_type,
                                agent_id=agent_id
                            ).set(cpu_eff)
                        
                        # Memory efficiency
                        if b'memory_efficiency' in perf_data:
                            mem_eff = float(perf_data[b'memory_efficiency'])
                            self.metrics['agent_memory_efficiency'].labels(
                                agent_type=agent_type,
                                agent_id=agent_id
                            ).set(mem_eff)
                            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def update_agent_metric(
        self, 
        agent_id: str, 
        metric_type: AgentMetricType, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Update a specific agent metric immediately"""
        try:
            # Store in Redis for persistence
            metric_key = f"agent_metric:{agent_id}:{metric_type.value}"
            metric_data = {
                "value": value,
                "timestamp": time.time(),
                "labels": json.dumps(labels or {})
            }
            
            await self.redis.hset(metric_key, mapping=metric_data)
            await self.redis.expire(metric_key, 3600)  # 1 hour expiry
            
        except Exception as e:
            logger.error(f"Error updating agent metric: {e}")

# FastAPI integration
def setup_custom_metrics_endpoint(app: FastAPI, exporter: CustomMetricsExporter):
    """Setup metrics endpoint for Prometheus scraping"""
    
    @app.get("/metrics/custom")
    async def custom_metrics():
        """Endpoint for custom metrics in Prometheus format"""
        metrics_data = exporter.get_metrics()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    
    @app.get("/metrics/agent/{agent_id}")
    async def agent_specific_metrics(agent_id: str):
        """Get metrics for a specific agent"""
        # Implementation for agent-specific metrics
        return {"agent_id": agent_id, "metrics": "agent_specific_data"}

# Utility functions for metric calculation
class MetricsCalculator:
    """Utility class for calculating complex metrics"""
    
    @staticmethod
    def calculate_design_complexity(task_data: Dict[str, Any]) -> float:
        """Calculate design complexity score (0-100)"""
        complexity_factors = {
            'component_count': task_data.get('components', 0) * 5,
            'integration_points': task_data.get('integrations', 0) * 8,
            'data_models': task_data.get('models', 0) * 3,
            'external_apis': task_data.get('external_apis', 0) * 10,
            'security_requirements': task_data.get('security_level', 1) * 15
        }
        
        total_complexity = sum(complexity_factors.values())
        # Normalize to 0-100 scale
        return min(total_complexity / 2, 100.0)
    
    @staticmethod
    def calculate_coordination_complexity(agents_data: List[Dict[str, Any]]) -> float:
        """Calculate coordination complexity for meta agents (0-100)"""
        if not agents_data:
            return 0.0
        
        agent_count = len(agents_data)
        active_tasks = sum(agent.get('active_tasks', 0) for agent in agents_data)
        conflicts = sum(agent.get('conflicts', 0) for agent in agents_data)
        
        # Coordination complexity increases with agent count and conflicts
        base_complexity = min(agent_count * 5, 50)
        task_complexity = min(active_tasks * 2, 30)
        conflict_complexity = min(conflicts * 10, 20)
        
        return min(base_complexity + task_complexity + conflict_complexity, 100.0)
    
    @staticmethod
    def calculate_performance_score(
        tasks_completed: int,
        tasks_failed: int,
        avg_completion_time: float,
        quality_score: float
    ) -> float:
        """Calculate agent performance score (0-100)"""
        if tasks_completed == 0:
            return 0.0
        
        # Success rate component (0-40)
        total_tasks = tasks_completed + tasks_failed
        success_rate = tasks_completed / total_tasks
        success_component = success_rate * 40
        
        # Speed component (0-30) - faster is better, normalized
        speed_component = max(0, 30 - (avg_completion_time / 60))  # Penalty for > 60 min
        
        # Quality component (0-30)
        quality_component = (quality_score / 100) * 30
        
        return min(success_component + speed_component + quality_component, 100.0)