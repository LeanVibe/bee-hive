"""
Cost Monitoring and Resource Utilization Tracking System.

Comprehensive cost monitoring for API usage, infrastructure resources,
and optimization recommendations for the Context Engine.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

import numpy as np
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..core.database import get_async_session
from ..core.redis import get_redis_client

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs to track."""
    EMBEDDING_API = "embedding_api"
    COMPUTE_RESOURCES = "compute_resources"
    STORAGE = "storage"
    NETWORK = "network"
    MONITORING = "monitoring"
    THIRD_PARTY_SERVICES = "third_party_services"


class ResourceType(Enum):
    """Types of resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    API_CALLS = "api_calls"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_MEMORY = "cache_memory"


@dataclass
class CostEntry:
    """Represents a cost entry."""
    entry_id: str
    category: CostCategory
    service_name: str
    resource_type: str
    amount_usd: float
    quantity: float
    unit: str  # tokens, requests, GB, hours, etc.
    unit_cost: float  # cost per unit
    timestamp: datetime
    billing_period: str  # hourly, daily, monthly
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Represents resource usage measurement."""
    measurement_id: str
    resource_type: ResourceType
    component: str
    current_usage: float
    max_capacity: float
    unit: str
    utilization_percentage: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudget:
    """Represents a cost budget with alerts."""
    budget_id: str
    name: str
    category: Optional[CostCategory]
    amount_usd: float
    period: str  # daily, weekly, monthly
    alert_thresholds: List[float]  # percentage thresholds for alerts
    current_spend: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reset: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation."""
    recommendation_id: str
    category: CostCategory
    title: str
    description: str
    potential_savings_usd: float
    potential_savings_percentage: float
    implementation_effort: str  # low, medium, high
    priority: int  # 1-5
    implementation_steps: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostMonitor:
    """
    Comprehensive cost monitoring and resource utilization tracking system.
    
    Features:
    - Real-time cost tracking across multiple categories
    - Resource utilization monitoring and alerting
    - Cost budgets with automated alerts
    - Cost optimization recommendations
    - Trend analysis and forecasting
    - Resource efficiency analysis
    - Multi-dimensional cost attribution
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize the cost monitor.
        
        Args:
            redis_client: Redis client for real-time data
            db_session: Database session for persistent storage
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        
        # Cost tracking
        self.cost_entries: deque[CostEntry] = deque(maxlen=100000)
        self.resource_usage: Dict[str, ResourceUsage] = {}
        self.cost_budgets: Dict[str, CostBudget] = {}
        
        # Aggregated data
        self.cost_aggregates: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.resource_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minutes
        
        # Cost optimization
        self.optimization_recommendations: List[CostOptimizationRecommendation] = []
        self.cost_anomalies: List[Dict[str, Any]] = []
        
        # Configuration
        self.api_cost_rates = {
            "openai": {
                "text-embedding-ada-002": 0.0001 / 1000,  # $0.0001 per 1K tokens
                "text-embedding-3-small": 0.00002 / 1000,  # $0.00002 per 1K tokens
                "text-embedding-3-large": 0.00013 / 1000,  # $0.00013 per 1K tokens
            },
            "anthropic": {
                "claude-3-haiku": 0.00025 / 1000,  # $0.00025 per 1K tokens
                "claude-3-sonnet": 0.003 / 1000,   # $0.003 per 1K tokens
                "claude-3-opus": 0.015 / 1000,     # $0.015 per 1K tokens
            }
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("Cost Monitor initialized")
    
    async def start(self) -> None:
        """Start the cost monitor background processes."""
        logger.info("Starting cost monitor")
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._cost_aggregator()),
            asyncio.create_task(self._resource_monitor()),
            asyncio.create_task(self._budget_tracker()),
            asyncio.create_task(self._optimization_analyzer()),
            asyncio.create_task(self._anomaly_detector()),
            asyncio.create_task(self._data_cleanup())
        ])
    
    async def stop(self) -> None:
        """Stop the cost monitor."""
        logger.info("Stopping cost monitor")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def record_api_cost(
        self,
        provider: str,
        model: str,
        tokens: int,
        request_type: str = "embedding",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostEntry:
        """
        Record API usage cost.
        
        Args:
            provider: API provider (openai, anthropic, etc.)
            model: Model used
            tokens: Number of tokens processed
            request_type: Type of request (embedding, completion, etc.)
            metadata: Additional metadata
            
        Returns:
            CostEntry for the recorded cost
        """
        try:
            # Calculate cost
            unit_cost = self.api_cost_rates.get(provider, {}).get(model, 0.001 / 1000)  # Default rate
            total_cost = tokens * unit_cost
            
            cost_entry = CostEntry(
                entry_id=str(uuid.uuid4()),
                category=CostCategory.EMBEDDING_API,
                service_name=f"{provider}:{model}",
                resource_type="tokens",
                amount_usd=total_cost,
                quantity=tokens,
                unit="tokens",
                unit_cost=unit_cost,
                timestamp=datetime.utcnow(),
                billing_period="immediate",
                metadata=metadata or {},
                tags={
                    "provider": provider,
                    "model": model,
                    "request_type": request_type
                }
            )
            
            # Store cost entry
            self.cost_entries.append(cost_entry)
            
            # Store in Redis for real-time access
            await self._store_cost_entry_redis(cost_entry)
            
            # Update budgets
            await self._update_budget_spending(cost_entry)
            
            logger.debug(f"Recorded API cost: ${total_cost:.6f} for {tokens} tokens ({provider}:{model})")
            
            return cost_entry
            
        except Exception as e:
            logger.error(f"Failed to record API cost: {e}")
            raise
    
    async def record_resource_usage(
        self,
        resource_type: ResourceType,
        component: str,
        current_usage: float,
        max_capacity: float,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ResourceUsage:
        """
        Record resource usage measurement.
        
        Args:
            resource_type: Type of resource
            component: Component using the resource
            current_usage: Current usage amount
            max_capacity: Maximum capacity
            unit: Unit of measurement
            metadata: Additional metadata
            
        Returns:
            ResourceUsage measurement
        """
        try:
            utilization_percentage = (current_usage / max_capacity * 100) if max_capacity > 0 else 0
            
            resource_usage = ResourceUsage(
                measurement_id=str(uuid.uuid4()),
                resource_type=resource_type,
                component=component,
                current_usage=current_usage,
                max_capacity=max_capacity,
                unit=unit,
                utilization_percentage=utilization_percentage,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Store current usage
            resource_key = f"{resource_type.value}:{component}"
            self.resource_usage[resource_key] = resource_usage
            
            # Store trend data
            self.resource_trends[resource_key].append({
                "timestamp": resource_usage.timestamp,
                "usage": current_usage,
                "utilization": utilization_percentage
            })
            
            # Store in Redis
            await self._store_resource_usage_redis(resource_usage)
            
            # Check for resource alerts
            await self._check_resource_alerts(resource_usage)
            
            return resource_usage
            
        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}")
            raise
    
    async def create_cost_budget(
        self,
        name: str,
        amount_usd: float,
        period: str,
        category: Optional[CostCategory] = None,
        alert_thresholds: Optional[List[float]] = None
    ) -> CostBudget:
        """
        Create a cost budget with alerts.
        
        Args:
            name: Budget name
            amount_usd: Budget amount in USD
            period: Budget period (daily, weekly, monthly)
            category: Optional cost category filter
            alert_thresholds: Percentage thresholds for alerts (default: [50, 80, 100])
            
        Returns:
            Created CostBudget
        """
        try:
            budget = CostBudget(
                budget_id=str(uuid.uuid4()),
                name=name,
                category=category,
                amount_usd=amount_usd,
                period=period,
                alert_thresholds=alert_thresholds or [50.0, 80.0, 100.0]
            )
            
            self.cost_budgets[budget.budget_id] = budget
            
            # Store in Redis
            await self.redis_client.setex(
                f"cost_budget:{budget.budget_id}",
                86400 * 365,  # 1 year TTL
                json.dumps(asdict(budget), default=str)
            )
            
            logger.info(f"Created cost budget: {name} (${amount_usd}/{period})")
            
            return budget
            
        except Exception as e:
            logger.error(f"Failed to create cost budget: {e}")
            raise
    
    async def get_cost_summary(
        self,
        time_window_hours: int = 24,
        category_filter: Optional[CostCategory] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive cost summary.
        
        Args:
            time_window_hours: Time window for analysis
            category_filter: Optional category filter
            
        Returns:
            Cost summary with breakdown and trends
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter cost entries
            relevant_entries = [
                entry for entry in self.cost_entries
                if entry.timestamp >= cutoff_time and
                (category_filter is None or entry.category == category_filter)
            ]
            
            if not relevant_entries:
                return {"error": "No cost data available for the specified period"}
            
            # Calculate totals
            total_cost = sum(entry.amount_usd for entry in relevant_entries)
            
            # Breakdown by category
            cost_by_category = defaultdict(float)
            cost_by_service = defaultdict(float)
            cost_by_hour = defaultdict(float)
            
            for entry in relevant_entries:
                cost_by_category[entry.category.value] += entry.amount_usd
                cost_by_service[entry.service_name] += entry.amount_usd
                
                # Group by hour for trends
                hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
                cost_by_hour[hour_key] += entry.amount_usd
            
            # Calculate projections
            hours_analyzed = max(time_window_hours, 1)
            hourly_rate = total_cost / hours_analyzed
            
            # Usage statistics
            total_tokens = sum(
                entry.quantity for entry in relevant_entries
                if entry.unit == "tokens"
            )
            
            total_requests = len([
                entry for entry in relevant_entries
                if entry.category == CostCategory.EMBEDDING_API
            ])
            
            # Top cost drivers
            service_costs = list(cost_by_service.items())
            service_costs.sort(key=lambda x: x[1], reverse=True)
            top_services = service_costs[:10]
            
            summary = {
                "analysis_period": {
                    "hours": time_window_hours,
                    "start_time": cutoff_time.isoformat(),
                    "end_time": datetime.utcnow().isoformat()
                },
                "total_cost_usd": round(total_cost, 4),
                "cost_breakdown": {
                    "by_category": dict(cost_by_category),
                    "by_service": dict(cost_by_service)
                },
                "cost_trends": {
                    "hourly": dict(cost_by_hour),
                    "hourly_rate": round(hourly_rate, 4)
                },
                "projections": {
                    "daily_cost": round(hourly_rate * 24, 4),
                    "weekly_cost": round(hourly_rate * 24 * 7, 4),
                    "monthly_cost": round(hourly_rate * 24 * 30, 4)
                },
                "usage_statistics": {
                    "total_requests": total_requests,
                    "total_tokens": total_tokens,
                    "avg_cost_per_request": round(total_cost / max(total_requests, 1), 6),
                    "avg_cost_per_token": round(total_cost / max(total_tokens, 1), 8)
                },
                "top_cost_drivers": [
                    {"service": service, "cost_usd": round(cost, 4)}
                    for service, cost in top_services
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {"error": str(e)}
    
    async def get_resource_utilization_report(self) -> Dict[str, Any]:
        """Get comprehensive resource utilization report."""
        try:
            current_time = datetime.utcnow()
            
            # Current resource utilization
            current_utilization = {}
            resource_alerts = []
            
            for resource_key, usage in self.resource_usage.items():
                resource_type, component = resource_key.split(":", 1)
                
                if resource_type not in current_utilization:
                    current_utilization[resource_type] = []
                
                utilization_data = {
                    "component": component,
                    "current_usage": usage.current_usage,
                    "max_capacity": usage.max_capacity,
                    "utilization_percentage": usage.utilization_percentage,
                    "unit": usage.unit,
                    "last_updated": usage.timestamp.isoformat()
                }
                
                current_utilization[resource_type].append(utilization_data)
                
                # Check for high utilization
                if usage.utilization_percentage > 85:
                    resource_alerts.append({
                        "resource_type": resource_type,
                        "component": component,
                        "utilization": usage.utilization_percentage,
                        "severity": "critical" if usage.utilization_percentage > 95 else "warning"
                    })
            
            # Calculate trend data
            resource_trends = {}
            for resource_key, trend_data in self.resource_trends.items():
                if trend_data:
                    resource_type, component = resource_key.split(":", 1)
                    
                    recent_data = list(trend_data)[-60:]  # Last hour
                    if recent_data:
                        avg_utilization = statistics.mean(d["utilization"] for d in recent_data)
                        max_utilization = max(d["utilization"] for d in recent_data)
                        
                        if resource_type not in resource_trends:
                            resource_trends[resource_type] = {}
                        
                        resource_trends[resource_type][component] = {
                            "avg_utilization_1h": round(avg_utilization, 2),
                            "max_utilization_1h": round(max_utilization, 2),
                            "data_points": len(recent_data)
                        }
            
            # Resource efficiency analysis
            efficiency_scores = {}
            for resource_type, components in current_utilization.items():
                total_usage = sum(c["current_usage"] for c in components)
                total_capacity = sum(c["max_capacity"] for c in components)
                
                if total_capacity > 0:
                    overall_utilization = (total_usage / total_capacity) * 100
                    
                    # Efficiency score based on utilization (optimal around 70-80%)
                    if 70 <= overall_utilization <= 80:
                        efficiency_score = 100
                    elif overall_utilization < 70:
                        efficiency_score = max(0, 100 - (70 - overall_utilization) * 2)
                    else:
                        efficiency_score = max(0, 100 - (overall_utilization - 80) * 3)
                    
                    efficiency_scores[resource_type] = {
                        "overall_utilization": round(overall_utilization, 2),
                        "efficiency_score": round(efficiency_score, 2),
                        "recommendation": self._get_efficiency_recommendation(overall_utilization)
                    }
            
            report = {
                "timestamp": current_time.isoformat(),
                "current_utilization": current_utilization,
                "resource_trends": resource_trends,
                "efficiency_analysis": efficiency_scores,
                "resource_alerts": resource_alerts,
                "total_resources_monitored": len(self.resource_usage),
                "summary": {
                    "total_alerts": len(resource_alerts),
                    "critical_alerts": len([a for a in resource_alerts if a["severity"] == "critical"]),
                    "avg_efficiency": round(
                        statistics.mean(e["efficiency_score"] for e in efficiency_scores.values()) 
                        if efficiency_scores else 0, 2
                    )
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to get resource utilization report: {e}")
            return {"error": str(e)}
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Get status of all cost budgets."""
        try:
            budget_status = []
            
            for budget in self.cost_budgets.values():
                # Calculate current spend for budget period
                current_spend = await self._calculate_budget_spend(budget)
                
                # Update budget current spend
                budget.current_spend = current_spend
                
                spend_percentage = (current_spend / budget.amount_usd * 100) if budget.amount_usd > 0 else 0
                
                # Determine status
                status = "on_track"
                if spend_percentage >= 100:
                    status = "over_budget"
                elif spend_percentage >= max(budget.alert_thresholds):
                    status = "critical"
                elif spend_percentage >= 80:
                    status = "warning"
                
                # Check which thresholds have been crossed
                crossed_thresholds = [
                    threshold for threshold in budget.alert_thresholds
                    if spend_percentage >= threshold
                ]
                
                budget_info = {
                    "budget_id": budget.budget_id,
                    "name": budget.name,
                    "category": budget.category.value if budget.category else "all",
                    "amount_usd": budget.amount_usd,
                    "period": budget.period,
                    "current_spend": round(current_spend, 4),
                    "remaining_budget": round(budget.amount_usd - current_spend, 4),
                    "spend_percentage": round(spend_percentage, 2),
                    "status": status,
                    "crossed_thresholds": crossed_thresholds,
                    "days_remaining": self._calculate_days_remaining(budget),
                    "last_reset": budget.last_reset.isoformat()
                }
                
                budget_status.append(budget_info)
            
            # Overall budget summary
            total_budgets = len(budget_status)
            over_budget_count = len([b for b in budget_status if b["status"] == "over_budget"])
            warning_count = len([b for b in budget_status if b["status"] in ["warning", "critical"]])
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "budgets": budget_status,
                "summary": {
                    "total_budgets": total_budgets,
                    "over_budget": over_budget_count,
                    "warning_or_critical": warning_count,
                    "on_track": total_budgets - over_budget_count - warning_count
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get budget status: {e}")
            return {"error": str(e)}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations."""
        try:
            return [asdict(rec) for rec in self.optimization_recommendations]
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    # Background task methods
    async def _cost_aggregator(self) -> None:
        """Background task to aggregate cost data."""
        logger.info("Starting cost aggregator")
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Aggregate costs by various dimensions
                await self._aggregate_hourly_costs()
                await self._aggregate_daily_costs()
                
                # Clean up old aggregates
                await self._cleanup_old_aggregates()
                
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cost aggregator error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Cost aggregator stopped")
    
    async def _resource_monitor(self) -> None:
        """Background task to monitor resource utilization."""
        logger.info("Starting resource monitor")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect system resource metrics
                await self._collect_system_resources()
                
                # Check for resource optimization opportunities
                await self._analyze_resource_efficiency()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Resource monitor stopped")
    
    async def _budget_tracker(self) -> None:
        """Background task to track budget spending and alerts."""
        logger.info("Starting budget tracker")
        
        while not self._shutdown_event.is_set():
            try:
                for budget in self.cost_budgets.values():
                    # Check if budget period has reset
                    if self._should_reset_budget(budget):
                        await self._reset_budget(budget)
                    
                    # Check for budget alerts
                    await self._check_budget_alerts(budget)
                
                await asyncio.sleep(300)  # Check budgets every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Budget tracker error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Budget tracker stopped")
    
    async def _optimization_analyzer(self) -> None:
        """Background task to analyze costs and generate optimization recommendations."""
        logger.info("Starting optimization analyzer")
        
        while not self._shutdown_event.is_set():
            try:
                # Analyze API costs for optimization opportunities
                await self._analyze_api_cost_optimization()
                
                # Analyze resource utilization for optimization
                await self._analyze_resource_optimization()
                
                # Clean up old recommendations
                self._cleanup_old_recommendations()
                
                await asyncio.sleep(1800)  # Analyze every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization analyzer error: {e}")
                await asyncio.sleep(1800)
        
        logger.info("Optimization analyzer stopped")
    
    async def _anomaly_detector(self) -> None:
        """Background task to detect cost anomalies."""
        logger.info("Starting cost anomaly detector")
        
        while not self._shutdown_event.is_set():
            try:
                # Detect unusual cost spikes
                await self._detect_cost_anomalies()
                
                await asyncio.sleep(600)  # Check for anomalies every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Anomaly detector error: {e}")
                await asyncio.sleep(600)
        
        logger.info("Cost anomaly detector stopped")
    
    async def _data_cleanup(self) -> None:
        """Background task for data cleanup and maintenance."""
        logger.info("Starting data cleanup")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old cost entries (keep last 30 days)
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                original_count = len(self.cost_entries)
                self.cost_entries = deque([
                    entry for entry in self.cost_entries
                    if entry.timestamp > cutoff_time
                ], maxlen=100000)
                
                cleaned_count = original_count - len(self.cost_entries)
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old cost entries")
                
                # Clean up old resource trends
                for resource_key in list(self.resource_trends.keys()):
                    trend_data = self.resource_trends[resource_key]
                    if trend_data:
                        # Keep only last 24 hours
                        cutoff_time_trend = datetime.utcnow() - timedelta(hours=24)
                        filtered_data = deque([
                            data for data in trend_data
                            if data["timestamp"] > cutoff_time_trend
                        ], maxlen=1440)
                        
                        self.resource_trends[resource_key] = filtered_data
                
                # Clean up Redis data
                await self._cleanup_redis_data()
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Data cleanup stopped")
    
    # Helper methods
    async def _store_cost_entry_redis(self, cost_entry: CostEntry) -> None:
        """Store cost entry in Redis for real-time access."""
        try:
            await self.redis_client.lpush(
                "cost_monitor:entries",
                json.dumps(asdict(cost_entry), default=str)
            )
            
            # Keep only recent entries
            await self.redis_client.ltrim("cost_monitor:entries", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to store cost entry in Redis: {e}")
    
    async def _store_resource_usage_redis(self, resource_usage: ResourceUsage) -> None:
        """Store resource usage in Redis."""
        try:
            resource_key = f"{resource_usage.resource_type.value}:{resource_usage.component}"
            
            await self.redis_client.setex(
                f"resource_usage:{resource_key}",
                3600,  # 1 hour TTL
                json.dumps(asdict(resource_usage), default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to store resource usage in Redis: {e}")
    
    async def _update_budget_spending(self, cost_entry: CostEntry) -> None:
        """Update budget spending with new cost entry."""
        try:
            for budget in self.cost_budgets.values():
                # Check if cost entry applies to this budget
                if budget.category is None or budget.category == cost_entry.category:
                    # Check if cost entry is within budget period
                    if self._is_within_budget_period(cost_entry.timestamp, budget):
                        budget.current_spend += cost_entry.amount_usd
            
        except Exception as e:
            logger.error(f"Failed to update budget spending: {e}")
    
    async def _check_resource_alerts(self, resource_usage: ResourceUsage) -> None:
        """Check for resource utilization alerts."""
        try:
            if resource_usage.utilization_percentage > 95:
                logger.critical(f"CRITICAL: {resource_usage.component} {resource_usage.resource_type.value} utilization at {resource_usage.utilization_percentage:.1f}%")
            elif resource_usage.utilization_percentage > 85:
                logger.warning(f"WARNING: {resource_usage.component} {resource_usage.resource_type.value} utilization at {resource_usage.utilization_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to check resource alerts: {e}")
    
    async def _calculate_budget_spend(self, budget: CostBudget) -> float:
        """Calculate current spending for a budget."""
        try:
            period_start = self._get_budget_period_start(budget)
            
            total_spend = 0.0
            
            for entry in self.cost_entries:
                if entry.timestamp >= period_start:
                    # Check if entry applies to this budget
                    if budget.category is None or budget.category == entry.category:
                        total_spend += entry.amount_usd
            
            return total_spend
            
        except Exception as e:
            logger.error(f"Failed to calculate budget spend: {e}")
            return 0.0
    
    def _get_budget_period_start(self, budget: CostBudget) -> datetime:
        """Get the start time for the current budget period."""
        now = datetime.utcnow()
        
        if budget.period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == "weekly":
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == "monthly":
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return budget.last_reset
    
    def _calculate_days_remaining(self, budget: CostBudget) -> int:
        """Calculate days remaining in budget period."""
        now = datetime.utcnow()
        
        if budget.period == "daily":
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return (tomorrow - now).days
        elif budget.period == "weekly":
            days_since_monday = now.weekday()
            next_monday = now + timedelta(days=(7 - days_since_monday))
            next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            return (next_monday - now).days
        elif budget.period == "monthly":
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return (next_month - now).days
        
        return 0
    
    def _should_reset_budget(self, budget: CostBudget) -> bool:
        """Check if budget should be reset for new period."""
        period_start = self._get_budget_period_start(budget)
        return budget.last_reset < period_start
    
    async def _reset_budget(self, budget: CostBudget) -> None:
        """Reset budget for new period."""
        try:
            budget.current_spend = 0.0
            budget.last_reset = datetime.utcnow()
            
            logger.info(f"Reset budget {budget.name} for new {budget.period} period")
            
        except Exception as e:
            logger.error(f"Failed to reset budget: {e}")
    
    def _is_within_budget_period(self, timestamp: datetime, budget: CostBudget) -> bool:
        """Check if timestamp is within current budget period."""
        period_start = self._get_budget_period_start(budget)
        return timestamp >= period_start
    
    async def _check_budget_alerts(self, budget: CostBudget) -> None:
        """Check for budget threshold alerts."""
        try:
            if budget.amount_usd == 0:
                return
            
            spend_percentage = (budget.current_spend / budget.amount_usd) * 100
            
            for threshold in budget.alert_thresholds:
                if spend_percentage >= threshold:
                    # Check if we already alerted for this threshold
                    alert_key = f"budget_alert:{budget.budget_id}:{threshold}"
                    already_alerted = await self.redis_client.get(alert_key)
                    
                    if not already_alerted:
                        logger.warning(f"BUDGET ALERT: {budget.name} has reached {spend_percentage:.1f}% of budget (${budget.current_spend:.2f}/${budget.amount_usd:.2f})")
                        
                        # Set alert flag to prevent duplicate alerts
                        await self.redis_client.setex(alert_key, 86400, "true")  # 24 hour TTL
            
        except Exception as e:
            logger.error(f"Failed to check budget alerts: {e}")
    
    async def _aggregate_hourly_costs(self) -> None:
        """Aggregate costs by hour for trending."""
        try:
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            hour_key = current_hour.strftime("%Y-%m-%d %H:00")
            
            # Calculate costs for current hour
            hour_costs = defaultdict(float)
            
            for entry in self.cost_entries:
                if entry.timestamp >= current_hour and entry.timestamp < current_hour + timedelta(hours=1):
                    hour_costs[entry.category.value] += entry.amount_usd
            
            # Store aggregated data
            if hour_costs:
                await self.redis_client.setex(
                    f"cost_aggregate:hourly:{hour_key}",
                    86400 * 7,  # 7 days TTL
                    json.dumps(dict(hour_costs))
                )
            
        except Exception as e:
            logger.error(f"Failed to aggregate hourly costs: {e}")
    
    async def _aggregate_daily_costs(self) -> None:
        """Aggregate costs by day."""
        try:
            current_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            day_key = current_day.strftime("%Y-%m-%d")
            
            # Calculate costs for current day
            day_costs = defaultdict(float)
            
            for entry in self.cost_entries:
                if entry.timestamp >= current_day and entry.timestamp < current_day + timedelta(days=1):
                    day_costs[entry.category.value] += entry.amount_usd
            
            # Store aggregated data
            if day_costs:
                await self.redis_client.setex(
                    f"cost_aggregate:daily:{day_key}",
                    86400 * 30,  # 30 days TTL
                    json.dumps(dict(day_costs))
                )
            
        except Exception as e:
            logger.error(f"Failed to aggregate daily costs: {e}")
    
    async def _cleanup_old_aggregates(self) -> None:
        """Clean up old aggregate data."""
        try:
            # Clean up hourly aggregates older than 7 days
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            hourly_keys = await self.redis_client.keys("cost_aggregate:hourly:*")
            
            for key in hourly_keys:
                try:
                    # Extract timestamp from key
                    timestamp_str = key.decode().split(":")[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:00")
                    
                    if timestamp < cutoff_time:
                        await self.redis_client.delete(key)
                except (ValueError, AttributeError):
                    continue
            
            # Clean up daily aggregates older than 30 days
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            daily_keys = await self.redis_client.keys("cost_aggregate:daily:*")
            
            for key in daily_keys:
                try:
                    # Extract timestamp from key
                    timestamp_str = key.decode().split(":")[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                    
                    if timestamp < cutoff_time:
                        await self.redis_client.delete(key)
                except (ValueError, AttributeError):
                    continue
            
        except Exception as e:
            logger.error(f"Failed to cleanup old aggregates: {e}")
    
    async def _collect_system_resources(self) -> None:
        """Collect system resource metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            await self.record_resource_usage(
                ResourceType.CPU,
                "system",
                cpu_usage,
                100.0,
                "percent"
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_resource_usage(
                ResourceType.MEMORY,
                "system",
                memory.used,
                memory.total,
                "bytes"
            )
            
            # Storage usage
            disk = psutil.disk_usage('/')
            await self.record_resource_usage(
                ResourceType.STORAGE,
                "system",
                disk.used,
                disk.total,
                "bytes"
            )
            
        except ImportError:
            logger.warning("psutil not available for system resource monitoring")
        except Exception as e:
            logger.error(f"Failed to collect system resources: {e}")
    
    async def _analyze_resource_efficiency(self) -> None:
        """Analyze resource efficiency and generate recommendations."""
        try:
            for resource_key, usage in self.resource_usage.items():
                resource_type, component = resource_key.split(":", 1)
                
                # Generate recommendations based on utilization
                if usage.utilization_percentage < 30:
                    # Under-utilized resource
                    recommendation = CostOptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        category=CostCategory.COMPUTE_RESOURCES,
                        title=f"Under-utilized {resource_type} in {component}",
                        description=f"{resource_type} utilization is only {usage.utilization_percentage:.1f}%. Consider rightsizing or consolidation.",
                        potential_savings_usd=10.0,  # Estimated savings
                        potential_savings_percentage=20.0,
                        implementation_effort="medium",
                        priority=3,
                        implementation_steps=[
                            f"Analyze {component} resource requirements",
                            f"Consider reducing {resource_type} allocation",
                            "Monitor performance after changes",
                            "Implement gradual scaling adjustments"
                        ]
                    )
                    
                    # Add to recommendations if not already present
                    existing = any(
                        r.title == recommendation.title 
                        for r in self.optimization_recommendations
                    )
                    
                    if not existing:
                        self.optimization_recommendations.append(recommendation)
                
                elif usage.utilization_percentage > 90:
                    # Over-utilized resource
                    recommendation = CostOptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        category=CostCategory.COMPUTE_RESOURCES,
                        title=f"Over-utilized {resource_type} in {component}",
                        description=f"{resource_type} utilization is {usage.utilization_percentage:.1f}%. Consider scaling up to prevent performance issues.",
                        potential_savings_usd=-5.0,  # Negative savings (cost increase)
                        potential_savings_percentage=-10.0,
                        implementation_effort="low",
                        priority=1,
                        implementation_steps=[
                            f"Increase {resource_type} allocation for {component}",
                            "Monitor performance improvements",
                            "Adjust scaling policies if applicable"
                        ]
                    )
                    
                    # Add to recommendations if not already present
                    existing = any(
                        r.title == recommendation.title 
                        for r in self.optimization_recommendations
                    )
                    
                    if not existing:
                        self.optimization_recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Failed to analyze resource efficiency: {e}")
    
    async def _analyze_api_cost_optimization(self) -> None:
        """Analyze API costs for optimization opportunities."""
        try:
            # Analyze recent API costs
            recent_entries = [
                entry for entry in self.cost_entries
                if (entry.category == CostCategory.EMBEDDING_API and
                    (datetime.utcnow() - entry.timestamp).total_seconds() < 86400)  # Last 24 hours
            ]
            
            if not recent_entries:
                return
            
            # Analyze by provider and model
            provider_costs = defaultdict(float)
            model_usage = defaultdict(lambda: {"cost": 0.0, "tokens": 0})
            
            for entry in recent_entries:
                provider = entry.tags.get("provider", "unknown")
                model = entry.tags.get("model", "unknown")
                
                provider_costs[provider] += entry.amount_usd
                model_usage[f"{provider}:{model}"]["cost"] += entry.amount_usd
                model_usage[f"{provider}:{model}"]["tokens"] += entry.quantity
            
            # Check for expensive models
            for model_key, usage_data in model_usage.items():
                if usage_data["tokens"] > 0:
                    cost_per_token = usage_data["cost"] / usage_data["tokens"]
                    
                    # If cost per token is high, recommend cheaper alternatives
                    if cost_per_token > 0.0001:  # Threshold for expensive models
                        recommendation = CostOptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            category=CostCategory.EMBEDDING_API,
                            title=f"High-cost model usage: {model_key}",
                            description=f"Model {model_key} has high cost per token ({cost_per_token:.6f}). Consider switching to more cost-effective alternatives.",
                            potential_savings_usd=usage_data["cost"] * 0.5,  # Estimate 50% savings
                            potential_savings_percentage=50.0,
                            implementation_effort="low",
                            priority=2,
                            implementation_steps=[
                                "Evaluate cheaper embedding models",
                                "Test model performance on your use case",
                                "Implement gradual migration",
                                "Monitor quality metrics"
                            ]
                        )
                        
                        # Add to recommendations if not already present
                        existing = any(
                            r.title == recommendation.title 
                            for r in self.optimization_recommendations
                        )
                        
                        if not existing:
                            self.optimization_recommendations.append(recommendation)
            
            # Check for high overall API costs
            total_daily_cost = sum(entry.amount_usd for entry in recent_entries)
            if total_daily_cost > 5.0:  # More than $5 per day
                recommendation = CostOptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category=CostCategory.EMBEDDING_API,
                    title="High daily API costs",
                    description=f"Daily API costs are high (${total_daily_cost:.2f}). Consider implementing aggressive caching and batching strategies.",
                    potential_savings_usd=total_daily_cost * 0.3,  # Estimate 30% savings
                    potential_savings_percentage=30.0,
                    implementation_effort="medium",
                    priority=1,
                    implementation_steps=[
                        "Implement aggressive embedding caching",
                        "Add duplicate content detection",
                        "Batch API requests where possible",
                        "Review embedding frequency and necessity"
                    ]
                )
                
                # Add to recommendations if not already present
                existing = any(
                    r.title == recommendation.title 
                    for r in self.optimization_recommendations
                )
                
                if not existing:
                    self.optimization_recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Failed to analyze API cost optimization: {e}")
    
    def _cleanup_old_recommendations(self) -> None:
        """Clean up old optimization recommendations."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            self.optimization_recommendations = [
                rec for rec in self.optimization_recommendations
                if rec.created_at > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to cleanup old recommendations: {e}")
    
    async def _detect_cost_anomalies(self) -> None:
        """Detect unusual cost spikes or patterns."""
        try:
            # Get hourly costs for anomaly detection
            hourly_costs = []
            current_time = datetime.utcnow()
            
            for hour_offset in range(24):  # Last 24 hours
                hour_start = current_time - timedelta(hours=hour_offset + 1)
                hour_end = hour_start + timedelta(hours=1)
                
                hour_cost = sum(
                    entry.amount_usd for entry in self.cost_entries
                    if hour_start <= entry.timestamp < hour_end
                )
                
                hourly_costs.append(hour_cost)
            
            if len(hourly_costs) >= 6:  # Need at least 6 hours of data
                # Simple anomaly detection based on standard deviation
                mean_cost = statistics.mean(hourly_costs)
                stdev_cost = statistics.stdev(hourly_costs) if len(hourly_costs) > 1 else 0
                
                current_hour_cost = hourly_costs[0]  # Most recent hour
                
                # Check if current hour is anomalous
                if stdev_cost > 0:
                    z_score = abs(current_hour_cost - mean_cost) / stdev_cost
                    
                    if z_score > 2.0:  # More than 2 standard deviations
                        anomaly = {
                            "anomaly_id": str(uuid.uuid4()),
                            "type": "cost_spike",
                            "detected_at": current_time.isoformat(),
                            "current_hour_cost": current_hour_cost,
                            "mean_cost": mean_cost,
                            "z_score": z_score,
                            "description": f"Unusual cost spike detected: ${current_hour_cost:.4f} vs average ${mean_cost:.4f}"
                        }
                        
                        self.cost_anomalies.append(anomaly)
                        
                        logger.warning(f"Cost anomaly detected: {anomaly['description']}")
                        
                        # Keep only recent anomalies
                        cutoff_time = datetime.utcnow() - timedelta(hours=24)
                        self.cost_anomalies = [
                            a for a in self.cost_anomalies
                            if datetime.fromisoformat(a["detected_at"]) > cutoff_time
                        ]
            
        except Exception as e:
            logger.error(f"Failed to detect cost anomalies: {e}")
    
    def _get_efficiency_recommendation(self, utilization: float) -> str:
        """Get efficiency recommendation based on utilization."""
        if utilization < 30:
            return "Consider reducing resource allocation or consolidating workloads"
        elif utilization < 50:
            return "Resource allocation appears conservative, monitor for optimization opportunities"
        elif utilization < 70:
            return "Good resource utilization"
        elif utilization < 85:
            return "Optimal resource utilization"
        elif utilization < 95:
            return "High utilization, monitor for performance impact"
        else:
            return "Critical utilization, consider scaling up immediately"
    
    async def _cleanup_redis_data(self) -> None:
        """Clean up old Redis data."""
        try:
            # Clean up old cost entries
            await self.redis_client.ltrim("cost_monitor:entries", 0, 9999)
            
            # Clean up old resource usage data
            resource_keys = await self.redis_client.keys("resource_usage:*")
            current_time = datetime.utcnow()
            
            for key in resource_keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        usage_data = json.loads(data)
                        timestamp = datetime.fromisoformat(usage_data["timestamp"])
                        
                        # Remove data older than 24 hours
                        if (current_time - timestamp).total_seconds() > 86400:
                            await self.redis_client.delete(key)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Remove invalid data
                    await self.redis_client.delete(key)
            
        except Exception as e:
            logger.error(f"Failed to cleanup Redis data: {e}")


# Global instance
_cost_monitor: Optional[CostMonitor] = None


async def get_cost_monitor() -> CostMonitor:
    """Get singleton cost monitor instance."""
    global _cost_monitor
    
    if _cost_monitor is None:
        _cost_monitor = CostMonitor()
        await _cost_monitor.start()
    
    return _cost_monitor


async def cleanup_cost_monitor() -> None:
    """Cleanup cost monitor resources."""
    global _cost_monitor
    
    if _cost_monitor:
        await _cost_monitor.stop()
        _cost_monitor = None