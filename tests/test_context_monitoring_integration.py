"""
Comprehensive Integration Tests for Context Monitoring System.

Tests the complete monitoring stack including performance monitoring,
cost tracking, capacity planning, alerting, and health monitoring.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.context_performance_monitor import (
    ContextPerformanceMonitor,
    ContextOperation,
    PerformanceIssueType
)
from app.core.cost_monitoring import (
    CostMonitor,
    CostCategory,
    ResourceType
)
from app.core.capacity_planning import (
    CapacityPlanner,
    CapacityMetric,
    GrowthTrend
)
from app.core.intelligent_alerting import (
    AlertManager,
    AlertSeverity,
    AlertStatus
)
from app.core.health_monitoring import (
    HealthMonitor,
    HealthStatus,
    ComponentType
)
from app.models.context import ContextType


class TestContextMonitoringIntegration:
    """Integration tests for the complete context monitoring system."""
    
    @pytest.fixture
    async def redis_mock(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.setex.return_value = True
        mock_redis.lpush.return_value = 1
        mock_redis.lrange.return_value = []
        mock_redis.ltrim.return_value = True
        mock_redis.keys.return_value = []
        mock_redis.delete.return_value = 1
        mock_redis.incr.return_value = 1
        return mock_redis
    
    @pytest.fixture
    async def db_mock(self):
        """Mock database session."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1000
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result
        mock_db.commit.return_value = None
        return mock_db
    
    @pytest.fixture
    async def performance_monitor(self, redis_mock, db_mock):
        """Create performance monitor instance."""
        monitor = ContextPerformanceMonitor(
            redis_client=redis_mock,
            db_session=db_mock
        )
        return monitor
    
    @pytest.fixture
    async def cost_monitor(self, redis_mock, db_mock):
        """Create cost monitor instance."""
        monitor = CostMonitor(
            redis_client=redis_mock,
            db_session=db_mock
        )
        return monitor
    
    @pytest.fixture
    async def capacity_planner(self, redis_mock, db_mock):
        """Create capacity planner instance."""
        planner = CapacityPlanner(
            redis_client=redis_mock,
            db_session=db_mock
        )
        return planner
    
    @pytest.fixture
    async def alert_manager(self, redis_mock):
        """Create alert manager instance."""
        manager = AlertManager(redis_client=redis_mock)
        return manager
    
    @pytest.fixture
    async def health_monitor(self, redis_mock, db_mock):
        """Create health monitor instance."""
        monitor = HealthMonitor(
            redis_client=redis_mock,
            db_session=db_mock
        )
        return monitor


class TestPerformanceMonitoringIntegration(TestContextMonitoringIntegration):
    """Test performance monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_record_context_operation_with_metrics(self, performance_monitor):
        """Test recording context operations and generating metrics."""
        # Record various operations
        operations = [
            (ContextOperation.CREATE, "context1", ContextType.CONVERSATION, 150.0, True),
            (ContextOperation.READ, "context2", ContextType.CODE, 50.0, True),
            (ContextOperation.SEARCH, "context3", ContextType.DOCUMENTATION, 250.0, True),
            (ContextOperation.UPDATE, "context1", ContextType.CONVERSATION, 100.0, True),
            (ContextOperation.SEARCH, "context4", ContextType.CODE, 1200.0, False)  # Slow search
        ]
        
        for operation, context_id, context_type, duration, success in operations:
            await performance_monitor.record_operation(
                operation=operation,
                context_id=context_id,
                context_type=context_type,
                duration_ms=duration,
                success=success
            )
        
        # Verify context metrics were updated
        assert len(performance_monitor.context_metrics) == 4  # 4 unique contexts
        
        # Check specific context metrics
        context1_metrics = performance_monitor.context_metrics["context1"]
        assert context1_metrics.operation_counts[ContextOperation.CREATE.value] == 1
        assert context1_metrics.operation_counts[ContextOperation.UPDATE.value] == 1
        assert context1_metrics.last_accessed is not None
        
        # Check operation times tracking
        assert len(performance_monitor.operation_times["search_code"]) == 2
        
        # Verify Redis storage was called
        performance_monitor.redis_client.lpush.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_performance_recording(self, performance_monitor):
        """Test search performance recording and analysis."""
        # Record search performance
        search_data = [
            ("vector_search", "test query 1", 10, 85.0, 0.85, False),
            ("hybrid_search", "test query 2", 15, 120.0, 0.92, True),
            ("vector_search", "test query 3", 8, 2000.0, 0.75, False),  # Slow search
            ("semantic_search", "test query 4", 20, 45.0, 0.88, True)
        ]
        
        for search_type, query, result_count, latency, quality, cache_hit in search_data:
            await performance_monitor.record_search_performance(
                search_type=search_type,
                query=query,
                result_count=result_count,
                latency_ms=latency,
                quality_score=quality,
                cache_hit=cache_hit
            )
        
        # Verify Prometheus metrics were updated
        assert performance_monitor.search_latency_seconds._child_samples()
        assert performance_monitor.search_result_count._child_samples()
        
        # Verify Redis storage
        performance_monitor.redis_client.lpush.assert_called_with(
            "context_monitor:search_metrics",
            json.dumps({
                "timestamp": pytest.approx(datetime.utcnow().isoformat(), abs=10),
                "search_type": "semantic_search",
                "query": "test query 4",
                "result_count": 20,
                "latency_ms": 45.0,
                "quality_score": 0.88,
                "cache_hit": True
            })
        )
    
    @pytest.mark.asyncio
    async def test_embedding_api_cost_tracking(self, performance_monitor):
        """Test embedding API cost tracking."""
        # Record API calls
        api_calls = [
            ("openai", "text-embedding-ada-002", 1000, 150.0, 0.0001, True),
            ("anthropic", "claude-3-haiku", 500, 200.0, 0.000125, True),
            ("openai", "text-embedding-3-small", 2000, 100.0, 0.00004, True),
            ("openai", "text-embedding-ada-002", 800, 300.0, 0.00008, False)  # Failed call
        ]
        
        for provider, model, tokens, duration, cost, success in api_calls:
            await performance_monitor.record_embedding_api_call(
                provider=provider,
                model=model,
                tokens=tokens,
                duration_ms=duration,
                cost_usd=cost,
                success=success
            )
        
        # Verify Prometheus metrics
        assert performance_monitor.embedding_api_calls_total._child_samples()
        assert performance_monitor.embedding_api_cost_usd._child_samples()
        
        # Verify cost tracking in Redis
        performance_monitor.redis_client.lpush.assert_called_with(
            "context_monitor:api_costs",
            pytest.approx(json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "tokens": 800,
                "duration_ms": 300.0,
                "cost_usd": 0.00008,
                "success": False
            }), abs=10)
        )
    
    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, performance_monitor):
        """Test comprehensive performance summary generation."""
        # Setup mock Redis data
        mock_search_metrics = [
            json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "search_type": "vector",
                "query": "test",
                "result_count": 10,
                "latency_ms": 150.0,
                "quality_score": 0.85,
                "cache_hit": True
            })
        ]
        
        mock_api_costs = [
            json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "provider": "openai",
                "model": "ada-002",
                "tokens": 1000,
                "duration_ms": 200.0,
                "cost_usd": 0.0001,
                "success": True
            })
        ]
        
        performance_monitor.redis_client.lrange.side_effect = [
            mock_search_metrics,
            mock_api_costs
        ]
        
        # Mock database response for context storage
        async def mock_query_context_storage(session):
            return {
                "total_contexts": 5000,
                "total_size_bytes": 1024 * 1024 * 50,  # 50MB
                "by_type": {
                    "conversation": {"count": 2000, "avg_importance": 0.8, "content_size_bytes": 20971520},
                    "code": {"count": 3000, "avg_importance": 0.9, "content_size_bytes": 31457280}
                }
            }
        
        performance_monitor._get_context_storage_summary = AsyncMock(
            return_value=mock_query_context_storage(None)
        )
        
        # Get performance summary
        summary = await performance_monitor.get_performance_summary(24)
        
        # Verify summary structure
        assert "time_window_hours" in summary
        assert "search_performance" in summary
        assert "api_costs" in summary
        assert "context_metrics" in summary
        
        # Verify search performance analysis
        search_perf = summary["search_performance"]
        assert "total_searches" in search_perf
        assert "avg_latency_ms" in search_perf
        assert "cache_hit_rate" in search_perf
        
        # Verify API cost analysis
        api_costs = summary["api_costs"]
        assert "total_calls" in api_costs
        assert "total_cost_usd" in api_costs


class TestCostMonitoringIntegration(TestContextMonitoringIntegration):
    """Test cost monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_api_cost_recording_and_analysis(self, cost_monitor):
        """Test API cost recording and analysis."""
        # Record API costs
        costs = [
            ("openai", "text-embedding-ada-002", 1000, "embedding"),
            ("anthropic", "claude-3-haiku", 500, "completion"),
            ("openai", "text-embedding-3-small", 2000, "embedding"),
            ("openai", "gpt-4", 800, "completion")
        ]
        
        for provider, model, tokens, request_type in costs:
            cost_entry = await cost_monitor.record_api_cost(
                provider=provider,
                model=model,
                tokens=tokens,
                request_type=request_type
            )
            
            assert cost_entry.category == CostCategory.EMBEDDING_API
            assert cost_entry.quantity == tokens
            assert cost_entry.amount_usd > 0
        
        # Verify cost entries stored
        assert len(cost_monitor.cost_entries) == 4
        
        # Test cost summary generation
        summary = await cost_monitor.get_cost_summary(24)
        
        assert "total_cost_usd" in summary
        assert "cost_breakdown" in summary
        assert "projections" in summary
        assert summary["usage_statistics"]["total_requests"] == 4
    
    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, cost_monitor):
        """Test resource usage tracking and monitoring."""
        # Record resource usage
        resources = [
            (ResourceType.CPU, "system", 75.0, 100.0, "percent"),
            (ResourceType.MEMORY, "application", 8589934592, 17179869184, "bytes"),  # 8GB/16GB
            (ResourceType.STORAGE, "database", 85899345920, 107374182400, "bytes"),  # 80GB/100GB
            (ResourceType.NETWORK_BANDWIDTH, "api", 1048576, 10485760, "bytes/sec")  # 1MB/10MB
        ]
        
        for resource_type, component, current, max_cap, unit in resources:
            usage = await cost_monitor.record_resource_usage(
                resource_type=resource_type,
                component=component,
                current_usage=current,
                max_capacity=max_cap,
                unit=unit
            )
            
            assert usage.utilization_percentage == (current / max_cap * 100)
            assert f"{resource_type.value}:{component}" in cost_monitor.resource_usage
        
        # Test resource utilization report
        report = await cost_monitor.get_resource_utilization_report()
        
        assert "current_utilization" in report
        assert "efficiency_analysis" in report
        assert "resource_alerts" in report
        
        # Check for high utilization alerts
        storage_alert = next(
            (alert for alert in report["resource_alerts"] 
             if alert["resource_type"] == "storage"), None
        )
        assert storage_alert is not None
        assert storage_alert["severity"] == "warning"  # 80% utilization
    
    @pytest.mark.asyncio
    async def test_cost_budget_management(self, cost_monitor):
        """Test cost budget creation and monitoring."""
        # Create budgets
        daily_budget = await cost_monitor.create_cost_budget(
            name="Daily API Budget",
            amount_usd=10.0,
            period="daily",
            category=CostCategory.EMBEDDING_API,
            alert_thresholds=[50.0, 80.0, 100.0]
        )
        
        monthly_budget = await cost_monitor.create_cost_budget(
            name="Monthly Total Budget",
            amount_usd=300.0,
            period="monthly",
            category=None,  # All categories
            alert_thresholds=[75.0, 90.0, 100.0]
        )
        
        # Record costs that will trigger budget alerts
        for i in range(10):
            await cost_monitor.record_api_cost(
                provider="openai",
                model="text-embedding-ada-002",
                tokens=10000,  # Large token count
                request_type="embedding"
            )
        
        # Check budget status
        budget_status = await cost_monitor.get_budget_status()
        
        assert "budgets" in budget_status
        assert len(budget_status["budgets"]) == 2
        
        # Find daily budget status
        daily_budget_status = next(
            (budget for budget in budget_status["budgets"] 
             if budget["name"] == "Daily API Budget"), None
        )
        
        assert daily_budget_status is not None
        assert daily_budget_status["spend_percentage"] > 0


class TestCapacityPlanningIntegration(TestContextMonitoringIntegration):
    """Test capacity planning integration."""
    
    @pytest.mark.asyncio
    async def test_capacity_metric_recording_and_forecasting(self, capacity_planner):
        """Test capacity metric recording and ML-based forecasting."""
        # Record capacity metrics over time
        base_time = datetime.utcnow() - timedelta(days=30)
        
        # Simulate growing context count
        for day in range(30):
            timestamp = base_time + timedelta(days=day)
            context_count = 1000 + (day * 50) + (day ** 1.1 * 10)  # Non-linear growth
            
            await capacity_planner.record_capacity_metric(
                metric=CapacityMetric.CONTEXT_COUNT,
                value=context_count,
                metadata={"day": day}
            )
        
        # Add training data to model
        model = capacity_planner.models[CapacityMetric.CONTEXT_COUNT]
        assert len(model.training_data) == 30
        
        # Train the model
        training_success = model.train()
        assert training_success
        assert model.model is not None
        assert model.accuracy_score >= 0  # Should have some accuracy
        
        # Test forecasting
        forecast = await capacity_planner.get_capacity_forecast(
            metric=CapacityMetric.CONTEXT_COUNT,
            days_ahead=30
        )
        
        assert forecast is not None
        assert forecast.metric == CapacityMetric.CONTEXT_COUNT
        assert len(forecast.predicted_values) == 30
        assert forecast.growth_rate > 0  # Should detect positive growth
        assert forecast.trend_type in [GrowthTrend.LINEAR, GrowthTrend.EXPONENTIAL]
    
    @pytest.mark.asyncio
    async def test_capacity_threshold_monitoring(self, capacity_planner):
        """Test capacity threshold monitoring and alerting."""
        # Add capacity threshold
        threshold = await capacity_planner.add_capacity_threshold(
            metric=CapacityMetric.STORAGE_SIZE,
            warning_threshold=80 * 1024**3,  # 80GB
            critical_threshold=100 * 1024**3,  # 100GB
            max_capacity=120 * 1024**3,  # 120GB
            unit="bytes"
        )
        
        assert threshold.metric == CapacityMetric.STORAGE_SIZE
        
        # Record storage metrics that approach threshold
        storage_sizes = [
            50 * 1024**3,   # 50GB - Normal
            70 * 1024**3,   # 70GB - Normal
            85 * 1024**3,   # 85GB - Warning
            105 * 1024**3,  # 105GB - Critical
        ]
        
        for size in storage_sizes:
            await capacity_planner.record_capacity_metric(
                metric=CapacityMetric.STORAGE_SIZE,
                value=size
            )
        
        # Verify threshold was stored
        assert CapacityMetric.STORAGE_SIZE in capacity_planner.thresholds
        
        # Verify Redis storage was called
        capacity_planner.redis_client.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_scaling_recommendations_generation(self, capacity_planner):
        """Test automated scaling recommendations."""
        # Setup forecast that indicates approaching threshold
        mock_forecast = MagicMock()
        mock_forecast.current_value = 8000
        mock_forecast.predicted_values = [
            (datetime.utcnow() + timedelta(days=5), 8500),
            (datetime.utcnow() + timedelta(days=10), 9200),
            (datetime.utcnow() + timedelta(days=15), 9800),  # Approaching warning
            (datetime.utcnow() + timedelta(days=20), 10500),  # Above warning
        ]
        mock_forecast.trend_type = GrowthTrend.LINEAR
        mock_forecast.growth_rate = 50  # per day
        mock_forecast.confidence_score = 0.85
        
        # Mock threshold
        mock_threshold = MagicMock()
        mock_threshold.warning_threshold = 10000
        mock_threshold.critical_threshold = 12000
        mock_threshold.max_capacity = 15000
        
        capacity_planner.current_forecasts[CapacityMetric.CONTEXT_COUNT] = mock_forecast
        capacity_planner.thresholds[CapacityMetric.CONTEXT_COUNT] = mock_threshold
        
        # Generate recommendations
        await capacity_planner._generate_scaling_recommendations()
        
        # Verify recommendations were generated
        recommendations = await capacity_planner.get_scaling_recommendations()
        assert len(recommendations) > 0
        
        # Check recommendation content
        warning_rec = next(
            (rec for rec in recommendations 
             if "warning threshold" in rec.description.lower()), None
        )
        assert warning_rec is not None
        assert warning_rec.priority <= 2  # High priority
        assert warning_rec.days_to_threshold <= 20


class TestAlertingIntegration(TestContextMonitoringIntegration):
    """Test intelligent alerting integration."""
    
    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_and_generation(self, alert_manager):
        """Test alert rule evaluation and alert generation."""
        # Test metrics that should trigger alerts
        metrics = {
            "search_avg_latency_ms": 1200.0,  # High latency
            "search_cache_hit_rate": 0.6,     # Low cache hit rate
            "api_cost_per_hour": 15.0,        # High API costs
            "search_quality_score": 0.6       # Low search quality
        }
        
        # Evaluate metrics against rules
        new_alerts = await alert_manager.evaluate_metrics(metrics)
        
        # Should generate multiple alerts
        assert len(new_alerts) >= 3  # At least 3 alert rules should trigger
        
        # Check specific alerts
        latency_alert = next(
            (alert for alert in new_alerts 
             if "latency" in alert.message.lower()), None
        )
        assert latency_alert is not None
        assert latency_alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        cache_alert = next(
            (alert for alert in new_alerts 
             if "cache" in alert.message.lower()), None
        )
        assert cache_alert is not None
        assert cache_alert.current_value == 0.6
    
    @pytest.mark.asyncio
    async def test_alert_lifecycle_management(self, alert_manager):
        """Test complete alert lifecycle."""
        # Create an alert
        metrics = {"search_avg_latency_ms": 2000.0}
        alerts = await alert_manager.evaluate_metrics(metrics)
        
        assert len(alerts) > 0
        alert = alerts[0]
        
        # Acknowledge the alert
        ack_success = await alert_manager.acknowledge_alert(
            alert.alert_id, 
            "test_user"
        )
        assert ack_success
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"
        
        # Resolve the alert
        resolve_success = await alert_manager.resolve_alert(
            alert.alert_id,
            "test_user",
            "Latency improved after optimization"
        )
        assert resolve_success
        assert alert.status == AlertStatus.RESOLVED
        assert alert.metadata["resolved_by"] == "test_user"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, alert_manager):
        """Test ML-based anomaly detection."""
        # Add normal metrics to establish baseline
        normal_metrics = [
            {"search_avg_latency_ms": 150.0},
            {"search_avg_latency_ms": 160.0},
            {"search_avg_latency_ms": 140.0},
            {"search_avg_latency_ms": 155.0},
            {"search_avg_latency_ms": 145.0},
        ]
        
        for metric in normal_metrics:
            await alert_manager.evaluate_metrics(metric)
        
        # Add anomalous metric
        anomalous_metric = {"search_avg_latency_ms": 2000.0}  # Significant spike
        
        # Mock anomaly detection to return True
        alert_manager.anomaly_detector.detect_anomaly = MagicMock(return_value=(True, 2.5))
        
        alerts = await alert_manager.evaluate_metrics(anomalous_metric)
        
        # Should detect anomaly
        anomaly_alert = next(
            (alert for alert in alerts 
             if alert.alert_type == "anomaly"), None
        )
        assert anomaly_alert is not None
        assert anomaly_alert.metadata["anomaly_score"] == 2.5


class TestHealthMonitoringIntegration(TestContextMonitoringIntegration):
    """Test health monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_component_health_checking(self, health_monitor):
        """Test comprehensive component health checking."""
        # Mock successful health check functions
        async def mock_db_check():
            return {"status": "healthy", "message": "Database operational"}
        
        async def mock_redis_check():
            return {"status": "healthy", "message": "Redis operational"}
        
        async def mock_system_check():
            return {
                "status": "healthy",
                "message": "System resources normal",
                "details": {
                    "cpu_percent": 45.0,
                    "memory_percent": 60.0,
                    "disk_percent": 70.0
                }
            }
        
        # Replace check functions
        health_monitor._check_database_health = mock_db_check
        health_monitor._check_redis_health = mock_redis_check
        health_monitor._check_system_resources = mock_system_check
        
        # Trigger health checks
        db_results = await health_monitor.trigger_health_check(ComponentType.DATABASE)
        redis_results = await health_monitor.trigger_health_check(ComponentType.REDIS)
        system_results = await health_monitor.trigger_health_check(ComponentType.SYSTEM_RESOURCES)
        
        # Verify results
        assert len(db_results) == 1
        assert db_results[0].status == HealthStatus.HEALTHY
        
        assert len(redis_results) == 1
        assert redis_results[0].status == HealthStatus.HEALTHY
        
        assert len(system_results) == 1
        assert system_results[0].status == HealthStatus.HEALTHY
        assert system_results[0].details["cpu_percent"] == 45.0
    
    @pytest.mark.asyncio
    async def test_system_health_aggregation(self, health_monitor):
        """Test system-wide health status aggregation."""
        # Setup component health states
        healthy_components = [
            ComponentType.DATABASE,
            ComponentType.REDIS,
            ComponentType.CONTEXT_ENGINE
        ]
        
        degraded_components = [
            ComponentType.VECTOR_SEARCH
        ]
        
        critical_components = [
            ComponentType.EMBEDDING_SERVICE
        ]
        
        # Mock component health
        for component in healthy_components:
            health = MagicMock()
            health.status = HealthStatus.HEALTHY
            health.issues = []
            health_monitor.component_health[component] = health
        
        for component in degraded_components:
            health = MagicMock()
            health.status = HealthStatus.DEGRADED
            health.issues = ["Performance degraded"]
            health_monitor.component_health[component] = health
        
        for component in critical_components:
            health = MagicMock()
            health.status = HealthStatus.CRITICAL
            health.issues = ["Service unavailable"]
            health_monitor.component_health[component] = health
        
        # Get system health
        system_health = await health_monitor.get_system_health()
        
        # Verify overall status (should be CRITICAL due to critical component)
        assert system_health.overall_status == HealthStatus.CRITICAL
        assert len(system_health.critical_issues) > 0
        assert ComponentType.VECTOR_SEARCH in system_health.degraded_components
    
    @pytest.mark.asyncio
    async def test_self_healing_integration(self, health_monitor):
        """Test self-healing capabilities."""
        # Setup unhealthy component
        redis_health = MagicMock()
        redis_health.status = HealthStatus.UNHEALTHY
        health_monitor.component_health[ComponentType.REDIS] = redis_health
        
        # Get Redis healing action
        redis_healing = health_monitor.healing_actions.get("restart_redis_connection")
        assert redis_healing is not None
        
        # Mock should_trigger_healing to return True
        health_monitor._should_trigger_healing = AsyncMock(return_value=True)
        
        # Mock healing function to succeed
        async def mock_redis_heal():
            return True
        
        redis_healing.action_function = mock_redis_heal
        
        # Execute healing
        await health_monitor._execute_healing_action(redis_healing)
        
        # Verify healing was attempted
        assert redis_healing.attempts == 1
        assert redis_healing.success_count == 1
    
    @pytest.mark.asyncio
    async def test_health_trends_analysis(self, health_monitor):
        """Test health trends and analytics."""
        # Setup mock health results
        mock_results = []
        base_time = datetime.utcnow() - timedelta(hours=12)
        
        for hour in range(12):
            timestamp = base_time + timedelta(hours=hour)
            
            # Create alternating healthy/unhealthy pattern
            status = HealthStatus.HEALTHY if hour % 2 == 0 else HealthStatus.DEGRADED
            
            result = MagicMock()
            result.component = ComponentType.DATABASE
            result.status = status
            result.timestamp = timestamp
            
            mock_results.append(result)
        
        # Add results to check_results
        health_monitor.check_results["db_check"] = deque(mock_results)
        
        # Get health trends
        trends = await health_monitor.get_health_trends(
            component=ComponentType.DATABASE,
            hours_back=12
        )
        
        # Verify trends analysis
        assert "total_checks" in trends
        assert trends["total_checks"] == 12
        assert "overall_health_rate" in trends
        assert trends["overall_health_rate"] == 50.0  # 50% healthy due to alternating pattern
        
        # Verify component breakdown
        assert "component_health" in trends
        db_health = trends["component_health"]["database"]
        assert db_health["health_percentage"] == 50.0
        assert db_health["total_checks"] == 12


class TestMonitoringSystemIntegration(TestContextMonitoringIntegration):
    """Test integration between all monitoring components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(
        self, 
        performance_monitor, 
        cost_monitor, 
        capacity_planner, 
        alert_manager, 
        health_monitor
    ):
        """Test complete end-to-end monitoring flow."""
        # 1. Record performance metrics
        await performance_monitor.record_operation(
            operation=ContextOperation.SEARCH,
            context_id="test_context",
            context_type=ContextType.CODE,
            duration_ms=1500.0,  # High latency
            success=True
        )
        
        # 2. Record API costs
        cost_entry = await cost_monitor.record_api_cost(
            provider="openai",
            model="text-embedding-ada-002",
            tokens=5000,
            request_type="embedding"
        )
        
        # 3. Record capacity metrics
        await capacity_planner.record_capacity_metric(
            metric=CapacityMetric.SEARCH_QUERIES,
            value=1000.0  # High query volume
        )
        
        # 4. Generate alerts based on metrics
        high_latency_metrics = {"search_avg_latency_ms": 1500.0}
        alerts = await alert_manager.evaluate_metrics(high_latency_metrics)
        
        # 5. Check system health
        system_health = await health_monitor.get_system_health()
        
        # Verify integration
        assert len(performance_monitor.context_metrics) == 1
        assert len(cost_monitor.cost_entries) == 1
        assert len(capacity_planner.capacity_data[CapacityMetric.SEARCH_QUERIES]) == 1
        assert len(alerts) > 0
        assert system_health.overall_status is not None
        
        # Verify data flows between components
        assert cost_entry.amount_usd > 0
        assert alerts[0].current_value == 1500.0
    
    @pytest.mark.asyncio
    async def test_monitoring_resilience_and_error_handling(
        self, 
        performance_monitor,
        redis_mock
    ):
        """Test monitoring system resilience to errors."""
        # Simulate Redis failure
        redis_mock.lpush.side_effect = Exception("Redis connection failed")
        
        # Operations should still work despite Redis failure
        await performance_monitor.record_operation(
            operation=ContextOperation.CREATE,
            context_id="test_context",
            context_type=ContextType.CONVERSATION,
            duration_ms=100.0,
            success=True
        )
        
        # Verify operation was recorded in memory despite Redis failure
        assert len(performance_monitor.context_metrics) == 1
        
        # Restore Redis functionality
        redis_mock.lpush.side_effect = None
        redis_mock.lpush.return_value = 1
        
        # Subsequent operations should work normally
        await performance_monitor.record_operation(
            operation=ContextOperation.READ,
            context_id="test_context",
            context_type=ContextType.CONVERSATION,
            duration_ms=50.0,
            success=True
        )
        
        # Verify both operations are tracked
        context_metrics = performance_monitor.context_metrics["test_context"]
        assert context_metrics.operation_counts[ContextOperation.CREATE.value] == 1
        assert context_metrics.operation_counts[ContextOperation.READ.value] == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_performance_impact(self, performance_monitor):
        """Test that monitoring doesn't significantly impact performance."""
        import time
        
        # Record timing for multiple operations
        start_time = time.time()
        
        # Perform 100 monitoring operations
        for i in range(100):
            await performance_monitor.record_operation(
                operation=ContextOperation.READ,
                context_id=f"context_{i}",
                context_type=ContextType.CODE,
                duration_ms=100.0,
                success=True
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Monitoring overhead should be minimal (< 1 second for 100 operations)
        assert total_time < 1.0
        
        # Verify all operations were recorded
        assert len(performance_monitor.context_metrics) == 100


@pytest.mark.asyncio
async def test_monitoring_cleanup_and_shutdown():
    """Test proper cleanup and shutdown of monitoring components."""
    # Create monitoring components
    redis_mock = AsyncMock()
    
    performance_monitor = ContextPerformanceMonitor(redis_client=redis_mock)
    cost_monitor = CostMonitor(redis_client=redis_mock)
    capacity_planner = CapacityPlanner(redis_client=redis_mock)
    alert_manager = AlertManager(redis_client=redis_mock)
    health_monitor = HealthMonitor(redis_client=redis_mock)
    
    # Start components
    await performance_monitor.start()
    await cost_monitor.start()
    await capacity_planner.start()
    await alert_manager.start()
    await health_monitor.start()
    
    # Verify background tasks started
    assert len(performance_monitor._background_tasks) > 0
    assert len(cost_monitor._background_tasks) > 0
    assert len(capacity_planner._background_tasks) > 0
    assert len(alert_manager._background_tasks) > 0
    assert len(health_monitor._background_tasks) > 0
    
    # Stop components
    await performance_monitor.stop()
    await cost_monitor.stop()
    await capacity_planner.stop()
    await alert_manager.stop()
    await health_monitor.stop()
    
    # Verify shutdown events were set
    assert performance_monitor._shutdown_event.is_set()
    assert cost_monitor._shutdown_event.is_set()
    assert capacity_planner._shutdown_event.is_set()
    assert alert_manager._shutdown_event.is_set()
    assert health_monitor._shutdown_event.is_set()