"""
Integration tests for the comprehensive sleep analytics system.

Tests the complete analytics pipeline including:
- Metrics collection and analysis
- Efficiency reporting and trend analysis
- API endpoint functionality
- Performance tracking and alerting
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from app.core.sleep_analytics import (
    SleepAnalyticsEngine, SleepEfficiencyMetrics, ConsolidationTrends,
    AnalyticsTimeRange, get_sleep_analytics_engine
)
from app.models.sleep_wake import SleepWakeCycle, SleepState, ConsolidationJob, ConsolidationStatus
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.performance_metric import PerformanceMetric


class TestSleepAnalyticsEngine:
    """Test the SleepAnalyticsEngine core functionality."""
    
    @pytest.fixture
    def analytics_engine(self):
        """Create analytics engine for testing."""
        return SleepAnalyticsEngine()
    
    @pytest.fixture
    def sample_agent(self):
        """Create sample agent for testing."""
        return Agent(
            id=uuid4(),
            name="test-analytics-agent",
            type=AgentType.WORKER,
            status=AgentStatus.ACTIVE
        )
    
    @pytest.fixture
    def sample_sleep_cycle(self, sample_agent):
        """Create sample sleep cycle for testing."""
        return SleepWakeCycle(
            id=uuid4(),
            agent_id=sample_agent.id,
            cycle_type="test_analytics",
            sleep_state=SleepState.SLEEPING,
            sleep_time=datetime.utcnow() - timedelta(hours=2),
            wake_time=datetime.utcnow() - timedelta(hours=1),
            token_reduction_achieved=0.45,
            consolidation_time_ms=15000.0,
            recovery_time_ms=2000.0
        )
    
    @pytest.fixture
    def sample_consolidation_jobs(self, sample_sleep_cycle):
        """Create sample consolidation jobs."""
        return [
            ConsolidationJob(
                id=uuid4(),
                cycle_id=sample_sleep_cycle.id,
                job_type="context_compression",
                status=ConsolidationStatus.COMPLETED,
                tokens_processed=10000,
                tokens_saved=4500,
                processing_time_ms=8000.0,
                priority=100
            ),
            ConsolidationJob(
                id=uuid4(),
                cycle_id=sample_sleep_cycle.id,
                job_type="vector_index_update",
                status=ConsolidationStatus.COMPLETED,
                tokens_processed=5000,
                tokens_saved=2000,
                processing_time_ms=3000.0,
                priority=80
            ),
            ConsolidationJob(
                id=uuid4(),
                cycle_id=sample_sleep_cycle.id,
                job_type="redis_stream_cleanup",
                status=ConsolidationStatus.COMPLETED,
                processing_time_ms=1000.0,
                priority=60
            )
        ]
    
    def test_analytics_time_range(self):
        """Test AnalyticsTimeRange functionality."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        time_range = AnalyticsTimeRange(start_date=start_date, end_date=end_date)
        start_dt, end_dt = time_range.to_datetime_range()
        
        assert start_dt.date() == start_date
        assert end_dt.date() == end_date
        assert start_dt.time() == datetime.min.time()
        assert end_dt.time() == datetime.max.time()
    
    def test_sleep_efficiency_metrics_calculation(self):
        """Test SleepEfficiencyMetrics calculations."""
        metrics = SleepEfficiencyMetrics(
            total_cycles=10,
            successful_cycles=9,
            failed_cycles=1,
            total_tokens_saved=50000,
            average_token_reduction=0.45,
            token_reduction_efficiency=0.82,
            average_consolidation_time_ms=12000.0,
            uptime_percentage=95.0,
            cpu_efficiency=85.0,
            memory_optimization=88.0
        )
        
        # Calculate success rate
        assert metrics.success_rate == 90.0  # 9/10 * 100
        
        # Calculate overall score
        overall_score = metrics.calculate_overall_score()
        assert 0 <= overall_score <= 100
        assert overall_score > 70  # Should be good with these metrics
    
    @pytest.mark.asyncio
    async def test_collect_sleep_cycle_metrics(self, analytics_engine, sample_sleep_cycle, sample_consolidation_jobs):
        """Test collecting metrics for a sleep cycle."""
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            # Mock database session
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Mock cycle retrieval
            mock_session_instance.get = AsyncMock(return_value=sample_sleep_cycle)
            
            # Mock consolidation jobs query
            mock_jobs_result = Mock()
            mock_jobs_result.scalars.return_value.all.return_value = sample_consolidation_jobs
            mock_session_instance.execute = AsyncMock(return_value=mock_jobs_result)
            
            # Mock helper methods
            analytics_engine._store_performance_metrics = AsyncMock()
            
            # Test metrics collection
            metrics = await analytics_engine.collect_sleep_cycle_metrics(sample_sleep_cycle.id)
            
            assert metrics["cycle_id"] == str(sample_sleep_cycle.id)
            assert "collection_timestamp" in metrics
            assert "metrics" in metrics
            
            # Verify cycle performance metrics
            cycle_metrics = metrics["metrics"]["cycle_performance"]
            assert cycle_metrics["cycle_type"] == "test_analytics"
            assert cycle_metrics["token_reduction_achieved"] == 0.45
            assert cycle_metrics["is_successful"] is True
            
            # Verify consolidation job metrics
            job_metrics = metrics["metrics"]["consolidation_jobs"]
            assert job_metrics["total_jobs"] == 3
            assert job_metrics["completed_jobs"] == 3
            assert job_metrics["failed_jobs"] == 0
            assert job_metrics["success_rate"] == 100.0
    
    @pytest.mark.asyncio
    async def test_generate_efficiency_report(self, analytics_engine, sample_agent):
        """Test generating efficiency report."""
        # Create sample cycles data
        sample_cycles = []
        for i in range(5):
            cycle = SleepWakeCycle(
                id=uuid4(),
                agent_id=sample_agent.id,
                cycle_type="test",
                sleep_state=SleepState.SLEEPING,
                token_reduction_achieved=0.4 + (i * 0.05),  # 0.4 to 0.6
                consolidation_time_ms=10000.0 + (i * 1000),
                recovery_time_ms=1500.0,
                created_at=datetime.utcnow() - timedelta(days=i)
            )
            
            # Add consolidation jobs
            cycle.consolidation_jobs = [
                ConsolidationJob(
                    cycle_id=cycle.id,
                    job_type="context_compression",
                    status=ConsolidationStatus.COMPLETED,
                    tokens_saved=8000 + (i * 500)
                )
            ]
            
            sample_cycles.append(cycle)
        
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            # Mock database session
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Mock cycles query
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = sample_cycles
            mock_session_instance.execute = AsyncMock(return_value=mock_result)
            
            # Test efficiency report generation
            time_range = AnalyticsTimeRange(
                start_date=date.today() - timedelta(days=7),
                end_date=date.today()
            )
            
            efficiency_metrics = await analytics_engine.generate_efficiency_report(sample_agent.id, time_range)
            
            assert efficiency_metrics.total_cycles == 5
            assert efficiency_metrics.successful_cycles == 5
            assert efficiency_metrics.failed_cycles == 0
            assert efficiency_metrics.success_rate == 100.0
            assert efficiency_metrics.average_token_reduction == 0.5  # Average of 0.4 to 0.6
            assert efficiency_metrics.total_tokens_saved == sum(8000 + (i * 500) for i in range(5))
    
    @pytest.mark.asyncio
    async def test_analyze_consolidation_trends(self, analytics_engine, sample_agent):
        """Test consolidation trends analysis."""
        # Create sample cycles with different performance over time
        sample_cycles = []
        base_date = datetime.utcnow() - timedelta(days=14)
        
        for i in range(14):  # 14 days of data
            for j in range(2):  # 2 cycles per day
                cycle = SleepWakeCycle(
                    id=uuid4(),
                    agent_id=sample_agent.id,
                    cycle_type="test",
                    sleep_state=SleepState.SLEEPING,
                    # Improving performance over time
                    token_reduction_achieved=0.3 + (i * 0.02),  # 0.3 to 0.56
                    created_at=base_date + timedelta(days=i, hours=j*12)
                )
                sample_cycles.append(cycle)
        
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            # Mock cycles for trend analysis
            analytics_engine._get_consolidation_candidates = AsyncMock(return_value=sample_cycles)
            
            trends = await analytics_engine.analyze_consolidation_trends(sample_agent.id, 14)
            
            assert len(trends.daily_efficiency) == 14
            assert trends.efficiency_trend == "improving"  # Should detect improvement
            assert trends.predicted_efficiency > 0
            assert len(trends.confidence_interval) == 2
            assert trends.recommendation_score > 70  # Should be good for improving trend
    
    @pytest.mark.asyncio
    async def test_update_daily_analytics(self, analytics_engine):
        """Test daily analytics update."""
        target_date = date.today() - timedelta(days=1)
        
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            # Mock database operations
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Mock agent IDs query
            mock_agents_result = Mock()
            mock_agents_result.fetchall.return_value = [(uuid4(),), (uuid4(),)]
            mock_session_instance.execute = AsyncMock(return_value=mock_agents_result)
            
            # Mock individual agent update methods
            analytics_engine._update_agent_daily_analytics = AsyncMock()
            analytics_engine._update_system_daily_analytics = AsyncMock()
            
            success = await analytics_engine.update_daily_analytics(target_date)
            
            assert success is True
            analytics_engine._update_agent_daily_analytics.assert_called()
            analytics_engine._update_system_daily_analytics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_data(self, analytics_engine):
        """Test real-time dashboard data generation."""
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            # Mock database session
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            # Mock helper methods
            analytics_engine._get_active_cycles = AsyncMock(return_value=[])
            analytics_engine._get_recent_performance = AsyncMock(return_value={})
            analytics_engine._get_system_health_indicators = AsyncMock(return_value={
                "status": "healthy",
                "error_rate": 2.0,
                "average_efficiency": 0.55
            })
            analytics_engine._get_current_efficiency_metrics = AsyncMock(return_value={
                "current_token_reduction": 0.52,
                "meets_target": False
            })
            analytics_engine._generate_performance_alerts = AsyncMock(return_value=[])
            
            dashboard_data = await analytics_engine.get_real_time_dashboard_data()
            
            assert "timestamp" in dashboard_data
            assert "active_cycles" in dashboard_data
            assert "health_indicators" in dashboard_data
            assert "current_metrics" in dashboard_data
            assert "alerts" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_export_analytics_report(self, analytics_engine, sample_agent):
        """Test analytics report export."""
        time_range = AnalyticsTimeRange(
            start_date=date.today() - timedelta(days=7),
            end_date=date.today()
        )
        
        # Mock helper methods
        analytics_engine.generate_efficiency_report = AsyncMock(return_value=SleepEfficiencyMetrics())
        analytics_engine.analyze_consolidation_trends = AsyncMock(return_value=ConsolidationTrends(
            daily_efficiency=[],
            weekly_patterns={},
            hourly_patterns={},
            efficiency_trend="stable",
            peak_performance_time=None,
            optimization_opportunities=[],
            predicted_efficiency=0.5,
            confidence_interval=(0.4, 0.6),
            recommendation_score=75.0
        ))
        analytics_engine._get_detailed_cycle_data = AsyncMock(return_value=[])
        
        # Test comprehensive report
        report = await analytics_engine.export_analytics_report(
            report_type="comprehensive",
            agent_id=sample_agent.id,
            time_range=time_range,
            format="json"
        )
        
        assert report["report_type"] == "comprehensive"
        assert report["agent_id"] == str(sample_agent.id)
        assert "efficiency_metrics" in report
        assert "consolidation_trends" in report
        assert "detailed_cycles" in report
        
        # Test summary report
        report = await analytics_engine.export_analytics_report(
            report_type="summary",
            agent_id=sample_agent.id,
            time_range=time_range,
            format="json"
        )
        
        assert report["report_type"] == "summary"
        assert "efficiency_metrics" in report
        assert "key_insights" in report
    
    def test_caching_functionality(self, analytics_engine):
        """Test analytics caching functionality."""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Test cache miss
        assert not analytics_engine._is_cache_valid(cache_key)
        
        # Test cache storage
        analytics_engine._cache_result(cache_key, test_data)
        assert analytics_engine._is_cache_valid(cache_key)
        assert analytics_engine._metrics_cache[cache_key] == test_data
        
        # Test cache expiry
        import datetime
        analytics_engine._cache_expiry[cache_key] = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
        assert not analytics_engine._is_cache_valid(cache_key)
    
    @pytest.mark.asyncio
    async def test_background_analytics_lifecycle(self, analytics_engine):
        """Test background analytics start/stop lifecycle."""
        # Mock the daily analytics loop
        with patch.object(analytics_engine, '_daily_analytics_loop') as mock_loop:
            mock_task = AsyncMock()
            mock_loop.return_value = mock_task
            
            with patch('asyncio.create_task', return_value=mock_task) as mock_create_task:
                # Test start
                await analytics_engine.start_background_analytics()
                assert len(analytics_engine._background_tasks) == 1
                mock_create_task.assert_called_once()
                
                # Test stop
                await analytics_engine.stop_background_analytics()
                assert len(analytics_engine._background_tasks) == 0
                mock_task.cancel.assert_called_once()


class TestSleepAnalyticsAPI:
    """Test the analytics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_analytics_endpoints_compilation(self):
        """Test that analytics API endpoints compile correctly."""
        try:
            from app.api.analytics import router
            assert router is not None
            assert router.prefix == "/analytics"
            assert "analytics" in router.tags
            
            # Test that all expected endpoints are registered
            routes = [route.path for route in router.routes]
            expected_routes = [
                "/dashboard",
                "/efficiency/{agent_id}",
                "/efficiency",
                "/trends/{agent_id}",
                "/trends",
                "/cycles/{cycle_id}/metrics",
                "/update-daily",
                "/export",
                "/health",
                "/quick/agent/{agent_id}/summary",
                "/quick/system/status"
            ]
            
            for expected_route in expected_routes:
                assert any(expected_route in route for route in routes), f"Route {expected_route} not found"
            
        except ImportError as e:
            pytest.fail(f"Failed to import analytics API: {e}")


class TestAnalyticsIntegration:
    """Test integration between analytics components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_flow(self):
        """Test complete analytics flow from data collection to reporting."""
        # This would test the complete flow in a real environment
        # For now, we test that the components integrate correctly
        
        engine = get_sleep_analytics_engine()
        assert engine is not None
        assert isinstance(engine, SleepAnalyticsEngine)
        
        # Test configuration
        assert engine.target_token_reduction == 0.55
        assert engine.max_consolidation_time_ms == 30000
        assert engine.min_success_rate == 95.0
        assert engine.min_uptime_percentage == 99.0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self):
        """Test integration with performance metrics system."""
        engine = SleepAnalyticsEngine()
        
        # Test that metrics storage integration works
        with patch('app.core.sleep_analytics.get_async_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            test_metrics = {
                "cycle_performance": {
                    "cycle_id": str(uuid4()),
                    "efficiency_score": 85.0,
                    "token_reduction_achieved": 0.55
                },
                "consolidation_jobs": {
                    "success_rate": 95.0,
                    "total_jobs": 5
                }
            }
            
            agent_id = uuid4()
            await engine._store_performance_metrics(mock_session_instance, agent_id, test_metrics)
            
            # Verify that PerformanceMetric objects were added to session
            assert mock_session_instance.add.called
            assert mock_session_instance.commit.called
    
    def test_analytics_data_structures(self):
        """Test analytics data structures and their relationships."""
        # Test SleepEfficiencyMetrics
        metrics = SleepEfficiencyMetrics(
            total_cycles=20,
            successful_cycles=18,
            failed_cycles=2,
            total_tokens_saved=100000,
            average_token_reduction=0.52
        )
        
        assert metrics.success_rate == 90.0
        assert metrics.calculate_overall_score() > 0
        
        # Test ConsolidationTrends
        trends = ConsolidationTrends(
            daily_efficiency=[0.4, 0.45, 0.5, 0.55, 0.6],
            weekly_patterns={"Monday": 0.5, "Tuesday": 0.55},
            hourly_patterns={2: 0.6, 14: 0.45},
            efficiency_trend="improving",
            peak_performance_time="02:00",
            optimization_opportunities=["Increase consolidation frequency"],
            predicted_efficiency=0.65,
            confidence_interval=(0.6, 0.7),
            recommendation_score=85.0
        )
        
        assert trends.efficiency_trend == "improving"
        assert trends.peak_performance_time == "02:00"
        assert len(trends.daily_efficiency) == 5
        assert len(trends.optimization_opportunities) == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])