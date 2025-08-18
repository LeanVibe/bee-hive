"""
Unit Tests for Performance Monitoring Components

Tests monitoring system components to ensure they provide accurate,
real-time performance data while maintaining minimal overhead on
the extraordinary performance of LeanVibe Agent Hive 2.0.

Test Categories:
- PerformanceMonitoringSystem unit tests
- IntelligentAlertingSystem unit tests
- CapacityPlanningSystem unit tests
- GrafanaDashboardManager unit tests
- Anomaly detection algorithm tests
- Dashboard data generation tests
"""

import asyncio
import pytest
import time
import json
import unittest.mock as mock
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.performance_monitoring_system import (
    PerformanceMonitoringSystem, SystemMetricsCollector, ApplicationMetricsCollector,
    BusinessMetricsCollector, UserExperienceCollector
)
from monitoring.intelligent_alerting_system import (
    IntelligentAlertingSystem, AnomalyDetection, AlertSeverity, AnomalyType,
    StatisticalAnomalyDetector, SeasonalAnomalyDetector, PerformanceRegressionDetector
)
from monitoring.capacity_planning_system import (
    CapacityPlanningSystem, CapacityForecast, ForecastHorizon, ResourceType,
    ScalingRecommendation
)
from monitoring.performance_dashboards.grafana_dashboard_manager import GrafanaDashboardManager


class TestPerformanceMonitoringSystem:
    """Unit tests for PerformanceMonitoringSystem."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create PerformanceMonitoringSystem instance."""
        return PerformanceMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_monitoring_initialization(self, monitoring_system):
        """Test monitoring system initialization."""
        success = await monitoring_system.initialize()
        
        assert success is True
        assert monitoring_system.system_collector is not None
        assert monitoring_system.app_collector is not None
        assert monitoring_system.business_collector is not None
        assert monitoring_system.ux_collector is not None
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, monitoring_system):
        """Test metrics collection from all collectors."""
        await monitoring_system.initialize()
        await monitoring_system.start_monitoring()
        
        # Allow collection to run briefly
        await asyncio.sleep(2)
        
        # Get dashboard data
        dashboard_data = monitoring_system.get_monitoring_dashboard_data()
        
        # Verify dashboard structure
        assert 'timestamp' in dashboard_data
        assert 'system_metrics' in dashboard_data
        assert 'application_performance' in dashboard_data
        assert 'business_metrics' in dashboard_data
        assert 'user_experience' in dashboard_data
        
        # Verify application performance metrics
        app_perf = dashboard_data['application_performance']
        assert 'metrics' in app_perf
        assert 'health_status' in app_perf
        
        # Should have key performance metrics
        metrics = app_perf['metrics']
        expected_metrics = [
            'task_assignment_latency_ms',
            'message_throughput_per_sec', 
            'memory_usage_mb',
            'error_rate_percent'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 'current' in metrics[metric]
            assert 'trend' in metrics[metric]
    
    @pytest.mark.asyncio
    async def test_system_metrics_collector(self, monitoring_system):
        """Test system metrics collection."""
        await monitoring_system.initialize()
        
        system_collector = monitoring_system.system_collector
        metrics = system_collector.collect_metrics()
        
        # Verify system metrics structure
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_io_read_mb_per_sec' in metrics
        assert 'network_bytes_sent_per_sec' in metrics
        
        # Verify metric values are reasonable
        assert 0 <= metrics['cpu_percent'] <= 100
        assert 0 <= metrics['memory_percent'] <= 100
        assert metrics['disk_io_read_mb_per_sec'] >= 0
        assert metrics['network_bytes_sent_per_sec'] >= 0
    
    @pytest.mark.asyncio
    async def test_application_metrics_collector(self, monitoring_system):
        """Test application metrics collection."""
        await monitoring_system.initialize()
        
        app_collector = monitoring_system.app_collector
        metrics = app_collector.collect_metrics()
        
        # Verify application metrics
        assert 'task_assignment_latency_ms' in metrics
        assert 'message_throughput_per_sec' in metrics
        assert 'agent_registration_latency_ms' in metrics
        assert 'workflow_execution_time_ms' in metrics
        assert 'cache_hit_rate_percent' in metrics
        assert 'error_rate_percent' in metrics
        
        # Verify performance targets are met
        assert metrics['task_assignment_latency_ms'] <= 0.1  # Should be excellent
        assert metrics['message_throughput_per_sec'] >= 10000  # Should be high
        assert 0 <= metrics['cache_hit_rate_percent'] <= 100
    
    @pytest.mark.asyncio
    async def test_business_metrics_collector(self, monitoring_system):
        """Test business metrics collection.""" 
        await monitoring_system.initialize()
        
        business_collector = monitoring_system.business_collector
        metrics = business_collector.collect_metrics()
        
        # Verify business metrics
        assert 'tasks_completed_per_minute' in metrics
        assert 'agent_success_rate_percent' in metrics
        assert 'system_availability_percent' in metrics
        assert 'sla_compliance_percent' in metrics
        
        # Verify business metric ranges
        assert metrics['tasks_completed_per_minute'] >= 0
        assert 0 <= metrics['agent_success_rate_percent'] <= 100
        assert 0 <= metrics['system_availability_percent'] <= 100
        assert 0 <= metrics['sla_compliance_percent'] <= 100
    
    @pytest.mark.asyncio 
    async def test_user_experience_collector(self, monitoring_system):
        """Test user experience metrics collection."""
        await monitoring_system.initialize()
        
        ux_collector = monitoring_system.ux_collector
        metrics = ux_collector.collect_metrics()
        
        # Verify UX metrics
        assert 'response_time_p99_ms' in metrics
        assert 'user_satisfaction_score' in metrics
        assert 'error_impact_score' in metrics
        
        # Verify UX metric ranges
        assert metrics['response_time_p99_ms'] >= 0
        assert 0 <= metrics['user_satisfaction_score'] <= 10
        assert metrics['error_impact_score'] >= 0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitoring_system):
        """Test monitoring start/stop lifecycle."""
        await monitoring_system.initialize()
        
        # Test starting monitoring
        await monitoring_system.start_monitoring()
        assert monitoring_system.monitoring_active is True
        
        # Allow brief monitoring
        await asyncio.sleep(1)
        
        # Test stopping monitoring
        await monitoring_system.stop_monitoring()
        assert monitoring_system.monitoring_active is False


class TestIntelligentAlertingSystem:
    """Unit tests for IntelligentAlertingSystem."""
    
    @pytest.fixture
    def alerting_system(self):
        """Create IntelligentAlertingSystem instance."""
        return IntelligentAlertingSystem()
    
    @pytest.fixture
    def mock_monitoring_system(self):
        """Create mock performance monitoring system."""
        mock_system = mock.MagicMock()
        mock_system.get_monitoring_dashboard_data.return_value = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.015, 'trend': 'stable'},
                    'message_throughput_per_sec': {'current': 48000, 'trend': 'increasing'},
                    'memory_usage_mb': {'current': 320, 'trend': 'stable'}
                }
            },
            'system_metrics': {
                'cpu_percent': 35.5,
                'memory_percent': 65.2
            }
        }
        return mock_system
    
    @pytest.mark.asyncio
    async def test_alerting_initialization(self, alerting_system, mock_monitoring_system):
        """Test alerting system initialization."""
        success = await alerting_system.initialize(mock_monitoring_system)
        
        assert success is True
        assert alerting_system.performance_monitor is not None
        assert alerting_system.statistical_detector is not None
        assert alerting_system.seasonal_detector is not None
        assert alerting_system.regression_detector is not None
    
    @pytest.mark.asyncio
    async def test_statistical_anomaly_detection(self, alerting_system, mock_monitoring_system):
        """Test statistical anomaly detection."""
        await alerting_system.initialize(mock_monitoring_system)
        
        detector = alerting_system.statistical_detector
        
        # Add normal data
        normal_values = [0.01, 0.012, 0.009, 0.011, 0.013, 0.010, 0.015, 0.008]
        for value in normal_values:
            detector.add_data_point('test_metric', value)
        
        # Test normal value (should not be anomaly)
        is_anomaly = detector.detect_anomaly('test_metric', 0.012)
        assert is_anomaly is False
        
        # Test anomalous value
        is_anomaly = detector.detect_anomaly('test_metric', 0.050)  # 5x normal
        assert is_anomaly is True
    
    @pytest.mark.asyncio
    async def test_seasonal_anomaly_detection(self, alerting_system, mock_monitoring_system):
        """Test seasonal anomaly detection."""
        await alerting_system.initialize(mock_monitoring_system)
        
        detector = alerting_system.seasonal_detector
        
        # Add seasonal pattern data (hourly pattern)
        base_time = datetime.utcnow()
        for hour in range(24):
            for day in range(7):  # Week of data
                timestamp = base_time + timedelta(days=day, hours=hour)
                # Simulate higher load during business hours
                if 9 <= hour <= 17:
                    value = 0.015 + np.random.normal(0, 0.002)
                else:
                    value = 0.008 + np.random.normal(0, 0.001)
                
                detector.add_data_point('test_metric', value, timestamp)
        
        # Test value during business hours (should be normal)
        business_hour_time = base_time + timedelta(hours=14)  # 2 PM
        is_anomaly = detector.detect_seasonal_anomaly('test_metric', 0.016, business_hour_time)
        assert is_anomaly is False
        
        # Test high value during off-hours (should be anomaly)
        night_time = base_time + timedelta(hours=2)  # 2 AM
        is_anomaly = detector.detect_seasonal_anomaly('test_metric', 0.020, night_time)
        assert is_anomaly is True
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, alerting_system, mock_monitoring_system):
        """Test performance regression detection."""
        await alerting_system.initialize(mock_monitoring_system)
        
        detector = alerting_system.regression_detector
        
        # Add baseline performance data
        baseline_values = [0.010, 0.011, 0.009, 0.012, 0.010, 0.013, 0.011]
        for value in baseline_values:
            detector.add_baseline_data('latency_metric', value)
        
        # Test non-regression (should be fine)
        is_regression = detector.detect_regression('latency_metric', 0.012)
        assert is_regression is False
        
        # Test clear regression (50% worse)
        is_regression = detector.detect_regression('latency_metric', 0.018)
        assert is_regression is True
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, alerting_system, mock_monitoring_system):
        """Test alert generation from anomaly detection."""
        await alerting_system.initialize(mock_monitoring_system)
        
        # Create anomaly detection
        anomaly = AnomalyDetection(
            anomaly_id='test_anomaly_001',
            timestamp=datetime.utcnow(),
            metric_name='task_assignment_latency_ms',
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            severity=AlertSeverity.WARNING,
            confidence=0.85,
            current_value=0.045,
            expected_value=0.012,
            deviation_score=3.2,
            pattern_description="Latency spike detected",
            detection_algorithm="statistical_z_score"
        )
        
        # Process anomaly (would generate alert)
        alerts = await alerting_system._generate_alerts_from_anomaly(anomaly)
        
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert['severity'] == 'warning'
        assert 'task_assignment_latency_ms' in alert['message']
        assert alert['confidence'] == 0.85
    
    @pytest.mark.asyncio
    async def test_alert_correlation(self, alerting_system, mock_monitoring_system):
        """Test alert correlation and noise reduction."""
        await alerting_system.initialize(mock_monitoring_system)
        
        # Create related anomalies
        anomaly1 = AnomalyDetection(
            anomaly_id='anomaly_001',
            timestamp=datetime.utcnow(),
            metric_name='task_assignment_latency_ms',
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            severity=AlertSeverity.WARNING,
            confidence=0.80,
            current_value=0.035
        )
        
        anomaly2 = AnomalyDetection(
            anomaly_id='anomaly_002',
            timestamp=datetime.utcnow(),
            metric_name='message_throughput_per_sec',
            anomaly_type=AnomalyType.PERFORMANCE_REGRESSION,
            severity=AlertSeverity.WARNING,
            confidence=0.75,
            current_value=25000
        )
        
        # Test correlation detection
        correlated = await alerting_system._correlate_anomalies([anomaly1, anomaly2])
        
        # Should detect correlation (latency up, throughput down)
        assert correlated is True
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, alerting_system, mock_monitoring_system):
        """Test alerting monitoring lifecycle."""
        await alerting_system.initialize(mock_monitoring_system)
        
        # Start monitoring
        await alerting_system.start_monitoring()
        status = await alerting_system.get_system_status()
        assert status['monitoring_active'] is True
        
        # Stop monitoring
        await alerting_system.stop_monitoring()
        status = await alerting_system.get_system_status()
        assert status['monitoring_active'] is False


class TestCapacityPlanningSystem:
    """Unit tests for CapacityPlanningSystem."""
    
    @pytest.fixture
    def capacity_planner(self):
        """Create CapacityPlanningSystem instance."""
        return CapacityPlanningSystem()
    
    @pytest.fixture
    def mock_monitoring_system(self):
        """Create mock performance monitoring system."""
        mock_system = mock.MagicMock()
        mock_system.get_monitoring_dashboard_data.return_value = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.012},
                    'message_throughput_per_sec': {'current': 47000},
                    'memory_usage_mb': {'current': 310}
                }
            },
            'system_metrics': {
                'cpu_percent': 42.5,
                'memory_percent': 68.3
            }
        }
        return mock_system
    
    @pytest.mark.asyncio
    async def test_capacity_planner_initialization(self, capacity_planner, mock_monitoring_system):
        """Test capacity planner initialization."""
        success = await capacity_planner.initialize(mock_monitoring_system)
        
        assert success is True
        assert capacity_planner.performance_monitor is not None
        assert capacity_planner.historical_data is not None
        assert capacity_planner.forecasting_models is not None
    
    @pytest.mark.asyncio
    async def test_capacity_forecasting(self, capacity_planner, mock_monitoring_system):
        """Test capacity forecasting capabilities."""
        await capacity_planner.initialize(mock_monitoring_system)
        
        # Add historical data for forecasting
        base_time = datetime.utcnow() - timedelta(days=30)
        for day in range(30):
            timestamp = base_time + timedelta(days=day)
            # Simulate gradual growth
            cpu_usage = 40 + (day * 0.5) + np.random.normal(0, 2)
            capacity_planner._add_historical_data_point(
                ResourceType.CPU, cpu_usage, timestamp
            )
        
        # Generate forecast
        forecast = await capacity_planner._generate_capacity_forecast(
            ResourceType.CPU, ForecastHorizon.MEDIUM_TERM
        )
        
        assert isinstance(forecast, CapacityForecast)
        assert forecast.resource_type == ResourceType.CPU
        assert forecast.forecast_horizon == ForecastHorizon.MEDIUM_TERM
        assert len(forecast.forecasted_values) > 0
        assert len(forecast.lower_confidence) == len(forecast.forecasted_values)
        assert len(forecast.upper_confidence) == len(forecast.forecasted_values)
    
    @pytest.mark.asyncio
    async def test_scaling_recommendations(self, capacity_planner, mock_monitoring_system):
        """Test scaling recommendations generation."""
        await capacity_planner.initialize(mock_monitoring_system)
        
        # Test scale-up recommendation (high utilization)
        recommendation = await capacity_planner._generate_scaling_recommendation(
            ResourceType.CPU, 
            current_utilization=85.0,
            forecasted_peak=92.0,
            time_horizon=ForecastHorizon.SHORT_TERM
        )
        
        assert recommendation == ScalingRecommendation.SCALE_UP
        
        # Test scale-out recommendation (very high utilization)
        recommendation = await capacity_planner._generate_scaling_recommendation(
            ResourceType.THROUGHPUT,
            current_utilization=95.0,
            forecasted_peak=98.0,
            time_horizon=ForecastHorizon.MEDIUM_TERM
        )
        
        assert recommendation == ScalingRecommendation.SCALE_OUT
        
        # Test maintain recommendation (normal utilization)
        recommendation = await capacity_planner._generate_scaling_recommendation(
            ResourceType.MEMORY,
            current_utilization=60.0,
            forecasted_peak=65.0,
            time_horizon=ForecastHorizon.SHORT_TERM
        )
        
        assert recommendation == ScalingRecommendation.MAINTAIN
    
    @pytest.mark.asyncio
    async def test_growth_trend_analysis(self, capacity_planner, mock_monitoring_system):
        """Test growth trend analysis."""
        await capacity_planner.initialize(mock_monitoring_system)
        
        # Add trending data (increasing load)
        base_time = datetime.utcnow() - timedelta(hours=24)
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            throughput = 35000 + (hour * 500)  # Linear growth
            capacity_planner._add_historical_data_point(
                ResourceType.THROUGHPUT, throughput, timestamp
            )
        
        # Analyze trend
        trend_analysis = await capacity_planner._analyze_growth_trends(ResourceType.THROUGHPUT)
        
        assert trend_analysis is not None
        assert 'trend_direction' in trend_analysis
        assert 'growth_rate' in trend_analysis
        assert trend_analysis['trend_direction'] == 'increasing'
        assert trend_analysis['growth_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_threshold_breach_prediction(self, capacity_planner, mock_monitoring_system):
        """Test threshold breach prediction."""
        await capacity_planner.initialize(mock_monitoring_system)
        
        # Simulate approaching threshold
        current_memory = 450  # MB, approaching 500 MB limit
        memory_growth_rate = 5  # MB per hour
        
        breach_prediction = await capacity_planner._predict_threshold_breach(
            ResourceType.MEMORY,
            current_value=current_memory,
            threshold=500,
            growth_rate=memory_growth_rate
        )
        
        assert breach_prediction is not None
        assert 'time_to_breach_hours' in breach_prediction
        assert 'breach_probability' in breach_prediction
        assert breach_prediction['time_to_breach_hours'] <= 12  # Should breach soon
        assert breach_prediction['breach_probability'] > 0.7   # High probability
    
    @pytest.mark.asyncio
    async def test_planning_lifecycle(self, capacity_planner, mock_monitoring_system):
        """Test capacity planning lifecycle."""
        await capacity_planner.initialize(mock_monitoring_system)
        
        # Start planning
        await capacity_planner.start_planning()
        status = await capacity_planner.get_planning_status()
        assert status['planning_active'] is True
        
        # Stop planning
        await capacity_planner.stop_planning()
        status = await capacity_planner.get_planning_status()
        assert status['planning_active'] is False


class TestGrafanaDashboardManager:
    """Unit tests for GrafanaDashboardManager."""
    
    @pytest.fixture
    def dashboard_manager(self):
        """Create GrafanaDashboardManager instance."""
        return GrafanaDashboardManager()
    
    @pytest.mark.asyncio
    async def test_dashboard_manager_initialization(self, dashboard_manager):
        """Test dashboard manager initialization."""
        success = await dashboard_manager.initialize()
        
        # Should succeed even without Grafana connection for testing
        assert success is True
        assert dashboard_manager.dashboard_configs is not None
        assert dashboard_manager.grafana_client is not None
    
    @pytest.mark.asyncio
    async def test_dashboard_config_loading(self, dashboard_manager):
        """Test dashboard configuration loading."""
        await dashboard_manager.initialize()
        
        configs = dashboard_manager._load_dashboard_configurations()
        
        assert configs is not None
        assert 'dashboards' in configs
        
        # Should have standard dashboards
        dashboards = configs['dashboards']
        expected_dashboards = ['system_overview', 'component_performance', 'business_metrics', 'infrastructure']
        
        for dashboard_name in expected_dashboards:
            assert dashboard_name in dashboards
            dashboard_config = dashboards[dashboard_name]
            assert 'title' in dashboard_config
            assert 'key_metrics' in dashboard_config
    
    @pytest.mark.asyncio
    async def test_dashboard_data_preparation(self, dashboard_manager):
        """Test dashboard data preparation."""
        await dashboard_manager.initialize()
        
        # Mock performance data
        mock_performance_data = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.012, 'trend': 'stable'},
                    'message_throughput_per_sec': {'current': 48500, 'trend': 'increasing'},
                    'memory_usage_mb': {'current': 295, 'trend': 'stable'}
                }
            },
            'system_metrics': {
                'cpu_percent': 38.5,
                'memory_percent': 62.1
            }
        }
        
        # Prepare dashboard data
        dashboard_data = dashboard_manager._prepare_dashboard_data(mock_performance_data)
        
        assert dashboard_data is not None
        assert 'panels' in dashboard_data
        assert 'variables' in dashboard_data
        
        # Should have panels for key metrics
        panels = dashboard_data['panels']
        assert len(panels) > 0
        
        # Verify panel structure
        for panel in panels:
            assert 'title' in panel
            assert 'type' in panel
            assert 'targets' in panel
    
    @pytest.mark.asyncio
    async def test_dashboard_creation_config(self, dashboard_manager):
        """Test dashboard creation configuration."""
        await dashboard_manager.initialize()
        
        # Test system overview dashboard config generation
        dashboard_json = dashboard_manager._create_system_overview_dashboard()
        
        assert dashboard_json is not None
        assert 'dashboard' in dashboard_json
        
        dashboard = dashboard_json['dashboard']
        assert dashboard['title'] == "LeanVibe Agent Hive 2.0 - System Overview"
        assert 'panels' in dashboard
        assert len(dashboard['panels']) > 0
        
        # Verify key metrics are included
        panel_titles = [panel['title'] for panel in dashboard['panels']]
        expected_metrics = ['Task Assignment Latency', 'Message Throughput', 'Memory Usage', 'Error Rate']
        
        for metric in expected_metrics:
            assert any(metric in title for title in panel_titles)
    
    @pytest.mark.asyncio
    async def test_dashboard_status_monitoring(self, dashboard_manager):
        """Test dashboard status monitoring."""
        await dashboard_manager.initialize()
        
        # Get dashboard status
        status = await dashboard_manager.get_dashboard_status()
        
        assert status is not None
        assert 'dashboards_configured' in status
        assert 'grafana_connected' in status
        assert 'last_update' in status
        
        # Should track dashboard configurations
        assert status['dashboards_configured'] >= 4  # Standard dashboards


if __name__ == "__main__":
    # Run tests
    import subprocess
    
    print("Running Performance Monitoring Components Unit Tests...")
    result = subprocess.run([
        'python', '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--asyncio-mode=auto'
    ], cwd=Path(__file__).parent.parent.parent)
    
    exit(result.returncode)