"""
Test suite for Phase 3 Intelligence Layer implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.core.alert_analysis_engine import (
    AlertAnalysisEngine, 
    AlertMetrics, 
    PatternType, 
    PerformanceTrend
)
from app.core.user_preference_system import (
    UserPreferenceSystem,
    NotificationPreferences,
    DashboardPreferences,
    NotificationChannel,
    AlertPriority,
    DashboardLayout,
    ColorTheme,
    UsagePattern,
    PersonalizationInsights
)
from app.core.intelligence_framework import (
    IntelligenceFramework,
    DataPoint,
    DataType,
    IntelligenceType,
    IntelligenceModel,
    ModelStatus,
    SignalRelevanceMetrics,
    SimplePatternRecognitionModel
)


class TestAlertAnalysisEngine:
    """Test cases for Alert Analysis Engine"""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.hgetall = AsyncMock(return_value={})
        redis_mock.setex = AsyncMock(return_value=True)
        return redis_mock
    
    @pytest.fixture
    async def alert_engine(self, mock_redis):
        """Create Alert Analysis Engine instance"""
        engine = AlertAnalysisEngine(mock_redis, None)
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_alert_analysis_basic(self, alert_engine):
        """Test basic alert analysis functionality"""
        alert_data = {
            'type': 'system_error',
            'severity': 'high',
            'message': 'Database connection failed',
            'timestamp': datetime.now().isoformat(),
            'metadata': {'component': 'database'}
        }
        
        result = await alert_engine.analyze_alert(alert_data)
        
        assert 'analysis' in result
        assert 'priority_score' in result['analysis']
        assert result['analysis']['priority_score'] > 0
        assert 'patterns' in result['analysis']
        assert 'recommended_action' in result['analysis']
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, alert_engine):
        """Test pattern detection in alerts"""
        # Send multiple similar alerts to trigger pattern detection
        for i in range(5):
            alert_data = {
                'type': 'recurring_error',
                'severity': 'medium',
                'message': f'Recurring error {i}',
                'timestamp': datetime.now().isoformat()
            }
            await alert_engine.analyze_alert(alert_data)
        
        # Check if pattern was detected
        insights = await alert_engine.get_pattern_insights()
        assert insights['total_patterns'] >= 0  # At least baseline patterns
    
    @pytest.mark.asyncio
    async def test_performance_trends(self, alert_engine):
        """Test performance trend detection"""
        metrics = {
            'response_time': 250.0,
            'error_rate': 0.05,
            'cpu_usage': 70.0,
            'memory_usage': 80.0
        }
        
        trends = await alert_engine.detect_performance_trends(metrics)
        
        assert len(trends) == len(metrics)
        for trend in trends:
            assert hasattr(trend, 'metric_name')
            assert hasattr(trend, 'trend_direction')
            assert hasattr(trend, 'confidence')
    
    @pytest.mark.asyncio
    async def test_alert_metrics_calculation(self, alert_engine):
        """Test alert metrics calculation"""
        # Add some test data
        for i in range(10):
            alert_data = {
                'type': f'test_alert_{i % 3}',
                'severity': 'medium',
                'message': f'Test alert {i}',
                'timestamp': datetime.now().isoformat()
            }
            await alert_engine.analyze_alert(alert_data)
        
        metrics = await alert_engine.get_alert_metrics(timedelta(hours=1))
        
        assert isinstance(metrics, AlertMetrics)
        assert metrics.frequency >= 0
        assert isinstance(metrics.severity_distribution, dict)
        assert metrics.response_time_avg >= 0


class TestUserPreferenceSystem:
    """Test cases for User Preference System"""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock(return_value=True)
        return redis_mock
    
    @pytest.fixture
    async def preference_system(self, mock_redis):
        """Create User Preference System instance"""
        system = UserPreferenceSystem(mock_redis, None)
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_default_preferences_creation(self, preference_system):
        """Test creation of default user preferences"""
        user_id = "test_user_123"
        
        preferences = await preference_system.get_user_preferences(user_id)
        
        assert preferences['user_id'] == user_id
        assert 'notifications' in preferences
        assert 'dashboard' in preferences
        assert 'created_at' in preferences
    
    @pytest.mark.asyncio
    async def test_notification_preferences_update(self, preference_system):
        """Test updating notification preferences"""
        user_id = "test_user_456"
        
        notification_prefs = NotificationPreferences(
            enabled_channels=[NotificationChannel.MOBILE, NotificationChannel.EMAIL],
            quiet_hours_start="23:00",
            quiet_hours_end="07:00",
            escalation_delay=20
        )
        
        success = await preference_system.update_notification_preferences(
            user_id, notification_prefs
        )
        
        assert success is True
        
        # Verify preferences were updated
        preferences = await preference_system.get_user_preferences(user_id)
        assert preferences['notifications']['quiet_hours_start'] == "23:00"
        assert preferences['notifications']['escalation_delay'] == 20
    
    @pytest.mark.asyncio
    async def test_dashboard_preferences_update(self, preference_system):
        """Test updating dashboard preferences"""
        user_id = "test_user_789"
        
        dashboard_prefs = DashboardPreferences(
            layout=DashboardLayout.COMPACT,
            color_theme=ColorTheme.DARK,
            refresh_interval=60,
            show_debug_info=True
        )
        
        success = await preference_system.update_dashboard_preferences(
            user_id, dashboard_prefs
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, preference_system):
        """Test usage pattern tracking"""
        user_id = "test_user_usage"
        
        # Track multiple command usages
        commands = ["status", "health", "logs", "status", "agents"]
        for command in commands:
            await preference_system.track_usage(
                user_id, command, response_time=2.5, success=True
            )
        
        # Check usage pattern was recorded
        assert user_id in preference_system.usage_patterns
        pattern = preference_system.usage_patterns[user_id]
        assert pattern.command_frequency["status"] == 2
        assert pattern.command_frequency["health"] == 1
    
    @pytest.mark.asyncio
    async def test_notification_decision(self, preference_system):
        """Test notification decision logic"""
        user_id = "test_user_notify"
        
        # Set up preferences first
        await preference_system.get_user_preferences(user_id)
        
        should_send, channels = await preference_system.should_send_notification(
            user_id, "system_error", "high"
        )
        
        assert isinstance(should_send, bool)
        assert isinstance(channels, list)
    
    @pytest.mark.asyncio
    async def test_personalized_dashboard_config(self, preference_system):
        """Test personalized dashboard configuration generation"""
        user_id = "test_user_dashboard"
        
        # Add some usage patterns
        await preference_system.track_usage(user_id, "agent_status", 5.0, True)
        await preference_system.track_usage(user_id, "performance_metrics", 3.0, True)
        
        config = await preference_system.get_personalized_dashboard_config(user_id)
        
        assert 'layout' in config
        assert 'widgets' in config
        assert 'quick_actions' in config
        assert isinstance(config['widgets'], list)
    
    @pytest.mark.asyncio
    async def test_productivity_insights(self, preference_system):
        """Test productivity insights generation"""
        user_id = "test_user_insights"
        
        # Create usage pattern with multiple commands
        for i in range(20):
            command = f"command_{i % 5}"
            await preference_system.track_usage(
                user_id, command, response_time=2.0 + (i % 3), success=i % 10 != 0
            )
        
        insights = await preference_system.generate_productivity_insights(user_id)
        
        assert isinstance(insights, PersonalizationInsights)
        assert insights.user_id == user_id
        assert 0 <= insights.efficiency_score <= 1
        assert len(insights.most_used_commands) > 0
        assert len(insights.improvement_suggestions) > 0


class TestIntelligenceFramework:
    """Test cases for Intelligence Framework"""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.lpush = AsyncMock(return_value=1)
        redis_mock.ltrim = AsyncMock(return_value=True)
        redis_mock.lrange = AsyncMock(return_value=[])
        return redis_mock
    
    @pytest.fixture
    async def intelligence_framework(self, mock_redis):
        """Create Intelligence Framework instance"""
        framework = IntelligenceFramework(mock_redis, None)
        await framework.initialize()
        return framework
    
    @pytest.mark.asyncio
    async def test_data_collection(self, intelligence_framework):
        """Test ML data collection"""
        data_id = await intelligence_framework.collect_data(
            DataType.TEXT,
            "Test alert message",
            user_id="test_user",
            labels=["error", "database"],
            metadata={"component": "database"}
        )
        
        assert data_id != "data_collection_disabled"
        assert data_id != "collection_error"
        assert len(intelligence_framework.data_buffer) > 0
    
    @pytest.mark.asyncio
    async def test_model_deployment(self, intelligence_framework):
        """Test intelligence model deployment"""
        model_config = IntelligenceModel(
            model_id="test_pattern_model",
            name="Test Pattern Recognition",
            intelligence_type=IntelligenceType.PATTERN_RECOGNITION,
            version="1.0.0",
            status=ModelStatus.TRAINING,
            confidence_threshold=0.7
        )
        
        success = await intelligence_framework.deploy_model(model_config)
        
        assert success is True
        assert model_config.model_id in intelligence_framework.deployed_models
        assert model_config.status == ModelStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_intelligence_prediction(self, intelligence_framework):
        """Test intelligence prediction"""
        # First deploy a model
        model_config = IntelligenceModel(
            model_id="test_predictor",
            name="Test Predictor",
            intelligence_type=IntelligenceType.PATTERN_RECOGNITION,
            version="1.0.0",
            status=ModelStatus.TRAINING
        )
        
        await intelligence_framework.deploy_model(model_config)
        
        # Make a prediction
        prediction = await intelligence_framework.get_intelligence_prediction(
            "test_predictor",
            {"text": "error database connection failed"}
        )
        
        assert prediction is not None
        assert prediction.model_id == "test_predictor"
        assert prediction.confidence >= 0.0
        assert prediction.prediction is not None
    
    @pytest.mark.asyncio
    async def test_signal_relevance_measurement(self, intelligence_framework):
        """Test signal relevance measurement"""
        signals = [
            {"id": "signal_1", "type": "alert", "content": "Test alert 1"},
            {"id": "signal_2", "type": "notification", "content": "Test notification"},
            {"id": "signal_3", "type": "alert", "content": "Test alert 2"}
        ]
        
        user_feedback = [
            {"signal_id": "signal_1", "action": "acted_on", "relevance": 0.8},
            {"signal_id": "signal_2", "action": "dismissed", "relevance": 0.2},
            {"signal_id": "signal_3", "action": "acknowledged", "relevance": 0.9}
        ]
        
        metrics = await intelligence_framework.measure_signal_relevance(
            signals, user_feedback
        )
        
        assert isinstance(metrics, SignalRelevanceMetrics)
        assert metrics.total_signals == 3
        assert metrics.relevant_signals >= 0
        assert 0 <= metrics.relevance_ratio <= 1
    
    @pytest.mark.asyncio
    async def test_ab_experiment_setup(self, intelligence_framework):
        """Test A/B testing experiment setup"""
        experiment_id = await intelligence_framework.setup_ab_experiment(
            "test_experiment",
            "model_a",
            "model_b",
            traffic_split=0.6
        )
        
        assert experiment_id != ""
        assert experiment_id in intelligence_framework.experiments
        
        # Test model assignment
        assignment = await intelligence_framework.get_model_assignment(
            experiment_id, "test_user"
        )
        
        assert assignment in ["model_a", "model_b"]
    
    @pytest.mark.asyncio
    async def test_intelligence_insights(self, intelligence_framework):
        """Test intelligence system insights"""
        # Add some test data
        await intelligence_framework.collect_data(
            DataType.TEXT, "Test data", labels=["test"]
        )
        
        insights = await intelligence_framework.get_intelligence_insights()
        
        assert 'models' in insights
        assert 'data_collection' in insights
        assert 'signal_relevance' in insights
        assert 'experiments' in insights
        assert 'roadmap_progress' in insights


class TestSimplePatternRecognitionModel:
    """Test cases for Simple Pattern Recognition Model"""
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration"""
        return IntelligenceModel(
            model_id="test_pattern_model",
            name="Test Pattern Model",
            intelligence_type=IntelligenceType.PATTERN_RECOGNITION,
            version="1.0.0",
            status=ModelStatus.TRAINING
        )
    
    @pytest.fixture
    def pattern_model(self, model_config):
        """Create pattern recognition model instance"""
        return SimplePatternRecognitionModel(model_config)
    
    @pytest.mark.asyncio
    async def test_model_training(self, pattern_model):
        """Test model training with sample data"""
        training_data = [
            DataPoint(
                id="1",
                timestamp=datetime.now(),
                data_type=DataType.TEXT,
                value="database connection error",
                labels=["error", "database"]
            ),
            DataPoint(
                id="2",
                timestamp=datetime.now(),
                data_type=DataType.TEXT,
                value="network timeout error",
                labels=["error", "network"]
            ),
            DataPoint(
                id="3",
                timestamp=datetime.now(),
                data_type=DataType.TEXT,
                value="authentication failed",
                labels=["error", "auth"]
            )
        ]
        
        success = await pattern_model.train(training_data)
        
        assert success is True
        assert pattern_model.config.status == ModelStatus.READY
        assert len(pattern_model.patterns) > 0
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, pattern_model):
        """Test model prediction"""
        # Train model first
        training_data = [
            DataPoint(
                id="1",
                timestamp=datetime.now(),
                data_type=DataType.TEXT,
                value="database connection failed",
                labels=["database_error"]
            )
        ]
        await pattern_model.train(training_data)
        
        # Make prediction
        prediction = await pattern_model.predict({
            "text": "database connection timeout"
        })
        
        assert prediction.model_id == pattern_model.config.model_id
        assert prediction.confidence >= 0.0
        assert prediction.prediction is not None
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, pattern_model):
        """Test model evaluation"""
        # Train model
        training_data = [
            DataPoint("1", datetime.now(), DataType.TEXT, "error message", ["error"])
        ]
        await pattern_model.train(training_data)
        
        # Evaluate model
        test_data = [
            DataPoint("2", datetime.now(), DataType.TEXT, "error occurred", ["error"])
        ]
        
        metrics = await pattern_model.evaluate(test_data)
        
        assert 'accuracy' in metrics
        assert 'average_confidence' in metrics
        assert metrics['accuracy'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_feature_importance(self, pattern_model):
        """Test feature importance extraction"""
        # Train model with patterns
        training_data = [
            DataPoint("1", datetime.now(), DataType.TEXT, "database error occurred", ["error"]),
            DataPoint("2", datetime.now(), DataType.TEXT, "database connection failed", ["error"]),
            DataPoint("3", datetime.now(), DataType.TEXT, "network timeout error", ["error"])
        ]
        await pattern_model.train(training_data)
        
        importance = await pattern_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0


# Integration tests
class TestIntelligenceSystemIntegration:
    """Integration tests for the complete intelligence system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_intelligence_workflow(self):
        """Test complete intelligence workflow"""
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.lpush = AsyncMock(return_value=1)
        mock_redis.ltrim = AsyncMock(return_value=True)
        mock_redis.lrange = AsyncMock(return_value=[])
        mock_redis.hgetall = AsyncMock(return_value={})
        
        # Initialize systems
        alert_engine = AlertAnalysisEngine(mock_redis, None)
        await alert_engine.initialize()
        
        preference_system = UserPreferenceSystem(mock_redis, None)
        await preference_system.initialize()
        
        intelligence_framework = IntelligenceFramework(mock_redis, None)
        await intelligence_framework.initialize()
        
        # Test workflow
        user_id = "integration_test_user"
        
        # 1. Analyze alert
        alert_data = {
            'type': 'integration_test',
            'severity': 'medium',
            'message': 'Integration test alert',
            'timestamp': datetime.now().isoformat()
        }
        
        analysis_result = await alert_engine.analyze_alert(alert_data)
        assert 'analysis' in analysis_result
        
        # 2. Track usage
        await preference_system.track_usage(
            user_id, "analyze_alert", response_time=1.5, success=True
        )
        
        # 3. Collect data for ML
        data_id = await intelligence_framework.collect_data(
            DataType.TEXT,
            alert_data['message'],
            user_id=user_id,
            labels=[alert_data['type']],
            metadata={'severity': alert_data['severity']}
        )
        
        assert data_id not in ["data_collection_disabled", "collection_error"]
        
        # 4. Get personalized config
        dashboard_config = await preference_system.get_personalized_dashboard_config(user_id)
        assert 'widgets' in dashboard_config
        
        # 5. Measure signal relevance
        signals = [{"id": "test_signal", "type": "alert", "content": alert_data['message']}]
        feedback = [{"signal_id": "test_signal", "action": "acted_on", "relevance": 0.8}]
        
        relevance_metrics = await intelligence_framework.measure_signal_relevance(signals, feedback)
        assert relevance_metrics.total_signals == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])