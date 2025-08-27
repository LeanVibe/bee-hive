"""
Business Intelligence Core Services

Epic 5: Business Intelligence & Analytics Engine
Transforms operational data into actionable business insights.
"""

from .executive_dashboard import ExecutiveDashboard, BusinessMetrics
from .user_behavior_analytics import (
    UserBehaviorAnalytics, UserBehaviorTracker, UserJourneyAnalyzer,
    FeatureUsageAnalyzer, SessionAnalyzer, UserJourney, UserBehaviorMetrics,
    get_user_behavior_analytics
)
from .agent_performance_insights import (
    AgentPerformanceAnalyzer, AgentOptimizationEngine, AgentCapacityPlanner,
    AgentBenchmarkTracker, AgentEfficiencyScore, AgentOptimizationRecommendation,
    AgentCapacityInsights, AgentBenchmarkMetrics,
    get_agent_performance_analyzer, get_agent_optimization_engine,
    get_agent_capacity_planner, get_agent_benchmark_tracker
)
from .predictive_business_modeling import (
    PredictiveAnalyticsEngine, BusinessGrowthModeler, CapacityForecastingEngine,
    AnomalyDetectionEngine, PredictiveBusinessModelingService,
    PredictiveMetric, BusinessGrowthForecast, CapacityPrediction, AnomalyAlert,
    TrendDirection, ForecastAccuracy,
    get_predictive_analytics_engine, get_business_growth_modeler,
    get_capacity_forecasting_engine, get_anomaly_detection_engine,
    get_predictive_business_modeling_service
)

# Future components for additional phases:
# from .business_intelligence_engine import BusinessIntelligenceEngine

__all__ = [
    "ExecutiveDashboard",
    "BusinessMetrics",
    # Epic 5 Phase 2: User Behavior Analytics (✅ Complete)
    "UserBehaviorAnalytics",
    "UserBehaviorTracker", 
    "UserJourneyAnalyzer",
    "FeatureUsageAnalyzer",
    "SessionAnalyzer",
    "UserJourney",
    "UserBehaviorMetrics",
    "get_user_behavior_analytics",
    # Epic 5 Phase 3: Agent Performance Insights (✅ Complete)
    "AgentPerformanceAnalyzer",
    "AgentOptimizationEngine",
    "AgentCapacityPlanner",
    "AgentBenchmarkTracker",
    "AgentEfficiencyScore",
    "AgentOptimizationRecommendation", 
    "AgentCapacityInsights",
    "AgentBenchmarkMetrics",
    "get_agent_performance_analyzer",
    "get_agent_optimization_engine",
    "get_agent_capacity_planner",
    "get_agent_benchmark_tracker",
    # Epic 5 Phase 4: Predictive Business Modeling (✅ Complete)
    "PredictiveAnalyticsEngine",
    "BusinessGrowthModeler",
    "CapacityForecastingEngine",
    "AnomalyDetectionEngine",
    "PredictiveBusinessModelingService",
    "PredictiveMetric",
    "BusinessGrowthForecast", 
    "CapacityPrediction",
    "AnomalyAlert",
    "TrendDirection",
    "ForecastAccuracy",
    "get_predictive_analytics_engine",
    "get_business_growth_modeler",
    "get_capacity_forecasting_engine",
    "get_anomaly_detection_engine",
    "get_predictive_business_modeling_service",
    # Future components for additional phases:
    # "BusinessIntelligenceEngine"
]