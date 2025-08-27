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

# Placeholder imports - will be implemented in subsequent phases
# from .predictive_business_model import PredictiveBusinessModel, BusinessForecast
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
    # Future components for Epic 5 phase 4:
    # "PredictiveBusinessModel",
    # "BusinessForecast",
    # "BusinessIntelligenceEngine"
]