"""
Business Intelligence Core Services

Epic 5: Business Intelligence & Analytics Engine
Transforms operational data into actionable business insights.
"""

from .executive_dashboard import ExecutiveDashboard, BusinessMetrics

# Placeholder imports - will be implemented in subsequent phases
# from .user_behavior_tracker import UserBehaviorTracker, UserJourney
# from .agent_performance_analyzer import AgentPerformanceAnalyzer, AgentInsights
# from .predictive_business_model import PredictiveBusinessModel, BusinessForecast
# from .business_intelligence_engine import BusinessIntelligenceEngine

__all__ = [
    "ExecutiveDashboard",
    "BusinessMetrics"
    # Future components for Epic 5 phases 2-4:
    # "UserBehaviorTracker",
    # "UserJourney",
    # "AgentPerformanceAnalyzer", 
    # "AgentInsights",
    # "PredictiveBusinessModel",
    # "BusinessForecast",
    # "BusinessIntelligenceEngine"
]