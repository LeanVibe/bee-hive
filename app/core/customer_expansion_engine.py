"""
Customer Expansion and Retention Engine
Comprehensive customer success, retention, and expansion automation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from decimal import Decimal
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
import redis.asyncio as redis
import aiohttp

from app.core.database import get_async_session
from app.core.redis import get_redis_client


class CustomerHealthStatus(Enum):
    """Customer health status levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 75-89
    FAIR = "fair"  # 60-74
    AT_RISK = "at_risk"  # 40-59
    CRITICAL = "critical"  # 0-39


class ExpansionReadiness(Enum):
    """Customer expansion readiness levels."""
    READY = "ready"  # High satisfaction, successful projects
    POTENTIAL = "potential"  # Good results, room for growth
    NURTURING = "nurturing"  # Need relationship building
    NOT_READY = "not_ready"  # Issues to resolve first


class RetentionStrategy(Enum):
    """Retention strategy types."""
    HIGH_SATISFACTION_EXPANSION = "high_satisfaction_expansion"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    AT_RISK_RECOVERY = "at_risk_recovery"
    CRITICAL_INTERVENTION = "critical_intervention"


class ExpansionOpportunity(Enum):
    """Types of expansion opportunities."""
    MVP_ENHANCEMENT = "mvp_enhancement"
    SCALING_SERVICES = "scaling_services"
    ADDITIONAL_PRODUCTS = "additional_products"
    TEAM_AUGMENTATION = "team_augmentation"
    MAINTENANCE_CONTRACT = "maintenance_contract"
    CONSULTING_SERVICES = "consulting_services"
    ENTERPRISE_UPGRADE = "enterprise_upgrade"


@dataclass
class CustomerHealthScore:
    """Customer health score breakdown."""
    customer_id: str
    overall_score: float
    component_scores: Dict[str, float]
    health_status: CustomerHealthStatus
    trending_direction: str  # "improving", "stable", "declining"
    risk_factors: List[str]
    positive_indicators: List[str]
    last_calculated: datetime
    next_review_date: datetime


@dataclass
class ExpansionOpportunityRecord:
    """Expansion opportunity record."""
    opportunity_id: str
    customer_id: str
    opportunity_type: ExpansionOpportunity
    title: str
    description: str
    estimated_value: Decimal
    probability: float  # 0.0 to 1.0
    timeline_months: int
    requirements: List[str]
    stakeholders: List[str]
    competitive_factors: List[str]
    success_factors: List[str]
    next_actions: List[str]
    created_at: datetime
    status: str = "identified"  # identified, qualified, proposal, negotiation, closed


@dataclass
class RetentionAction:
    """Customer retention action."""
    action_id: str
    customer_id: str
    action_type: str
    title: str
    description: str
    priority: str  # critical, high, medium, low
    estimated_impact: float  # Expected retention probability improvement
    cost: Decimal
    timeline_days: int
    responsible_team: str
    success_metrics: List[str]
    completion_criteria: List[str]
    status: str = "planned"  # planned, in_progress, completed, cancelled
    scheduled_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None


@dataclass
class CustomerExpansionProfile:
    """Comprehensive customer expansion profile."""
    customer_id: str
    customer_name: str
    current_services: List[str]
    health_score: CustomerHealthScore
    expansion_readiness: ExpansionReadiness
    expansion_opportunities: List[ExpansionOpportunityRecord]
    retention_actions: List[RetentionAction]
    relationship_strength: float  # 0.0 to 10.0
    decision_maker_engagement: float  # 0.0 to 10.0
    budget_capacity: str  # "high", "medium", "low"
    competitive_pressure: float  # 0.0 to 10.0
    churn_risk_probability: float  # 0.0 to 1.0
    lifetime_value: Decimal
    expansion_potential_value: Decimal
    last_interaction: datetime
    next_touchpoint: datetime


class CustomerHealthAnalyzer:
    """Advanced customer health analysis engine."""
    
    HEALTH_COMPONENTS = {
        "satisfaction_metrics": {
            "weight": 0.25,
            "sub_components": {
                "overall_satisfaction": 0.4,
                "nps_score": 0.3,
                "support_satisfaction": 0.2,
                "stakeholder_sentiment": 0.1
            }
        },
        "engagement_metrics": {
            "weight": 0.20,
            "sub_components": {
                "platform_usage": 0.3,
                "feature_adoption": 0.3,
                "communication_frequency": 0.2,
                "feedback_participation": 0.2
            }
        },
        "project_success_metrics": {
            "weight": 0.25,
            "sub_components": {
                "delivery_success_rate": 0.4,
                "timeline_adherence": 0.3,
                "quality_scores": 0.3
            }
        },
        "business_outcome_metrics": {
            "weight": 0.20,
            "sub_components": {
                "roi_achievement": 0.4,
                "business_impact": 0.3,
                "goal_attainment": 0.3
            }
        },
        "relationship_metrics": {
            "weight": 0.10,
            "sub_components": {
                "stakeholder_relationships": 0.4,
                "champion_strength": 0.3,
                "escalation_frequency": 0.3
            }
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def analyze_customer_health(
        self,
        customer_id: str,
        customer_data: Dict[str, Any]
    ) -> CustomerHealthScore:
        """Comprehensive customer health analysis."""
        
        self.logger.info(f"Analyzing customer health: {customer_id}")
        
        component_scores = {}
        
        # Calculate each health component
        for component_name, component_config in self.HEALTH_COMPONENTS.items():
            component_score = await self._calculate_component_score(
                component_name, 
                component_config,
                customer_data.get(component_name, {})
            )
            component_scores[component_name] = component_score
        
        # Calculate overall health score
        overall_score = sum(
            component_scores[component] * config["weight"]
            for component, config in self.HEALTH_COMPONENTS.items()
        )
        
        # Determine health status
        if overall_score >= 90:
            health_status = CustomerHealthStatus.EXCELLENT
        elif overall_score >= 75:
            health_status = CustomerHealthStatus.GOOD
        elif overall_score >= 60:
            health_status = CustomerHealthStatus.FAIR
        elif overall_score >= 40:
            health_status = CustomerHealthStatus.AT_RISK
        else:
            health_status = CustomerHealthStatus.CRITICAL
        
        # Analyze trending direction
        trending_direction = await self._analyze_trending_direction(customer_id, overall_score)
        
        # Identify risk factors and positive indicators
        risk_factors = await self._identify_risk_factors(customer_data, component_scores)
        positive_indicators = await self._identify_positive_indicators(customer_data, component_scores)
        
        health_score = CustomerHealthScore(
            customer_id=customer_id,
            overall_score=overall_score,
            component_scores=component_scores,
            health_status=health_status,
            trending_direction=trending_direction,
            risk_factors=risk_factors,
            positive_indicators=positive_indicators,
            last_calculated=datetime.now(),
            next_review_date=datetime.now() + timedelta(days=30)
        )
        
        # Store health score
        await self._store_health_score(health_score)
        
        return health_score
    
    async def _calculate_component_score(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        component_data: Dict[str, Any]
    ) -> float:
        """Calculate score for individual health component."""
        
        if component_name == "satisfaction_metrics":
            return await self._calculate_satisfaction_score(component_config, component_data)
        elif component_name == "engagement_metrics":
            return await self._calculate_engagement_score(component_config, component_data)
        elif component_name == "project_success_metrics":
            return await self._calculate_project_success_score(component_config, component_data)
        elif component_name == "business_outcome_metrics":
            return await self._calculate_business_outcome_score(component_config, component_data)
        elif component_name == "relationship_metrics":
            return await self._calculate_relationship_score(component_config, component_data)
        
        return 0.0
    
    async def _calculate_satisfaction_score(
        self,
        config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> float:
        """Calculate customer satisfaction component score."""
        
        sub_scores = {}
        
        # Overall satisfaction (1-10 scale)
        overall_satisfaction = data.get("overall_satisfaction", 5.0)
        sub_scores["overall_satisfaction"] = (overall_satisfaction / 10.0) * 100
        
        # NPS score (-100 to 100 scale)
        nps_score = data.get("nps_score", 0)
        sub_scores["nps_score"] = ((nps_score + 100) / 200) * 100
        
        # Support satisfaction (1-10 scale)
        support_satisfaction = data.get("support_satisfaction", 7.0)
        sub_scores["support_satisfaction"] = (support_satisfaction / 10.0) * 100
        
        # Stakeholder sentiment analysis (0-100 scale)
        stakeholder_sentiment = data.get("stakeholder_sentiment", 70.0)
        sub_scores["stakeholder_sentiment"] = stakeholder_sentiment
        
        # Weighted average
        weighted_score = sum(
            sub_scores[sub_component] * weight
            for sub_component, weight in config["sub_components"].items()
        )
        
        return weighted_score
    
    async def _calculate_engagement_score(
        self,
        config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> float:
        """Calculate customer engagement component score."""
        
        sub_scores = {}
        
        # Platform usage (percentage of expected usage)
        platform_usage = data.get("platform_usage_percentage", 50.0)
        sub_scores["platform_usage"] = min(platform_usage, 100.0)
        
        # Feature adoption (percentage of available features used)
        feature_adoption = data.get("feature_adoption_percentage", 40.0)
        sub_scores["feature_adoption"] = min(feature_adoption, 100.0)
        
        # Communication frequency (interactions per month vs expected)
        comm_frequency = data.get("communication_frequency_score", 60.0)
        sub_scores["communication_frequency"] = min(comm_frequency, 100.0)
        
        # Feedback participation (percentage participation in surveys/feedback)
        feedback_participation = data.get("feedback_participation_percentage", 30.0)
        sub_scores["feedback_participation"] = min(feedback_participation, 100.0)
        
        # Weighted average
        weighted_score = sum(
            sub_scores[sub_component] * weight
            for sub_component, weight in config["sub_components"].items()
        )
        
        return weighted_score


class ExpansionOpportunityEngine:
    """Engine for identifying and qualifying expansion opportunities."""
    
    OPPORTUNITY_TRIGGERS = {
        "mvp_enhancement": {
            "conditions": [
                "mvp_successfully_delivered",
                "customer_satisfaction >= 8.0",
                "feature_requests_pending"
            ],
            "estimated_value_multiplier": 0.3,  # 30% of original project value
            "probability_base": 0.7
        },
        "scaling_services": {
            "conditions": [
                "high_platform_usage",
                "team_growth_indicators",
                "performance_bottlenecks"
            ],
            "estimated_value_multiplier": 0.8,
            "probability_base": 0.6
        },
        "team_augmentation": {
            "conditions": [
                "hiring_challenges",
                "project_backlog_growth",
                "skill_gaps_identified"
            ],
            "estimated_value_multiplier": 1.2,
            "probability_base": 0.5
        },
        "enterprise_upgrade": {
            "conditions": [
                "company_growth >= 50%",
                "compliance_requirements_emerging",
                "enterprise_features_requested"
            ],
            "estimated_value_multiplier": 2.0,
            "probability_base": 0.4
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def identify_expansion_opportunities(
        self,
        customer_id: str,
        customer_profile: Dict[str, Any],
        project_history: List[Dict[str, Any]]
    ) -> List[ExpansionOpportunityRecord]:
        """Identify potential expansion opportunities for a customer."""
        
        self.logger.info(f"Identifying expansion opportunities for customer: {customer_id}")
        
        opportunities = []
        
        # Analyze each opportunity type
        for opportunity_type, trigger_config in self.OPPORTUNITY_TRIGGERS.items():
            opportunity_score = await self._evaluate_opportunity_triggers(
                opportunity_type,
                trigger_config,
                customer_profile,
                project_history
            )
            
            if opportunity_score["qualified"]:
                opportunity = await self._create_expansion_opportunity(
                    customer_id,
                    opportunity_type,
                    opportunity_score,
                    customer_profile
                )
                opportunities.append(opportunity)
        
        # Sort by estimated value and probability
        opportunities.sort(
            key=lambda x: float(x.estimated_value) * x.probability,
            reverse=True
        )
        
        return opportunities
    
    async def _evaluate_opportunity_triggers(
        self,
        opportunity_type: str,
        trigger_config: Dict[str, Any],
        customer_profile: Dict[str, Any],
        project_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate triggers for a specific opportunity type."""
        
        conditions_met = 0
        total_conditions = len(trigger_config["conditions"])
        condition_details = {}
        
        for condition in trigger_config["conditions"]:
            met = await self._check_condition(condition, customer_profile, project_history)
            condition_details[condition] = met
            if met:
                conditions_met += 1
        
        # Calculate qualification score
        qualification_score = conditions_met / total_conditions
        
        # Adjust probability based on qualification
        probability = trigger_config["probability_base"] * qualification_score
        
        return {
            "qualified": qualification_score >= 0.6,  # 60% of conditions must be met
            "qualification_score": qualification_score,
            "probability": probability,
            "conditions_met": conditions_met,
            "total_conditions": total_conditions,
            "condition_details": condition_details
        }
    
    async def _create_expansion_opportunity(
        self,
        customer_id: str,
        opportunity_type: str,
        opportunity_score: Dict[str, Any],
        customer_profile: Dict[str, Any]
    ) -> ExpansionOpportunityRecord:
        """Create expansion opportunity record."""
        
        opportunity_id = f"exp_{customer_id}_{opportunity_type}_{datetime.now().strftime('%Y%m%d')}"
        
        # Calculate estimated value
        base_value = customer_profile.get("project_values", {}).get("latest_project", 100000)
        multiplier = self.OPPORTUNITY_TRIGGERS[opportunity_type]["estimated_value_multiplier"]
        estimated_value = Decimal(str(base_value * multiplier))
        
        # Generate opportunity details
        opportunity_details = await self._generate_opportunity_details(
            opportunity_type, customer_profile
        )
        
        return ExpansionOpportunityRecord(
            opportunity_id=opportunity_id,
            customer_id=customer_id,
            opportunity_type=ExpansionOpportunity(opportunity_type),
            title=opportunity_details["title"],
            description=opportunity_details["description"],
            estimated_value=estimated_value,
            probability=opportunity_score["probability"],
            timeline_months=opportunity_details["timeline_months"],
            requirements=opportunity_details["requirements"],
            stakeholders=opportunity_details["stakeholders"],
            competitive_factors=opportunity_details["competitive_factors"],
            success_factors=opportunity_details["success_factors"],
            next_actions=opportunity_details["next_actions"],
            created_at=datetime.now()
        )


class CustomerRetentionEngine:
    """Advanced customer retention strategy engine."""
    
    RETENTION_STRATEGIES = {
        "high_satisfaction_expansion": {
            "trigger_conditions": {
                "health_score": 85.0,
                "satisfaction_score": 8.5,
                "project_success_rate": 90.0
            },
            "actions": [
                {
                    "type": "dedicated_account_manager",
                    "priority": "high",
                    "impact": 0.15,
                    "cost": 10000,
                    "timeline_days": 30
                },
                {
                    "type": "expansion_opportunity_presentation",
                    "priority": "high",
                    "impact": 0.25,
                    "cost": 5000,
                    "timeline_days": 14
                },
                {
                    "type": "strategic_partnership_discussion",
                    "priority": "medium",
                    "impact": 0.20,
                    "cost": 3000,
                    "timeline_days": 60
                }
            ]
        },
        "moderate_improvement": {
            "trigger_conditions": {
                "health_score": 65.0,
                "satisfaction_score": 7.0,
                "engagement_issues": True
            },
            "actions": [
                {
                    "type": "enhanced_support_package",
                    "priority": "high",
                    "impact": 0.20,
                    "cost": 8000,
                    "timeline_days": 30
                },
                {
                    "type": "success_manager_intervention",
                    "priority": "high",
                    "impact": 0.15,
                    "cost": 5000,
                    "timeline_days": 14
                },
                {
                    "type": "training_and_optimization",
                    "priority": "medium",
                    "impact": 0.10,
                    "cost": 3000,
                    "timeline_days": 45
                }
            ]
        },
        "at_risk_recovery": {
            "trigger_conditions": {
                "health_score": 45.0,
                "satisfaction_score": 6.0,
                "escalations_count": 2
            },
            "actions": [
                {
                    "type": "executive_intervention",
                    "priority": "critical",
                    "impact": 0.30,
                    "cost": 15000,
                    "timeline_days": 7
                },
                {
                    "type": "comprehensive_project_audit",
                    "priority": "critical",
                    "impact": 0.25,
                    "cost": 12000,
                    "timeline_days": 14
                },
                {
                    "type": "relationship_reset_program",
                    "priority": "high",
                    "impact": 0.20,
                    "cost": 10000,
                    "timeline_days": 30
                }
            ]
        },
        "critical_intervention": {
            "trigger_conditions": {
                "health_score": 25.0,
                "churn_risk": 0.8,
                "contract_renewal_risk": True
            },
            "actions": [
                {
                    "type": "ceo_intervention",
                    "priority": "critical",
                    "impact": 0.40,
                    "cost": 25000,
                    "timeline_days": 3
                },
                {
                    "type": "emergency_project_recovery",
                    "priority": "critical",
                    "impact": 0.35,
                    "cost": 50000,
                    "timeline_days": 7
                },
                {
                    "type": "full_service_credit_package",
                    "priority": "critical",
                    "impact": 0.30,
                    "cost": 100000,
                    "timeline_days": 1
                }
            ]
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def develop_retention_strategy(
        self,
        customer_id: str,
        health_score: CustomerHealthScore,
        customer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Develop comprehensive retention strategy based on customer health."""
        
        self.logger.info(f"Developing retention strategy for customer: {customer_id}")
        
        # Determine appropriate strategy
        strategy_type = await self._determine_retention_strategy(health_score, customer_profile)
        
        # Get strategy configuration
        strategy_config = self.RETENTION_STRATEGIES[strategy_type.value]
        
        # Generate retention actions
        retention_actions = []
        for action_config in strategy_config["actions"]:
            action = await self._create_retention_action(
                customer_id, action_config, customer_profile
            )
            retention_actions.append(action)
        
        # Calculate strategy effectiveness prediction
        effectiveness_prediction = await self._predict_strategy_effectiveness(
            strategy_type, retention_actions, customer_profile
        )
        
        # Generate implementation timeline
        implementation_timeline = await self._generate_implementation_timeline(retention_actions)
        
        return {
            "customer_id": customer_id,
            "strategy_type": strategy_type.value,
            "retention_actions": retention_actions,
            "effectiveness_prediction": effectiveness_prediction,
            "implementation_timeline": implementation_timeline,
            "total_investment": sum(action.cost for action in retention_actions),
            "expected_retention_improvement": effectiveness_prediction["retention_improvement"],
            "roi_projection": effectiveness_prediction["roi_projection"],
            "next_review_date": datetime.now() + timedelta(days=30)
        }
    
    async def _determine_retention_strategy(
        self,
        health_score: CustomerHealthScore,
        customer_profile: Dict[str, Any]
    ) -> RetentionStrategy:
        """Determine appropriate retention strategy based on customer health."""
        
        score = health_score.overall_score
        satisfaction = customer_profile.get("satisfaction_metrics", {}).get("overall_satisfaction", 5.0)
        escalations = customer_profile.get("escalations_count", 0)
        churn_risk = customer_profile.get("churn_risk_probability", 0.0)
        
        # Decision logic for strategy selection
        if score >= 85 and satisfaction >= 8.5:
            return RetentionStrategy.HIGH_SATISFACTION_EXPANSION
        elif score >= 65 and satisfaction >= 7.0:
            return RetentionStrategy.MODERATE_IMPROVEMENT
        elif score >= 45 or escalations >= 2:
            return RetentionStrategy.AT_RISK_RECOVERY
        else:
            return RetentionStrategy.CRITICAL_INTERVENTION
    
    async def _create_retention_action(
        self,
        customer_id: str,
        action_config: Dict[str, Any],
        customer_profile: Dict[str, Any]
    ) -> RetentionAction:
        """Create individual retention action."""
        
        action_id = f"ret_{customer_id}_{action_config['type']}_{datetime.now().strftime('%Y%m%d')}"
        
        # Generate action details
        action_details = await self._generate_action_details(
            action_config["type"], customer_profile
        )
        
        return RetentionAction(
            action_id=action_id,
            customer_id=customer_id,
            action_type=action_config["type"],
            title=action_details["title"],
            description=action_details["description"],
            priority=action_config["priority"],
            estimated_impact=action_config["impact"],
            cost=Decimal(str(action_config["cost"])),
            timeline_days=action_config["timeline_days"],
            responsible_team=action_details["responsible_team"],
            success_metrics=action_details["success_metrics"],
            completion_criteria=action_details["completion_criteria"],
            scheduled_date=datetime.now() + timedelta(days=1)
        )


class CustomerExpansionEngine:
    """Main customer expansion and retention orchestrator."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.health_analyzer: Optional[CustomerHealthAnalyzer] = None
        self.opportunity_engine: Optional[ExpansionOpportunityEngine] = None
        self.retention_engine: Optional[CustomerRetentionEngine] = None
    
    async def initialize(self):
        """Initialize the expansion engine and all components."""
        self.redis_client = await get_redis_client()
        self.health_analyzer = CustomerHealthAnalyzer(self.redis_client, self.logger)
        self.opportunity_engine = ExpansionOpportunityEngine(self.redis_client, self.logger)
        self.retention_engine = CustomerRetentionEngine(self.redis_client, self.logger)
        
        self.logger.info("Customer Expansion Engine initialized successfully")
    
    async def create_expansion_profile(
        self,
        customer_id: str,
        customer_data: Dict[str, Any]
    ) -> CustomerExpansionProfile:
        """Create comprehensive customer expansion profile."""
        
        self.logger.info(f"Creating expansion profile for customer: {customer_id}")
        
        # Analyze customer health
        health_score = await self.health_analyzer.analyze_customer_health(
            customer_id, customer_data
        )
        
        # Determine expansion readiness
        expansion_readiness = await self._assess_expansion_readiness(health_score, customer_data)
        
        # Identify expansion opportunities
        expansion_opportunities = await self.opportunity_engine.identify_expansion_opportunities(
            customer_id,
            customer_data.get("profile", {}),
            customer_data.get("project_history", [])
        )
        
        # Develop retention strategy
        retention_strategy = await self.retention_engine.develop_retention_strategy(
            customer_id, health_score, customer_data.get("profile", {})
        )
        
        # Calculate relationship metrics
        relationship_metrics = await self._calculate_relationship_metrics(customer_data)
        
        # Create expansion profile
        expansion_profile = CustomerExpansionProfile(
            customer_id=customer_id,
            customer_name=customer_data.get("customer_name", "Unknown"),
            current_services=customer_data.get("current_services", []),
            health_score=health_score,
            expansion_readiness=expansion_readiness,
            expansion_opportunities=expansion_opportunities,
            retention_actions=retention_strategy.get("retention_actions", []),
            relationship_strength=relationship_metrics["relationship_strength"],
            decision_maker_engagement=relationship_metrics["decision_maker_engagement"],
            budget_capacity=relationship_metrics["budget_capacity"],
            competitive_pressure=relationship_metrics["competitive_pressure"],
            churn_risk_probability=customer_data.get("churn_risk_probability", 0.0),
            lifetime_value=Decimal(str(customer_data.get("lifetime_value", 0))),
            expansion_potential_value=sum(opp.estimated_value for opp in expansion_opportunities),
            last_interaction=datetime.fromisoformat(customer_data.get("last_interaction", datetime.now().isoformat())),
            next_touchpoint=datetime.now() + timedelta(days=30)
        )
        
        # Store expansion profile
        await self._store_expansion_profile(expansion_profile)
        
        return expansion_profile
    
    async def execute_retention_actions(
        self,
        customer_id: str,
        action_ids: List[str]
    ) -> Dict[str, Any]:
        """Execute specific retention actions for a customer."""
        
        self.logger.info(f"Executing retention actions for customer: {customer_id}")
        
        execution_results = []
        
        for action_id in action_ids:
            result = await self._execute_individual_action(customer_id, action_id)
            execution_results.append(result)
        
        # Update customer profile with action results
        await self._update_customer_profile_with_actions(customer_id, execution_results)
        
        return {
            "customer_id": customer_id,
            "actions_executed": len(execution_results),
            "successful_actions": len([r for r in execution_results if r["status"] == "success"]),
            "failed_actions": len([r for r in execution_results if r["status"] == "failed"]),
            "execution_results": execution_results,
            "next_review_date": datetime.now() + timedelta(days=7)
        }
    
    async def get_expansion_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive expansion dashboard for a customer."""
        
        try:
            # Load expansion profile
            expansion_profile = await self._load_expansion_profile(customer_id)
            
            if not expansion_profile:
                return {"status": "error", "message": "Customer expansion profile not found"}
            
            return {
                "status": "success",
                "customer_id": customer_id,
                "customer_name": expansion_profile.customer_name,
                "health_summary": {
                    "overall_score": expansion_profile.health_score.overall_score,
                    "health_status": expansion_profile.health_score.health_status.value,
                    "trending_direction": expansion_profile.health_score.trending_direction,
                    "risk_factors": expansion_profile.health_score.risk_factors,
                    "positive_indicators": expansion_profile.health_score.positive_indicators
                },
                "expansion_readiness": expansion_profile.expansion_readiness.value,
                "expansion_opportunities": [
                    {
                        "opportunity_id": opp.opportunity_id,
                        "type": opp.opportunity_type.value,
                        "title": opp.title,
                        "estimated_value": float(opp.estimated_value),
                        "probability": opp.probability,
                        "timeline_months": opp.timeline_months,
                        "status": opp.status
                    }
                    for opp in expansion_profile.expansion_opportunities
                ],
                "retention_actions": [
                    {
                        "action_id": action.action_id,
                        "type": action.action_type,
                        "title": action.title,
                        "priority": action.priority,
                        "estimated_impact": action.estimated_impact,
                        "status": action.status,
                        "scheduled_date": action.scheduled_date.isoformat() if action.scheduled_date else None
                    }
                    for action in expansion_profile.retention_actions
                ],
                "key_metrics": {
                    "relationship_strength": expansion_profile.relationship_strength,
                    "churn_risk_probability": expansion_profile.churn_risk_probability,
                    "lifetime_value": float(expansion_profile.lifetime_value),
                    "expansion_potential_value": float(expansion_profile.expansion_potential_value)
                },
                "next_touchpoint": expansion_profile.next_touchpoint.isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get expansion dashboard: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _store_expansion_profile(self, profile: CustomerExpansionProfile):
        """Store expansion profile in Redis."""
        
        profile_data = {
            "customer_id": profile.customer_id,
            "customer_name": profile.customer_name,
            "current_services": profile.current_services,
            "health_score": {
                "overall_score": profile.health_score.overall_score,
                "health_status": profile.health_score.health_status.value,
                "trending_direction": profile.health_score.trending_direction,
                "component_scores": profile.health_score.component_scores,
                "risk_factors": profile.health_score.risk_factors,
                "positive_indicators": profile.health_score.positive_indicators
            },
            "expansion_readiness": profile.expansion_readiness.value,
            "expansion_opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "opportunity_type": opp.opportunity_type.value,
                    "title": opp.title,
                    "description": opp.description,
                    "estimated_value": str(opp.estimated_value),
                    "probability": opp.probability,
                    "timeline_months": opp.timeline_months,
                    "requirements": opp.requirements,
                    "next_actions": opp.next_actions,
                    "status": opp.status
                }
                for opp in profile.expansion_opportunities
            ],
            "retention_actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "title": action.title,
                    "priority": action.priority,
                    "estimated_impact": action.estimated_impact,
                    "cost": str(action.cost),
                    "timeline_days": action.timeline_days,
                    "status": action.status
                }
                for action in profile.retention_actions
            ],
            "relationship_strength": profile.relationship_strength,
            "churn_risk_probability": profile.churn_risk_probability,
            "lifetime_value": str(profile.lifetime_value),
            "expansion_potential_value": str(profile.expansion_potential_value),
            "last_interaction": profile.last_interaction.isoformat(),
            "next_touchpoint": profile.next_touchpoint.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        await self.redis_client.setex(
            f"expansion_profile:{profile.customer_id}",
            86400 * 90,  # 90 days TTL
            json.dumps(profile_data, default=str)
        )


# Global service instance
_expansion_engine: Optional[CustomerExpansionEngine] = None


async def get_expansion_engine() -> CustomerExpansionEngine:
    """Get the global customer expansion engine instance."""
    global _expansion_engine
    
    if _expansion_engine is None:
        _expansion_engine = CustomerExpansionEngine()
        await _expansion_engine.initialize()
    
    return _expansion_engine


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class CustomerExpansionEngineScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            async def test_expansion_engine():
            """Test the customer expansion engine."""

            engine = await get_expansion_engine()

            # Sample customer data
            customer_data = {
            "customer_name": "TechCorp Inc.",
            "current_services": ["mvp_development"],
            "satisfaction_metrics": {
            "overall_satisfaction": 8.7,
            "nps_score": 65,
            "support_satisfaction": 8.2,
            "stakeholder_sentiment": 85.0
            },
            "engagement_metrics": {
            "platform_usage_percentage": 78.0,
            "feature_adoption_percentage": 65.0,
            "communication_frequency_score": 82.0,
            "feedback_participation_percentage": 55.0
            },
            "project_success_metrics": {
            "delivery_success_rate": 95.0,
            "timeline_adherence": 88.0,
            "quality_scores": 92.0
            },
            "business_outcome_metrics": {
            "roi_achievement": 135.0,
            "business_impact": 88.0,
            "goal_attainment": 92.0
            },
            "relationship_metrics": {
            "stakeholder_relationships": 85.0,
            "champion_strength": 80.0,
            "escalation_frequency": 95.0
            },
            "profile": {
            "company_growth": 65,
            "hiring_challenges": True,
            "project_values": {"latest_project": 150000}
            },
            "project_history": [
            {
            "project_type": "mvp_development",
            "success_score": 92.0,
            "satisfaction_score": 8.7,
            "delivery_on_time": True
            }
            ],
            "lifetime_value": 150000,
            "last_interaction": datetime.now().isoformat()
            }

            # Create expansion profile
            expansion_profile = await engine.create_expansion_profile(
            "customer_techcorp", customer_data
            )

            self.logger.info("Customer Expansion Profile Created:")
            self.logger.info(f"Customer: {expansion_profile.customer_name}")
            self.logger.info(f"Health Score: {expansion_profile.health_score.overall_score:.1f}")
            self.logger.info(f"Health Status: {expansion_profile.health_score.health_status.value}")
            self.logger.info(f"Expansion Readiness: {expansion_profile.expansion_readiness.value}")
            self.logger.info(f"Expansion Opportunities: {len(expansion_profile.expansion_opportunities)}")
            self.logger.info(f"Retention Actions: {len(expansion_profile.retention_actions)}")
            self.logger.info(f"Expansion Potential Value: ${expansion_profile.expansion_potential_value:,.2f}")
            self.logger.info()

            # Get expansion dashboard
            dashboard = await engine.get_expansion_dashboard("customer_techcorp")
            self.logger.info("Expansion Dashboard:")
            self.logger.info(f"Overall Health: {dashboard['health_summary']['overall_score']:.1f}")
            self.logger.info(f"Risk Factors: {len(dashboard['health_summary']['risk_factors'])}")
            self.logger.info(f"Positive Indicators: {len(dashboard['health_summary']['positive_indicators'])}")
            self.logger.info(f"Top Opportunities: {len(dashboard['expansion_opportunities'])}")

            # Run test
            await test_expansion_engine()
            
            return {"status": "completed"}
    
    script_main(CustomerExpansionEngineScript)