"""
Enterprise Pilot Manager for LeanVibe Agent Hive 2.0

Manages Fortune 500 enterprise pilot programs with comprehensive ROI tracking,
success metrics, and conversion optimization. Designed for accelerated enterprise
market capture during competitive lead window.

Based on Gemini CLI strategic analysis for aggressive enterprise adoption.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from .database import get_session
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.agent import Agent

logger = structlog.get_logger()


class PilotStatus(Enum):
    """Enterprise pilot program status."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    CONVERTING = "converting"
    COMPLETED = "completed"
    EXTENDED = "extended"
    CANCELLED = "cancelled"


class PilotTier(Enum):
    """Enterprise pilot tier based on company size."""
    FORTUNE_50 = "fortune_50"
    FORTUNE_100 = "fortune_100"
    FORTUNE_500 = "fortune_500"
    ENTERPRISE = "enterprise"


@dataclass
class ROIMetrics:
    """ROI and success metrics for enterprise pilots."""
    baseline_velocity: float = 0.0  # Tasks per week before autonomous development
    autonomous_velocity: float = 0.0  # Tasks per week with autonomous development
    velocity_improvement: float = 0.0  # Multiplier improvement (target: 20x+)
    
    # Time savings metrics
    time_saved_hours: float = 0.0
    cost_savings_dollars: float = 0.0
    roi_percentage: float = 0.0
    
    # Quality metrics
    defect_reduction: float = 0.0
    test_coverage_improvement: float = 0.0
    documentation_completeness: float = 0.0
    
    # Business impact
    feature_delivery_acceleration: float = 0.0
    developer_satisfaction_score: float = 0.0
    time_to_market_improvement: float = 0.0
    
    def calculate_roi(self, pilot_cost: float, timeframe_weeks: int) -> None:
        """Calculate comprehensive ROI metrics."""
        if self.baseline_velocity > 0:
            self.velocity_improvement = self.autonomous_velocity / self.baseline_velocity
        
        # Conservative time savings calculation (assume $150/hour developer cost)
        weekly_time_saved = (self.autonomous_velocity - self.baseline_velocity) * 8  # 8 hours per task
        self.time_saved_hours = weekly_time_saved * timeframe_weeks
        self.cost_savings_dollars = self.time_saved_hours * 150
        
        if pilot_cost > 0:
            self.roi_percentage = ((self.cost_savings_dollars - pilot_cost) / pilot_cost) * 100


@dataclass
class EnterprisePilot:
    """Enterprise pilot program with comprehensive tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    company_name: str = ""
    company_tier: PilotTier = PilotTier.ENTERPRISE
    contact_name: str = ""
    contact_email: str = ""
    contact_title: str = ""
    
    # Pilot configuration
    status: PilotStatus = PilotStatus.PROPOSED
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_weeks: int = 4  # Standard 30-day pilot
    pilot_fee: float = 50000.0  # $50K pilot fee (credited toward license)
    
    # Success criteria
    target_velocity_improvement: float = 20.0  # Target 20x improvement
    success_threshold: float = 15.0  # Minimum 15x for pilot success
    guaranteed_roi: float = 1000.0  # Guaranteed 1000% ROI
    
    # Tracking
    roi_metrics: ROIMetrics = field(default_factory=ROIMetrics)
    tasks_completed: int = 0
    demos_conducted: int = 0
    stakeholders_engaged: int = 0
    
    # Enterprise specifics
    use_cases: List[str] = field(default_factory=list)
    technical_requirements: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)
    
    # Outcomes
    conversion_likelihood: float = 0.0  # 0-100 probability of conversion
    renewal_opportunity: float = 0.0  # Potential annual contract value
    reference_customer_potential: bool = False
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_successful(self) -> bool:
        """Determine if pilot meets success criteria."""
        return (
            self.roi_metrics.velocity_improvement >= self.success_threshold and
            self.roi_metrics.roi_percentage >= self.guaranteed_roi
        )
    
    def get_conversion_score(self) -> float:
        """Calculate conversion probability based on success metrics."""
        score = 0.0
        
        # Velocity improvement score (40% weight)
        if self.roi_metrics.velocity_improvement >= self.target_velocity_improvement:
            score += 40
        elif self.roi_metrics.velocity_improvement >= self.success_threshold:
            score += 30
        elif self.roi_metrics.velocity_improvement >= 10:
            score += 20
        
        # ROI score (30% weight)
        if self.roi_metrics.roi_percentage >= self.guaranteed_roi:
            score += 30
        elif self.roi_metrics.roi_percentage >= 500:
            score += 20
        elif self.roi_metrics.roi_percentage >= 200:
            score += 10
        
        # Engagement score (20% weight)
        if self.stakeholders_engaged >= 10:
            score += 20
        elif self.stakeholders_engaged >= 5:
            score += 15
        elif self.stakeholders_engaged >= 3:
            score += 10
        
        # Quality metrics (10% weight)
        if self.roi_metrics.developer_satisfaction_score >= 8.5:
            score += 10
        elif self.roi_metrics.developer_satisfaction_score >= 7.5:
            score += 7
        elif self.roi_metrics.developer_satisfaction_score >= 6.5:
            score += 5
        
        self.conversion_likelihood = min(score, 100)
        return self.conversion_likelihood


class EnterprisePilotManager:
    """
    Enterprise pilot program manager for Fortune 500 customer acquisition.
    
    Manages the complete pilot lifecycle from onboarding through conversion,
    with comprehensive ROI tracking and success optimization.
    """
    
    def __init__(self):
        self.active_pilots: Dict[str, EnterprisePilot] = {}
        self.success_threshold = 15.0  # Minimum 15x velocity improvement
        self.target_conversion_rate = 0.8  # 80% pilot-to-enterprise conversion
        
    async def create_pilot(
        self,
        company_name: str,
        company_tier: PilotTier,
        contact_info: Dict[str, str],
        use_cases: List[str],
        requirements: Dict[str, Any] = None
    ) -> EnterprisePilot:
        """Create new enterprise pilot program."""
        
        pilot = EnterprisePilot(
            company_name=company_name,
            company_tier=company_tier,
            contact_name=contact_info.get("name", ""),
            contact_email=contact_info.get("email", ""),
            contact_title=contact_info.get("title", ""),
            use_cases=use_cases,
            technical_requirements=requirements or {},
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(weeks=4)
        )
        
        # Set tier-specific parameters
        if company_tier == PilotTier.FORTUNE_50:
            pilot.pilot_fee = 100000.0  # $100K for Fortune 50
            pilot.target_velocity_improvement = 25.0  # Higher target
            pilot.renewal_opportunity = 3200000.0  # $3.2M annual potential
        elif company_tier == PilotTier.FORTUNE_100:
            pilot.pilot_fee = 75000.0  # $75K for Fortune 100
            pilot.target_velocity_improvement = 22.0
            pilot.renewal_opportunity = 1800000.0  # $1.8M annual potential
        else:
            pilot.pilot_fee = 50000.0  # $50K for Fortune 500
            pilot.target_velocity_improvement = 20.0
            pilot.renewal_opportunity = 900000.0  # $900K annual potential
        
        self.active_pilots[pilot.id] = pilot
        
        logger.info(
            "Enterprise pilot created",
            pilot_id=pilot.id,
            company=company_name,
            tier=company_tier.value,
            target_improvement=f"{pilot.target_velocity_improvement}x"
        )
        
        return pilot
    
    async def update_pilot_metrics(
        self,
        pilot_id: str,
        velocity_metrics: Dict[str, float],
        quality_metrics: Dict[str, float] = None
    ) -> None:
        """Update pilot ROI and success metrics."""
        
        if pilot_id not in self.active_pilots:
            raise ValueError(f"Pilot {pilot_id} not found")
        
        pilot = self.active_pilots[pilot_id]
        
        # Update velocity metrics
        pilot.roi_metrics.baseline_velocity = velocity_metrics.get("baseline_velocity", 0)
        pilot.roi_metrics.autonomous_velocity = velocity_metrics.get("autonomous_velocity", 0)
        
        # Update quality metrics if provided
        if quality_metrics:
            pilot.roi_metrics.defect_reduction = quality_metrics.get("defect_reduction", 0)
            pilot.roi_metrics.test_coverage_improvement = quality_metrics.get("test_coverage", 0)
            pilot.roi_metrics.developer_satisfaction_score = quality_metrics.get("satisfaction", 0)
        
        # Calculate ROI
        pilot.roi_metrics.calculate_roi(pilot.pilot_fee, pilot.duration_weeks)
        
        # Update conversion score
        pilot.get_conversion_score()
        
        pilot.updated_at = datetime.utcnow()
        
        logger.info(
            "Pilot metrics updated",
            pilot_id=pilot_id,
            velocity_improvement=f"{pilot.roi_metrics.velocity_improvement:.1f}x",
            roi_percentage=f"{pilot.roi_metrics.roi_percentage:.0f}%",
            conversion_likelihood=f"{pilot.conversion_likelihood:.0f}%"
        )
    
    async def track_pilot_activity(
        self,
        pilot_id: str,
        activity_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Track pilot program activities and engagement."""
        
        if pilot_id not in self.active_pilots:
            return
        
        pilot = self.active_pilots[pilot_id]
        
        if activity_type == "demo_conducted":
            pilot.demos_conducted += 1
        elif activity_type == "stakeholder_engagement":
            pilot.stakeholders_engaged += details.get("new_stakeholders", 1)
        elif activity_type == "task_completed":
            pilot.tasks_completed += details.get("task_count", 1)
        
        pilot.updated_at = datetime.utcnow()
        
        logger.info(
            "Pilot activity tracked",
            pilot_id=pilot_id,
            activity=activity_type,
            total_demos=pilot.demos_conducted,
            total_stakeholders=pilot.stakeholders_engaged
        )
    
    async def evaluate_pilot_success(self, pilot_id: str) -> Dict[str, Any]:
        """Comprehensive pilot success evaluation."""
        
        if pilot_id not in self.active_pilots:
            raise ValueError(f"Pilot {pilot_id} not found")
        
        pilot = self.active_pilots[pilot_id]
        
        # Update status based on success criteria
        if pilot.is_successful():
            pilot.status = PilotStatus.CONVERTING
            pilot.reference_customer_potential = True
        else:
            pilot.status = PilotStatus.EVALUATING
        
        conversion_score = pilot.get_conversion_score()
        
        evaluation = {
            "pilot_id": pilot_id,
            "company": pilot.company_name,
            "success_criteria_met": pilot.is_successful(),
            "velocity_improvement": pilot.roi_metrics.velocity_improvement,
            "roi_percentage": pilot.roi_metrics.roi_percentage,
            "conversion_likelihood": conversion_score,
            "renewal_opportunity": pilot.renewal_opportunity,
            "recommendation": self._get_conversion_recommendation(pilot),
            "next_steps": self._get_next_steps(pilot)
        }
        
        logger.info(
            "Pilot evaluation completed",
            pilot_id=pilot_id,
            success=pilot.is_successful(),
            conversion_score=f"{conversion_score:.0f}%",
            recommendation=evaluation["recommendation"]
        )
        
        return evaluation
    
    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for entire pilot portfolio."""
        
        total_pilots = len(self.active_pilots)
        successful_pilots = sum(1 for p in self.active_pilots.values() if p.is_successful())
        
        # Calculate average metrics
        if total_pilots > 0:
            avg_velocity_improvement = sum(
                p.roi_metrics.velocity_improvement for p in self.active_pilots.values()
            ) / total_pilots
            
            avg_roi = sum(
                p.roi_metrics.roi_percentage for p in self.active_pilots.values()
            ) / total_pilots
            
            avg_conversion_likelihood = sum(
                p.conversion_likelihood for p in self.active_pilots.values()
            ) / total_pilots
        else:
            avg_velocity_improvement = avg_roi = avg_conversion_likelihood = 0
        
        # Calculate pipeline value
        total_pipeline = sum(p.renewal_opportunity for p in self.active_pilots.values())
        qualified_pipeline = sum(
            p.renewal_opportunity for p in self.active_pilots.values()
            if p.conversion_likelihood >= 70
        )
        
        return {
            "total_pilots": total_pilots,
            "successful_pilots": successful_pilots,
            "success_rate": (successful_pilots / total_pilots * 100) if total_pilots > 0 else 0,
            "average_velocity_improvement": avg_velocity_improvement,
            "average_roi_percentage": avg_roi,
            "average_conversion_likelihood": avg_conversion_likelihood,
            "total_pipeline_value": total_pipeline,
            "qualified_pipeline_value": qualified_pipeline,
            "projected_conversions": total_pilots * (avg_conversion_likelihood / 100),
            "pilot_by_tier": self._get_tier_breakdown(),
            "status_breakdown": self._get_status_breakdown()
        }
    
    def _get_conversion_recommendation(self, pilot: EnterprisePilot) -> str:
        """Get conversion recommendation based on pilot performance."""
        
        if pilot.conversion_likelihood >= 80:
            return "STRONG_CONVERT"  # Immediate enterprise license conversion
        elif pilot.conversion_likelihood >= 60:
            return "LIKELY_CONVERT"  # High probability, continue engagement
        elif pilot.conversion_likelihood >= 40:
            return "EXTEND_PILOT"  # Extend pilot to improve metrics
        else:
            return "NURTURE_LONG_TERM"  # Long-term relationship building
    
    def _get_next_steps(self, pilot: EnterprisePilot) -> List[str]:
        """Get recommended next steps for pilot conversion."""
        
        steps = []
        
        if pilot.conversion_likelihood >= 70:
            steps.extend([
                "Schedule enterprise license presentation",
                "Prepare detailed ROI business case",
                "Engage procurement and legal teams",
                "Create implementation timeline"
            ])
        elif pilot.conversion_likelihood >= 50:
            steps.extend([
                "Address remaining stakeholder concerns",
                "Demonstrate additional use cases",
                "Provide competitive differentiation analysis",
                "Schedule executive sponsor meeting"
            ])
        else:
            steps.extend([
                "Identify blockers and concerns",
                "Provide additional training and support",
                "Consider pilot extension",
                "Re-evaluate success criteria"
            ])
        
        return steps
    
    def _get_tier_breakdown(self) -> Dict[str, int]:
        """Get breakdown of pilots by company tier."""
        breakdown = {tier.value: 0 for tier in PilotTier}
        
        for pilot in self.active_pilots.values():
            breakdown[pilot.company_tier.value] += 1
        
        return breakdown
    
    def _get_status_breakdown(self) -> Dict[str, int]:
        """Get breakdown of pilots by status."""
        breakdown = {status.value: 0 for status in PilotStatus}
        
        for pilot in self.active_pilots.values():
            breakdown[pilot.status.value] += 1
        
        return breakdown


# Global pilot manager instance
_pilot_manager: Optional[EnterprisePilotManager] = None


async def get_pilot_manager() -> EnterprisePilotManager:
    """Get or create enterprise pilot manager instance."""
    global _pilot_manager
    if _pilot_manager is None:
        _pilot_manager = EnterprisePilotManager()
    return _pilot_manager