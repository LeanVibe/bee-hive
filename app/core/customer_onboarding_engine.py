"""
Customer Onboarding Engine
Comprehensive customer acquisition, qualification, and onboarding automation.
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
from jinja2 import Template

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.services.customer_success_service import get_success_service, CustomerSuccessService


class LeadQualificationScore(Enum):
    """Lead qualification score levels."""
    HIGHLY_QUALIFIED = "highly_qualified"  # 80-100
    QUALIFIED = "qualified"  # 60-79
    MODERATELY_QUALIFIED = "moderately_qualified"  # 40-59
    UNQUALIFIED = "unqualified"  # 0-39


class ProjectFeasibility(Enum):
    """Project feasibility levels."""
    HIGHLY_FEASIBLE = "highly_feasible"  # 85-100
    FEASIBLE = "feasible"  # 70-84
    MARGINALLY_FEASIBLE = "marginally_feasible"  # 55-69
    INFEASIBLE = "infeasible"  # 0-54


@dataclass
class LeadProfile:
    """Lead profile data structure."""
    lead_id: str
    organization_name: str
    contact_information: Dict[str, str]
    project_requirements: Dict[str, Any]
    technical_readiness: Dict[str, Any]
    business_readiness: Dict[str, Any]
    financial_readiness: Dict[str, Any]
    organizational_readiness: Dict[str, Any]
    qualification_score: float = 0.0
    qualification_level: LeadQualificationScore = LeadQualificationScore.UNQUALIFIED
    disqualifiers: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class FeasibilityAssessment:
    """Project feasibility assessment result."""
    assessment_id: str
    lead_id: str
    technical_feasibility: float
    timeline_feasibility: float
    resource_feasibility: float
    overall_feasibility: float
    feasibility_level: ProjectFeasibility
    risk_factors: List[str] = field(default_factory=list)
    success_probability: float = 0.0
    recommended_approach: str = ""
    potential_blockers: List[str] = field(default_factory=list)


class LeadQualificationEngine:
    """Advanced lead qualification with AI-powered scoring."""
    
    QUALIFICATION_CRITERIA = {
        "technical_readiness": {
            "existing_codebase": 25,  # Points for having existing code
            "development_team": 20,   # Points for having technical team
            "project_documentation": 15,  # Points for clear requirements
            "technical_stack_clarity": 10,   # Points for tech stack decisions
            "infrastructure_readiness": 10  # Points for deployment infrastructure
        },
        "business_readiness": {
            "clear_requirements": 30,     # Most important factor
            "defined_timeline": 20,       # Project urgency
            "stakeholder_alignment": 15,  # Decision maker involvement
            "success_criteria": 10,       # Measurable goals
            "change_management": 5        # Ready for organizational change
        },
        "financial_readiness": {
            "budget_defined": 25,         # Budget allocated
            "decision_authority": 20,     # Can make financial decisions
            "contract_timeline": 10,      # Can move quickly
            "payment_terms": 5            # Acceptable payment terms
        },
        "organizational_readiness": {
            "change_management": 15,      # Ready for change
            "team_availability": 10,      # Team can participate
            "communication_structure": 10, # Clear communication lines
            "project_priority": 5         # Project has organizational priority
        }
    }
    
    SERVICE_FIT_MATRIX = {
        "mvp_development": {
            "ideal_score_range": (70, 100),
            "minimum_score": 60,
            "key_factors": ["clear_requirements", "defined_timeline", "budget_defined"],
            "disqualifiers": ["no_stakeholder_alignment", "unrealistic_timeline"]
        },
        "legacy_modernization": {
            "ideal_score_range": (75, 100),
            "minimum_score": 65,
            "key_factors": ["existing_codebase", "technical_readiness", "change_management"],
            "disqualifiers": ["no_technical_team", "business_critical_without_fallback"]
        },
        "team_augmentation": {
            "ideal_score_range": (65, 100),
            "minimum_score": 55,
            "key_factors": ["development_team", "clear_requirements", "communication_structure"],
            "disqualifiers": ["no_existing_team", "poor_communication_structure"]
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def qualify_lead(self, lead_data: dict) -> LeadProfile:
        """Qualify a lead and return detailed assessment."""
        
        lead_id = f"lead_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"Qualifying lead: {lead_id}")
        
        # Create lead profile
        lead_profile = LeadProfile(
            lead_id=lead_id,
            organization_name=lead_data.get("organization_name", "Unknown"),
            contact_information=lead_data.get("contact_information", {}),
            project_requirements=lead_data.get("project_requirements", {}),
            technical_readiness=lead_data.get("technical_readiness", {}),
            business_readiness=lead_data.get("business_readiness", {}),
            financial_readiness=lead_data.get("financial_readiness", {}),
            organizational_readiness=lead_data.get("organizational_readiness", {})
        )
        
        # Calculate qualification scores
        scores = {}
        
        # Technical readiness scoring
        scores["technical"] = await self._score_technical_readiness(
            lead_profile.technical_readiness
        )
        
        # Business readiness scoring
        scores["business"] = await self._score_business_readiness(
            lead_profile.business_readiness
        )
        
        # Financial readiness scoring
        scores["financial"] = await self._score_financial_readiness(
            lead_profile.financial_readiness
        )
        
        # Organizational readiness scoring
        scores["organizational"] = await self._score_organizational_readiness(
            lead_profile.organizational_readiness
        )
        
        # Calculate overall qualification score
        lead_profile.qualification_score = sum(scores.values()) / len(scores)
        
        # Determine qualification level
        if lead_profile.qualification_score >= 80:
            lead_profile.qualification_level = LeadQualificationScore.HIGHLY_QUALIFIED
        elif lead_profile.qualification_score >= 60:
            lead_profile.qualification_level = LeadQualificationScore.QUALIFIED
        elif lead_profile.qualification_score >= 40:
            lead_profile.qualification_level = LeadQualificationScore.MODERATELY_QUALIFIED
        else:
            lead_profile.qualification_level = LeadQualificationScore.UNQUALIFIED
        
        # Check for disqualifiers
        lead_profile.disqualifiers = await self._check_disqualifiers(lead_profile)
        
        # Generate next steps
        lead_profile.next_steps = await self._generate_next_steps(lead_profile)
        
        # Store lead profile
        await self._store_lead_profile(lead_profile)
        
        return lead_profile
    
    async def _score_technical_readiness(self, technical_data: dict) -> float:
        """Score technical readiness component."""
        
        score = 0.0
        criteria = self.QUALIFICATION_CRITERIA["technical_readiness"]
        
        # Existing codebase
        if technical_data.get("has_existing_codebase", False):
            score += criteria["existing_codebase"]
        
        # Development team
        team_size = technical_data.get("development_team_size", 0)
        if team_size >= 3:
            score += criteria["development_team"]
        elif team_size >= 1:
            score += criteria["development_team"] * 0.6
        
        # Project documentation
        doc_quality = technical_data.get("documentation_quality", "none")
        if doc_quality == "comprehensive":
            score += criteria["project_documentation"]
        elif doc_quality == "partial":
            score += criteria["project_documentation"] * 0.6
        
        # Technical stack clarity
        if technical_data.get("preferred_tech_stack"):
            score += criteria["technical_stack_clarity"]
        
        # Infrastructure readiness
        if technical_data.get("has_infrastructure_plan", False):
            score += criteria["infrastructure_readiness"]
        
        return min(score, 100.0)  # Cap at 100
    
    async def _score_business_readiness(self, business_data: dict) -> float:
        """Score business readiness component."""
        
        score = 0.0
        criteria = self.QUALIFICATION_CRITERIA["business_readiness"]
        
        # Clear requirements
        req_clarity = business_data.get("requirements_clarity", "vague")
        if req_clarity == "very_clear":
            score += criteria["clear_requirements"]
        elif req_clarity == "mostly_clear":
            score += criteria["clear_requirements"] * 0.8
        elif req_clarity == "somewhat_clear":
            score += criteria["clear_requirements"] * 0.5
        
        # Defined timeline
        timeline_urgency = business_data.get("timeline_urgency", "flexible")
        if timeline_urgency == "urgent":
            score += criteria["defined_timeline"]
        elif timeline_urgency == "moderate":
            score += criteria["defined_timeline"] * 0.7
        
        # Stakeholder alignment
        stakeholder_alignment = business_data.get("stakeholder_alignment", "poor")
        if stakeholder_alignment == "excellent":
            score += criteria["stakeholder_alignment"]
        elif stakeholder_alignment == "good":
            score += criteria["stakeholder_alignment"] * 0.8
        elif stakeholder_alignment == "fair":
            score += criteria["stakeholder_alignment"] * 0.5
        
        # Success criteria
        if business_data.get("has_success_criteria", False):
            score += criteria["success_criteria"]
        
        # Change management
        if business_data.get("change_management_experience", False):
            score += criteria["change_management"]
        
        return min(score, 100.0)
    
    async def _score_financial_readiness(self, financial_data: dict) -> float:
        """Score financial readiness component."""
        
        score = 0.0
        criteria = self.QUALIFICATION_CRITERIA["financial_readiness"]
        
        # Budget defined
        budget_status = financial_data.get("budget_status", "undefined")
        if budget_status == "approved":
            score += criteria["budget_defined"]
        elif budget_status == "preliminary":
            score += criteria["budget_defined"] * 0.6
        
        # Decision authority
        decision_authority = financial_data.get("decision_authority", "none")
        if decision_authority == "full":
            score += criteria["decision_authority"]
        elif decision_authority == "partial":
            score += criteria["decision_authority"] * 0.6
        
        # Contract timeline
        contract_timeline = financial_data.get("contract_timeline_weeks", 12)
        if contract_timeline <= 2:
            score += criteria["contract_timeline"]
        elif contract_timeline <= 4:
            score += criteria["contract_timeline"] * 0.7
        elif contract_timeline <= 8:
            score += criteria["contract_timeline"] * 0.4
        
        # Payment terms
        if financial_data.get("payment_terms_acceptable", True):
            score += criteria["payment_terms"]
        
        return min(score, 100.0)
    
    async def _score_organizational_readiness(self, org_data: dict) -> float:
        """Score organizational readiness component."""
        
        score = 0.0
        criteria = self.QUALIFICATION_CRITERIA["organizational_readiness"]
        
        # Change management
        change_readiness = org_data.get("change_readiness", "resistant")
        if change_readiness == "enthusiastic":
            score += criteria["change_management"]
        elif change_readiness == "open":
            score += criteria["change_management"] * 0.8
        elif change_readiness == "cautious":
            score += criteria["change_management"] * 0.5
        
        # Team availability
        team_availability = org_data.get("team_availability", "limited")
        if team_availability == "full":
            score += criteria["team_availability"]
        elif team_availability == "partial":
            score += criteria["team_availability"] * 0.6
        
        # Communication structure
        comm_structure = org_data.get("communication_structure", "poor")
        if comm_structure == "excellent":
            score += criteria["communication_structure"]
        elif comm_structure == "good":
            score += criteria["communication_structure"] * 0.8
        elif comm_structure == "fair":
            score += criteria["communication_structure"] * 0.5
        
        # Project priority
        if org_data.get("is_high_priority", False):
            score += criteria["project_priority"]
        
        return min(score, 100.0)
    
    async def _check_disqualifiers(self, lead_profile: LeadProfile) -> List[str]:
        """Check for lead disqualifiers."""
        
        disqualifiers = []
        
        # No decision maker involvement
        if lead_profile.business_readiness.get("stakeholder_alignment") == "poor":
            disqualifiers.append("no_stakeholder_alignment")
        
        # Unrealistic timeline
        timeline_weeks = lead_profile.project_requirements.get("timeline_weeks", 0)
        complexity = lead_profile.project_requirements.get("complexity", "medium")
        
        if complexity == "high" and timeline_weeks < 8:
            disqualifiers.append("unrealistic_timeline")
        elif complexity == "medium" and timeline_weeks < 4:
            disqualifiers.append("unrealistic_timeline")
        
        # No technical team for team augmentation
        service_type = lead_profile.project_requirements.get("service_type")
        if service_type == "team_augmentation":
            if lead_profile.technical_readiness.get("development_team_size", 0) == 0:
                disqualifiers.append("no_existing_team")
        
        # Business critical without fallback for legacy modernization
        if service_type == "legacy_modernization":
            is_critical = lead_profile.project_requirements.get("business_criticality") == "critical"
            has_fallback = lead_profile.technical_readiness.get("has_fallback_plan", False)
            
            if is_critical and not has_fallback:
                disqualifiers.append("business_critical_without_fallback")
        
        return disqualifiers
    
    async def _generate_next_steps(self, lead_profile: LeadProfile) -> List[str]:
        """Generate recommended next steps based on qualification."""
        
        next_steps = []
        
        if lead_profile.qualification_level == LeadQualificationScore.HIGHLY_QUALIFIED:
            next_steps = [
                "Schedule technical architecture review",
                "Prepare service tier recommendation",
                "Initiate customer education sequence",
                "Create success guarantee proposal"
            ]
        elif lead_profile.qualification_level == LeadQualificationScore.QUALIFIED:
            next_steps = [
                "Conduct detailed requirements clarification",
                "Assess project feasibility",
                "Schedule stakeholder alignment session",
                "Prepare preliminary proposal"
            ]
        elif lead_profile.qualification_level == LeadQualificationScore.MODERATELY_QUALIFIED:
            next_steps = [
                "Address key qualification gaps",
                "Provide educational content about autonomous development",
                "Schedule discovery workshop",
                "Re-evaluate after improvements"
            ]
        else:  # UNQUALIFIED
            next_steps = [
                "Provide market education resources",
                "Suggest preparatory steps",
                "Schedule follow-up in 3-6 months",
                "Offer consultation services"
            ]
        
        return next_steps
    
    async def _store_lead_profile(self, lead_profile: LeadProfile):
        """Store lead profile in Redis for tracking."""
        
        profile_data = {
            "lead_id": lead_profile.lead_id,
            "organization_name": lead_profile.organization_name,
            "qualification_score": lead_profile.qualification_score,
            "qualification_level": lead_profile.qualification_level.value,
            "disqualifiers": lead_profile.disqualifiers,
            "next_steps": lead_profile.next_steps,
            "created_at": datetime.now().isoformat(),
            "contact_information": lead_profile.contact_information
        }
        
        await self.redis.setex(
            f"lead_profile:{lead_profile.lead_id}",
            86400 * 30,  # 30 days TTL
            json.dumps(profile_data, default=str)
        )


class ProjectFeasibilityAnalyzer:
    """Comprehensive project feasibility assessment."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def analyze_project_feasibility(
        self, 
        project_request: dict,
        lead_profile: LeadProfile
    ) -> FeasibilityAssessment:
        """Analyze project feasibility with risk assessment."""
        
        assessment_id = f"feasibility_{lead_profile.lead_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Analyzing project feasibility: {assessment_id}")
        
        # Technical feasibility analysis
        technical_score = await self._assess_technical_feasibility(project_request)
        
        # Timeline feasibility
        timeline_score = await self._assess_timeline_feasibility(
            project_request.get("timeline_weeks", 8),
            project_request.get("requirements", [])
        )
        
        # Resource feasibility
        resource_score = await self._assess_resource_feasibility(
            project_request.get("budget_usd", 0),
            project_request.get("team_size", 1)
        )
        
        # Calculate overall feasibility
        overall_feasibility = (
            technical_score * 0.4 + 
            timeline_score * 0.3 + 
            resource_score * 0.3
        )
        
        # Determine feasibility level
        if overall_feasibility >= 85:
            feasibility_level = ProjectFeasibility.HIGHLY_FEASIBLE
        elif overall_feasibility >= 70:
            feasibility_level = ProjectFeasibility.FEASIBLE
        elif overall_feasibility >= 55:
            feasibility_level = ProjectFeasibility.MARGINALLY_FEASIBLE
        else:
            feasibility_level = ProjectFeasibility.INFEASIBLE
        
        # Create assessment
        assessment = FeasibilityAssessment(
            assessment_id=assessment_id,
            lead_id=lead_profile.lead_id,
            technical_feasibility=technical_score,
            timeline_feasibility=timeline_score,
            resource_feasibility=resource_score,
            overall_feasibility=overall_feasibility,
            feasibility_level=feasibility_level
        )
        
        # Success probability based on historical data
        assessment.success_probability = await self._calculate_success_probability(
            overall_feasibility,
            lead_profile.qualification_score
        )
        
        # Risk factors identification
        assessment.risk_factors = await self._identify_risk_factors(
            project_request, lead_profile, assessment
        )
        
        # Recommended approach
        assessment.recommended_approach = await self._recommend_approach(
            project_request, assessment
        )
        
        # Potential blockers
        assessment.potential_blockers = await self._identify_potential_blockers(
            project_request, lead_profile
        )
        
        # Store assessment
        await self._store_feasibility_assessment(assessment)
        
        return assessment
    
    async def _assess_technical_feasibility(self, project_request: dict) -> float:
        """Assess technical feasibility score."""
        
        score = 100.0  # Start with full score and deduct
        
        # Complexity assessment
        complexity = project_request.get("complexity", "medium")
        if complexity == "very_high":
            score -= 30
        elif complexity == "high":
            score -= 20
        elif complexity == "medium":
            score -= 10
        
        # Technology stack maturity
        tech_stack = project_request.get("technology_preferences", [])
        immature_techs = ["experimental", "alpha", "bleeding_edge"]
        
        for tech in tech_stack:
            if any(keyword in tech.lower() for keyword in immature_techs):
                score -= 15
        
        # Integration complexity
        integrations = project_request.get("required_integrations", [])
        score -= min(len(integrations) * 5, 25)  # Max 25 point deduction
        
        # Compliance requirements
        compliance_reqs = project_request.get("compliance_requirements", [])
        high_compliance = ["hipaa", "sox", "pci", "gdpr"]
        
        for req in compliance_reqs:
            if req.lower() in high_compliance:
                score -= 10
        
        return max(score, 0.0)
    
    async def _assess_timeline_feasibility(
        self, 
        timeline_weeks: int, 
        requirements: List[str]
    ) -> float:
        """Assess timeline feasibility."""
        
        # Estimate minimum time based on requirements
        base_weeks = 4  # Minimum for any project
        
        # Add time for each requirement (simplified)
        estimated_weeks = base_weeks + len(requirements) * 0.5
        
        # Add complexity multiplier
        if len(requirements) > 20:
            estimated_weeks *= 1.3
        elif len(requirements) > 10:
            estimated_weeks *= 1.2
        
        # Calculate feasibility score
        if timeline_weeks >= estimated_weeks:
            score = 100.0
        else:
            # Penalize based on shortage
            shortage_ratio = timeline_weeks / estimated_weeks
            score = shortage_ratio * 100
        
        return max(score, 0.0)
    
    async def _assess_resource_feasibility(
        self, 
        budget_usd: int, 
        team_size: int
    ) -> float:
        """Assess resource feasibility."""
        
        # Rough budget estimation (simplified)
        min_budget = 50000  # Minimum viable budget
        optimal_budget = 150000  # Optimal budget for most projects
        
        budget_score = 100.0
        if budget_usd < min_budget:
            budget_score = (budget_usd / min_budget) * 100
        elif budget_usd < optimal_budget:
            budget_score = 80 + ((budget_usd - min_budget) / (optimal_budget - min_budget)) * 20
        
        # Team size assessment
        team_score = 100.0
        if team_size < 1:
            team_score = 0
        elif team_size < 3:
            team_score = 70
        
        return (budget_score + team_score) / 2
    
    async def _calculate_success_probability(
        self, 
        feasibility_score: float, 
        qualification_score: float
    ) -> float:
        """Calculate success probability based on scores."""
        
        # Weighted combination
        combined_score = (feasibility_score * 0.6 + qualification_score * 0.4)
        
        # Apply historical success rate adjustments
        if combined_score >= 85:
            return 95.0
        elif combined_score >= 75:
            return 85.0
        elif combined_score >= 65:
            return 75.0
        elif combined_score >= 55:
            return 60.0
        else:
            return 40.0
    
    async def _store_feasibility_assessment(self, assessment: FeasibilityAssessment):
        """Store feasibility assessment in Redis."""
        
        assessment_data = {
            "assessment_id": assessment.assessment_id,
            "lead_id": assessment.lead_id,
            "technical_feasibility": assessment.technical_feasibility,
            "timeline_feasibility": assessment.timeline_feasibility,
            "resource_feasibility": assessment.resource_feasibility,
            "overall_feasibility": assessment.overall_feasibility,
            "feasibility_level": assessment.feasibility_level.value,
            "success_probability": assessment.success_probability,
            "risk_factors": assessment.risk_factors,
            "recommended_approach": assessment.recommended_approach,
            "potential_blockers": assessment.potential_blockers,
            "created_at": datetime.now().isoformat()
        }
        
        await self.redis.setex(
            f"feasibility_assessment:{assessment.assessment_id}",
            86400 * 30,  # 30 days TTL
            json.dumps(assessment_data, default=str)
        )


# Global service instances
_qualification_engine: Optional[LeadQualificationEngine] = None
_feasibility_analyzer: Optional[ProjectFeasibilityAnalyzer] = None


async def get_qualification_engine() -> LeadQualificationEngine:
    """Get the global lead qualification engine instance."""
    global _qualification_engine
    
    if _qualification_engine is None:
        redis_client = await get_redis_client()
        logger = logging.getLogger(__name__)
        _qualification_engine = LeadQualificationEngine(redis_client, logger)
    
    return _qualification_engine


async def get_feasibility_analyzer() -> ProjectFeasibilityAnalyzer:
    """Get the global project feasibility analyzer instance."""
    global _feasibility_analyzer
    
    if _feasibility_analyzer is None:
        redis_client = await get_redis_client()
        logger = logging.getLogger(__name__)
        _feasibility_analyzer = ProjectFeasibilityAnalyzer(redis_client, logger)
    
    return _feasibility_analyzer


# Usage example and testing
if __name__ == "__main__":
    async def test_customer_onboarding():
        """Test the customer onboarding engine."""
        
        # Sample lead data
        lead_data = {
            "organization_name": "TechCorp Inc.",
            "contact_information": {
                "primary_contact": "John Smith",
                "email": "john.smith@techcorp.com",
                "phone": "+1-555-0123"
            },
            "project_requirements": {
                "service_type": "mvp_development",
                "timeline_weeks": 8,
                "complexity": "medium",
                "requirements": [
                    "User authentication system",
                    "Product catalog with search",
                    "Shopping cart functionality",
                    "Payment processing integration",
                    "Admin dashboard"
                ],
                "technology_preferences": ["React", "Node.js", "PostgreSQL"],
                "budget_usd": 150000,
                "compliance_requirements": ["PCI DSS"]
            },
            "technical_readiness": {
                "has_existing_codebase": False,
                "development_team_size": 2,
                "documentation_quality": "partial",
                "preferred_tech_stack": ["React", "Node.js"],
                "has_infrastructure_plan": True
            },
            "business_readiness": {
                "requirements_clarity": "mostly_clear",
                "timeline_urgency": "moderate",
                "stakeholder_alignment": "good",
                "has_success_criteria": True,
                "change_management_experience": True
            },
            "financial_readiness": {
                "budget_status": "approved",
                "decision_authority": "full",
                "contract_timeline_weeks": 3,
                "payment_terms_acceptable": True
            },
            "organizational_readiness": {
                "change_readiness": "open",
                "team_availability": "partial",
                "communication_structure": "good",
                "is_high_priority": True
            }
        }
        
        # Test lead qualification
        qualification_engine = await get_qualification_engine()
        lead_profile = await qualification_engine.qualify_lead(lead_data)
        
        print("Lead Qualification Result:")
        print(f"Organization: {lead_profile.organization_name}")
        print(f"Qualification Score: {lead_profile.qualification_score:.1f}")
        print(f"Qualification Level: {lead_profile.qualification_level.value}")
        print(f"Disqualifiers: {lead_profile.disqualifiers}")
        print(f"Next Steps: {lead_profile.next_steps}")
        print()
        
        # Test feasibility analysis
        feasibility_analyzer = await get_feasibility_analyzer()
        feasibility_assessment = await feasibility_analyzer.analyze_project_feasibility(
            lead_data["project_requirements"],
            lead_profile
        )
        
        print("Feasibility Assessment Result:")
        print(f"Overall Feasibility: {feasibility_assessment.overall_feasibility:.1f}")
        print(f"Feasibility Level: {feasibility_assessment.feasibility_level.value}")
        print(f"Success Probability: {feasibility_assessment.success_probability:.1f}%")
        print(f"Technical Feasibility: {feasibility_assessment.technical_feasibility:.1f}")
        print(f"Timeline Feasibility: {feasibility_assessment.timeline_feasibility:.1f}")
        print(f"Resource Feasibility: {feasibility_assessment.resource_feasibility:.1f}")
        print(f"Risk Factors: {feasibility_assessment.risk_factors}")
        print(f"Recommended Approach: {feasibility_assessment.recommended_approach}")
    
    # Run test
    asyncio.run(test_customer_onboarding())