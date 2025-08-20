"""
Weekly Milestone Framework
Comprehensive weekly milestone tracking and validation for 30-day success guarantee.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
import redis.asyncio as redis
import aiohttp

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.services.customer_success_service import get_success_service


class MilestoneStatus(Enum):
    """Milestone status levels."""
    SUCCESS = "success"  # >= minimum_success_threshold
    AT_RISK = "at_risk"  # >= escalation_threshold but < minimum_success_threshold
    FAILED = "failed"  # < escalation_threshold
    PENDING = "pending"  # Not yet validated


class ValidationMethod(Enum):
    """Validation methods for milestone criteria."""
    STAKEHOLDER_SIGN_OFF = "stakeholder_sign_off"
    TECHNICAL_REVIEW = "technical_review"
    AUTOMATED_VERIFICATION = "automated_verification"
    STAKEHOLDER_FEEDBACK = "stakeholder_feedback"
    AUTOMATED_TESTING = "automated_testing"
    INTEGRATION_TESTING = "integration_testing"
    SECURITY_AUDIT = "security_audit"
    STAKEHOLDER_DEMO = "stakeholder_demo"
    AUTOMATED_QUALITY_CHECKS = "automated_quality_checks"
    PERFORMANCE_TESTING = "performance_testing"
    USER_TESTING = "user_testing"
    DEPLOYMENT_VERIFICATION = "deployment_verification"
    DOCUMENTATION_REVIEW = "documentation_review"
    TRAINING_COMPLETION = "training_completion"
    SATISFACTION_SURVEY = "satisfaction_survey"


@dataclass
class SuccessCriterion:
    """Individual success criterion for a milestone."""
    criterion_id: str
    criterion_name: str
    description: str
    weight: float  # 0.0 to 1.0
    validation_method: ValidationMethod
    deliverable: str
    target_value: Optional[Union[str, float, int]] = None
    current_value: Optional[Union[str, float, int]] = None
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    stakeholder_confirmation: Dict[str, Any] = field(default_factory=dict)
    completion_status: bool = False


@dataclass
class WeeklyMilestone:
    """Weekly milestone definition and tracking."""
    milestone_id: str
    guarantee_id: str
    week_number: int
    title: str
    description: str
    success_criteria: List[SuccessCriterion]
    minimum_success_threshold: float  # Percentage required to pass
    escalation_threshold: float  # Percentage that triggers escalation
    overall_score: float = 0.0
    status: MilestoneStatus = MilestoneStatus.PENDING
    validation_timestamp: Optional[datetime] = None
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    next_actions: List[str] = field(default_factory=list)
    escalation_actions: List[str] = field(default_factory=list)


@dataclass
class MilestoneValidationResult:
    """Result of milestone validation."""
    validation_id: str
    milestone_id: str
    overall_score: float
    status: MilestoneStatus
    criteria_results: List[Dict[str, Any]]
    validation_timestamp: datetime
    evidence_summary: Dict[str, Any]
    recommendations: List[str]
    next_actions: List[str]
    escalation_required: bool


class WeeklyMilestoneFramework:
    """Comprehensive weekly milestone tracking and validation."""
    
    MILESTONE_TEMPLATES = {
        "mvp_development": {
            "week_1": {
                "title": "Foundation & Architecture",
                "description": "Establish project foundation, complete requirements analysis, and design system architecture.",
                "success_criteria": [
                    {
                        "criterion_name": "Requirements Analysis Complete",
                        "description": "All functional and non-functional requirements documented and approved by stakeholders.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.STAKEHOLDER_SIGN_OFF,
                        "deliverable": "Requirements specification document with stakeholder sign-off",
                        "target_value": "100% requirements documented and approved"
                    },
                    {
                        "criterion_name": "System Architecture Designed",
                        "description": "Complete system architecture with technology stack, data flow, and component design.",
                        "weight": 0.4,
                        "validation_method": ValidationMethod.TECHNICAL_REVIEW,
                        "deliverable": "Architecture diagrams, technical specification, and technology decisions document",
                        "target_value": "Architecture review passed with >= 8.5/10 score"
                    },
                    {
                        "criterion_name": "Development Environment Setup",
                        "description": "Complete development, testing, and CI/CD environments configured and operational.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.AUTOMATED_VERIFICATION,
                        "deliverable": "Working development environment with CI/CD pipeline and automated testing setup",
                        "target_value": "All environments operational with green build status"
                    },
                    {
                        "criterion_name": "Team Communication Established",
                        "description": "Communication channels, daily standups, and reporting processes established.",
                        "weight": 0.1,
                        "validation_method": ValidationMethod.STAKEHOLDER_FEEDBACK,
                        "deliverable": "Communication channels setup, daily standup process, and stakeholder reporting schedule",
                        "target_value": "Communication effectiveness score >= 8.0/10"
                    }
                ],
                "minimum_success_threshold": 85.0,
                "escalation_threshold": 70.0
            },
            "week_2": {
                "title": "Core Development Sprint",
                "description": "Implement core features, APIs, and security measures with comprehensive testing.",
                "success_criteria": [
                    {
                        "criterion_name": "Core Features Implementation",
                        "description": "Primary business logic and core features implemented with comprehensive test coverage.",
                        "weight": 0.5,
                        "validation_method": ValidationMethod.AUTOMATED_TESTING,
                        "deliverable": "Working core features with >= 90% test coverage and passing all unit tests",
                        "target_value": "90% test coverage with all critical features functional"
                    },
                    {
                        "criterion_name": "API Development & Integration",
                        "description": "RESTful APIs implemented with proper documentation and integration tests.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.INTEGRATION_TESTING,
                        "deliverable": "Complete API endpoints with OpenAPI documentation and integration test suite",
                        "target_value": "All API endpoints functional with comprehensive documentation"
                    },
                    {
                        "criterion_name": "Security Implementation",
                        "description": "Security measures implemented including authentication, authorization, and data protection.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.SECURITY_AUDIT,
                        "deliverable": "Security implementation with vulnerability scan results showing zero high-risk issues",
                        "target_value": "Security audit passed with zero high/critical vulnerabilities"
                    }
                ],
                "minimum_success_threshold": 80.0,
                "escalation_threshold": 65.0
            },
            "week_3": {
                "title": "Feature Completion & Quality Assurance",
                "description": "Complete all features, pass quality gates, and conduct user acceptance testing.",
                "success_criteria": [
                    {
                        "criterion_name": "All Features Implemented",
                        "description": "All required features implemented and functional according to specifications.",
                        "weight": 0.4,
                        "validation_method": ValidationMethod.STAKEHOLDER_DEMO,
                        "deliverable": "Complete feature set demonstration matching all requirements",
                        "target_value": "100% of required features implemented and demonstrated"
                    },
                    {
                        "criterion_name": "Quality Gates Passed",
                        "description": "All quality gates passed including test coverage, code quality, and performance benchmarks.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.AUTOMATED_QUALITY_CHECKS,
                        "deliverable": "Quality reports showing >= 95% test coverage and code quality score > 8.5",
                        "target_value": "95% test coverage, 8.5+ code quality score"
                    },
                    {
                        "criterion_name": "Performance Benchmarks Met",
                        "description": "System performance meets or exceeds specified benchmarks under expected load.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.PERFORMANCE_TESTING,
                        "deliverable": "Performance test results meeting all SLA requirements",
                        "target_value": "All performance SLAs met or exceeded"
                    },
                    {
                        "criterion_name": "User Acceptance Testing",
                        "description": "Stakeholder user acceptance testing completed with approval.",
                        "weight": 0.1,
                        "validation_method": ValidationMethod.USER_TESTING,
                        "deliverable": "UAT results with stakeholder approval and feedback incorporation",
                        "target_value": "UAT passed with >= 95% stakeholder approval"
                    }
                ],
                "minimum_success_threshold": 85.0,
                "escalation_threshold": 70.0
            },
            "week_4": {
                "title": "Delivery & Knowledge Transfer",
                "description": "Deploy to production, complete documentation, and transfer knowledge to customer team.",
                "success_criteria": [
                    {
                        "criterion_name": "Production Deployment",
                        "description": "Successful deployment to production environment with monitoring and alerting.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.DEPLOYMENT_VERIFICATION,
                        "deliverable": "Live production system with monitoring, logging, and alerting configured",
                        "target_value": "Production deployment successful with 99.9% uptime in first 48 hours"
                    },
                    {
                        "criterion_name": "Documentation Complete",
                        "description": "Complete technical and user documentation including deployment and maintenance guides.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.DOCUMENTATION_REVIEW,
                        "deliverable": "Complete technical documentation, user guides, and operational runbooks",
                        "target_value": "Documentation completeness score >= 90%"
                    },
                    {
                        "criterion_name": "Knowledge Transfer Sessions",
                        "description": "Comprehensive knowledge transfer sessions for customer team with recorded materials.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.TRAINING_COMPLETION,
                        "deliverable": "Knowledge transfer sessions completed with recorded materials and Q&A",
                        "target_value": "All planned training sessions completed with >= 95% attendance"
                    },
                    {
                        "criterion_name": "Customer Satisfaction Survey",
                        "description": "Customer satisfaction survey completed with high satisfaction scores.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.SATISFACTION_SURVEY,
                        "deliverable": "Customer satisfaction survey results with detailed feedback",
                        "target_value": "Customer satisfaction score >= 8.5/10"
                    }
                ],
                "minimum_success_threshold": 90.0,
                "escalation_threshold": 75.0
            }
        },
        "legacy_modernization": {
            "week_1": {
                "title": "Legacy Analysis & Modernization Plan",
                "description": "Complete legacy system analysis and create comprehensive modernization roadmap.",
                "success_criteria": [
                    {
                        "criterion_name": "Legacy System Analysis",
                        "description": "Comprehensive analysis of existing legacy system including architecture, dependencies, and risks.",
                        "weight": 0.4,
                        "validation_method": ValidationMethod.TECHNICAL_REVIEW,
                        "deliverable": "Legacy system analysis report with architecture assessment and dependency mapping",
                        "target_value": "Analysis completeness score >= 95% with all critical components documented"
                    },
                    {
                        "criterion_name": "Modernization Strategy Defined",
                        "description": "Clear modernization strategy with approach, timeline, and risk mitigation plans.",
                        "weight": 0.3,
                        "validation_method": ValidationMethod.STAKEHOLDER_SIGN_OFF,
                        "deliverable": "Modernization strategy document with stakeholder approval",
                        "target_value": "Strategy approved by all key stakeholders"
                    },
                    {
                        "criterion_name": "Risk Assessment Complete",
                        "description": "Comprehensive risk assessment with mitigation strategies for business continuity.",
                        "weight": 0.2,
                        "validation_method": ValidationMethod.TECHNICAL_REVIEW,
                        "deliverable": "Risk assessment document with mitigation plans for all identified risks",
                        "target_value": "All high and critical risks have documented mitigation plans"
                    },
                    {
                        "criterion_name": "Modernization Environment Setup",
                        "description": "Development and testing environments configured for modernization work.",
                        "weight": 0.1,
                        "validation_method": ValidationMethod.AUTOMATED_VERIFICATION,
                        "deliverable": "Working modernization environments with CI/CD pipeline",
                        "target_value": "All environments operational with green build status"
                    }
                ],
                "minimum_success_threshold": 85.0,
                "escalation_threshold": 70.0
            }
            # Additional weeks would be defined similarly...
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def create_milestone_plan(
        self,
        guarantee_id: str,
        service_type: str,
        custom_criteria: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, WeeklyMilestone]:
        """Create complete 4-week milestone plan for a service type."""
        
        self.logger.info(f"Creating milestone plan for guarantee: {guarantee_id}, service type: {service_type}")
        
        if service_type not in self.MILESTONE_TEMPLATES:
            raise ValueError(f"Unsupported service type: {service_type}")
        
        template = self.MILESTONE_TEMPLATES[service_type]
        milestone_plan = {}
        
        for week_key, week_config in template.items():
            week_number = int(week_key.split('_')[1])
            
            # Create success criteria
            success_criteria = []
            for i, criterion_config in enumerate(week_config["success_criteria"]):
                criterion_id = f"criterion_{guarantee_id}_w{week_number}_{i+1}"
                
                criterion = SuccessCriterion(
                    criterion_id=criterion_id,
                    criterion_name=criterion_config["criterion_name"],
                    description=criterion_config["description"],
                    weight=criterion_config["weight"],
                    validation_method=ValidationMethod(criterion_config["validation_method"]),
                    deliverable=criterion_config["deliverable"],
                    target_value=criterion_config["target_value"]
                )
                
                success_criteria.append(criterion)
            
            # Create milestone
            milestone_id = f"milestone_{guarantee_id}_week_{week_number}"
            
            milestone = WeeklyMilestone(
                milestone_id=milestone_id,
                guarantee_id=guarantee_id,
                week_number=week_number,
                title=week_config["title"],
                description=week_config["description"],
                success_criteria=success_criteria,
                minimum_success_threshold=week_config["minimum_success_threshold"],
                escalation_threshold=week_config["escalation_threshold"]
            )
            
            milestone_plan[f"week_{week_number}"] = milestone
        
        # Apply custom criteria if provided
        if custom_criteria:
            milestone_plan = await self._apply_custom_criteria(milestone_plan, custom_criteria)
        
        # Store milestone plan
        await self._store_milestone_plan(guarantee_id, milestone_plan)
        
        return milestone_plan
    
    async def validate_weekly_milestone(
        self,
        milestone_id: str,
        validation_data: Dict[str, Any]
    ) -> MilestoneValidationResult:
        """Validate weekly milestone completion with comprehensive evidence."""
        
        validation_id = f"validation_{milestone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Validating milestone: {milestone_id}")
        
        # Retrieve milestone details
        milestone = await self._get_milestone_details(milestone_id)
        if not milestone:
            raise ValueError(f"Milestone not found: {milestone_id}")
        
        # Initialize validation result
        validation_result = MilestoneValidationResult(
            validation_id=validation_id,
            milestone_id=milestone_id,
            overall_score=0.0,
            status=MilestoneStatus.PENDING,
            criteria_results=[],
            validation_timestamp=datetime.now(),
            evidence_summary={},
            recommendations=[],
            next_actions=[],
            escalation_required=False
        )
        
        total_weighted_score = 0.0
        
        # Validate each success criterion
        for criterion in milestone.success_criteria:
            criterion_result = await self._validate_criterion(
                criterion,
                validation_data.get(criterion.criterion_id, {})
            )
            
            weighted_score = criterion_result["score"] * criterion.weight
            total_weighted_score += weighted_score
            
            validation_result.criteria_results.append({
                "criterion_id": criterion.criterion_id,
                "criterion_name": criterion.criterion_name,
                "score": criterion_result["score"],
                "weight": criterion.weight,
                "weighted_score": weighted_score,
                "validation_evidence": criterion_result["evidence"],
                "deliverable_status": criterion_result["deliverable_status"],
                "completion_status": criterion_result["score"] >= 80.0,  # 80% threshold for individual criteria
                "stakeholder_feedback": criterion_result.get("stakeholder_feedback", {})
            })
        
        validation_result.overall_score = total_weighted_score
        
        # Determine milestone status
        if validation_result.overall_score >= milestone.minimum_success_threshold:
            validation_result.status = MilestoneStatus.SUCCESS
        elif validation_result.overall_score >= milestone.escalation_threshold:
            validation_result.status = MilestoneStatus.AT_RISK
            validation_result.escalation_required = True
        else:
            validation_result.status = MilestoneStatus.FAILED
            validation_result.escalation_required = True
        
        # Generate recommendations and next actions
        validation_result.recommendations = await self._generate_recommendations(
            milestone, validation_result
        )
        
        validation_result.next_actions = await self._generate_next_actions(
            milestone, validation_result
        )
        
        # Update milestone status
        milestone.overall_score = validation_result.overall_score
        milestone.status = validation_result.status
        milestone.validation_timestamp = validation_result.validation_timestamp
        milestone.next_actions = validation_result.next_actions
        
        # Store validation results
        await self._store_validation_result(validation_result)
        await self._update_milestone_status(milestone)
        
        # Trigger alerts if escalation required
        if validation_result.escalation_required:
            await self._trigger_escalation_alerts(validation_result, milestone)
        
        return validation_result
    
    async def _validate_criterion(
        self,
        criterion: SuccessCriterion,
        validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate individual success criterion."""
        
        result = {
            "score": 0.0,
            "evidence": {},
            "deliverable_status": "not_provided",
            "validation_notes": ""
        }
        
        # Validation based on method type
        if criterion.validation_method == ValidationMethod.STAKEHOLDER_SIGN_OFF:
            result = await self._validate_stakeholder_signoff(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.TECHNICAL_REVIEW:
            result = await self._validate_technical_review(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.AUTOMATED_VERIFICATION:
            result = await self._validate_automated_verification(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.AUTOMATED_TESTING:
            result = await self._validate_automated_testing(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.INTEGRATION_TESTING:
            result = await self._validate_integration_testing(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.SECURITY_AUDIT:
            result = await self._validate_security_audit(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.PERFORMANCE_TESTING:
            result = await self._validate_performance_testing(criterion, validation_data)
        
        elif criterion.validation_method == ValidationMethod.SATISFACTION_SURVEY:
            result = await self._validate_satisfaction_survey(criterion, validation_data)
        
        else:
            # Default validation for other methods
            result = await self._validate_default_method(criterion, validation_data)
        
        return result
    
    async def _validate_stakeholder_signoff(
        self,
        criterion: SuccessCriterion,
        validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate stakeholder sign-off criterion."""
        
        result = {
            "score": 0.0,
            "evidence": {},
            "deliverable_status": "not_provided",
            "validation_notes": ""
        }
        
        # Check for stakeholder approval data
        stakeholder_data = validation_data.get("stakeholder_approval", {})
        
        if not stakeholder_data:
            result["validation_notes"] = "No stakeholder approval data provided"
            return result
        
        # Calculate approval score
        total_stakeholders = stakeholder_data.get("total_stakeholders", 1)
        approved_stakeholders = stakeholder_data.get("approved_count", 0)
        
        if total_stakeholders > 0:
            approval_percentage = (approved_stakeholders / total_stakeholders) * 100
            result["score"] = min(approval_percentage, 100.0)
        
        # Evidence collection
        result["evidence"] = {
            "approval_percentage": approval_percentage,
            "stakeholder_feedback": stakeholder_data.get("feedback", []),
            "approval_timestamps": stakeholder_data.get("approval_timestamps", []),
            "deliverable_url": validation_data.get("deliverable_url", "")
        }
        
        result["deliverable_status"] = "provided" if result["score"] >= 80.0 else "incomplete"
        result["validation_notes"] = f"Stakeholder approval: {approval_percentage:.1f}%"
        
        return result
    
    async def _validate_automated_testing(
        self,
        criterion: SuccessCriterion,
        validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate automated testing criterion."""
        
        result = {
            "score": 0.0,
            "evidence": {},
            "deliverable_status": "not_provided",
            "validation_notes": ""
        }
        
        test_data = validation_data.get("test_results", {})
        
        if not test_data:
            result["validation_notes"] = "No test results provided"
            return result
        
        # Check test coverage
        test_coverage = test_data.get("coverage_percentage", 0.0)
        passing_tests = test_data.get("passing_tests", 0)
        total_tests = test_data.get("total_tests", 1)
        
        # Calculate score based on coverage and pass rate
        pass_rate = (passing_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Weighted score: 60% coverage, 40% pass rate
        result["score"] = (test_coverage * 0.6) + (pass_rate * 0.4)
        
        result["evidence"] = {
            "test_coverage": test_coverage,
            "pass_rate": pass_rate,
            "passing_tests": passing_tests,
            "total_tests": total_tests,
            "test_report_url": validation_data.get("test_report_url", ""),
            "failed_tests": test_data.get("failed_tests", [])
        }
        
        result["deliverable_status"] = "provided" if result["score"] >= 80.0 else "incomplete"
        result["validation_notes"] = f"Test coverage: {test_coverage:.1f}%, Pass rate: {pass_rate:.1f}%"
        
        return result
    
    async def _validate_performance_testing(
        self,
        criterion: SuccessCriterion,
        validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate performance testing criterion."""
        
        result = {
            "score": 0.0,
            "evidence": {},
            "deliverable_status": "not_provided",
            "validation_notes": ""
        }
        
        perf_data = validation_data.get("performance_results", {})
        
        if not perf_data:
            result["validation_notes"] = "No performance test results provided"
            return result
        
        # Check performance metrics
        response_time = perf_data.get("avg_response_time_ms", 0)
        throughput = perf_data.get("requests_per_second", 0)
        error_rate = perf_data.get("error_rate_percentage", 100)
        
        # Performance targets (configurable)
        target_response_time = perf_data.get("target_response_time_ms", 500)
        target_throughput = perf_data.get("target_requests_per_second", 100)
        max_error_rate = perf_data.get("max_error_rate_percentage", 1.0)
        
        # Calculate scores for each metric
        response_score = max(0, 100 - ((response_time - target_response_time) / target_response_time * 100)) if target_response_time > 0 else 0
        throughput_score = min(100, (throughput / target_throughput) * 100) if target_throughput > 0 else 0
        error_score = max(0, 100 - (error_rate / max_error_rate * 100)) if max_error_rate > 0 else 0
        
        # Overall performance score (weighted average)
        result["score"] = (response_score * 0.4) + (throughput_score * 0.3) + (error_score * 0.3)
        
        result["evidence"] = {
            "response_time_ms": response_time,
            "target_response_time_ms": target_response_time,
            "throughput_rps": throughput,
            "target_throughput_rps": target_throughput,
            "error_rate": error_rate,
            "max_error_rate": max_error_rate,
            "performance_report_url": validation_data.get("performance_report_url", "")
        }
        
        result["deliverable_status"] = "provided" if result["score"] >= 80.0 else "incomplete"
        result["validation_notes"] = f"Performance score: {result['score']:.1f}%"
        
        return result
    
    async def _validate_satisfaction_survey(
        self,
        criterion: SuccessCriterion,
        validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate customer satisfaction survey criterion."""
        
        result = {
            "score": 0.0,
            "evidence": {},
            "deliverable_status": "not_provided",
            "validation_notes": ""
        }
        
        survey_data = validation_data.get("satisfaction_survey", {})
        
        if not survey_data:
            result["validation_notes"] = "No satisfaction survey data provided"
            return result
        
        # Calculate satisfaction score
        overall_score = survey_data.get("overall_satisfaction", 0.0)  # Scale 1-10
        response_rate = survey_data.get("response_rate", 0.0)  # Percentage
        
        # Normalize to 100-point scale and factor in response rate
        satisfaction_score = (overall_score / 10.0) * 100
        response_weight = min(response_rate / 80.0, 1.0)  # 80% response rate = full weight
        
        result["score"] = satisfaction_score * response_weight
        
        result["evidence"] = {
            "overall_satisfaction": overall_score,
            "response_rate": response_rate,
            "total_responses": survey_data.get("total_responses", 0),
            "detailed_feedback": survey_data.get("detailed_feedback", []),
            "nps_score": survey_data.get("nps_score", 0),
            "survey_results_url": validation_data.get("survey_results_url", "")
        }
        
        result["deliverable_status"] = "provided" if result["score"] >= 80.0 else "incomplete"
        result["validation_notes"] = f"Satisfaction: {overall_score:.1f}/10, Response rate: {response_rate:.1f}%"
        
        return result
    
    async def _store_milestone_plan(
        self,
        guarantee_id: str,
        milestone_plan: Dict[str, WeeklyMilestone]
    ):
        """Store milestone plan in Redis."""
        
        plan_data = {}
        for week_key, milestone in milestone_plan.items():
            milestone_dict = {
                "milestone_id": milestone.milestone_id,
                "guarantee_id": milestone.guarantee_id,
                "week_number": milestone.week_number,
                "title": milestone.title,
                "description": milestone.description,
                "minimum_success_threshold": milestone.minimum_success_threshold,
                "escalation_threshold": milestone.escalation_threshold,
                "status": milestone.status.value,
                "success_criteria": []
            }
            
            for criterion in milestone.success_criteria:
                criterion_dict = {
                    "criterion_id": criterion.criterion_id,
                    "criterion_name": criterion.criterion_name,
                    "description": criterion.description,
                    "weight": criterion.weight,
                    "validation_method": criterion.validation_method.value,
                    "deliverable": criterion.deliverable,
                    "target_value": criterion.target_value
                }
                milestone_dict["success_criteria"].append(criterion_dict)
            
            plan_data[week_key] = milestone_dict
        
        await self.redis.setex(
            f"milestone_plan:{guarantee_id}",
            86400 * 35,  # 35 days TTL
            json.dumps(plan_data, default=str)
        )
    
    async def get_milestone_status(self, milestone_id: str) -> Dict[str, Any]:
        """Get current status of a milestone."""
        
        try:
            # Get milestone from stored plan
            milestone = await self._get_milestone_details(milestone_id)
            
            if not milestone:
                return {"status": "error", "message": "Milestone not found"}
            
            # Get validation history
            validation_history = await self._get_validation_history(milestone_id)
            
            return {
                "status": "success",
                "milestone_id": milestone_id,
                "title": milestone.title,
                "week_number": milestone.week_number,
                "current_status": milestone.status.value,
                "overall_score": milestone.overall_score,
                "minimum_success_threshold": milestone.minimum_success_threshold,
                "escalation_threshold": milestone.escalation_threshold,
                "validation_timestamp": milestone.validation_timestamp.isoformat() if milestone.validation_timestamp else None,
                "success_criteria": [
                    {
                        "criterion_name": criterion.criterion_name,
                        "weight": criterion.weight,
                        "completion_status": criterion.completion_status,
                        "target_value": criterion.target_value,
                        "current_value": criterion.current_value
                    }
                    for criterion in milestone.success_criteria
                ],
                "next_actions": milestone.next_actions,
                "validation_history": validation_history
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get milestone status: {e}")
            return {"status": "error", "message": str(e)}


# Global service instance
_milestone_framework: Optional[WeeklyMilestoneFramework] = None


async def get_milestone_framework() -> WeeklyMilestoneFramework:
    """Get the global weekly milestone framework instance."""
    global _milestone_framework
    
    if _milestone_framework is None:
        redis_client = await get_redis_client()
        logger = logging.getLogger(__name__)
        _milestone_framework = WeeklyMilestoneFramework(redis_client, logger)
    
    return _milestone_framework


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class WeeklyMilestoneFrameworkScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            async def test_milestone_framework():
            """Test the weekly milestone framework."""

            framework = await get_milestone_framework()

            # Create milestone plan
            guarantee_id = "guarantee_test_20250801"
            milestone_plan = await framework.create_milestone_plan(
            guarantee_id, "mvp_development"
            )

            self.logger.info(f"Created milestone plan with {len(milestone_plan)} weeks")

            # Test milestone validation
            week_1_milestone = milestone_plan["week_1"]

            # Sample validation data
            validation_data = {
            week_1_milestone.success_criteria[0].criterion_id: {
            "stakeholder_approval": {
            "total_stakeholders": 3,
            "approved_count": 3,
            "feedback": ["Excellent requirements analysis", "Very thorough", "Ready to proceed"],
            "approval_timestamps": ["2025-08-01T10:00:00Z", "2025-08-01T11:00:00Z", "2025-08-01T12:00:00Z"]
            },
            "deliverable_url": "https://docs.company.com/requirements-spec"
            },
            week_1_milestone.success_criteria[1].criterion_id: {
            "technical_review": {
            "review_score": 9.2,
            "reviewer_feedback": "Solid architecture design with good scalability considerations",
            "review_date": "2025-08-01T15:00:00Z"
            },
            "deliverable_url": "https://docs.company.com/architecture-design"
            }
            }

            validation_result = await framework.validate_weekly_milestone(
            week_1_milestone.milestone_id,
            validation_data
            )

            self.logger.info(f"Validation Result:")
            self.logger.info(f"Overall Score: {validation_result.overall_score:.1f}")
            self.logger.info(f"Status: {validation_result.status.value}")
            self.logger.info(f"Escalation Required: {validation_result.escalation_required}")
            self.logger.info(f"Recommendations: {validation_result.recommendations}")

            # Run test
            await test_milestone_framework()
            
            return {"status": "completed"}
    
    script_main(WeeklyMilestoneFrameworkScript)