"""
TaskExecutionAPI Scheduling - Unified Intelligent Scheduling

Consolidated intelligent scheduling functionality from:
- app/api/intelligent_scheduling.py (ML-based scheduling with pattern analysis)
- app/api/v1/orchestrator_core.py (core scheduling patterns)
- app/api/v1/team_coordination.py (team-based scheduling coordination)

Features:
- Machine learning-based pattern analysis and optimization
- Predictive scheduling with resource optimization  
- Conflict resolution and automated rescheduling
- Real-time performance monitoring and adaptation
- Epic 1 ConsolidatedProductionOrchestrator integration
- Team coordination and load balancing
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
import structlog

from .models import (
    ScheduleRequest, ScheduleResponse, PatternAnalysisRequest, 
    ConflictResolutionRequest, OperationResponse
)
from .middleware import (
    require_schedule_optimize, require_task_read, require_admin_access,
    check_rate_limit_dependency, audit_operation, with_circuit_breaker,
    scheduling_circuit_breaker
)

from app.core.production_orchestrator import create_production_orchestrator
from app.core.redis_integration import get_redis_service
from app.core.intelligent_sleep_manager import get_intelligent_sleep_manager
from app.core.sleep_analytics import get_sleep_analytics_engine


logger = structlog.get_logger(__name__)
router = APIRouter()


# ===============================================================================
# INTELLIGENT SCHEDULING SERVICE
# ===============================================================================

class IntelligentSchedulingService:
    """
    Unified intelligent scheduling service consolidating ML-based scheduling,
    pattern analysis, and conflict resolution capabilities.
    
    Integrates with Epic 1 ConsolidatedProductionOrchestrator for comprehensive
    task and workflow scheduling optimization.
    """
    
    def __init__(self):
        self.orchestrator = None
        self.redis_service = None
        self.intelligent_manager = None
        self.analytics_engine = None
        
    async def initialize(self):
        """Initialize scheduling service dependencies."""
        self.orchestrator = create_production_orchestrator()
        self.redis_service = get_redis_service()
        self.intelligent_manager = get_intelligent_sleep_manager()
        self.analytics_engine = get_sleep_analytics_engine()
    
    async def analyze_activity_patterns_comprehensive(
        self,
        request: PatternAnalysisRequest,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive activity pattern analysis with ML insights.
        
        Args:
            request: Pattern analysis parameters
            user_id: User requesting analysis
            
        Returns:
            Dict containing pattern analysis results and recommendations
        """
        try:
            logger.info("Analyzing activity patterns",
                       agent_id=request.agent_id,
                       analysis_period_days=request.analysis_period_days,
                       pattern_types=request.pattern_types,
                       user_id=user_id)
            
            # Perform pattern analysis through intelligent manager
            patterns = await self.intelligent_manager.analyze_activity_patterns(
                agent_id=request.agent_id,
                analysis_period=timedelta(days=request.analysis_period_days),
                pattern_types=request.pattern_types
            )
            
            # Get predictive insights if requested
            predictions = {}
            if request.include_predictions:
                predictions = await self.analytics_engine.get_predictive_insights(
                    agent_id=request.agent_id,
                    forecast_hours=24
                )
                patterns["predictions"] = predictions
            
            # Generate optimization insights
            insights = await self.intelligent_manager.generate_optimization_insights(
                patterns, request.agent_id
            )
            
            # Generate actionable recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                patterns, insights, request.pattern_types
            )
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_pattern_performance_metrics(
                patterns, request.agent_id
            )
            
            result = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_id": str(request.agent_id) if request.agent_id else "system-wide",
                "analysis_period_days": request.analysis_period_days,
                "pattern_types": request.pattern_types,
                "patterns": patterns,
                "insights": insights,
                "predictions": predictions,
                "recommendations": recommendations,
                "performance_metrics": performance_metrics,
                "confidence_score": self._calculate_analysis_confidence(patterns)
            }
            
            # Cache analysis results
            if self.redis_service:
                cache_key = f"pattern_analysis:{request.agent_id or 'system'}:{request.analysis_period_days}"
                await self.redis_service.cache_set(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error("Pattern analysis failed",
                        agent_id=request.agent_id,
                        error=str(e))
            raise
    
    async def generate_optimal_schedule_advanced(
        self,
        request: ScheduleRequest,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate optimal schedule using advanced ML optimization.
        
        Args:
            request: Schedule generation parameters
            user_id: User requesting scheduling
            
        Returns:
            Dict containing optimal schedule and validation results
        """
        try:
            logger.info("Generating optimal schedule",
                       agent_id=request.agent_id,
                       optimization_goal=request.optimization_goal,
                       time_horizon_hours=request.time_horizon_hours,
                       user_id=user_id)
            
            # Generate optimal schedule through intelligent manager
            optimal_schedule = await self.intelligent_manager.generate_optimal_schedule(
                agent_id=request.agent_id,
                optimization_goal=request.optimization_goal,
                time_horizon=timedelta(hours=request.time_horizon_hours),
                constraints=request.constraints,
                resource_limits=request.resource_limits,
                blackout_periods=request.blackout_periods
            )
            
            # Validate schedule feasibility
            validation_results = await self.intelligent_manager.validate_schedule_feasibility(
                optimal_schedule, request.agent_id
            )
            
            # Calculate performance predictions
            performance_predictions = await self.intelligent_manager.predict_schedule_performance(
                optimal_schedule, request.agent_id
            )
            
            # Generate task assignments from schedule
            task_assignments = await self._generate_task_assignments_from_schedule(
                optimal_schedule, request
            )
            
            # Generate time slots allocation
            time_slots = await self._generate_time_slots_from_schedule(
                optimal_schedule, request.time_horizon_hours
            )
            
            # Calculate confidence and efficiency scores
            confidence_score = self._calculate_schedule_confidence(
                optimal_schedule, validation_results, performance_predictions
            )
            
            efficiency_score = self._calculate_schedule_efficiency(
                optimal_schedule, performance_predictions
            )
            
            result = {
                "schedule_id": optimal_schedule["id"],
                "schedule_name": f"Optimal Schedule - {request.optimization_goal}",
                "optimization_goal": request.optimization_goal,
                "agent_id": str(request.agent_id) if request.agent_id else None,
                "schedule": optimal_schedule,
                "task_assignments": task_assignments,
                "time_slots": time_slots,
                "validation_results": validation_results,
                "performance_predictions": performance_predictions,
                "confidence_score": confidence_score,
                "efficiency_score": efficiency_score,
                "created_at": datetime.utcnow().isoformat(),
                "valid_from": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=request.time_horizon_hours)).isoformat(),
                "resource_utilization": await self._calculate_resource_utilization(optimal_schedule),
                "optimization_insights": await self._generate_optimization_insights(optimal_schedule, request)
            }
            
            # Cache schedule for fast retrieval
            if self.redis_service:
                await self.redis_service.cache_set(
                    f"schedule:{optimal_schedule['id']}", 
                    result, 
                    ttl=request.time_horizon_hours * 3600
                )
            
            return result
            
        except Exception as e:
            logger.error("Schedule generation failed",
                        agent_id=request.agent_id,
                        error=str(e))
            raise
    
    async def resolve_schedule_conflicts_intelligent(
        self,
        request: ConflictResolutionRequest,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Intelligent resolution of scheduling conflicts with optimization.
        
        Args:
            request: Conflict resolution parameters
            user_id: User requesting resolution
            
        Returns:
            Dict containing conflict resolution results
        """
        try:
            logger.info("Resolving schedule conflicts",
                       strategy=request.conflict_resolution_strategy,
                       allow_rescheduling=request.allow_rescheduling,
                       user_id=user_id)
            
            # Detect existing conflicts
            conflicts = await self.orchestrator.detect_schedule_conflicts()
            
            if not conflicts:
                return {
                    "conflicts_found": False,
                    "message": "No schedule conflicts detected",
                    "timestamp": datetime.utcnow().isoformat(),
                    "resolution_summary": {
                        "total_conflicts": 0,
                        "resolved_conflicts": 0,
                        "remaining_conflicts": 0
                    }
                }
            
            logger.info("Schedule conflicts detected",
                       total_conflicts=len(conflicts))
            
            # Analyze conflict complexity and impact
            conflict_analysis = await self._analyze_conflict_complexity(conflicts)
            
            # Resolve conflicts using specified strategy
            resolution_results = await self.orchestrator.resolve_conflicts(
                conflicts=conflicts,
                strategy=request.conflict_resolution_strategy,
                priority_weights=request.priority_weights,
                allow_rescheduling=request.allow_rescheduling,
                force_resolution=request.force_resolution
            )
            
            # Generate optimization recommendations for remaining conflicts
            optimization_recommendations = []
            if resolution_results.get("unresolved"):
                optimization_recommendations = await self._generate_conflict_optimization_recommendations(
                    resolution_results["unresolved"]
                )
            
            # Calculate resolution metrics
            resolution_metrics = await self._calculate_resolution_metrics(
                conflicts, resolution_results
            )
            
            # Update caches and notify relevant systems
            if self.redis_service:
                # Invalidate affected schedule caches
                await self._invalidate_affected_schedule_caches(resolution_results)
                
                # Publish conflict resolution event
                await self.redis_service.publish("scheduling_events", {
                    "event": "conflicts_resolved",
                    "total_conflicts": len(conflicts),
                    "resolved_count": len(resolution_results.get("resolved", [])),
                    "strategy": request.conflict_resolution_strategy,
                    "resolved_by": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return {
                "conflicts_found": True,
                "total_conflicts": len(conflicts),
                "conflicts_resolved": len(resolution_results.get("resolved", [])),
                "conflicts_remaining": len(resolution_results.get("unresolved", [])),
                "resolution_strategy": request.conflict_resolution_strategy,
                "resolution_details": resolution_results,
                "conflict_analysis": conflict_analysis,
                "resolution_metrics": resolution_metrics,
                "optimization_recommendations": optimization_recommendations,
                "rescheduled_operations": resolution_results.get("rescheduled", []),
                "timestamp": datetime.utcnow().isoformat(),
                "resolution_summary": {
                    "success_rate": (len(resolution_results.get("resolved", [])) / len(conflicts)) * 100,
                    "average_resolution_time": resolution_metrics.get("average_resolution_time", 0),
                    "resource_impact": resolution_metrics.get("resource_impact", {})
                }
            }
            
        except Exception as e:
            logger.error("Conflict resolution failed", error=str(e))
            raise
    
    async def get_predictive_forecast_comprehensive(
        self,
        agent_id: Optional[str],
        forecast_hours: int,
        include_confidence: bool,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive predictive forecast with ML insights.
        
        Args:
            agent_id: Agent ID for forecast (None for system-wide)
            forecast_hours: Forecast time horizon
            include_confidence: Include confidence intervals
            user_id: User requesting forecast
            
        Returns:
            Dict containing comprehensive forecast data
        """
        try:
            logger.info("Generating predictive forecast",
                       agent_id=agent_id,
                       forecast_hours=forecast_hours,
                       include_confidence=include_confidence,
                       user_id=user_id)
            
            # Generate predictive forecast through intelligent manager
            forecast = await self.intelligent_manager.generate_predictive_forecast(
                agent_id=agent_id,
                forecast_horizon=timedelta(hours=forecast_hours),
                include_confidence=include_confidence
            )
            
            # Add behavioral predictions
            behavioral_predictions = await self.analytics_engine.predict_agent_behavior(
                agent_id=agent_id,
                prediction_window=timedelta(hours=forecast_hours)
            )
            
            # Generate optimization opportunities
            optimization_opportunities = await self.intelligent_manager.identify_optimization_opportunities(
                forecast, behavioral_predictions
            )
            
            # Calculate risk assessment
            risk_assessment = await self._calculate_forecast_risk_assessment(
                forecast, behavioral_predictions
            )
            
            # Generate scenario analysis
            scenario_analysis = await self._generate_scenario_analysis(
                forecast, forecast_hours
            )
            
            result = {
                "forecast_id": f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "agent_id": str(agent_id) if agent_id else "system-wide",
                "forecast_horizon_hours": forecast_hours,
                "generated_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=forecast_hours)).isoformat(),
                "forecast": forecast,
                "behavioral_predictions": behavioral_predictions,
                "optimization_opportunities": optimization_opportunities,
                "risk_assessment": risk_assessment,
                "scenario_analysis": scenario_analysis,
                "confidence_metrics": {
                    "overall_confidence": self._calculate_forecast_confidence(forecast),
                    "prediction_accuracy": await self._get_historical_prediction_accuracy(agent_id),
                    "data_quality_score": await self._calculate_forecast_data_quality(agent_id)
                }
            }
            
            # Cache forecast results
            if self.redis_service:
                await self.redis_service.cache_set(
                    f"forecast:{result['forecast_id']}", 
                    result, 
                    ttl=min(forecast_hours * 3600, 86400)  # Max 24 hours
                )
            
            return result
            
        except Exception as e:
            logger.error("Predictive forecast failed",
                        agent_id=agent_id,
                        error=str(e))
            raise
    
    # ===============================================================================
    # HELPER METHODS
    # ===============================================================================
    
    async def _generate_comprehensive_recommendations(
        self, 
        patterns: Dict[str, Any], 
        insights: Dict[str, Any],
        pattern_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive actionable recommendations."""
        recommendations = []
        
        try:
            # Analyze sleep patterns
            if "sleep" in pattern_types and "sleep_patterns" in patterns:
                sleep_efficiency = patterns["sleep_patterns"].get("efficiency", 0)
                if sleep_efficiency < 0.8:
                    recommendations.append({
                        "type": "optimization",
                        "priority": "high",
                        "category": "sleep_efficiency",
                        "title": "Improve Sleep Cycle Efficiency", 
                        "description": f"Sleep efficiency is {sleep_efficiency:.1%}, below optimal threshold",
                        "action": "Review sleep timing and consolidation parameters",
                        "expected_impact": "15-25% performance improvement",
                        "implementation_effort": "low"
                    })
            
            # Analyze activity patterns
            if "activity" in pattern_types and "activity_patterns" in patterns:
                activity_variance = patterns["activity_patterns"].get("variance", 0)
                if activity_variance > 0.5:
                    recommendations.append({
                        "type": "scheduling",
                        "priority": "medium",
                        "category": "activity_stabilization",
                        "title": "Stabilize Activity Patterns",
                        "description": f"High activity variance detected ({activity_variance:.1%})",
                        "action": "Consider more consistent scheduling intervals",
                        "expected_impact": "10-20% consistency improvement",
                        "implementation_effort": "medium"
                    })
            
            # Analyze resource utilization
            if "resource_utilization" in insights:
                cpu_efficiency = insights["resource_utilization"].get("cpu_efficiency", 1.0)
                if cpu_efficiency < 0.7:
                    recommendations.append({
                        "type": "resource",
                        "priority": "medium", 
                        "category": "resource_optimization",
                        "title": "Optimize Resource Usage",
                        "description": f"CPU efficiency is {cpu_efficiency:.1%}",
                        "action": "Review consolidation algorithms and background tasks",
                        "expected_impact": "20-30% resource efficiency gain",
                        "implementation_effort": "high"
                    })
            
            # Analyze performance patterns
            if "performance" in pattern_types and "performance_patterns" in patterns:
                response_time_trend = patterns["performance_patterns"].get("response_time_trend", "stable")
                if response_time_trend == "increasing":
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "category": "performance_degradation",
                        "title": "Address Performance Degradation",
                        "description": "Response times showing increasing trend",
                        "action": "Investigate and optimize critical path operations",
                        "expected_impact": "25-40% response time improvement",
                        "implementation_effort": "high"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            return [{
                "type": "error",
                "priority": "low",
                "category": "system_error",
                "title": "Recommendation Generation Failed",
                "description": str(e),
                "action": "Review system logs for details"
            }]
    
    def _calculate_analysis_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern analysis."""
        try:
            # Base confidence on data quality and pattern consistency
            data_points = patterns.get("metadata", {}).get("data_points", 0)
            pattern_consistency = patterns.get("metadata", {}).get("consistency_score", 0.5)
            
            # Higher confidence with more data points and consistent patterns
            data_confidence = min(1.0, data_points / 1000.0)  # Normalize to 1000 data points
            
            overall_confidence = (data_confidence * 0.6) + (pattern_consistency * 0.4)
            return round(overall_confidence, 3)
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    async def _calculate_pattern_performance_metrics(
        self,
        patterns: Dict[str, Any],
        agent_id: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for pattern analysis."""
        try:
            metrics = {
                "analysis_completeness": 0.0,
                "data_quality_score": 0.0,
                "pattern_strength": 0.0,
                "predictive_accuracy": 0.0
            }
            
            # Calculate analysis completeness
            expected_patterns = ["activity_patterns", "sleep_patterns", "performance_patterns"]
            found_patterns = [p for p in expected_patterns if p in patterns]
            metrics["analysis_completeness"] = len(found_patterns) / len(expected_patterns)
            
            # Estimate data quality based on pattern metadata
            if "metadata" in patterns:
                metadata = patterns["metadata"]
                metrics["data_quality_score"] = min(1.0, metadata.get("data_points", 0) / 500.0)
                metrics["pattern_strength"] = metadata.get("strength_score", 0.5)
            
            # Get historical accuracy if available
            if agent_id and self.analytics_engine:
                accuracy = await self._get_historical_prediction_accuracy(agent_id)
                metrics["predictive_accuracy"] = accuracy
            
            return metrics
            
        except Exception as e:
            logger.warning("Failed to calculate pattern performance metrics", error=str(e))
            return {"error": str(e)}
    
    async def _generate_task_assignments_from_schedule(
        self,
        schedule: Dict[str, Any],
        request: ScheduleRequest
    ) -> List[Dict[str, Any]]:
        """Generate task assignments from optimal schedule."""
        try:
            assignments = []
            
            # Extract task assignments from schedule structure
            if "task_slots" in schedule:
                for slot in schedule["task_slots"]:
                    assignment = {
                        "task_id": slot.get("task_id"),
                        "agent_id": slot.get("agent_id"),
                        "start_time": slot.get("start_time"),
                        "end_time": slot.get("end_time"),
                        "estimated_effort": slot.get("estimated_effort"),
                        "priority": slot.get("priority"),
                        "resource_requirements": slot.get("resources", {}),
                        "confidence": slot.get("assignment_confidence", 0.8)
                    }
                    assignments.append(assignment)
            
            return assignments
            
        except Exception as e:
            logger.warning("Failed to generate task assignments", error=str(e))
            return []
    
    async def _generate_time_slots_from_schedule(
        self,
        schedule: Dict[str, Any],
        time_horizon_hours: int
    ) -> List[Dict[str, Any]]:
        """Generate time slot allocations from schedule."""
        try:
            time_slots = []
            
            # Generate hourly time slots
            start_time = datetime.utcnow()
            for hour in range(time_horizon_hours):
                slot_start = start_time + timedelta(hours=hour)
                slot_end = slot_start + timedelta(hours=1)
                
                slot = {
                    "slot_id": f"slot_{hour:03d}",
                    "start_time": slot_start.isoformat(),
                    "end_time": slot_end.isoformat(),
                    "allocated_tasks": [],
                    "resource_utilization": 0.0,
                    "availability": "available"
                }
                
                # Check if any tasks are scheduled in this slot
                if "task_slots" in schedule:
                    for task_slot in schedule["task_slots"]:
                        task_start = datetime.fromisoformat(task_slot.get("start_time", slot_start.isoformat()))
                        task_end = datetime.fromisoformat(task_slot.get("end_time", slot_end.isoformat()))
                        
                        # Check for overlap
                        if task_start < slot_end and task_end > slot_start:
                            slot["allocated_tasks"].append({
                                "task_id": task_slot.get("task_id"),
                                "agent_id": task_slot.get("agent_id"),
                                "overlap_duration": min(slot_end, task_end) - max(slot_start, task_start)
                            })
                            slot["resource_utilization"] += 0.8  # Estimate utilization
                            slot["availability"] = "busy" if slot["resource_utilization"] >= 1.0 else "partial"
                
                time_slots.append(slot)
            
            return time_slots
            
        except Exception as e:
            logger.warning("Failed to generate time slots", error=str(e))
            return []
    
    def _calculate_schedule_confidence(
        self,
        schedule: Dict[str, Any],
        validation_results: Dict[str, Any],
        performance_predictions: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for generated schedule."""
        try:
            # Base confidence on validation and prediction accuracy
            validation_score = validation_results.get("feasibility_score", 0.5)
            prediction_accuracy = performance_predictions.get("accuracy_estimate", 0.5)
            resource_availability = validation_results.get("resource_availability", 0.5)
            
            # Weight different factors
            confidence = (
                validation_score * 0.4 +
                prediction_accuracy * 0.3 +
                resource_availability * 0.3
            )
            
            return round(confidence, 3)
            
        except Exception:
            return 0.7  # Default reasonable confidence
    
    def _calculate_schedule_efficiency(
        self,
        schedule: Dict[str, Any],
        performance_predictions: Dict[str, Any]
    ) -> float:
        """Calculate efficiency score for generated schedule."""
        try:
            # Calculate efficiency based on resource utilization and task distribution
            resource_efficiency = schedule.get("resource_utilization", {}).get("overall", 0.7)
            task_distribution_efficiency = schedule.get("task_distribution_score", 0.7)
            time_efficiency = performance_predictions.get("time_efficiency", 0.7)
            
            efficiency = (
                resource_efficiency * 0.4 +
                task_distribution_efficiency * 0.3 +
                time_efficiency * 0.3
            )
            
            return round(efficiency, 3)
            
        except Exception:
            return 0.7  # Default efficiency score
    
    async def _calculate_resource_utilization(
        self,
        schedule: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate resource utilization metrics for schedule."""
        try:
            utilization = {
                "cpu": 0.0,
                "memory": 0.0, 
                "network": 0.0,
                "agents": 0.0,
                "overall": 0.0
            }
            
            # Extract utilization from schedule if available
            if "resource_metrics" in schedule:
                metrics = schedule["resource_metrics"]
                utilization.update({
                    "cpu": metrics.get("cpu_utilization", 0.0),
                    "memory": metrics.get("memory_utilization", 0.0),
                    "network": metrics.get("network_utilization", 0.0),
                    "agents": metrics.get("agent_utilization", 0.0)
                })
                
                # Calculate overall utilization
                utilization["overall"] = sum(utilization.values()) / 4
            
            return utilization
            
        except Exception as e:
            logger.warning("Failed to calculate resource utilization", error=str(e))
            return {"overall": 0.5, "error": str(e)}
    
    async def _generate_optimization_insights(
        self,
        schedule: Dict[str, Any],
        request: ScheduleRequest
    ) -> List[Dict[str, Any]]:
        """Generate optimization insights for the schedule."""
        try:
            insights = []
            
            # Analyze resource utilization patterns
            resource_util = await self._calculate_resource_utilization(schedule)
            if resource_util.get("overall", 0) < 0.6:
                insights.append({
                    "type": "underutilization",
                    "title": "Resource Underutilization Detected",
                    "description": f"Overall resource utilization is {resource_util.get('overall', 0):.1%}",
                    "recommendation": "Consider consolidating tasks or reducing resource allocation",
                    "impact": "medium"
                })
            
            # Analyze scheduling efficiency
            if "task_distribution_score" in schedule and schedule["task_distribution_score"] < 0.7:
                insights.append({
                    "type": "distribution",
                    "title": "Suboptimal Task Distribution",
                    "description": "Task distribution could be improved for better load balancing",
                    "recommendation": "Review agent capabilities and workload distribution",
                    "impact": "high"
                })
            
            # Analyze time horizon efficiency
            if request.time_horizon_hours > 24:
                insights.append({
                    "type": "time_horizon",
                    "title": "Extended Time Horizon",
                    "description": f"Long time horizon ({request.time_horizon_hours}h) may reduce accuracy",
                    "recommendation": "Consider shorter planning horizons for better accuracy",
                    "impact": "low"
                })
            
            return insights
            
        except Exception as e:
            logger.warning("Failed to generate optimization insights", error=str(e))
            return []
    
    async def _analyze_conflict_complexity(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze complexity and impact of scheduling conflicts."""
        try:
            analysis = {
                "total_conflicts": len(conflicts),
                "complexity_score": 0.0,
                "impact_assessment": {},
                "conflict_types": {},
                "affected_resources": set(),
                "time_span_hours": 0.0
            }
            
            # Analyze conflict types and complexity
            for conflict in conflicts:
                conflict_type = conflict.get("type", "unknown")
                analysis["conflict_types"][conflict_type] = analysis["conflict_types"].get(conflict_type, 0) + 1
                
                # Track affected resources
                if "affected_agents" in conflict:
                    analysis["affected_resources"].update(conflict["affected_agents"])
                if "affected_tasks" in conflict:
                    analysis["affected_resources"].update(conflict["affected_tasks"])
            
            # Calculate complexity score based on number and types of conflicts
            base_complexity = min(1.0, len(conflicts) / 10.0)  # Normalize to 10 conflicts
            type_diversity = len(analysis["conflict_types"]) / 5.0  # Assume max 5 types
            resource_impact = len(analysis["affected_resources"]) / 20.0  # Normalize to 20 resources
            
            analysis["complexity_score"] = (base_complexity + type_diversity + resource_impact) / 3.0
            analysis["affected_resources"] = list(analysis["affected_resources"])  # Convert to list for JSON
            
            return analysis
            
        except Exception as e:
            logger.warning("Failed to analyze conflict complexity", error=str(e))
            return {"error": str(e)}
    
    async def _generate_conflict_optimization_recommendations(
        self,
        unresolved_conflicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations for unresolved conflicts."""
        recommendations = []
        
        try:
            for conflict in unresolved_conflicts:
                conflict_type = conflict.get("type", "unknown")
                
                if conflict_type == "resource_overallocation":
                    recommendations.append({
                        "conflict_id": conflict.get("id"),
                        "type": "resource_scaling",
                        "title": "Scale Resources to Resolve Overallocation",
                        "description": "Add additional agents or increase resource limits",
                        "priority": "high",
                        "estimated_effort": "medium"
                    })
                
                elif conflict_type == "time_overlap":
                    recommendations.append({
                        "conflict_id": conflict.get("id"),
                        "type": "time_adjustment",
                        "title": "Adjust Task Scheduling to Avoid Overlap",
                        "description": "Reschedule conflicting tasks to different time slots",
                        "priority": "medium",
                        "estimated_effort": "low"
                    })
                
                elif conflict_type == "dependency_violation":
                    recommendations.append({
                        "conflict_id": conflict.get("id"),
                        "type": "dependency_reorder",
                        "title": "Reorder Tasks to Respect Dependencies",
                        "description": "Adjust task sequence to maintain dependency constraints",
                        "priority": "critical",
                        "estimated_effort": "high"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.warning("Failed to generate conflict recommendations", error=str(e))
            return []
    
    async def _calculate_resolution_metrics(
        self,
        original_conflicts: List[Dict[str, Any]],
        resolution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for conflict resolution performance."""
        try:
            resolved_count = len(resolution_results.get("resolved", []))
            total_count = len(original_conflicts)
            
            metrics = {
                "resolution_rate": (resolved_count / total_count) * 100 if total_count > 0 else 0,
                "total_conflicts": total_count,
                "resolved_conflicts": resolved_count,
                "unresolved_conflicts": total_count - resolved_count,
                "average_resolution_time": 0.0,
                "resource_impact": {},
                "success_by_type": {}
            }
            
            # Calculate average resolution time if available
            if "resolved" in resolution_results:
                total_time = sum(
                    conflict.get("resolution_time", 0) 
                    for conflict in resolution_results["resolved"]
                )
                metrics["average_resolution_time"] = total_time / resolved_count if resolved_count > 0 else 0
            
            # Analyze success rate by conflict type
            for conflict in original_conflicts:
                conflict_type = conflict.get("type", "unknown")
                if conflict_type not in metrics["success_by_type"]:
                    metrics["success_by_type"][conflict_type] = {"total": 0, "resolved": 0}
                
                metrics["success_by_type"][conflict_type]["total"] += 1
                
                # Check if this conflict was resolved
                conflict_id = conflict.get("id")
                if any(r.get("id") == conflict_id for r in resolution_results.get("resolved", [])):
                    metrics["success_by_type"][conflict_type]["resolved"] += 1
            
            # Calculate success rates by type
            for conflict_type, data in metrics["success_by_type"].items():
                data["success_rate"] = (data["resolved"] / data["total"]) * 100 if data["total"] > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.warning("Failed to calculate resolution metrics", error=str(e))
            return {"error": str(e)}
    
    async def _invalidate_affected_schedule_caches(
        self,
        resolution_results: Dict[str, Any]
    ):
        """Invalidate schedule caches affected by conflict resolution."""
        try:
            if not self.redis_service:
                return
                
            # Collect affected schedule IDs
            affected_schedules = set()
            
            for resolved_conflict in resolution_results.get("resolved", []):
                if "affected_schedules" in resolved_conflict:
                    affected_schedules.update(resolved_conflict["affected_schedules"])
            
            # Invalidate caches
            for schedule_id in affected_schedules:
                await self.redis_service.cache_delete(f"schedule:{schedule_id}")
                
            logger.info("Invalidated schedule caches", 
                       affected_schedules=len(affected_schedules))
            
        except Exception as e:
            logger.warning("Failed to invalidate schedule caches", error=str(e))
    
    async def _calculate_forecast_risk_assessment(
        self,
        forecast: Dict[str, Any],
        behavioral_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk assessment for forecast."""
        try:
            risk_factors = []
            overall_risk = "low"
            
            # Analyze prediction confidence
            confidence = forecast.get("confidence", 0.8)
            if confidence < 0.6:
                risk_factors.append({
                    "type": "low_confidence",
                    "severity": "medium",
                    "description": f"Forecast confidence is {confidence:.1%}",
                    "mitigation": "Increase historical data or reduce forecast horizon"
                })
                overall_risk = "medium"
            
            # Analyze behavioral volatility
            volatility = behavioral_predictions.get("volatility_score", 0.3)
            if volatility > 0.7:
                risk_factors.append({
                    "type": "high_volatility", 
                    "severity": "high",
                    "description": f"High behavioral volatility detected ({volatility:.1%})",
                    "mitigation": "Use shorter prediction windows and frequent updates"
                })
                overall_risk = "high"
            
            # Analyze external factors
            external_factors = forecast.get("external_factors", {})
            if external_factors.get("instability_indicators", []):
                risk_factors.append({
                    "type": "external_instability",
                    "severity": "medium",
                    "description": "External instability indicators detected",
                    "mitigation": "Monitor external conditions and adjust predictions"
                })
            
            return {
                "overall_risk": overall_risk,
                "risk_factors": risk_factors,
                "risk_score": len(risk_factors) / 3.0,  # Normalize to 3 potential factors
                "mitigation_strategies": [factor.get("mitigation") for factor in risk_factors]
            }
            
        except Exception as e:
            logger.warning("Failed to calculate forecast risk assessment", error=str(e))
            return {"overall_risk": "unknown", "error": str(e)}
    
    async def _generate_scenario_analysis(
        self,
        forecast: Dict[str, Any],
        forecast_hours: int
    ) -> Dict[str, Any]:
        """Generate scenario analysis for forecast."""
        try:
            scenarios = {
                "optimistic": {},
                "realistic": {},
                "pessimistic": {}
            }
            
            base_performance = forecast.get("performance_prediction", 0.8)
            
            # Generate optimistic scenario (20% better)
            scenarios["optimistic"] = {
                "performance_multiplier": 1.2,
                "expected_performance": base_performance * 1.2,
                "probability": 0.15,
                "description": "Best case scenario with optimal conditions"
            }
            
            # Realistic scenario (base case)
            scenarios["realistic"] = {
                "performance_multiplier": 1.0,
                "expected_performance": base_performance,
                "probability": 0.7,
                "description": "Most likely scenario based on current trends"
            }
            
            # Pessimistic scenario (20% worse)
            scenarios["pessimistic"] = {
                "performance_multiplier": 0.8,
                "expected_performance": base_performance * 0.8,
                "probability": 0.15,
                "description": "Worst case scenario with adverse conditions"
            }
            
            # Calculate expected value across scenarios
            expected_value = sum(
                scenario["expected_performance"] * scenario["probability"]
                for scenario in scenarios.values()
            )
            
            return {
                "scenarios": scenarios,
                "expected_value": expected_value,
                "confidence_interval": {
                    "lower": scenarios["pessimistic"]["expected_performance"],
                    "upper": scenarios["optimistic"]["expected_performance"]
                },
                "recommendation": "Plan for realistic scenario but prepare for pessimistic case"
            }
            
        except Exception as e:
            logger.warning("Failed to generate scenario analysis", error=str(e))
            return {"error": str(e)}
    
    def _calculate_forecast_confidence(self, forecast: Dict[str, Any]) -> float:
        """Calculate overall confidence for forecast."""
        try:
            # Factors that affect confidence
            data_quality = forecast.get("data_quality", 0.8)
            model_accuracy = forecast.get("model_accuracy", 0.7)
            time_horizon_factor = max(0.5, 1.0 - (forecast.get("horizon_hours", 24) / 168))  # Reduce with longer horizon
            
            confidence = (data_quality * 0.4 + model_accuracy * 0.4 + time_horizon_factor * 0.2)
            return round(confidence, 3)
            
        except Exception:
            return 0.7  # Default reasonable confidence
    
    async def _get_historical_prediction_accuracy(self, agent_id: Optional[str]) -> float:
        """Get historical prediction accuracy for agent or system."""
        try:
            if self.analytics_engine and agent_id:
                accuracy_data = await self.analytics_engine.get_prediction_accuracy_history(agent_id)
                return accuracy_data.get("overall_accuracy", 0.75)
            else:
                # Return system-wide average accuracy
                return 0.75  # Default accuracy estimate
                
        except Exception as e:
            logger.warning("Failed to get historical accuracy", error=str(e))
            return 0.75
    
    async def _calculate_forecast_data_quality(self, agent_id: Optional[str]) -> float:
        """Calculate data quality score for forecast."""
        try:
            if self.analytics_engine and agent_id:
                data_quality = await self.analytics_engine.assess_data_quality(agent_id)
                return data_quality.get("overall_score", 0.8)
            else:
                # Return system-wide data quality estimate
                return 0.8
                
        except Exception as e:
            logger.warning("Failed to calculate data quality", error=str(e))
            return 0.8


# Global service instance
scheduling_service = IntelligentSchedulingService()


# ===============================================================================
# INTELLIGENT SCHEDULING ENDPOINTS
# ===============================================================================

@router.on_event("startup")
async def startup_scheduling_service():
    """Initialize intelligent scheduling service."""
    await scheduling_service.initialize()


@router.post("/patterns/analyze", response_model=Dict[str, Any])
@audit_operation("analyze_patterns", "scheduling")
@with_circuit_breaker(scheduling_circuit_breaker)
async def analyze_patterns(
    request: PatternAnalysisRequest,
    user: Dict[str, Any] = Depends(require_schedule_optimize),
    _: None = Depends(check_rate_limit_dependency)
) -> Dict[str, Any]:
    """
    Analyze activity patterns with machine learning insights.
    
    Features:
    - Historical pattern recognition with ML algorithms
    - Predictive behavior modeling and forecasting
    - Anomaly detection and alerting capabilities
    - Optimization opportunity identification
    - Performance target: <2s analysis time
    
    Args:
        request: Pattern analysis parameters
        user: Authenticated user information
        
    Returns:
        Dict containing comprehensive pattern analysis results
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Starting pattern analysis",
                   agent_id=request.agent_id,
                   analysis_period_days=request.analysis_period_days,
                   user_id=user.get("user_id"))
        
        # Perform comprehensive pattern analysis
        analysis_result = await scheduling_service.analyze_activity_patterns_comprehensive(
            request=request,
            user_id=user.get("user_id")
        )
        
        # Performance monitoring
        analysis_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if analysis_time_ms > 2000:
            logger.warning("Pattern analysis exceeded performance target",
                          analysis_time_ms=analysis_time_ms,
                          target_ms=2000)
        
        # Add performance metadata
        analysis_result["performance_metadata"] = {
            "analysis_time_ms": analysis_time_ms,
            "data_points_analyzed": analysis_result.get("patterns", {}).get("metadata", {}).get("data_points", 0),
            "patterns_detected": len(analysis_result.get("patterns", {}))
        }
        
        logger.info("Pattern analysis completed successfully",
                   agent_id=request.agent_id,
                   analysis_time_ms=analysis_time_ms,
                   patterns_found=len(analysis_result.get("patterns", {})))
        
        return analysis_result
        
    except Exception as e:
        logger.error("Pattern analysis failed",
                    agent_id=request.agent_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pattern analysis failed: {str(e)}"
        )


@router.post("/optimize", response_model=ScheduleResponse)
@audit_operation("optimize_schedule", "scheduling")
@with_circuit_breaker(scheduling_circuit_breaker)
async def optimize_schedule(
    request: ScheduleRequest,
    user: Dict[str, Any] = Depends(require_schedule_optimize),
    _: None = Depends(check_rate_limit_dependency)
) -> ScheduleResponse:
    """
    Generate optimal schedule using machine learning optimization.
    
    Features:
    - Multi-objective optimization algorithms
    - Constraint satisfaction and resource allocation
    - Performance prediction and validation
    - Epic 1 ConsolidatedProductionOrchestrator integration
    - Performance target: <2s optimization time
    
    Args:
        request: Schedule optimization parameters
        user: Authenticated user information
        
    Returns:
        ScheduleResponse with optimal schedule and metrics
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Starting schedule optimization",
                   agent_id=request.agent_id,
                   optimization_goal=request.optimization_goal,
                   time_horizon_hours=request.time_horizon_hours,
                   user_id=user.get("user_id"))
        
        # Generate optimal schedule
        optimization_result = await scheduling_service.generate_optimal_schedule_advanced(
            request=request,
            user_id=user.get("user_id")
        )
        
        # Build response
        response = ScheduleResponse(
            schedule_id=optimization_result["schedule_id"],
            schedule_name=optimization_result["schedule_name"],
            optimization_goal=optimization_result["optimization_goal"],
            schedule=optimization_result["schedule"],
            task_assignments=optimization_result["task_assignments"],
            time_slots=optimization_result["time_slots"],
            validation_results=optimization_result["validation_results"],
            performance_predictions=optimization_result["performance_predictions"],
            confidence_score=optimization_result["confidence_score"],
            created_at=datetime.fromisoformat(optimization_result["created_at"]),
            valid_from=datetime.fromisoformat(optimization_result["valid_from"]),
            expires_at=datetime.fromisoformat(optimization_result["expires_at"]),
            efficiency_score=optimization_result["efficiency_score"],
            resource_utilization=optimization_result["resource_utilization"],
            optimization_insights=optimization_result["optimization_insights"]
        )
        
        # Performance monitoring
        optimization_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if optimization_time_ms > 2000:
            logger.warning("Schedule optimization exceeded performance target",
                          optimization_time_ms=optimization_time_ms,
                          target_ms=2000)
        
        logger.info("Schedule optimization completed successfully",
                   schedule_id=optimization_result["schedule_id"],
                   optimization_time_ms=optimization_time_ms,
                   confidence_score=response.confidence_score,
                   efficiency_score=response.efficiency_score)
        
        return response
        
    except Exception as e:
        logger.error("Schedule optimization failed",
                    agent_id=request.agent_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule optimization failed: {str(e)}"
        )


@router.post("/conflicts/resolve", response_model=Dict[str, Any])
@audit_operation("resolve_conflicts", "scheduling")
@with_circuit_breaker(scheduling_circuit_breaker)
async def resolve_conflicts(
    request: ConflictResolutionRequest,
    user: Dict[str, Any] = Depends(require_schedule_optimize),
    _: None = Depends(check_rate_limit_dependency)
) -> Dict[str, Any]:
    """
    Intelligent resolution of scheduling conflicts.
    
    Features:
    - Multi-criteria conflict resolution algorithms
    - Optimal rescheduling with minimal impact
    - Stakeholder impact minimization
    - Performance preservation guarantees
    - Performance target: <5s resolution time
    
    Args:
        request: Conflict resolution parameters
        user: Authenticated user information
        
    Returns:
        Dict containing conflict resolution results and metrics
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Starting conflict resolution",
                   strategy=request.conflict_resolution_strategy,
                   allow_rescheduling=request.allow_rescheduling,
                   user_id=user.get("user_id"))
        
        # Resolve conflicts through intelligent service
        resolution_result = await scheduling_service.resolve_schedule_conflicts_intelligent(
            request=request,
            user_id=user.get("user_id")
        )
        
        # Performance monitoring
        resolution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if resolution_time_ms > 5000:
            logger.warning("Conflict resolution exceeded performance target",
                          resolution_time_ms=resolution_time_ms,
                          target_ms=5000)
        
        # Add performance metadata
        resolution_result["performance_metadata"] = {
            "resolution_time_ms": resolution_time_ms,
            "strategy_used": request.conflict_resolution_strategy,
            "conflicts_processed": resolution_result.get("total_conflicts", 0)
        }
        
        logger.info("Conflict resolution completed",
                   total_conflicts=resolution_result.get("total_conflicts", 0),
                   resolved_conflicts=resolution_result.get("conflicts_resolved", 0),
                   resolution_time_ms=resolution_time_ms)
        
        return resolution_result
        
    except Exception as e:
        logger.error("Conflict resolution failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conflict resolution failed: {str(e)}"
        )


@router.get("/forecast", response_model=Dict[str, Any])
@with_circuit_breaker(scheduling_circuit_breaker)
async def get_predictive_forecast(
    agent_id: Optional[str] = Query(None, description="Agent ID for forecast"),
    forecast_hours: int = Query(24, ge=1, le=168, description="Forecast time horizon"),
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    user: Dict[str, Any] = Depends(require_task_read)
) -> Dict[str, Any]:
    """
    Get predictive forecast for agent behavior and optimal scheduling.
    
    Features:
    - Machine learning-based predictions with confidence intervals
    - Scenario analysis and what-if modeling capabilities
    - Risk assessment and mitigation suggestions
    - Performance optimization opportunities identification
    - Performance target: <1s forecast generation
    
    Args:
        agent_id: Agent ID for forecast (None for system-wide)
        forecast_hours: Forecast time horizon in hours
        include_confidence: Include confidence intervals in results
        user: Authenticated user information
        
    Returns:
        Dict containing comprehensive predictive forecast
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Generating predictive forecast",
                   agent_id=agent_id,
                   forecast_hours=forecast_hours,
                   user_id=user.get("user_id"))
        
        # Generate comprehensive forecast
        forecast_result = await scheduling_service.get_predictive_forecast_comprehensive(
            agent_id=agent_id,
            forecast_hours=forecast_hours,
            include_confidence=include_confidence,
            user_id=user.get("user_id")
        )
        
        # Performance monitoring
        forecast_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if forecast_time_ms > 1000:
            logger.warning("Forecast generation exceeded performance target",
                          forecast_time_ms=forecast_time_ms,
                          target_ms=1000)
        
        # Add performance metadata
        forecast_result["performance_metadata"] = {
            "generation_time_ms": forecast_time_ms,
            "data_points_used": forecast_result.get("forecast", {}).get("data_points", 0),
            "scenarios_analyzed": len(forecast_result.get("scenario_analysis", {}).get("scenarios", {}))
        }
        
        logger.info("Predictive forecast generated successfully",
                   agent_id=agent_id,
                   forecast_hours=forecast_hours,
                   generation_time_ms=forecast_time_ms,
                   confidence=forecast_result.get("confidence_metrics", {}).get("overall_confidence", 0))
        
        return forecast_result
        
    except Exception as e:
        logger.error("Predictive forecast failed",
                    agent_id=agent_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Predictive forecast failed: {str(e)}"
        )