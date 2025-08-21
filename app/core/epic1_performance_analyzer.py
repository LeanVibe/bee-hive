#!/usr/bin/env python3
"""
Epic 1 Performance Analyzer - Deep System Performance Analysis

Specialized performance analysis tool for Epic 1: Performance Excellence & Optimization.
Provides detailed component-level performance analysis, bottleneck identification,
and optimization opportunity assessment.
"""

import asyncio
import time
import tracemalloc
import gc
import psutil
import sys
import importlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

import structlog
import httpx

logger = structlog.get_logger(__name__)


@dataclass
class ComponentPerformanceProfile:
    """Performance profile for a system component."""
    name: str
    memory_usage_mb: float
    import_time_ms: float
    initialization_time_ms: float
    cpu_usage_percent: float
    optimization_potential: str
    recommendations: List[str]


@dataclass
class APIEndpointProfile:
    """Performance profile for an API endpoint."""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    success: bool
    optimization_priority: str


class Epic1PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for Epic 1 optimization.
    
    Provides deep analysis of:
    - Component-level memory usage and performance
    - API endpoint response times and bottlenecks
    - Import and initialization performance
    - Optimization opportunity identification
    """
    
    def __init__(self):
        self.component_profiles: Dict[str, ComponentPerformanceProfile] = {}
        self.api_profiles: List[APIEndpointProfile] = []
        self.system_baseline: Dict[str, float] = {}
        
        # Start memory tracing
        tracemalloc.start()
        
        logger.info("Epic 1 Performance Analyzer initialized")
    
    async def analyze_component_performance(self, component_name: str, import_path: str) -> ComponentPerformanceProfile:
        """Analyze performance characteristics of a specific component."""
        logger.info(f"Analyzing component: {component_name}")
        
        # Measure memory before import
        gc.collect()  # Clean up before measurement
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Measure import time
        start_time = time.time()
        try:
            module = importlib.import_module(import_path)
            import_time = (time.time() - start_time) * 1000
            import_success = True
        except Exception as e:
            import_time = (time.time() - start_time) * 1000
            import_success = False
            logger.warning(f"Failed to import {component_name}: {e}")
        
        # Measure memory after import
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        # Measure initialization time if applicable
        initialization_time = 0.0
        if import_success and hasattr(module, '__init__'):
            try:
                start_time = time.time()
                # Try to initialize if it's a class
                if hasattr(module, '__name__') and any(
                    hasattr(getattr(module, attr), '__init__') 
                    for attr in dir(module) 
                    if not attr.startswith('_')
                ):
                    initialization_time = (time.time() - start_time) * 1000
            except:
                pass
        
        # CPU usage (current)
        cpu_usage = process.cpu_percent()
        
        # Determine optimization potential
        optimization_potential = self._assess_optimization_potential(
            memory_usage, import_time, initialization_time
        )
        
        # Generate recommendations
        recommendations = self._generate_component_recommendations(
            component_name, memory_usage, import_time, initialization_time
        )
        
        profile = ComponentPerformanceProfile(
            name=component_name,
            memory_usage_mb=memory_usage,
            import_time_ms=import_time,
            initialization_time_ms=initialization_time,
            cpu_usage_percent=cpu_usage,
            optimization_potential=optimization_potential,
            recommendations=recommendations
        )
        
        self.component_profiles[component_name] = profile
        return profile
    
    def _assess_optimization_potential(self, memory_mb: float, import_ms: float, init_ms: float) -> str:
        """Assess optimization potential based on performance metrics."""
        if memory_mb > 20 or import_ms > 200 or init_ms > 100:
            return "HIGH"
        elif memory_mb > 10 or import_ms > 100 or init_ms > 50:
            return "MEDIUM"
        elif memory_mb > 5 or import_ms > 50 or init_ms > 25:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_component_recommendations(self, name: str, memory_mb: float, import_ms: float, init_ms: float) -> List[str]:
        """Generate optimization recommendations for a component."""
        recommendations = []
        
        if memory_mb > 20:
            recommendations.append("Critical memory optimization needed - implement lazy loading")
        elif memory_mb > 10:
            recommendations.append("Memory optimization opportunity - review data structures")
        
        if import_ms > 200:
            recommendations.append("Import time critical - restructure imports or use lazy imports")
        elif import_ms > 100:
            recommendations.append("Import optimization opportunity - review dependencies")
        
        if init_ms > 100:
            recommendations.append("Initialization optimization needed - implement async initialization")
        elif init_ms > 50:
            recommendations.append("Initialization could be optimized - review startup logic")
        
        if not recommendations:
            recommendations.append("Well optimized - monitor for regression")
        
        return recommendations
    
    async def analyze_api_endpoints(self, base_url: str = "http://localhost:8000") -> List[APIEndpointProfile]:
        """Analyze API endpoint performance."""
        logger.info("Analyzing API endpoint performance")
        
        # Define test endpoints
        test_endpoints = [
            ("GET", "/health"),
            ("GET", "/metrics"),
            ("GET", "/api/v2/agents"),
            ("GET", "/api/v2/tasks"),
            ("GET", "/api/v2/sessions"),
            ("POST", "/api/v2/agents"),  # Will likely fail but measure time
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ]
        
        profiles = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for method, endpoint in test_endpoints:
                try:
                    start_time = time.time()
                    
                    if method == "GET":
                        response = await client.get(f"{base_url}{endpoint}")
                    elif method == "POST":
                        response = await client.post(f"{base_url}{endpoint}", json={})
                    else:
                        continue
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    # Determine optimization priority
                    if response_time > 100:
                        priority = "CRITICAL"
                    elif response_time > 50:
                        priority = "HIGH"
                    elif response_time > 25:
                        priority = "MEDIUM"
                    else:
                        priority = "LOW"
                    
                    profile = APIEndpointProfile(
                        endpoint=endpoint,
                        method=method,
                        response_time_ms=response_time,
                        status_code=response.status_code,
                        success=200 <= response.status_code < 400,
                        optimization_priority=priority
                    )
                    
                    profiles.append(profile)
                    
                except Exception as e:
                    logger.warning(f"Failed to test {method} {endpoint}: {e}")
                    # Create profile for failed endpoint
                    profile = APIEndpointProfile(
                        endpoint=endpoint,
                        method=method,
                        response_time_ms=999.0,  # High time to indicate failure
                        status_code=0,
                        success=False,
                        optimization_priority="ERROR"
                    )
                    profiles.append(profile)
        
        self.api_profiles = profiles
        return profiles
    
    async def analyze_memory_usage_patterns(self) -> Dict[str, Any]:
        """Analyze detailed memory usage patterns."""
        logger.info("Analyzing memory usage patterns")
        
        # Get current memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        
        # Get system memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get memory breakdown by component
        gc.collect()  # Force garbage collection
        
        analysis = {
            "total_memory_mb": memory_info.rss / 1024 / 1024,
            "peak_memory_mb": peak / 1024 / 1024,
            "current_traced_mb": current / 1024 / 1024,
            "virtual_memory_mb": memory_info.vms / 1024 / 1024,
            "memory_efficiency": (current / peak) * 100 if peak > 0 else 100,
            "gc_objects": len(gc.get_objects()),
            "gc_collections": gc.get_stats() if hasattr(gc, 'get_stats') else None,
            "optimization_recommendations": []
        }
        
        # Generate memory optimization recommendations
        if analysis["total_memory_mb"] > 100:
            analysis["optimization_recommendations"].extend([
                "Critical: Memory usage exceeds 100MB - implement aggressive optimization",
                "Review component loading patterns for lazy initialization",
                "Implement memory pooling for frequently allocated objects"
            ])
        elif analysis["total_memory_mb"] > 80:
            analysis["optimization_recommendations"].extend([
                "Memory usage above Epic 1 target - optimization needed",
                "Profile component memory usage for reduction opportunities"
            ])
        
        if analysis["memory_efficiency"] < 70:
            analysis["optimization_recommendations"].append(
                "Low memory efficiency - review object lifecycle management"
            )
        
        return analysis
    
    async def identify_optimization_opportunities(self) -> Dict[str, List[str]]:
        """Identify top optimization opportunities across all components."""
        opportunities = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Component-based opportunities
        for name, profile in self.component_profiles.items():
            if profile.optimization_potential == "HIGH":
                opportunities["critical"].extend([
                    f"{name}: {rec}" for rec in profile.recommendations
                ])
            elif profile.optimization_potential == "MEDIUM":
                opportunities["high"].extend([
                    f"{name}: {rec}" for rec in profile.recommendations
                ])
            elif profile.optimization_potential == "LOW":
                opportunities["medium"].extend([
                    f"{name}: {rec}" for rec in profile.recommendations
                ])
        
        # API-based opportunities
        for profile in self.api_profiles:
            if profile.optimization_priority == "CRITICAL":
                opportunities["critical"].append(
                    f"API {profile.endpoint}: Response time {profile.response_time_ms:.2f}ms needs optimization"
                )
            elif profile.optimization_priority == "HIGH":
                opportunities["high"].append(
                    f"API {profile.endpoint}: Response time {profile.response_time_ms:.2f}ms can be improved"
                )
        
        return opportunities
    
    async def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate comprehensive optimization plan for Epic 1."""
        opportunities = await self.identify_optimization_opportunities()
        memory_analysis = await self.analyze_memory_usage_patterns()
        
        plan = {
            "epic1_targets": {
                "api_response_time": "<50ms for 95th percentile",
                "memory_usage": "<80MB consistent",
                "concurrent_agents": "200+ without degradation",
                "ml_monitoring": "80% prediction accuracy"
            },
            "current_status": {
                "total_memory_mb": memory_analysis["total_memory_mb"],
                "components_analyzed": len(self.component_profiles),
                "api_endpoints_tested": len(self.api_profiles),
                "critical_issues": len(opportunities["critical"]),
                "high_priority_issues": len(opportunities["high"])
            },
            "optimization_priorities": {
                "phase_1_critical": opportunities["critical"][:5],  # Top 5 critical
                "phase_2_high": opportunities["high"][:10],        # Top 10 high
                "phase_3_medium": opportunities["medium"][:5]      # Top 5 medium
            },
            "estimated_impact": self._estimate_optimization_impact(memory_analysis, opportunities),
            "implementation_strategy": [
                "1. Address critical memory optimization opportunities",
                "2. Optimize high-impact API endpoints",
                "3. Implement component-level optimizations",
                "4. Enable ML-based monitoring and prediction",
                "5. Validate all Epic 1 targets achieved"
            ]
        }
        
        return plan
    
    def _estimate_optimization_impact(self, memory_analysis: Dict, opportunities: Dict) -> Dict[str, str]:
        """Estimate the impact of optimization efforts."""
        impact = {}
        
        # Memory impact
        current_memory = memory_analysis["total_memory_mb"]
        if len(opportunities["critical"]) > 0:
            estimated_reduction = min(40, len(opportunities["critical"]) * 8)
            impact["memory_reduction"] = f"Up to {estimated_reduction}MB reduction possible"
        else:
            impact["memory_reduction"] = "10-20MB reduction through general optimization"
        
        # API performance impact
        critical_api_issues = [op for op in opportunities["critical"] if "API" in op]
        if critical_api_issues:
            impact["api_performance"] = "Significant response time improvements possible"
        else:
            impact["api_performance"] = "Fine-tuning for sub-50ms target"
        
        # Overall impact
        total_issues = len(opportunities["critical"]) + len(opportunities["high"])
        if total_issues > 10:
            impact["overall"] = "Major performance improvements expected"
        elif total_issues > 5:
            impact["overall"] = "Moderate performance improvements expected"
        else:
            impact["overall"] = "Incremental optimizations for Epic 1 targets"
        
        return impact

    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete Epic 1 performance analysis."""
        logger.info("Starting comprehensive Epic 1 performance analysis")
        
        # Analyze core components
        core_components = {
            "SimpleOrchestrator": "app.core.simple_orchestrator",
            "BaseManager": "app.core.unified_managers.base_manager",
            "LifecycleManager": "app.core.unified_managers.lifecycle_manager",
            "CommunicationManager": "app.core.unified_managers.communication_manager",
            "SecurityManager": "app.core.unified_managers.security_manager",
            "PerformanceManager": "app.core.unified_managers.performance_manager",
            "ConfigurationManager": "app.core.unified_managers.configuration_manager",
            "PerformanceFramework": "app.core.performance_optimization_framework"
        }
        
        component_results = {}
        for name, path in core_components.items():
            try:
                profile = await self.analyze_component_performance(name, path)
                component_results[name] = profile
            except Exception as e:
                logger.warning(f"Failed to analyze {name}: {e}")
        
        # Analyze API performance
        api_results = await self.analyze_api_endpoints()
        
        # Generate optimization plan
        optimization_plan = await self.generate_optimization_plan()
        
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "component_analysis": component_results,
            "api_analysis": api_results,
            "optimization_plan": optimization_plan,
            "epic1_readiness": self._assess_epic1_readiness(optimization_plan)
        }
    
    def _assess_epic1_readiness(self, plan: Dict) -> Dict[str, Any]:
        """Assess readiness for Epic 1 optimization phases."""
        current_memory = plan["current_status"]["total_memory_mb"]
        critical_issues = plan["current_status"]["critical_issues"]
        
        readiness = {
            "memory_optimization_ready": current_memory < 150,  # Not too high to start
            "api_optimization_ready": critical_issues < 10,     # Manageable number of critical issues
            "overall_readiness": "READY" if critical_issues < 5 and current_memory < 130 else "NEEDS_PREP",
            "recommended_start_phase": "Phase 1.2 - API Optimization" if critical_issues < 3 else "Phase 1.1 - Critical Issues"
        }
        
        return readiness


# Global analyzer instance
_epic1_analyzer: Optional[Epic1PerformanceAnalyzer] = None


def get_epic1_analyzer() -> Epic1PerformanceAnalyzer:
    """Get or create the Epic 1 performance analyzer."""
    global _epic1_analyzer
    
    if _epic1_analyzer is None:
        _epic1_analyzer = Epic1PerformanceAnalyzer()
    
    return _epic1_analyzer


async def run_epic1_analysis() -> Dict[str, Any]:
    """Run comprehensive Epic 1 performance analysis."""
    analyzer = get_epic1_analyzer()
    return await analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    # Example usage
    async def main():
        analyzer = get_epic1_analyzer()
        results = await analyzer.run_comprehensive_analysis()
        
        print("🔍 EPIC 1 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Print key findings
        plan = results["optimization_plan"]
        print(f"\\nCurrent Memory Usage: {plan['current_status']['total_memory_mb']:.2f}MB")
        print(f"Components Analyzed: {plan['current_status']['components_analyzed']}")
        print(f"API Endpoints Tested: {plan['current_status']['api_endpoints_tested']}")
        print(f"Critical Issues: {plan['current_status']['critical_issues']}")
        
        # Print top optimization opportunities
        print("\\n🎯 TOP OPTIMIZATION OPPORTUNITIES:")
        for priority, opportunities in plan["optimization_priorities"].items():
            if opportunities:
                print(f"\\n{priority.upper()}:")
                for opp in opportunities[:3]:  # Top 3 per category
                    print(f"  • {opp}")
        
        # Print Epic 1 readiness
        readiness = results["epic1_readiness"]
        print(f"\\n🚀 Epic 1 Readiness: {readiness['overall_readiness']}")
        print(f"Recommended Start: {readiness['recommended_start_phase']}")
    
    asyncio.run(main())