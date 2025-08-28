#!/usr/bin/env python3
"""
EPIC D PHASE 1: Production Deployment Validation - CI/CD Workflow Analysis
Comprehensive analysis of existing 23+ GitHub Actions workflows for optimization opportunities.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class WorkflowMetrics:
    """Metrics for a single GitHub Actions workflow"""
    name: str
    file_path: str
    job_count: int
    step_count: int
    dependencies: List[str]
    parallel_potential: int
    optimization_score: float
    estimated_runtime: int  # minutes
    cache_utilization: bool
    security_scanning: bool
    test_coverage: bool

class CICDWorkflowAnalyzer:
    """Analyzes existing CI/CD workflows for performance optimization"""
    
    def __init__(self):
        self.workflows_dir = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/.github/workflows")
        self.analysis_results = {}
        
    def analyze_all_workflows(self) -> Dict[str, Any]:
        """Analyze all GitHub Actions workflows"""
        print("🔍 Starting comprehensive CI/CD workflow analysis...")
        
        workflows = list(self.workflows_dir.glob("*.yml"))
        total_workflows = len(workflows)
        
        print(f"📊 Found {total_workflows} workflows to analyze")
        
        analysis_summary = {
            "total_workflows": total_workflows,
            "workflows": {},
            "optimization_opportunities": [],
            "performance_metrics": {
                "total_estimated_runtime": 0,
                "parallelizable_jobs": 0,
                "cache_optimizations": 0,
                "dependency_optimizations": 0
            },
            "recommendations": []
        }
        
        for workflow_file in workflows:
            try:
                metrics = self._analyze_workflow(workflow_file)
                analysis_summary["workflows"][metrics.name] = metrics
                analysis_summary["performance_metrics"]["total_estimated_runtime"] += metrics.estimated_runtime
                
                if metrics.parallel_potential > 1:
                    analysis_summary["performance_metrics"]["parallelizable_jobs"] += metrics.parallel_potential
                
                if not metrics.cache_utilization:
                    analysis_summary["performance_metrics"]["cache_optimizations"] += 1
                    
            except Exception as e:
                print(f"❌ Error analyzing {workflow_file.name}: {e}")
                
        self._generate_optimization_recommendations(analysis_summary)
        return analysis_summary
    
    def _analyze_workflow(self, workflow_file: Path) -> WorkflowMetrics:
        """Analyze a single workflow file"""
        print(f"  📋 Analyzing {workflow_file.name}...")
        
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        jobs = workflow_data.get('jobs', {})
        total_steps = 0
        dependencies = []
        cache_found = False
        security_scanning = False
        test_coverage = False
        
        for job_name, job_config in jobs.items():
            steps = job_config.get('steps', [])
            total_steps += len(steps)
            
            # Check for dependencies
            if 'needs' in job_config:
                needs = job_config['needs']
                if isinstance(needs, str):
                    dependencies.append(needs)
                elif isinstance(needs, list):
                    dependencies.extend(needs)
            
            # Check for cache usage
            for step in steps:
                if 'uses' in step and 'actions/cache' in step['uses']:
                    cache_found = True
                
                # Check for security scanning
                if any(keyword in str(step).lower() for keyword in ['security', 'bandit', 'safety', 'audit']):
                    security_scanning = True
                
                # Check for test coverage
                if any(keyword in str(step).lower() for keyword in ['pytest', 'coverage', 'test']):
                    test_coverage = True
        
        # Calculate parallel potential (jobs that could run in parallel)
        parallel_potential = len(jobs) - len(set(dependencies))
        
        # Calculate optimization score (0-100)
        optimization_score = self._calculate_optimization_score(
            job_count=len(jobs),
            cache_utilization=cache_found,
            security_scanning=security_scanning,
            test_coverage=test_coverage,
            parallel_potential=parallel_potential
        )
        
        # Estimate runtime based on complexity
        estimated_runtime = self._estimate_runtime(len(jobs), total_steps, cache_found)
        
        return WorkflowMetrics(
            name=workflow_data.get('name', workflow_file.stem),
            file_path=str(workflow_file),
            job_count=len(jobs),
            step_count=total_steps,
            dependencies=dependencies,
            parallel_potential=parallel_potential,
            optimization_score=optimization_score,
            estimated_runtime=estimated_runtime,
            cache_utilization=cache_found,
            security_scanning=security_scanning,
            test_coverage=test_coverage
        )
    
    def _calculate_optimization_score(self, job_count: int, cache_utilization: bool, 
                                    security_scanning: bool, test_coverage: bool,
                                    parallel_potential: int) -> float:
        """Calculate optimization score (0-100) for a workflow"""
        score = 0.0
        
        # Base score for having multiple jobs (parallelization potential)
        if job_count > 1:
            score += 20
        
        # Cache utilization bonus
        if cache_utilization:
            score += 25
        
        # Security scanning bonus
        if security_scanning:
            score += 20
        
        # Test coverage bonus
        if test_coverage:
            score += 20
        
        # Parallelization potential bonus
        if parallel_potential > 0:
            score += min(15, parallel_potential * 5)
        
        return min(100.0, score)
    
    def _estimate_runtime(self, job_count: int, step_count: int, has_cache: bool) -> int:
        """Estimate workflow runtime in minutes"""
        base_time = step_count * 0.5  # 30 seconds per step base
        
        # Parallel jobs reduce total time
        if job_count > 1:
            base_time = base_time / min(job_count, 3)  # Max 3 concurrent jobs assumption
        
        # Cache reduces time
        if has_cache:
            base_time *= 0.7
        
        return max(1, int(base_time))
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]):
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Analyze workflows for common optimization opportunities
        for workflow_name, metrics in analysis["workflows"].items():
            if isinstance(metrics, WorkflowMetrics):
                # Cache optimization recommendation
                if not metrics.cache_utilization and metrics.step_count > 5:
                    recommendations.append({
                        "type": "cache_optimization",
                        "workflow": workflow_name,
                        "description": f"Add dependency caching to reduce build time (estimated 30% improvement)",
                        "priority": "high",
                        "estimated_improvement": "30%"
                    })
                
                # Parallelization recommendation
                if metrics.parallel_potential > 2:
                    recommendations.append({
                        "type": "parallelization",
                        "workflow": workflow_name,
                        "description": f"Optimize job dependencies to enable {metrics.parallel_potential} parallel jobs",
                        "priority": "medium",
                        "estimated_improvement": f"{min(50, metrics.parallel_potential * 10)}%"
                    })
                
                # Runtime optimization for long workflows
                if metrics.estimated_runtime > 10:
                    recommendations.append({
                        "type": "runtime_optimization",
                        "workflow": workflow_name,
                        "description": f"Optimize {metrics.estimated_runtime}-minute workflow with selective testing and build optimization",
                        "priority": "high" if metrics.estimated_runtime > 20 else "medium",
                        "estimated_improvement": "40%"
                    })
        
        # Global recommendations
        total_runtime = analysis["performance_metrics"]["total_estimated_runtime"]
        if total_runtime > 60:  # More than 1 hour total
            recommendations.append({
                "type": "global_optimization",
                "workflow": "all",
                "description": f"Implement global workflow optimization strategy for {total_runtime}-minute total runtime",
                "priority": "critical",
                "estimated_improvement": "50%"
            })
        
        analysis["recommendations"] = recommendations

def main():
    """Execute CI/CD workflow analysis"""
    analyzer = CICDWorkflowAnalyzer()
    
    start_time = time.time()
    analysis_results = analyzer.analyze_all_workflows()
    analysis_time = time.time() - start_time
    
    # Generate comprehensive report
    report = {
        "analysis_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_duration": f"{analysis_time:.2f} seconds",
            "epic": "D",
            "phase": "1",
            "focus": "CI/CD Workflow Performance Analysis"
        },
        "executive_summary": {
            "total_workflows": analysis_results["total_workflows"],
            "total_estimated_runtime": analysis_results["performance_metrics"]["total_estimated_runtime"],
            "optimization_opportunities": len(analysis_results["recommendations"]),
            "high_priority_optimizations": len([r for r in analysis_results["recommendations"] if r.get("priority") == "high"]),
            "potential_time_savings": "45-60%" if analysis_results["performance_metrics"]["total_estimated_runtime"] > 30 else "25-35%"
        },
        "detailed_analysis": analysis_results
    }
    
    # Save analysis results
    report_file = f"/Users/bogdan/work/leanvibe-dev/bee-hive/epic_d_phase1_cicd_analysis_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n🎯 CI/CD Analysis Complete!")
    print(f"📊 Analyzed {report['executive_summary']['total_workflows']} workflows")
    print(f"⏱️  Total estimated runtime: {report['executive_summary']['total_estimated_runtime']} minutes")
    print(f"🚀 Optimization opportunities: {report['executive_summary']['optimization_opportunities']}")
    print(f"⚡ Potential time savings: {report['executive_summary']['potential_time_savings']}")
    print(f"📁 Report saved: {report_file}")
    
    # Print top 3 recommendations
    if analysis_results["recommendations"]:
        print(f"\n🔧 Top Optimization Recommendations:")
        for i, rec in enumerate(analysis_results["recommendations"][:3], 1):
            print(f"  {i}. {rec['type'].replace('_', ' ').title()}: {rec['description']}")
    
    return report_file

if __name__ == "__main__":
    main()