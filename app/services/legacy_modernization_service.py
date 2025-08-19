"""
Legacy System Modernization Service
Production-grade service for automated legacy system analysis and incremental refactoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import subprocess
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import redis.asyncio as redis

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.models.workflow import WorkflowExecution
from app.schemas.workflow import WorkflowStatus, WorkflowType


class ModernizationPhase(Enum):
    """Phases of legacy system modernization."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    VALIDATION = "validation"


class TechnologyStack(Enum):
    """Supported technology stacks for modernization."""
    JAVA_SPRING = "java_spring"
    DOTNET_CORE = "dotnet_core"
    PYTHON_DJANGO = "python_django"
    NODEJS_EXPRESS = "nodejs_express"
    REACT_FRONTEND = "react_frontend"
    ANGULAR_FRONTEND = "angular_frontend"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"


@dataclass
class LegacySystemProfile:
    """Complete profile of a legacy system."""
    system_id: str
    name: str
    technology_stack: List[str]
    lines_of_code: int
    dependencies: Dict[str, str]
    database_schema: Dict[str, Any]
    api_endpoints: List[Dict[str, Any]]
    business_logic_modules: List[str]
    technical_debt_score: float
    security_vulnerabilities: List[Dict[str, Any]]
    performance_bottlenecks: List[Dict[str, Any]]
    compliance_requirements: List[str] = field(default_factory=list)
    estimated_modernization_cost: Optional[float] = None
    estimated_timeline_months: Optional[int] = None


@dataclass
class ModernizationStrategy:
    """Comprehensive modernization strategy."""
    strategy_id: str
    system_profile: LegacySystemProfile
    target_architecture: Dict[str, Any]
    migration_phases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    rollback_strategy: Dict[str, Any]
    success_criteria: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, datetime]


@dataclass
class ModernizationProgress:
    """Real-time modernization progress tracking."""
    execution_id: str
    current_phase: ModernizationPhase
    phase_progress: float  # 0.0 to 1.0
    overall_progress: float  # 0.0 to 1.0
    tasks_completed: int
    tasks_total: int
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    risk_indicators: List[str]
    estimated_completion: datetime
    last_updated: datetime


class LegacyAnalysisAgent:
    """Specialized agent for legacy system analysis."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        
    async def analyze_codebase(self, codebase_path: str) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        self.logger.info(f"Starting codebase analysis for: {codebase_path}")
        
        analysis_results = {
            "technology_detection": await self._detect_technologies(codebase_path),
            "dependency_analysis": await self._analyze_dependencies(codebase_path),
            "code_metrics": await self._calculate_code_metrics(codebase_path),
            "security_scan": await self._security_vulnerability_scan(codebase_path),
            "performance_analysis": await self._performance_bottleneck_analysis(codebase_path),
            "business_logic_extraction": await self._extract_business_logic(codebase_path),
            "api_discovery": await self._discover_api_endpoints(codebase_path)
        }
        
        # Cache analysis results
        await self.redis.setex(
            f"legacy_analysis:{codebase_path.replace('/', '_')}",
            3600,  # 1 hour TTL
            json.dumps(analysis_results, default=str)
        )
        
        return analysis_results
    
    async def _detect_technologies(self, codebase_path: str) -> Dict[str, Any]:
        """Detect technologies used in the codebase."""
        tech_indicators = {
            "java": [".java", "pom.xml", "build.gradle"],
            "python": [".py", "requirements.txt", "setup.py", "pyproject.toml"],
            "javascript": [".js", "package.json", "node_modules/"],
            "csharp": [".cs", ".csproj", ".sln"],
            "php": [".php", "composer.json"],
            "ruby": [".rb", "Gemfile"],
            "go": [".go", "go.mod"]
        }
        
        detected_technologies = {}
        
        for tech, indicators in tech_indicators.items():
            for indicator in indicators:
                if await self._find_files_with_pattern(codebase_path, indicator):
                    detected_technologies[tech] = await self._get_tech_details(codebase_path, tech)
                    break
        
        return detected_technologies
    
    async def _analyze_dependencies(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze project dependencies and their versions."""
        dependency_files = {
            "package.json": "npm",
            "requirements.txt": "pip",
            "pom.xml": "maven",
            "build.gradle": "gradle",
            "Gemfile": "gem",
            "composer.json": "composer",
            "go.mod": "go mod"
        }
        
        dependencies = {}
        
        for dep_file, manager in dependency_files.items():
            file_path = Path(codebase_path) / dep_file
            if file_path.exists():
                dependencies[manager] = await self._parse_dependency_file(file_path, manager)
        
        # Analyze for outdated dependencies
        for manager, deps in dependencies.items():
            dependencies[manager]["outdated"] = await self._check_outdated_dependencies(deps, manager)
            dependencies[manager]["security_alerts"] = await self._check_security_vulnerabilities(deps, manager)
        
        return dependencies
    
    async def _calculate_code_metrics(self, codebase_path: str) -> Dict[str, Any]:
        """Calculate comprehensive code quality metrics."""
        metrics = {
            "lines_of_code": 0,
            "complexity_score": 0.0,
            "duplication_percentage": 0.0,
            "maintainability_index": 0.0,
            "test_coverage": 0.0,
            "file_count": 0,
            "function_count": 0,
            "class_count": 0
        }
        
        # Use multiple tools for comprehensive analysis
        try:
            # Line counting
            result = await self._run_command(f"find {codebase_path} -name '*.py' -o -name '*.js' -o -name '*.java' | xargs wc -l")
            if result.returncode == 0:
                metrics["lines_of_code"] = self._parse_line_count(result.stdout)
            
            # Complexity analysis using radon (for Python) or similar tools
            if await self._find_files_with_pattern(codebase_path, "*.py"):
                complexity_result = await self._run_command(f"radon cc {codebase_path} -a")
                if complexity_result.returncode == 0:
                    metrics["complexity_score"] = self._parse_complexity_score(complexity_result.stdout)
            
            # Code duplication analysis
            duplication_result = await self._run_command(f"jscpd {codebase_path} --format json")
            if duplication_result.returncode == 0:
                metrics["duplication_percentage"] = self._parse_duplication_percentage(duplication_result.stdout)
                
        except Exception as e:
            self.logger.warning(f"Code metrics calculation failed: {e}")
        
        return metrics
    
    async def _security_vulnerability_scan(self, codebase_path: str) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Use multiple security scanning tools
            scanners = [
                ("bandit", f"bandit -r {codebase_path} -f json"),  # Python
                ("npm_audit", f"cd {codebase_path} && npm audit --json"),  # JavaScript
                ("safety", f"safety check --json"),  # Python dependencies
                ("snyk", f"cd {codebase_path} && snyk test --json")  # Multi-language
            ]
            
            for scanner_name, command in scanners:
                try:
                    result = await self._run_command(command)
                    if result.returncode == 0:
                        scanner_results = self._parse_security_results(result.stdout, scanner_name)
                        vulnerabilities.extend(scanner_results)
                except Exception as e:
                    self.logger.debug(f"Scanner {scanner_name} failed: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Security vulnerability scan failed: {e}")
        
        return vulnerabilities
    
    async def _performance_bottleneck_analysis(self, codebase_path: str) -> List[Dict[str, Any]]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        # Pattern-based analysis for common performance issues
        performance_patterns = {
            "n_plus_one_queries": [r"for.*in.*:\s*.*\.query\(", r"for.*in.*:\s*.*\.get\("],
            "large_loops": [r"for.*in.*range\(\d{4,}\)", r"while.*\d{4,}"],
            "inefficient_string_concat": [r"\+=.*str", r"\".*\"\s*\+\s*\".*\""],
            "blocking_io": [r"time\.sleep\(", r"requests\.get\(", r"open\("],
            "memory_leaks": [r"global\s+\w+\s*=\s*\[\]", r"cache\s*=\s*\{\}"]
        }
        
        for issue_type, patterns in performance_patterns.items():
            for pattern in patterns:
                matches = await self._find_pattern_in_codebase(codebase_path, pattern)
                for match in matches:
                    bottlenecks.append({
                        "type": issue_type,
                        "file": match["file"],
                        "line": match["line"],
                        "code": match["code"],
                        "severity": self._assess_bottleneck_severity(issue_type)
                    })
        
        return bottlenecks


class ModernizationStrategyAgent:
    """Agent responsible for creating modernization strategies."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def create_modernization_strategy(
        self, 
        system_profile: LegacySystemProfile,
        target_requirements: Dict[str, Any]
    ) -> ModernizationStrategy:
        """Create comprehensive modernization strategy."""
        
        self.logger.info(f"Creating modernization strategy for system: {system_profile.name}")
        
        # Analyze current state and target state
        gap_analysis = await self._perform_gap_analysis(system_profile, target_requirements)
        
        # Design target architecture
        target_architecture = await self._design_target_architecture(system_profile, target_requirements)
        
        # Create migration phases
        migration_phases = await self._plan_migration_phases(system_profile, target_architecture)
        
        # Assess risks and create mitigation strategies
        risk_assessment = await self._assess_migration_risks(system_profile, migration_phases)
        
        # Create rollback strategy
        rollback_strategy = await self._create_rollback_strategy(system_profile, migration_phases)
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(system_profile, target_requirements)
        
        strategy = ModernizationStrategy(
            strategy_id=f"modernization_{system_profile.system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            system_profile=system_profile,
            target_architecture=target_architecture,
            migration_phases=migration_phases,
            risk_assessment=risk_assessment,
            rollback_strategy=rollback_strategy,
            success_criteria=success_criteria,
            resource_requirements=await self._calculate_resource_requirements(migration_phases),
            timeline=await self._create_timeline(migration_phases)
        )
        
        # Cache strategy
        await self.redis.setex(
            f"modernization_strategy:{strategy.strategy_id}",
            86400,  # 24 hours TTL
            json.dumps(strategy.__dict__, default=str)
        )
        
        return strategy
    
    async def _perform_gap_analysis(
        self, 
        system_profile: LegacySystemProfile, 
        target_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze gaps between current and target state."""
        
        gaps = {
            "technology_gaps": [],
            "architecture_gaps": [],
            "performance_gaps": [],
            "security_gaps": [],
            "compliance_gaps": [],
            "scalability_gaps": []
        }
        
        # Technology gap analysis
        current_tech = set(system_profile.technology_stack)
        target_tech = set(target_requirements.get("technology_stack", []))
        
        gaps["technology_gaps"] = {
            "technologies_to_add": list(target_tech - current_tech),
            "technologies_to_remove": list(current_tech - target_tech),
            "technologies_to_upgrade": await self._identify_upgrade_candidates(system_profile.technology_stack)
        }
        
        # Performance gap analysis
        current_performance = system_profile.performance_bottlenecks
        target_performance = target_requirements.get("performance_requirements", {})
        
        gaps["performance_gaps"] = {
            "response_time_improvement_needed": target_performance.get("max_response_time", 500) - self._get_current_response_time(current_performance),
            "throughput_improvement_needed": target_performance.get("min_throughput", 1000) - self._get_current_throughput(current_performance),
            "bottlenecks_to_address": len(current_performance)
        }
        
        return gaps
    
    async def _design_target_architecture(
        self, 
        system_profile: LegacySystemProfile, 
        target_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design the target system architecture."""
        
        architecture_patterns = {
            "microservices": target_requirements.get("microservices", False),
            "serverless": target_requirements.get("serverless", False),
            "event_driven": target_requirements.get("event_driven", False),
            "api_first": target_requirements.get("api_first", True),
            "cloud_native": target_requirements.get("cloud_native", True)
        }
        
        # Select appropriate architecture pattern
        if architecture_patterns["microservices"]:
            return await self._design_microservices_architecture(system_profile, target_requirements)
        elif architecture_patterns["serverless"]:
            return await self._design_serverless_architecture(system_profile, target_requirements)
        else:
            return await self._design_modernized_monolith_architecture(system_profile, target_requirements)
    
    async def _plan_migration_phases(
        self, 
        system_profile: LegacySystemProfile, 
        target_architecture: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan incremental migration phases."""
        
        phases = []
        
        # Phase 1: Foundation and Infrastructure
        phases.append({
            "phase": 1,
            "name": "Foundation Setup",
            "duration_weeks": 4,
            "objectives": [
                "Set up CI/CD pipeline",
                "Implement monitoring and logging",
                "Create development and staging environments",
                "Establish security baselines"
            ],
            "deliverables": [
                "CI/CD pipeline operational",
                "Monitoring dashboards configured",
                "Security scanning integrated",
                "Environment provisioning automated"
            ],
            "risk_level": "low",
            "parallel_execution": True
        })
        
        # Phase 2: Data Layer Modernization
        phases.append({
            "phase": 2,
            "name": "Data Layer Modernization",
            "duration_weeks": 6,
            "objectives": [
                "Migrate database to modern technology",
                "Implement data access layer",
                "Create data migration tools",
                "Establish data validation processes"
            ],
            "deliverables": [
                "New database schema implemented",
                "Data migration tools created",
                "Data validation framework",
                "Performance benchmarking completed"
            ],
            "risk_level": "medium",
            "parallel_execution": False
        })
        
        # Phase 3: Business Logic Migration
        business_modules = system_profile.business_logic_modules
        module_batches = self._create_module_batches(business_modules, batch_size=3)
        
        for i, batch in enumerate(module_batches):
            phases.append({
                "phase": i + 3,
                "name": f"Business Logic Migration - Batch {i + 1}",
                "duration_weeks": 8,
                "objectives": [
                    f"Migrate modules: {', '.join(batch)}",
                    "Implement modern design patterns",
                    "Add comprehensive testing",
                    "Ensure API compatibility"
                ],
                "deliverables": [
                    "Modernized business logic modules",
                    "Comprehensive test coverage",
                    "API documentation updated",
                    "Performance validation completed"
                ],
                "risk_level": "high",
                "parallel_execution": True,
                "modules": batch
            })
        
        # Final Phase: Integration and Optimization
        phases.append({
            "phase": len(phases) + 1,
            "name": "Integration and Optimization",
            "duration_weeks": 4,
            "objectives": [
                "Complete system integration",
                "Performance optimization",
                "Security hardening",
                "Production deployment"
            ],
            "deliverables": [
                "Fully integrated system",
                "Performance benchmarks met",
                "Security audit completed",
                "Production deployment successful"
            ],
            "risk_level": "medium",
            "parallel_execution": False
        })
        
        return phases


class ModernizationExecutionAgent:
    """Agent responsible for executing modernization tasks."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.active_executions: Dict[str, ModernizationProgress] = {}
    
    async def execute_modernization_phase(
        self, 
        strategy: ModernizationStrategy, 
        phase_number: int,
        execution_id: str
    ) -> ModernizationProgress:
        """Execute a specific modernization phase."""
        
        self.logger.info(f"Starting modernization phase {phase_number} for execution {execution_id}")
        
        phase = strategy.migration_phases[phase_number - 1]
        
        # Initialize progress tracking
        progress = ModernizationProgress(
            execution_id=execution_id,
            current_phase=ModernizationPhase(phase["name"].lower().replace(" ", "_")),
            phase_progress=0.0,
            overall_progress=0.0,
            tasks_completed=0,
            tasks_total=len(phase["objectives"]),
            quality_metrics={},
            performance_metrics={},
            risk_indicators=[],
            estimated_completion=datetime.now() + timedelta(weeks=phase["duration_weeks"]),
            last_updated=datetime.now()
        )
        
        self.active_executions[execution_id] = progress
        
        # Execute phase objectives
        for i, objective in enumerate(phase["objectives"]):
            try:
                await self._execute_objective(objective, strategy, phase, progress)
                progress.tasks_completed += 1
                progress.phase_progress = progress.tasks_completed / progress.tasks_total
                progress.last_updated = datetime.now()
                
                # Update cache
                await self._update_progress_cache(execution_id, progress)
                
            except Exception as e:
                self.logger.error(f"Failed to execute objective '{objective}': {e}")
                progress.risk_indicators.append(f"Objective failure: {objective}")
        
        # Validate phase completion
        if await self._validate_phase_completion(phase, strategy):
            progress.phase_progress = 1.0
            self.logger.info(f"Phase {phase_number} completed successfully")
        else:
            self.logger.warning(f"Phase {phase_number} validation failed")
            progress.risk_indicators.append("Phase validation failed")
        
        return progress


class LegacyModernizationService:
    """Main service orchestrating legacy system modernization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.analysis_agent: Optional[LegacyAnalysisAgent] = None
        self.strategy_agent: Optional[ModernizationStrategyAgent] = None
        self.execution_agent: Optional[ModernizationExecutionAgent] = None
    
    async def initialize(self):
        """Initialize the service and its components."""
        self.redis_client = await get_redis_client()
        self.analysis_agent = LegacyAnalysisAgent(self.redis_client, self.logger)
        self.strategy_agent = ModernizationStrategyAgent(self.redis_client, self.logger)
        self.execution_agent = ModernizationExecutionAgent(self.redis_client, self.logger)
        
        self.logger.info("Legacy Modernization Service initialized successfully")
    
    async def start_modernization_project(
        self,
        project_config: Dict[str, Any],
        customer_id: str
    ) -> Dict[str, Any]:
        """Start a complete legacy modernization project."""
        
        project_id = f"legacy_mod_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Phase 1: System Analysis
            self.logger.info(f"Starting system analysis for project {project_id}")
            
            analysis_results = await self.analysis_agent.analyze_codebase(
                project_config["codebase_path"]
            )
            
            # Create system profile
            system_profile = LegacySystemProfile(
                system_id=project_id,
                name=project_config["system_name"],
                technology_stack=analysis_results["technology_detection"],
                lines_of_code=analysis_results["code_metrics"]["lines_of_code"],
                dependencies=analysis_results["dependency_analysis"],
                database_schema=project_config.get("database_schema", {}),
                api_endpoints=analysis_results["api_discovery"],
                business_logic_modules=analysis_results["business_logic_extraction"],
                technical_debt_score=self._calculate_technical_debt_score(analysis_results),
                security_vulnerabilities=analysis_results["security_scan"],
                performance_bottlenecks=analysis_results["performance_analysis"],
                compliance_requirements=project_config.get("compliance_requirements", [])
            )
            
            # Phase 2: Strategy Creation
            self.logger.info(f"Creating modernization strategy for project {project_id}")
            
            modernization_strategy = await self.strategy_agent.create_modernization_strategy(
                system_profile,
                project_config["target_requirements"]
            )
            
            # Phase 3: Execution Planning
            execution_plan = {
                "project_id": project_id,
                "customer_id": customer_id,
                "system_profile": system_profile.__dict__,
                "modernization_strategy": modernization_strategy.__dict__,
                "execution_timeline": modernization_strategy.timeline,
                "risk_mitigation_plan": modernization_strategy.risk_assessment,
                "success_criteria": modernization_strategy.success_criteria,
                "estimated_cost": self._calculate_project_cost(modernization_strategy),
                "estimated_roi": self._calculate_project_roi(system_profile, modernization_strategy),
                "next_steps": [
                    "Customer approval of modernization strategy",
                    "Resource allocation and team assignment",
                    "Environment setup and tooling configuration",
                    "Phase 1 execution initiation"
                ]
            }
            
            # Store project data
            await self.redis_client.setex(
                f"modernization_project:{project_id}",
                86400 * 30,  # 30 days TTL
                json.dumps(execution_plan, default=str)
            )
            
            return {
                "status": "success",
                "project_id": project_id,
                "analysis_completed": datetime.now().isoformat(),
                "strategy_created": datetime.now().isoformat(),
                "execution_plan": execution_plan,
                "next_phase": "customer_approval"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start modernization project: {e}")
            return {
                "status": "error",
                "project_id": project_id,
                "error_message": str(e),
                "recommended_action": "Review project configuration and retry"
            }
    
    async def execute_modernization_phase(
        self,
        project_id: str,
        phase_number: int
    ) -> Dict[str, Any]:
        """Execute a specific phase of the modernization project."""
        
        try:
            # Retrieve project data
            project_data = await self.redis_client.get(f"modernization_project:{project_id}")
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            project = json.loads(project_data)
            strategy = ModernizationStrategy(**project["modernization_strategy"])
            
            # Execute the phase
            execution_id = f"{project_id}_phase_{phase_number}"
            progress = await self.execution_agent.execute_modernization_phase(
                strategy, phase_number, execution_id
            )
            
            return {
                "status": "success",
                "project_id": project_id,
                "phase_number": phase_number,
                "execution_id": execution_id,
                "progress": progress.__dict__,
                "next_steps": self._get_next_steps(progress, strategy)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute modernization phase: {e}")
            return {
                "status": "error",
                "project_id": project_id,
                "phase_number": phase_number,
                "error_message": str(e)
            }
    
    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get current status of a modernization project."""
        
        try:
            project_data = await self.redis_client.get(f"modernization_project:{project_id}")
            if not project_data:
                return {"status": "error", "message": "Project not found"}
            
            project = json.loads(project_data)
            
            # Get all phase progress
            phase_progress = {}
            for i, phase in enumerate(project["modernization_strategy"]["migration_phases"]):
                execution_id = f"{project_id}_phase_{i + 1}"
                progress_data = await self.redis_client.get(f"modernization_progress:{execution_id}")
                if progress_data:
                    phase_progress[f"phase_{i + 1}"] = json.loads(progress_data)
            
            return {
                "status": "success",
                "project_id": project_id,
                "project_overview": {
                    "system_name": project["system_profile"]["name"],
                    "total_phases": len(project["modernization_strategy"]["migration_phases"]),
                    "estimated_timeline": project["execution_timeline"],
                    "estimated_cost": project["estimated_cost"],
                    "estimated_roi": project["estimated_roi"]
                },
                "phase_progress": phase_progress,
                "overall_status": self._calculate_overall_status(phase_progress)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project status: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_technical_debt_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate technical debt score based on analysis results."""
        
        score = 0.0
        
        # Code quality factors
        complexity_score = analysis_results["code_metrics"].get("complexity_score", 0)
        duplication_percentage = analysis_results["code_metrics"].get("duplication_percentage", 0)
        
        score += min(complexity_score * 10, 30)  # Max 30 points for complexity
        score += min(duplication_percentage * 2, 20)  # Max 20 points for duplication
        
        # Security vulnerabilities
        high_vulns = len([v for v in analysis_results["security_scan"] if v.get("severity") == "high"])
        medium_vulns = len([v for v in analysis_results["security_scan"] if v.get("severity") == "medium"])
        
        score += high_vulns * 10  # 10 points per high vulnerability
        score += medium_vulns * 5  # 5 points per medium vulnerability
        
        # Performance bottlenecks
        critical_bottlenecks = len([b for b in analysis_results["performance_analysis"] if b.get("severity") == "high"])
        score += critical_bottlenecks * 5  # 5 points per critical bottleneck
        
        # Outdated dependencies
        for manager, deps in analysis_results["dependency_analysis"].items():
            if isinstance(deps, dict) and "outdated" in deps:
                outdated_count = len(deps["outdated"])
                score += outdated_count * 2  # 2 points per outdated dependency
        
        return min(score, 100.0)  # Cap at 100
    
    def _calculate_project_cost(self, strategy: ModernizationStrategy) -> Dict[str, float]:
        """Calculate estimated project cost."""
        
        base_cost_per_week = 50000  # $50K per week base cost
        
        total_weeks = sum(phase["duration_weeks"] for phase in strategy.migration_phases)
        base_cost = total_weeks * base_cost_per_week
        
        # Complexity multipliers
        complexity_multiplier = 1.0
        if strategy.system_profile.technical_debt_score > 70:
            complexity_multiplier += 0.5
        if len(strategy.system_profile.security_vulnerabilities) > 10:
            complexity_multiplier += 0.3
        if strategy.system_profile.lines_of_code > 500000:
            complexity_multiplier += 0.4
        
        total_cost = base_cost * complexity_multiplier
        
        return {
            "base_cost": base_cost,
            "complexity_multiplier": complexity_multiplier,
            "total_estimated_cost": total_cost,
            "cost_breakdown": {
                "analysis_and_planning": total_cost * 0.2,
                "development_and_migration": total_cost * 0.6,
                "testing_and_validation": total_cost * 0.15,
                "deployment_and_support": total_cost * 0.05
            }
        }
    
    def _calculate_project_roi(
        self, 
        system_profile: LegacySystemProfile, 
        strategy: ModernizationStrategy
    ) -> Dict[str, float]:
        """Calculate estimated return on investment."""
        
        # Current system costs (annual)
        maintenance_cost = system_profile.lines_of_code * 0.50  # $0.50 per LOC annually
        security_incident_cost = len(system_profile.security_vulnerabilities) * 25000  # $25K per vulnerability risk
        performance_cost = len(system_profile.performance_bottlenecks) * 15000  # $15K per bottleneck annually
        
        current_annual_cost = maintenance_cost + security_incident_cost + performance_cost
        
        # Post-modernization costs (annual)
        modernized_maintenance_cost = current_annual_cost * 0.3  # 70% reduction
        modernized_security_cost = security_incident_cost * 0.1  # 90% reduction
        modernized_performance_cost = performance_cost * 0.2  # 80% reduction
        
        modernized_annual_cost = modernized_maintenance_cost + modernized_security_cost + modernized_performance_cost
        
        annual_savings = current_annual_cost - modernized_annual_cost
        project_cost = self._calculate_project_cost(strategy)["total_estimated_cost"]
        
        payback_period_years = project_cost / annual_savings if annual_savings > 0 else float('inf')
        five_year_roi = ((annual_savings * 5) - project_cost) / project_cost * 100 if project_cost > 0 else 0
        
        return {
            "current_annual_cost": current_annual_cost,
            "modernized_annual_cost": modernized_annual_cost,
            "annual_savings": annual_savings,
            "payback_period_years": payback_period_years,
            "five_year_roi_percentage": five_year_roi,
            "five_year_total_savings": annual_savings * 5
        }


# Global service instance
_modernization_service: Optional[LegacyModernizationService] = None


async def get_modernization_service() -> LegacyModernizationService:
    """Get the global modernization service instance."""
    global _modernization_service
    
    if _modernization_service is None:
        _modernization_service = LegacyModernizationService()
        await _modernization_service.initialize()
    
    return _modernization_service


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class LegacyModernizationService(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            async def test_modernization_service():
                """Test the legacy modernization service."""

            service = await get_modernization_service()

            # Sample project configuration
            project_config = {
            "system_name": "Legacy ERP System",
            "codebase_path": "/path/to/legacy/erp",
            "target_requirements": {
            "technology_stack": ["python_django", "react_frontend", "postgresql"],
            "microservices": True,
            "cloud_native": True,
            "performance_requirements": {
            "max_response_time": 200,
            "min_throughput": 5000
            },
            "compliance_requirements": ["SOX", "PCI_DSS"]
            },
            "compliance_requirements": ["SOX", "PCI_DSS"],
            "database_schema": {"tables": 150, "size_gb": 500}
            }

            # Start modernization project
            result = await service.start_modernization_project(project_config, "customer_123")
            self.logger.info("Project start result:", json.dumps(result, indent=2, default=str))

            if result["status"] == "success":
                project_id = result["project_id"]

                # Execute first phase
                phase_result = await service.execute_modernization_phase(project_id, 1)
                self.logger.info("Phase execution result:", json.dumps(phase_result, indent=2, default=str))

                # Get project status
                status = await service.get_project_status(project_id)
                self.logger.info("Project status:", json.dumps(status, indent=2, default=str))

            # Run test
            await test_modernization_service()
            
            return {"status": "completed"}
    
    script_main(LegacyModernizationService)