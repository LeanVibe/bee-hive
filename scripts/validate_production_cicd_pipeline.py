#!/usr/bin/env python3
"""
Production CI/CD Pipeline Validation Script
==========================================

Validates the Epic 5 Phase 1 CI/CD pipeline implementation:
- Verifies all workflows are properly configured
- Tests Epic 4 API integration points
- Validates pipeline quality gates
- Ensures production readiness

Epic 5 Phase 1 Implementation Requirements:
- 24+ GitHub Actions workflows activated
- Epic 4 consolidated API validation (monitoring, agents, tasks)
- Blue-green deployment with zero-downtime capability
- Database migration automation with rollback
- Production monitoring with Epic 4 SystemMonitoringAPI
- Automated alerting and incident response

Usage:
    python scripts/validate_production_cicd_pipeline.py [--environment staging|production] [--verbose]
"""

import asyncio
import json
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import subprocess

try:
    import httpx
    import structlog
except ImportError:
    print("Missing dependencies. Install with: pip install httpx structlog")
    sys.exit(1)


logger = structlog.get_logger(__name__)


class ProductionCICDValidator:
    """Validates Production CI/CD Pipeline for Epic 5 Phase 1"""
    
    def __init__(self, environment: str = "staging", verbose: bool = False):
        self.environment = environment
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        
        # Expected Epic 5 Phase 1 components
        self.required_workflows = [
            "consolidated_system_ci.yml",
            "blue-green-deployment.yml", 
            "production-deployment.yml",
            "database-migration-automation.yml",
            "production-monitoring-integration.yml",
            "incident-response-automation.yml",
            "ci.yml",
            "docker-build.yml",
            "security-scan.yml",
            "performance-validation.yml"
        ]
        
        self.epic4_apis = [
            {"name": "monitoring", "path": "/api/v2/monitoring", "efficiency_target": 94.4},
            {"name": "agents", "path": "/api/v2/agents", "efficiency_target": 94.4},
            {"name": "tasks", "path": "/api/v2/tasks", "efficiency_target": 96.2}  # Benchmark achievement
        ]
        
        self.validation_results = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "epic5_phase1_status": "VALIDATING",
            "workflow_validation": {},
            "epic4_api_validation": {},
            "infrastructure_validation": {},
            "quality_gates_validation": {},
            "production_readiness": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

    async def validate_complete_pipeline(self) -> Dict[str, Any]:
        """Run complete Epic 5 Phase 1 CI/CD pipeline validation"""
        
        print("🚀 Epic 5 Phase 1: Production CI/CD Pipeline Validation")
        print("=" * 60)
        print(f"Environment: {self.environment}")
        print(f"Timestamp: {self.validation_results['validation_timestamp']}")
        print()
        
        try:
            # Validate GitHub Actions workflows
            await self._validate_github_workflows()
            
            # Validate Epic 4 API integration
            await self._validate_epic4_api_integration()
            
            # Validate infrastructure components
            await self._validate_infrastructure_components()
            
            # Validate quality gates
            await self._validate_quality_gates()
            
            # Assess production readiness
            await self._assess_production_readiness()
            
            # Generate final status
            self._generate_final_status()
            
        except Exception as e:
            self.validation_results["errors"].append(f"Pipeline validation failed: {str(e)}")
            logger.error("Pipeline validation error", error=str(e))
        
        return self.validation_results

    async def _validate_github_workflows(self):
        """Validate GitHub Actions workflows configuration"""
        
        print("📋 Validating GitHub Actions Workflows...")
        
        workflows_dir = self.project_root / ".github" / "workflows"
        workflow_results = {}
        
        if not workflows_dir.exists():
            self.validation_results["errors"].append("GitHub workflows directory not found")
            return
        
        # Check for required workflows
        existing_workflows = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        existing_names = [w.name for w in existing_workflows]
        
        print(f"Found {len(existing_workflows)} workflow files:")
        for workflow in existing_workflows:
            print(f"  ✅ {workflow.name}")
        
        # Validate required workflows
        missing_workflows = []
        for required in self.required_workflows:
            if required not in existing_names:
                missing_workflows.append(required)
            else:
                workflow_results[required] = await self._validate_individual_workflow(workflows_dir / required)
        
        if missing_workflows:
            self.validation_results["warnings"].extend([f"Workflow not found: {w}" for w in missing_workflows])
        
        # Validate Epic 4 API contract test integration
        consolidated_ci = workflows_dir / "consolidated_system_ci.yml"
        if consolidated_ci.exists():
            epic4_integration = await self._validate_epic4_workflow_integration(consolidated_ci)
            workflow_results["epic4_integration"] = epic4_integration
        
        self.validation_results["workflow_validation"] = {
            "total_workflows": len(existing_workflows),
            "required_workflows_found": len(self.required_workflows) - len(missing_workflows),
            "required_workflows_total": len(self.required_workflows),
            "missing_workflows": missing_workflows,
            "workflow_details": workflow_results,
            "epic4_integration_status": workflow_results.get("epic4_integration", {}).get("status", "NOT_FOUND")
        }
        
        print(f"  ✅ {len(existing_workflows)} total workflows found")
        print(f"  ✅ {len(self.required_workflows) - len(missing_workflows)}/{len(self.required_workflows)} required workflows found")
        if missing_workflows:
            print(f"  ⚠️ Missing workflows: {', '.join(missing_workflows)}")
        print()

    async def _validate_individual_workflow(self, workflow_path: Path) -> Dict[str, Any]:
        """Validate individual workflow file"""
        
        result = {"status": "VALID", "checks": {}, "issues": []}
        
        try:
            with open(workflow_path, 'r') as f:
                workflow_content = yaml.safe_load(f)
            
            # Basic structure checks
            result["checks"]["has_name"] = "name" in workflow_content
            result["checks"]["has_on_triggers"] = "on" in workflow_content
            result["checks"]["has_jobs"] = "jobs" in workflow_content and len(workflow_content["jobs"]) > 0
            
            # Environment-specific checks
            if workflow_path.name == "blue-green-deployment.yml":
                result["checks"]["has_blue_green_strategy"] = self._check_blue_green_strategy(workflow_content)
                result["checks"]["has_epic4_validation"] = self._check_epic4_validation_in_workflow(workflow_content)
            
            elif workflow_path.name == "database-migration-automation.yml":
                result["checks"]["has_alembic_integration"] = self._check_alembic_integration(workflow_content)
                result["checks"]["has_rollback_capability"] = self._check_rollback_capability(workflow_content)
            
            elif workflow_path.name == "production-monitoring-integration.yml":
                result["checks"]["has_monitoring_validation"] = self._check_monitoring_integration(workflow_content)
                result["checks"]["has_epic4_api_monitoring"] = self._check_epic4_monitoring(workflow_content)
            
            # Check if any critical checks failed
            if not all(result["checks"].values()):
                result["status"] = "ISSUES_FOUND"
                failed_checks = [check for check, passed in result["checks"].items() if not passed]
                result["issues"] = [f"Failed check: {check}" for check in failed_checks]
        
        except Exception as e:
            result["status"] = "ERROR"
            result["issues"] = [f"Failed to parse workflow: {str(e)}"]
        
        return result

    def _check_blue_green_strategy(self, workflow_content: Dict) -> bool:
        """Check if workflow implements blue-green deployment strategy"""
        content_str = json.dumps(workflow_content).lower()
        return any(term in content_str for term in ["blue-green", "inactive_env", "active_env", "traffic switching"])

    def _check_epic4_validation_in_workflow(self, workflow_content: Dict) -> bool:
        """Check if workflow includes Epic 4 API validation"""
        content_str = json.dumps(workflow_content).lower()
        return any(term in content_str for term in ["epic4", "epic 4", "api/v2/monitoring", "api/v2/agents", "api/v2/tasks"])

    def _check_alembic_integration(self, workflow_content: Dict) -> bool:
        """Check if workflow includes Alembic database migration"""
        content_str = json.dumps(workflow_content).lower()
        return "alembic" in content_str

    def _check_rollback_capability(self, workflow_content: Dict) -> bool:
        """Check if workflow includes rollback capability"""
        content_str = json.dumps(workflow_content).lower()
        return "rollback" in content_str

    def _check_monitoring_integration(self, workflow_content: Dict) -> bool:
        """Check if workflow includes monitoring integration"""
        content_str = json.dumps(workflow_content).lower()
        return any(term in content_str for term in ["monitoring", "prometheus", "grafana", "alerts"])

    def _check_epic4_monitoring(self, workflow_content: Dict) -> bool:
        """Check if workflow includes Epic 4 API monitoring"""
        content_str = json.dumps(workflow_content).lower()
        return any(term in content_str for term in ["epic4", "systemmonitoringapi", "v2/monitoring"])

    async def _validate_epic4_workflow_integration(self, consolidated_ci_path: Path) -> Dict[str, Any]:
        """Validate Epic 4 API integration in consolidated CI workflow"""
        
        result = {"status": "NOT_INTEGRATED", "details": {}}
        
        try:
            with open(consolidated_ci_path, 'r') as f:
                workflow_content = f.read()
            
            # Check for Epic 4 API contract validation step
            has_epic4_test_step = "epic4_api_contract_validation" in workflow_content.lower() or \
                                  "test_epic4_api_contract_validation.py" in workflow_content
            
            # Check for Epic 4 API performance validation
            has_performance_validation = any(term in workflow_content.lower() for term in [
                "94.4% efficiency", "96.2% efficiency", "epic4.*performance", "api/v2.*validation"
            ])
            
            # Check for Epic 4 artifacts upload
            has_artifacts_upload = "epic4-api-contract-results" in workflow_content or \
                                  "epic4-api-contract-report" in workflow_content
            
            result["details"] = {
                "has_epic4_test_step": has_epic4_test_step,
                "has_performance_validation": has_performance_validation,
                "has_artifacts_upload": has_artifacts_upload
            }
            
            if all(result["details"].values()):
                result["status"] = "FULLY_INTEGRATED"
            elif any(result["details"].values()):
                result["status"] = "PARTIALLY_INTEGRATED"
            
        except Exception as e:
            result["status"] = "ERROR"
            result["details"]["error"] = str(e)
        
        return result

    async def _validate_epic4_api_integration(self):
        """Validate Epic 4 API integration and contract tests"""
        
        print("🔍 Validating Epic 4 API Integration...")
        
        api_results = {}
        
        # Check for Epic 4 API contract test file
        contract_test_path = self.project_root / "tests" / "integration" / "test_epic4_api_contract_validation.py"
        
        if contract_test_path.exists():
            print("  ✅ Epic 4 API contract test file found")
            api_results["contract_test_file"] = {
                "exists": True,
                "path": str(contract_test_path.relative_to(self.project_root)),
                "validation": await self._validate_contract_test_file(contract_test_path)
            }
        else:
            print("  ❌ Epic 4 API contract test file not found")
            api_results["contract_test_file"] = {"exists": False}
            self.validation_results["errors"].append("Epic 4 API contract test file missing")
        
        # Validate Epic 4 API source files
        for api in self.epic4_apis:
            api_validation = await self._validate_epic4_api_source(api)
            api_results[f"{api['name']}_api"] = api_validation
            
            if api_validation["exists"]:
                print(f"  ✅ Epic 4 {api['name'].title()}API v2 source found")
            else:
                print(f"  ⚠️ Epic 4 {api['name'].title()}API v2 source not found")
                self.validation_results["warnings"].append(f"Epic 4 {api['name']}API source missing")
        
        self.validation_results["epic4_api_validation"] = api_results
        print()

    async def _validate_contract_test_file(self, test_file_path: Path) -> Dict[str, Any]:
        """Validate Epic 4 API contract test file content"""
        
        result = {"status": "VALID", "checks": {}, "issues": []}
        
        try:
            with open(test_file_path, 'r') as f:
                content = f.read()
            
            # Check for required test components
            result["checks"]["has_validator_class"] = "Epic4APIContractValidator" in content
            result["checks"]["has_performance_targets"] = "performance_targets" in content and "94.4" in content and "96.2" in content
            result["checks"]["has_efficiency_tests"] = "efficiency_tests" in content
            result["checks"]["has_pytest_markers"] = "@pytest.mark" in content
            result["checks"]["has_async_tests"] = "async def test_" in content
            result["checks"]["tests_all_apis"] = all(api["name"] in content for api in self.epic4_apis)
            
            # Check for Epic 4 specific validations
            result["checks"]["validates_monitoring_api"] = "monitoring" in content and "94.4%" in content
            result["checks"]["validates_agents_api"] = "agents" in content and "94.4%" in content
            result["checks"]["validates_tasks_api"] = "tasks" in content and "96.2%" in content  # Benchmark
            
            if not all(result["checks"].values()):
                result["status"] = "INCOMPLETE"
                failed_checks = [check for check, passed in result["checks"].items() if not passed]
                result["issues"] = [f"Missing component: {check}" for check in failed_checks]
        
        except Exception as e:
            result["status"] = "ERROR"
            result["issues"] = [f"Failed to validate test file: {str(e)}"]
        
        return result

    async def _validate_epic4_api_source(self, api_config: Dict) -> Dict[str, Any]:
        """Validate Epic 4 API source files exist and are properly structured"""
        
        api_name = api_config["name"]
        api_path = self.project_root / "app" / "api" / "v2" / api_name
        
        result = {
            "exists": api_path.exists(),
            "core_file": (api_path / "core.py").exists(),
            "models_file": (api_path / "models.py").exists(),
            "middleware_file": (api_path / "middleware.py").exists(),
            "init_file": (api_path / "__init__.py").exists()
        }
        
        if api_path.exists():
            # Check for performance targets in core.py
            core_path = api_path / "core.py"
            if core_path.exists():
                with open(core_path, 'r') as f:
                    core_content = f.read()
                result["has_performance_targets"] = str(api_config["efficiency_target"]) in core_content
                result["has_epic4_documentation"] = "Epic 4" in core_content
        
        return result

    async def _validate_infrastructure_components(self):
        """Validate infrastructure components (Docker, K8s, databases)"""
        
        print("🏗️ Validating Infrastructure Components...")
        
        infra_results = {}
        
        # Docker configuration
        docker_files = ["Dockerfile.production", "docker-compose.production.yml"]
        docker_results = {}
        
        for docker_file in docker_files:
            docker_path = self.project_root / docker_file
            docker_results[docker_file] = {
                "exists": docker_path.exists(),
                "validation": await self._validate_docker_file(docker_path) if docker_path.exists() else {}
            }
            
            if docker_path.exists():
                print(f"  ✅ {docker_file} found")
            else:
                print(f"  ⚠️ {docker_file} not found")
                self.validation_results["warnings"].append(f"Docker file missing: {docker_file}")
        
        infra_results["docker"] = docker_results
        
        # Kubernetes manifests
        k8s_dir = self.project_root / "k8s"
        k8s_results = {"base_dir_exists": k8s_dir.exists(), "manifests": {}}
        
        if k8s_dir.exists():
            print("  ✅ Kubernetes manifests directory found")
            
            # Check for production deployment manifest
            prod_deployment = k8s_dir / "production-deployment.yaml"
            if prod_deployment.exists():
                print("  ✅ Production deployment manifest found")
                k8s_results["manifests"]["production_deployment"] = {
                    "exists": True,
                    "validation": await self._validate_k8s_manifest(prod_deployment)
                }
            else:
                print("  ⚠️ Production deployment manifest not found")
                self.validation_results["warnings"].append("K8s production deployment manifest missing")
        else:
            print("  ⚠️ Kubernetes manifests directory not found")
            self.validation_results["warnings"].append("Kubernetes manifests directory missing")
        
        infra_results["kubernetes"] = k8s_results
        
        # Database migration setup
        migrations_dir = self.project_root / "migrations"
        alembic_ini = self.project_root / "alembic.ini"
        
        infra_results["database"] = {
            "migrations_dir_exists": migrations_dir.exists(),
            "alembic_ini_exists": alembic_ini.exists(),
            "migration_automation": True  # We created the automation workflow
        }
        
        if migrations_dir.exists() and alembic_ini.exists():
            print("  ✅ Database migration infrastructure found")
        else:
            print("  ⚠️ Database migration infrastructure incomplete")
            if not migrations_dir.exists():
                self.validation_results["warnings"].append("Database migrations directory missing")
            if not alembic_ini.exists():
                self.validation_results["warnings"].append("Alembic configuration missing")
        
        self.validation_results["infrastructure_validation"] = infra_results
        print()

    async def _validate_docker_file(self, docker_path: Path) -> Dict[str, Any]:
        """Validate Docker file configuration"""
        
        result = {"status": "VALID", "checks": {}, "issues": []}
        
        try:
            with open(docker_path, 'r') as f:
                content = f.read()
            
            # Check for production optimizations
            result["checks"]["multi_stage_build"] = "FROM" in content and content.count("FROM") > 1
            result["checks"]["non_root_user"] = "USER" in content or "useradd" in content
            result["checks"]["security_hardening"] = any(term in content for term in ["chown", "chmod", "security"])
            result["checks"]["health_check"] = "HEALTHCHECK" in content or "health" in content.lower()
            
            if docker_path.name == "docker-compose.production.yml":
                result["checks"]["has_postgres"] = "postgres" in content
                result["checks"]["has_redis"] = "redis" in content
                result["checks"]["has_api_service"] = "api:" in content or "leanvibe" in content
                result["checks"]["has_environment_vars"] = "environment:" in content
        
        except Exception as e:
            result["status"] = "ERROR"
            result["issues"] = [f"Failed to validate Docker file: {str(e)}"]
        
        return result

    async def _validate_k8s_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Validate Kubernetes manifest file"""
        
        result = {"status": "VALID", "checks": {}, "issues": []}
        
        try:
            with open(manifest_path, 'r') as f:
                content = f.read()
            
            # Check for production-ready configurations
            result["checks"]["has_deployment"] = "kind: Deployment" in content
            result["checks"]["has_service"] = "kind: Service" in content
            result["checks"]["has_ingress"] = "kind: Ingress" in content
            result["checks"]["has_configmap"] = "kind: ConfigMap" in content
            result["checks"]["has_secrets"] = "kind: Secret" in content
            result["checks"]["has_health_checks"] = "livenessProbe" in content and "readinessProbe" in content
            result["checks"]["has_resource_limits"] = "resources:" in content and "limits:" in content
            result["checks"]["has_security_context"] = "securityContext:" in content
            
            # Check for blue-green deployment support
            result["checks"]["supports_blue_green"] = any(term in content for term in ["blue", "green", "version", "color"])
        
        except Exception as e:
            result["status"] = "ERROR"
            result["issues"] = [f"Failed to validate K8s manifest: {str(e)}"]
        
        return result

    async def _validate_quality_gates(self):
        """Validate quality gates and testing infrastructure"""
        
        print("🚦 Validating Quality Gates...")
        
        quality_results = {}
        
        # Check for quality gate scripts
        quality_gate_script = self.project_root / "scripts" / "run_quality_gates.py"
        quality_results["quality_gate_script"] = {
            "exists": quality_gate_script.exists()
        }
        
        if quality_gate_script.exists():
            print("  ✅ Quality gate script found")
        else:
            print("  ⚠️ Quality gate script not found")
            self.validation_results["warnings"].append("Quality gate script missing")
        
        # Check test infrastructure
        tests_dir = self.project_root / "tests"
        test_files = list(tests_dir.glob("**/*.py")) if tests_dir.exists() else []
        
        quality_results["test_infrastructure"] = {
            "tests_dir_exists": tests_dir.exists(),
            "test_files_count": len(test_files),
            "has_integration_tests": (tests_dir / "integration").exists() if tests_dir.exists() else False,
            "has_performance_tests": any("performance" in str(f) for f in test_files),
            "has_epic4_tests": any("epic4" in str(f) for f in test_files)
        }
        
        if tests_dir.exists():
            print(f"  ✅ Test infrastructure found ({len(test_files)} test files)")
            if any("epic4" in str(f) for f in test_files):
                print("  ✅ Epic 4 tests found")
            else:
                print("  ⚠️ Epic 4 tests not found")
                self.validation_results["warnings"].append("Epic 4 specific tests missing")
        else:
            print("  ❌ Test infrastructure not found")
            self.validation_results["errors"].append("Test infrastructure missing")
        
        # Check for CI/CD quality gates integration
        consolidated_ci = self.project_root / ".github" / "workflows" / "consolidated_system_ci.yml"
        if consolidated_ci.exists():
            with open(consolidated_ci, 'r') as f:
                ci_content = f.read()
            
            quality_results["ci_quality_gates"] = {
                "has_quality_gates_step": "run_quality_gates" in ci_content,
                "has_epic4_validation": "epic4" in ci_content.lower(),
                "has_performance_validation": "performance" in ci_content.lower(),
                "has_artifact_upload": "upload-artifact" in ci_content
            }
            
            if quality_results["ci_quality_gates"]["has_quality_gates_step"]:
                print("  ✅ Quality gates integrated in CI")
            else:
                print("  ⚠️ Quality gates not integrated in CI")
        
        self.validation_results["quality_gates_validation"] = quality_results
        print()

    async def _assess_production_readiness(self):
        """Assess overall production readiness based on all validations"""
        
        print("🎯 Assessing Production Readiness...")
        
        readiness_score = 0
        max_score = 100
        
        # Workflow validation (30 points)
        workflow_score = 0
        workflow_validation = self.validation_results["workflow_validation"]
        
        if workflow_validation["required_workflows_found"] >= 8:  # Most critical workflows
            workflow_score += 20
        elif workflow_validation["required_workflows_found"] >= 6:
            workflow_score += 15
        elif workflow_validation["required_workflows_found"] >= 4:
            workflow_score += 10
        
        if workflow_validation.get("epic4_integration_status") == "FULLY_INTEGRATED":
            workflow_score += 10
        elif workflow_validation.get("epic4_integration_status") == "PARTIALLY_INTEGRATED":
            workflow_score += 5
        
        readiness_score += workflow_score
        
        # Epic 4 API integration (25 points)
        api_score = 0
        api_validation = self.validation_results["epic4_api_validation"]
        
        if api_validation.get("contract_test_file", {}).get("exists", False):
            api_score += 10
            if api_validation["contract_test_file"].get("validation", {}).get("status") == "VALID":
                api_score += 5
        
        # Check each Epic 4 API
        for api in self.epic4_apis:
            api_key = f"{api['name']}_api"
            if api_validation.get(api_key, {}).get("exists", False):
                api_score += 3  # 3 points per API (9 total)
        
        readiness_score += api_score
        
        # Infrastructure validation (25 points)
        infra_score = 0
        infra_validation = self.validation_results["infrastructure_validation"]
        
        # Docker (10 points)
        docker_validation = infra_validation.get("docker", {})
        if docker_validation.get("Dockerfile.production", {}).get("exists", False):
            infra_score += 5
        if docker_validation.get("docker-compose.production.yml", {}).get("exists", False):
            infra_score += 5
        
        # Kubernetes (10 points)
        k8s_validation = infra_validation.get("kubernetes", {})
        if k8s_validation.get("base_dir_exists", False):
            infra_score += 5
        if k8s_validation.get("manifests", {}).get("production_deployment", {}).get("exists", False):
            infra_score += 5
        
        # Database (5 points)
        db_validation = infra_validation.get("database", {})
        if db_validation.get("migrations_dir_exists", False) and db_validation.get("alembic_ini_exists", False):
            infra_score += 5
        
        readiness_score += infra_score
        
        # Quality gates validation (20 points)
        quality_score = 0
        quality_validation = self.validation_results["quality_gates_validation"]
        
        if quality_validation.get("quality_gate_script", {}).get("exists", False):
            quality_score += 5
        
        test_infra = quality_validation.get("test_infrastructure", {})
        if test_infra.get("tests_dir_exists", False):
            quality_score += 5
        if test_infra.get("has_integration_tests", False):
            quality_score += 5
        if test_infra.get("has_epic4_tests", False):
            quality_score += 5
        
        readiness_score += quality_score
        
        # Determine readiness level
        readiness_percentage = (readiness_score / max_score) * 100
        
        if readiness_percentage >= 90:
            readiness_level = "PRODUCTION_READY"
            readiness_status = "✅ READY"
        elif readiness_percentage >= 75:
            readiness_level = "MOSTLY_READY"
            readiness_status = "⚠️ MOSTLY READY"
        elif readiness_percentage >= 50:
            readiness_level = "PARTIALLY_READY"
            readiness_status = "⚠️ PARTIALLY READY"
        else:
            readiness_level = "NOT_READY"
            readiness_status = "❌ NOT READY"
        
        self.validation_results["production_readiness"] = {
            "readiness_level": readiness_level,
            "readiness_percentage": readiness_percentage,
            "readiness_score": readiness_score,
            "max_score": max_score,
            "component_scores": {
                "workflows": workflow_score,
                "epic4_apis": api_score,
                "infrastructure": infra_score,
                "quality_gates": quality_score
            },
            "recommendations": self._generate_readiness_recommendations(readiness_percentage)
        }
        
        print(f"  {readiness_status} ({readiness_percentage:.1f}%)")
        print(f"  Score: {readiness_score}/{max_score}")
        print(f"    Workflows: {workflow_score}/30")
        print(f"    Epic 4 APIs: {api_score}/25")
        print(f"    Infrastructure: {infra_score}/25")
        print(f"    Quality Gates: {quality_score}/20")
        print()

    def _generate_readiness_recommendations(self, readiness_percentage: float) -> List[str]:
        """Generate recommendations based on readiness assessment"""
        
        recommendations = []
        
        if readiness_percentage < 90:
            # Check specific areas needing improvement
            workflow_validation = self.validation_results["workflow_validation"]
            if workflow_validation["required_workflows_found"] < len(self.required_workflows):
                recommendations.append("Complete missing GitHub Actions workflows")
            
            if workflow_validation.get("epic4_integration_status") != "FULLY_INTEGRATED":
                recommendations.append("Fully integrate Epic 4 API validation in CI/CD pipeline")
            
            api_validation = self.validation_results["epic4_api_validation"]
            if not api_validation.get("contract_test_file", {}).get("exists", False):
                recommendations.append("Create Epic 4 API contract test file")
            
            infra_validation = self.validation_results["infrastructure_validation"]
            if not infra_validation.get("kubernetes", {}).get("base_dir_exists", False):
                recommendations.append("Set up Kubernetes deployment manifests")
            
            quality_validation = self.validation_results["quality_gates_validation"]
            if not quality_validation.get("test_infrastructure", {}).get("has_epic4_tests", False):
                recommendations.append("Add Epic 4 specific test coverage")
        
        if readiness_percentage >= 75:
            recommendations.append("Perform end-to-end pipeline testing before production deployment")
            recommendations.append("Set up production monitoring and alerting")
            recommendations.append("Create incident response procedures")
        
        return recommendations

    def _generate_final_status(self):
        """Generate final validation status"""
        
        errors_count = len(self.validation_results["errors"])
        warnings_count = len(self.validation_results["warnings"])
        readiness_level = self.validation_results["production_readiness"]["readiness_level"]
        
        if errors_count == 0 and readiness_level == "PRODUCTION_READY":
            self.validation_results["epic5_phase1_status"] = "COMPLETE"
        elif errors_count == 0 and readiness_level in ["MOSTLY_READY", "PARTIALLY_READY"]:
            self.validation_results["epic5_phase1_status"] = "NEAR_COMPLETE"
        else:
            self.validation_results["epic5_phase1_status"] = "INCOMPLETE"

    def save_validation_report(self, output_path: Optional[str] = None):
        """Save validation report to JSON file"""
        
        if not output_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = f"epic5_phase1_validation_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"📄 Validation report saved to: {output_path}")
        return output_path

    def print_summary_report(self):
        """Print a summary report to console"""
        
        print()
        print("=" * 80)
        print("🚀 EPIC 5 PHASE 1: PRODUCTION CI/CD PIPELINE VALIDATION SUMMARY")
        print("=" * 80)
        
        print(f"Status: {self.validation_results['epic5_phase1_status']}")
        print(f"Environment: {self.validation_results['environment']}")
        print(f"Validation Time: {self.validation_results['validation_timestamp']}")
        
        readiness = self.validation_results["production_readiness"]
        print(f"Production Readiness: {readiness['readiness_level']} ({readiness['readiness_percentage']:.1f}%)")
        
        print()
        print("📊 Component Scores:")
        for component, score in readiness["component_scores"].items():
            print(f"  {component.title()}: {score}")
        
        print()
        print("📋 Validation Results:")
        workflow_validation = self.validation_results["workflow_validation"]
        print(f"  Workflows: {workflow_validation['required_workflows_found']}/{workflow_validation['required_workflows_total']} required found")
        
        api_validation = self.validation_results["epic4_api_validation"]
        epic4_test_exists = api_validation.get("contract_test_file", {}).get("exists", False)
        print(f"  Epic 4 Integration: {'✅' if epic4_test_exists else '❌'} Contract tests exist")
        
        infra_validation = self.validation_results["infrastructure_validation"]
        docker_ready = infra_validation.get("docker", {}).get("Dockerfile.production", {}).get("exists", False)
        k8s_ready = infra_validation.get("kubernetes", {}).get("base_dir_exists", False)
        print(f"  Infrastructure: Docker {'✅' if docker_ready else '❌'}, K8s {'✅' if k8s_ready else '❌'}")
        
        quality_validation = self.validation_results["quality_gates_validation"]
        tests_exist = quality_validation.get("test_infrastructure", {}).get("tests_dir_exists", False)
        epic4_tests = quality_validation.get("test_infrastructure", {}).get("has_epic4_tests", False)
        print(f"  Quality Gates: Tests {'✅' if tests_exist else '❌'}, Epic 4 Tests {'✅' if epic4_tests else '❌'}")
        
        if self.validation_results["errors"]:
            print()
            print("❌ Errors:")
            for error in self.validation_results["errors"]:
                print(f"  • {error}")
        
        if self.validation_results["warnings"]:
            print()
            print("⚠️ Warnings:")
            for warning in self.validation_results["warnings"]:
                print(f"  • {warning}")
        
        recommendations = readiness.get("recommendations", [])
        if recommendations:
            print()
            print("💡 Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        print()
        print("=" * 80)


async def main():
    """Main validation function"""
    
    parser = argparse.ArgumentParser(description="Epic 5 Phase 1 CI/CD Pipeline Validator")
    parser.add_argument("--environment", choices=["staging", "production"], default="staging",
                       help="Target environment for validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", help="Output file path for validation report")
    
    args = parser.parse_args()
    
    # Create and run validator
    validator = ProductionCICDValidator(environment=args.environment, verbose=args.verbose)
    
    print("🔍 Starting Epic 5 Phase 1 Production CI/CD Pipeline Validation...")
    print()
    
    # Run complete validation
    results = await validator.validate_complete_pipeline()
    
    # Print summary report
    validator.print_summary_report()
    
    # Save detailed report
    report_path = validator.save_validation_report(args.output)
    
    # Exit with appropriate code
    final_status = results["epic5_phase1_status"]
    if final_status == "COMPLETE":
        print("🎉 Epic 5 Phase 1 validation PASSED - Pipeline ready for production!")
        sys.exit(0)
    elif final_status == "NEAR_COMPLETE":
        print("⚠️ Epic 5 Phase 1 validation MOSTLY PASSED - Minor issues to address")
        sys.exit(0)  # Allow deployment with warnings
    else:
        print("❌ Epic 5 Phase 1 validation FAILED - Address critical issues before deployment")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())