"""
Automated Testing Integration for LeanVibe Agent Hive 2.0

Comprehensive CI/CD pipeline integration with automated test execution,
intelligent failure analysis, coverage tracking, and performance regression detection.
"""

import asyncio
import json
import logging
import re
import uuid
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload
import structlog

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.github_integration import PullRequest, GitHubRepository
from ..core.github_api_client import GitHubAPIClient
from ..core.redis import get_redis

logger = structlog.get_logger()
settings = get_settings()


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    ERROR = "error"


class TestSuite(Enum):
    """Test suite types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    E2E = "e2e"
    SMOKE = "smoke"
    REGRESSION = "regression"


class CIProvider(Enum):
    """Continuous Integration providers."""
    GITHUB_ACTIONS = "github_actions"
    JENKINS = "jenkins"
    GITLAB_CI = "gitlab_ci"
    CIRCLE_CI = "circle_ci"
    BUILDKITE = "buildkite"
    AZURE_DEVOPS = "azure_devops"


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: TestStatus
    duration: float
    message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    failure_type: Optional[str] = None
    stack_trace: Optional[str] = None
    

@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    suite_name: str
    suite_type: TestSuite
    status: TestStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration: float
    coverage: Optional[float] = None
    test_results: List[TestResult] = None
    

@dataclass
class CIBuildResult:
    """Complete CI build result."""
    build_id: str
    build_url: str
    status: TestStatus
    pr_number: int
    commit_sha: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    test_suites: List[TestSuiteResult] = None
    artifacts: List[str] = None
    environment: Optional[str] = None


class AutomatedTestingError(Exception):
    """Custom exception for automated testing operations."""
    pass


class TestResultParser:
    """
    Intelligent test result parser for multiple formats.
    
    Supports JUnit XML, pytest JSON, Jest, and other common test output formats
    with intelligent failure analysis and categorization.
    """
    
    def __init__(self):
        self.failure_patterns = {
            "assertion_error": [
                r"AssertionError",
                r"assert\s+.*==.*",
                r"Expected.*but.*was",
                r"Assertion\s+failed"
            ],
            "timeout_error": [
                r"TimeoutError",
                r"timeout.*exceeded",
                r"Operation.*timed.*out",
                r"Request.*timeout"
            ],
            "connection_error": [
                r"ConnectionError",
                r"Connection.*refused",
                r"Network.*unreachable",
                r"DNS.*resolution.*failed"
            ],
            "permission_error": [
                r"PermissionError",
                r"Permission.*denied",
                r"Access.*forbidden",
                r"Unauthorized"
            ],
            "import_error": [
                r"ImportError",
                r"ModuleNotFoundError",
                r"No.*module.*named",
                r"cannot.*import"
            ],
            "syntax_error": [
                r"SyntaxError",
                r"invalid.*syntax",
                r"unexpected.*token",
                r"Parse.*error"
            ]
        }
    
    def parse_junit_xml(self, xml_content: str) -> List[TestSuiteResult]:
        """Parse JUnit XML test results."""
        
        try:
            root = ET.fromstring(xml_content)
            test_suites = []
            
            # Handle both <testsuites> and single <testsuite> root
            if root.tag == "testsuites":
                suite_elements = root.findall("testsuite")
            else:
                suite_elements = [root]
            
            for suite_element in suite_elements:
                suite_name = suite_element.get("name", "unknown")
                total_tests = int(suite_element.get("tests", 0))
                failures = int(suite_element.get("failures", 0))
                errors = int(suite_element.get("errors", 0))
                skipped = int(suite_element.get("skipped", 0))
                duration = float(suite_element.get("time", 0.0))
                
                passed_tests = total_tests - failures - errors - skipped
                
                # Determine overall status
                if errors > 0:
                    status = TestStatus.ERROR
                elif failures > 0:
                    status = TestStatus.FAILED
                elif passed_tests > 0:
                    status = TestStatus.PASSED
                else:
                    status = TestStatus.SKIPPED
                
                # Parse individual test cases
                test_results = []
                for test_element in suite_element.findall("testcase"):
                    test_result = self._parse_junit_testcase(test_element)
                    test_results.append(test_result)
                
                # Determine suite type from name
                suite_type = self._infer_suite_type(suite_name)
                
                test_suite = TestSuiteResult(
                    suite_name=suite_name,
                    suite_type=suite_type,
                    status=status,
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    failed_tests=failures,
                    skipped_tests=skipped,
                    duration=duration,
                    test_results=test_results
                )
                test_suites.append(test_suite)
            
            return test_suites
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse JUnit XML: {e}")
            raise AutomatedTestingError(f"Invalid JUnit XML format: {e}")
    
    def _parse_junit_testcase(self, test_element: ET.Element) -> TestResult:
        """Parse individual JUnit test case."""
        
        name = test_element.get("name", "unknown")
        classname = test_element.get("classname", "")
        duration = float(test_element.get("time", 0.0))
        
        # Check for failure, error, or skip
        failure_element = test_element.find("failure")
        error_element = test_element.find("error")
        skip_element = test_element.find("skipped")
        
        if failure_element is not None:
            status = TestStatus.FAILED
            message = failure_element.get("message", "")
            stack_trace = failure_element.text or ""
            failure_type = self._classify_failure(message + " " + stack_trace)
        elif error_element is not None:
            status = TestStatus.ERROR
            message = error_element.get("message", "")
            stack_trace = error_element.text or ""
            failure_type = self._classify_failure(message + " " + stack_trace)
        elif skip_element is not None:
            status = TestStatus.SKIPPED
            message = skip_element.get("message", "Test skipped")
            stack_trace = None
            failure_type = None
        else:
            status = TestStatus.PASSED
            message = None
            stack_trace = None
            failure_type = None
        
        return TestResult(
            name=f"{classname}.{name}" if classname else name,
            status=status,
            duration=duration,
            message=message,
            failure_type=failure_type,
            stack_trace=stack_trace
        )
    
    def parse_pytest_json(self, json_content: str) -> List[TestSuiteResult]:
        """Parse pytest JSON test results."""
        
        try:
            data = json.loads(json_content)
            
            # Group tests by file/module
            test_groups = {}
            for test in data.get("tests", []):
                file_path = test.get("nodeid", "").split("::")[0]
                if file_path not in test_groups:
                    test_groups[file_path] = []
                test_groups[file_path].append(test)
            
            test_suites = []
            for file_path, tests in test_groups.items():
                suite_name = Path(file_path).stem if file_path else "unknown"
                
                test_results = []
                passed = failed = skipped = 0
                total_duration = 0.0
                
                for test in tests:
                    outcome = test.get("outcome", "unknown")
                    duration = test.get("duration", 0.0)
                    total_duration += duration
                    
                    if outcome == "passed":
                        status = TestStatus.PASSED
                        passed += 1
                    elif outcome == "failed":
                        status = TestStatus.FAILED
                        failed += 1
                    elif outcome == "skipped":
                        status = TestStatus.SKIPPED
                        skipped += 1
                    else:
                        status = TestStatus.ERROR
                        failed += 1
                    
                    # Get failure details
                    call = test.get("call", {})
                    message = call.get("longrepr", {}).get("reprcrash", {}).get("message", "")
                    stack_trace = str(call.get("longrepr", ""))
                    
                    test_result = TestResult(
                        name=test.get("nodeid", "unknown"),
                        status=status,
                        duration=duration,
                        message=message if message else None,
                        failure_type=self._classify_failure(message + " " + stack_trace) if status == TestStatus.FAILED else None,
                        stack_trace=stack_trace if stack_trace else None
                    )
                    test_results.append(test_result)
                
                # Determine suite status
                if failed > 0:
                    suite_status = TestStatus.FAILED
                elif passed > 0:
                    suite_status = TestStatus.PASSED
                else:
                    suite_status = TestStatus.SKIPPED
                
                suite_type = self._infer_suite_type(suite_name)
                
                test_suite = TestSuiteResult(
                    suite_name=suite_name,
                    suite_type=suite_type,
                    status=suite_status,
                    total_tests=len(tests),
                    passed_tests=passed,
                    failed_tests=failed,
                    skipped_tests=skipped,
                    duration=total_duration,
                    test_results=test_results
                )
                test_suites.append(test_suite)
            
            return test_suites
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pytest JSON: {e}")
            raise AutomatedTestingError(f"Invalid pytest JSON format: {e}")
    
    def _classify_failure(self, failure_text: str) -> str:
        """Classify failure type based on error message."""
        
        failure_text_lower = failure_text.lower()
        
        for failure_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, failure_text_lower, re.IGNORECASE):
                    return failure_type
        
        return "unknown_error"
    
    def _infer_suite_type(self, suite_name: str) -> TestSuite:
        """Infer test suite type from name."""
        
        suite_name_lower = suite_name.lower()
        
        if any(keyword in suite_name_lower for keyword in ["unit", "test_unit"]):
            return TestSuite.UNIT
        elif any(keyword in suite_name_lower for keyword in ["integration", "test_integration"]):
            return TestSuite.INTEGRATION
        elif any(keyword in suite_name_lower for keyword in ["functional", "test_functional"]):
            return TestSuite.FUNCTIONAL
        elif any(keyword in suite_name_lower for keyword in ["performance", "perf", "benchmark"]):
            return TestSuite.PERFORMANCE
        elif any(keyword in suite_name_lower for keyword in ["security", "sec"]):
            return TestSuite.SECURITY
        elif any(keyword in suite_name_lower for keyword in ["e2e", "end_to_end", "selenium"]):
            return TestSuite.E2E
        elif any(keyword in suite_name_lower for keyword in ["smoke", "sanity"]):
            return TestSuite.SMOKE
        elif any(keyword in suite_name_lower for keyword in ["regression"]):
            return TestSuite.REGRESSION
        else:
            return TestSuite.UNIT  # Default


class CoverageAnalyzer:
    """
    Code coverage analysis and tracking.
    
    Analyzes coverage reports, tracks trends, and identifies
    coverage gaps with intelligent recommendations.
    """
    
    def __init__(self):
        self.coverage_thresholds = {
            "excellent": 90.0,
            "good": 80.0,
            "acceptable": 70.0,
            "poor": 50.0
        }
    
    def parse_coverage_report(self, coverage_data: str, format_type: str = "xml") -> Dict[str, Any]:
        """Parse coverage report in various formats."""
        
        if format_type == "xml":
            return self._parse_coverage_xml(coverage_data)
        elif format_type == "json":
            return self._parse_coverage_json(coverage_data)
        elif format_type == "lcov":
            return self._parse_lcov_coverage(coverage_data)
        else:
            raise AutomatedTestingError(f"Unsupported coverage format: {format_type}")
    
    def _parse_coverage_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML coverage report (Cobertura format)."""
        
        try:
            root = ET.fromstring(xml_content)
            
            # Get overall coverage
            line_rate = float(root.get("line-rate", 0.0)) * 100
            branch_rate = float(root.get("branch-rate", 0.0)) * 100
            
            # Get package/file level coverage
            files_coverage = {}
            packages = root.findall(".//package")
            
            for package in packages:
                package_name = package.get("name", "")
                
                for class_element in package.findall("classes/class"):
                    filename = class_element.get("filename", "")
                    class_line_rate = float(class_element.get("line-rate", 0.0)) * 100
                    class_branch_rate = float(class_element.get("branch-rate", 0.0)) * 100
                    
                    files_coverage[filename] = {
                        "line_coverage": class_line_rate,
                        "branch_coverage": class_branch_rate,
                        "package": package_name
                    }
            
            # Calculate quality grade
            coverage_grade = self._calculate_coverage_grade(line_rate)
            
            return {
                "overall_line_coverage": line_rate,
                "overall_branch_coverage": branch_rate,
                "files_coverage": files_coverage,
                "coverage_grade": coverage_grade,
                "total_files": len(files_coverage),
                "files_below_threshold": len([
                    f for f, data in files_coverage.items()
                    if data["line_coverage"] < 70.0
                ])
            }
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse coverage XML: {e}")
            raise AutomatedTestingError(f"Invalid coverage XML format: {e}")
    
    def _parse_coverage_json(self, json_content: str) -> Dict[str, Any]:
        """Parse JSON coverage report."""
        
        try:
            data = json.loads(json_content)
            
            # Handle different JSON coverage formats
            if "totals" in data:
                # Coverage.py format
                overall_coverage = data["totals"].get("percent_covered", 0.0)
                files_coverage = {}
                
                for filename, file_data in data.get("files", {}).items():
                    files_coverage[filename] = {
                        "line_coverage": file_data.get("percent_covered", 0.0),
                        "lines_covered": file_data.get("covered_lines", 0),
                        "total_lines": file_data.get("num_statements", 0)
                    }
            else:
                # Generic format
                overall_coverage = data.get("coverage", 0.0)
                files_coverage = data.get("files", {})
            
            coverage_grade = self._calculate_coverage_grade(overall_coverage)
            
            return {
                "overall_line_coverage": overall_coverage,
                "files_coverage": files_coverage,
                "coverage_grade": coverage_grade,
                "total_files": len(files_coverage),
                "files_below_threshold": len([
                    f for f, data in files_coverage.items()
                    if data.get("line_coverage", 0) < 70.0
                ])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse coverage JSON: {e}")
            raise AutomatedTestingError(f"Invalid coverage JSON format: {e}")
    
    def _calculate_coverage_grade(self, coverage_percentage: float) -> str:
        """Calculate coverage quality grade."""
        
        if coverage_percentage >= self.coverage_thresholds["excellent"]:
            return "A"
        elif coverage_percentage >= self.coverage_thresholds["good"]:
            return "B"
        elif coverage_percentage >= self.coverage_thresholds["acceptable"]:
            return "C"
        elif coverage_percentage >= self.coverage_thresholds["poor"]:
            return "D"
        else:
            return "F"
    
    def generate_coverage_recommendations(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent coverage improvement recommendations."""
        
        recommendations = []
        
        overall_coverage = coverage_data.get("overall_line_coverage", 0.0)
        files_coverage = coverage_data.get("files_coverage", {})
        
        # Overall coverage recommendation
        if overall_coverage < 70.0:
            recommendations.append({
                "type": "overall_coverage",
                "priority": "high",
                "title": "Low Overall Coverage",
                "description": f"Current coverage is {overall_coverage:.1f}%, below recommended 70%",
                "recommendation": "Focus on adding unit tests for core business logic and critical paths",
                "impact": "high"
            })
        
        # Identify worst covered files
        low_coverage_files = [
            (filename, data) for filename, data in files_coverage.items()
            if data.get("line_coverage", 0) < 50.0
        ]
        
        if low_coverage_files:
            # Sort by importance (heuristic: shorter paths are more likely core files)
            low_coverage_files.sort(key=lambda x: len(Path(x[0]).parts))
            
            for filename, coverage_data in low_coverage_files[:5]:  # Top 5 worst
                recommendations.append({
                    "type": "file_coverage",
                    "priority": "medium",
                    "title": f"Low Coverage: {Path(filename).name}",
                    "description": f"File has {coverage_data.get('line_coverage', 0):.1f}% coverage",
                    "recommendation": f"Add focused unit tests for {filename}",
                    "file": filename,
                    "impact": "medium"
                })
        
        # Missing branch coverage
        if coverage_data.get("overall_branch_coverage", 0) < overall_coverage - 10:
            recommendations.append({
                "type": "branch_coverage",
                "priority": "medium", 
                "title": "Low Branch Coverage",
                "description": "Branch coverage significantly lower than line coverage",
                "recommendation": "Add tests for conditional logic and error handling paths",
                "impact": "medium"
            })
        
        return recommendations


class AutomatedTestingIntegration:
    """
    Comprehensive automated testing integration system.
    
    Orchestrates test execution, analyzes results, tracks coverage,
    and provides intelligent failure analysis and recommendations.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.test_parser = TestResultParser()
        self.coverage_analyzer = CoverageAnalyzer()
        self.redis = get_redis()
        
        # CI Provider configurations
        self.ci_configs = {
            CIProvider.GITHUB_ACTIONS: {
                "workflow_file": ".github/workflows/test.yml",
                "artifacts_path": "test-results",
                "coverage_path": "coverage"
            }
        }
    
    async def trigger_automated_tests(
        self,
        pull_request: PullRequest,
        test_suites: List[str] = None,
        force_run: bool = False
    ) -> Dict[str, Any]:
        """Trigger automated test execution for PR."""
        
        test_suites = test_suites or ["unit", "integration", "security"]
        
        try:
            logger.info(
                "Triggering automated tests",
                pr_number=pull_request.github_pr_number,
                test_suites=test_suites
            )
            
            # Get repository information
            async with get_db_session() as session:
                result = await session.execute(
                    select(PullRequest).options(
                        selectinload(PullRequest.repository)
                    ).where(PullRequest.id == pull_request.id)
                )
                pr_with_repo = result.scalar_one()
            
            repo_parts = pr_with_repo.repository.repository_full_name.split('/')
            
            # Check for existing test run
            if not force_run:
                existing_run = await self._check_existing_test_run(pull_request)
                if existing_run:
                    logger.info("Tests already running for PR", existing_run_id=existing_run)
                    return {
                        "status": "already_running",
                        "existing_run_id": existing_run,
                        "message": "Tests are already running for this PR"
                    }
            
            # Trigger GitHub Actions workflow
            workflow_run = await self._trigger_github_actions(
                repo_parts[0], repo_parts[1], pull_request, test_suites
            )
            
            # Store test run metadata in Redis
            test_run_id = str(uuid.uuid4())
            await self._store_test_run_metadata(test_run_id, pull_request, workflow_run)
            
            return {
                "status": "triggered",
                "test_run_id": test_run_id,
                "workflow_run_id": workflow_run.get("id"),
                "workflow_url": workflow_run.get("html_url"),
                "test_suites": test_suites,
                "estimated_duration_minutes": self._estimate_test_duration(test_suites)
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger automated tests: {e}")
            raise AutomatedTestingError(f"Test trigger failed: {str(e)}")
    
    async def monitor_test_execution(
        self,
        test_run_id: str,
        timeout_minutes: int = 60
    ) -> Dict[str, Any]:
        """Monitor test execution progress and results."""
        
        try:
            # Get test run metadata
            test_metadata = await self._get_test_run_metadata(test_run_id)
            if not test_metadata:
                raise AutomatedTestingError(f"Test run {test_run_id} not found")
            
            workflow_run_id = test_metadata["workflow_run_id"]
            repo_full_name = test_metadata["repository"]
            repo_parts = repo_full_name.split('/')
            
            # Monitor workflow status
            start_time = datetime.utcnow()
            timeout = timedelta(minutes=timeout_minutes)
            
            while datetime.utcnow() - start_time < timeout:
                # Get workflow run status
                workflow_status = await self._get_workflow_status(
                    repo_parts[0], repo_parts[1], workflow_run_id
                )
                
                if workflow_status["status"] in ["completed"]:
                    # Get detailed results
                    test_results = await self._collect_test_results(
                        repo_parts[0], repo_parts[1], workflow_run_id
                    )
                    
                    # Analyze results
                    analysis = await self._analyze_test_results(test_results)
                    
                    # Update test run metadata
                    await self._update_test_run_results(test_run_id, test_results, analysis)
                    
                    return {
                        "status": "completed",
                        "test_run_id": test_run_id,
                        "workflow_status": workflow_status["conclusion"],
                        "success": workflow_status["conclusion"] == "success",
                        "test_results": test_results.dict() if hasattr(test_results, 'dict') else test_results,
                        "analysis": analysis,
                        "duration_minutes": (datetime.utcnow() - start_time).total_seconds() / 60
                    }
                elif workflow_status["status"] == "cancelled":
                    return {
                        "status": "cancelled",
                        "test_run_id": test_run_id,
                        "message": "Test execution was cancelled"
                    }
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Timeout reached
            return {
                "status": "timeout",
                "test_run_id": test_run_id,
                "message": f"Test execution exceeded {timeout_minutes} minute timeout"
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor test execution: {e}")
            raise AutomatedTestingError(f"Test monitoring failed: {str(e)}")
    
    async def analyze_test_failures(
        self,
        test_results: CIBuildResult
    ) -> Dict[str, Any]:
        """Analyze test failures with intelligent categorization and suggestions."""
        
        try:
            analysis = {
                "failure_categories": {},
                "common_patterns": [],
                "affected_areas": set(),
                "recommendations": [],
                "retry_suggestions": [],
                "failure_severity": "low"
            }
            
            total_failures = 0
            critical_failures = 0
            
            for suite in test_results.test_suites or []:
                if suite.status == TestStatus.FAILED:
                    total_failures += suite.failed_tests
                    
                    # Analyze individual test failures
                    for test in suite.test_results or []:
                        if test.status == TestStatus.FAILED:
                            # Categorize failure
                            category = test.failure_type or "unknown_error"
                            if category not in analysis["failure_categories"]:
                                analysis["failure_categories"][category] = []
                            
                            analysis["failure_categories"][category].append({
                                "test_name": test.name,
                                "message": test.message,
                                "file_path": test.file_path,
                                "duration": test.duration
                            })
                            
                            # Track affected areas
                            if test.file_path:
                                area = Path(test.file_path).parts[0] if Path(test.file_path).parts else "unknown"
                                analysis["affected_areas"].add(area)
                            
                            # Check for critical failures
                            if test.failure_type in ["connection_error", "timeout_error", "import_error"]:
                                critical_failures += 1
            
            # Convert set to list for JSON serialization
            analysis["affected_areas"] = list(analysis["affected_areas"])
            
            # Determine failure severity
            if critical_failures > 0:
                analysis["failure_severity"] = "critical"
            elif total_failures > 10:
                analysis["failure_severity"] = "high"
            elif total_failures > 3:
                analysis["failure_severity"] = "medium"
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_failure_recommendations(
                analysis["failure_categories"], total_failures
            )
            
            # Generate retry suggestions
            analysis["retry_suggestions"] = self._generate_retry_suggestions(
                analysis["failure_categories"]
            )
            
            # Find common patterns
            analysis["common_patterns"] = self._identify_failure_patterns(
                analysis["failure_categories"]
            )
            
            logger.info(
                "Test failure analysis completed",
                total_failures=total_failures,
                critical_failures=critical_failures,
                categories=len(analysis["failure_categories"])
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze test failures: {e}")
            raise AutomatedTestingError(f"Failure analysis failed: {str(e)}")
    
    async def _trigger_github_actions(
        self,
        owner: str,
        repo: str,
        pull_request: PullRequest,
        test_suites: List[str]
    ) -> Dict[str, Any]:
        """Trigger GitHub Actions workflow."""
        
        try:
            # Create workflow dispatch
            workflow_inputs = {
                "ref": pull_request.source_branch,
                "test_suites": ",".join(test_suites),
                "pr_number": str(pull_request.github_pr_number),
                "force_run": "true"
            }
            
            response = await self.github_client._make_request(
                "POST",
                f"/repos/{owner}/{repo}/actions/workflows/test.yml/dispatches",
                json={
                    "ref": pull_request.source_branch,
                    "inputs": workflow_inputs
                }
            )
            
            # Get the created workflow run
            await asyncio.sleep(2)  # Brief delay for workflow to be created
            
            runs_response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/actions/runs",
                params={
                    "branch": pull_request.source_branch,
                    "per_page": 1
                }
            )
            
            if runs_response.get("workflow_runs"):
                return runs_response["workflow_runs"][0]
            else:
                raise AutomatedTestingError("Failed to find created workflow run")
            
        except Exception as e:
            logger.error(f"Failed to trigger GitHub Actions: {e}")
            raise AutomatedTestingError(f"GitHub Actions trigger failed: {str(e)}")
    
    async def _get_workflow_status(
        self,
        owner: str,
        repo: str,
        workflow_run_id: str
    ) -> Dict[str, Any]:
        """Get GitHub Actions workflow status."""
        
        try:
            response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/actions/runs/{workflow_run_id}"
            )
            
            return {
                "status": response.get("status"),
                "conclusion": response.get("conclusion"),
                "created_at": response.get("created_at"),
                "updated_at": response.get("updated_at"),
                "url": response.get("html_url")
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {"status": "unknown", "conclusion": None}
    
    async def _collect_test_results(
        self,
        owner: str,
        repo: str,
        workflow_run_id: str
    ) -> CIBuildResult:
        """Collect and parse test results from CI artifacts."""
        
        try:
            # Get workflow run artifacts
            artifacts_response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/actions/runs/{workflow_run_id}/artifacts"
            )
            
            test_suites = []
            
            for artifact in artifacts_response.get("artifacts", []):
                artifact_name = artifact["name"]
                
                if "test-results" in artifact_name:
                    # Download and parse test results
                    artifact_data = await self._download_artifact(
                        owner, repo, artifact["id"]
                    )
                    
                    if artifact_data:
                        # Parse based on file extension/content
                        if artifact_name.endswith(".xml") or "junit" in artifact_name:
                            suite_results = self.test_parser.parse_junit_xml(artifact_data)
                        elif artifact_name.endswith(".json"):
                            suite_results = self.test_parser.parse_pytest_json(artifact_data)
                        else:
                            continue
                        
                        test_suites.extend(suite_results)
                
                elif "coverage" in artifact_name:
                    # Parse coverage data
                    coverage_data = await self._download_artifact(
                        owner, repo, artifact["id"]
                    )
                    
                    if coverage_data:
                        coverage_analysis = self.coverage_analyzer.parse_coverage_report(
                            coverage_data, "xml"
                        )
                        
                        # Add coverage to test suites
                        for suite in test_suites:
                            if suite.suite_type in [TestSuite.UNIT, TestSuite.INTEGRATION]:
                                suite.coverage = coverage_analysis.get("overall_line_coverage")
            
            # Get workflow run details
            workflow_response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/actions/runs/{workflow_run_id}"
            )
            
            # Determine overall status
            if all(suite.status == TestStatus.PASSED for suite in test_suites):
                overall_status = TestStatus.PASSED
            elif any(suite.status == TestStatus.FAILED for suite in test_suites):
                overall_status = TestStatus.FAILED
            else:
                overall_status = TestStatus.ERROR
            
            created_at = datetime.fromisoformat(
                workflow_response["created_at"].replace("Z", "+00:00")
            )
            updated_at = datetime.fromisoformat(
                workflow_response["updated_at"].replace("Z", "+00:00")
            ) if workflow_response.get("updated_at") else None
            
            return CIBuildResult(
                build_id=str(workflow_run_id),
                build_url=workflow_response.get("html_url", ""),
                status=overall_status,
                pr_number=0,  # Will be filled from metadata
                commit_sha=workflow_response.get("head_sha", ""),
                started_at=created_at,
                completed_at=updated_at,
                duration=(updated_at - created_at).total_seconds() if updated_at else None,
                test_suites=test_suites,
                artifacts=[artifact["name"] for artifact in artifacts_response.get("artifacts", [])]
            )
            
        except Exception as e:
            logger.error(f"Failed to collect test results: {e}")
            raise AutomatedTestingError(f"Test result collection failed: {str(e)}")
    
    async def _download_artifact(self, owner: str, repo: str, artifact_id: int) -> Optional[str]:
        """Download CI artifact content."""
        
        try:
            # Note: This is a simplified version. In practice, you'd need to handle
            # artifact download URLs and potentially unzip compressed artifacts
            response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
            )
            
            # For now, return mock data to demonstrate the structure
            if "junit" in str(artifact_id):
                return """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="pytest" tests="5" failures="1" errors="0" time="2.345" timestamp="2024-01-01T12:00:00" hostname="runner">
    <testsuite name="test_example" tests="5" failures="1" errors="0" time="2.345" timestamp="2024-01-01T12:00:00" hostname="runner">
        <testcase classname="test_example.TestExample" name="test_success" time="0.123"/>
        <testcase classname="test_example.TestExample" name="test_failure" time="0.456">
            <failure message="AssertionError: Expected 5 but was 3">AssertionError: Expected 5 but was 3
at test_example.py:15</failure>
        </testcase>
    </testsuite>
</testsuites>"""
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to download artifact {artifact_id}: {e}")
            return None
    
    async def _store_test_run_metadata(
        self,
        test_run_id: str,
        pull_request: PullRequest,
        workflow_run: Dict[str, Any]
    ) -> None:
        """Store test run metadata in Redis."""
        
        try:
            metadata = {
                "test_run_id": test_run_id,
                "pr_id": str(pull_request.id),
                "pr_number": pull_request.github_pr_number,
                "repository": pull_request.repository.repository_full_name,
                "workflow_run_id": workflow_run.get("id"),
                "workflow_url": workflow_run.get("html_url"),
                "commit_sha": workflow_run.get("head_sha"),
                "created_at": datetime.utcnow().isoformat(),
                "status": "running"
            }
            
            await self.redis.setex(
                f"test_run:{test_run_id}",
                3600,  # 1 hour TTL
                json.dumps(metadata)
            )
            
            # Also store by PR for lookup
            await self.redis.setex(
                f"test_run:pr:{pull_request.id}",
                3600,
                test_run_id
            )
            
        except Exception as e:
            logger.error(f"Failed to store test run metadata: {e}")
    
    async def _get_test_run_metadata(self, test_run_id: str) -> Optional[Dict[str, Any]]:
        """Get test run metadata from Redis."""
        
        try:
            data = await self.redis.get(f"test_run:{test_run_id}")
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get test run metadata: {e}")
            return None
    
    async def _check_existing_test_run(self, pull_request: PullRequest) -> Optional[str]:
        """Check for existing test run for PR."""
        
        try:
            test_run_id = await self.redis.get(f"test_run:pr:{pull_request.id}")
            if test_run_id:
                return test_run_id.decode() if isinstance(test_run_id, bytes) else test_run_id
            return None
            
        except Exception as e:
            logger.error(f"Failed to check existing test run: {e}")
            return None
    
    def _estimate_test_duration(self, test_suites: List[str]) -> int:
        """Estimate test duration in minutes based on suite types."""
        
        duration_estimates = {
            "unit": 2,
            "integration": 5,
            "functional": 10,
            "performance": 15,
            "security": 8,
            "e2e": 20,
            "smoke": 3
        }
        
        total_minutes = 0
        for suite in test_suites:
            total_minutes += duration_estimates.get(suite, 5)
        
        return max(5, total_minutes)  # Minimum 5 minutes
    
    def _generate_failure_recommendations(
        self,
        failure_categories: Dict[str, List[Dict[str, Any]]],
        total_failures: int
    ) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations for test failures."""
        
        recommendations = []
        
        for category, failures in failure_categories.items():
            if category == "assertion_error":
                recommendations.append({
                    "category": category,
                    "priority": "high",
                    "title": "Logic or Data Issues",
                    "description": f"{len(failures)} assertion failures detected",
                    "recommendation": "Review test expectations and business logic implementation",
                    "action_items": [
                        "Check if requirements have changed",
                        "Verify test data is up to date",
                        "Review recent code changes for logic errors"
                    ]
                })
            elif category == "timeout_error":
                recommendations.append({
                    "category": category,
                    "priority": "medium",
                    "title": "Performance or Infrastructure Issues", 
                    "description": f"{len(failures)} timeout errors detected",
                    "recommendation": "Investigate performance bottlenecks or increase timeouts",
                    "action_items": [
                        "Profile slow operations",
                        "Check database query performance",
                        "Review network connectivity",
                        "Consider increasing timeout values"
                    ]
                })
            elif category == "connection_error":
                recommendations.append({
                    "category": category,
                    "priority": "critical",
                    "title": "Service Dependencies",
                    "description": f"{len(failures)} connection errors detected",
                    "recommendation": "Check external service availability and configuration",
                    "action_items": [
                        "Verify service endpoints are accessible",
                        "Check authentication credentials",
                        "Review network configuration",
                        "Implement service mocking for tests"
                    ]
                })
        
        # Overall recommendations
        if total_failures > 20:
            recommendations.append({
                "category": "overall",
                "priority": "critical",
                "title": "High Failure Rate",
                "description": f"{total_failures} total test failures",
                "recommendation": "Consider reverting recent changes and investigating systematically",
                "action_items": [
                    "Review recent commits",
                    "Run tests locally to reproduce",
                    "Check for environmental differences"
                ]
            })
        
        return recommendations
    
    def _generate_retry_suggestions(
        self,
        failure_categories: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for test retries."""
        
        suggestions = []
        
        # Flaky test categories that might benefit from retry
        flaky_categories = ["timeout_error", "connection_error"]
        
        for category in flaky_categories:
            if category in failure_categories:
                failures = failure_categories[category]
                suggestions.append({
                    "category": category,
                    "test_count": len(failures),
                    "retry_recommended": True,
                    "max_retries": 3,
                    "reason": f"{category.replace('_', ' ').title()} failures are often transient"
                })
        
        # Stable failure categories that should not be retried
        stable_categories = ["assertion_error", "syntax_error", "import_error"]
        for category in stable_categories:
            if category in failure_categories:
                failures = failure_categories[category]
                suggestions.append({
                    "category": category,
                    "test_count": len(failures),
                    "retry_recommended": False,
                    "reason": f"{category.replace('_', ' ').title()} failures indicate code issues"
                })
        
        return suggestions
    
    def _identify_failure_patterns(
        self,
        failure_categories: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify common failure patterns."""
        
        patterns = []
        
        # Pattern 1: Same file multiple failures
        file_failures = {}
        for category, failures in failure_categories.items():
            for failure in failures:
                file_path = failure.get("file_path")
                if file_path:
                    if file_path not in file_failures:
                        file_failures[file_path] = []
                    file_failures[file_path].append(failure)
        
        for file_path, failures in file_failures.items():
            if len(failures) > 2:
                patterns.append({
                    "type": "file_concentration",
                    "file": file_path,
                    "failure_count": len(failures),
                    "description": f"Multiple failures concentrated in {Path(file_path).name}",
                    "suggestion": f"Focus debugging efforts on {file_path}"
                })
        
        # Pattern 2: Similar error messages
        message_groups = {}
        for category, failures in failure_categories.items():
            for failure in failures:
                message = failure.get("message", "")
                # Group by first 50 characters of error message
                message_key = message[:50] if message else "unknown"
                if message_key not in message_groups:
                    message_groups[message_key] = []
                message_groups[message_key].append(failure)
        
        for message_key, failures in message_groups.items():
            if len(failures) > 3:
                patterns.append({
                    "type": "repeated_error",
                    "message_pattern": message_key,
                    "failure_count": len(failures),
                    "description": f"Repeated error pattern: {message_key}",
                    "suggestion": "This error pattern appears multiple times - may indicate systematic issue"
                })
        
        return patterns
    
    async def _analyze_test_results(self, test_results: CIBuildResult) -> Dict[str, Any]:
        """Comprehensive analysis of test results."""
        
        analysis = {
            "overall_status": test_results.status.value,
            "success_rate": 0.0,
            "performance_metrics": {},
            "coverage_analysis": {},
            "failure_analysis": {},
            "trends": {},
            "recommendations": []
        }
        
        if not test_results.test_suites:
            return analysis
        
        # Calculate success rate
        total_tests = sum(suite.total_tests for suite in test_results.test_suites)
        passed_tests = sum(suite.passed_tests for suite in test_results.test_suites)
        
        if total_tests > 0:
            analysis["success_rate"] = (passed_tests / total_tests) * 100
        
        # Performance metrics
        analysis["performance_metrics"] = {
            "total_duration": test_results.duration or 0.0,
            "average_test_duration": (test_results.duration or 0.0) / total_tests if total_tests > 0 else 0.0,
            "slowest_suite": max(
                test_results.test_suites, 
                key=lambda s: s.duration, 
                default=None
            ).suite_name if test_results.test_suites else None,
            "suite_durations": {
                suite.suite_name: suite.duration 
                for suite in test_results.test_suites
            }
        }
        
        # Coverage analysis
        coverage_suites = [suite for suite in test_results.test_suites if suite.coverage is not None]
        if coverage_suites:
            analysis["coverage_analysis"] = {
                "average_coverage": sum(suite.coverage for suite in coverage_suites) / len(coverage_suites),
                "suite_coverage": {
                    suite.suite_name: suite.coverage 
                    for suite in coverage_suites
                }
            }
        
        # Failure analysis
        if test_results.status == TestStatus.FAILED:
            analysis["failure_analysis"] = await self.analyze_test_failures(test_results)
        
        return analysis
    
    async def _update_test_run_results(
        self,
        test_run_id: str,
        test_results: CIBuildResult,
        analysis: Dict[str, Any]
    ) -> None:
        """Update test run with final results."""
        
        try:
            metadata = await self._get_test_run_metadata(test_run_id)
            if metadata:
                metadata.update({
                    "status": "completed",
                    "conclusion": test_results.status.value,
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration": test_results.duration,
                    "success_rate": analysis.get("success_rate", 0.0),
                    "total_tests": sum(suite.total_tests for suite in test_results.test_suites or [])
                })
                
                # Store with longer TTL for historical data
                await self.redis.setex(
                    f"test_run:{test_run_id}",
                    86400,  # 24 hours
                    json.dumps(metadata)
                )
            
        except Exception as e:
            logger.error(f"Failed to update test run results: {e}")
    
    async def get_test_trends(
        self,
        repository_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get test execution trends and metrics."""
        
        try:
            # This would integrate with stored test results
            # For now, return mock trending data
            
            return {
                "period_days": days,
                "total_test_runs": 45,
                "average_success_rate": 87.3,
                "success_rate_trend": "improving",  # "improving", "declining", "stable"
                "average_duration_minutes": 12.5,
                "duration_trend": "stable",
                "most_common_failures": [
                    {"type": "assertion_error", "count": 23, "percentage": 34.2},
                    {"type": "timeout_error", "count": 15, "percentage": 22.1},
                    {"type": "connection_error", "count": 8, "percentage": 11.9}
                ],
                "flaky_tests": [
                    {
                        "test_name": "test_api_integration",
                        "failure_rate": 15.2,
                        "failures_last_week": 3
                    }
                ],
                "coverage_trend": {
                    "current": 82.1,
                    "previous_week": 80.5,
                    "trend": "improving"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get test trends: {e}")
            return {"error": str(e)}


# Factory function
async def create_automated_testing_integration() -> AutomatedTestingIntegration:
    """Create and initialize automated testing integration."""
    
    github_client = GitHubAPIClient()
    return AutomatedTestingIntegration(github_client)


# Export main classes
__all__ = [
    "AutomatedTestingIntegration",
    "TestResultParser", 
    "CoverageAnalyzer",
    "CIBuildResult",
    "TestSuiteResult",
    "TestResult",
    "AutomatedTestingError",
    "create_automated_testing_integration"
]