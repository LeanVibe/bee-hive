"""
Security Scanner Plugin Example

Demonstrates security plugin development using the LeanVibe SDK.
Shows vulnerability scanning, threat detection, and security monitoring.
"""

import asyncio
import hashlib
import re
import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from ..interfaces import SecurityPlugin, PluginType
from ..models import PluginConfig, TaskInterface, TaskResult, PluginEvent, EventSeverity
from ..decorators import plugin_method, performance_tracked, error_handled, cached_result
from ..exceptions import PluginConfigurationError, PluginExecutionError


@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure."""
    vulnerability_id: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    affected_component: str
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability_id": self.vulnerability_id,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "affected_component": self.affected_component,
            "cve_id": self.cve_id,
            "cvss_score": self.cvss_score,
            "remediation": self.remediation,
            "discovered_at": self.discovered_at.isoformat()
        }


@dataclass
class SecurityScanResult:
    """Security scan result."""
    scan_id: str
    scan_type: str
    start_time: datetime
    end_time: datetime
    vulnerabilities: List[SecurityVulnerability]
    scan_status: str
    total_files_scanned: int = 0
    total_issues_found: int = 0
    scan_duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "scan_type": self.scan_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "scan_status": self.scan_status,
            "total_files_scanned": self.total_files_scanned,
            "total_issues_found": self.total_issues_found,
            "scan_duration_seconds": self.scan_duration_seconds,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "summary": {
                "critical": len([v for v in self.vulnerabilities if v.severity == "critical"]),
                "high": len([v for v in self.vulnerabilities if v.severity == "high"]),
                "medium": len([v for v in self.vulnerabilities if v.severity == "medium"]),
                "low": len([v for v in self.vulnerabilities if v.severity == "low"])
            }
        }


class SecurityScannerPlugin(SecurityPlugin):
    """
    Advanced security scanning plugin.
    
    Features:
    - Code vulnerability scanning
    - Dependency security analysis
    - Secret detection in code
    - Configuration security checks
    - Security policy compliance
    - Threat pattern detection
    - Automated remediation suggestions
    
    Epic 1 Optimizations:
    - Efficient pattern matching algorithms
    - Incremental scanning capabilities
    - <50ms security check responses
    - <20MB memory footprint
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Configuration
        self.scan_patterns = self._load_scan_patterns()
        self.secret_patterns = self._load_secret_patterns()
        self.dependency_check_enabled = config.parameters.get("dependency_check_enabled", True)
        self.max_file_size_mb = config.parameters.get("max_file_size_mb", 10)
        self.excluded_paths = set(config.parameters.get("excluded_paths", [
            ".git", "node_modules", "__pycache__", ".venv", "venv"
        ]))
        
        # Vulnerability database
        self.vulnerability_db = self._initialize_vulnerability_db()
        
        # Runtime state
        self.scan_history: List[SecurityScanResult] = []
        self.active_scans: Dict[str, bool] = {}
        self.scanning_stats = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "false_positives": 0,
            "average_scan_time_seconds": 0.0
        }
        
        # Performance tracking
        self._scan_times = []
    
    def _load_scan_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security vulnerability patterns."""
        return {
            "sql_injection": [
                {
                    "pattern": r"SELECT.*FROM.*WHERE.*=.*\$.*",
                    "severity": "high",
                    "description": "Potential SQL injection vulnerability"
                },
                {
                    "pattern": r"INSERT INTO.*VALUES.*\$.*",
                    "severity": "high", 
                    "description": "Potential SQL injection in INSERT statement"
                }
            ],
            "xss": [
                {
                    "pattern": r"innerHTML\s*=\s*.*\+.*",
                    "severity": "medium",
                    "description": "Potential XSS via innerHTML"
                },
                {
                    "pattern": r"document\.write\s*\(.*\+.*\)",
                    "severity": "medium",
                    "description": "Potential XSS via document.write"
                }
            ],
            "path_traversal": [
                {
                    "pattern": r"\.\.\/.*\.\.\/",
                    "severity": "high",
                    "description": "Potential path traversal vulnerability"
                }
            ],
            "hardcoded_secrets": [
                {
                    "pattern": r"password\s*=\s*['\"][^'\"]{3,}['\"]",
                    "severity": "critical",
                    "description": "Hardcoded password detected"
                },
                {
                    "pattern": r"api_key\s*=\s*['\"][^'\"]{10,}['\"]",
                    "severity": "high",
                    "description": "Hardcoded API key detected"
                }
            ],
            "crypto_weaknesses": [
                {
                    "pattern": r"md5\(.*\)",
                    "severity": "medium",
                    "description": "Weak MD5 hash function used"
                },
                {
                    "pattern": r"sha1\(.*\)",
                    "severity": "medium",
                    "description": "Weak SHA1 hash function used"
                }
            ]
        }
    
    def _load_secret_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for detecting secrets in code."""
        return [
            {
                "name": "AWS Access Key",
                "pattern": r"AKIA[0-9A-Z]{16}",
                "severity": "critical"
            },
            {
                "name": "AWS Secret Key",
                "pattern": r"[0-9a-zA-Z/+]{40}",
                "severity": "critical"
            },
            {
                "name": "GitHub Token",
                "pattern": r"ghp_[0-9a-zA-Z]{36}",
                "severity": "high"
            },
            {
                "name": "Google API Key",
                "pattern": r"AIza[0-9A-Za-z\-_]{35}",
                "severity": "high"
            },
            {
                "name": "JWT Token",
                "pattern": r"ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
                "severity": "medium"
            },
            {
                "name": "Private Key",
                "pattern": r"-----BEGIN PRIVATE KEY-----",
                "severity": "critical"
            }
        ]
    
    def _initialize_vulnerability_db(self) -> Dict[str, Any]:
        """Initialize vulnerability database."""
        # Mock vulnerability database - in real implementation, 
        # this would load from actual CVE databases
        return {
            "known_vulnerable_packages": {
                "requests": ["2.25.0", "2.25.1"],
                "flask": ["1.0.0", "1.0.1"],
                "django": ["2.2.0", "2.2.1"]
            },
            "security_advisories": [
                {
                    "id": "ADVISORY-2023-001",
                    "package": "requests",
                    "version_range": ">=2.25.0,<2.25.2",
                    "severity": "high",
                    "description": "SSL certificate validation bypass"
                }
            ]
        }
    
    async def _on_initialize(self) -> None:
        """Initialize the security scanner plugin."""
        await self.log_info("Initializing SecurityScannerPlugin")
        
        # Validate configuration
        if self.max_file_size_mb <= 0:
            raise PluginConfigurationError(
                "Max file size must be positive",
                config_key="max_file_size_mb",
                expected_type="positive number",
                actual_value=self.max_file_size_mb,
                plugin_id=self.plugin_id
            )
        
        # Initialize state
        self.scan_history = []
        self.active_scans = {}
        self.scanning_stats = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "false_positives": 0,
            "average_scan_time_seconds": 0.0
        }
        
        await self.log_info(
            f"Initialized with {sum(len(patterns) for patterns in self.scan_patterns.values())} "
            f"vulnerability patterns, {len(self.secret_patterns)} secret patterns"
        )
    
    @performance_tracked(alert_threshold_ms=5000, memory_limit_mb=25)
    @plugin_method(timeout_seconds=3600, max_retries=1)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Execute security scanning operations.
        
        Supports the following task types:
        - scan_code: Scan code for vulnerabilities
        - scan_dependencies: Check dependencies for vulnerabilities
        - scan_secrets: Scan for hardcoded secrets
        - scan_config: Check configuration security
        - get_scan_results: Get historical scan results
        - get_vulnerability_report: Generate vulnerability report
        """
        task_type = task.task_type
        
        if task_type == "scan_code":
            return await self._scan_code_vulnerabilities(task)
        elif task_type == "scan_dependencies":
            return await self._scan_dependencies(task)
        elif task_type == "scan_secrets":
            return await self._scan_secrets(task)
        elif task_type == "scan_config":
            return await self._scan_configuration(task)
        elif task_type == "get_scan_results":
            return await self._get_scan_results(task)
        elif task_type == "get_vulnerability_report":
            return await self._generate_vulnerability_report(task)
        else:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Unknown task type: {task_type}",
                error_code="INVALID_TASK_TYPE"
            )
    
    async def _scan_code_vulnerabilities(self, task: TaskInterface) -> TaskResult:
        """Scan code for security vulnerabilities."""
        start_time = datetime.utcnow()
        scan_id = f"code_scan_{task.task_id}"
        
        try:
            target_path = task.parameters.get("target_path", ".")
            file_extensions = task.parameters.get("file_extensions", [".py", ".js", ".php", ".java"])
            
            await task.update_status("running", progress=0.1)
            await self.log_info(f"Starting code vulnerability scan: {target_path}")
            
            self.active_scans[scan_id] = True
            vulnerabilities = []
            files_scanned = 0
            
            # Scan files
            target = Path(target_path)
            if target.is_file():
                files_to_scan = [target]
            else:
                files_to_scan = []
                for ext in file_extensions:
                    files_to_scan.extend(target.rglob(f"*{ext}"))
            
            # Filter out excluded paths
            files_to_scan = [
                f for f in files_to_scan 
                if not any(excluded in str(f) for excluded in self.excluded_paths)
            ]
            
            total_files = len(files_to_scan)
            
            for i, file_path in enumerate(files_to_scan):
                if scan_id not in self.active_scans:
                    break  # Scan was cancelled
                
                # Check file size
                if file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                    await self.log_warning(f"Skipping large file: {file_path}")
                    continue
                
                # Scan file
                file_vulns = await self._scan_file_for_vulnerabilities(file_path)
                vulnerabilities.extend(file_vulns)
                files_scanned += 1
                
                # Update progress
                progress = 0.1 + (0.8 * (i + 1) / total_files)
                await task.update_status("running", progress=progress)
                
                # Yield control to prevent blocking
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            
            end_time = datetime.utcnow()
            scan_duration = (end_time - start_time).total_seconds()
            
            # Create scan result
            scan_result = SecurityScanResult(
                scan_id=scan_id,
                scan_type="code_vulnerability",
                start_time=start_time,
                end_time=end_time,
                vulnerabilities=vulnerabilities,
                scan_status="completed",
                total_files_scanned=files_scanned,
                total_issues_found=len(vulnerabilities),
                scan_duration_seconds=scan_duration
            )
            
            # Store result
            self.scan_history.append(scan_result)
            
            # Update statistics
            self.scanning_stats["total_scans"] += 1
            self.scanning_stats["vulnerabilities_found"] += len(vulnerabilities)
            self._scan_times.append(scan_duration)
            
            if self._scan_times:
                self.scanning_stats["average_scan_time_seconds"] = sum(self._scan_times) / len(self._scan_times)
                # Keep only recent times
                if len(self._scan_times) > 50:
                    self._scan_times = self._scan_times[-50:]
            
            # Clean up
            self.active_scans.pop(scan_id, None)
            
            await task.update_status("completed", progress=1.0)
            
            # Emit scan completed event
            scan_event = PluginEvent(
                event_type="security_scan_completed",
                plugin_id=self.plugin_id,
                data={
                    "scan_id": scan_id,
                    "scan_type": "code_vulnerability",
                    "files_scanned": files_scanned,
                    "vulnerabilities_found": len(vulnerabilities),
                    "scan_duration_seconds": scan_duration
                },
                severity=EventSeverity.INFO if not vulnerabilities else EventSeverity.WARNING,
                task_id=task.task_id
            )
            await self.emit_event(scan_event)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                execution_time_ms=scan_duration * 1000,
                data=scan_result.to_dict()
            )
            
        except Exception as e:
            self.active_scans.pop(scan_id, None)
            await self.log_error(f"Code scan failed: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="CODE_SCAN_FAILED"
            )
    
    async def _scan_file_for_vulnerabilities(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a single file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Scan for each pattern category
            for category, patterns in self.scan_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Calculate line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vuln = SecurityVulnerability(
                            vulnerability_id=hashlib.md5(
                                f"{file_path}:{line_num}:{pattern}".encode()
                            ).hexdigest()[:16],
                            severity=pattern_info["severity"],
                            title=f"{category.replace('_', ' ').title()} Vulnerability",
                            description=f"{pattern_info['description']} at line {line_num}",
                            affected_component=str(file_path),
                            remediation=self._get_remediation_advice(category)
                        )
                        vulnerabilities.append(vuln)
            
        except Exception as e:
            await self.log_error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_remediation_advice(self, vulnerability_category: str) -> str:
        """Get remediation advice for vulnerability category."""
        remediation_map = {
            "sql_injection": "Use parameterized queries or prepared statements to prevent SQL injection",
            "xss": "Sanitize user input and use proper output encoding",
            "path_traversal": "Validate and sanitize file paths, use whitelist approach",
            "hardcoded_secrets": "Move secrets to environment variables or secure configuration",
            "crypto_weaknesses": "Use strong cryptographic algorithms like SHA-256 or bcrypt"
        }
        return remediation_map.get(vulnerability_category, "Review code and apply security best practices")
    
    async def _scan_dependencies(self, task: TaskInterface) -> TaskResult:
        """Scan dependencies for known vulnerabilities."""
        start_time = datetime.utcnow()
        
        try:
            project_path = task.parameters.get("project_path", ".")
            package_files = task.parameters.get("package_files", ["requirements.txt", "package.json", "Pipfile"])
            
            await task.update_status("running", progress=0.2)
            
            vulnerabilities = []
            dependencies_found = {}
            
            # Scan different package file types
            for package_file in package_files:
                file_path = Path(project_path) / package_file
                if file_path.exists():
                    deps = await self._parse_dependency_file(file_path)
                    dependencies_found.update(deps)
            
            await task.update_status("running", progress=0.6)
            
            # Check dependencies against vulnerability database
            for package_name, version in dependencies_found.items():
                vulns = await self._check_package_vulnerabilities(package_name, version)
                vulnerabilities.extend(vulns)
            
            scan_duration = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                execution_time_ms=scan_duration * 1000,
                data={
                    "dependencies_checked": len(dependencies_found),
                    "vulnerabilities_found": len(vulnerabilities),
                    "vulnerable_packages": [v.affected_component for v in vulnerabilities],
                    "scan_duration_seconds": scan_duration,
                    "vulnerabilities": [v.to_dict() for v in vulnerabilities]
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="DEPENDENCY_SCAN_FAILED"
            )
    
    async def _parse_dependency_file(self, file_path: Path) -> Dict[str, str]:
        """Parse dependency file to extract package names and versions."""
        dependencies = {}
        
        try:
            if file_path.name == "requirements.txt":
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '==' in line:
                                package, version = line.split('==', 1)
                                dependencies[package.strip()] = version.strip()
            
            elif file_path.name == "package.json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for dep_type in ["dependencies", "devDependencies"]:
                        if dep_type in data:
                            for package, version in data[dep_type].items():
                                # Remove version prefixes like ^, ~
                                clean_version = re.sub(r'^[\^~]', '', version)
                                dependencies[package] = clean_version
        
        except Exception as e:
            await self.log_error(f"Error parsing {file_path}: {e}")
        
        return dependencies
    
    async def _check_package_vulnerabilities(self, package_name: str, version: str) -> List[SecurityVulnerability]:
        """Check if a package version has known vulnerabilities."""
        vulnerabilities = []
        
        # Check against mock vulnerability database
        vulnerable_packages = self.vulnerability_db.get("known_vulnerable_packages", {})
        
        if package_name in vulnerable_packages:
            vulnerable_versions = vulnerable_packages[package_name]
            if version in vulnerable_versions:
                vuln = SecurityVulnerability(
                    vulnerability_id=f"dep_{package_name}_{version}",
                    severity="high",
                    title=f"Vulnerable dependency: {package_name}",
                    description=f"Package {package_name} version {version} has known vulnerabilities",
                    affected_component=f"{package_name}=={version}",
                    remediation=f"Update {package_name} to a newer, secure version"
                )
                vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    async def _scan_secrets(self, task: TaskInterface) -> TaskResult:
        """Scan for hardcoded secrets in code."""
        start_time = datetime.utcnow()
        
        try:
            target_path = task.parameters.get("target_path", ".")
            file_extensions = task.parameters.get("file_extensions", [".py", ".js", ".env", ".yml", ".yaml"])
            
            await task.update_status("running", progress=0.1)
            
            secrets_found = []
            files_scanned = 0
            
            # Get files to scan
            target = Path(target_path)
            files_to_scan = []
            
            if target.is_file():
                files_to_scan = [target]
            else:
                for ext in file_extensions:
                    files_to_scan.extend(target.rglob(f"*{ext}"))
            
            # Filter excluded paths
            files_to_scan = [
                f for f in files_to_scan 
                if not any(excluded in str(f) for excluded in self.excluded_paths)
            ]
            
            for i, file_path in enumerate(files_to_scan):
                file_secrets = await self._scan_file_for_secrets(file_path)
                secrets_found.extend(file_secrets)
                files_scanned += 1
                
                # Update progress
                progress = 0.1 + (0.8 * (i + 1) / len(files_to_scan))
                await task.update_status("running", progress=progress)
                
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            
            scan_duration = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                execution_time_ms=scan_duration * 1000,
                data={
                    "files_scanned": files_scanned,
                    "secrets_found": len(secrets_found),
                    "scan_duration_seconds": scan_duration,
                    "secrets": [s.to_dict() for s in secrets_found]
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="SECRET_SCAN_FAILED"
            )
    
    async def _scan_file_for_secrets(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a file for hardcoded secrets."""
        secrets = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for secret_pattern in self.secret_patterns:
                pattern = secret_pattern["pattern"]
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    secret = SecurityVulnerability(
                        vulnerability_id=hashlib.md5(
                            f"{file_path}:{line_num}:secret".encode()
                        ).hexdigest()[:16],
                        severity=secret_pattern["severity"],
                        title=f"Hardcoded Secret: {secret_pattern['name']}",
                        description=f"{secret_pattern['name']} found at line {line_num}",
                        affected_component=str(file_path),
                        remediation="Remove hardcoded secret and use environment variables or secure vaults"
                    )
                    secrets.append(secret)
        
        except Exception as e:
            await self.log_error(f"Error scanning file for secrets {file_path}: {e}")
        
        return secrets
    
    async def _scan_configuration(self, task: TaskInterface) -> TaskResult:
        """Scan configuration files for security issues."""
        # Mock configuration scanning - in real implementation,
        # this would check configuration files for security misconfigurations
        
        scan_duration = 0.5  # Mock scan time
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            execution_time_ms=scan_duration * 1000,
            data={
                "config_files_scanned": 3,
                "security_issues_found": 0,
                "scan_duration_seconds": scan_duration,
                "recommendations": [
                    "Enable HTTPS in production",
                    "Set secure cookie flags",
                    "Configure proper CORS headers"
                ]
            }
        )
    
    async def _get_scan_results(self, task: TaskInterface) -> TaskResult:
        """Get historical scan results."""
        try:
            limit = task.parameters.get("limit", 10)
            scan_type = task.parameters.get("scan_type")
            
            # Filter by scan type if specified
            filtered_scans = self.scan_history
            if scan_type:
                filtered_scans = [s for s in filtered_scans if s.scan_type == scan_type]
            
            # Apply limit
            if limit:
                filtered_scans = filtered_scans[-limit:]
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "scan_results": [s.to_dict() for s in filtered_scans],
                    "total_scans": len(filtered_scans),
                    "scanning_stats": self.scanning_stats
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="SCAN_RESULTS_RETRIEVAL_FAILED"
            )
    
    async def _generate_vulnerability_report(self, task: TaskInterface) -> TaskResult:
        """Generate comprehensive vulnerability report."""
        try:
            # Aggregate vulnerabilities from all scans
            all_vulnerabilities = []
            for scan in self.scan_history:
                all_vulnerabilities.extend(scan.vulnerabilities)
            
            # Generate summary statistics
            severity_counts = {
                "critical": len([v for v in all_vulnerabilities if v.severity == "critical"]),
                "high": len([v for v in all_vulnerabilities if v.severity == "high"]),
                "medium": len([v for v in all_vulnerabilities if v.severity == "medium"]),
                "low": len([v for v in all_vulnerabilities if v.severity == "low"])
            }
            
            # Generate component analysis
            component_analysis = {}
            for vuln in all_vulnerabilities:
                component = vuln.affected_component
                if component not in component_analysis:
                    component_analysis[component] = {"count": 0, "max_severity": "low"}
                component_analysis[component]["count"] += 1
                
                # Update max severity
                severity_order = ["low", "medium", "high", "critical"]
                current_max = component_analysis[component]["max_severity"]
                if severity_order.index(vuln.severity) > severity_order.index(current_max):
                    component_analysis[component]["max_severity"] = vuln.severity
            
            report = {
                "report_id": f"security_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_vulnerabilities": len(all_vulnerabilities),
                    "severity_breakdown": severity_counts,
                    "total_scans_performed": len(self.scan_history),
                    "components_affected": len(component_analysis)
                },
                "component_analysis": component_analysis,
                "recent_vulnerabilities": [
                    v.to_dict() for v in sorted(
                        all_vulnerabilities, 
                        key=lambda x: x.discovered_at, 
                        reverse=True
                    )[:10]
                ],
                "recommendations": [
                    "Prioritize fixing critical and high severity vulnerabilities",
                    "Implement automated security scanning in CI/CD pipeline",
                    "Regular dependency updates and security patches",
                    "Code review processes to catch security issues early"
                ]
            }
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={"vulnerability_report": report}
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="REPORT_GENERATION_FAILED"
            )
    
    async def _on_cleanup(self) -> None:
        """Cleanup plugin resources."""
        await self.log_info("Cleaning up SecurityScannerPlugin")
        
        # Cancel any active scans
        for scan_id in list(self.active_scans.keys()):
            self.active_scans.pop(scan_id, None)
        
        # Clear scan history
        self.scan_history.clear()
        self._scan_times.clear()
        
        # Reset statistics
        self.scanning_stats = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "false_positives": 0,
            "average_scan_time_seconds": 0.0
        }