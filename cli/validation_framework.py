"""
Validation Framework

Comprehensive validation and health monitoring system for Project Index installations.
Provides deep validation of services, configurations, performance metrics, and
overall system health with automated remediation suggestions.
"""

import os
import json
import time
import asyncio
import subprocess
import requests
import psutil
import docker
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import concurrent.futures
import socket
import ssl

class ValidationLevel(Enum):
    """Validation depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ENTERPRISE = "enterprise"

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

class ValidationCategory(Enum):
    """Categories of validation checks"""
    SYSTEM_REQUIREMENTS = "system_requirements"
    DOCKER_INFRASTRUCTURE = "docker_infrastructure"
    SERVICE_HEALTH = "service_health"
    API_CONNECTIVITY = "api_connectivity"
    DATABASE_INTEGRITY = "database_integrity"
    PERFORMANCE_METRICS = "performance_metrics"
    SECURITY_CONFIGURATION = "security_configuration"
    CONFIGURATION_VALIDITY = "configuration_validity"
    INTEGRATION_STATUS = "integration_status"
    DATA_CONSISTENCY = "data_consistency"

@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    category: ValidationCategory
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration_ms: int
    timestamp: datetime
    remediation: Optional[str] = None
    impact_level: str = "low"  # low, medium, high, critical

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    load_average: Tuple[float, float, float]
    uptime_seconds: int
    processes_count: int

@dataclass
class ServiceMetrics:
    """Service-specific metrics"""
    service_name: str
    status: str
    uptime_seconds: int
    memory_usage_mb: float
    cpu_usage_percent: float
    restart_count: int
    response_time_ms: Optional[float]
    error_rate: Optional[float]
    connections_active: Optional[int]

@dataclass
class HealthReport:
    """Comprehensive health report"""
    overall_status: HealthStatus
    validation_level: ValidationLevel
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    critical_issues: int
    validation_results: List[ValidationResult]
    system_metrics: SystemMetrics
    service_metrics: List[ServiceMetrics]
    recommendations: List[str]
    report_timestamp: datetime
    validation_duration_ms: int

class ValidationFramework:
    """Comprehensive validation and health monitoring framework"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.docker_client = None
        self.validation_start_time = None
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
    
    async def run_validation(self, validation_level: ValidationLevel = ValidationLevel.STANDARD,
                           config_path: Optional[Path] = None,
                           project_path: Optional[Path] = None) -> HealthReport:
        """Run comprehensive validation based on specified level"""
        
        self.validation_start_time = time.time()
        self.logger.info(f"Starting {validation_level.value} validation...")
        
        validation_results = []
        
        # Gather system metrics
        system_metrics = self._gather_system_metrics()
        
        # Run validation checks based on level
        checks_to_run = self._get_checks_for_level(validation_level)
        
        # Execute validation checks
        for check_name, check_function in checks_to_run.items():
            try:
                start_time = time.time()
                result = await check_function(config_path, project_path)
                duration_ms = int((time.time() - start_time) * 1000)
                
                if isinstance(result, list):
                    for res in result:
                        res.duration_ms = duration_ms
                        validation_results.append(res)
                else:
                    result.duration_ms = duration_ms
                    validation_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Validation check '{check_name}' failed: {e}")
                validation_results.append(ValidationResult(
                    check_name=check_name,
                    category=ValidationCategory.SYSTEM_REQUIREMENTS,
                    status=HealthStatus.CRITICAL,
                    message=f"Check failed with error: {str(e)}",
                    details={"error": str(e)},
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation="Check logs and fix the underlying issue",
                    impact_level="high"
                ))
        
        # Gather service metrics
        service_metrics = self._gather_service_metrics()
        
        # Generate report
        report = self._generate_health_report(
            validation_level, validation_results, system_metrics, service_metrics
        )
        
        total_duration = int((time.time() - self.validation_start_time) * 1000)
        self.logger.info(f"Validation completed in {total_duration}ms")
        
        return report
    
    def _get_checks_for_level(self, level: ValidationLevel) -> Dict[str, Any]:
        """Get validation checks for the specified level"""
        
        basic_checks = {
            "system_requirements": self._check_system_requirements,
            "docker_availability": self._check_docker_availability,
            "basic_connectivity": self._check_basic_connectivity
        }
        
        standard_checks = {
            **basic_checks,
            "service_health": self._check_service_health,
            "api_endpoints": self._check_api_endpoints,
            "database_connection": self._check_database_connection,
            "configuration_validity": self._check_configuration_validity
        }
        
        comprehensive_checks = {
            **standard_checks,
            "performance_metrics": self._check_performance_metrics,
            "security_configuration": self._check_security_configuration,
            "data_consistency": self._check_data_consistency,
            "integration_status": self._check_integration_status
        }
        
        enterprise_checks = {
            **comprehensive_checks,
            "advanced_security": self._check_advanced_security,
            "compliance_validation": self._check_compliance_validation,
            "disaster_recovery": self._check_disaster_recovery,
            "monitoring_systems": self._check_monitoring_systems
        }
        
        level_mapping = {
            ValidationLevel.BASIC: basic_checks,
            ValidationLevel.STANDARD: standard_checks,
            ValidationLevel.COMPREHENSIVE: comprehensive_checks,
            ValidationLevel.ENTERPRISE: enterprise_checks
        }
        
        return level_mapping[level]
    
    async def _check_system_requirements(self, config_path: Optional[Path], 
                                       project_path: Optional[Path]) -> List[ValidationResult]:
        """Check system requirements and capabilities"""
        
        results = []
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 4:
            status = HealthStatus.HEALTHY
            message = f"Sufficient memory available: {memory_gb:.1f}GB"
            remediation = None
        elif memory_gb >= 2:
            status = HealthStatus.WARNING
            message = f"Limited memory: {memory_gb:.1f}GB (recommended: 4GB+)"
            remediation = "Consider upgrading memory for better performance"
        else:
            status = HealthStatus.CRITICAL
            message = f"Insufficient memory: {memory_gb:.1f}GB (minimum: 2GB)"
            remediation = "Upgrade system memory to at least 2GB"
        
        results.append(ValidationResult(
            check_name="Memory Requirements",
            category=ValidationCategory.SYSTEM_REQUIREMENTS,
            status=status,
            message=message,
            details={"total_memory_gb": memory_gb, "available_memory_gb": memory.available / (1024**3)},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="high" if status == HealthStatus.CRITICAL else "medium"
        ))
        
        # CPU check
        cpu_count = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_count >= 4 and cpu_usage < 80:
            status = HealthStatus.HEALTHY
            message = f"Adequate CPU resources: {cpu_count} cores, {cpu_usage}% usage"
            remediation = None
        elif cpu_count >= 2 and cpu_usage < 90:
            status = HealthStatus.WARNING
            message = f"Limited CPU resources: {cpu_count} cores, {cpu_usage}% usage"
            remediation = "Monitor CPU usage during operation"
        else:
            status = HealthStatus.CRITICAL
            message = f"Insufficient CPU resources: {cpu_count} cores, {cpu_usage}% usage"
            remediation = "Reduce system load or upgrade CPU"
        
        results.append(ValidationResult(
            check_name="CPU Requirements",
            category=ValidationCategory.SYSTEM_REQUIREMENTS,
            status=status,
            message=message,
            details={"cpu_count": cpu_count, "cpu_usage_percent": cpu_usage},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="medium"
        ))
        
        # Disk space check
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        disk_usage_percent = (disk.used / disk.total) * 100
        
        if disk_free_gb >= 5 and disk_usage_percent < 80:
            status = HealthStatus.HEALTHY
            message = f"Adequate disk space: {disk_free_gb:.1f}GB free"
            remediation = None
        elif disk_free_gb >= 1 and disk_usage_percent < 90:
            status = HealthStatus.WARNING
            message = f"Limited disk space: {disk_free_gb:.1f}GB free"
            remediation = "Clean up disk space or monitor usage"
        else:
            status = HealthStatus.CRITICAL
            message = f"Insufficient disk space: {disk_free_gb:.1f}GB free"
            remediation = "Free up disk space immediately"
        
        results.append(ValidationResult(
            check_name="Disk Space Requirements",
            category=ValidationCategory.SYSTEM_REQUIREMENTS,
            status=status,
            message=message,
            details={"free_space_gb": disk_free_gb, "usage_percent": disk_usage_percent},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="high" if status == HealthStatus.CRITICAL else "medium"
        ))
        
        return results
    
    async def _check_docker_availability(self, config_path: Optional[Path], 
                                       project_path: Optional[Path]) -> ValidationResult:
        """Check Docker daemon availability and functionality"""
        
        if not self.docker_client:
            return ValidationResult(
                check_name="Docker Availability",
                category=ValidationCategory.DOCKER_INFRASTRUCTURE,
                status=HealthStatus.CRITICAL,
                message="Docker client could not be initialized",
                details={},
                duration_ms=0,
                timestamp=datetime.now(),
                remediation="Install Docker and ensure daemon is running",
                impact_level="critical"
            )
        
        try:
            # Test Docker connectivity
            self.docker_client.ping()
            version_info = self.docker_client.version()
            
            # Check Docker Compose availability
            compose_available = False
            try:
                result = subprocess.run(['docker', 'compose', 'version'], 
                                      capture_output=True, text=True)
                compose_available = result.returncode == 0
            except:
                pass
            
            if compose_available:
                status = HealthStatus.HEALTHY
                message = f"Docker is fully functional (version {version_info['Version']})"
                remediation = None
            else:
                status = HealthStatus.WARNING
                message = f"Docker available but Compose missing (version {version_info['Version']})"
                remediation = "Install Docker Compose for full functionality"
            
            return ValidationResult(
                check_name="Docker Availability",
                category=ValidationCategory.DOCKER_INFRASTRUCTURE,
                status=status,
                message=message,
                details={
                    "docker_version": version_info['Version'],
                    "api_version": version_info['ApiVersion'],
                    "compose_available": compose_available
                },
                duration_ms=0,
                timestamp=datetime.now(),
                remediation=remediation,
                impact_level="high" if not compose_available else "low"
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="Docker Availability",
                category=ValidationCategory.DOCKER_INFRASTRUCTURE,
                status=HealthStatus.CRITICAL,
                message=f"Docker daemon not accessible: {str(e)}",
                details={"error": str(e)},
                duration_ms=0,
                timestamp=datetime.now(),
                remediation="Start Docker daemon and check connectivity",
                impact_level="critical"
            )
    
    async def _check_basic_connectivity(self, config_path: Optional[Path], 
                                      project_path: Optional[Path]) -> List[ValidationResult]:
        """Check basic network connectivity and port availability"""
        
        results = []
        
        # Check if common ports are available
        ports_to_check = [8100, 8101, 5432, 6379, 9090]
        
        for port in ports_to_check:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    status = HealthStatus.HEALTHY
                    message = f"Port {port} is available"
                    remediation = None
                    impact = "low"
                except OSError:
                    status = HealthStatus.WARNING
                    message = f"Port {port} is in use"
                    remediation = f"Stop service using port {port} or configure alternative port"
                    impact = "medium"
                
                results.append(ValidationResult(
                    check_name=f"Port {port} Availability",
                    category=ValidationCategory.API_CONNECTIVITY,
                    status=status,
                    message=message,
                    details={"port": port},
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation=remediation,
                    impact_level=impact
                ))
        
        # Check internet connectivity
        try:
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "Internet connectivity is working"
                remediation = None
            else:
                status = HealthStatus.WARNING
                message = f"Internet connectivity issues (status: {response.status_code})"
                remediation = "Check network configuration"
                
        except Exception as e:
            status = HealthStatus.WARNING
            message = f"No internet connectivity: {str(e)}"
            remediation = "Check network connection for Docker image downloads"
        
        results.append(ValidationResult(
            check_name="Internet Connectivity",
            category=ValidationCategory.API_CONNECTIVITY,
            status=status,
            message=message,
            details={},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="medium"
        ))
        
        return results
    
    async def _check_service_health(self, config_path: Optional[Path], 
                                  project_path: Optional[Path]) -> List[ValidationResult]:
        """Check health of Project Index services"""
        
        results = []
        
        if not self.docker_client:
            return [ValidationResult(
                check_name="Service Health Check",
                category=ValidationCategory.SERVICE_HEALTH,
                status=HealthStatus.CRITICAL,
                message="Cannot check services - Docker unavailable",
                details={},
                duration_ms=0,
                timestamp=datetime.now(),
                remediation="Fix Docker connectivity first"
            )]
        
        # Expected services
        expected_services = ["postgres", "redis", "project-index-api"]
        
        try:
            containers = self.docker_client.containers.list(all=True)
            project_containers = {}
            
            for container in containers:
                for service in expected_services:
                    if service in container.name:
                        project_containers[service] = container
                        break
            
            for service_name in expected_services:
                if service_name in project_containers:
                    container = project_containers[service_name]
                    
                    if container.status == 'running':
                        # Check health if health check is configured
                        health_status = container.attrs.get('State', {}).get('Health', {})
                        
                        if health_status:
                            health = health_status.get('Status', 'unknown')
                            if health == 'healthy':
                                status = HealthStatus.HEALTHY
                                message = f"{service_name} is running and healthy"
                                remediation = None
                            elif health == 'unhealthy':
                                status = HealthStatus.CRITICAL
                                message = f"{service_name} is running but unhealthy"
                                remediation = f"Check {service_name} logs and restart if necessary"
                            else:
                                status = HealthStatus.WARNING
                                message = f"{service_name} health status unknown"
                                remediation = f"Monitor {service_name} for issues"
                        else:
                            status = HealthStatus.HEALTHY
                            message = f"{service_name} is running"
                            remediation = None
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"{service_name} is not running (status: {container.status})"
                        remediation = f"Start {service_name} service"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"{service_name} container not found"
                    remediation = f"Deploy {service_name} service"
                
                results.append(ValidationResult(
                    check_name=f"{service_name} Health",
                    category=ValidationCategory.SERVICE_HEALTH,
                    status=status,
                    message=message,
                    details={"service": service_name},
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation=remediation,
                    impact_level="high"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="Service Health Check",
                category=ValidationCategory.SERVICE_HEALTH,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check service health: {str(e)}",
                details={"error": str(e)},
                duration_ms=0,
                timestamp=datetime.now(),
                remediation="Check Docker and service configuration"
            ))
        
        return results
    
    async def _check_api_endpoints(self, config_path: Optional[Path], 
                                 project_path: Optional[Path]) -> List[ValidationResult]:
        """Check API endpoint availability and functionality"""
        
        results = []
        
        # API endpoints to test
        endpoints = [
            {"path": "/health", "method": "GET", "expected_status": 200},
            {"path": "/api/status", "method": "GET", "expected_status": 200},
            {"path": "/api/search", "method": "GET", "expected_status": [200, 400]},  # 400 is OK for missing query
        ]
        
        base_url = "http://localhost:8100"  # Default API URL
        
        for endpoint in endpoints:
            try:
                if endpoint["method"] == "GET":
                    response = requests.get(f"{base_url}{endpoint['path']}", timeout=10)
                else:
                    response = requests.request(endpoint["method"], 
                                              f"{base_url}{endpoint['path']}", timeout=10)
                
                expected_status = endpoint["expected_status"]
                if isinstance(expected_status, list):
                    status_ok = response.status_code in expected_status
                else:
                    status_ok = response.status_code == expected_status
                
                if status_ok:
                    status = HealthStatus.HEALTHY
                    message = f"Endpoint {endpoint['path']} is responding correctly"
                    remediation = None
                    impact = "low"
                else:
                    status = HealthStatus.WARNING
                    message = f"Endpoint {endpoint['path']} returned status {response.status_code}"
                    remediation = "Check API service logs"
                    impact = "medium"
                
                results.append(ValidationResult(
                    check_name=f"API Endpoint {endpoint['path']}",
                    category=ValidationCategory.API_CONNECTIVITY,
                    status=status,
                    message=message,
                    details={
                        "endpoint": endpoint['path'],
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    },
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation=remediation,
                    impact_level=impact
                ))
                
            except requests.ConnectionError:
                results.append(ValidationResult(
                    check_name=f"API Endpoint {endpoint['path']}",
                    category=ValidationCategory.API_CONNECTIVITY,
                    status=HealthStatus.CRITICAL,
                    message=f"Cannot connect to API endpoint {endpoint['path']}",
                    details={"endpoint": endpoint['path']},
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation="Start Project Index API service",
                    impact_level="high"
                ))
                break  # If we can't connect, other endpoints will fail too
                
            except Exception as e:
                results.append(ValidationResult(
                    check_name=f"API Endpoint {endpoint['path']}",
                    category=ValidationCategory.API_CONNECTIVITY,
                    status=HealthStatus.WARNING,
                    message=f"Error testing endpoint {endpoint['path']}: {str(e)}",
                    details={"endpoint": endpoint['path'], "error": str(e)},
                    duration_ms=0,
                    timestamp=datetime.now(),
                    remediation="Check API service configuration",
                    impact_level="medium"
                ))
        
        return results
    
    async def _check_database_connection(self, config_path: Optional[Path], 
                                       project_path: Optional[Path]) -> ValidationResult:
        """Check database connectivity and basic functionality"""
        
        try:
            # Try to connect to PostgreSQL using psql if available
            result = subprocess.run([
                'docker', 'exec', '-t', 'project-index_postgres', 
                'pg_isready', '-U', 'project_index'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Try a simple query
                query_result = subprocess.run([
                    'docker', 'exec', '-t', 'project-index_postgres',
                    'psql', '-U', 'project_index', '-d', 'project_index_db',
                    '-c', 'SELECT 1;'
                ], capture_output=True, text=True, timeout=10)
                
                if query_result.returncode == 0:
                    status = HealthStatus.HEALTHY
                    message = "Database is accessible and functional"
                    remediation = None
                else:
                    status = HealthStatus.WARNING
                    message = "Database is accessible but queries may be failing"
                    remediation = "Check database configuration and permissions"
            else:
                status = HealthStatus.CRITICAL
                message = "Database is not accessible"
                remediation = "Start PostgreSQL service and check connectivity"
            
        except subprocess.TimeoutExpired:
            status = HealthStatus.WARNING
            message = "Database check timed out"
            remediation = "Check database performance and load"
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Database check failed: {str(e)}"
            remediation = "Check database service status and configuration"
        
        return ValidationResult(
            check_name="Database Connection",
            category=ValidationCategory.DATABASE_INTEGRITY,
            status=status,
            message=message,
            details={},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="high"
        )
    
    async def _check_configuration_validity(self, config_path: Optional[Path], 
                                          project_path: Optional[Path]) -> List[ValidationResult]:
        """Check configuration file validity and completeness"""
        
        results = []
        
        # Check for configuration files
        config_files = [
            "docker-compose.yml",
            ".env",
            "project-index-config.json"
        ]
        
        for config_file in config_files:
            file_path = Path(config_file)
            if config_path:
                file_path = config_path / config_file
            
            if file_path.exists():
                try:
                    if config_file.endswith('.yml') or config_file.endswith('.yaml'):
                        import yaml
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f)
                        
                    elif config_file.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json.load(f)
                    
                    status = HealthStatus.HEALTHY
                    message = f"Configuration file {config_file} is valid"
                    remediation = None
                    
                except Exception as e:
                    status = HealthStatus.CRITICAL
                    message = f"Configuration file {config_file} is invalid: {str(e)}"
                    remediation = f"Fix syntax errors in {config_file}"
            else:
                status = HealthStatus.WARNING
                message = f"Configuration file {config_file} not found"
                remediation = f"Create or restore {config_file}"
            
            results.append(ValidationResult(
                check_name=f"Config File {config_file}",
                category=ValidationCategory.CONFIGURATION_VALIDITY,
                status=status,
                message=message,
                details={"file": config_file},
                duration_ms=0,
                timestamp=datetime.now(),
                remediation=remediation,
                impact_level="medium"
            ))
        
        return results
    
    async def _check_performance_metrics(self, config_path: Optional[Path], 
                                       project_path: Optional[Path]) -> List[ValidationResult]:
        """Check performance metrics and resource usage"""
        
        results = []
        
        # Check system load
        load_avg = psutil.getloadavg()
        cpu_count = psutil.cpu_count()
        
        if load_avg[0] < cpu_count * 0.7:
            status = HealthStatus.HEALTHY
            message = f"System load is normal: {load_avg[0]:.2f}"
            remediation = None
        elif load_avg[0] < cpu_count * 1.5:
            status = HealthStatus.WARNING
            message = f"System load is elevated: {load_avg[0]:.2f}"
            remediation = "Monitor system performance"
        else:
            status = HealthStatus.CRITICAL
            message = f"System load is very high: {load_avg[0]:.2f}"
            remediation = "Reduce system load or scale resources"
        
        results.append(ValidationResult(
            check_name="System Load",
            category=ValidationCategory.PERFORMANCE_METRICS,
            status=status,
            message=message,
            details={"load_average": load_avg, "cpu_count": cpu_count},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="medium"
        ))
        
        # Check memory pressure
        memory = psutil.virtual_memory()
        memory_pressure = memory.percent
        
        if memory_pressure < 70:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is normal: {memory_pressure:.1f}%"
            remediation = None
        elif memory_pressure < 85:
            status = HealthStatus.WARNING
            message = f"Memory usage is elevated: {memory_pressure:.1f}%"
            remediation = "Monitor memory usage"
        else:
            status = HealthStatus.CRITICAL
            message = f"Memory usage is critical: {memory_pressure:.1f}%"
            remediation = "Free memory or add more RAM"
        
        results.append(ValidationResult(
            check_name="Memory Pressure",
            category=ValidationCategory.PERFORMANCE_METRICS,
            status=status,
            message=message,
            details={"memory_percent": memory_pressure, "available_gb": memory.available / (1024**3)},
            duration_ms=0,
            timestamp=datetime.now(),
            remediation=remediation,
            impact_level="high"
        ))
        
        return results
    
    # Placeholder methods for additional checks
    async def _check_security_configuration(self, config_path: Optional[Path], 
                                          project_path: Optional[Path]) -> List[ValidationResult]:
        """Check security configuration"""
        # Implementation would check SSL, authentication, etc.
        return []
    
    async def _check_data_consistency(self, config_path: Optional[Path], 
                                    project_path: Optional[Path]) -> List[ValidationResult]:
        """Check data consistency and integrity"""
        # Implementation would check database consistency, index integrity, etc.
        return []
    
    async def _check_integration_status(self, config_path: Optional[Path], 
                                      project_path: Optional[Path]) -> List[ValidationResult]:
        """Check integration status with external systems"""
        # Implementation would check framework integrations, webhooks, etc.
        return []
    
    async def _check_advanced_security(self, config_path: Optional[Path], 
                                     project_path: Optional[Path]) -> List[ValidationResult]:
        """Check advanced security features"""
        # Implementation would check enterprise security features
        return []
    
    async def _check_compliance_validation(self, config_path: Optional[Path], 
                                         project_path: Optional[Path]) -> List[ValidationResult]:
        """Check compliance requirements"""
        # Implementation would check compliance standards
        return []
    
    async def _check_disaster_recovery(self, config_path: Optional[Path], 
                                     project_path: Optional[Path]) -> List[ValidationResult]:
        """Check disaster recovery preparedness"""
        # Implementation would check backup systems, recovery procedures
        return []
    
    async def _check_monitoring_systems(self, config_path: Optional[Path], 
                                      project_path: Optional[Path]) -> List[ValidationResult]:
        """Check monitoring and alerting systems"""
        # Implementation would check Prometheus, Grafana, alerting
        return []
    
    def _gather_system_metrics(self) -> SystemMetrics:
        """Gather current system metrics"""
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            memory_usage_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io={"bytes_sent": network.bytes_sent, "bytes_recv": network.bytes_recv},
            load_average=psutil.getloadavg(),
            uptime_seconds=int(time.time() - psutil.boot_time()),
            processes_count=len(psutil.pids())
        )
    
    def _gather_service_metrics(self) -> List[ServiceMetrics]:
        """Gather service-specific metrics"""
        
        service_metrics = []
        
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                
                for container in containers:
                    if "project-index" in container.name or any(service in container.name 
                                                             for service in ["postgres", "redis"]):
                        
                        # Get basic stats
                        stats = container.stats(stream=False)
                        
                        # Calculate memory usage
                        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                        
                        # Calculate CPU usage (simplified)
                        cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] - 
                                   stats['precpu_stats']['cpu_usage']['total_usage'])
                        system_delta = (stats['cpu_stats']['system_cpu_usage'] - 
                                      stats['precpu_stats']['system_cpu_usage'])
                        cpu_usage = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                        
                        # Get uptime
                        created_time = container.attrs['Created']
                        uptime = int((datetime.now() - datetime.fromisoformat(
                            created_time.replace('Z', '+00:00'))).total_seconds())
                        
                        service_metrics.append(ServiceMetrics(
                            service_name=container.name,
                            status=container.status,
                            uptime_seconds=uptime,
                            memory_usage_mb=memory_usage,
                            cpu_usage_percent=cpu_usage,
                            restart_count=container.attrs['RestartCount'],
                            response_time_ms=None,  # Would need to test endpoints
                            error_rate=None,  # Would need to check logs
                            connections_active=None  # Would need service-specific checks
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Failed to gather service metrics: {e}")
        
        return service_metrics
    
    def _generate_health_report(self, validation_level: ValidationLevel,
                              validation_results: List[ValidationResult],
                              system_metrics: SystemMetrics,
                              service_metrics: List[ServiceMetrics]) -> HealthReport:
        """Generate comprehensive health report"""
        
        # Calculate overall status
        critical_count = sum(1 for r in validation_results if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in validation_results if r.status == HealthStatus.WARNING)
        healthy_count = sum(1 for r in validation_results if r.status == HealthStatus.HEALTHY)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > healthy_count:
            overall_status = HealthStatus.DEGRADED
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, system_metrics)
        
        # Calculate validation duration
        validation_duration = int((time.time() - self.validation_start_time) * 1000)
        
        return HealthReport(
            overall_status=overall_status,
            validation_level=validation_level,
            total_checks=len(validation_results),
            passed_checks=healthy_count,
            failed_checks=critical_count,
            warning_checks=warning_count,
            critical_issues=critical_count,
            validation_results=validation_results,
            system_metrics=system_metrics,
            service_metrics=service_metrics,
            recommendations=recommendations,
            report_timestamp=datetime.now(),
            validation_duration_ms=validation_duration
        )
    
    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                system_metrics: SystemMetrics) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        
        recommendations = []
        
        # Critical issues first
        critical_results = [r for r in validation_results if r.status == HealthStatus.CRITICAL]
        if critical_results:
            recommendations.append("Address critical issues immediately:")
            for result in critical_results:
                if result.remediation:
                    recommendations.append(f"  - {result.remediation}")
        
        # System optimization recommendations
        if system_metrics.memory_usage_percent > 80:
            recommendations.append("Consider adding more memory to improve performance")
        
        if system_metrics.cpu_usage_percent > 80:
            recommendations.append("Monitor CPU usage and consider scaling resources")
        
        if system_metrics.disk_usage_percent > 85:
            recommendations.append("Clean up disk space to prevent storage issues")
        
        # Warning-level recommendations
        warning_results = [r for r in validation_results if r.status == HealthStatus.WARNING]
        if warning_results:
            recommendations.append("Monitor these potential issues:")
            for result in warning_results[:3]:  # Limit to top 3
                if result.remediation:
                    recommendations.append(f"  - {result.remediation}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is healthy - continue monitoring")
        
        return recommendations
    
    def export_report(self, report: HealthReport, output_path: Path, 
                     format_type: str = "json") -> Path:
        """Export health report to file"""
        
        if format_type == "json":
            file_path = output_path / f"health-report-{int(time.time())}.json"
            with open(file_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format_type == "html":
            file_path = output_path / f"health-report-{int(time.time())}.html"
            html_content = self._generate_html_report(report)
            with open(file_path, 'w') as f:
                f.write(html_content)
        
        self.logger.info(f"Health report exported to {file_path}")
        return file_path
    
    def _generate_html_report(self, report: HealthReport) -> str:
        """Generate HTML version of the health report"""
        
        status_colors = {
            HealthStatus.HEALTHY: "#28a745",
            HealthStatus.WARNING: "#ffc107", 
            HealthStatus.CRITICAL: "#dc3545",
            HealthStatus.DEGRADED: "#fd7e14",
            HealthStatus.UNKNOWN: "#6c757d"
        }
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Project Index Health Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .status {{ font-weight: bold; color: {status_colors[report.overall_status]}; }}
        .metric {{ margin: 10px 0; }}
        .checks {{ margin-top: 20px; }}
        .check {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
        .check.healthy {{ border-color: {status_colors[HealthStatus.HEALTHY]}; }}
        .check.warning {{ border-color: {status_colors[HealthStatus.WARNING]}; }}
        .check.critical {{ border-color: {status_colors[HealthStatus.CRITICAL]}; }}
        .recommendations {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Project Index Health Report</h1>
        <div class="status">Overall Status: {report.overall_status.value.upper()}</div>
        <div class="metric">Generated: {report.report_timestamp}</div>
        <div class="metric">Validation Level: {report.validation_level.value}</div>
        <div class="metric">Total Checks: {report.total_checks}</div>
        <div class="metric">Passed: {report.passed_checks} | Warnings: {report.warning_checks} | Failed: {report.failed_checks}</div>
    </div>
    
    <div class="checks">
        <h2>Validation Results</h2>
        {self._generate_html_checks(report.validation_results)}
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {"".join(f"<li>{rec}</li>" for rec in report.recommendations)}
        </ul>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_html_checks(self, validation_results: List[ValidationResult]) -> str:
        """Generate HTML for validation checks"""
        
        html_parts = []
        
        for result in validation_results:
            css_class = result.status.value
            
            html_parts.append(f"""
            <div class="check {css_class}">
                <h3>{result.check_name}</h3>
                <p><strong>Status:</strong> {result.status.value.upper()}</p>
                <p><strong>Message:</strong> {result.message}</p>
                {f"<p><strong>Remediation:</strong> {result.remediation}</p>" if result.remediation else ""}
                <p><small>Category: {result.category.value} | Duration: {result.duration_ms}ms</small></p>
            </div>
            """)
        
        return "".join(html_parts)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize validation framework
        validator = ValidationFramework()
        
        # Run validation
        report = await validator.run_validation(
            validation_level=ValidationLevel.STANDARD,
            config_path=Path("."),
            project_path=Path(".")
        )
        
        print(f"\\n=== Project Index Health Report ===")
        print(f"Overall Status: {report.overall_status.value.upper()}")
        print(f"Total Checks: {report.total_checks}")
        print(f"Passed: {report.passed_checks} | Warnings: {report.warning_checks} | Failed: {report.failed_checks}")
        print(f"Critical Issues: {report.critical_issues}")
        
        print(f"\\n=== System Metrics ===")
        print(f"CPU Usage: {report.system_metrics.cpu_usage_percent:.1f}%")
        print(f"Memory Usage: {report.system_metrics.memory_usage_percent:.1f}%")
        print(f"Disk Usage: {report.system_metrics.disk_usage_percent:.1f}%")
        
        print(f"\\n=== Critical Issues ===")
        for result in report.validation_results:
            if result.status == HealthStatus.CRITICAL:
                print(f"❌ {result.check_name}: {result.message}")
                if result.remediation:
                    print(f"   Fix: {result.remediation}")
        
        print(f"\\n=== Recommendations ===")
        for rec in report.recommendations:
            print(f"• {rec}")
        
        # Export report
        output_path = Path("./validation_output")
        output_path.mkdir(exist_ok=True)
        
        json_file = validator.export_report(report, output_path, "json")
        html_file = validator.export_report(report, output_path, "html")
        
        print(f"\\nReports exported:")
        print(f"  JSON: {json_file}")
        print(f"  HTML: {html_file}")
    
    asyncio.run(main())