"""
Configuration Generator

Intelligent configuration generation system that creates optimized Project Index
configurations based on project analysis, system capabilities, and user preferences.
Provides smart defaults, performance tuning, and environment-specific optimizations.
"""

import os
import json
import yaml
import toml
import re
import subprocess
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import secrets
import string

from cli.project_detector import ProjectAnalysis, Language, ProjectType
from cli.docker_manager import DeploymentProfile

class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"
    PYTHON = "python"
    JAVASCRIPT = "javascript"

class OptimizationLevel(Enum):
    """Configuration optimization levels"""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"

@dataclass
class SystemCapabilities:
    """System capabilities and constraints"""
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores: int
    cpu_frequency_ghz: float
    disk_space_gb: float
    disk_type: str  # "SSD", "HDD", "NVMe"
    network_bandwidth_mbps: Optional[float]
    docker_memory_limit_gb: Optional[float]
    supports_gpu: bool
    os_type: str
    architecture: str

@dataclass
class PerformanceProfile:
    """Performance optimization profile"""
    name: str
    memory_allocation: Dict[str, str]  # service -> memory limit
    cpu_allocation: Dict[str, float]   # service -> CPU limit
    connection_limits: Dict[str, int]  # service -> max connections
    cache_settings: Dict[str, Any]
    indexing_settings: Dict[str, Any]
    batch_sizes: Dict[str, int]
    timeout_settings: Dict[str, int]

@dataclass
class SecurityConfiguration:
    """Security configuration settings"""
    api_key_length: int
    password_complexity: Dict[str, bool]
    encryption_settings: Dict[str, Any]
    access_control: Dict[str, Any]
    rate_limiting: Dict[str, Any]
    cors_settings: Dict[str, Any]
    ssl_settings: Dict[str, Any]

@dataclass
class ProjectIndexConfiguration:
    """Complete Project Index configuration"""
    project_info: Dict[str, Any]
    deployment_profile: DeploymentProfile
    optimization_level: OptimizationLevel
    performance_profile: PerformanceProfile
    security_config: SecurityConfiguration
    service_configs: Dict[str, Dict[str, Any]]
    environment_variables: Dict[str, str]
    feature_flags: Dict[str, bool]
    monitoring_config: Dict[str, Any]
    backup_config: Dict[str, Any]
    integration_configs: Dict[str, Dict[str, Any]]

class ConfigurationGenerator:
    """Advanced configuration generator with intelligent optimization"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Default performance profiles
        self.performance_profiles = {
            OptimizationLevel.MINIMAL: PerformanceProfile(
                name="Minimal Resource Usage",
                memory_allocation={
                    "postgres": "256m",
                    "redis": "64m",
                    "api": "512m",
                    "indexer": "256m"
                },
                cpu_allocation={
                    "postgres": 0.5,
                    "redis": 0.25,
                    "api": 0.5,
                    "indexer": 0.25
                },
                connection_limits={
                    "postgres": 20,
                    "redis": 100,
                    "api": 50
                },
                cache_settings={
                    "redis_max_memory": "64m",
                    "postgres_shared_buffers": "32m",
                    "query_cache_size": 100
                },
                indexing_settings={
                    "batch_size": 10,
                    "index_frequency": "1h",
                    "concurrent_indexers": 1
                },
                batch_sizes={
                    "file_processing": 10,
                    "search_results": 50
                },
                timeout_settings={
                    "api_timeout": 30,
                    "db_timeout": 15,
                    "index_timeout": 300
                }
            ),
            OptimizationLevel.BALANCED: PerformanceProfile(
                name="Balanced Performance",
                memory_allocation={
                    "postgres": "512m",
                    "redis": "128m",
                    "api": "1g",
                    "indexer": "512m"
                },
                cpu_allocation={
                    "postgres": 1.0,
                    "redis": 0.5,
                    "api": 1.0,
                    "indexer": 0.5
                },
                connection_limits={
                    "postgres": 50,
                    "redis": 200,
                    "api": 100
                },
                cache_settings={
                    "redis_max_memory": "128m",
                    "postgres_shared_buffers": "128m",
                    "query_cache_size": 500
                },
                indexing_settings={
                    "batch_size": 50,
                    "index_frequency": "30m",
                    "concurrent_indexers": 2
                },
                batch_sizes={
                    "file_processing": 50,
                    "search_results": 100
                },
                timeout_settings={
                    "api_timeout": 60,
                    "db_timeout": 30,
                    "index_timeout": 600
                }
            ),
            OptimizationLevel.PERFORMANCE: PerformanceProfile(
                name="High Performance",
                memory_allocation={
                    "postgres": "1g",
                    "redis": "256m",
                    "api": "2g",
                    "indexer": "1g"
                },
                cpu_allocation={
                    "postgres": 2.0,
                    "redis": 1.0,
                    "api": 2.0,
                    "indexer": 1.0
                },
                connection_limits={
                    "postgres": 100,
                    "redis": 500,
                    "api": 200
                },
                cache_settings={
                    "redis_max_memory": "256m",
                    "postgres_shared_buffers": "256m",
                    "query_cache_size": 1000
                },
                indexing_settings={
                    "batch_size": 100,
                    "index_frequency": "15m",
                    "concurrent_indexers": 4
                },
                batch_sizes={
                    "file_processing": 100,
                    "search_results": 200
                },
                timeout_settings={
                    "api_timeout": 120,
                    "db_timeout": 60,
                    "index_timeout": 1200
                }
            ),
            OptimizationLevel.ENTERPRISE: PerformanceProfile(
                name="Enterprise Scale",
                memory_allocation={
                    "postgres": "4g",
                    "redis": "1g",
                    "api": "4g",
                    "indexer": "2g"
                },
                cpu_allocation={
                    "postgres": 4.0,
                    "redis": 2.0,
                    "api": 4.0,
                    "indexer": 2.0
                },
                connection_limits={
                    "postgres": 200,
                    "redis": 1000,
                    "api": 500
                },
                cache_settings={
                    "redis_max_memory": "1g",
                    "postgres_shared_buffers": "1g",
                    "query_cache_size": 5000
                },
                indexing_settings={
                    "batch_size": 500,
                    "index_frequency": "5m",
                    "concurrent_indexers": 8
                },
                batch_sizes={
                    "file_processing": 500,
                    "search_results": 500
                },
                timeout_settings={
                    "api_timeout": 300,
                    "db_timeout": 120,
                    "index_timeout": 3600
                }
            )
        }
    
    def analyze_system_capabilities(self) -> SystemCapabilities:
        """Analyze current system capabilities and constraints"""
        self.logger.info("Analyzing system capabilities...")
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # CPU information
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_frequency = psutil.cpu_freq()
        cpu_frequency_ghz = cpu_frequency.current / 1000 if cpu_frequency else 2.0
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_space_gb = disk.free / (1024**3)
        
        # Detect disk type (best effort)
        disk_type = "Unknown"
        try:
            if platform.system() == "Linux":
                result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and '0' in result.stdout:
                    disk_type = "SSD"
                else:
                    disk_type = "HDD"
            elif platform.system() == "Darwin":
                # macOS - assume SSD for modern Macs
                disk_type = "SSD"
            elif platform.system() == "Windows":
                # Windows detection would require WMI
                disk_type = "SSD"  # Modern assumption
        except:
            pass
        
        # Docker memory limits
        docker_memory_limit_gb = None
        try:
            result = subprocess.run(['docker', 'system', 'info', '--format', '{{.MemTotal}}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                docker_memory_limit_gb = int(result.stdout.strip()) / (1024**3)
        except:
            pass
        
        # GPU detection
        supports_gpu = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            supports_gpu = result.returncode == 0
        except:
            pass
        
        # OS and architecture
        os_type = platform.system()
        architecture = platform.machine()
        
        capabilities = SystemCapabilities(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            cpu_cores=cpu_cores,
            cpu_frequency_ghz=cpu_frequency_ghz,
            disk_space_gb=disk_space_gb,
            disk_type=disk_type,
            network_bandwidth_mbps=None,  # Not easily detectable
            docker_memory_limit_gb=docker_memory_limit_gb,
            supports_gpu=supports_gpu,
            os_type=os_type,
            architecture=architecture
        )
        
        self.logger.info(f"System capabilities: {total_memory_gb:.1f}GB RAM, {cpu_cores} CPU cores, {disk_space_gb:.1f}GB disk")
        return capabilities
    
    def recommend_optimization_level(self, analysis: ProjectAnalysis, 
                                   capabilities: SystemCapabilities) -> OptimizationLevel:
        """Recommend optimization level based on project and system analysis"""
        
        score = 0
        
        # Project complexity factors
        if analysis.estimated_complexity == "enterprise":
            score += 4
        elif analysis.estimated_complexity == "high":
            score += 3
        elif analysis.estimated_complexity == "medium":
            score += 2
        else:
            score += 1
        
        # File count factors
        if analysis.file_count > 50000:
            score += 3
        elif analysis.file_count > 10000:
            score += 2
        elif analysis.file_count > 1000:
            score += 1
        
        # System capability factors
        if capabilities.total_memory_gb >= 16:
            score += 2
        elif capabilities.total_memory_gb >= 8:
            score += 1
        
        if capabilities.cpu_cores >= 8:
            score += 2
        elif capabilities.cpu_cores >= 4:
            score += 1
        
        if capabilities.disk_type in ["SSD", "NVMe"]:
            score += 1
        
        # Map score to optimization level
        if score >= 10:
            return OptimizationLevel.ENTERPRISE
        elif score >= 7:
            return OptimizationLevel.PERFORMANCE
        elif score >= 4:
            return OptimizationLevel.BALANCED
        else:
            return OptimizationLevel.MINIMAL
    
    def generate_security_config(self, optimization_level: OptimizationLevel) -> SecurityConfiguration:
        """Generate security configuration based on optimization level"""
        
        base_config = SecurityConfiguration(
            api_key_length=32,
            password_complexity={
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True
            },
            encryption_settings={
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 90,
                "encrypt_at_rest": True
            },
            access_control={
                "enable_rbac": False,
                "default_permissions": "read",
                "session_timeout_minutes": 60
            },
            rate_limiting={
                "api_requests_per_minute": 100,
                "search_requests_per_minute": 50,
                "burst_allowance": 20
            },
            cors_settings={
                "allowed_origins": ["http://localhost:3000", "http://localhost:8080"],
                "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_credentials": True
            },
            ssl_settings={
                "enabled": False,
                "min_tls_version": "1.2",
                "cipher_suites": ["ECDHE-RSA-AES256-GCM-SHA384"]
            }
        )
        
        # Adjust based on optimization level
        if optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE]:
            base_config.access_control["enable_rbac"] = True
            base_config.ssl_settings["enabled"] = True
            base_config.rate_limiting["api_requests_per_minute"] = 500
            base_config.rate_limiting["search_requests_per_minute"] = 200
        
        if optimization_level == OptimizationLevel.ENTERPRISE:
            base_config.api_key_length = 64
            base_config.encryption_settings["key_rotation_days"] = 30
            base_config.access_control["session_timeout_minutes"] = 30
            base_config.ssl_settings["min_tls_version"] = "1.3"
        
        return base_config
    
    def optimize_for_project_type(self, base_config: Dict[str, Any], 
                                 analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Optimize configuration based on project type"""
        
        config = base_config.copy()
        
        if analysis.project_type == ProjectType.WEB_APPLICATION:
            # Web applications need responsive search
            config["indexing"]["real_time_updates"] = True
            config["api"]["response_timeout"] = 5000
            config["search"]["max_results"] = 100
        
        elif analysis.project_type == ProjectType.MOBILE_APPLICATION:
            # Mobile apps often have smaller codebases but need fast responses
            config["indexing"]["batch_size"] = 20
            config["api"]["response_timeout"] = 3000
            config["search"]["max_results"] = 50
        
        elif analysis.project_type == ProjectType.MONOREPO:
            # Monorepos need powerful indexing and organization
            config["indexing"]["concurrent_indexers"] = min(8, config["indexing"]["concurrent_indexers"] * 2)
            config["indexing"]["workspace_detection"] = True
            config["search"]["support_workspace_scoping"] = True
        
        elif analysis.project_type == ProjectType.DATA_SCIENCE:
            # Data science projects have notebooks and large files
            config["indexing"]["notebook_support"] = True
            config["indexing"]["large_file_threshold"] = "50MB"
            config["search"]["include_notebook_cells"] = True
        
        elif analysis.project_type == ProjectType.INFRASTRUCTURE:
            # Infrastructure projects need configuration analysis
            config["indexing"]["config_file_analysis"] = True
            config["search"]["infrastructure_patterns"] = True
            config["analysis"]["security_scanning"] = True
        
        # Language-specific optimizations
        if Language.PYTHON in analysis.languages:
            config["analysis"]["python_ast_parsing"] = True
            config["indexing"]["virtual_env_detection"] = True
        
        if Language.JAVASCRIPT in analysis.languages or Language.TYPESCRIPT in analysis.languages:
            config["analysis"]["node_modules_handling"] = "smart_ignore"
            config["indexing"]["package_json_analysis"] = True
        
        if Language.JAVA in analysis.languages:
            config["analysis"]["maven_gradle_support"] = True
            config["indexing"]["class_hierarchy_analysis"] = True
        
        return config
    
    def generate_configuration(self, analysis: ProjectAnalysis, 
                             deployment_profile: DeploymentProfile,
                             user_preferences: Dict[str, Any] = None) -> ProjectIndexConfiguration:
        """Generate complete Project Index configuration"""
        
        self.logger.info("Generating Project Index configuration...")
        
        # Analyze system capabilities
        capabilities = self.analyze_system_capabilities()
        
        # Determine optimization level
        optimization_level = self.recommend_optimization_level(analysis, capabilities)
        if user_preferences and "optimization_level" in user_preferences:
            optimization_level = OptimizationLevel(user_preferences["optimization_level"])
        
        self.logger.info(f"Using optimization level: {optimization_level.value}")
        
        # Get performance profile
        performance_profile = self.performance_profiles[optimization_level]
        
        # Generate security configuration
        security_config = self.generate_security_config(optimization_level)
        
        # Project information
        project_info = {
            "name": analysis.project_name,
            "path": analysis.project_path,
            "type": analysis.project_type.value,
            "primary_language": analysis.primary_language.value,
            "languages": {lang.value: percentage for lang, percentage in analysis.languages.items()},
            "frameworks": [f.name for f in analysis.frameworks],
            "file_count": analysis.file_count,
            "line_count": analysis.line_count,
            "complexity": analysis.estimated_complexity
        }
        
        # Service configurations
        service_configs = self._generate_service_configs(
            performance_profile, capabilities, analysis
        )
        
        # Environment variables
        environment_variables = self._generate_environment_variables(
            analysis, deployment_profile, capabilities
        )
        
        # Feature flags
        feature_flags = self._generate_feature_flags(
            optimization_level, analysis, capabilities
        )
        
        # Monitoring configuration
        monitoring_config = self._generate_monitoring_config(
            optimization_level, deployment_profile
        )
        
        # Backup configuration
        backup_config = self._generate_backup_config(
            optimization_level, deployment_profile
        )
        
        # Integration configurations
        integration_configs = self._generate_integration_configs(analysis)
        
        return ProjectIndexConfiguration(
            project_info=project_info,
            deployment_profile=deployment_profile,
            optimization_level=optimization_level,
            performance_profile=performance_profile,
            security_config=security_config,
            service_configs=service_configs,
            environment_variables=environment_variables,
            feature_flags=feature_flags,
            monitoring_config=monitoring_config,
            backup_config=backup_config,
            integration_configs=integration_configs
        )
    
    def _generate_service_configs(self, performance_profile: PerformanceProfile,
                                capabilities: SystemCapabilities,
                                analysis: ProjectAnalysis) -> Dict[str, Dict[str, Any]]:
        """Generate service-specific configurations"""
        
        configs = {}
        
        # PostgreSQL configuration
        configs["postgres"] = {
            "shared_buffers": performance_profile.cache_settings["postgres_shared_buffers"],
            "effective_cache_size": f"{int(capabilities.available_memory_gb * 0.75)}GB",
            "work_mem": "16MB",
            "maintenance_work_mem": "256MB",
            "max_connections": performance_profile.connection_limits["postgres"],
            "log_statement": "none",
            "log_min_duration_statement": 1000,
            "extensions": ["pg_trgm", "btree_gin", "vector"],
            "checkpoint_completion_target": 0.9,
            "wal_buffers": "16MB",
            "default_statistics_target": 100
        }
        
        # Redis configuration
        configs["redis"] = {
            "maxmemory": performance_profile.cache_settings["redis_max_memory"],
            "maxmemory_policy": "allkeys-lru",
            "save": ["900 1", "300 10", "60 10000"],
            "tcp_keepalive": 300,
            "timeout": 0,
            "databases": 16,
            "hash_max_ziplist_entries": 512,
            "hash_max_ziplist_value": 64
        }
        
        # Project Index API configuration
        configs["api"] = {
            "memory_limit": performance_profile.memory_allocation["api"],
            "worker_processes": min(capabilities.cpu_cores, 4),
            "worker_connections": performance_profile.connection_limits["api"],
            "request_timeout": performance_profile.timeout_settings["api_timeout"],
            "database_timeout": performance_profile.timeout_settings["db_timeout"],
            "search_timeout": 30,
            "batch_size": performance_profile.batch_sizes["search_results"],
            "cache_ttl": 300,
            "enable_compression": True,
            "cors_enabled": True
        }
        
        # Indexer service configuration
        configs["indexer"] = {
            "memory_limit": performance_profile.memory_allocation["indexer"],
            "concurrent_workers": performance_profile.indexing_settings["concurrent_indexers"],
            "batch_size": performance_profile.indexing_settings["batch_size"],
            "index_frequency": performance_profile.indexing_settings["index_frequency"],
            "file_size_limit": "10MB",
            "ignore_patterns": self._get_ignore_patterns_for_project(analysis),
            "language_detection": True,
            "content_extraction": True,
            "incremental_updates": True
        }
        
        return configs
    
    def _generate_environment_variables(self, analysis: ProjectAnalysis,
                                      deployment_profile: DeploymentProfile,
                                      capabilities: SystemCapabilities) -> Dict[str, str]:
        """Generate environment variables"""
        
        env_vars = {
            # Project information
            "PROJECT_NAME": analysis.project_name,
            "PROJECT_PATH": analysis.project_path,
            "PROJECT_TYPE": analysis.project_type.value,
            "DEPLOYMENT_PROFILE": deployment_profile.value,
            
            # Database
            "POSTGRES_DB": f"{analysis.project_name}_index",
            "POSTGRES_USER": "project_index",
            "POSTGRES_PASSWORD": self._generate_secure_password(),
            
            # Redis
            "REDIS_PASSWORD": self._generate_secure_password(),
            
            # API
            "API_SECRET_KEY": self._generate_secure_password(64),
            "API_DEBUG": "false",
            "API_LOG_LEVEL": "INFO",
            
            # System
            "PYTHONPATH": "/app",
            "TZ": "UTC",
            
            # Performance
            "WORKERS": str(min(capabilities.cpu_cores, 4)),
            "MEMORY_LIMIT": f"{int(capabilities.available_memory_gb * 0.8)}g"
        }
        
        return env_vars
    
    def _generate_feature_flags(self, optimization_level: OptimizationLevel,
                              analysis: ProjectAnalysis,
                              capabilities: SystemCapabilities) -> Dict[str, bool]:
        """Generate feature flags based on configuration"""
        
        flags = {
            # Core features
            "enable_real_time_indexing": True,
            "enable_search_api": True,
            "enable_file_monitoring": True,
            
            # Performance features
            "enable_query_caching": True,
            "enable_result_caching": True,
            "enable_compression": True,
            
            # Advanced features
            "enable_ml_analysis": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE],
            "enable_semantic_search": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE],
            "enable_code_intelligence": optimization_level != OptimizationLevel.MINIMAL,
            
            # Security features
            "enable_api_authentication": optimization_level != OptimizationLevel.MINIMAL,
            "enable_rate_limiting": True,
            "enable_audit_logging": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE],
            
            # Monitoring features
            "enable_metrics": optimization_level != OptimizationLevel.MINIMAL,
            "enable_health_checks": True,
            "enable_performance_monitoring": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE],
            
            # Enterprise features
            "enable_multi_user": optimization_level == OptimizationLevel.ENTERPRISE,
            "enable_access_control": optimization_level == OptimizationLevel.ENTERPRISE,
            "enable_backup": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE]
        }
        
        # Project-specific features
        if analysis.project_type == ProjectType.MONOREPO:
            flags["enable_workspace_detection"] = True
            flags["enable_multi_project_support"] = True
        
        if Language.PYTHON in analysis.languages:
            flags["enable_python_ast_analysis"] = True
        
        if any(f.name in ["React", "Vue.js", "Angular"] for f in analysis.frameworks):
            flags["enable_frontend_analysis"] = True
        
        return flags
    
    def _generate_monitoring_config(self, optimization_level: OptimizationLevel,
                                  deployment_profile: DeploymentProfile) -> Dict[str, Any]:
        """Generate monitoring configuration"""
        
        if optimization_level == OptimizationLevel.MINIMAL:
            return {"enabled": False}
        
        config = {
            "enabled": True,
            "metrics": {
                "enabled": True,
                "retention_days": 7 if optimization_level == OptimizationLevel.BALANCED else 30,
                "scrape_interval": "30s",
                "evaluation_interval": "30s"
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "retention_days": 7 if optimization_level == OptimizationLevel.BALANCED else 30
            },
            "health_checks": {
                "enabled": True,
                "interval": "30s",
                "timeout": "10s",
                "unhealthy_threshold": 3
            },
            "alerts": {
                "enabled": optimization_level in [OptimizationLevel.PERFORMANCE, OptimizationLevel.ENTERPRISE],
                "channels": ["log"],  # Can be extended to email, slack, etc.
                "rules": [
                    {
                        "name": "High Memory Usage",
                        "condition": "memory_usage > 90",
                        "severity": "warning"
                    },
                    {
                        "name": "API Response Time",
                        "condition": "api_response_time > 5000",
                        "severity": "warning"
                    },
                    {
                        "name": "Indexing Errors",
                        "condition": "indexing_error_rate > 0.1",
                        "severity": "critical"
                    }
                ]
            }
        }
        
        return config
    
    def _generate_backup_config(self, optimization_level: OptimizationLevel,
                              deployment_profile: DeploymentProfile) -> Dict[str, Any]:
        """Generate backup configuration"""
        
        if optimization_level == OptimizationLevel.MINIMAL:
            return {"enabled": False}
        
        config = {
            "enabled": True,
            "schedule": {
                "database": "0 2 * * *",  # Daily at 2 AM
                "index": "0 3 * * 0",     # Weekly on Sunday at 3 AM
                "config": "0 4 * * 0"     # Weekly on Sunday at 4 AM
            },
            "retention": {
                "daily": 7,
                "weekly": 4,
                "monthly": 3 if optimization_level == OptimizationLevel.ENTERPRISE else 1
            },
            "compression": True,
            "encryption": optimization_level == OptimizationLevel.ENTERPRISE,
            "destinations": ["local"],
            "verification": optimization_level == OptimizationLevel.ENTERPRISE
        }
        
        return config
    
    def _generate_integration_configs(self, analysis: ProjectAnalysis) -> Dict[str, Dict[str, Any]]:
        """Generate framework-specific integration configurations"""
        
        configs = {}
        
        for framework in analysis.frameworks:
            framework_name = framework.name.lower()
            
            if framework_name == "flask":
                configs["flask"] = {
                    "middleware_enabled": True,
                    "auto_context_tracking": True,
                    "cli_commands": True,
                    "template_integration": True
                }
            
            elif framework_name == "react":
                configs["react"] = {
                    "hooks_enabled": True,
                    "context_provider": True,
                    "dev_tools": True,
                    "hot_reload_integration": True
                }
            
            elif framework_name == "vue.js":
                configs["vue"] = {
                    "plugin_enabled": True,
                    "composition_api": True,
                    "devtools_integration": True
                }
            
            elif framework_name == "django":
                configs["django"] = {
                    "middleware_enabled": True,
                    "admin_integration": True,
                    "management_commands": True,
                    "template_tags": True
                }
        
        return configs
    
    def _get_ignore_patterns_for_project(self, analysis: ProjectAnalysis) -> List[str]:
        """Get appropriate ignore patterns for the project"""
        
        base_patterns = [
            ".git/", "node_modules/", "__pycache__/", ".pytest_cache/",
            "venv/", ".venv/", "env/", ".env", "dist/", "build/",
            "target/", ".idea/", ".vscode/", "coverage/", ".coverage",
            "*.egg-info/", ".tox/", ".mypy_cache/"
        ]
        
        # Language-specific patterns
        if Language.JAVASCRIPT in analysis.languages or Language.TYPESCRIPT in analysis.languages:
            base_patterns.extend([
                "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*",
                ".next/", ".nuxt/", "out/", "public/build/"
            ])
        
        if Language.PYTHON in analysis.languages:
            base_patterns.extend([
                "*.pyc", "*.pyo", "*.pyd", "__pycache__/",
                ".Python", "build/", "develop-eggs/", "downloads/",
                "eggs/", ".eggs/", "lib/", "lib64/", "parts/",
                "sdist/", "var/", "wheels/"
            ])
        
        if Language.JAVA in analysis.languages:
            base_patterns.extend([
                "*.class", "*.jar", "*.war", "*.ear",
                "target/", ".gradle/", "build/"
            ])
        
        return base_patterns
    
    def _generate_secure_password(self, length: int = 32) -> str:
        """Generate a secure random password"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def export_configuration(self, config: ProjectIndexConfiguration, 
                           output_path: Path, formats: List[ConfigFormat]) -> List[Path]:
        """Export configuration to specified formats"""
        
        created_files = []
        
        # Prepare configuration data for export
        config_data = {
            "project": config.project_info,
            "deployment": {
                "profile": config.deployment_profile.value,
                "optimization_level": config.optimization_level.value
            },
            "services": config.service_configs,
            "security": asdict(config.security_config),
            "performance": asdict(config.performance_profile),
            "features": config.feature_flags,
            "monitoring": config.monitoring_config,
            "backup": config.backup_config,
            "integrations": config.integration_configs
        }
        
        for format_type in formats:
            if format_type == ConfigFormat.JSON:
                file_path = output_path / "project-index-config.json"
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
                created_files.append(file_path)
            
            elif format_type == ConfigFormat.YAML:
                file_path = output_path / "project-index-config.yaml"
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                created_files.append(file_path)
            
            elif format_type == ConfigFormat.ENV:
                file_path = output_path / ".env"
                with open(file_path, 'w') as f:
                    f.write("# Project Index Environment Configuration\\n")
                    f.write(f"# Generated on {datetime.now().isoformat()}\\n\\n")
                    
                    for key, value in config.environment_variables.items():
                        f.write(f"{key}={value}\\n")
                
                created_files.append(file_path)
            
            elif format_type == ConfigFormat.TOML:
                file_path = output_path / "project-index-config.toml"
                with open(file_path, 'w') as f:
                    toml.dump(config_data, f)
                created_files.append(file_path)
        
        self.logger.info(f"Exported configuration to {len(created_files)} files")
        return created_files
    
    def validate_configuration(self, config: ProjectIndexConfiguration) -> Tuple[bool, List[str]]:
        """Validate the generated configuration"""
        
        errors = []
        warnings = []
        
        # Validate memory allocations
        total_memory = sum(
            self._parse_memory_string(mem) 
            for mem in config.performance_profile.memory_allocation.values()
        )
        
        if total_memory > 8:  # More than 8GB seems excessive
            warnings.append(f"Total memory allocation is high: {total_memory:.1f}GB")
        
        # Validate CPU allocations
        total_cpu = sum(config.performance_profile.cpu_allocation.values())
        if total_cpu > 16:  # More than 16 CPU cores seems excessive
            warnings.append(f"Total CPU allocation is high: {total_cpu} cores")
        
        # Validate connection limits
        postgres_connections = config.performance_profile.connection_limits.get("postgres", 0)
        if postgres_connections > 200:
            warnings.append(f"PostgreSQL connection limit is very high: {postgres_connections}")
        
        # Validate security settings
        if not config.security_config.password_complexity["require_symbols"]:
            warnings.append("Password complexity could be stronger")
        
        if config.security_config.api_key_length < 32:
            warnings.append("API key length should be at least 32 characters")
        
        # Check for required features
        if not config.feature_flags.get("enable_health_checks", False):
            errors.append("Health checks should always be enabled")
        
        # Log warnings
        for warning in warnings:
            self.logger.warning(warning)
        
        return len(errors) == 0, errors
    
    def _parse_memory_string(self, memory_str: str) -> float:
        """Parse memory string like '1g', '512m' to GB"""
        memory_str = memory_str.lower()
        if memory_str.endswith('g'):
            return float(memory_str[:-1])
        elif memory_str.endswith('m'):
            return float(memory_str[:-1]) / 1024
        elif memory_str.endswith('k'):
            return float(memory_str[:-1]) / (1024 * 1024)
        else:
            return float(memory_str) / (1024 * 1024 * 1024)  # Assume bytes


# Example usage
if __name__ == "__main__":
    import sys
    from cli.project_detector import ProjectDetector
    
    if len(sys.argv) != 2:
        print("Usage: python config_generator.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    
    # Analyze project
    detector = ProjectDetector()
    analysis = detector.analyze_project(project_path)
    
    # Generate configuration
    generator = ConfigurationGenerator()
    config = generator.generate_configuration(
        analysis=analysis,
        deployment_profile=DeploymentProfile.MEDIUM,
        user_preferences={}
    )
    
    print(f"Generated configuration for {config.project_info['name']}")
    print(f"Optimization level: {config.optimization_level.value}")
    print(f"Performance profile: {config.performance_profile.name}")
    
    # Validate configuration
    valid, errors = generator.validate_configuration(config)
    if valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Export configuration
    output_path = Path("./config_output")
    output_path.mkdir(exist_ok=True)
    
    formats = [ConfigFormat.JSON, ConfigFormat.YAML, ConfigFormat.ENV]
    created_files = generator.export_configuration(config, output_path, formats)
    
    print(f"\\nExported configuration to {len(created_files)} files:")
    for file_path in created_files:
        print(f"  - {file_path}")