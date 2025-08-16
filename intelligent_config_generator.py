#!/usr/bin/env python3
"""
Intelligent Configuration Generator
===================================

Advanced configuration template system that generates optimal Project Index configurations
based on project detection results, best practices, and performance considerations.

Features:
- Language-specific optimization templates
- Framework-aware configuration generation
- Performance scaling based on project size
- Security and compliance settings
- CI/CD integration templates
- Custom rule generation

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Configuration optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"


class SecurityLevel(Enum):
    """Security configuration levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


@dataclass
class PerformanceProfile:
    """Performance configuration profile."""
    max_concurrent_analyses: int
    analysis_batch_size: int
    cache_enabled: bool
    cache_ttl_seconds: int
    memory_limit_mb: int
    timeout_seconds: int
    debounce_interval: float


@dataclass
class SecurityProfile:
    """Security configuration profile."""
    scan_dependencies: bool
    check_vulnerabilities: bool
    validate_licenses: bool
    audit_sensitive_files: bool
    enable_sandboxing: bool
    restrict_file_access: bool
    security_reporting: bool


@dataclass
class MonitoringProfile:
    """Monitoring and observability configuration."""
    enable_metrics: bool
    enable_health_checks: bool
    enable_logging: bool
    log_level: str
    metrics_retention_days: int
    performance_tracking: bool
    error_alerting: bool


@dataclass
class ProjectIndexConfiguration:
    """Complete Project Index configuration."""
    # Project metadata
    project_name: str
    project_path: str
    configuration_version: str
    generated_timestamp: str
    
    # Detection metadata
    detection_metadata: Dict[str, Any]
    
    # Core configuration
    analysis: Dict[str, Any]
    file_patterns: Dict[str, List[str]]
    ignore_patterns: List[str]
    monitoring: Dict[str, Any]
    optimization: Dict[str, Any]
    performance: Dict[str, Any]
    
    # Advanced settings
    security: Dict[str, Any]
    integrations: Dict[str, Any]
    custom_rules: List[Dict[str, Any]]
    
    # Documentation
    configuration_notes: List[str]
    recommendations: List[str]


class IntelligentConfigGenerator:
    """
    Intelligent configuration generator that creates optimal Project Index
    configurations based on project characteristics and requirements.
    """
    
    # Performance profiles for different project sizes
    PERFORMANCE_PROFILES = {
        'small': PerformanceProfile(
            max_concurrent_analyses=2,
            analysis_batch_size=25,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            memory_limit_mb=256,
            timeout_seconds=15,
            debounce_interval=1.0
        ),
        'medium': PerformanceProfile(
            max_concurrent_analyses=4,
            analysis_batch_size=50,
            cache_enabled=True,
            cache_ttl_seconds=7200,
            memory_limit_mb=512,
            timeout_seconds=30,
            debounce_interval=2.0
        ),
        'large': PerformanceProfile(
            max_concurrent_analyses=6,
            analysis_batch_size=100,
            cache_enabled=True,
            cache_ttl_seconds=14400,
            memory_limit_mb=1024,
            timeout_seconds=60,
            debounce_interval=3.0
        ),
        'enterprise': PerformanceProfile(
            max_concurrent_analyses=8,
            analysis_batch_size=200,
            cache_enabled=True,
            cache_ttl_seconds=21600,
            memory_limit_mb=2048,
            timeout_seconds=120,
            debounce_interval=5.0
        )
    }
    
    # Security profiles
    SECURITY_PROFILES = {
        SecurityLevel.BASIC: SecurityProfile(
            scan_dependencies=False,
            check_vulnerabilities=False,
            validate_licenses=False,
            audit_sensitive_files=False,
            enable_sandboxing=False,
            restrict_file_access=False,
            security_reporting=False
        ),
        SecurityLevel.STANDARD: SecurityProfile(
            scan_dependencies=True,
            check_vulnerabilities=True,
            validate_licenses=False,
            audit_sensitive_files=True,
            enable_sandboxing=False,
            restrict_file_access=False,
            security_reporting=True
        ),
        SecurityLevel.STRICT: SecurityProfile(
            scan_dependencies=True,
            check_vulnerabilities=True,
            validate_licenses=True,
            audit_sensitive_files=True,
            enable_sandboxing=True,
            restrict_file_access=True,
            security_reporting=True
        ),
        SecurityLevel.ENTERPRISE: SecurityProfile(
            scan_dependencies=True,
            check_vulnerabilities=True,
            validate_licenses=True,
            audit_sensitive_files=True,
            enable_sandboxing=True,
            restrict_file_access=True,
            security_reporting=True
        )
    }
    
    # Language-specific configurations
    LANGUAGE_CONFIGS = {
        'python': {
            'extensions': ['.py', '.pyi', '.pyx', '.pyw'],
            'ignore_patterns': [
                '**/__pycache__/**',
                '**/*.pyc',
                '**/*.pyo',
                '**/.venv/**',
                '**/venv/**',
                '**/env/**',
                '**/.pytest_cache/**',
                '**/build/**',
                '**/dist/**',
                '**/*.egg-info/**'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_style': True,
                'extract_docstrings': True
            },
            'security_patterns': [
                '**/*secret*',
                '**/*password*',
                '**/*token*',
                '**/.env',
                '**/config.py'
            ]
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.mjs', '.cjs'],
            'ignore_patterns': [
                '**/node_modules/**',
                '**/build/**',
                '**/dist/**',
                '**/.next/**',
                '**/coverage/**',
                '**/*.min.js',
                '**/*.bundle.js'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_style': False,
                'extract_jsdoc': True
            },
            'security_patterns': [
                '**/.env*',
                '**/config.js',
                '**/secrets.js'
            ]
        },
        'typescript': {
            'extensions': ['.ts', '.tsx', '.d.ts'],
            'ignore_patterns': [
                '**/node_modules/**',
                '**/build/**',
                '**/dist/**',
                '**/.next/**',
                '**/coverage/**',
                '**/*.js.map',
                '**/*.d.ts.map'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_types': True,
                'extract_tsdoc': True
            },
            'security_patterns': [
                '**/.env*',
                '**/config.ts',
                '**/secrets.ts'
            ]
        },
        'go': {
            'extensions': ['.go'],
            'ignore_patterns': [
                '**/vendor/**',
                '**/bin/**',
                '**/*.exe',
                '**/*.so',
                '**/*.dylib'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_formatting': True,
                'extract_godoc': True
            },
            'security_patterns': [
                '**/*secret*',
                '**/*password*',
                '**/*token*'
            ]
        },
        'rust': {
            'extensions': ['.rs'],
            'ignore_patterns': [
                '**/target/**',
                '**/*.rlib',
                '**/*.rmeta',
                '**/Cargo.lock'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_clippy': True,
                'extract_rustdoc': True
            },
            'security_patterns': [
                '**/*secret*',
                '**/*password*',
                '**/*token*'
            ]
        },
        'java': {
            'extensions': ['.java'],
            'ignore_patterns': [
                '**/target/**',
                '**/build/**',
                '**/*.class',
                '**/*.jar'
            ],
            'analysis_settings': {
                'parse_ast': True,
                'extract_imports': True,
                'analyze_complexity': True,
                'check_style': True,
                'extract_javadoc': True
            },
            'security_patterns': [
                '**/*secret*',
                '**/*password*',
                '**/*token*',
                '**/application.properties',
                '**/application.yml'
            ]
        }
    }
    
    # Framework-specific enhancements
    FRAMEWORK_CONFIGS = {
        'django': {
            'additional_patterns': ['**/migrations/**', '**/static/**', '**/templates/**'],
            'ignore_patterns': ['**/migrations/**', '**/static/admin/**'],
            'security_focus': ['**/settings.py', '**/urls.py', '**/models.py'],
            'performance_settings': {
                'priority_files': ['models.py', 'views.py', 'urls.py'],
                'cache_templates': True
            }
        },
        'react': {
            'additional_patterns': ['**/components/**', '**/hooks/**', '**/contexts/**'],
            'ignore_patterns': ['**/build/**', '**/.next/**'],
            'security_focus': ['.env*', 'config.js'],
            'performance_settings': {
                'priority_files': ['App.js', 'index.js'],
                'bundle_analysis': True
            }
        },
        'express': {
            'additional_patterns': ['**/routes/**', '**/middleware/**', '**/controllers/**'],
            'ignore_patterns': ['**/node_modules/**', '**/logs/**'],
            'security_focus': ['config/', 'middleware/', '.env*'],
            'performance_settings': {
                'priority_files': ['app.js', 'server.js'],
                'monitor_endpoints': True
            }
        },
        'spring': {
            'additional_patterns': ['**/src/main/java/**', '**/src/main/resources/**'],
            'ignore_patterns': ['**/target/**', '**/logs/**'],
            'security_focus': ['application.properties', 'application.yml'],
            'performance_settings': {
                'priority_files': ['Application.java', 'Controller.java'],
                'jvm_monitoring': True
            }
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent configuration generator."""
        self.config = config or {}
        self.default_optimization_level = OptimizationLevel(
            self.config.get('optimization_level', 'balanced')
        )
        self.default_security_level = SecurityLevel(
            self.config.get('security_level', 'standard')
        )
        
        logger.info("Intelligent configuration generator initialized")
    
    def generate_configuration(
        self,
        detection_result: Dict[str, Any],
        optimization_level: Optional[OptimizationLevel] = None,
        security_level: Optional[SecurityLevel] = None,
        custom_overrides: Optional[Dict[str, Any]] = None
    ) -> ProjectIndexConfiguration:
        """
        Generate comprehensive Project Index configuration.
        
        Args:
            detection_result: Results from project detection
            optimization_level: Performance optimization level
            security_level: Security configuration level
            custom_overrides: Custom configuration overrides
            
        Returns:
            Complete Project Index configuration
        """
        optimization_level = optimization_level or self.default_optimization_level
        security_level = security_level or self.default_security_level
        custom_overrides = custom_overrides or {}
        
        logger.info("Generating intelligent configuration",
                   optimization=optimization_level.value,
                   security=security_level.value)
        
        # Extract detection information
        project_path = detection_result.get('project_path', '.')
        project_name = Path(project_path).name
        primary_language = detection_result.get('primary_language', {}).get('language')
        frameworks = [f.get('framework') for f in detection_result.get('detected_frameworks', [])]
        project_size = detection_result.get('size_analysis', {}).get('size_category', 'medium')
        
        # Generate core configuration components
        analysis_config = self._generate_analysis_config(
            primary_language, frameworks, project_size, optimization_level
        )
        
        file_patterns = self._generate_file_patterns(
            primary_language, frameworks, detection_result
        )
        
        ignore_patterns = self._generate_ignore_patterns(
            primary_language, frameworks, detection_result
        )
        
        monitoring_config = self._generate_monitoring_config(
            project_size, optimization_level
        )
        
        optimization_config = self._generate_optimization_config(
            project_size, optimization_level
        )
        
        performance_config = self._generate_performance_config(
            project_size, optimization_level
        )
        
        security_config = self._generate_security_config(
            primary_language, frameworks, security_level
        )
        
        integrations_config = self._generate_integrations_config(
            frameworks, detection_result
        )
        
        custom_rules = self._generate_custom_rules(
            primary_language, frameworks, detection_result
        )
        
        # Generate documentation
        config_notes, recommendations = self._generate_documentation(
            detection_result, optimization_level, security_level
        )
        
        # Apply custom overrides
        if custom_overrides:
            analysis_config = self._apply_overrides(analysis_config, custom_overrides.get('analysis', {}))
            file_patterns = self._apply_overrides(file_patterns, custom_overrides.get('file_patterns', {}))
            # Apply other overrides as needed
        
        from datetime import datetime
        
        return ProjectIndexConfiguration(
            project_name=project_name,
            project_path=project_path,
            configuration_version="2.0",
            generated_timestamp=datetime.utcnow().isoformat(),
            detection_metadata={
                'primary_language': primary_language,
                'frameworks': frameworks,
                'project_size': project_size,
                'confidence_score': detection_result.get('confidence_score', 0.0),
                'optimization_level': optimization_level.value,
                'security_level': security_level.value
            },
            analysis=analysis_config,
            file_patterns=file_patterns,
            ignore_patterns=ignore_patterns,
            monitoring=monitoring_config,
            optimization=optimization_config,
            performance=performance_config,
            security=security_config,
            integrations=integrations_config,
            custom_rules=custom_rules,
            configuration_notes=config_notes,
            recommendations=recommendations
        )
    
    def _generate_analysis_config(
        self,
        primary_language: Optional[str],
        frameworks: List[str],
        project_size: str,
        optimization_level: OptimizationLevel
    ) -> Dict[str, Any]:
        """Generate analysis configuration."""
        
        # Base configuration
        config = {
            'enabled': True,
            'parse_ast': True,
            'extract_dependencies': True,
            'calculate_complexity': True,
            'max_file_size_mb': 10,
            'max_line_count': 50000,
            'timeout_seconds': 30,
            'parallel_processing': True
        }
        
        # Language-specific settings
        if primary_language and primary_language in self.LANGUAGE_CONFIGS:
            lang_config = self.LANGUAGE_CONFIGS[primary_language]
            config.update(lang_config['analysis_settings'])
        
        # Optimization level adjustments
        if optimization_level == OptimizationLevel.MINIMAL:
            config.update({
                'parse_ast': False,
                'calculate_complexity': False,
                'timeout_seconds': 15
            })
        elif optimization_level == OptimizationLevel.PERFORMANCE:
            config.update({
                'max_file_size_mb': 20,
                'max_line_count': 100000,
                'timeout_seconds': 60,
                'parallel_processing': True,
                'aggressive_caching': True
            })
        elif optimization_level == OptimizationLevel.ENTERPRISE:
            config.update({
                'max_file_size_mb': 50,
                'max_line_count': 200000,
                'timeout_seconds': 120,
                'parallel_processing': True,
                'aggressive_caching': True,
                'deep_analysis': True,
                'cross_file_analysis': True
            })
        
        # Project size adjustments
        size_multipliers = {
            'small': 0.5,
            'medium': 1.0,
            'large': 1.5,
            'enterprise': 2.0
        }
        
        multiplier = size_multipliers.get(project_size, 1.0)
        config['timeout_seconds'] = int(config['timeout_seconds'] * multiplier)
        
        return config
    
    def _generate_file_patterns(
        self,
        primary_language: Optional[str],
        frameworks: List[str],
        detection_result: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate file inclusion patterns."""
        
        include_patterns = []
        exclude_patterns = []
        
        # Language-specific patterns
        if primary_language and primary_language in self.LANGUAGE_CONFIGS:
            lang_config = self.LANGUAGE_CONFIGS[primary_language]
            for ext in lang_config['extensions']:
                include_patterns.append(f"**/*{ext}")
        
        # Framework-specific patterns
        for framework in frameworks:
            if framework in self.FRAMEWORK_CONFIGS:
                framework_config = self.FRAMEWORK_CONFIGS[framework]
                include_patterns.extend(framework_config.get('additional_patterns', []))
        
        # Always include common configuration files
        include_patterns.extend([
            '**/package.json',
            '**/requirements.txt',
            '**/pyproject.toml',
            '**/Cargo.toml',
            '**/go.mod',
            '**/pom.xml',
            '**/build.gradle',
            '**/composer.json',
            '**/*.yml',
            '**/*.yaml',
            '**/*.toml',
            '**/*.json',
            '**/Makefile',
            '**/Dockerfile',
            '**/*.md'
        ])
        
        # Default exclude patterns
        exclude_patterns.extend([
            '**/.*',  # Hidden files/directories
            '**/*.log',
            '**/*.tmp',
            '**/*.temp',
            '**/backup/**',
            '**/backups/**'
        ])
        
        return {
            'include': list(set(include_patterns)),  # Remove duplicates
            'exclude': list(set(exclude_patterns))
        }
    
    def _generate_ignore_patterns(
        self,
        primary_language: Optional[str],
        frameworks: List[str],
        detection_result: Dict[str, Any]
    ) -> List[str]:
        """Generate ignore patterns."""
        
        ignore_patterns = [
            # Version control
            '**/.git/**',
            '**/.svn/**',
            '**/.hg/**',
            
            # General build artifacts
            '**/build/**',
            '**/dist/**',
            '**/out/**',
            '**/output/**',
            
            # Logs
            '**/*.log',
            '**/logs/**',
            
            # Temporary files
            '**/*.tmp',
            '**/*.temp',
            '**/*~',
            
            # IDE files
            '**/.vscode/**',
            '**/.idea/**',
            '**/*.sublime-*',
            
            # OS files
            '**/.DS_Store',
            '**/Thumbs.db',
            '**/desktop.ini'
        ]
        
        # Language-specific ignore patterns
        if primary_language and primary_language in self.LANGUAGE_CONFIGS:
            lang_config = self.LANGUAGE_CONFIGS[primary_language]
            ignore_patterns.extend(lang_config['ignore_patterns'])
        
        # Framework-specific ignore patterns
        for framework in frameworks:
            if framework in self.FRAMEWORK_CONFIGS:
                framework_config = self.FRAMEWORK_CONFIGS[framework]
                ignore_patterns.extend(framework_config.get('ignore_patterns', []))
        
        return list(set(ignore_patterns))  # Remove duplicates
    
    def _generate_monitoring_config(
        self,
        project_size: str,
        optimization_level: OptimizationLevel
    ) -> Dict[str, Any]:
        """Generate monitoring configuration."""
        
        base_config = {
            'enabled': True,
            'debounce_seconds': 2.0,
            'watch_subdirectories': True,
            'max_file_size_mb': 10,
            'batch_events': True,
            'event_queue_size': 1000
        }
        
        # Size-based adjustments
        size_configs = {
            'small': {'debounce_seconds': 1.0, 'event_queue_size': 100},
            'medium': {'debounce_seconds': 2.0, 'event_queue_size': 500},
            'large': {'debounce_seconds': 3.0, 'event_queue_size': 1000},
            'enterprise': {'debounce_seconds': 5.0, 'event_queue_size': 2000}
        }
        
        if project_size in size_configs:
            base_config.update(size_configs[project_size])
        
        # Optimization level adjustments
        if optimization_level == OptimizationLevel.PERFORMANCE:
            base_config.update({
                'parallel_monitoring': True,
                'advanced_filtering': True,
                'predictive_analysis': True
            })
        elif optimization_level == OptimizationLevel.ENTERPRISE:
            base_config.update({
                'parallel_monitoring': True,
                'advanced_filtering': True,
                'predictive_analysis': True,
                'distributed_monitoring': True,
                'metrics_collection': True
            })
        
        return base_config
    
    def _generate_optimization_config(
        self,
        project_size: str,
        optimization_level: OptimizationLevel
    ) -> Dict[str, Any]:
        """Generate optimization configuration."""
        
        base_config = {
            'context_optimization_enabled': True,
            'max_context_files': 50,
            'relevance_threshold': 0.7,
            'intelligent_caching': True,
            'incremental_updates': True
        }
        
        # Size-based adjustments
        size_configs = {
            'small': {'max_context_files': 25, 'relevance_threshold': 0.6},
            'medium': {'max_context_files': 50, 'relevance_threshold': 0.7},
            'large': {'max_context_files': 100, 'relevance_threshold': 0.75},
            'enterprise': {'max_context_files': 200, 'relevance_threshold': 0.8}
        }
        
        if project_size in size_configs:
            base_config.update(size_configs[project_size])
        
        # Optimization level enhancements
        if optimization_level == OptimizationLevel.PERFORMANCE:
            base_config.update({
                'aggressive_optimization': True,
                'predictive_loading': True,
                'smart_prioritization': True
            })
        elif optimization_level == OptimizationLevel.ENTERPRISE:
            base_config.update({
                'aggressive_optimization': True,
                'predictive_loading': True,
                'smart_prioritization': True,
                'ml_optimization': True,
                'distributed_optimization': True
            })
        
        return base_config
    
    def _generate_performance_config(
        self,
        project_size: str,
        optimization_level: OptimizationLevel
    ) -> Dict[str, Any]:
        """Generate performance configuration."""
        
        # Get base performance profile
        profile = self.PERFORMANCE_PROFILES.get(project_size, self.PERFORMANCE_PROFILES['medium'])
        
        config = {
            'max_concurrent_analyses': profile.max_concurrent_analyses,
            'analysis_batch_size': profile.analysis_batch_size,
            'cache_enabled': profile.cache_enabled,
            'cache_ttl_seconds': profile.cache_ttl_seconds,
            'memory_limit_mb': profile.memory_limit_mb,
            'timeout_seconds': profile.timeout_seconds,
            'debounce_interval': profile.debounce_interval,
            'batch_insert_size': profile.analysis_batch_size * 2
        }
        
        # Optimization level adjustments
        if optimization_level == OptimizationLevel.PERFORMANCE:
            config.update({
                'max_concurrent_analyses': config['max_concurrent_analyses'] * 2,
                'aggressive_caching': True,
                'memory_optimization': True,
                'cpu_optimization': True
            })
        elif optimization_level == OptimizationLevel.ENTERPRISE:
            config.update({
                'max_concurrent_analyses': config['max_concurrent_analyses'] * 3,
                'analysis_batch_size': config['analysis_batch_size'] * 2,
                'memory_limit_mb': config['memory_limit_mb'] * 2,
                'aggressive_caching': True,
                'memory_optimization': True,
                'cpu_optimization': True,
                'distributed_processing': True,
                'load_balancing': True
            })
        
        return config
    
    def _generate_security_config(
        self,
        primary_language: Optional[str],
        frameworks: List[str],
        security_level: SecurityLevel
    ) -> Dict[str, Any]:
        """Generate security configuration."""
        
        # Get base security profile
        profile = self.SECURITY_PROFILES[security_level]
        
        config = {
            'enabled': True,
            'scan_dependencies': profile.scan_dependencies,
            'check_vulnerabilities': profile.check_vulnerabilities,
            'validate_licenses': profile.validate_licenses,
            'audit_sensitive_files': profile.audit_sensitive_files,
            'enable_sandboxing': profile.enable_sandboxing,
            'restrict_file_access': profile.restrict_file_access,
            'security_reporting': profile.security_reporting
        }
        
        # Language-specific security patterns
        sensitive_patterns = []
        if primary_language and primary_language in self.LANGUAGE_CONFIGS:
            lang_config = self.LANGUAGE_CONFIGS[primary_language]
            sensitive_patterns.extend(lang_config.get('security_patterns', []))
        
        # Framework-specific security patterns
        for framework in frameworks:
            if framework in self.FRAMEWORK_CONFIGS:
                framework_config = self.FRAMEWORK_CONFIGS[framework]
                sensitive_patterns.extend(framework_config.get('security_focus', []))
        
        if sensitive_patterns:
            config['sensitive_file_patterns'] = list(set(sensitive_patterns))
        
        # Additional security settings based on level
        if security_level in [SecurityLevel.STRICT, SecurityLevel.ENTERPRISE]:
            config.update({
                'code_analysis_security': True,
                'dependency_audit': True,
                'secret_detection': True,
                'compliance_checking': True
            })
        
        return config
    
    def _generate_integrations_config(
        self,
        frameworks: List[str],
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate integrations configuration."""
        
        config = {
            'enabled': True,
            'webhooks': {
                'enabled': False,
                'endpoints': []
            },
            'ci_cd': {
                'enabled': False,
                'platforms': []
            },
            'ide': {
                'vscode_extension': True,
                'intellij_plugin': False
            },
            'monitoring': {
                'prometheus': False,
                'grafana': False,
                'datadog': False
            }
        }
        
        # Detect CI/CD platforms from build system analysis
        build_analysis = detection_result.get('build_system_analysis', {})
        ci_cd_files = build_analysis.get('ci_cd_files', [])
        
        detected_platforms = []
        if any('github' in f for f in ci_cd_files):
            detected_platforms.append('github_actions')
        if any('gitlab' in f for f in ci_cd_files):
            detected_platforms.append('gitlab_ci')
        if any('jenkins' in f for f in ci_cd_files):
            detected_platforms.append('jenkins')
        
        if detected_platforms:
            config['ci_cd']['enabled'] = True
            config['ci_cd']['platforms'] = detected_platforms
        
        # Framework-specific integrations
        if 'prometheus' in str(detection_result).lower():
            config['monitoring']['prometheus'] = True
        
        return config
    
    def _generate_custom_rules(
        self,
        primary_language: Optional[str],
        frameworks: List[str],
        detection_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate custom analysis rules."""
        
        rules = []
        
        # Language-specific rules
        if primary_language == 'python':
            rules.extend([
                {
                    'name': 'django_migration_detector',
                    'pattern': '**/migrations/*.py',
                    'action': 'low_priority',
                    'reason': 'Django migrations are auto-generated'
                },
                {
                    'name': 'test_file_prioritization',
                    'pattern': '**/test_*.py',
                    'action': 'test_context',
                    'reason': 'Separate test analysis context'
                }
            ])
        
        elif primary_language == 'javascript':
            rules.extend([
                {
                    'name': 'minified_js_exclusion',
                    'pattern': '**/*.min.js',
                    'action': 'exclude',
                    'reason': 'Minified files are not source code'
                },
                {
                    'name': 'component_priority',
                    'pattern': '**/components/**/*.js',
                    'action': 'high_priority',
                    'reason': 'React components are core logic'
                }
            ])
        
        # Framework-specific rules
        if 'django' in frameworks:
            rules.append({
                'name': 'django_settings_security',
                'pattern': '**/settings.py',
                'action': 'security_scan',
                'reason': 'Django settings contain sensitive configuration'
            })
        
        if 'react' in frameworks:
            rules.append({
                'name': 'react_build_exclusion',
                'pattern': '**/build/**',
                'action': 'exclude',
                'reason': 'React build artifacts should be ignored'
            })
        
        # Project size-based rules
        project_size = detection_result.get('size_analysis', {}).get('size_category')
        if project_size == 'enterprise':
            rules.append({
                'name': 'enterprise_batch_processing',
                'pattern': '**/*',
                'action': 'batch_large',
                'reason': 'Enterprise projects need larger batch sizes'
            })
        
        return rules
    
    def _generate_documentation(
        self,
        detection_result: Dict[str, Any],
        optimization_level: OptimizationLevel,
        security_level: SecurityLevel
    ) -> Tuple[List[str], List[str]]:
        """Generate configuration documentation and recommendations."""
        
        notes = [
            f"Configuration generated for {optimization_level.value} optimization level",
            f"Security level set to {security_level.value}",
            "This configuration is automatically generated and can be customized",
            "Review security settings before deploying to production"
        ]
        
        recommendations = []
        
        # Size-based recommendations
        project_size = detection_result.get('size_analysis', {}).get('size_category')
        if project_size == 'enterprise':
            recommendations.extend([
                "Consider distributed processing for better performance",
                "Enable enterprise security features for compliance",
                "Set up monitoring and alerting for production use"
            ])
        elif project_size == 'small':
            recommendations.extend([
                "Current configuration is optimized for small projects",
                "Consider upgrading to balanced optimization as project grows"
            ])
        
        # Language-specific recommendations
        primary_language = detection_result.get('primary_language', {}).get('language')
        if primary_language == 'python':
            recommendations.append("Enable virtual environment detection for better dependency analysis")
        elif primary_language == 'javascript':
            recommendations.append("Consider enabling bundle analysis for better performance insights")
        
        # Framework-specific recommendations
        frameworks = [f.get('framework') for f in detection_result.get('detected_frameworks', [])]
        if 'django' in frameworks:
            recommendations.append("Enable Django-specific security scans for settings.py")
        if 'react' in frameworks:
            recommendations.append("Configure component-based analysis for better React insights")
        
        # Testing recommendations
        testing_analysis = detection_result.get('testing_analysis', {})
        if testing_analysis.get('testing_strategy') == 'none':
            recommendations.append("Consider adding unit tests to improve code quality analysis")
        
        return notes, recommendations
    
    def _apply_overrides(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom overrides to configuration."""
        result = base_config.copy()
        
        for key, value in overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._apply_overrides(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def export_configuration(
        self,
        config: ProjectIndexConfiguration,
        output_path: Path,
        format: str = 'json'
    ) -> None:
        """Export configuration to file."""
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Configuration exported to {output_path}")
    
    def generate_quick_start_script(
        self,
        config: ProjectIndexConfiguration,
        output_path: Path
    ) -> None:
        """Generate a quick start script for the configuration."""
        
        script_content = f"""#!/bin/bash
# Quick Start Script for Project Index
# Generated configuration for: {config.project_name}
# Optimization Level: {config.detection_metadata['optimization_level']}
# Security Level: {config.detection_metadata['security_level']}

echo "🚀 Setting up Project Index for {config.project_name}"

# Create configuration directory
mkdir -p .project-index
cp project-index-config.json .project-index/

# Initialize Project Index
echo "📊 Initializing Project Index..."
python -m project_index.cli init --config .project-index/project-index-config.json

# Start monitoring
echo "👁️  Starting file monitoring..."
python -m project_index.cli monitor --daemon

# Run initial analysis
echo "🔍 Running initial project analysis..."
python -m project_index.cli analyze --full

echo "✅ Project Index setup complete!"
echo "💡 View dashboard at: http://localhost:8000/dashboard"
echo "📚 Documentation: https://docs.leanvibe.dev/project-index"

# Configuration notes:
"""
        
        for note in config.configuration_notes:
            script_content += f"# - {note}\n"
        
        script_content += "\n# Recommendations:\n"
        for rec in config.recommendations:
            script_content += f"# - {rec}\n"
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        output_path.chmod(0o755)
        
        logger.info(f"Quick start script generated: {output_path}")


# CLI interface for standalone usage
def main():
    """CLI entry point for configuration generation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Intelligent Configuration Generator")
    parser.add_argument("detection_result", help="Path to detection result JSON file")
    parser.add_argument("--output", "-o", help="Output configuration file")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--optimization", choices=["minimal", "balanced", "performance", "enterprise"], 
                       default="balanced", help="Optimization level")
    parser.add_argument("--security", choices=["basic", "standard", "strict", "enterprise"], 
                       default="standard", help="Security level")
    parser.add_argument("--generate-script", action="store_true", help="Generate quick start script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load detection result
        with open(args.detection_result, 'r') as f:
            detection_result = json.load(f)
        
        # Generate configuration
        generator = IntelligentConfigGenerator()
        config = generator.generate_configuration(
            detection_result,
            OptimizationLevel(args.optimization),
            SecurityLevel(args.security)
        )
        
        # Export configuration
        if args.output:
            output_path = Path(args.output)
            generator.export_configuration(config, output_path, args.format)
            
            # Generate quick start script if requested
            if args.generate_script:
                script_path = output_path.parent / "setup-project-index.sh"
                generator.generate_quick_start_script(config, script_path)
            
            print(f"✅ Configuration generated: {output_path}")
        else:
            # Print to stdout
            config_dict = asdict(config)
            if args.format == 'json':
                print(json.dumps(config_dict, indent=2))
            else:
                print(yaml.dump(config_dict, default_flow_style=False))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()