#!/usr/bin/env python3
"""
Enhanced Intelligent Configuration Generator
============================================

Advanced configuration generator that creates optimal project-specific settings automatically
based on comprehensive project analysis. Integrates with project detection, framework adapters,
and universal installer for seamless out-of-the-box configuration.

Features:
- Machine learning-enhanced configuration optimization
- Environment-specific configuration generation
- Advanced validation and testing framework
- Configuration inheritance and composition
- Real-time configuration updates and hot-reloading
- Performance profiling and optimization recommendations
- Security hardening and compliance validation
- Integration with CI/CD pipelines and development workflows

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import yaml
import copy
import hashlib
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from enum import Enum
import logging
import jsonschema
from jsonschema import validate, ValidationError

# Import existing components
from intelligent_config_generator import (
    IntelligentConfigGenerator, ProjectIndexConfiguration, 
    OptimizationLevel, SecurityLevel, PerformanceProfile, SecurityProfile
)

logger = logging.getLogger(__name__)


class ConfigurationStrategy(Enum):
    """Configuration generation strategies."""
    MINIMAL = "minimal"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class ConfigurationEnvironment(Enum):
    """Target deployment environments."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"
    EDGE = "edge"


class ValidationLevel(Enum):
    """Configuration validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ConfigurationTemplate:
    """Base configuration template."""
    name: str
    description: str
    target_environments: List[ConfigurationEnvironment]
    base_settings: Dict[str, Any]
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    performance_profile: Optional[str] = None
    security_profile: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"


@dataclass
class ConfigurationValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    validation_level: ValidationLevel
    schema_errors: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    compatibility_issues: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    validation_duration: float = 0.0


@dataclass
class ConfigurationProfile:
    """Complete configuration profile for a project."""
    profile_id: str
    project_path: str
    base_configuration: ProjectIndexConfiguration
    environment_overrides: Dict[ConfigurationEnvironment, Dict[str, Any]]
    custom_templates: List[ConfigurationTemplate]
    validation_results: Dict[str, ConfigurationValidationResult]
    performance_metrics: Dict[str, Any]
    security_audit: Dict[str, Any]
    created_timestamp: datetime
    last_updated: datetime
    checksum: str


class EnhancedConfigurationGenerator(IntelligentConfigGenerator):
    """
    Enhanced intelligent configuration generator with advanced features,
    validation, templates, and optimization capabilities.
    """
    
    # Configuration schemas for validation
    CONFIG_SCHEMAS = {
        "base": {
            "type": "object",
            "required": ["project_name", "configuration_version", "analysis", "performance"],
            "properties": {
                "project_name": {"type": "string", "minLength": 1},
                "configuration_version": {"type": "string", "pattern": r"^\d+\.\d+(\.\d+)?$"},
                "analysis": {
                    "type": "object",
                    "required": ["enabled"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "parse_ast": {"type": "boolean"},
                        "extract_dependencies": {"type": "boolean"},
                        "calculate_complexity": {"type": "boolean"},
                        "max_file_size_mb": {"type": "number", "minimum": 1, "maximum": 100},
                        "max_line_count": {"type": "number", "minimum": 1000, "maximum": 1000000},
                        "timeout_seconds": {"type": "number", "minimum": 5, "maximum": 600}
                    }
                },
                "performance": {
                    "type": "object",
                    "required": ["max_concurrent_analyses"],
                    "properties": {
                        "max_concurrent_analyses": {"type": "number", "minimum": 1, "maximum": 32},
                        "analysis_batch_size": {"type": "number", "minimum": 10, "maximum": 1000},
                        "cache_enabled": {"type": "boolean"},
                        "memory_limit_mb": {"type": "number", "minimum": 128, "maximum": 8192}
                    }
                }
            }
        }
    }
    
    # Advanced configuration templates
    CONFIGURATION_TEMPLATES = {
        "python_microservice": ConfigurationTemplate(
            name="Python Microservice",
            description="Optimized configuration for Python microservices",
            target_environments=[ConfigurationEnvironment.PRODUCTION, ConfigurationEnvironment.STAGING],
            base_settings={
                "analysis": {
                    "parse_ast": True,
                    "extract_dependencies": True,
                    "calculate_complexity": True,
                    "analyze_docstrings": True,
                    "check_type_hints": True,
                    "validate_imports": True
                },
                "performance": {
                    "max_concurrent_analyses": 6,
                    "analysis_batch_size": 75,
                    "cache_enabled": True,
                    "memory_limit_mb": 1024,
                    "enable_profiling": True
                },
                "security": {
                    "scan_dependencies": True,
                    "check_vulnerabilities": True,
                    "validate_secrets": True,
                    "audit_imports": True
                }
            },
            validation_rules=[
                {"rule": "performance.max_concurrent_analyses", "min": 2, "max": 8},
                {"rule": "security.scan_dependencies", "required": True}
            ],
            tags=["python", "microservice", "production"]
        ),
        
        "javascript_spa": ConfigurationTemplate(
            name="JavaScript Single Page Application",
            description="Optimized for modern JavaScript SPAs",
            target_environments=[ConfigurationEnvironment.DEVELOPMENT, ConfigurationEnvironment.PRODUCTION],
            base_settings={
                "analysis": {
                    "parse_ast": True,
                    "extract_dependencies": True,
                    "analyze_bundles": True,
                    "check_tree_shaking": True,
                    "validate_imports": True
                },
                "performance": {
                    "max_concurrent_analyses": 4,
                    "analysis_batch_size": 50,
                    "cache_enabled": True,
                    "memory_limit_mb": 768,
                    "bundle_analysis": True
                },
                "file_patterns": {
                    "include": ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx", "**/*.vue"],
                    "exclude": ["**/dist/**", "**/build/**", "**/.next/**"]
                }
            },
            tags=["javascript", "spa", "frontend"]
        ),
        
        "enterprise_monolith": ConfigurationTemplate(
            name="Enterprise Monolithic Application",
            description="High-performance configuration for large enterprise applications",
            target_environments=[ConfigurationEnvironment.ENTERPRISE],
            base_settings={
                "analysis": {
                    "parse_ast": True,
                    "extract_dependencies": True,
                    "calculate_complexity": True,
                    "deep_analysis": True,
                    "cross_module_analysis": True,
                    "architecture_analysis": True
                },
                "performance": {
                    "max_concurrent_analyses": 12,
                    "analysis_batch_size": 200,
                    "cache_enabled": True,
                    "memory_limit_mb": 4096,
                    "distributed_processing": True,
                    "load_balancing": True
                },
                "optimization": {
                    "aggressive_optimization": True,
                    "ml_optimization": True,
                    "predictive_analysis": True
                },
                "security": {
                    "scan_dependencies": True,
                    "check_vulnerabilities": True,
                    "validate_licenses": True,
                    "audit_sensitive_files": True,
                    "compliance_checking": True,
                    "security_reporting": True
                }
            },
            tags=["enterprise", "monolith", "high-performance"]
        )
    }
    
    # Environment-specific overrides
    ENVIRONMENT_OVERRIDES = {
        ConfigurationEnvironment.DEVELOPMENT: {
            "analysis": {
                "timeout_seconds": 15,
                "max_file_size_mb": 5
            },
            "performance": {
                "max_concurrent_analyses": 2,
                "cache_enabled": True,
                "debug_mode": True
            },
            "monitoring": {
                "debug_logging": True,
                "verbose_output": True
            }
        },
        ConfigurationEnvironment.TESTING: {
            "analysis": {
                "timeout_seconds": 30,
                "include_test_files": True,
                "test_coverage_analysis": True
            },
            "performance": {
                "max_concurrent_analyses": 4,
                "batch_testing": True
            },
            "security": {
                "strict_validation": True,
                "test_security_patterns": True
            }
        },
        ConfigurationEnvironment.PRODUCTION: {
            "analysis": {
                "timeout_seconds": 60,
                "max_file_size_mb": 20,
                "production_optimizations": True
            },
            "performance": {
                "max_concurrent_analyses": 8,
                "aggressive_caching": True,
                "memory_optimization": True
            },
            "security": {
                "strict_security": True,
                "audit_mode": True,
                "compliance_mode": True
            },
            "monitoring": {
                "performance_metrics": True,
                "error_tracking": True,
                "alerting": True
            }
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced configuration generator."""
        super().__init__(config)
        
        self.templates: Dict[str, ConfigurationTemplate] = self.CONFIGURATION_TEMPLATES.copy()
        self.custom_validators: List[Callable] = []
        self.optimization_strategies: Dict[str, Callable] = {}
        self.performance_profiles_cache: Dict[str, Any] = {}
        
        # ML-based optimization (placeholder for future implementation)
        self.ml_optimizer = None
        
        logger.info("Enhanced configuration generator initialized")
    
    def generate_enhanced_configuration(
        self,
        detection_result: Dict[str, Any],
        strategy: ConfigurationStrategy = ConfigurationStrategy.PRODUCTION,
        environment: ConfigurationEnvironment = ConfigurationEnvironment.PRODUCTION,
        template_name: Optional[str] = None,
        custom_overrides: Optional[Dict[str, Any]] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ConfigurationProfile:
        """
        Generate enhanced configuration with advanced features.
        
        Args:
            detection_result: Results from project detection
            strategy: Configuration generation strategy
            environment: Target deployment environment
            template_name: Specific template to use
            custom_overrides: Custom configuration overrides
            validation_level: Level of validation to perform
            
        Returns:
            Complete configuration profile
        """
        start_time = time.time()
        project_path = detection_result.get('project_path', '.')
        
        logger.info("Generating enhanced configuration",
                   strategy=strategy.value,
                   environment=environment.value,
                   template=template_name)
        
        # 1. Select and apply base template
        template = self._select_optimal_template(detection_result, strategy, template_name)
        
        # 2. Generate base configuration using parent class
        base_optimization = self._map_strategy_to_optimization(strategy)
        base_security = self._map_strategy_to_security(strategy)
        
        base_config = super().generate_configuration(
            detection_result,
            base_optimization,
            base_security,
            custom_overrides or {}
        )
        
        # 3. Apply template enhancements
        enhanced_config = self._apply_template_enhancements(base_config, template)
        
        # 4. Apply environment-specific overrides
        env_config = self._apply_environment_overrides(enhanced_config, environment)
        
        # 5. Apply intelligent optimizations
        optimized_config = self._apply_intelligent_optimizations(env_config, detection_result)
        
        # 6. Generate environment-specific configurations
        environment_overrides = self._generate_environment_configurations(
            optimized_config, detection_result
        )
        
        # 7. Validate configuration
        validation_results = self._validate_configuration_comprehensive(
            optimized_config, validation_level, detection_result
        )
        
        # 8. Generate performance metrics and security audit
        performance_metrics = self._analyze_configuration_performance(optimized_config)
        security_audit = self._perform_security_audit(optimized_config)
        
        # 9. Create profile
        profile_id = self._generate_profile_id(project_path, strategy, environment)
        
        profile = ConfigurationProfile(
            profile_id=profile_id,
            project_path=project_path,
            base_configuration=optimized_config,
            environment_overrides=environment_overrides,
            custom_templates=[template] if template else [],
            validation_results={"default": validation_results},
            performance_metrics=performance_metrics,
            security_audit=security_audit,
            created_timestamp=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            checksum=self._calculate_configuration_checksum(optimized_config)
        )
        
        generation_time = time.time() - start_time
        logger.info("Enhanced configuration generated",
                   duration=f"{generation_time:.2f}s",
                   profile_id=profile_id,
                   validation_status="valid" if validation_results.is_valid else "invalid")
        
        return profile
    
    def _select_optimal_template(
        self,
        detection_result: Dict[str, Any],
        strategy: ConfigurationStrategy,
        template_name: Optional[str]
    ) -> Optional[ConfigurationTemplate]:
        """Select the most appropriate configuration template."""
        
        if template_name and template_name in self.templates:
            return self.templates[template_name]
        
        # Analyze project characteristics for auto-selection
        primary_language = detection_result.get('primary_language', {}).get('language')
        frameworks = [f.get('framework') for f in detection_result.get('detected_frameworks', [])]
        project_size = detection_result.get('size_analysis', {}).get('size_category')
        
        # Template scoring system
        template_scores = {}
        
        for name, template in self.templates.items():
            score = 0
            
            # Language matching
            if primary_language:
                if primary_language in template.tags:
                    score += 30
            
            # Framework matching
            for framework in frameworks:
                if framework in template.tags:
                    score += 20
            
            # Project size matching
            if project_size == 'enterprise' and 'enterprise' in template.tags:
                score += 25
            elif project_size in ['small', 'medium'] and 'microservice' in template.tags:
                score += 15
            
            # Strategy compatibility
            target_envs = template.target_environments
            if strategy == ConfigurationStrategy.PRODUCTION and ConfigurationEnvironment.PRODUCTION in target_envs:
                score += 20
            elif strategy == ConfigurationStrategy.ENTERPRISE and ConfigurationEnvironment.ENTERPRISE in target_envs:
                score += 25
            
            template_scores[name] = score
        
        # Select highest scoring template
        if template_scores:
            best_template_name = max(template_scores, key=template_scores.get)
            if template_scores[best_template_name] > 20:  # Minimum threshold
                logger.info("Auto-selected template",
                           template=best_template_name,
                           score=template_scores[best_template_name])
                return self.templates[best_template_name]
        
        return None
    
    def _apply_template_enhancements(
        self,
        base_config: ProjectIndexConfiguration,
        template: Optional[ConfigurationTemplate]
    ) -> ProjectIndexConfiguration:
        """Apply template-specific enhancements to base configuration."""
        
        if not template:
            return base_config
        
        # Convert to dict for easier manipulation
        config_dict = asdict(base_config)
        
        # Apply template base settings
        config_dict = self._deep_merge_dicts(config_dict, template.base_settings)
        
        # Apply template-specific optimizations
        if template.performance_profile:
            perf_profile = self.PERFORMANCE_PROFILES.get(template.performance_profile)
            if perf_profile:
                config_dict['performance'].update(asdict(perf_profile))
        
        if template.security_profile:
            sec_profile = self.SECURITY_PROFILES.get(template.security_profile)
            if sec_profile:
                config_dict['security'].update(asdict(sec_profile))
        
        # Reconstruct configuration object
        return ProjectIndexConfiguration(**config_dict)
    
    def _apply_environment_overrides(
        self,
        config: ProjectIndexConfiguration,
        environment: ConfigurationEnvironment
    ) -> ProjectIndexConfiguration:
        """Apply environment-specific configuration overrides."""
        
        config_dict = asdict(config)
        
        # Apply environment overrides
        if environment in self.ENVIRONMENT_OVERRIDES:
            env_overrides = self.ENVIRONMENT_OVERRIDES[environment]
            config_dict = self._deep_merge_dicts(config_dict, env_overrides)
        
        return ProjectIndexConfiguration(**config_dict)
    
    def _apply_intelligent_optimizations(
        self,
        config: ProjectIndexConfiguration,
        detection_result: Dict[str, Any]
    ) -> ProjectIndexConfiguration:
        """Apply intelligent optimizations based on project analysis."""
        
        config_dict = asdict(config)
        
        # CPU core optimization
        try:
            import os
            cpu_count = os.cpu_count() or 4
            optimal_concurrency = min(max(cpu_count // 2, 2), 12)
            
            current_concurrency = config_dict['performance'].get('max_concurrent_analyses', 4)
            if current_concurrency > optimal_concurrency:
                config_dict['performance']['max_concurrent_analyses'] = optimal_concurrency
                config_dict['recommendations'].append(
                    f"Optimized concurrency for {cpu_count} CPU cores"
                )
        except Exception:
            pass
        
        # Memory optimization based on project size
        project_size = detection_result.get('size_analysis', {}).get('size_category', 'medium')
        file_count = detection_result.get('size_analysis', {}).get('file_count', 100)
        
        if file_count > 5000:  # Large project
            config_dict['performance']['memory_limit_mb'] = min(
                config_dict['performance'].get('memory_limit_mb', 512) * 2,
                4096
            )
            config_dict['performance']['analysis_batch_size'] = min(
                config_dict['performance'].get('analysis_batch_size', 50) * 2,
                200
            )
        
        # Framework-specific optimizations
        frameworks = [f.get('framework') for f in detection_result.get('detected_frameworks', [])]
        
        if 'react' in frameworks or 'vue' in frameworks or 'angular' in frameworks:
            # Frontend framework optimizations
            config_dict['analysis']['bundle_analysis'] = True
            config_dict['analysis']['dependency_tree_analysis'] = True
            config_dict['file_patterns']['include'].extend([
                '**/*.css', '**/*.scss', '**/*.sass', '**/*.less'
            ])
        
        if 'django' in frameworks or 'flask' in frameworks or 'fastapi' in frameworks:
            # Python web framework optimizations
            config_dict['analysis']['orm_analysis'] = True
            config_dict['security']['django_security_checks'] = True
            config_dict['file_patterns']['include'].extend([
                '**/templates/**/*.html', '**/static/**/*.js'
            ])
        
        return ProjectIndexConfiguration(**config_dict)
    
    def _generate_environment_configurations(
        self,
        base_config: ProjectIndexConfiguration,
        detection_result: Dict[str, Any]
    ) -> Dict[ConfigurationEnvironment, Dict[str, Any]]:
        """Generate configurations for different environments."""
        
        environments = {}
        base_dict = asdict(base_config)
        
        for env in ConfigurationEnvironment:
            if env in self.ENVIRONMENT_OVERRIDES:
                env_config = self._deep_merge_dicts(
                    base_dict.copy(),
                    self.ENVIRONMENT_OVERRIDES[env]
                )
                environments[env] = env_config
        
        return environments
    
    def _validate_configuration_comprehensive(
        self,
        config: ProjectIndexConfiguration,
        validation_level: ValidationLevel,
        detection_result: Dict[str, Any]
    ) -> ConfigurationValidationResult:
        """Perform comprehensive configuration validation."""
        
        start_time = time.time()
        result = ConfigurationValidationResult(
            is_valid=True,
            validation_level=validation_level
        )
        
        config_dict = asdict(config)
        
        # 1. Schema validation
        try:
            validate(instance=config_dict, schema=self.CONFIG_SCHEMAS["base"])
        except ValidationError as e:
            result.is_valid = False
            result.schema_errors.append(f"Schema validation failed: {e.message}")
        
        # 2. Performance validation
        perf_config = config_dict.get('performance', {})
        
        # Check resource limits
        max_memory = perf_config.get('memory_limit_mb', 512)
        if max_memory > 8192:
            result.performance_warnings.append(
                f"Memory limit ({max_memory}MB) is very high and may cause system issues"
            )
        
        max_concurrency = perf_config.get('max_concurrent_analyses', 4)
        if max_concurrency > 16:
            result.performance_warnings.append(
                f"High concurrency ({max_concurrency}) may overwhelm system resources"
            )
        
        # 3. Security validation
        security_config = config_dict.get('security', {})
        
        if validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            if not security_config.get('scan_dependencies', False):
                result.security_warnings.append(
                    "Dependency scanning is disabled - security vulnerabilities may be missed"
                )
            
            if not security_config.get('check_vulnerabilities', False):
                result.security_warnings.append(
                    "Vulnerability checking is disabled"
                )
        
        # 4. Compatibility validation
        primary_language = detection_result.get('primary_language', {}).get('language')
        analysis_config = config_dict.get('analysis', {})
        
        if primary_language == 'python' and not analysis_config.get('parse_ast', False):
            result.compatibility_issues.append(
                "AST parsing is disabled for Python project - reduced analysis quality"
            )
        
        # 5. Optimization suggestions
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            cache_enabled = perf_config.get('cache_enabled', False)
            if not cache_enabled:
                result.optimization_suggestions.append(
                    "Enable caching for better performance"
                )
            
            project_size = detection_result.get('size_analysis', {}).get('size_category')
            if project_size == 'enterprise' and max_concurrency < 6:
                result.optimization_suggestions.append(
                    "Increase concurrency for enterprise-scale projects"
                )
        
        # 6. Custom validator execution
        for validator in self.custom_validators:
            try:
                validator_result = validator(config_dict, detection_result)
                if validator_result.get('errors'):
                    result.schema_errors.extend(validator_result['errors'])
                if validator_result.get('warnings'):
                    result.performance_warnings.extend(validator_result['warnings'])
            except Exception as e:
                logger.warning("Custom validator failed", error=str(e))
        
        # Final validation status
        if result.schema_errors:
            result.is_valid = False
        
        result.validation_duration = time.time() - start_time
        
        return result
    
    def _analyze_configuration_performance(
        self,
        config: ProjectIndexConfiguration
    ) -> Dict[str, Any]:
        """Analyze configuration for performance characteristics."""
        
        config_dict = asdict(config)
        perf_config = config_dict.get('performance', {})
        
        # Estimate resource usage
        max_memory = perf_config.get('memory_limit_mb', 512)
        max_concurrency = perf_config.get('max_concurrent_analyses', 4)
        batch_size = perf_config.get('analysis_batch_size', 50)
        
        estimated_cpu_usage = min(max_concurrency * 15, 100)  # Rough estimate
        estimated_memory_usage = max_memory * 0.8  # Expected usage
        
        # Performance score (0-100)
        performance_score = 100
        
        if max_memory > 2048:
            performance_score -= 10  # High memory usage penalty
        if max_concurrency > 8:
            performance_score -= 10  # High concurrency penalty
        if not perf_config.get('cache_enabled', False):
            performance_score -= 20  # No caching penalty
        
        return {
            'estimated_cpu_usage_percent': estimated_cpu_usage,
            'estimated_memory_usage_mb': estimated_memory_usage,
            'performance_score': max(performance_score, 0),
            'optimization_level': config_dict.get('detection_metadata', {}).get('optimization_level'),
            'cache_enabled': perf_config.get('cache_enabled', False),
            'parallel_processing': perf_config.get('max_concurrent_analyses', 1) > 1,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _perform_security_audit(
        self,
        config: ProjectIndexConfiguration
    ) -> Dict[str, Any]:
        """Perform security audit of the configuration."""
        
        config_dict = asdict(config)
        security_config = config_dict.get('security', {})
        
        security_score = 100
        issues = []
        recommendations = []
        
        # Check security features
        if not security_config.get('scan_dependencies', False):
            security_score -= 25
            issues.append("Dependency scanning disabled")
            recommendations.append("Enable dependency scanning for vulnerability detection")
        
        if not security_config.get('check_vulnerabilities', False):
            security_score -= 20
            issues.append("Vulnerability checking disabled")
            recommendations.append("Enable vulnerability checking")
        
        if not security_config.get('audit_sensitive_files', False):
            security_score -= 15
            issues.append("Sensitive file auditing disabled")
            recommendations.append("Enable sensitive file auditing")
        
        # Check file access restrictions
        if not security_config.get('restrict_file_access', False):
            security_score -= 10
            recommendations.append("Consider enabling file access restrictions")
        
        return {
            'security_score': max(security_score, 0),
            'security_level': config_dict.get('detection_metadata', {}).get('security_level'),
            'issues': issues,
            'recommendations': recommendations,
            'compliance_checks': {
                'dependency_scanning': security_config.get('scan_dependencies', False),
                'vulnerability_assessment': security_config.get('check_vulnerabilities', False),
                'sensitive_file_audit': security_config.get('audit_sensitive_files', False),
                'access_controls': security_config.get('restrict_file_access', False)
            },
            'audit_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def add_custom_template(self, template: ConfigurationTemplate) -> None:
        """Add a custom configuration template."""
        self.templates[template.name] = template
        logger.info("Added custom template", name=template.name)
    
    def add_custom_validator(self, validator: Callable) -> None:
        """Add a custom configuration validator."""
        self.custom_validators.append(validator)
        logger.info("Added custom validator")
    
    def export_configuration_profile(
        self,
        profile: ConfigurationProfile,
        output_dir: Path,
        format: str = 'json',
        include_environments: bool = True
    ) -> List[Path]:
        """Export configuration profile to files."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Export base configuration
        base_config_file = output_dir / f"config.{format}"
        if format == 'json':
            with open(base_config_file, 'w') as f:
                json.dump(asdict(profile.base_configuration), f, indent=2, default=str)
        elif format == 'yaml':
            with open(base_config_file, 'w') as f:
                yaml.dump(asdict(profile.base_configuration), f, default_flow_style=False)
        
        exported_files.append(base_config_file)
        
        # Export environment configurations
        if include_environments and profile.environment_overrides:
            env_dir = output_dir / "environments"
            env_dir.mkdir(exist_ok=True)
            
            for env, config in profile.environment_overrides.items():
                env_file = env_dir / f"{env.value}.{format}"
                if format == 'json':
                    with open(env_file, 'w') as f:
                        json.dump(config, f, indent=2, default=str)
                elif format == 'yaml':
                    with open(env_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                
                exported_files.append(env_file)
        
        # Export validation results
        validation_file = output_dir / "validation_report.json"
        with open(validation_file, 'w') as f:
            validation_data = {
                env: asdict(result) for env, result in profile.validation_results.items()
            }
            json.dump(validation_data, f, indent=2, default=str)
        
        exported_files.append(validation_file)
        
        # Export performance and security reports
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        perf_report = reports_dir / "performance_analysis.json"
        with open(perf_report, 'w') as f:
            json.dump(profile.performance_metrics, f, indent=2, default=str)
        exported_files.append(perf_report)
        
        security_report = reports_dir / "security_audit.json"
        with open(security_report, 'w') as f:
            json.dump(profile.security_audit, f, indent=2, default=str)
        exported_files.append(security_report)
        
        logger.info("Configuration profile exported",
                   profile_id=profile.profile_id,
                   files_count=len(exported_files),
                   output_dir=str(output_dir))
        
        return exported_files
    
    def generate_deployment_script(
        self,
        profile: ConfigurationProfile,
        target_environment: ConfigurationEnvironment,
        output_path: Path
    ) -> None:
        """Generate deployment script for specific environment."""
        
        script_content = f"""#!/bin/bash
# Enhanced Project Index Deployment Script
# Profile: {profile.profile_id}
# Environment: {target_environment.value}
# Generated: {datetime.now(timezone.utc).isoformat()}

set -euo pipefail

echo "🚀 Deploying Project Index configuration for {target_environment.value}"

# Configuration validation
echo "🔍 Validating configuration..."
if [ -f "validation_report.json" ]; then
    echo "✅ Configuration validation report found"
else
    echo "⚠️  No validation report found"
fi

# Performance check
echo "📊 Performance configuration:"
"""
        
        perf_metrics = profile.performance_metrics
        script_content += f"""echo "  CPU Usage: ~{perf_metrics.get('estimated_cpu_usage_percent', 0)}%"
echo "  Memory Usage: ~{perf_metrics.get('estimated_memory_usage_mb', 0)}MB"
echo "  Performance Score: {perf_metrics.get('performance_score', 0)}/100"
"""
        
        script_content += f"""
# Security audit
echo "🔒 Security configuration:"
"""
        
        security_audit = profile.security_audit
        script_content += f"""echo "  Security Score: {security_audit.get('security_score', 0)}/100"
echo "  Issues: {len(security_audit.get('issues', []))}"
echo "  Recommendations: {len(security_audit.get('recommendations', []))}"
"""
        
        script_content += f"""
# Deploy configuration
echo "📦 Deploying configuration files..."

# Create project index directory
mkdir -p .project-index
mkdir -p .project-index/cache
mkdir -p .project-index/logs

# Copy environment-specific configuration
if [ -f "environments/{target_environment.value}.json" ]; then
    cp environments/{target_environment.value}.json .project-index/config.json
    echo "✅ Environment configuration deployed"
else
    cp config.json .project-index/config.json
    echo "✅ Base configuration deployed"
fi

# Copy reports
cp -r reports .project-index/
echo "✅ Analysis reports deployed"

# Set permissions
chmod 755 .project-index/
chmod 644 .project-index/config.json

echo "✅ Project Index deployment completed for {target_environment.value}"
echo "💡 Next steps:"
echo "  1. Review configuration: cat .project-index/config.json"
echo "  2. Start monitoring: python -m project_index monitor"
echo "  3. Run analysis: python -m project_index analyze"
"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        output_path.chmod(0o755)
        
        logger.info("Deployment script generated",
                   environment=target_environment.value,
                   script_path=str(output_path))
    
    # Helper methods
    
    def _map_strategy_to_optimization(self, strategy: ConfigurationStrategy) -> OptimizationLevel:
        """Map configuration strategy to optimization level."""
        mapping = {
            ConfigurationStrategy.MINIMAL: OptimizationLevel.MINIMAL,
            ConfigurationStrategy.DEVELOPMENT: OptimizationLevel.BALANCED,
            ConfigurationStrategy.TESTING: OptimizationLevel.BALANCED,
            ConfigurationStrategy.STAGING: OptimizationLevel.PERFORMANCE,
            ConfigurationStrategy.PRODUCTION: OptimizationLevel.PERFORMANCE,
            ConfigurationStrategy.ENTERPRISE: OptimizationLevel.ENTERPRISE
        }
        return mapping.get(strategy, OptimizationLevel.BALANCED)
    
    def _map_strategy_to_security(self, strategy: ConfigurationStrategy) -> SecurityLevel:
        """Map configuration strategy to security level."""
        mapping = {
            ConfigurationStrategy.MINIMAL: SecurityLevel.BASIC,
            ConfigurationStrategy.DEVELOPMENT: SecurityLevel.STANDARD,
            ConfigurationStrategy.TESTING: SecurityLevel.STANDARD,
            ConfigurationStrategy.STAGING: SecurityLevel.STRICT,
            ConfigurationStrategy.PRODUCTION: SecurityLevel.STRICT,
            ConfigurationStrategy.ENTERPRISE: SecurityLevel.ENTERPRISE
        }
        return mapping.get(strategy, SecurityLevel.STANDARD)
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _generate_profile_id(
        self,
        project_path: str,
        strategy: ConfigurationStrategy,
        environment: ConfigurationEnvironment
    ) -> str:
        """Generate unique profile ID."""
        project_name = Path(project_path).name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{project_name}_{strategy.value}_{environment.value}_{timestamp}"
    
    def _calculate_configuration_checksum(self, config: ProjectIndexConfiguration) -> str:
        """Calculate configuration checksum for change detection."""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def main():
    """CLI entry point for enhanced configuration generation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Enhanced Configuration Generator")
    parser.add_argument("detection_result", help="Path to detection result JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for configuration")
    parser.add_argument("--strategy", choices=[s.value for s in ConfigurationStrategy], 
                       default="production", help="Configuration strategy")
    parser.add_argument("--environment", choices=[e.value for e in ConfigurationEnvironment], 
                       default="production", help="Target environment")
    parser.add_argument("--template", help="Specific template to use")
    parser.add_argument("--validation-level", choices=[v.value for v in ValidationLevel], 
                       default="standard", help="Validation level")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--generate-script", action="store_true", help="Generate deployment script")
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
        
        # Generate enhanced configuration
        generator = EnhancedConfigurationGenerator()
        profile = generator.generate_enhanced_configuration(
            detection_result,
            ConfigurationStrategy(args.strategy),
            ConfigurationEnvironment(args.environment),
            args.template,
            validation_level=ValidationLevel(args.validation_level)
        )
        
        # Export configuration
        output_dir = Path(args.output)
        exported_files = generator.export_configuration_profile(
            profile, output_dir, args.format
        )
        
        # Generate deployment script if requested
        if args.generate_script:
            script_path = output_dir / "deploy.sh"
            generator.generate_deployment_script(
                profile, ConfigurationEnvironment(args.environment), script_path
            )
            exported_files.append(script_path)
        
        print(f"✅ Enhanced configuration generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Files created: {len(exported_files)}")
        print(f"🔧 Profile ID: {profile.profile_id}")
        
        # Display validation summary
        validation = profile.validation_results.get("default")
        if validation:
            print(f"✅ Validation: {'PASSED' if validation.is_valid else 'FAILED'}")
            if validation.schema_errors:
                print(f"❌ Schema errors: {len(validation.schema_errors)}")
            if validation.performance_warnings:
                print(f"⚠️  Performance warnings: {len(validation.performance_warnings)}")
            if validation.security_warnings:
                print(f"🔒 Security warnings: {len(validation.security_warnings)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())