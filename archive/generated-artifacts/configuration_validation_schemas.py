#!/usr/bin/env python3
"""
Configuration Validation Schemas
================================

Comprehensive JSON schemas and validation rules for Project Index configurations.
Provides schema definitions, custom validators, and compliance checking for
different environments and use cases.

Features:
- JSON Schema definitions for all configuration components
- Environment-specific validation rules
- Framework-specific validation patterns
- Performance constraint validation
- Security compliance checking
- Custom validation rule engine

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
import json
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import re


class SchemaLevel(Enum):
    """Schema validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class ConfigurationSchemas:
    """
    Comprehensive schema definitions for Project Index configurations.
    """
    
    # Base schema for core configuration structure
    BASE_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Project Index Configuration",
        "description": "Core configuration schema for Project Index",
        "required": [
            "project_name",
            "project_path",
            "configuration_version",
            "detection_metadata",
            "analysis",
            "performance"
        ],
        "properties": {
            "project_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 255,
                "pattern": "^[a-zA-Z0-9_-]+$",
                "description": "Project name identifier"
            },
            "project_path": {
                "type": "string",
                "minLength": 1,
                "description": "Absolute path to project directory"
            },
            "configuration_version": {
                "type": "string",
                "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
                "description": "Configuration version in semver format"
            },
            "generated_timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "ISO timestamp of configuration generation"
            },
            "detection_metadata": {
                "$ref": "#/definitions/detectionMetadata"
            },
            "analysis": {
                "$ref": "#/definitions/analysisConfig"
            },
            "file_patterns": {
                "$ref": "#/definitions/filePatterns"
            },
            "ignore_patterns": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1
                },
                "description": "File patterns to ignore during analysis"
            },
            "monitoring": {
                "$ref": "#/definitions/monitoringConfig"
            },
            "optimization": {
                "$ref": "#/definitions/optimizationConfig"
            },
            "performance": {
                "$ref": "#/definitions/performanceConfig"
            },
            "security": {
                "$ref": "#/definitions/securityConfig"
            },
            "integrations": {
                "$ref": "#/definitions/integrationsConfig"
            },
            "custom_rules": {
                "type": "array",
                "items": {
                    "$ref": "#/definitions/customRule"
                },
                "description": "Custom analysis rules"
            },
            "configuration_notes": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Configuration generation notes"
            },
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Configuration recommendations"
            }
        },
        "definitions": {
            "detectionMetadata": {
                "type": "object",
                "required": ["optimization_level", "security_level"],
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "enum": ["python", "javascript", "typescript", "go", "rust", "java", "csharp", "php", None]
                    },
                    "frameworks": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "project_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large", "enterprise"]
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "optimization_level": {
                        "type": "string",
                        "enum": ["minimal", "balanced", "performance", "enterprise"]
                    },
                    "security_level": {
                        "type": "string",
                        "enum": ["basic", "standard", "strict", "enterprise"]
                    }
                }
            },
            "analysisConfig": {
                "type": "object",
                "required": ["enabled"],
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable/disable analysis"
                    },
                    "parse_ast": {
                        "type": "boolean",
                        "description": "Enable AST parsing"
                    },
                    "extract_dependencies": {
                        "type": "boolean",
                        "description": "Extract dependency information"
                    },
                    "calculate_complexity": {
                        "type": "boolean",
                        "description": "Calculate code complexity metrics"
                    },
                    "analyze_docstrings": {
                        "type": "boolean",
                        "description": "Analyze documentation strings"
                    },
                    "max_file_size_mb": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum file size to analyze (MB)"
                    },
                    "max_line_count": {
                        "type": "number",
                        "minimum": 100,
                        "maximum": 1000000,
                        "description": "Maximum line count to analyze"
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 5,
                        "maximum": 600,
                        "description": "Analysis timeout in seconds"
                    },
                    "parallel_processing": {
                        "type": "boolean",
                        "description": "Enable parallel processing"
                    }
                }
            },
            "filePatterns": {
                "type": "object",
                "required": ["include"],
                "properties": {
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[*]{0,2}.*$"
                        },
                        "minItems": 1,
                        "description": "File patterns to include"
                    },
                    "exclude": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[*]{0,2}.*$"
                        },
                        "description": "File patterns to exclude"
                    }
                }
            },
            "monitoringConfig": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable file monitoring"
                    },
                    "debounce_seconds": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 60.0,
                        "description": "File change debounce interval"
                    },
                    "watch_subdirectories": {
                        "type": "boolean",
                        "description": "Monitor subdirectories"
                    },
                    "max_file_size_mb": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum file size to monitor"
                    },
                    "batch_events": {
                        "type": "boolean",
                        "description": "Batch file change events"
                    },
                    "event_queue_size": {
                        "type": "number",
                        "minimum": 10,
                        "maximum": 10000,
                        "description": "Event queue size"
                    }
                }
            },
            "optimizationConfig": {
                "type": "object",
                "properties": {
                    "context_optimization_enabled": {
                        "type": "boolean",
                        "description": "Enable context optimization"
                    },
                    "max_context_files": {
                        "type": "number",
                        "minimum": 10,
                        "maximum": 1000,
                        "description": "Maximum files in context"
                    },
                    "relevance_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "File relevance threshold"
                    },
                    "intelligent_caching": {
                        "type": "boolean",
                        "description": "Enable intelligent caching"
                    },
                    "incremental_updates": {
                        "type": "boolean",
                        "description": "Enable incremental updates"
                    },
                    "aggressive_optimization": {
                        "type": "boolean",
                        "description": "Enable aggressive optimization"
                    },
                    "predictive_loading": {
                        "type": "boolean",
                        "description": "Enable predictive loading"
                    },
                    "smart_prioritization": {
                        "type": "boolean",
                        "description": "Enable smart prioritization"
                    }
                }
            },
            "performanceConfig": {
                "type": "object",
                "required": ["max_concurrent_analyses"],
                "properties": {
                    "max_concurrent_analyses": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 32,
                        "description": "Maximum concurrent analysis processes"
                    },
                    "analysis_batch_size": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Analysis batch size"
                    },
                    "cache_enabled": {
                        "type": "boolean",
                        "description": "Enable result caching"
                    },
                    "cache_ttl_seconds": {
                        "type": "number",
                        "minimum": 60,
                        "maximum": 86400,
                        "description": "Cache TTL in seconds"
                    },
                    "memory_limit_mb": {
                        "type": "number",
                        "minimum": 128,
                        "maximum": 16384,
                        "description": "Memory limit in MB"
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 5,
                        "maximum": 600,
                        "description": "Operation timeout in seconds"
                    },
                    "debounce_interval": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 60.0,
                        "description": "Debounce interval in seconds"
                    },
                    "batch_insert_size": {
                        "type": "number",
                        "minimum": 10,
                        "maximum": 1000,
                        "description": "Database batch insert size"
                    }
                }
            },
            "securityConfig": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable security features"
                    },
                    "scan_dependencies": {
                        "type": "boolean",
                        "description": "Scan dependencies for vulnerabilities"
                    },
                    "check_vulnerabilities": {
                        "type": "boolean",
                        "description": "Check for known vulnerabilities"
                    },
                    "validate_licenses": {
                        "type": "boolean",
                        "description": "Validate dependency licenses"
                    },
                    "audit_sensitive_files": {
                        "type": "boolean",
                        "description": "Audit sensitive files"
                    },
                    "enable_sandboxing": {
                        "type": "boolean",
                        "description": "Enable execution sandboxing"
                    },
                    "restrict_file_access": {
                        "type": "boolean",
                        "description": "Restrict file system access"
                    },
                    "security_reporting": {
                        "type": "boolean",
                        "description": "Enable security reporting"
                    },
                    "sensitive_file_patterns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Patterns for sensitive files"
                    }
                }
            },
            "integrationsConfig": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable integrations"
                    },
                    "webhooks": {
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean"
                            },
                            "endpoints": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    },
                    "ci_cd": {
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean"
                            },
                            "platforms": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["github_actions", "gitlab_ci", "jenkins", "travis", "circleci"]
                                }
                            }
                        }
                    },
                    "ide": {
                        "type": "object",
                        "properties": {
                            "vscode_extension": {
                                "type": "boolean"
                            },
                            "intellij_plugin": {
                                "type": "boolean"
                            }
                        }
                    },
                    "monitoring": {
                        "type": "object",
                        "properties": {
                            "prometheus": {
                                "type": "boolean"
                            },
                            "grafana": {
                                "type": "boolean"
                            },
                            "datadog": {
                                "type": "boolean"
                            }
                        }
                    }
                }
            },
            "customRule": {
                "type": "object",
                "required": ["name", "pattern", "action"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Rule name"
                    },
                    "pattern": {
                        "type": "string",
                        "minLength": 1,
                        "description": "File pattern to match"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["include", "exclude", "high_priority", "low_priority", "security_scan", "test_context"],
                        "description": "Action to take when pattern matches"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the rule"
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether rule is enabled"
                    }
                }
            }
        }
    }
    
    # Environment-specific schemas
    ENVIRONMENT_SCHEMAS = {
        "development": {
            "type": "object",
            "properties": {
                "analysis": {
                    "properties": {
                        "timeout_seconds": {
                            "maximum": 30  # Shorter timeouts for dev
                        }
                    }
                },
                "performance": {
                    "properties": {
                        "max_concurrent_analyses": {
                            "maximum": 4  # Limited concurrency for dev
                        },
                        "memory_limit_mb": {
                            "maximum": 1024  # Lower memory for dev
                        }
                    }
                }
            }
        },
        "production": {
            "type": "object",
            "properties": {
                "security": {
                    "required": ["enabled", "scan_dependencies", "check_vulnerabilities"],
                    "properties": {
                        "enabled": {
                            "const": True  # Security must be enabled in production
                        },
                        "scan_dependencies": {
                            "const": True  # Dependency scanning required
                        },
                        "check_vulnerabilities": {
                            "const": True  # Vulnerability checking required
                        }
                    }
                },
                "monitoring": {
                    "required": ["enabled"],
                    "properties": {
                        "enabled": {
                            "const": True  # Monitoring required in production
                        }
                    }
                }
            }
        },
        "enterprise": {
            "type": "object",
            "properties": {
                "security": {
                    "required": ["enabled", "scan_dependencies", "check_vulnerabilities", "validate_licenses", "security_reporting"],
                    "properties": {
                        "enabled": {
                            "const": True
                        },
                        "scan_dependencies": {
                            "const": True
                        },
                        "check_vulnerabilities": {
                            "const": True
                        },
                        "validate_licenses": {
                            "const": True  # License validation required
                        },
                        "security_reporting": {
                            "const": True  # Security reporting required
                        }
                    }
                },
                "performance": {
                    "properties": {
                        "max_concurrent_analyses": {
                            "minimum": 4  # Higher concurrency for enterprise
                        },
                        "memory_limit_mb": {
                            "minimum": 1024  # Higher memory for enterprise
                        }
                    }
                }
            }
        }
    }
    
    # Framework-specific validation rules
    FRAMEWORK_SCHEMAS = {
        "python": {
            "analysis": {
                "required_properties": ["parse_ast", "extract_dependencies"],
                "recommended_properties": ["analyze_docstrings"]
            },
            "file_patterns": {
                "required_includes": ["**/*.py"],
                "recommended_includes": ["**/requirements*.txt", "**/pyproject.toml", "**/setup.py"]
            }
        },
        "javascript": {
            "analysis": {
                "required_properties": ["parse_ast"],
                "recommended_properties": ["extract_dependencies"]
            },
            "file_patterns": {
                "required_includes": ["**/*.js"],
                "recommended_includes": ["**/package.json", "**/*.jsx", "**/*.ts", "**/*.tsx"]
            }
        },
        "django": {
            "security": {
                "required_properties": ["scan_dependencies", "audit_sensitive_files"],
                "sensitive_patterns": ["**/settings.py", "**/urls.py", "**/.env"]
            }
        },
        "react": {
            "file_patterns": {
                "required_includes": ["**/*.jsx", "**/*.tsx"],
                "ignore_patterns": ["**/build/**", "**/dist/**"]
            }
        }
    }


class ConfigurationValidator:
    """
    Advanced configuration validator with multiple validation levels
    and framework-specific rules.
    """
    
    def __init__(self, schema_level: SchemaLevel = SchemaLevel.STANDARD):
        """Initialize the validator with specified schema level."""
        self.schema_level = schema_level
        self.schemas = ConfigurationSchemas()
        self.custom_validators: List[Callable] = []
        
        # Create JSON schema validator
        self.base_validator = Draft7Validator(self.schemas.BASE_SCHEMA)
    
    def validate_configuration(
        self,
        config: Dict[str, Any],
        environment: Optional[str] = None,
        frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            environment: Target environment (development, production, enterprise)
            frameworks: List of detected frameworks
            
        Returns:
            Validation result with errors, warnings, and suggestions
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "schema_level": self.schema_level.value,
            "validation_timestamp": json.dumps(None, default=str)  # Will be set to current time
        }
        
        # 1. Base schema validation
        schema_errors = self._validate_base_schema(config)
        if schema_errors:
            result["valid"] = False
            result["errors"].extend(schema_errors)
        
        # 2. Environment-specific validation
        if environment and environment in self.schemas.ENVIRONMENT_SCHEMAS:
            env_errors, env_warnings = self._validate_environment_schema(config, environment)
            if env_errors:
                result["valid"] = False
                result["errors"].extend(env_errors)
            result["warnings"].extend(env_warnings)
        
        # 3. Framework-specific validation
        if frameworks:
            framework_errors, framework_warnings, framework_suggestions = self._validate_framework_requirements(
                config, frameworks
            )
            if framework_errors:
                result["valid"] = False
                result["errors"].extend(framework_errors)
            result["warnings"].extend(framework_warnings)
            result["suggestions"].extend(framework_suggestions)
        
        # 4. Performance validation
        perf_warnings, perf_suggestions = self._validate_performance_constraints(config)
        result["warnings"].extend(perf_warnings)
        result["suggestions"].extend(perf_suggestions)
        
        # 5. Security validation
        sec_errors, sec_warnings = self._validate_security_requirements(config, environment)
        if sec_errors:
            result["valid"] = False
            result["errors"].extend(sec_errors)
        result["warnings"].extend(sec_warnings)
        
        # 6. Custom validator execution
        for validator in self.custom_validators:
            try:
                custom_result = validator(config)
                if custom_result.get("errors"):
                    result["valid"] = False
                    result["errors"].extend(custom_result["errors"])
                if custom_result.get("warnings"):
                    result["warnings"].extend(custom_result["warnings"])
                if custom_result.get("suggestions"):
                    result["suggestions"].extend(custom_result["suggestions"])
            except Exception as e:
                result["warnings"].append(f"Custom validator failed: {str(e)}")
        
        # 7. Schema level specific validation
        if self.schema_level in [SchemaLevel.STRICT, SchemaLevel.ENTERPRISE]:
            strict_errors, strict_warnings = self._validate_strict_requirements(config)
            if strict_errors:
                result["valid"] = False
                result["errors"].extend(strict_errors)
            result["warnings"].extend(strict_warnings)
        
        return result
    
    def _validate_base_schema(self, config: Dict[str, Any]) -> List[str]:
        """Validate against base JSON schema."""
        errors = []
        
        try:
            self.base_validator.validate(config)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            
            # Add more specific error details
            if e.path:
                path = " -> ".join(str(p) for p in e.path)
                errors.append(f"Error path: {path}")
        
        return errors
    
    def _validate_environment_schema(
        self,
        config: Dict[str, Any],
        environment: str
    ) -> Tuple[List[str], List[str]]:
        """Validate environment-specific requirements."""
        errors = []
        warnings = []
        
        env_schema = self.schemas.ENVIRONMENT_SCHEMAS.get(environment, {})
        
        # Validate environment-specific constraints
        try:
            # Create a temporary validator for environment schema
            env_validator = Draft7Validator(env_schema)
            env_validator.validate(config)
        except ValidationError as e:
            errors.append(f"Environment validation error for {environment}: {e.message}")
        
        # Environment-specific checks
        if environment == "production":
            # Production must have security enabled
            if not config.get("security", {}).get("enabled", False):
                errors.append("Security must be enabled in production environment")
            
            # Production should have monitoring
            if not config.get("monitoring", {}).get("enabled", False):
                warnings.append("Monitoring should be enabled in production environment")
        
        elif environment == "development":
            # Development can have relaxed timeouts
            timeout = config.get("analysis", {}).get("timeout_seconds", 30)
            if timeout > 30:
                warnings.append("Consider shorter timeouts for development environment")
        
        return errors, warnings
    
    def _validate_framework_requirements(
        self,
        config: Dict[str, Any],
        frameworks: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Validate framework-specific requirements."""
        errors = []
        warnings = []
        suggestions = []
        
        for framework in frameworks:
            if framework in self.schemas.FRAMEWORK_SCHEMAS:
                framework_rules = self.schemas.FRAMEWORK_SCHEMAS[framework]
                
                # Check analysis requirements
                if "analysis" in framework_rules:
                    analysis_rules = framework_rules["analysis"]
                    analysis_config = config.get("analysis", {})
                    
                    # Required properties
                    for prop in analysis_rules.get("required_properties", []):
                        if not analysis_config.get(prop, False):
                            errors.append(f"Framework {framework} requires analysis.{prop} to be enabled")
                    
                    # Recommended properties
                    for prop in analysis_rules.get("recommended_properties", []):
                        if not analysis_config.get(prop, False):
                            suggestions.append(f"Consider enabling analysis.{prop} for {framework} projects")
                
                # Check file pattern requirements
                if "file_patterns" in framework_rules:
                    pattern_rules = framework_rules["file_patterns"]
                    file_patterns = config.get("file_patterns", {}).get("include", [])
                    
                    # Required includes
                    for pattern in pattern_rules.get("required_includes", []):
                        if pattern not in file_patterns:
                            warnings.append(f"Framework {framework} should include pattern: {pattern}")
                    
                    # Recommended includes
                    for pattern in pattern_rules.get("recommended_includes", []):
                        if pattern not in file_patterns:
                            suggestions.append(f"Consider including pattern for {framework}: {pattern}")
                
                # Check security requirements
                if "security" in framework_rules:
                    security_rules = framework_rules["security"]
                    security_config = config.get("security", {})
                    
                    # Required security properties
                    for prop in security_rules.get("required_properties", []):
                        if not security_config.get(prop, False):
                            errors.append(f"Framework {framework} requires security.{prop} to be enabled")
                    
                    # Sensitive file patterns
                    sensitive_patterns = security_rules.get("sensitive_patterns", [])
                    current_patterns = security_config.get("sensitive_file_patterns", [])
                    
                    for pattern in sensitive_patterns:
                        if pattern not in current_patterns:
                            suggestions.append(f"Consider adding sensitive file pattern for {framework}: {pattern}")
        
        return errors, warnings, suggestions
    
    def _validate_performance_constraints(
        self,
        config: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Validate performance configuration constraints."""
        warnings = []
        suggestions = []
        
        perf_config = config.get("performance", {})
        
        # Check resource usage
        max_memory = perf_config.get("memory_limit_mb", 512)
        max_concurrency = perf_config.get("max_concurrent_analyses", 4)
        
        # Memory warnings
        if max_memory > 4096:
            warnings.append(f"High memory limit ({max_memory}MB) may cause system instability")
        elif max_memory < 256:
            warnings.append(f"Low memory limit ({max_memory}MB) may cause performance issues")
        
        # Concurrency warnings
        if max_concurrency > 16:
            warnings.append(f"High concurrency ({max_concurrency}) may overwhelm system resources")
        elif max_concurrency == 1:
            suggestions.append("Consider increasing concurrency for better performance")
        
        # Cache configuration
        if not perf_config.get("cache_enabled", False):
            suggestions.append("Enable caching for improved performance")
        
        # Timeout validation
        analysis_timeout = config.get("analysis", {}).get("timeout_seconds", 30)
        perf_timeout = perf_config.get("timeout_seconds", 30)
        
        if analysis_timeout > perf_timeout:
            warnings.append("Analysis timeout exceeds performance timeout")
        
        return warnings, suggestions
    
    def _validate_security_requirements(
        self,
        config: Dict[str, Any],
        environment: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Validate security configuration requirements."""
        errors = []
        warnings = []
        
        security_config = config.get("security", {})
        
        # Environment-specific security requirements
        if environment in ["production", "enterprise"]:
            required_security_features = [
                "scan_dependencies",
                "check_vulnerabilities"
            ]
            
            for feature in required_security_features:
                if not security_config.get(feature, False):
                    errors.append(f"Security feature '{feature}' is required for {environment} environment")
        
        # Security feature consistency
        if security_config.get("enabled", False):
            if not any([
                security_config.get("scan_dependencies", False),
                security_config.get("check_vulnerabilities", False),
                security_config.get("audit_sensitive_files", False)
            ]):
                warnings.append("Security is enabled but no security features are active")
        
        # Sensitive file patterns
        sensitive_patterns = security_config.get("sensitive_file_patterns", [])
        common_sensitive_patterns = [
            "**/.env", "**/config.py", "**/settings.py", "**/*secret*", "**/*password*"
        ]
        
        for pattern in common_sensitive_patterns:
            if pattern not in sensitive_patterns:
                warnings.append(f"Consider adding sensitive file pattern: {pattern}")
        
        return errors, warnings
    
    def _validate_strict_requirements(
        self,
        config: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Validate strict/enterprise requirements."""
        errors = []
        warnings = []
        
        if self.schema_level == SchemaLevel.ENTERPRISE:
            # Enterprise requirements
            required_features = {
                "security.enabled": True,
                "security.scan_dependencies": True,
                "security.check_vulnerabilities": True,
                "security.validate_licenses": True,
                "monitoring.enabled": True,
                "performance.cache_enabled": True
            }
            
            for feature_path, required_value in required_features.items():
                keys = feature_path.split(".")
                value = config
                
                try:
                    for key in keys:
                        value = value[key]
                    
                    if value != required_value:
                        errors.append(f"Enterprise requirement: {feature_path} must be {required_value}")
                except (KeyError, TypeError):
                    errors.append(f"Enterprise requirement: {feature_path} is missing")
        
        elif self.schema_level == SchemaLevel.STRICT:
            # Strict mode warnings
            if not config.get("analysis", {}).get("parse_ast", False):
                warnings.append("AST parsing should be enabled in strict mode")
            
            if not config.get("performance", {}).get("cache_enabled", False):
                warnings.append("Caching should be enabled in strict mode")
        
        return errors, warnings
    
    def add_custom_validator(self, validator: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Add a custom validation function."""
        self.custom_validators.append(validator)
    
    def validate_file_patterns(self, patterns: List[str]) -> List[str]:
        """Validate file pattern syntax."""
        errors = []
        
        for pattern in patterns:
            # Check for valid glob pattern syntax
            if not pattern:
                errors.append("Empty file pattern is not allowed")
                continue
            
            # Check for dangerous patterns
            if pattern == "**" or pattern == "*":
                errors.append(f"Overly broad pattern '{pattern}' may cause performance issues")
            
            # Check for valid characters
            invalid_chars = ['<', '>', '|', '"']
            for char in invalid_chars:
                if char in pattern:
                    errors.append(f"Invalid character '{char}' in pattern: {pattern}")
        
        return errors
    
    def suggest_optimizations(self, config: Dict[str, Any]) -> List[str]:
        """Suggest configuration optimizations."""
        suggestions = []
        
        perf_config = config.get("performance", {})
        analysis_config = config.get("analysis", {})
        
        # Performance optimizations
        if not perf_config.get("cache_enabled", False):
            suggestions.append("Enable caching to improve analysis performance")
        
        if perf_config.get("max_concurrent_analyses", 1) == 1:
            suggestions.append("Increase concurrency for better performance on multi-core systems")
        
        # Analysis optimizations
        if not analysis_config.get("parallel_processing", False):
            suggestions.append("Enable parallel processing for faster analysis")
        
        # Security optimizations
        security_config = config.get("security", {})
        if security_config.get("enabled", False) and not security_config.get("security_reporting", False):
            suggestions.append("Enable security reporting for better visibility")
        
        return suggestions


# Example custom validators
def validate_python_project(config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom validator for Python projects."""
    errors = []
    warnings = []
    suggestions = []
    
    analysis_config = config.get("analysis", {})
    file_patterns = config.get("file_patterns", {}).get("include", [])
    
    # Python-specific requirements
    if "**/*.py" not in file_patterns:
        errors.append("Python projects must include '**/*.py' in file patterns")
    
    if not analysis_config.get("parse_ast", False):
        warnings.append("AST parsing is highly recommended for Python projects")
    
    if not analysis_config.get("extract_dependencies", False):
        suggestions.append("Consider enabling dependency extraction for Python projects")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions
    }


def validate_web_application(config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom validator for web applications."""
    errors = []
    warnings = []
    suggestions = []
    
    security_config = config.get("security", {})
    
    # Web application security requirements
    if not security_config.get("scan_dependencies", False):
        warnings.append("Web applications should have dependency scanning enabled")
    
    if not security_config.get("audit_sensitive_files", False):
        suggestions.append("Consider enabling sensitive file auditing for web applications")
    
    # Check for web-specific patterns
    file_patterns = config.get("file_patterns", {}).get("include", [])
    web_patterns = ["**/*.html", "**/*.css", "**/*.js"]
    
    missing_patterns = [p for p in web_patterns if p not in file_patterns]
    if missing_patterns:
        suggestions.extend([f"Consider including pattern: {p}" for p in missing_patterns])
    
    return {
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions
    }


if __name__ == "__main__":
    """Example usage of the validation system."""
    
    # Example configuration to validate
    example_config = {
        "project_name": "test_project",
        "project_path": "/path/to/project",
        "configuration_version": "2.0",
        "detection_metadata": {
            "primary_language": "python",
            "frameworks": ["django"],
            "project_size": "medium",
            "optimization_level": "performance",
            "security_level": "strict"
        },
        "analysis": {
            "enabled": True,
            "parse_ast": True,
            "extract_dependencies": True,
            "calculate_complexity": True,
            "max_file_size_mb": 10,
            "timeout_seconds": 30
        },
        "file_patterns": {
            "include": ["**/*.py", "**/*.html"]
        },
        "performance": {
            "max_concurrent_analyses": 4,
            "cache_enabled": True,
            "memory_limit_mb": 1024
        },
        "security": {
            "enabled": True,
            "scan_dependencies": True,
            "check_vulnerabilities": True
        }
    }
    
    # Validate the configuration
    validator = ConfigurationValidator(SchemaLevel.STRICT)
    validator.add_custom_validator(validate_python_project)
    
    result = validator.validate_configuration(
        example_config,
        environment="production",
        frameworks=["django"]
    )
    
    print("Validation Result:")
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    if result['suggestions']:
        print("Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")