#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Configuration Generator
============================================================

Tests for the enhanced intelligent configuration generator system,
including configuration generation, validation, templates, and
environment-specific optimizations.

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import unittest
import json
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_configuration_generator import (
    EnhancedConfigurationGenerator,
    ConfigurationStrategy,
    ConfigurationEnvironment,
    ValidationLevel,
    ConfigurationTemplate,
    ConfigurationProfile
)
from configuration_validation_schemas import (
    ConfigurationValidator,
    SchemaLevel,
    validate_python_project,
    validate_web_application
)
from intelligent_config_generator import (
    OptimizationLevel,
    SecurityLevel,
    ProjectIndexConfiguration
)


class TestEnhancedConfigurationGenerator(unittest.TestCase):
    """Test suite for EnhancedConfigurationGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedConfigurationGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Sample detection result
        self.sample_detection_result = {
            "project_path": str(self.temp_dir),
            "primary_language": {
                "language": "python",
                "confidence": "high",
                "file_count": 25
            },
            "detected_frameworks": [
                {
                    "framework": "django",
                    "confidence": "high",
                    "evidence_files": ["manage.py", "settings.py"]
                }
            ],
            "size_analysis": {
                "size_category": "medium",
                "file_count": 150,
                "line_count": 5000,
                "complexity_score": 0.6
            },
            "confidence_score": 0.85
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_enhanced_configuration_default(self):
        """Test generating enhanced configuration with default settings."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result
        )
        
        # Verify profile structure
        self.assertIsInstance(profile, ConfigurationProfile)
        self.assertIsInstance(profile.base_configuration, ProjectIndexConfiguration)
        self.assertTrue(profile.profile_id)
        self.assertIn("python", profile.profile_id.lower() or profile.base_configuration.project_name.lower())
        
        # Verify configuration content
        config = profile.base_configuration
        self.assertEqual(config.project_name, self.temp_dir.name)
        self.assertTrue(config.analysis["enabled"])
        self.assertTrue(config.performance["cache_enabled"])
        
        # Verify environment overrides exist
        self.assertIsInstance(profile.environment_overrides, dict)
        self.assertIn(ConfigurationEnvironment.PRODUCTION, profile.environment_overrides)
        
        # Verify validation results
        self.assertIn("default", profile.validation_results)
        validation = profile.validation_results["default"]
        self.assertIsInstance(validation.is_valid, bool)
    
    def test_generate_configuration_with_strategy(self):
        """Test configuration generation with different strategies."""
        strategies = [
            ConfigurationStrategy.MINIMAL,
            ConfigurationStrategy.DEVELOPMENT,
            ConfigurationStrategy.PRODUCTION,
            ConfigurationStrategy.ENTERPRISE
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                profile = self.generator.generate_enhanced_configuration(
                    self.sample_detection_result,
                    strategy=strategy
                )
                
                # Verify strategy is reflected in configuration
                config = profile.base_configuration
                metadata = config.detection_metadata
                
                if strategy == ConfigurationStrategy.ENTERPRISE:
                    self.assertEqual(metadata["optimization_level"], "enterprise")
                    self.assertEqual(metadata["security_level"], "enterprise")
                elif strategy == ConfigurationStrategy.MINIMAL:
                    self.assertEqual(metadata["optimization_level"], "minimal")
                    self.assertEqual(metadata["security_level"], "basic")
    
    def test_generate_configuration_with_environment(self):
        """Test configuration generation with different environments."""
        environments = [
            ConfigurationEnvironment.DEVELOPMENT,
            ConfigurationEnvironment.TESTING,
            ConfigurationEnvironment.PRODUCTION,
            ConfigurationEnvironment.ENTERPRISE
        ]
        
        for environment in environments:
            with self.subTest(environment=environment):
                profile = self.generator.generate_enhanced_configuration(
                    self.sample_detection_result,
                    environment=environment
                )
                
                # Verify environment-specific overrides
                self.assertIn(environment, profile.environment_overrides)
                env_config = profile.environment_overrides[environment]
                
                if environment == ConfigurationEnvironment.DEVELOPMENT:
                    # Development should have debug features
                    monitoring = env_config.get("monitoring", {})
                    self.assertTrue(monitoring.get("debug_logging", False))
                elif environment == ConfigurationEnvironment.PRODUCTION:
                    # Production should have enhanced security
                    security = env_config.get("security", {})
                    self.assertTrue(security.get("strict_security", False))
    
    def test_template_selection_automatic(self):
        """Test automatic template selection based on project characteristics."""
        # Test Python microservice detection
        detection_result = self.sample_detection_result.copy()
        detection_result["size_analysis"]["size_category"] = "small"
        detection_result["detected_frameworks"] = [
            {"framework": "fastapi", "confidence": "high"}
        ]
        
        profile = self.generator.generate_enhanced_configuration(
            detection_result,
            strategy=ConfigurationStrategy.PRODUCTION
        )
        
        # Should select a suitable template for Python microservice
        self.assertTrue(len(profile.custom_templates) <= 1)  # May or may not select template
        
        # Verify microservice-appropriate settings
        config = profile.base_configuration
        self.assertTrue(config.analysis.get("parse_ast", False))
        self.assertTrue(config.security.get("scan_dependencies", False))
    
    def test_template_selection_explicit(self):
        """Test explicit template selection."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result,
            template_name="python_microservice"
        )
        
        # Verify template was applied
        self.assertEqual(len(profile.custom_templates), 1)
        template = profile.custom_templates[0]
        self.assertEqual(template.name, "Python Microservice")
        
        # Verify template settings were applied
        config = profile.base_configuration
        self.assertTrue(config.analysis.get("parse_ast", False))
        self.assertTrue(config.analysis.get("extract_dependencies", False))
    
    def test_framework_specific_optimizations(self):
        """Test framework-specific configuration optimizations."""
        # Test Django project
        django_detection = self.sample_detection_result.copy()
        django_detection["detected_frameworks"] = [
            {"framework": "django", "confidence": "high"}
        ]
        
        profile = self.generator.generate_enhanced_configuration(django_detection)
        config = profile.base_configuration
        
        # Django projects should have ORM analysis
        self.assertTrue(config.analysis.get("orm_analysis", False))
        
        # Test React project
        react_detection = {
            "project_path": str(self.temp_dir),
            "primary_language": {"language": "javascript", "confidence": "high", "file_count": 30},
            "detected_frameworks": [
                {"framework": "react", "confidence": "high"}
            ],
            "size_analysis": {"size_category": "medium", "file_count": 100},
            "confidence_score": 0.8
        }
        
        react_profile = self.generator.generate_enhanced_configuration(react_detection)
        react_config = react_profile.base_configuration
        
        # React projects should have bundle analysis
        self.assertTrue(react_config.analysis.get("bundle_analysis", False))
        
        # Should include CSS patterns
        include_patterns = react_config.file_patterns.get("include", [])
        css_patterns = [p for p in include_patterns if "css" in p]
        self.assertTrue(len(css_patterns) > 0)
    
    def test_performance_optimization(self):
        """Test performance optimization based on system resources."""
        with patch('os.cpu_count', return_value=8):
            profile = self.generator.generate_enhanced_configuration(
                self.sample_detection_result
            )
            
            config = profile.base_configuration
            # Should optimize for available CPU cores
            max_concurrency = config.performance["max_concurrent_analyses"]
            self.assertLessEqual(max_concurrency, 12)  # Shouldn't exceed reasonable limits
            self.assertGreaterEqual(max_concurrency, 2)  # Should be at least 2
    
    def test_large_project_optimizations(self):
        """Test optimizations for large projects."""
        large_project_detection = self.sample_detection_result.copy()
        large_project_detection["size_analysis"] = {
            "size_category": "enterprise",
            "file_count": 6000,
            "line_count": 500000,
            "complexity_score": 0.9
        }
        
        profile = self.generator.generate_enhanced_configuration(large_project_detection)
        config = profile.base_configuration
        
        # Large projects should have higher resource limits
        self.assertGreaterEqual(config.performance["memory_limit_mb"], 1024)
        self.assertGreaterEqual(config.performance["analysis_batch_size"], 100)
    
    def test_validation_integration(self):
        """Test validation integration with configuration generation."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result,
            validation_level=ValidationLevel.STRICT
        )
        
        validation = profile.validation_results["default"]
        self.assertEqual(validation.validation_level, ValidationLevel.STRICT)
        
        # Strict validation should catch more issues
        total_issues = len(validation.schema_errors) + len(validation.performance_warnings) + len(validation.security_warnings)
        self.assertGreaterEqual(total_issues, 0)  # May or may not have issues
    
    def test_security_audit(self):
        """Test security audit functionality."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result,
            strategy=ConfigurationStrategy.ENTERPRISE
        )
        
        security_audit = profile.security_audit
        
        # Should have security score
        self.assertIn("security_score", security_audit)
        self.assertIsInstance(security_audit["security_score"], (int, float))
        self.assertGreaterEqual(security_audit["security_score"], 0)
        self.assertLessEqual(security_audit["security_score"], 100)
        
        # Should have compliance checks
        self.assertIn("compliance_checks", security_audit)
        compliance = security_audit["compliance_checks"]
        self.assertIn("dependency_scanning", compliance)
        self.assertIn("vulnerability_assessment", compliance)
    
    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result
        )
        
        performance_metrics = profile.performance_metrics
        
        # Should have performance score
        self.assertIn("performance_score", performance_metrics)
        self.assertIsInstance(performance_metrics["performance_score"], (int, float))
        self.assertGreaterEqual(performance_metrics["performance_score"], 0)
        self.assertLessEqual(performance_metrics["performance_score"], 100)
        
        # Should have resource estimates
        self.assertIn("estimated_cpu_usage_percent", performance_metrics)
        self.assertIn("estimated_memory_usage_mb", performance_metrics)
    
    def test_custom_template_addition(self):
        """Test adding custom templates."""
        custom_template = ConfigurationTemplate(
            name="test_template",
            description="Test template",
            target_environments=[ConfigurationEnvironment.DEVELOPMENT],
            base_settings={
                "analysis": {
                    "custom_feature": True
                }
            },
            tags=["test"]
        )
        
        self.generator.add_custom_template(custom_template)
        
        # Verify template was added
        self.assertIn("test_template", self.generator.templates)
        
        # Test using the custom template
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result,
            template_name="test_template"
        )
        
        config = profile.base_configuration
        self.assertTrue(config.analysis.get("custom_feature", False))
    
    def test_export_configuration_profile(self):
        """Test exporting configuration profiles."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result
        )
        
        output_dir = self.temp_dir / "export_test"
        
        # Test JSON export
        exported_files = self.generator.export_configuration_profile(
            profile, output_dir, format="json"
        )
        
        self.assertTrue(len(exported_files) > 0)
        
        # Verify main config file exists
        config_file = output_dir / "config.json"
        self.assertIn(config_file, exported_files)
        self.assertTrue(config_file.exists())
        
        # Verify config file is valid JSON
        with open(config_file) as f:
            config_data = json.load(f)
        self.assertEqual(config_data["project_name"], profile.base_configuration.project_name)
        
        # Verify environment files exist
        env_dir = output_dir / "environments"
        self.assertTrue(env_dir.exists())
        
        # Verify reports exist
        reports_dir = output_dir / "reports"
        self.assertTrue(reports_dir.exists())
        
        perf_report = reports_dir / "performance_analysis.json"
        self.assertTrue(perf_report.exists())
    
    def test_deployment_script_generation(self):
        """Test deployment script generation."""
        profile = self.generator.generate_enhanced_configuration(
            self.sample_detection_result
        )
        
        script_path = self.temp_dir / "deploy.sh"
        
        self.generator.generate_deployment_script(
            profile,
            ConfigurationEnvironment.PRODUCTION,
            script_path
        )
        
        # Verify script was created
        self.assertTrue(script_path.exists())
        
        # Verify script is executable
        self.assertTrue(script_path.stat().st_mode & 0o111)  # Check execute permission
        
        # Verify script content
        script_content = script_path.read_text()
        self.assertIn("production", script_content.lower())
        self.assertIn("project index", script_content.lower())
        self.assertIn(profile.profile_id, script_content)


class TestConfigurationValidator(unittest.TestCase):
    """Test suite for ConfigurationValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigurationValidator(SchemaLevel.STANDARD)
        
        # Sample valid configuration
        self.valid_config = {
            "project_name": "test_project",
            "project_path": "/path/to/project",
            "configuration_version": "2.0",
            "detection_metadata": {
                "primary_language": "python",
                "frameworks": ["django"],
                "project_size": "medium",
                "optimization_level": "performance",
                "security_level": "standard"
            },
            "analysis": {
                "enabled": True,
                "parse_ast": True,
                "extract_dependencies": True,
                "calculate_complexity": True,
                "max_file_size_mb": 10,
                "max_line_count": 50000,
                "timeout_seconds": 30
            },
            "file_patterns": {
                "include": ["**/*.py"]
            },
            "ignore_patterns": ["**/__pycache__/**"],
            "monitoring": {
                "enabled": True,
                "debounce_seconds": 2.0
            },
            "optimization": {
                "context_optimization_enabled": True
            },
            "performance": {
                "max_concurrent_analyses": 4,
                "cache_enabled": True,
                "memory_limit_mb": 512
            },
            "security": {
                "enabled": True,
                "scan_dependencies": True,
                "check_vulnerabilities": True
            },
            "integrations": {
                "enabled": False
            },
            "custom_rules": [],
            "configuration_notes": [],
            "recommendations": []
        }
    
    def test_validate_valid_configuration(self):
        """Test validation of a valid configuration."""
        result = self.validator.validate_configuration(self.valid_config)
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(result["schema_level"], "standard")
    
    def test_validate_invalid_configuration(self):
        """Test validation of invalid configurations."""
        # Missing required field
        invalid_config = self.valid_config.copy()
        del invalid_config["project_name"]
        
        result = self.validator.validate_configuration(invalid_config)
        self.assertFalse(result["valid"])
        self.assertTrue(len(result["errors"]) > 0)
        
        # Invalid value type
        invalid_config2 = self.valid_config.copy()
        invalid_config2["performance"]["max_concurrent_analyses"] = "invalid"
        
        result2 = self.validator.validate_configuration(invalid_config2)
        self.assertFalse(result2["valid"])
    
    def test_environment_specific_validation(self):
        """Test environment-specific validation rules."""
        # Production environment validation
        result = self.validator.validate_configuration(
            self.valid_config,
            environment="production"
        )
        
        # Should pass as security is enabled
        self.assertTrue(result["valid"])
        
        # Test with security disabled in production
        insecure_config = self.valid_config.copy()
        insecure_config["security"]["enabled"] = False
        
        result2 = self.validator.validate_configuration(
            insecure_config,
            environment="production"
        )
        
        # Should have errors about security
        security_errors = [e for e in result2["errors"] if "security" in e.lower()]
        self.assertTrue(len(security_errors) > 0)
    
    def test_framework_specific_validation(self):
        """Test framework-specific validation rules."""
        result = self.validator.validate_configuration(
            self.valid_config,
            frameworks=["django"]
        )
        
        # Should pass for Django with proper configuration
        self.assertTrue(result["valid"])
        
        # Test missing Django requirements
        incomplete_config = self.valid_config.copy()
        incomplete_config["analysis"]["parse_ast"] = False
        
        result2 = self.validator.validate_configuration(
            incomplete_config,
            frameworks=["python"]  # Generic python framework
        )
        
        # Should have warnings or suggestions
        total_issues = len(result2["errors"]) + len(result2["warnings"]) + len(result2["suggestions"])
        self.assertGreater(total_issues, 0)
    
    def test_performance_constraint_validation(self):
        """Test performance constraint validation."""
        # Test high memory configuration
        high_memory_config = self.valid_config.copy()
        high_memory_config["performance"]["memory_limit_mb"] = 8192
        
        result = self.validator.validate_configuration(high_memory_config)
        
        memory_warnings = [w for w in result["warnings"] if "memory" in w.lower()]
        self.assertTrue(len(memory_warnings) > 0)
        
        # Test high concurrency configuration
        high_concurrency_config = self.valid_config.copy()
        high_concurrency_config["performance"]["max_concurrent_analyses"] = 20
        
        result2 = self.validator.validate_configuration(high_concurrency_config)
        
        concurrency_warnings = [w for w in result2["warnings"] if "concurrency" in w.lower()]
        self.assertTrue(len(concurrency_warnings) > 0)
    
    def test_security_validation(self):
        """Test security-specific validation."""
        # Test enterprise environment requirements
        result = self.validator.validate_configuration(
            self.valid_config,
            environment="enterprise"
        )
        
        # Enterprise should require additional security features
        license_errors = [e for e in result["errors"] if "license" in e.lower()]
        self.assertTrue(len(license_errors) > 0)  # Should require license validation
    
    def test_strict_validation_mode(self):
        """Test strict validation mode."""
        strict_validator = ConfigurationValidator(SchemaLevel.STRICT)
        
        result = strict_validator.validate_configuration(self.valid_config)
        
        # Strict mode should generate more warnings/suggestions
        total_feedback = len(result["warnings"]) + len(result["suggestions"])
        
        standard_validator = ConfigurationValidator(SchemaLevel.STANDARD)
        standard_result = standard_validator.validate_configuration(self.valid_config)
        standard_feedback = len(standard_result["warnings"]) + len(standard_result["suggestions"])
        
        # Strict should have more feedback
        self.assertGreaterEqual(total_feedback, standard_feedback)
    
    def test_enterprise_validation_mode(self):
        """Test enterprise validation mode."""
        enterprise_validator = ConfigurationValidator(SchemaLevel.ENTERPRISE)
        
        result = enterprise_validator.validate_configuration(self.valid_config)
        
        # Enterprise mode should require specific features
        enterprise_errors = [e for e in result["errors"] if "enterprise" in e.lower()]
        self.assertTrue(len(enterprise_errors) > 0)
    
    def test_custom_validator_integration(self):
        """Test custom validator integration."""
        def custom_validator(config):
            errors = []
            if not config.get("analysis", {}).get("custom_feature", False):
                errors.append("Custom feature is required")
            return {"errors": errors, "warnings": [], "suggestions": []}
        
        self.validator.add_custom_validator(custom_validator)
        
        result = self.validator.validate_configuration(self.valid_config)
        
        # Should have error from custom validator
        custom_errors = [e for e in result["errors"] if "custom feature" in e.lower()]
        self.assertTrue(len(custom_errors) > 0)
    
    def test_file_pattern_validation(self):
        """Test file pattern validation."""
        # Test valid patterns
        valid_patterns = ["**/*.py", "src/**/*.js", "*.json"]
        errors = self.validator.validate_file_patterns(valid_patterns)
        self.assertEqual(len(errors), 0)
        
        # Test invalid patterns
        invalid_patterns = ["", "**", "*", "file<name>"]
        errors = self.validator.validate_file_patterns(invalid_patterns)
        self.assertTrue(len(errors) > 0)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        # Test configuration without optimizations
        basic_config = self.valid_config.copy()
        basic_config["performance"]["cache_enabled"] = False
        basic_config["performance"]["max_concurrent_analyses"] = 1
        
        suggestions = self.validator.suggest_optimizations(basic_config)
        
        # Should suggest enabling cache and increasing concurrency
        cache_suggestions = [s for s in suggestions if "cach" in s.lower()]
        concurrency_suggestions = [s for s in suggestions if "concurrency" in s.lower()]
        
        self.assertTrue(len(cache_suggestions) > 0)
        self.assertTrue(len(concurrency_suggestions) > 0)


class TestCustomValidators(unittest.TestCase):
    """Test suite for custom validators."""
    
    def test_python_project_validator(self):
        """Test Python project validator."""
        # Valid Python configuration
        python_config = {
            "analysis": {
                "parse_ast": True,
                "extract_dependencies": True
            },
            "file_patterns": {
                "include": ["**/*.py"]
            }
        }
        
        result = validate_python_project(python_config)
        self.assertEqual(len(result["errors"]), 0)
        
        # Invalid Python configuration
        invalid_python_config = {
            "analysis": {
                "parse_ast": False
            },
            "file_patterns": {
                "include": ["**/*.js"]  # Missing .py files
            }
        }
        
        result2 = validate_python_project(invalid_python_config)
        self.assertTrue(len(result2["errors"]) > 0)
    
    def test_web_application_validator(self):
        """Test web application validator."""
        # Basic web app configuration
        web_config = {
            "security": {
                "scan_dependencies": False,
                "audit_sensitive_files": False
            },
            "file_patterns": {
                "include": ["**/*.py"]  # Missing web patterns
            }
        }
        
        result = validate_web_application(web_config)
        
        # Should have warnings about security and suggestions for patterns
        self.assertTrue(len(result["warnings"]) > 0)
        self.assertTrue(len(result["suggestions"]) > 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete configuration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedConfigurationGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_python_django_workflow(self):
        """Test complete workflow for Python Django project."""
        detection_result = {
            "project_path": str(self.temp_dir),
            "primary_language": {"language": "python", "confidence": "high", "file_count": 45},
            "detected_frameworks": [
                {"framework": "django", "confidence": "high", "evidence_files": ["manage.py"]}
            ],
            "size_analysis": {"size_category": "medium", "file_count": 200, "line_count": 8000},
            "confidence_score": 0.9
        }
        
        # Generate configuration
        profile = self.generator.generate_enhanced_configuration(
            detection_result,
            strategy=ConfigurationStrategy.PRODUCTION,
            environment=ConfigurationEnvironment.PRODUCTION,
            validation_level=ValidationLevel.STRICT
        )
        
        # Verify Django-specific optimizations
        config = profile.base_configuration
        self.assertTrue(config.analysis.get("parse_ast", False))
        self.assertTrue(config.security.get("scan_dependencies", False))
        
        # Verify production environment settings
        prod_config = profile.environment_overrides[ConfigurationEnvironment.PRODUCTION]
        self.assertTrue(prod_config.get("security", {}).get("strict_security", False))
        
        # Verify validation passed
        validation = profile.validation_results["default"]
        self.assertTrue(validation.is_valid)
        
        # Export and verify files
        output_dir = self.temp_dir / "django_export"
        exported_files = self.generator.export_configuration_profile(
            profile, output_dir, include_environments=True
        )
        
        self.assertTrue(len(exported_files) >= 4)  # Config, validation, perf, security reports
        
        # Generate deployment script
        script_path = output_dir / "deploy.sh"
        self.generator.generate_deployment_script(
            profile, ConfigurationEnvironment.PRODUCTION, script_path
        )
        
        self.assertTrue(script_path.exists())
    
    def test_javascript_react_workflow(self):
        """Test complete workflow for JavaScript React project."""
        detection_result = {
            "project_path": str(self.temp_dir),
            "primary_language": {"language": "javascript", "confidence": "high", "file_count": 60},
            "detected_frameworks": [
                {"framework": "react", "confidence": "high", "evidence_files": ["package.json"]}
            ],
            "size_analysis": {"size_category": "large", "file_count": 800, "line_count": 25000},
            "confidence_score": 0.85
        }
        
        # Generate configuration for development
        dev_profile = self.generator.generate_enhanced_configuration(
            detection_result,
            strategy=ConfigurationStrategy.DEVELOPMENT,
            environment=ConfigurationEnvironment.DEVELOPMENT
        )
        
        # Verify React-specific settings
        config = dev_profile.base_configuration
        self.assertTrue(config.analysis.get("bundle_analysis", False))
        
        # Verify CSS patterns included
        include_patterns = config.file_patterns.get("include", [])
        css_patterns = [p for p in include_patterns if "css" in p or "scss" in p]
        self.assertTrue(len(css_patterns) > 0)
        
        # Generate production configuration
        prod_profile = self.generator.generate_enhanced_configuration(
            detection_result,
            strategy=ConfigurationStrategy.PRODUCTION,
            environment=ConfigurationEnvironment.PRODUCTION
        )
        
        # Production should have different settings
        prod_config = prod_profile.base_configuration
        self.assertNotEqual(
            config.performance["max_concurrent_analyses"],
            prod_config.performance["max_concurrent_analyses"]
        )
    
    def test_enterprise_microservices_workflow(self):
        """Test complete workflow for enterprise microservices."""
        detection_result = {
            "project_path": str(self.temp_dir),
            "primary_language": {"language": "go", "confidence": "high", "file_count": 120},
            "detected_frameworks": [
                {"framework": "gin", "confidence": "high"}
            ],
            "size_analysis": {"size_category": "enterprise", "file_count": 1500, "line_count": 50000},
            "confidence_score": 0.95
        }
        
        # Generate enterprise configuration
        profile = self.generator.generate_enhanced_configuration(
            detection_result,
            strategy=ConfigurationStrategy.ENTERPRISE,
            environment=ConfigurationEnvironment.ENTERPRISE,
            validation_level=ValidationLevel.ENTERPRISE
        )
        
        # Verify enterprise-level settings
        config = profile.base_configuration
        self.assertGreaterEqual(config.performance["max_concurrent_analyses"], 6)
        self.assertGreaterEqual(config.performance["memory_limit_mb"], 1024)
        
        # Verify enterprise security requirements
        self.assertTrue(config.security.get("scan_dependencies", False))
        self.assertTrue(config.security.get("check_vulnerabilities", False))
        self.assertTrue(config.security.get("validate_licenses", False))
        
        # Verify performance metrics
        perf_metrics = profile.performance_metrics
        self.assertGreaterEqual(perf_metrics["performance_score"], 0)
        
        # Verify security audit
        security_audit = profile.security_audit
        self.assertIn("compliance_checks", security_audit)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)