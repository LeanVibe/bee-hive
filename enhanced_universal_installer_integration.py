#!/usr/bin/env python3
"""
Enhanced Universal Installer Integration
========================================

Advanced integration between the enhanced configuration generator and universal installer
for seamless, intelligent project setup with optimal configurations out-of-the-box.

Features:
- Automatic project detection and analysis
- Smart configuration generation with templates
- Environment-specific configuration deployment
- Comprehensive validation and testing
- Interactive configuration customization
- Real-time performance monitoring
- Security compliance validation
- CI/CD integration setup

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import yaml

# Import enhanced components
from enhanced_configuration_generator import (
    EnhancedConfigurationGenerator,
    ConfigurationStrategy,
    ConfigurationEnvironment,
    ValidationLevel,
    ConfigurationProfile
)
from configuration_validation_schemas import (
    ConfigurationValidator,
    SchemaLevel
)
from universal_installer_integration import UniversalInstallerIntegration
from app.project_index.intelligent_detector import IntelligentProjectDetector

logger = logging.getLogger(__name__)


class EnhancedInstallerWorkflow:
    """
    Enhanced installer workflow with intelligent configuration generation,
    validation, and deployment capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced installer workflow."""
        self.config = config or {}
        
        # Initialize components
        self.detector = IntelligentProjectDetector()
        self.config_generator = EnhancedConfigurationGenerator()
        self.validator = ConfigurationValidator(SchemaLevel.STANDARD)
        self.base_installer = UniversalInstallerIntegration(config)
        
        # Workflow settings
        self.interactive_mode = self.config.get('interactive', True)
        self.auto_validate = self.config.get('auto_validate', True)
        self.auto_optimize = self.config.get('auto_optimize', True)
        self.backup_existing = self.config.get('backup_existing', True)
        
        logger.info("Enhanced installer workflow initialized")
    
    def install_with_enhanced_detection(
        self,
        project_path: str,
        strategy: ConfigurationStrategy = ConfigurationStrategy.PRODUCTION,
        environment: ConfigurationEnvironment = ConfigurationEnvironment.PRODUCTION,
        template_name: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced installation with intelligent configuration generation.
        
        Args:
            project_path: Path to the project directory
            strategy: Configuration generation strategy
            environment: Target deployment environment
            template_name: Specific template to use
            validation_level: Configuration validation level
            options: Additional installation options
            
        Returns:
            Installation result with enhanced metadata
        """
        project_path = Path(project_path).resolve()
        options = options or {}
        
        print(f"🚀 Enhanced Project Index Installation")
        print(f"📂 Project: {project_path.name}")
        print(f"🎯 Strategy: {strategy.value}")
        print(f"🌍 Environment: {environment.value}")
        print(f"🔍 Validation: {validation_level.value}")
        print("=" * 60)
        
        # Step 1: Pre-installation validation
        if not self._validate_installation_prerequisites(project_path):
            raise ValueError(f"Installation prerequisites not met for: {project_path}")
        
        # Step 2: Backup existing configuration if present
        backup_info = None
        if self.backup_existing:
            backup_info = self._backup_existing_configuration(project_path)
        
        # Step 3: Enhanced project detection
        print(f"\n🔍 Running enhanced project detection...")
        detection_result = self._run_enhanced_detection(project_path)
        self._display_enhanced_detection_summary(detection_result)
        
        # Step 4: Interactive configuration customization
        custom_overrides = {}
        if self.interactive_mode:
            custom_overrides = self._interactive_configuration_setup(
                detection_result, strategy, environment, template_name
            )
        
        # Step 5: Generate enhanced configuration
        print(f"\n⚙️ Generating enhanced configuration...")
        config_profile = self._generate_enhanced_configuration(
            detection_result, strategy, environment, template_name,
            validation_level, custom_overrides
        )
        
        # Step 6: Validate configuration
        print(f"\n✅ Validating configuration...")
        validation_result = self._validate_configuration_comprehensive(
            config_profile, detection_result
        )
        
        if not validation_result["valid"] and validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
            print(f"❌ Configuration validation failed with {validation_level.value} validation")
            self._display_validation_errors(validation_result)
            
            if self.interactive_mode:
                if not self._confirm_proceed_with_errors():
                    print("Installation cancelled by user")
                    return {"success": False, "reason": "User cancelled due to validation errors"}
        
        # Step 7: Deploy configuration
        print(f"\n📦 Deploying configuration...")
        deployment_result = self._deploy_enhanced_configuration(
            project_path, config_profile, environment, options
        )
        
        # Step 8: Post-installation setup
        print(f"\n🔧 Running post-installation setup...")
        post_install_result = self._enhanced_post_installation_setup(
            project_path, config_profile, detection_result, environment
        )
        
        # Step 9: Generate comprehensive documentation
        print(f"\n📚 Generating documentation...")
        docs_result = self._generate_comprehensive_documentation(
            project_path, config_profile, detection_result, validation_result
        )
        
        # Step 10: Setup monitoring and maintenance
        print(f"\n👁️ Setting up monitoring...")
        monitoring_result = self._setup_enhanced_monitoring(
            project_path, config_profile, environment
        )
        
        # Compile final result
        result = {
            "success": deployment_result["success"],
            "project_path": str(project_path),
            "installation_timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "strategy": strategy.value,
                "environment": environment.value,
                "validation_level": validation_level.value,
                "template_used": template_name,
                "profile_id": config_profile.profile_id
            },
            "detection_result": self._serialize_detection_result(detection_result),
            "validation_result": validation_result,
            "deployment_result": deployment_result,
            "backup_info": backup_info,
            "documentation": docs_result,
            "monitoring": monitoring_result,
            "next_steps": self._generate_enhanced_next_steps(
                project_path, config_profile, detection_result, environment
            ),
            "performance_metrics": config_profile.performance_metrics,
            "security_audit": config_profile.security_audit
        }
        
        # Display final summary
        self._display_installation_summary(result)
        
        return result
    
    def _validate_installation_prerequisites(self, project_path: Path) -> bool:
        """Validate installation prerequisites."""
        print(f"🔍 Validating installation prerequisites...")
        
        # Check project directory
        if not project_path.exists():
            print(f"❌ Project directory does not exist: {project_path}")
            return False
        
        if not project_path.is_dir():
            print(f"❌ Path is not a directory: {project_path}")
            return False
        
        # Check permissions
        if not os.access(project_path, os.W_OK):
            print(f"❌ No write permission for directory: {project_path}")
            return False
        
        # Check disk space (at least 100MB)
        try:
            stat = shutil.disk_usage(project_path)
            free_mb = stat.free / (1024 * 1024)
            if free_mb < 100:
                print(f"⚠️ Low disk space: {free_mb:.1f}MB available")
        except Exception:
            pass
        
        # Check for existing Project Index installation
        existing_pi = project_path / '.project-index'
        if existing_pi.exists():
            print(f"⚠️ Existing Project Index installation found")
            if self.interactive_mode:
                response = input("Continue with upgrade/reinstall? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    return False
        
        print(f"✅ Prerequisites validated")
        return True
    
    def _backup_existing_configuration(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Backup existing Project Index configuration."""
        existing_pi = project_path / '.project-index'
        
        if not existing_pi.exists():
            return None
        
        print(f"💾 Backing up existing configuration...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = project_path / f'.project-index.backup.{timestamp}'
        
        try:
            shutil.copytree(existing_pi, backup_dir)
            print(f"✅ Backup created: {backup_dir.name}")
            
            return {
                "backup_path": str(backup_dir),
                "backup_timestamp": timestamp,
                "original_size": sum(f.stat().st_size for f in existing_pi.rglob('*') if f.is_file())
            }
        except Exception as e:
            print(f"⚠️ Backup failed: {e}")
            return None
    
    def _run_enhanced_detection(self, project_path: Path) -> Dict[str, Any]:
        """Run enhanced project detection with additional analysis."""
        detection_result = self.detector.detect_project(project_path)
        
        # Convert to dictionary format for easier handling
        enhanced_result = {
            "project_path": str(project_path),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "primary_language": {
                "language": detection_result.primary_language.language if detection_result.primary_language else None,
                "confidence": detection_result.primary_language.confidence.value if detection_result.primary_language else None,
                "file_count": detection_result.primary_language.file_count if detection_result.primary_language else 0
            },
            "detected_frameworks": [
                {
                    "framework": f.framework,
                    "confidence": f.confidence.value,
                    "evidence_files": f.evidence_files[:3]  # Limit for display
                }
                for f in detection_result.detected_frameworks
            ],
            "size_analysis": {
                "size_category": detection_result.size_analysis.size_category.value,
                "file_count": detection_result.size_analysis.file_count,
                "line_count": detection_result.size_analysis.line_count,
                "complexity_score": detection_result.size_analysis.complexity_score
            },
            "confidence_score": detection_result.confidence_score,
            "recommendations": detection_result.recommendations
        }
        
        # Add enhanced analysis
        enhanced_result["enhanced_analysis"] = self._perform_enhanced_analysis(project_path)
        
        return enhanced_result
    
    def _perform_enhanced_analysis(self, project_path: Path) -> Dict[str, Any]:
        """Perform additional enhanced analysis."""
        analysis = {
            "git_analysis": self._analyze_git_repository(project_path),
            "ci_cd_analysis": self._analyze_ci_cd_setup(project_path),
            "documentation_analysis": self._analyze_documentation(project_path),
            "testing_analysis": self._analyze_testing_setup(project_path),
            "performance_indicators": self._analyze_performance_indicators(project_path)
        }
        
        return analysis
    
    def _analyze_git_repository(self, project_path: Path) -> Dict[str, Any]:
        """Analyze Git repository characteristics."""
        git_dir = project_path / '.git'
        
        if not git_dir.exists():
            return {"is_git_repo": False}
        
        analysis = {"is_git_repo": True}
        
        try:
            # Get branch information
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  cwd=project_path, capture_output=True, text=True)
            if result.returncode == 0:
                analysis["current_branch"] = result.stdout.strip()
            
            # Get commit count
            result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                  cwd=project_path, capture_output=True, text=True)
            if result.returncode == 0:
                analysis["commit_count"] = int(result.stdout.strip())
            
            # Check for common files
            gitignore = project_path / '.gitignore'
            analysis["has_gitignore"] = gitignore.exists()
            
            if gitignore.exists():
                gitignore_content = gitignore.read_text()
                analysis["gitignore_includes_project_index"] = '.project-index' in gitignore_content
            
        except Exception as e:
            logger.debug(f"Git analysis error: {e}")
        
        return analysis
    
    def _analyze_ci_cd_setup(self, project_path: Path) -> Dict[str, Any]:
        """Analyze CI/CD setup and configuration."""
        ci_indicators = {
            "github_actions": (project_path / '.github' / 'workflows').exists(),
            "gitlab_ci": (project_path / '.gitlab-ci.yml').exists(),
            "travis": (project_path / '.travis.yml').exists(),
            "jenkins": (project_path / 'Jenkinsfile').exists(),
            "circle_ci": (project_path / '.circleci').exists(),
            "azure_pipelines": (project_path / 'azure-pipelines.yml').exists()
        }
        
        active_platforms = [platform for platform, active in ci_indicators.items() if active]
        
        return {
            "platforms": ci_indicators,
            "active_platforms": active_platforms,
            "has_ci_cd": len(active_platforms) > 0,
            "multiple_platforms": len(active_platforms) > 1
        }
    
    def _analyze_documentation(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project documentation."""
        doc_files = {
            "readme": any((project_path / name).exists() for name in 
                         ['README.md', 'README.rst', 'README.txt', 'readme.md']),
            "changelog": any((project_path / name).exists() for name in 
                           ['CHANGELOG.md', 'HISTORY.md', 'CHANGES.md']),
            "contributing": any((project_path / name).exists() for name in 
                              ['CONTRIBUTING.md', 'CONTRIBUTE.md']),
            "license": any((project_path / name).exists() for name in 
                         ['LICENSE', 'LICENSE.md', 'LICENSE.txt']),
            "docs_directory": (project_path / 'docs').exists() or (project_path / 'documentation').exists()
        }
        
        documentation_score = sum(doc_files.values()) / len(doc_files)
        
        return {
            "files": doc_files,
            "documentation_score": documentation_score,
            "well_documented": documentation_score >= 0.6
        }
    
    def _analyze_testing_setup(self, project_path: Path) -> Dict[str, Any]:
        """Analyze testing setup and coverage."""
        test_indicators = {
            "test_directory": any((project_path / name).exists() for name in ['tests', 'test']),
            "pytest_config": any((project_path / name).exists() for name in 
                                ['pytest.ini', 'pyproject.toml', 'setup.cfg']),
            "jest_config": any((project_path / name).exists() for name in 
                             ['jest.config.js', 'jest.config.json']),
            "coverage_config": any((project_path / name).exists() for name in 
                                 ['.coveragerc', 'coverage.ini']),
            "tox_config": (project_path / 'tox.ini').exists()
        }
        
        # Count test files
        test_file_count = 0
        test_patterns = ['**/test_*.py', '**/test*.py', '**/*_test.py', 
                        '**/*.test.js', '**/*.spec.js', '**/*.test.ts']
        
        for pattern in test_patterns:
            test_file_count += len(list(project_path.glob(pattern)))
        
        return {
            "indicators": test_indicators,
            "test_file_count": test_file_count,
            "has_testing_setup": any(test_indicators.values()) or test_file_count > 0,
            "testing_maturity": "high" if test_file_count > 10 else "medium" if test_file_count > 0 else "low"
        }
    
    def _analyze_performance_indicators(self, project_path: Path) -> Dict[str, Any]:
        """Analyze performance-related indicators."""
        # Calculate project complexity metrics
        total_files = len(list(project_path.rglob('*')))
        code_files = len(list(project_path.glob('**/*.py'))) + len(list(project_path.glob('**/*.js'))) + len(list(project_path.glob('**/*.ts')))
        
        # Check for performance-related files
        perf_indicators = {
            "docker": (project_path / 'Dockerfile').exists(),
            "docker_compose": any((project_path / name).exists() for name in 
                                ['docker-compose.yml', 'docker-compose.yaml']),
            "makefile": any((project_path / name).exists() for name in 
                          ['Makefile', 'makefile']),
            "requirements_files": len(list(project_path.glob('*requirements*.txt'))) > 0,
            "package_lock": any((project_path / name).exists() for name in 
                              ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'])
        }
        
        # Estimate project complexity
        complexity_factors = {
            "file_count": min(total_files / 1000, 1.0),
            "code_file_ratio": code_files / max(total_files, 1),
            "directory_depth": min(len(list(project_path.rglob('*/'))) / 50, 1.0)
        }
        
        complexity_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        return {
            "performance_indicators": perf_indicators,
            "complexity_factors": complexity_factors,
            "complexity_score": complexity_score,
            "estimated_build_time": "fast" if complexity_score < 0.3 else "medium" if complexity_score < 0.7 else "slow"
        }
    
    def _display_enhanced_detection_summary(self, detection_result: Dict[str, Any]):
        """Display enhanced detection results summary."""
        print(f"\n📊 ENHANCED DETECTION RESULTS")
        print(f"━" * 60)
        
        # Basic detection info
        primary_lang = detection_result.get("primary_language", {})
        if primary_lang.get("language"):
            print(f"🎯 Primary Language: {primary_lang['language'].title()}")
            print(f"   Confidence: {primary_lang.get('confidence', 'unknown').replace('_', ' ').title()}")
            print(f"   Files: {primary_lang.get('file_count', 0):,}")
        
        # Frameworks
        frameworks = detection_result.get("detected_frameworks", [])
        if frameworks:
            print(f"🛠️ Frameworks ({len(frameworks)}):")
            for fw in frameworks[:3]:
                print(f"   • {fw['framework'].title()}: {fw['confidence'].replace('_', ' ').title()}")
        
        # Project size and complexity
        size_info = detection_result.get("size_analysis", {})
        print(f"📏 Project Size: {size_info.get('size_category', 'unknown').title()}")
        print(f"📁 Files: {size_info.get('file_count', 0):,}")
        print(f"📝 Lines: {size_info.get('line_count', 0):,}")
        
        # Enhanced analysis
        enhanced = detection_result.get("enhanced_analysis", {})
        
        # Git repository info
        git_info = enhanced.get("git_analysis", {})
        if git_info.get("is_git_repo"):
            print(f"🗂️ Git Repository:")
            print(f"   Branch: {git_info.get('current_branch', 'unknown')}")
            print(f"   Commits: {git_info.get('commit_count', 0):,}")
            print(f"   .gitignore: {'✅' if git_info.get('has_gitignore') else '❌'}")
        
        # CI/CD setup
        ci_info = enhanced.get("ci_cd_analysis", {})
        if ci_info.get("has_ci_cd"):
            platforms = ci_info.get("active_platforms", [])
            print(f"🔄 CI/CD: {', '.join(p.replace('_', ' ').title() for p in platforms)}")
        
        # Documentation
        doc_info = enhanced.get("documentation_analysis", {})
        doc_score = doc_info.get("documentation_score", 0)
        print(f"📚 Documentation: {doc_score:.1%} ({'Good' if doc_score >= 0.6 else 'Basic' if doc_score >= 0.3 else 'Limited'})")
        
        # Testing
        test_info = enhanced.get("testing_analysis", {})
        test_maturity = test_info.get("testing_maturity", "low")
        test_count = test_info.get("test_file_count", 0)
        print(f"🧪 Testing: {test_maturity.title()} ({test_count} test files)")
        
        # Overall confidence
        confidence = detection_result.get("confidence_score", 0)
        print(f"📈 Detection Confidence: {confidence:.1%}")
    
    def _interactive_configuration_setup(
        self,
        detection_result: Dict[str, Any],
        strategy: ConfigurationStrategy,
        environment: ConfigurationEnvironment,
        template_name: Optional[str]
    ) -> Dict[str, Any]:
        """Interactive configuration customization."""
        print(f"\n🎛️ INTERACTIVE CONFIGURATION SETUP")
        print(f"━" * 60)
        
        overrides = {}
        
        # Ask about strategy
        print(f"Current strategy: {strategy.value}")
        if input("Customize strategy? [y/N]: ").strip().lower() in ['y', 'yes']:
            strategies = list(ConfigurationStrategy)
            print("Available strategies:")
            for i, s in enumerate(strategies, 1):
                print(f"  {i}. {s.value}")
            
            try:
                choice = int(input("Select strategy (number): ")) - 1
                if 0 <= choice < len(strategies):
                    strategy = strategies[choice]
            except ValueError:
                pass
        
        # Ask about performance settings
        if input("Customize performance settings? [y/N]: ").strip().lower() in ['y', 'yes']:
            performance_overrides = {}
            
            try:
                concurrency = input(f"Max concurrent analyses [default: auto]: ").strip()
                if concurrency and concurrency.isdigit():
                    performance_overrides["max_concurrent_analyses"] = int(concurrency)
                
                memory = input(f"Memory limit in MB [default: auto]: ").strip()
                if memory and memory.isdigit():
                    performance_overrides["memory_limit_mb"] = int(memory)
                
                if performance_overrides:
                    overrides["performance"] = performance_overrides
            except ValueError:
                pass
        
        # Ask about security settings
        if input("Customize security settings? [y/N]: ").strip().lower() in ['y', 'yes']:
            security_overrides = {}
            
            if input("Enable dependency scanning? [Y/n]: ").strip().lower() not in ['n', 'no']:
                security_overrides["scan_dependencies"] = True
            
            if input("Enable vulnerability checking? [Y/n]: ").strip().lower() not in ['n', 'no']:
                security_overrides["check_vulnerabilities"] = True
            
            if input("Enable license validation? [y/N]: ").strip().lower() in ['y', 'yes']:
                security_overrides["validate_licenses"] = True
            
            if security_overrides:
                overrides["security"] = security_overrides
        
        return overrides
    
    def _generate_enhanced_configuration(
        self,
        detection_result: Dict[str, Any],
        strategy: ConfigurationStrategy,
        environment: ConfigurationEnvironment,
        template_name: Optional[str],
        validation_level: ValidationLevel,
        custom_overrides: Dict[str, Any]
    ) -> ConfigurationProfile:
        """Generate enhanced configuration profile."""
        return self.config_generator.generate_enhanced_configuration(
            detection_result,
            strategy=strategy,
            environment=environment,
            template_name=template_name,
            custom_overrides=custom_overrides,
            validation_level=validation_level
        )
    
    def _validate_configuration_comprehensive(
        self,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive configuration validation."""
        config_dict = self._convert_config_to_dict(config_profile.base_configuration)
        
        frameworks = [f["framework"] for f in detection_result.get("detected_frameworks", [])]
        environment = None  # Extract from profile if needed
        
        return self.validator.validate_configuration(
            config_dict,
            environment=environment,
            frameworks=frameworks
        )
    
    def _display_validation_errors(self, validation_result: Dict[str, Any]):
        """Display validation errors and warnings."""
        print(f"\n❌ VALIDATION ISSUES")
        print(f"━" * 60)
        
        if validation_result.get("errors"):
            print("🚫 Errors:")
            for error in validation_result["errors"]:
                print(f"   • {error}")
        
        if validation_result.get("warnings"):
            print("⚠️ Warnings:")
            for warning in validation_result["warnings"]:
                print(f"   • {warning}")
        
        if validation_result.get("suggestions"):
            print("💡 Suggestions:")
            for suggestion in validation_result["suggestions"]:
                print(f"   • {suggestion}")
    
    def _confirm_proceed_with_errors(self) -> bool:
        """Ask user to confirm proceeding with validation errors."""
        print("\nValidation errors were found.")
        response = input("Proceed with installation anyway? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    
    def _deploy_enhanced_configuration(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        environment: ConfigurationEnvironment,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy enhanced configuration to project."""
        try:
            # Create Project Index directory structure
            pi_dir = project_path / '.project-index'
            pi_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            subdirs = ['cache', 'logs', 'temp', 'config', 'reports', 'scripts']
            for subdir in subdirs:
                (pi_dir / subdir).mkdir(exist_ok=True)
            
            # Export configuration profile
            exported_files = self.config_generator.export_configuration_profile(
                config_profile,
                pi_dir / 'config',
                format='json',
                include_environments=True
            )
            
            # Create main configuration symlink
            main_config = pi_dir / 'config.json'
            if not main_config.exists():
                env_config_file = pi_dir / 'config' / f'{environment.value}.json'
                if env_config_file.exists():
                    if os.name == 'nt':  # Windows
                        shutil.copy2(env_config_file, main_config)
                    else:  # Unix-like
                        main_config.symlink_to(f'config/{environment.value}.json')
                else:
                    # Use base config
                    base_config_file = pi_dir / 'config' / 'config.json'
                    if base_config_file.exists():
                        if os.name == 'nt':
                            shutil.copy2(base_config_file, main_config)
                        else:
                            main_config.symlink_to('config/config.json')
            
            # Generate deployment script
            deploy_script = pi_dir / 'scripts' / 'deploy.sh'
            self.config_generator.generate_deployment_script(
                config_profile, environment, deploy_script
            )
            
            # Create CLI wrapper script
            self._create_enhanced_cli_script(pi_dir, config_profile)
            
            # Set up gitignore
            self._update_gitignore(project_path)
            
            return {
                "success": True,
                "exported_files": [str(f) for f in exported_files],
                "main_config": str(main_config),
                "deployment_script": str(deploy_script)
            }
            
        except Exception as e:
            logger.error(f"Configuration deployment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_enhanced_cli_script(self, pi_dir: Path, config_profile: ConfigurationProfile):
        """Create enhanced CLI script."""
        cli_script = pi_dir / 'cli.py'
        
        cli_content = f'''#!/usr/bin/env python3
"""
Enhanced Project Index CLI
Generated for profile: {config_profile.profile_id}
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Enhanced Project Index CLI")
    parser.add_argument("--status", action="store_true", help="Show detailed status")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--performance", action="store_true", help="Show performance metrics")
    parser.add_argument("--security", action="store_true", help="Show security audit")
    parser.add_argument("--config", help="Show configuration")
    parser.add_argument("--environment", help="Switch environment")
    
    args = parser.parse_args()
    
    pi_dir = Path(__file__).parent
    
    if args.status:
        print("📊 Enhanced Project Index Status")
        print("=" * 40)
        
        # Load main config
        config_file = pi_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(f"Project: {{config.get('project_name', 'Unknown')}}")
            print(f"Profile ID: {config_profile.profile_id}")
            print(f"Configuration Version: {{config.get('configuration_version', 'Unknown')}}")
            
            # Show performance metrics
            perf_file = pi_dir / "reports" / "performance_analysis.json"
            if perf_file.exists():
                with open(perf_file) as f:
                    perf = json.load(f)
                print(f"Performance Score: {{perf.get('performance_score', 0)}}/100")
            
            # Show security audit
            security_file = pi_dir / "reports" / "security_audit.json"
            if security_file.exists():
                with open(security_file) as f:
                    security = json.load(f)
                print(f"Security Score: {{security.get('security_score', 0)}}/100")
        else:
            print("❌ No configuration found")
    
    elif args.performance:
        perf_file = pi_dir / "reports" / "performance_analysis.json"
        if perf_file.exists():
            with open(perf_file) as f:
                perf = json.load(f)
            print(json.dumps(perf, indent=2))
        else:
            print("❌ No performance data found")
    
    elif args.security:
        security_file = pi_dir / "reports" / "security_audit.json"
        if security_file.exists():
            with open(security_file) as f:
                security = json.load(f)
            print(json.dumps(security, indent=2))
        else:
            print("❌ No security audit found")
    
    elif args.validate:
        validation_file = pi_dir / "validation_report.json"
        if validation_file.exists():
            with open(validation_file) as f:
                validation = json.load(f)
            print("✅ Configuration Validation Results")
            for env, result in validation.items():
                status = "✅ VALID" if result.get("valid") else "❌ INVALID"
                print(f"{{env}}: {{status}}")
                if result.get("errors"):
                    for error in result["errors"]:
                        print(f"  Error: {{error}}")
        else:
            print("❌ No validation data found")
    
    elif args.config:
        config_file = pi_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(json.dumps(config, indent=2))
        else:
            print("❌ No configuration found")
    
    elif args.analyze:
        print("🔍 Running project analysis...")
        print("Analysis complete. (Enhanced analysis implementation pending)")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''
        
        cli_script.write_text(cli_content)
        cli_script.chmod(0o755)
    
    def _update_gitignore(self, project_path: Path):
        """Update .gitignore to include Project Index patterns."""
        gitignore_path = project_path / '.gitignore'
        
        pi_patterns = [
            "# Project Index",
            ".project-index/cache/",
            ".project-index/logs/",
            ".project-index/temp/",
            ".project-index/*.log",
            ".project-index/*.tmp"
        ]
        
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            if ".project-index" not in gitignore_content:
                gitignore_content += "\n\n" + "\n".join(pi_patterns) + "\n"
                gitignore_path.write_text(gitignore_content)
        else:
            gitignore_path.write_text("\n".join(pi_patterns) + "\n")
    
    def _enhanced_post_installation_setup(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any],
        environment: ConfigurationEnvironment
    ) -> Dict[str, Any]:
        """Enhanced post-installation setup."""
        setup_results = {}
        
        # Initialize database if needed
        if self.config.get('setup_database', True):
            setup_results['database'] = self._setup_enhanced_database(project_path, config_profile)
        
        # Setup monitoring
        if self.config.get('setup_monitoring', True):
            setup_results['monitoring'] = self._setup_enhanced_monitoring(project_path, config_profile, environment)
        
        # Setup CI/CD integration
        ci_info = detection_result.get("enhanced_analysis", {}).get("ci_cd_analysis", {})
        if ci_info.get("has_ci_cd"):
            setup_results['ci_cd'] = self._setup_ci_cd_integration(project_path, config_profile, ci_info)
        
        # Setup IDE integration
        setup_results['ide'] = self._setup_ide_integration(project_path, config_profile)
        
        return setup_results
    
    def _setup_enhanced_database(self, project_path: Path, config_profile: ConfigurationProfile) -> Dict[str, Any]:
        """Setup enhanced database configuration."""
        pi_dir = project_path / '.project-index'
        
        db_config = {
            "type": "sqlite",
            "path": str(pi_dir / "project_index.db"),
            "auto_migrate": True,
            "performance_optimizations": {
                "wal_mode": True,
                "synchronous": "NORMAL",
                "cache_size": "10000",
                "temp_store": "MEMORY"
            },
            "backup": {
                "enabled": True,
                "interval_hours": 24,
                "retention_days": 7
            }
        }
        
        db_config_file = pi_dir / "database.json"
        with open(db_config_file, 'w') as f:
            json.dump(db_config, f, indent=2)
        
        return {"config_file": str(db_config_file), "type": "sqlite"}
    
    def _setup_enhanced_monitoring(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        environment: ConfigurationEnvironment
    ) -> Dict[str, Any]:
        """Setup enhanced monitoring configuration."""
        pi_dir = project_path / '.project-index'
        
        monitoring_config = {
            "enabled": True,
            "environment": environment.value,
            "file_monitoring": {
                "watch_patterns": config_profile.base_configuration.file_patterns.get("include", []),
                "ignore_patterns": config_profile.base_configuration.ignore_patterns,
                "debounce_seconds": config_profile.base_configuration.monitoring.get("debounce_seconds", 2.0),
                "recursive": True
            },
            "performance_monitoring": {
                "enabled": True,
                "metrics": ["cpu_usage", "memory_usage", "analysis_time", "queue_size"],
                "alert_thresholds": {
                    "cpu_usage_percent": 80,
                    "memory_usage_mb": config_profile.base_configuration.performance.get("memory_limit_mb", 512) * 0.9,
                    "analysis_timeout_seconds": config_profile.base_configuration.performance.get("timeout_seconds", 30) * 2
                }
            },
            "health_checks": {
                "enabled": True,
                "interval_seconds": 60,
                "endpoints": ["/health", "/metrics", "/status"]
            },
            "logging": {
                "level": "INFO" if environment == ConfigurationEnvironment.PRODUCTION else "DEBUG",
                "file": str(pi_dir / "logs" / "monitoring.log"),
                "rotation": {
                    "max_size_mb": 10,
                    "backup_count": 5
                }
            }
        }
        
        monitoring_config_file = pi_dir / "monitoring.json"
        with open(monitoring_config_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        return {"config_file": str(monitoring_config_file)}
    
    def _setup_ci_cd_integration(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        ci_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Setup CI/CD integration configurations."""
        integrations = {}
        
        # GitHub Actions integration
        if ci_info.get("platforms", {}).get("github_actions"):
            integrations["github_actions"] = self._create_github_actions_integration(project_path, config_profile)
        
        # GitLab CI integration
        if ci_info.get("platforms", {}).get("gitlab_ci"):
            integrations["gitlab_ci"] = self._create_gitlab_ci_integration(project_path, config_profile)
        
        return integrations
    
    def _create_github_actions_integration(self, project_path: Path, config_profile: ConfigurationProfile) -> Dict[str, str]:
        """Create GitHub Actions integration."""
        workflow_dir = project_path / '.github' / 'workflows'
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = f"""name: Project Index Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  project-index-analysis:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Project Index
      run: |
        python .project-index/cli.py --status
        python .project-index/cli.py --validate
    
    - name: Run Analysis
      run: |
        python .project-index/cli.py --analyze
    
    - name: Performance Check
      run: |
        python .project-index/cli.py --performance
    
    - name: Security Audit
      run: |
        python .project-index/cli.py --security
"""
        
        workflow_file = workflow_dir / 'project-index.yml'
        workflow_file.write_text(workflow_content)
        
        return {"workflow_file": str(workflow_file)}
    
    def _create_gitlab_ci_integration(self, project_path: Path, config_profile: ConfigurationProfile) -> Dict[str, str]:
        """Create GitLab CI integration."""
        gitlab_ci_content = f"""
project_index_analysis:
  stage: test
  script:
    - python .project-index/cli.py --status
    - python .project-index/cli.py --validate
    - python .project-index/cli.py --analyze
    - python .project-index/cli.py --performance
    - python .project-index/cli.py --security
  artifacts:
    reports:
      junit: .project-index/reports/junit.xml
    paths:
      - .project-index/reports/
    expire_in: 1 week
"""
        
        gitlab_ci_file = project_path / '.gitlab-ci-project-index.yml'
        gitlab_ci_file.write_text(gitlab_ci_content)
        
        return {"config_file": str(gitlab_ci_file)}
    
    def _setup_ide_integration(self, project_path: Path, config_profile: ConfigurationProfile) -> Dict[str, Any]:
        """Setup IDE integration configurations."""
        integrations = {}
        
        # VS Code integration
        vscode_dir = project_path / '.vscode'
        if vscode_dir.exists() or self.config.get('create_vscode_config', True):
            integrations["vscode"] = self._create_vscode_integration(project_path, config_profile)
        
        return integrations
    
    def _create_vscode_integration(self, project_path: Path, config_profile: ConfigurationProfile) -> Dict[str, str]:
        """Create VS Code integration."""
        vscode_dir = project_path / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        # Create tasks.json for Project Index commands
        tasks_config = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Project Index: Status",
                    "type": "shell",
                    "command": "python",
                    "args": [".project-index/cli.py", "--status"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Project Index: Analyze",
                    "type": "shell",
                    "command": "python",
                    "args": [".project-index/cli.py", "--analyze"],
                    "group": "build"
                },
                {
                    "label": "Project Index: Validate",
                    "type": "shell",
                    "command": "python",
                    "args": [".project-index/cli.py", "--validate"],
                    "group": "test"
                }
            ]
        }
        
        tasks_file = vscode_dir / 'tasks.json'
        with open(tasks_file, 'w') as f:
            json.dump(tasks_config, f, indent=2)
        
        return {"tasks_file": str(tasks_file)}
    
    def _generate_comprehensive_documentation(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive documentation."""
        pi_dir = project_path / '.project-index'
        docs_dir = pi_dir / 'docs'
        docs_dir.mkdir(exist_ok=True)
        
        docs = {}
        
        # Main README
        docs["readme"] = self._create_main_readme(docs_dir, config_profile, detection_result)
        
        # Configuration guide
        docs["config_guide"] = self._create_configuration_guide(docs_dir, config_profile)
        
        # Quick start guide
        docs["quickstart"] = self._create_quickstart_guide(docs_dir, config_profile, detection_result)
        
        # Troubleshooting guide
        docs["troubleshooting"] = self._create_troubleshooting_guide(docs_dir, config_profile, validation_result)
        
        return docs
    
    def _create_main_readme(
        self,
        docs_dir: Path,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any]
    ) -> str:
        """Create main README documentation."""
        readme_content = f"""# Project Index Configuration

## Overview

This Project Index configuration was automatically generated using enhanced intelligent detection and optimization.

**Profile ID:** `{config_profile.profile_id}`
**Generated:** {config_profile.created_timestamp.isoformat()}

## Project Analysis

### Detected Characteristics

- **Primary Language:** {detection_result.get('primary_language', {}).get('language', 'Unknown')}
- **Project Size:** {detection_result.get('size_analysis', {}).get('size_category', 'unknown')}
- **Frameworks:** {', '.join(f['framework'] for f in detection_result.get('detected_frameworks', []))}
- **Confidence Score:** {detection_result.get('confidence_score', 0):.1%}

### Performance Configuration

- **Optimization Level:** {config_profile.base_configuration.detection_metadata.get('optimization_level', 'unknown')}
- **Concurrent Analyses:** {config_profile.base_configuration.performance.get('max_concurrent_analyses', 'unknown')}
- **Memory Limit:** {config_profile.base_configuration.performance.get('memory_limit_mb', 'unknown')}MB
- **Cache Enabled:** {'Yes' if config_profile.base_configuration.performance.get('cache_enabled') else 'No'}

### Security Configuration

- **Security Level:** {config_profile.base_configuration.detection_metadata.get('security_level', 'unknown')}
- **Dependency Scanning:** {'Enabled' if config_profile.base_configuration.security.get('scan_dependencies') else 'Disabled'}
- **Vulnerability Checking:** {'Enabled' if config_profile.base_configuration.security.get('check_vulnerabilities') else 'Disabled'}

## Quick Commands

```bash
# Show status
python .project-index/cli.py --status

# Run analysis
python .project-index/cli.py --analyze

# Validate configuration
python .project-index/cli.py --validate

# Check performance metrics
python .project-index/cli.py --performance

# Security audit
python .project-index/cli.py --security
```

## Files and Directories

- `config/` - Configuration files for different environments
- `reports/` - Analysis and audit reports
- `scripts/` - Deployment and utility scripts
- `logs/` - Application logs
- `cache/` - Analysis cache files

## Support

For issues and questions:
- Check the troubleshooting guide: `docs/troubleshooting.md`
- Review configuration guide: `docs/configuration.md`
- Project Index documentation: https://docs.leanvibe.dev/project-index
"""
        
        readme_file = docs_dir / 'README.md'
        readme_file.write_text(readme_content)
        
        return str(readme_file)
    
    def _create_configuration_guide(self, docs_dir: Path, config_profile: ConfigurationProfile) -> str:
        """Create configuration guide."""
        guide_content = f"""# Configuration Guide

## Configuration Structure

The Project Index configuration consists of several main sections:

### Analysis Configuration

Controls how code analysis is performed:

```json
{json.dumps(config_profile.base_configuration.analysis, indent=2)}
```

### Performance Configuration

Controls system resource usage and performance:

```json
{json.dumps(config_profile.base_configuration.performance, indent=2)}
```

### Security Configuration

Controls security scanning and auditing:

```json
{json.dumps(config_profile.base_configuration.security, indent=2)}
```

## Environment Configurations

Different configurations are available for different environments:

{chr(10).join(f"- `{env.value}.json` - {env.value.title()} environment" for env in config_profile.environment_overrides.keys())}

## Customization

To customize the configuration:

1. Edit the appropriate environment file in `config/`
2. Validate changes: `python .project-index/cli.py --validate`
3. Restart monitoring if needed

## Performance Tuning

### For Small Projects
- Reduce `max_concurrent_analyses` to 1-2
- Lower `memory_limit_mb` to 256-512MB
- Shorter `timeout_seconds`

### For Large Projects
- Increase `max_concurrent_analyses` to 6-8
- Higher `memory_limit_mb` to 1-2GB
- Enable `aggressive_caching`

### For Enterprise Projects
- Maximum `max_concurrent_analyses` (8-12)
- High `memory_limit_mb` (2-4GB)
- Enable `distributed_processing`
- Enable all security features
"""
        
        guide_file = docs_dir / 'configuration.md'
        guide_file.write_text(guide_content)
        
        return str(guide_file)
    
    def _create_quickstart_guide(
        self,
        docs_dir: Path,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any]
    ) -> str:
        """Create quickstart guide."""
        quickstart_content = f"""# Quick Start Guide

## Getting Started

Welcome to Project Index! Your configuration has been optimized for your {detection_result.get('primary_language', {}).get('language', 'unknown')} project.

## 1. Verify Installation

```bash
python .project-index/cli.py --status
```

Expected output should show:
- Project name and profile ID
- Performance and security scores
- Configuration version

## 2. Run Initial Analysis

```bash
python .project-index/cli.py --analyze
```

This will:
- Scan your project files
- Build the project index
- Generate analysis reports

## 3. Check Results

### Performance Metrics
```bash
python .project-index/cli.py --performance
```

### Security Audit
```bash
python .project-index/cli.py --security
```

### Configuration Validation
```bash
python .project-index/cli.py --validate
```

## 4. Monitor Changes

Project Index automatically monitors file changes. You can check the monitoring status in the logs:

```bash
tail -f .project-index/logs/monitoring.log
```

## 5. Integration

### CI/CD Integration
{f"GitHub Actions workflow has been created in `.github/workflows/project-index.yml`" if detection_result.get('enhanced_analysis', {}).get('ci_cd_analysis', {}).get('platforms', {}).get('github_actions') else ""}
{f"GitLab CI configuration available in `.gitlab-ci-project-index.yml`" if detection_result.get('enhanced_analysis', {}).get('ci_cd_analysis', {}).get('platforms', {}).get('gitlab_ci') else ""}

### IDE Integration
- VS Code tasks are available in `.vscode/tasks.json`
- Use Ctrl+Shift+P → "Tasks: Run Task" → "Project Index: Status"

## Next Steps

1. **Customize Configuration**: Edit files in `.project-index/config/`
2. **Set Up Alerts**: Configure monitoring thresholds
3. **Review Reports**: Check `.project-index/reports/` regularly
4. **Update Documentation**: Keep project docs in sync

## Troubleshooting

If you encounter issues:
1. Check `.project-index/logs/` for error messages
2. Validate configuration: `python .project-index/cli.py --validate`
3. Review troubleshooting guide: `docs/troubleshooting.md`
"""
        
        quickstart_file = docs_dir / 'quickstart.md'
        quickstart_file.write_text(quickstart_content)
        
        return str(quickstart_file)
    
    def _create_troubleshooting_guide(
        self,
        docs_dir: Path,
        config_profile: ConfigurationProfile,
        validation_result: Dict[str, Any]
    ) -> str:
        """Create troubleshooting guide."""
        troubleshooting_content = f"""# Troubleshooting Guide

## Common Issues

### Configuration Validation Errors

{chr(10).join(f"- {error}" for error in validation_result.get('errors', [])) if validation_result.get('errors') else "No validation errors found."}

### Performance Issues

**High Memory Usage:**
- Reduce `memory_limit_mb` in configuration
- Lower `max_concurrent_analyses`
- Clear cache: `rm -rf .project-index/cache/*`

**Slow Analysis:**
- Increase `timeout_seconds`
- Enable caching if disabled
- Reduce file pattern scope

**High CPU Usage:**
- Lower `max_concurrent_analyses`
- Increase `debounce_interval`
- Check for file monitoring loops

### Security Warnings

{chr(10).join(f"- {warning}" for warning in validation_result.get('warnings', [])) if validation_result.get('warnings') else "No security warnings found."}

## Log Files

Check these log files for detailed error information:

- **General Logs**: `.project-index/logs/application.log`
- **Monitoring Logs**: `.project-index/logs/monitoring.log`
- **Analysis Logs**: `.project-index/logs/analysis.log`

## Configuration Reset

To reset to default configuration:

```bash
# Backup current config
cp .project-index/config.json .project-index/config.backup.json

# Regenerate configuration
python -m enhanced_configuration_generator detection_result.json --output .project-index/config/
```

## Performance Diagnostics

```bash
# Check system resources
python .project-index/cli.py --performance

# Validate all configurations
python .project-index/cli.py --validate

# Check file patterns
grep -E "include|exclude" .project-index/config.json
```

## Getting Help

1. **Documentation**: https://docs.leanvibe.dev/project-index
2. **Issues**: https://github.com/leanvibe/bee-hive/issues
3. **Community**: https://discord.gg/leanvibe

## Configuration Debugging

### Common Configuration Issues

1. **Invalid File Patterns**
   - Check glob pattern syntax
   - Ensure no conflicting include/exclude patterns

2. **Resource Limits**
   - Memory limit too high for system
   - Concurrency exceeds CPU cores

3. **Environment Mismatches**
   - Production settings in development
   - Missing security requirements

### Validation Commands

```bash
# Validate specific environment
python -c "
from configuration_validation_schemas import ConfigurationValidator, SchemaLevel
import json

with open('.project-index/config.json') as f:
    config = json.load(f)

validator = ConfigurationValidator(SchemaLevel.STRICT)
result = validator.validate_configuration(config, environment='production')
print(json.dumps(result, indent=2))
"
```
"""
        
        troubleshooting_file = docs_dir / 'troubleshooting.md'
        troubleshooting_file.write_text(troubleshooting_content)
        
        return str(troubleshooting_file)
    
    def _generate_enhanced_next_steps(
        self,
        project_path: Path,
        config_profile: ConfigurationProfile,
        detection_result: Dict[str, Any],
        environment: ConfigurationEnvironment
    ) -> List[str]:
        """Generate enhanced next steps."""
        steps = [
            "📖 Read the Quick Start Guide: .project-index/docs/quickstart.md",
            "✅ Check installation status: python .project-index/cli.py --status",
            "🔍 Run initial analysis: python .project-index/cli.py --analyze",
            "🔧 Validate configuration: python .project-index/cli.py --validate"
        ]
        
        # Add environment-specific steps
        if environment == ConfigurationEnvironment.PRODUCTION:
            steps.extend([
                "🔒 Review security audit: python .project-index/cli.py --security",
                "📊 Monitor performance: python .project-index/cli.py --performance",
                "🚨 Set up alerting thresholds in .project-index/monitoring.json"
            ])
        elif environment == ConfigurationEnvironment.DEVELOPMENT:
            steps.extend([
                "🔧 Customize configuration: .project-index/config/development.json",
                "🧪 Integrate with test workflow: Add to your test scripts"
            ])
        
        # Add framework-specific steps
        frameworks = [f["framework"] for f in detection_result.get("detected_frameworks", [])]
        if "django" in frameworks:
            steps.append("🐍 Review Django-specific settings in configuration")
        if "react" in frameworks:
            steps.append("⚛️ Enable bundle analysis for React optimization")
        
        # Add CI/CD integration steps
        ci_info = detection_result.get("enhanced_analysis", {}).get("ci_cd_analysis", {})
        if ci_info.get("has_ci_cd"):
            active_platforms = ci_info.get("active_platforms", [])
            for platform in active_platforms:
                if platform == "github_actions":
                    steps.append("🔄 Review GitHub Actions workflow: .github/workflows/project-index.yml")
                elif platform == "gitlab_ci":
                    steps.append("🔄 Integrate GitLab CI: Include .gitlab-ci-project-index.yml")
        
        # Add documentation steps
        doc_info = detection_result.get("enhanced_analysis", {}).get("documentation_analysis", {})
        if not doc_info.get("well_documented"):
            steps.append("📝 Improve project documentation for better analysis")
        
        # Add final steps
        steps.extend([
            "🌐 Explore the web dashboard (if available): http://localhost:8000/dashboard",
            "🔌 Set up IDE integration: .vscode/tasks.json",
            "📚 Read full documentation: https://docs.leanvibe.dev/project-index"
        ])
        
        return steps
    
    def _display_installation_summary(self, result: Dict[str, Any]):
        """Display comprehensive installation summary."""
        print(f"\n🎉 ENHANCED INSTALLATION COMPLETE!")
        print(f"━" * 60)
        
        if result["success"]:
            print(f"✅ Status: Successfully installed")
            print(f"📁 Project: {Path(result['project_path']).name}")
            print(f"🆔 Profile ID: {result['configuration']['profile_id']}")
            print(f"🎯 Strategy: {result['configuration']['strategy']}")
            print(f"🌍 Environment: {result['configuration']['environment']}")
            
            # Performance summary
            perf_metrics = result.get("performance_metrics", {})
            if perf_metrics:
                print(f"📊 Performance Score: {perf_metrics.get('performance_score', 0)}/100")
                print(f"💾 Memory Usage: ~{perf_metrics.get('estimated_memory_usage_mb', 0)}MB")
            
            # Security summary
            security_audit = result.get("security_audit", {})
            if security_audit:
                print(f"🔒 Security Score: {security_audit.get('security_score', 0)}/100")
            
            # Validation summary
            validation = result.get("validation_result", {})
            if validation:
                status = "✅ PASSED" if validation.get("valid") else "❌ FAILED"
                print(f"✅ Validation: {status}")
                
                if validation.get("errors"):
                    print(f"❌ Errors: {len(validation['errors'])}")
                if validation.get("warnings"):
                    print(f"⚠️ Warnings: {len(validation['warnings'])}")
            
            # Documentation
            docs = result.get("documentation", {})
            if docs:
                print(f"📚 Documentation: {len(docs)} files generated")
            
            print(f"\n🎯 NEXT STEPS")
            print(f"━" * 30)
            next_steps = result.get("next_steps", [])
            for i, step in enumerate(next_steps[:5], 1):
                print(f"{i}. {step}")
            
            if len(next_steps) > 5:
                print(f"   ...and {len(next_steps) - 5} more steps in the Quick Start Guide")
            
        else:
            print(f"❌ Status: Installation failed")
            print(f"💥 Reason: {result.get('reason', 'Unknown error')}")
    
    # Helper methods
    
    def _convert_config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, '_asdict'):
            return config._asdict()
        else:
            return dict(config) if isinstance(config, dict) else {}
    
    def _serialize_detection_result(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize detection result for JSON storage."""
        # Remove any non-serializable objects
        serializable_result = {}
        
        for key, value in detection_result.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                serializable_result[key] = value
            elif value is None:
                serializable_result[key] = None
            else:
                serializable_result[key] = str(value)
        
        return serializable_result


def main():
    """CLI entry point for enhanced universal installer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Universal Project Index Installer")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--strategy", choices=[s.value for s in ConfigurationStrategy], 
                       default="production", help="Configuration strategy")
    parser.add_argument("--environment", choices=[e.value for e in ConfigurationEnvironment], 
                       default="production", help="Target environment")
    parser.add_argument("--template", help="Specific template to use")
    parser.add_argument("--validation-level", choices=[v.value for v in ValidationLevel], 
                       default="standard", help="Validation level")
    parser.add_argument("--non-interactive", action="store_true", help="Run without user prompts")
    parser.add_argument("--auto-optimize", action="store_true", default=True, help="Enable auto-optimization")
    parser.add_argument("--backup", action="store_true", default=True, help="Backup existing configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Prepare workflow configuration
    workflow_config = {
        'interactive': not args.non_interactive,
        'auto_optimize': args.auto_optimize,
        'backup_existing': args.backup
    }
    
    try:
        # Run enhanced installation
        workflow = EnhancedInstallerWorkflow(workflow_config)
        result = workflow.install_with_enhanced_detection(
            args.project_path,
            ConfigurationStrategy(args.strategy),
            ConfigurationEnvironment(args.environment),
            args.template,
            ValidationLevel(args.validation_level)
        )
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())