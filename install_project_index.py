#!/usr/bin/env python3
"""
Project Index Universal Installer - Complete Integration

This script ties together all the CLI components to provide a seamless
one-command installation experience for the Project Index system.

Usage:
    python install_project_index.py [options]
    python install_project_index.py --help
"""

import sys
import os
import argparse
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Add the CLI modules to the path
sys.path.insert(0, str(Path(__file__).parent))

from project_index_cli import ProjectIndexCLI, InstallationProfile, Colors
from cli.project_detector import ProjectDetector, ProjectAnalysis
from cli.docker_manager import DockerManager, DeploymentProfile
from cli.framework_adapters import FrameworkAdapterManager
from cli.config_generator import ConfigurationGenerator, OptimizationLevel, ConfigFormat
from cli.validation_framework import ValidationFramework, ValidationLevel

class ProjectIndexInstaller:
    """Complete Project Index installation orchestrator"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logging()
        
        # Initialize all components
        self.cli = ProjectIndexCLI()
        self.project_detector = ProjectDetector(self.logger)
        self.docker_manager = DockerManager(self.logger)
        self.framework_manager = FrameworkAdapterManager(self.logger)
        self.config_generator = ConfigurationGenerator(self.logger)
        self.validator = ValidationFramework(self.logger)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("project-index-installer")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "installation.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def install_complete_system(self, project_path: str, 
                                    profile: Optional[InstallationProfile] = None,
                                    auto_confirm: bool = False,
                                    skip_validation: bool = False) -> bool:
        """Complete system installation workflow"""
        
        try:
            # Step 1: Show banner and initialize
            self.cli.show_banner()
            print(Colors.header("🚀 Starting Complete Project Index Installation"))
            
            # Step 2: System validation
            print(Colors.step("Step 1: System Requirements Validation"))
            system_info = self.cli.get_system_info()
            valid, errors = self.cli.validate_system_requirements(system_info)
            
            if not valid:
                print(Colors.error("System validation failed. Cannot proceed."))
                for error in errors:
                    print(f"  - {error}")
                return False
            
            # Step 3: Project analysis
            print(Colors.step("Step 2: Project Analysis and Detection"))
            analysis = self.project_detector.analyze_project(project_path)
            
            print(Colors.info(f"Project: {analysis.project_name}"))
            print(Colors.info(f"Type: {analysis.project_type.value}"))
            print(Colors.info(f"Primary Language: {analysis.primary_language.value}"))
            print(Colors.info(f"Files: {analysis.file_count:,}"))
            print(Colors.info(f"Frameworks: {len(analysis.frameworks)} detected"))
            
            if analysis.frameworks:
                for framework in analysis.frameworks[:5]:  # Show top 5
                    confidence = "★" * int(framework.confidence * 5)
                    print(f"  • {framework.name} {confidence}")
            
            # Step 4: Configuration generation
            print(Colors.step("Step 3: Intelligent Configuration Generation"))
            
            # Map installation profile to deployment profile
            profile_mapping = {
                InstallationProfile.SMALL: DeploymentProfile.SMALL,
                InstallationProfile.MEDIUM: DeploymentProfile.MEDIUM,
                InstallationProfile.LARGE: DeploymentProfile.LARGE,
                InstallationProfile.ENTERPRISE: DeploymentProfile.ENTERPRISE
            }
            
            if not profile:
                # Recommend profile based on analysis
                file_count = analysis.file_count
                if file_count < 1000:
                    profile = InstallationProfile.SMALL
                elif file_count < 10000:
                    profile = InstallationProfile.MEDIUM
                else:
                    profile = InstallationProfile.LARGE
            
            deployment_profile = profile_mapping[profile]
            
            print(Colors.info(f"Selected profile: {profile.value}"))
            
            # Generate configuration
            config = self.config_generator.generate_configuration(
                analysis=analysis,
                deployment_profile=deployment_profile,
                user_preferences={}
            )
            
            print(Colors.success(f"Configuration generated (optimization: {config.optimization_level.value})"))
            
            # Step 5: Docker infrastructure setup
            print(Colors.step("Step 4: Docker Infrastructure Preparation"))
            
            # Validate Docker environment
            docker_valid, docker_errors = self.docker_manager.validate_docker_environment()
            if not docker_valid:
                print(Colors.error("Docker environment validation failed:"))
                for error in docker_errors:
                    print(f"  - {error}")
                return False
            
            # Generate Docker infrastructure config
            ports = {
                'api': 8100,
                'dashboard': 8101,
                'metrics': 9090
            }
            
            passwords = {
                'database': config.environment_variables.get('POSTGRES_PASSWORD'),
                'redis': config.environment_variables.get('REDIS_PASSWORD')
            }
            
            infra_config = self.docker_manager.generate_infrastructure_config(
                profile=deployment_profile,
                project_name=analysis.project_name,
                project_path=analysis.project_path,
                detected_frameworks=[f.name for f in analysis.frameworks],
                ports=ports,
                passwords=passwords
            )
            
            print(Colors.success(f"Infrastructure config generated ({len(infra_config.services)} services)"))
            
            # Step 6: Framework integrations
            print(Colors.step("Step 5: Framework Integration Generation"))
            
            integrations = self.framework_manager.generate_all_integrations(
                analysis, {'api_url': f'http://localhost:{ports["api"]}'}
            )
            
            if integrations:
                print(Colors.success(f"Generated {len(integrations)} framework integrations"))
                for integration in integrations:
                    print(f"  • {integration.framework_name}: {len(integration.generated_files)} files")
            else:
                print(Colors.info("No framework integrations available"))
            
            # Step 7: Create output directory and files
            print(Colors.step("Step 6: Creating Installation Files"))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"project-index-installation-{timestamp}")
            output_dir.mkdir(exist_ok=True)
            
            # Generate Docker files
            compose_file = self.docker_manager.generate_docker_compose_file(infra_config, output_dir)
            env_file = self.docker_manager.generate_environment_file(infra_config, output_dir)
            self.docker_manager.generate_supporting_files(infra_config, output_dir)
            
            # Export configuration
            config_formats = [ConfigFormat.JSON, ConfigFormat.YAML, ConfigFormat.ENV]
            config_files = self.config_generator.export_configuration(config, output_dir, config_formats)
            
            # Write integration files
            if integrations:
                integration_files = self.framework_manager.write_integration_files(integrations, output_dir)
                print(Colors.success(f"Created {len(integration_files)} integration files"))
            
            print(Colors.success(f"Installation files created in: {output_dir}"))
            
            # Step 8: User confirmation
            if not auto_confirm:
                print(Colors.header("📋 Installation Summary"))
                print(f"Project: {Colors.colored(analysis.project_name, Colors.BOLD)}")
                print(f"Profile: {Colors.colored(profile.value, Colors.GREEN)}")
                print(f"Services: {len(infra_config.services)}")
                print(f"Integrations: {len(integrations)}")
                print(f"Output Directory: {output_dir}")
                print()
                
                if not self.cli.ask_yes_no("Proceed with deployment?", default=True):
                    print(Colors.info("Installation cancelled. Files are available in the output directory."))
                    return True
            
            # Step 9: Deploy infrastructure
            print(Colors.step("Step 7: Deploying Infrastructure"))
            
            deployment_success = self.docker_manager.deploy_infrastructure(infra_config, output_dir)
            
            if not deployment_success:
                print(Colors.error("Infrastructure deployment failed"))
                return False
            
            print(Colors.success("Infrastructure deployed successfully"))
            
            # Step 10: Post-deployment validation
            if not skip_validation:
                print(Colors.step("Step 8: Post-Deployment Validation"))
                
                # Wait a moment for services to start
                print(Colors.info("Waiting for services to initialize..."))
                await asyncio.sleep(30)
                
                # Run validation
                validation_report = await self.validator.run_validation(
                    validation_level=ValidationLevel.STANDARD,
                    config_path=output_dir,
                    project_path=Path(analysis.project_path)
                )
                
                print(f"Validation Status: {Colors.colored(validation_report.overall_status.value.upper(), Colors.GREEN if validation_report.overall_status.value == 'healthy' else Colors.YELLOW)}")
                print(f"Checks: {validation_report.passed_checks} passed, {validation_report.warning_checks} warnings, {validation_report.failed_checks} failed")
                
                # Export validation report
                validation_file = self.validator.export_report(validation_report, output_dir, "html")
                print(Colors.info(f"Validation report: {validation_file}"))
                
                if validation_report.critical_issues > 0:
                    print(Colors.warning("Critical issues detected - check validation report"))
                    for result in validation_report.validation_results:
                        if result.status.value == 'critical':
                            print(f"  ❌ {result.check_name}: {result.message}")
            
            # Step 11: Success and next steps
            print(Colors.header("🎉 Installation Complete!"))
            
            print(Colors.success("Project Index has been successfully installed!"))
            print()
            print(Colors.info("📊 Services Available:"))
            print(f"  • API: http://localhost:{ports['api']}")
            print(f"  • API Documentation: http://localhost:{ports['api']}/docs")
            print(f"  • Health Check: http://localhost:{ports['api']}/health")
            
            if config.monitoring_enabled:
                print(f"  • Metrics: http://localhost:{ports['metrics']}")
            
            print()
            print(Colors.info("📁 Installation Directory:"))
            print(f"  {output_dir.absolute()}")
            
            print()
            print(Colors.info("🔧 Management Commands:"))
            print(f"  cd {output_dir}")
            print("  ./scripts/health-check.sh     # Check system health")
            print("  docker-compose logs           # View service logs")
            print("  docker-compose stop           # Stop all services")
            print("  docker-compose start          # Start all services")
            
            if integrations:
                print()
                print(Colors.info("🔗 Framework Integrations:"))
                for integration in integrations:
                    print(f"  • {integration.framework_name}: integrations/{integration.framework_name.lower()}/")
            
            print()
            print(Colors.info("📚 Next Steps:"))
            print("  1. Test the API endpoints to ensure everything is working")
            print("  2. Review the framework integrations for your project")
            print("  3. Configure your development environment")
            print("  4. Set up monitoring and alerts if needed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            print(Colors.error(f"Installation failed: {e}"))
            return False

def main():
    """Main entry point for the installer"""
    
    parser = argparse.ArgumentParser(
        description="Project Index Universal Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_project_index.py /path/to/project
  python install_project_index.py /path/to/project --profile large
  python install_project_index.py /path/to/project --auto-confirm
  python install_project_index.py /path/to/project --skip-validation
        """
    )
    
    parser.add_argument("project_path", help="Path to the project to analyze and index")
    parser.add_argument("--profile", 
                       choices=[p.value for p in InstallationProfile],
                       help="Installation profile (auto-detected if not specified)")
    parser.add_argument("--auto-confirm", action="store_true",
                       help="Automatically confirm all prompts")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip post-deployment validation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(Colors.error(f"Project path does not exist: {project_path}"))
        sys.exit(1)
    
    if not project_path.is_dir():
        print(Colors.error(f"Project path is not a directory: {project_path}"))
        sys.exit(1)
    
    # Convert profile string to enum
    profile = None
    if args.profile:
        profile = InstallationProfile(args.profile)
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create installer and run
    installer = ProjectIndexInstaller()
    
    try:
        # Run the async installation
        success = asyncio.run(installer.install_complete_system(
            project_path=str(project_path),
            profile=profile,
            auto_confirm=args.auto_confirm,
            skip_validation=args.skip_validation
        ))
        
        if success:
            print(Colors.success("\\n✅ Installation completed successfully!"))
            sys.exit(0)
        else:
            print(Colors.error("\\n❌ Installation failed!"))
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(Colors.warning("\\n⚠️  Installation interrupted by user"))
        sys.exit(1)
    except Exception as e:
        print(Colors.error(f"\\n❌ Unexpected error: {e}"))
        sys.exit(1)

if __name__ == "__main__":
    main()