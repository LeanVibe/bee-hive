#!/usr/bin/env python3
"""
Universal Installer Integration for Intelligent Project Detection
================================================================

Integrates the intelligent project detection system with the universal installer
to provide automatic configuration generation and optimal setup for any codebase.

Features:
- Auto-detection during installation
- Smart configuration generation
- Framework-specific optimizations
- Performance tuning based on project size
- Security configuration
- CI/CD integration templates

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class UniversalInstallerIntegration:
    """
    Integration layer between the intelligent project detection system
    and the universal installer for seamless project setup.
    """
    
    def __init__(self, installer_config: Optional[Dict[str, Any]] = None):
        """Initialize the integration system."""
        self.config = installer_config or {}
        self.auto_detect = self.config.get('auto_detect', True)
        self.auto_configure = self.config.get('auto_configure', True)
        self.interactive_mode = self.config.get('interactive', True)
        
        # Initialize detection components
        self.detector = None
        self.dependency_analyzer = None
        self.structure_analyzer = None
        self.config_generator = None
        
        self._initialize_components()
        
        logger.info("Universal installer integration initialized")
    
    def _initialize_components(self):
        """Initialize detection system components."""
        try:
            from app.project_index.intelligent_detector import IntelligentProjectDetector
            from enhanced_dependency_analyzer import EnhancedDependencyAnalyzer
            from advanced_structure_analyzer import AdvancedStructureAnalyzer
            from intelligent_config_generator import IntelligentConfigGenerator
            
            self.detector = IntelligentProjectDetector()
            self.dependency_analyzer = EnhancedDependencyAnalyzer()
            self.structure_analyzer = AdvancedStructureAnalyzer()
            self.config_generator = IntelligentConfigGenerator()
            
            logger.info("Detection components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to initialize detection components: {e}")
            # Fallback to manual configuration
            self.auto_detect = False
            self.auto_configure = False
    
    def install_with_detection(
        self,
        project_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Install Project Index with automatic detection and configuration.
        
        Args:
            project_path: Path to the project directory
            options: Installation options and overrides
            
        Returns:
            Installation result with detection metadata
        """
        project_path = Path(project_path).resolve()
        options = options or {}
        
        print(f"🚀 Installing Project Index for: {project_path.name}")
        print(f"📂 Project Path: {project_path}")
        
        # Step 1: Validate project directory
        if not self._validate_project_directory(project_path):
            raise ValueError(f"Invalid project directory: {project_path}")
        
        # Step 2: Run intelligent detection
        detection_result = None
        if self.auto_detect and self.detector:
            print(f"\n🔍 Running intelligent project detection...")
            try:
                detection_result = self._run_comprehensive_detection(project_path)
                self._display_detection_summary(detection_result)
                
                # Ask for confirmation if interactive
                if self.interactive_mode:
                    if not self._confirm_detection_results(detection_result):
                        detection_result = None
                        
            except Exception as e:
                logger.warning(f"Detection failed: {e}")
                print(f"⚠️  Detection failed, proceeding with manual configuration")
        
        # Step 3: Generate configuration
        config = None
        if self.auto_configure and detection_result and self.config_generator:
            print(f"\n⚙️  Generating optimal configuration...")
            try:
                config = self._generate_intelligent_config(detection_result, options)
                self._display_config_summary(config)
                
                if self.interactive_mode:
                    if not self._confirm_configuration(config):
                        config = None
                        
            except Exception as e:
                logger.warning(f"Config generation failed: {e}")
                print(f"⚠️  Config generation failed, using default configuration")
        
        # Step 4: Install Project Index
        print(f"\n📦 Installing Project Index components...")
        installation_result = self._install_project_index(project_path, config, options)
        
        # Step 5: Post-installation setup
        if installation_result['success']:
            print(f"\n🔧 Running post-installation setup...")
            self._post_installation_setup(project_path, config, detection_result)
            
            # Generate quick start guide
            self._generate_quick_start_guide(project_path, config, detection_result)
        
        # Step 6: Prepare result
        result = {
            'success': installation_result['success'],
            'project_path': str(project_path),
            'detection_result': detection_result,
            'configuration': config,
            'installation_log': installation_result.get('log', []),
            'next_steps': self._generate_next_steps(project_path, config, detection_result)
        }
        
        if installation_result['success']:
            print(f"\n✅ Project Index installation completed successfully!")
            self._display_next_steps(result['next_steps'])
        else:
            print(f"\n❌ Installation failed. Check logs for details.")
        
        return result
    
    def _validate_project_directory(self, project_path: Path) -> bool:
        """Validate that the project directory is suitable for Project Index."""
        if not project_path.exists():
            print(f"❌ Project directory does not exist: {project_path}")
            return False
        
        if not project_path.is_dir():
            print(f"❌ Path is not a directory: {project_path}")
            return False
        
        # Check if it's a git repository (recommended but not required)
        if not (project_path / '.git').exists():
            print(f"⚠️  Warning: Not a git repository. Project Index works best with version control.")
        
        # Check for write permissions
        if not os.access(project_path, os.W_OK):
            print(f"❌ No write permission for directory: {project_path}")
            return False
        
        return True
    
    def _run_comprehensive_detection(self, project_path: Path) -> Dict[str, Any]:
        """Run comprehensive project detection analysis."""
        start_time = time.time()
        
        # Run primary detection
        detection_result = self.detector.detect_project(project_path)
        
        # Run additional analyses
        dependency_graph = self.dependency_analyzer.analyze_project_dependencies(project_path)
        structure_analysis = self.structure_analyzer.analyze_project_structure(project_path)
        
        # Combine results
        comprehensive_result = {
            'primary_detection': {
                'project_path': detection_result.project_path,
                'primary_language': {
                    'language': detection_result.primary_language.language if detection_result.primary_language else None,
                    'confidence': detection_result.primary_language.confidence.value if detection_result.primary_language else None,
                    'file_count': detection_result.primary_language.file_count if detection_result.primary_language else 0
                },
                'secondary_languages': [
                    {
                        'language': lang.language,
                        'confidence': lang.confidence.value,
                        'file_count': lang.file_count
                    }
                    for lang in detection_result.secondary_languages
                ],
                'detected_frameworks': [
                    {
                        'framework': f.framework,
                        'confidence': f.confidence.value,
                        'evidence_files': f.evidence_files
                    }
                    for f in detection_result.detected_frameworks
                ],
                'size_analysis': {
                    'size_category': detection_result.size_analysis.size_category.value,
                    'file_count': detection_result.size_analysis.file_count,
                    'line_count': detection_result.size_analysis.line_count,
                    'complexity_score': detection_result.size_analysis.complexity_score
                },
                'confidence_score': detection_result.confidence_score
            },
            'dependency_analysis': {
                'total_dependencies': dependency_graph.total_count,
                'by_type': {k.value: v for k, v in dependency_graph.by_type.items()},
                'by_package_manager': {k.value: v for k, v in dependency_graph.by_package_manager.items()},
                'security_summary': {k.value: v for k, v in dependency_graph.security_summary.items()}
            },
            'structure_analysis': {
                'architecture_pattern': structure_analysis.architecture_pattern.value,
                'project_type': structure_analysis.project_type.value,
                'total_directories': structure_analysis.total_directories,
                'total_files': structure_analysis.total_files,
                'code_files': structure_analysis.code_files,
                'test_files': structure_analysis.test_files,
                'testing_strategy': structure_analysis.testing_analysis.testing_strategy.value,
                'confidence_scores': structure_analysis.confidence_scores
            },
            'detection_metadata': {
                'analysis_duration': time.time() - start_time,
                'timestamp': time.time()
            }
        }
        
        return comprehensive_result
    
    def _display_detection_summary(self, detection_result: Dict[str, Any]):
        """Display detection results summary."""
        primary = detection_result['primary_detection']
        structure = detection_result['structure_analysis']
        deps = detection_result['dependency_analysis']
        
        print(f"\n📊 DETECTION RESULTS")
        print(f"━" * 50)
        
        # Primary language
        if primary['primary_language']['language']:
            print(f"🎯 Primary Language: {primary['primary_language']['language'].title()}")
            print(f"   Confidence: {primary['primary_language']['confidence'].replace('_', ' ').title()}")
            print(f"   Files: {primary['primary_language']['file_count']:,}")
        
        # Secondary languages
        if primary['secondary_languages']:
            print(f"🔀 Secondary Languages:")
            for lang in primary['secondary_languages'][:3]:
                print(f"   • {lang['language'].title()}: {lang['file_count']} files")
        
        # Frameworks
        if primary['detected_frameworks']:
            print(f"🛠️  Detected Frameworks:")
            for framework in primary['detected_frameworks'][:3]:
                print(f"   • {framework['framework'].title()}: {framework['confidence'].replace('_', ' ').title()}")
        
        # Project characteristics
        print(f"📏 Project Size: {primary['size_analysis']['size_category'].title()}")
        print(f"🏗️  Architecture: {structure['architecture_pattern'].replace('_', ' ').title()}")
        print(f"📱 Project Type: {structure['project_type'].replace('_', ' ').title()}")
        
        # Dependencies
        if deps['total_dependencies'] > 0:
            print(f"📦 Dependencies: {deps['total_dependencies']} total")
            package_managers = [k for k, v in deps['by_package_manager'].items() if v > 0]
            if package_managers:
                print(f"   Package Managers: {', '.join(package_managers)}")
        
        # Overall confidence
        print(f"📈 Overall Confidence: {primary['confidence_score']:.1%}")
    
    def _confirm_detection_results(self, detection_result: Dict[str, Any]) -> bool:
        """Ask user to confirm detection results."""
        primary = detection_result['primary_detection']
        
        print(f"\n❓ Detection Summary:")
        print(f"   Language: {primary['primary_language']['language'] or 'Unknown'}")
        print(f"   Frameworks: {len(primary['detected_frameworks'])} detected")
        print(f"   Size: {primary['size_analysis']['size_category']}")
        print(f"   Confidence: {primary['confidence_score']:.1%}")
        
        while True:
            response = input(f"\n✅ Use these detection results? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please answer 'y' or 'n'")
    
    def _generate_intelligent_config(
        self,
        detection_result: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent configuration from detection results."""
        
        # Convert to format expected by config generator
        config_input = {
            'project_path': detection_result['primary_detection']['project_path'],
            'primary_language': detection_result['primary_detection']['primary_language'],
            'detected_frameworks': detection_result['primary_detection']['detected_frameworks'],
            'size_analysis': detection_result['primary_detection']['size_analysis'],
            'confidence_score': detection_result['primary_detection']['confidence_score']
        }
        
        # Determine optimization and security levels from options or detection
        optimization_level = options.get('optimization_level', 'balanced')
        security_level = options.get('security_level', 'standard')
        
        # Adjust based on project characteristics
        size_category = detection_result['primary_detection']['size_analysis']['size_category']
        if size_category == 'enterprise':
            optimization_level = options.get('optimization_level', 'enterprise')
            security_level = options.get('security_level', 'strict')
        elif size_category == 'small':
            optimization_level = options.get('optimization_level', 'minimal')
        
        from intelligent_config_generator import OptimizationLevel, SecurityLevel
        
        config = self.config_generator.generate_configuration(
            config_input,
            OptimizationLevel(optimization_level),
            SecurityLevel(security_level),
            options.get('custom_overrides', {})
        )
        
        # Convert to dict for easier handling
        config_dict = {
            'project_name': config.project_name,
            'project_path': config.project_path,
            'configuration_version': config.configuration_version,
            'detection_metadata': config.detection_metadata,
            'analysis': config.analysis,
            'file_patterns': config.file_patterns,
            'ignore_patterns': config.ignore_patterns,
            'monitoring': config.monitoring,
            'optimization': config.optimization,
            'performance': config.performance,
            'security': config.security,
            'integrations': config.integrations,
            'custom_rules': config.custom_rules,
            'configuration_notes': config.configuration_notes,
            'recommendations': config.recommendations
        }
        
        return config_dict
    
    def _display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary."""
        print(f"\n⚙️  CONFIGURATION SUMMARY")
        print(f"━" * 50)
        
        # Core settings
        print(f"📊 Analysis Settings:")
        analysis = config['analysis']
        print(f"   • AST Parsing: {'✅' if analysis.get('parse_ast') else '❌'}")
        print(f"   • Dependency Extraction: {'✅' if analysis.get('extract_dependencies') else '❌'}")
        print(f"   • Complexity Analysis: {'✅' if analysis.get('calculate_complexity') else '❌'}")
        print(f"   • Max File Size: {analysis.get('max_file_size_mb', 10)} MB")
        print(f"   • Timeout: {analysis.get('timeout_seconds', 30)}s")
        
        # Performance settings
        perf = config['performance']
        print(f"🚀 Performance Settings:")
        print(f"   • Concurrent Analyses: {perf.get('max_concurrent_analyses', 4)}")
        print(f"   • Batch Size: {perf.get('analysis_batch_size', 50)}")
        print(f"   • Cache Enabled: {'✅' if perf.get('cache_enabled') else '❌'}")
        print(f"   • Memory Limit: {perf.get('memory_limit_mb', 512)} MB")
        
        # File patterns
        patterns = config['file_patterns']
        print(f"📁 File Patterns:")
        print(f"   • Include Patterns: {len(patterns.get('include', []))}")
        print(f"   • Exclude Patterns: {len(patterns.get('exclude', []))}")
        print(f"   • Ignore Patterns: {len(config.get('ignore_patterns', []))}")
        
        # Security settings
        security = config['security']
        if security.get('enabled'):
            print(f"🔒 Security Settings:")
            print(f"   • Dependency Scanning: {'✅' if security.get('scan_dependencies') else '❌'}")
            print(f"   • Vulnerability Checks: {'✅' if security.get('check_vulnerabilities') else '❌'}")
            print(f"   • Sensitive File Audit: {'✅' if security.get('audit_sensitive_files') else '❌'}")
        
        # Optimization level
        metadata = config['detection_metadata']
        print(f"⚡ Optimization Level: {metadata.get('optimization_level', 'balanced').title()}")
        print(f"🛡️  Security Level: {metadata.get('security_level', 'standard').title()}")
    
    def _confirm_configuration(self, config: Dict[str, Any]) -> bool:
        """Ask user to confirm configuration."""
        metadata = config['detection_metadata']
        
        print(f"\n❓ Configuration Summary:")
        print(f"   Optimization: {metadata.get('optimization_level', 'balanced').title()}")
        print(f"   Security: {metadata.get('security_level', 'standard').title()}")
        print(f"   Concurrent Analyses: {config['performance'].get('max_concurrent_analyses', 4)}")
        print(f"   Memory Limit: {config['performance'].get('memory_limit_mb', 512)} MB")
        
        while True:
            response = input(f"\n✅ Use this configuration? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please answer 'y' or 'n'")
    
    def _install_project_index(
        self,
        project_path: Path,
        config: Optional[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Install Project Index with the given configuration."""
        
        installation_log = []
        
        try:
            # Create Project Index directory
            pi_dir = project_path / '.project-index'
            pi_dir.mkdir(exist_ok=True)
            installation_log.append("Created .project-index directory")
            
            # Save configuration
            if config:
                config_path = pi_dir / 'config.json'
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                installation_log.append(f"Saved configuration to {config_path}")
            
            # Copy Project Index scripts and templates
            self._install_project_index_files(project_path, installation_log)
            
            # Set up database if needed
            if options.get('setup_database', True):
                self._setup_database(project_path, installation_log)
            
            # Initialize monitoring
            if options.get('setup_monitoring', True):
                self._setup_monitoring(project_path, installation_log)
            
            installation_log.append("Project Index installation completed successfully")
            
            return {
                'success': True,
                'log': installation_log
            }
            
        except Exception as e:
            installation_log.append(f"Installation failed: {e}")
            logger.error(f"Installation failed: {e}")
            
            return {
                'success': False,
                'log': installation_log,
                'error': str(e)
            }
    
    def _install_project_index_files(self, project_path: Path, log: List[str]):
        """Install Project Index files and scripts."""
        pi_dir = project_path / '.project-index'
        
        # Create necessary subdirectories
        (pi_dir / 'cache').mkdir(exist_ok=True)
        (pi_dir / 'logs').mkdir(exist_ok=True)
        (pi_dir / 'temp').mkdir(exist_ok=True)
        
        # Create basic CLI script
        cli_script = pi_dir / 'cli.py'
        cli_script.write_text("""#!/usr/bin/env python3
\"\"\"
Project Index CLI
Basic command-line interface for project operations.
\"\"\"

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Project Index CLI")
    parser.add_argument("--status", action="store_true", help="Show project status")
    parser.add_argument("--analyze", action="store_true", help="Run project analysis")
    parser.add_argument("--config", help="Show or update configuration")
    
    args = parser.parse_args()
    
    if args.status:
        print("Project Index Status: Active")
        config_path = Path(".project-index/config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print(f"Project: {config.get('project_name', 'Unknown')}")
            print(f"Configuration Version: {config.get('configuration_version', 'Unknown')}")
    
    elif args.analyze:
        print("Running project analysis...")
        print("Analysis complete. (Placeholder)")
    
    elif args.config:
        config_path = Path(".project-index/config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print(json.dumps(config, indent=2))
        else:
            print("No configuration found")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
""")
        cli_script.chmod(0o755)
        log.append("Created CLI script")
        
        # Create gitignore for Project Index
        gitignore = pi_dir / '.gitignore'
        gitignore.write_text("""
# Project Index cache and temporary files
cache/
logs/
temp/
*.log
*.tmp
""")
        log.append("Created .gitignore for Project Index")
    
    def _setup_database(self, project_path: Path, log: List[str]):
        """Set up database for Project Index."""
        # For now, just create a placeholder database config
        pi_dir = project_path / '.project-index'
        db_config = pi_dir / 'database.json'
        
        db_config.write_text(json.dumps({
            'type': 'sqlite',
            'path': str(pi_dir / 'project_index.db'),
            'auto_migrate': True
        }, indent=2))
        
        log.append("Created database configuration")
    
    def _setup_monitoring(self, project_path: Path, log: List[str]):
        """Set up file monitoring for the project."""
        pi_dir = project_path / '.project-index'
        
        # Create monitoring configuration
        monitor_config = pi_dir / 'monitoring.json'
        monitor_config.write_text(json.dumps({
            'enabled': True,
            'watch_patterns': ['**/*.py', '**/*.js', '**/*.ts', '**/*.json'],
            'ignore_patterns': ['**/.git/**', '**/node_modules/**', '**/__pycache__/**'],
            'debounce_seconds': 2.0
        }, indent=2))
        
        log.append("Created monitoring configuration")
    
    def _post_installation_setup(
        self,
        project_path: Path,
        config: Optional[Dict[str, Any]],
        detection_result: Optional[Dict[str, Any]]
    ):
        """Run post-installation setup tasks."""
        
        # Create README for Project Index
        pi_dir = project_path / '.project-index'
        readme = pi_dir / 'README.md'
        
        readme_content = f"""# Project Index

This directory contains Project Index configuration and data for your project.

## Generated Configuration

"""
        
        if detection_result:
            primary = detection_result['primary_detection']
            readme_content += f"""### Project Detection Results

- **Primary Language**: {primary['primary_language']['language'] or 'Unknown'}
- **Project Size**: {primary['size_analysis']['size_category']}
- **Frameworks**: {len(primary['detected_frameworks'])} detected
- **Confidence**: {primary['confidence_score']:.1%}

"""
        
        if config:
            readme_content += f"""### Configuration Summary

- **Optimization Level**: {config['detection_metadata'].get('optimization_level', 'balanced')}
- **Security Level**: {config['detection_metadata'].get('security_level', 'standard')}
- **Analysis Features**: {'AST parsing, ' if config['analysis'].get('parse_ast') else ''}{'Dependency extraction, ' if config['analysis'].get('extract_dependencies') else ''}{'Complexity analysis' if config['analysis'].get('calculate_complexity') else ''}

"""
        
        readme_content += """## Usage

Use the CLI to interact with Project Index:

```bash
# Check status
python .project-index/cli.py --status

# Run analysis
python .project-index/cli.py --analyze

# View configuration
python .project-index/cli.py --config
```

## Files

- `config.json` - Main configuration
- `database.json` - Database settings
- `monitoring.json` - File monitoring settings
- `cli.py` - Command-line interface
- `cache/` - Analysis cache
- `logs/` - Log files

## Documentation

For more information, visit: https://docs.leanvibe.dev/project-index
"""
        
        readme.write_text(readme_content)
        print("📝 Created Project Index documentation")
    
    def _generate_quick_start_guide(
        self,
        project_path: Path,
        config: Optional[Dict[str, Any]],
        detection_result: Optional[Dict[str, Any]]
    ):
        """Generate a quick start guide for the user."""
        
        guide_path = project_path / 'PROJECT_INDEX_QUICKSTART.md'
        
        guide_content = f"""# Project Index Quick Start Guide

Welcome to Project Index! Your project has been automatically configured for optimal code intelligence.

## 🎯 What Was Detected

"""
        
        if detection_result:
            primary = detection_result['primary_detection']
            structure = detection_result['structure_analysis']
            
            guide_content += f"""- **Language**: {primary['primary_language']['language'] or 'Multiple'} ({primary['primary_language']['file_count']} files)
- **Architecture**: {structure['architecture_pattern'].replace('_', ' ').title()}
- **Project Type**: {structure['project_type'].replace('_', ' ').title()}
- **Size**: {primary['size_analysis']['size_category'].title()} project
- **Testing**: {structure['testing_strategy'].replace('_', ' ').title()} strategy

"""
            
            if primary['detected_frameworks']:
                guide_content += "**Frameworks Detected**:\n"
                for framework in primary['detected_frameworks'][:3]:
                    guide_content += f"- {framework['framework'].title()}\n"
                guide_content += "\n"
        
        guide_content += """## 🚀 Getting Started

### 1. Check Status
```bash
python .project-index/cli.py --status
```

### 2. Run Initial Analysis
```bash
python .project-index/cli.py --analyze
```

### 3. View Configuration
```bash
python .project-index/cli.py --config
```

## ⚙️ Configuration

Your project has been optimized with these settings:

"""
        
        if config:
            metadata = config['detection_metadata']
            perf = config['performance']
            
            guide_content += f"""- **Optimization Level**: {metadata.get('optimization_level', 'balanced').title()}
- **Security Level**: {metadata.get('security_level', 'standard').title()}
- **Concurrent Analyses**: {perf.get('max_concurrent_analyses', 4)}
- **Memory Limit**: {perf.get('memory_limit_mb', 512)} MB
- **Cache**: {'Enabled' if perf.get('cache_enabled') else 'Disabled'}

"""
        
        guide_content += """## 📁 File Organization

Project Index will monitor these file types:
"""
        
        if config and config.get('file_patterns', {}).get('include'):
            for pattern in config['file_patterns']['include'][:5]:
                guide_content += f"- `{pattern}`\n"
            if len(config['file_patterns']['include']) > 5:
                guide_content += f"- ...and {len(config['file_patterns']['include']) - 5} more patterns\n"
        
        guide_content += """
## 🔧 Customization

To modify your configuration:

1. Edit `.project-index/config.json`
2. Restart Project Index monitoring
3. Re-run analysis if needed

## 📚 Next Steps

"""
        
        if config and config.get('recommendations'):
            for rec in config['recommendations'][:3]:
                guide_content += f"- {rec}\n"
        
        guide_content += """
## 🆘 Support

- Documentation: https://docs.leanvibe.dev/project-index
- Issues: https://github.com/leanvibe/bee-hive/issues
- Community: https://discord.gg/leanvibe

---

*This guide was automatically generated based on your project characteristics.*
"""
        
        guide_path.write_text(guide_content)
        print(f"📖 Created quick start guide: {guide_path.name}")
    
    def _generate_next_steps(
        self,
        project_path: Path,
        config: Optional[Dict[str, Any]],
        detection_result: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate list of next steps for the user."""
        
        steps = [
            "Read the Quick Start Guide: PROJECT_INDEX_QUICKSTART.md",
            "Check Project Index status: python .project-index/cli.py --status",
            "Run initial analysis: python .project-index/cli.py --analyze"
        ]
        
        if config:
            # Add configuration-specific steps
            if config['security'].get('enabled'):
                steps.append("Review security settings in .project-index/config.json")
            
            if config.get('recommendations'):
                steps.extend(config['recommendations'][:2])
        
        if detection_result:
            # Add detection-specific steps
            structure = detection_result['structure_analysis']
            if structure['testing_strategy'] == 'none':
                steps.append("Consider adding unit tests to improve code quality analysis")
            
            if structure['architecture_pattern'] == 'monolithic':
                steps.append("Consider modularizing large components for better analysis")
        
        # Add general steps
        steps.extend([
            "Explore the Project Index dashboard (if available)",
            "Integrate with your IDE or editor",
            "Set up CI/CD integration for automated analysis"
        ])
        
        return steps
    
    def _display_next_steps(self, next_steps: List[str]):
        """Display next steps to the user."""
        print(f"\n🎯 NEXT STEPS")
        print(f"━" * 50)
        
        for i, step in enumerate(next_steps[:5], 1):
            print(f"{i}. {step}")
        
        if len(next_steps) > 5:
            print(f"   ...and {len(next_steps) - 5} more recommendations in the Quick Start Guide")


def main():
    """CLI entry point for universal installer integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Project Index Installer")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--optimization", choices=["minimal", "balanced", "performance", "enterprise"], 
                       default="balanced", help="Optimization level")
    parser.add_argument("--security", choices=["basic", "standard", "strict", "enterprise"], 
                       default="standard", help="Security level")
    parser.add_argument("--non-interactive", action="store_true", help="Run without user prompts")
    parser.add_argument("--skip-detection", action="store_true", help="Skip automatic detection")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Prepare installer configuration
    installer_config = {
        'auto_detect': not args.skip_detection,
        'auto_configure': True,
        'interactive': not args.non_interactive
    }
    
    # Prepare installation options
    options = {
        'optimization_level': args.optimization,
        'security_level': args.security,
        'setup_database': True,
        'setup_monitoring': True
    }
    
    try:
        # Run installation
        installer = UniversalInstallerIntegration(installer_config)
        result = installer.install_with_detection(args.project_path, options)
        
        if result['success']:
            print(f"\n🎉 Installation completed successfully!")
            return 0
        else:
            print(f"\n💥 Installation failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())