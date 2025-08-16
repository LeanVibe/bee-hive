#!/usr/bin/env python3
"""
Standalone Intelligent Project Detection System
==============================================

A comprehensive project analysis tool that can detect languages, frameworks,
dependencies, and generate optimal Project Index configurations for any codebase.

Usage:
    python standalone_project_detector.py /path/to/project [--output config.json] [--verbose]

Features:
- Language detection with confidence scoring
- Framework and tool identification
- Dependency analysis and parsing
- Project structure analysis
- Intelligent configuration generation
- Size estimation and complexity analysis

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import the intelligent detector
try:
    from app.project_index.intelligent_detector import (
        IntelligentProjectDetector, ProjectDetectionResult
    )
except ImportError:
    print("Error: Could not import IntelligentProjectDetector.")
    print("Make sure you're running this from the bee-hive project root directory.")
    sys.exit(1)


def format_detection_summary(result: ProjectDetectionResult) -> str:
    """Format detection result into a human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("🔍 INTELLIGENT PROJECT DETECTION RESULTS")
    lines.append("=" * 60)
    
    # Project overview
    lines.append(f"\n📂 Project: {result.project_path}")
    lines.append(f"📅 Analysis Date: {result.detection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"⏱️  Analysis Duration: {result.analysis_duration:.2f} seconds")
    lines.append(f"📊 Overall Confidence: {result.confidence_score:.2%}")
    
    # Primary language
    if result.primary_language:
        lang = result.primary_language
        lines.append(f"\n🎯 PRIMARY LANGUAGE")
        lines.append(f"   Language: {lang.language.title()}")
        lines.append(f"   Confidence: {lang.confidence.value.replace('_', ' ').title()}")
        lines.append(f"   Files: {lang.file_count}")
        lines.append(f"   Lines of Code: {lang.total_lines:,}")
        if lang.primary_files:
            lines.append(f"   Key Files: {', '.join(lang.primary_files[:3])}")
    
    # Secondary languages
    if result.secondary_languages:
        lines.append(f"\n🔀 SECONDARY LANGUAGES")
        for lang in result.secondary_languages[:3]:
            lines.append(f"   • {lang.language.title()}: {lang.file_count} files, {lang.total_lines:,} lines")
    
    # Detected frameworks
    if result.detected_frameworks:
        lines.append(f"\n🛠️  DETECTED FRAMEWORKS & TOOLS")
        for framework in result.detected_frameworks[:5]:
            confidence_icon = {
                'very_high': '🟢',
                'high': '🔵', 
                'medium': '🟡',
                'low': '🔴'
            }.get(framework.confidence.value, '⚪')
            
            lines.append(f"   {confidence_icon} {framework.framework.title()}")
            if framework.version:
                lines.append(f"      Version: {framework.version}")
            lines.append(f"      Confidence: {framework.confidence.value.replace('_', ' ').title()}")
            if framework.evidence_files:
                lines.append(f"      Evidence: {', '.join(framework.evidence_files[:2])}")
    
    # Dependencies
    if result.dependency_analysis:
        lines.append(f"\n📦 DEPENDENCY ANALYSIS")
        for dep in result.dependency_analysis:
            lines.append(f"   • {dep.package_manager.title()}")
            lines.append(f"     Total Dependencies: {dep.total_dependencies}")
            lines.append(f"     Production: {dep.production_dependencies}")
            lines.append(f"     Development: {dep.dev_dependencies}")
            if dep.major_dependencies:
                major_deps = [d['name'] for d in dep.major_dependencies[:3]]
                lines.append(f"     Key Dependencies: {', '.join(major_deps)}")
    
    # Project size and structure
    size = result.size_analysis
    lines.append(f"\n📏 PROJECT SIZE & COMPLEXITY")
    lines.append(f"   Size Category: {size.size_category.value.title()}")
    lines.append(f"   Files: {size.file_count:,}")
    lines.append(f"   Lines of Code: {size.line_count:,}")
    lines.append(f"   Directories: {size.directory_count}")
    lines.append(f"   Complexity Score: {size.complexity_score:.2f}/1.0")
    lines.append(f"   Estimated Team Size: {size.estimated_team_size}")
    lines.append(f"   Development Stage: {size.development_stage.title()}")
    
    # Structure analysis
    structure = result.structure_analysis
    lines.append(f"\n🏗️  PROJECT STRUCTURE")
    lines.append(f"   Structure Type: {structure.structure_type.title()}")
    if structure.entry_points:
        lines.append(f"   Entry Points: {', '.join(structure.entry_points[:3])}")
    if structure.core_directories:
        lines.append(f"   Core Directories: {', '.join(structure.core_directories[:3])}")
    if structure.test_directories:
        lines.append(f"   Test Directories: {', '.join(structure.test_directories[:2])}")
    
    # Configuration summary
    config = result.recommended_config
    lines.append(f"\n⚙️  RECOMMENDED CONFIGURATION")
    lines.append(f"   File Patterns: {len(config.file_patterns.get('include', []))} include patterns")
    lines.append(f"   Ignore Patterns: {len(config.ignore_patterns)} ignore patterns")
    lines.append(f"   Max File Size: {config.analysis_settings.get('max_file_size_mb', 10)} MB")
    lines.append(f"   Analysis Timeout: {config.analysis_settings.get('timeout_seconds', 30)}s")
    lines.append(f"   Context Optimization: {'Enabled' if config.optimization_settings.get('context_optimization_enabled') else 'Disabled'}")
    
    # Warnings and recommendations
    if result.warnings:
        lines.append(f"\n⚠️  WARNINGS")
        for warning in result.warnings:
            lines.append(f"   • {warning}")
    
    if result.recommendations:
        lines.append(f"\n💡 RECOMMENDATIONS")
        for rec in result.recommendations:
            lines.append(f"   • {rec}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def generate_project_index_config(result: ProjectDetectionResult) -> Dict[str, Any]:
    """Generate a complete Project Index configuration from detection results."""
    config = result.recommended_config
    
    # Convert to Project Index configuration format
    project_config = {
        "project_name": Path(result.project_path).name,
        "project_path": result.project_path,
        "detection_metadata": {
            "detection_timestamp": result.detection_timestamp.isoformat(),
            "primary_language": result.primary_language.language if result.primary_language else None,
            "frameworks": [f.framework for f in result.detected_frameworks],
            "project_size": result.size_analysis.size_category.value,
            "confidence_score": result.confidence_score
        },
        
        # Analysis configuration
        "analysis": {
            "enabled": True,
            "parse_ast": config.analysis_settings.get("parse_ast", True),
            "extract_dependencies": config.analysis_settings.get("extract_dependencies", True),
            "calculate_complexity": config.analysis_settings.get("calculate_complexity", True),
            "analyze_docstrings": config.analysis_settings.get("analyze_docstrings", True),
            "max_file_size_mb": config.analysis_settings.get("max_file_size_mb", 10),
            "max_line_count": config.analysis_settings.get("max_line_count", 50000),
            "timeout_seconds": config.analysis_settings.get("timeout_seconds", 30)
        },
        
        # File patterns
        "file_patterns": {
            "include": config.file_patterns.get("include", []),
            "exclude": config.file_patterns.get("exclude", [])
        },
        "ignore_patterns": config.ignore_patterns,
        
        # Monitoring configuration
        "monitoring": {
            "enabled": config.monitoring_config.get("enabled", True),
            "debounce_seconds": config.monitoring_config.get("debounce_seconds", 2.0),
            "watch_subdirectories": config.monitoring_config.get("watch_subdirectories", True),
            "max_file_size_mb": config.monitoring_config.get("max_file_size_mb", 10)
        },
        
        # Optimization settings
        "optimization": {
            "context_optimization_enabled": config.optimization_settings.get("context_optimization_enabled", True),
            "max_context_files": config.optimization_settings.get("max_context_files", 50),
            "relevance_threshold": config.optimization_settings.get("relevance_threshold", 0.7)
        },
        
        # Performance settings
        "performance": {
            "max_concurrent_analyses": config.performance_settings.get("max_concurrent_analyses", 4),
            "analysis_batch_size": config.performance_settings.get("analysis_batch_size", 50),
            "cache_enabled": config.performance_settings.get("cache_enabled", True),
            "batch_insert_size": config.performance_settings.get("batch_insert_size", 100)
        }
    }
    
    return project_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Project Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/my-project
  %(prog)s /path/to/my-project --output project-config.json
  %(prog)s . --verbose --export-summary
        """
    )
    
    parser.add_argument(
        "project_path",
        help="Path to the project directory to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for generated configuration (JSON format)"
    )
    
    parser.add_argument(
        "--export-summary",
        action="store_true",
        help="Export human-readable summary to [project-name]-analysis.txt"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--config",
        help="Custom detector configuration file (JSON)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "summary"],
        default="summary",
        help="Output format (default: summary)"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"❌ Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)
    
    if not project_path.is_dir():
        print(f"❌ Error: Project path is not a directory: {project_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load custom configuration if provided
    detector_config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    detector_config = json.load(f)
                if args.verbose:
                    print(f"📝 Loaded custom configuration from {config_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load configuration file: {e}", file=sys.stderr)
    
    # Initialize detector
    print(f"🔍 Analyzing project: {project_path}")
    print(f"⏱️  Starting intelligent detection...")
    
    try:
        detector = IntelligentProjectDetector(config=detector_config)
        start_time = time.time()
        
        # Perform detection
        result = detector.detect_project(project_path)
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Output results based on format
        if args.format == "json":
            # Export raw JSON result
            result_dict = result.__dict__.copy()
            
            # Convert complex objects to serializable format
            def make_serializable(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                else:
                    return str(obj)
            
            result_dict = make_serializable(result_dict)
            print(json.dumps(result_dict, indent=2, default=str))
            
        elif args.format == "summary":
            # Human-readable summary
            summary = format_detection_summary(result)
            print(summary)
        
        # Export configuration if requested
        if args.output:
            output_path = Path(args.output)
            project_config = generate_project_index_config(result)
            
            with open(output_path, 'w') as f:
                json.dump(project_config, f, indent=2, default=str)
            
            print(f"\n📄 Configuration exported to: {output_path}")
        
        # Export summary if requested
        if args.export_summary:
            summary_path = project_path.parent / f"{project_path.name}-analysis.txt"
            summary = format_detection_summary(result)
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print(f"📄 Analysis summary exported to: {summary_path}")
        
        # Show quick setup instructions
        print(f"\n🚀 QUICK SETUP INSTRUCTIONS")
        print(f"To enable Project Index for this project:")
        print(f"1. Copy the generated configuration to your project")
        print(f"2. Run: python enable_project_index.py {project_path}")
        print(f"3. Start the Project Index service")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())