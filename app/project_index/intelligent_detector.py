"""
Intelligent Project Detection System for LeanVibe Agent Hive 2.0

Analyzes any codebase and automatically generates optimal Project Index configurations.
Provides language detection, framework identification, project structure analysis,
dependency parsing, and intelligent configuration generation.
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import logging

import structlog

logger = structlog.get_logger()


class ProjectSize(Enum):
    """Project size categories based on complexity metrics."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class LanguageConfidence(Enum):
    """Confidence levels for language detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis."""
    language: str
    confidence: LanguageConfidence
    file_count: int
    total_lines: int
    primary_files: List[str]
    evidence: Dict[str, Any]


@dataclass
class FrameworkDetectionResult:
    """Result of framework detection analysis."""
    framework: str
    version: Optional[str]
    confidence: LanguageConfidence
    evidence_files: List[str]
    configuration_files: List[str]
    characteristics: Dict[str, Any]


@dataclass
class DependencyAnalysisResult:
    """Result of dependency analysis."""
    package_manager: str
    total_dependencies: int
    production_dependencies: int
    dev_dependencies: int
    dependency_files: List[str]
    major_dependencies: List[Dict[str, Any]]


@dataclass
class ProjectSizeAnalysis:
    """Analysis of project size and complexity."""
    size_category: ProjectSize
    file_count: int
    line_count: int
    directory_count: int
    complexity_score: float
    estimated_team_size: int
    development_stage: str


@dataclass
class ProjectStructureAnalysis:
    """Analysis of project organization and structure."""
    structure_type: str  # "monolithic", "modular", "microservice", "monorepo"
    entry_points: List[str]
    core_directories: List[str]
    test_directories: List[str]
    documentation_files: List[str]
    configuration_files: List[str]
    build_files: List[str]


@dataclass
class IntelligentProjectConfig:
    """Generated configuration for the project."""
    analysis_settings: Dict[str, Any]
    file_patterns: Dict[str, List[str]]
    ignore_patterns: List[str]
    optimization_settings: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    performance_settings: Dict[str, Any]


@dataclass
class ProjectDetectionResult:
    """Comprehensive project detection result."""
    project_path: str
    detection_timestamp: datetime
    analysis_duration: float
    
    # Core detection results
    primary_language: LanguageDetectionResult
    secondary_languages: List[LanguageDetectionResult]
    detected_frameworks: List[FrameworkDetectionResult]
    dependency_analysis: List[DependencyAnalysisResult]
    size_analysis: ProjectSizeAnalysis
    structure_analysis: ProjectStructureAnalysis
    
    # Generated configuration
    recommended_config: IntelligentProjectConfig
    
    # Metadata
    confidence_score: float
    warnings: List[str]
    recommendations: List[str]


class IntelligentProjectDetector:
    """
    Intelligent system for detecting project characteristics and generating
    optimal Project Index configurations.
    """
    
    # Language detection patterns and file extensions
    LANGUAGE_PATTERNS = {
        'python': {
            'extensions': ['.py', '.pyi', '.pyx', '.pyw'],
            'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', 'environment.yml'],
            'directories': ['__pycache__', '.venv', 'venv', 'env'],
            'patterns': [
                r'#!/usr/bin/env python',
                r'#!/usr/bin/python',
                r'# -\*- coding: utf-8 -\*-',
                r'from __future__ import',
                r'import \w+',
                r'def \w+\(',
                r'class \w+\(',
            ]
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.mjs', '.cjs'],
            'files': ['package.json', 'package-lock.json', 'yarn.lock', '.babelrc', 'webpack.config.js'],
            'directories': ['node_modules', 'dist', 'build'],
            'patterns': [
                r'#!/usr/bin/env node',
                r'require\(',
                r'import .* from',
                r'export (default|const|function)',
                r'function \w+\(',
                r'const \w+ =',
                r'let \w+ =',
                r'var \w+ =',
            ]
        },
        'typescript': {
            'extensions': ['.ts', '.tsx', '.d.ts'],
            'files': ['tsconfig.json', 'tslint.json', '.eslintrc.ts'],
            'directories': ['@types', 'types'],
            'patterns': [
                r'interface \w+',
                r'type \w+ =',
                r'enum \w+',
                r': \w+\[\]',
                r'<.*>',
                r'implements \w+',
            ]
        },
        'go': {
            'extensions': ['.go'],
            'files': ['go.mod', 'go.sum', 'Gopkg.toml', 'Gopkg.lock'],
            'directories': ['vendor'],
            'patterns': [
                r'package \w+',
                r'import \(',
                r'func \w+\(',
                r'type \w+ struct',
                r'var \w+ \w+',
                r'go \w+\(',
            ]
        },
        'rust': {
            'extensions': ['.rs'],
            'files': ['Cargo.toml', 'Cargo.lock'],
            'directories': ['target', 'src'],
            'patterns': [
                r'fn \w+\(',
                r'struct \w+',
                r'enum \w+',
                r'impl \w+',
                r'use \w+',
                r'mod \w+',
                r'pub fn',
            ]
        },
        'java': {
            'extensions': ['.java', '.class', '.jar'],
            'files': ['pom.xml', 'build.gradle', 'gradle.properties', 'settings.gradle'],
            'directories': ['src/main/java', 'src/test/java', 'target', 'build'],
            'patterns': [
                r'package \w+',
                r'import \w+',
                r'public class \w+',
                r'private \w+ \w+',
                r'public static void main',
                r'@\w+',
            ]
        },
        'csharp': {
            'extensions': ['.cs', '.csx', '.dll', '.exe'],
            'files': ['*.csproj', '*.sln', 'packages.config', 'nuget.config'],
            'directories': ['bin', 'obj', 'packages'],
            'patterns': [
                r'using \w+',
                r'namespace \w+',
                r'public class \w+',
                r'private \w+ \w+',
                r'\[.*\]',
                r'public static void Main',
            ]
        },
        'php': {
            'extensions': ['.php', '.phtml', '.php3', '.php4', '.php5', '.phps'],
            'files': ['composer.json', 'composer.lock', 'index.php'],
            'directories': ['vendor', 'public'],
            'patterns': [
                r'<\?php',
                r'namespace \w+',
                r'use \w+',
                r'class \w+',
                r'function \w+\(',
                r'\$\w+',
            ]
        }
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'python': {
            'django': {
                'files': ['manage.py', 'django.py', 'settings.py'],
                'directories': ['django_project', 'mysite'],
                'patterns': [r'from django', r'DJANGO_SETTINGS_MODULE', r'django.setup()'],
                'dependencies': ['Django', 'django']
            },
            'flask': {
                'files': ['app.py', 'application.py', 'run.py'],
                'patterns': [r'from flask import', r'Flask\(__name__\)', r'@app.route'],
                'dependencies': ['Flask', 'flask']
            },
            'fastapi': {
                'files': ['main.py', 'app.py'],
                'patterns': [r'from fastapi import', r'FastAPI\(', r'@app\.(get|post|put|delete)'],
                'dependencies': ['fastapi', 'FastAPI']
            },
            'poetry': {
                'files': ['pyproject.toml'],
                'patterns': [r'\[tool\.poetry\]', r'poetry.lock'],
                'dependencies': []
            }
        },
        'javascript': {
            'react': {
                'files': ['.babelrc', 'webpack.config.js'],
                'patterns': [r'import React', r'from [\'"]react[\'"]', r'ReactDOM.render', r'useState', r'useEffect'],
                'dependencies': ['react', 'react-dom', '@types/react']
            },
            'vue': {
                'files': ['vue.config.js'],
                'patterns': [r'import Vue', r'from [\'"]vue[\'"]', r'new Vue\(', r'<template>', r'<script>'],
                'dependencies': ['vue', '@vue/cli']
            },
            'angular': {
                'files': ['angular.json', '.angular-cli.json'],
                'patterns': [r'import.*@angular', r'@Component\(', r'@Injectable\(', r'@NgModule\('],
                'dependencies': ['@angular/core', '@angular/cli']
            },
            'express': {
                'files': ['server.js', 'app.js'],
                'patterns': [r'require\([\'"]express[\'"]\)', r'express\(\)', r'app\.(get|post|put|delete)'],
                'dependencies': ['express']
            },
            'next': {
                'files': ['next.config.js', 'pages'],
                'patterns': [r'import.*next', r'getStaticProps', r'getServerSideProps'],
                'dependencies': ['next', 'react']
            }
        },
        'typescript': {
            'nest': {
                'files': ['nest-cli.json'],
                'patterns': [r'import.*@nestjs', r'@Controller\(', r'@Injectable\(', r'@Module\('],
                'dependencies': ['@nestjs/core', '@nestjs/common']
            }
        },
        'go': {
            'gin': {
                'patterns': [r'gin\.Default\(\)', r'gin\.Engine', r'c \*gin\.Context'],
                'dependencies': ['github.com/gin-gonic/gin']
            },
            'echo': {
                'patterns': [r'echo\.New\(\)', r'echo\.Echo', r'c echo\.Context'],
                'dependencies': ['github.com/labstack/echo']
            }
        },
        'rust': {
            'axum': {
                'patterns': [r'use axum', r'axum::Router', r'axum::extract'],
                'dependencies': ['axum']
            },
            'rocket': {
                'patterns': [r'use rocket', r'#\[rocket::', r'rocket::launch'],
                'dependencies': ['rocket']
            }
        },
        'java': {
            'spring': {
                'files': ['application.properties', 'application.yml'],
                'patterns': [r'@SpringBootApplication', r'@RestController', r'@Service', r'@Repository'],
                'dependencies': ['spring-boot', 'springframework']
            }
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent project detector."""
        self.config = config or {}
        self.analysis_cache: Dict[str, Any] = {}
        
        # Analysis settings
        self.max_file_scan_depth = self.config.get('max_file_scan_depth', 10)
        self.max_files_to_analyze = self.config.get('max_files_to_analyze', 1000)
        self.sample_file_size_limit = self.config.get('sample_file_size_limit', 1024 * 1024)  # 1MB
        
        logger.info("Intelligent project detector initialized", config=self.config)
    
    def detect_project(self, project_path: Union[str, Path]) -> ProjectDetectionResult:
        """
        Perform comprehensive project detection and analysis.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Complete project detection result with configuration
        """
        start_time = time.time()
        project_path = Path(project_path).resolve()
        
        logger.info("Starting project detection", path=str(project_path))
        
        if not project_path.exists() or not project_path.is_dir():
            raise ValueError(f"Project path does not exist or is not a directory: {project_path}")
        
        warnings = []
        recommendations = []
        
        try:
            # 1. Scan project structure
            logger.info("Scanning project structure")
            structure_info = self._scan_project_structure(project_path)
            
            # 2. Detect languages
            logger.info("Detecting programming languages")
            language_results = self._detect_languages(project_path, structure_info)
            
            # 3. Identify frameworks
            logger.info("Identifying frameworks and tools")
            framework_results = self._detect_frameworks(project_path, language_results, structure_info)
            
            # 4. Analyze dependencies
            logger.info("Analyzing project dependencies")
            dependency_results = self._analyze_dependencies(project_path, language_results)
            
            # 5. Estimate project size and complexity
            logger.info("Estimating project size and complexity")
            size_analysis = self._analyze_project_size(project_path, structure_info, language_results)
            
            # 6. Analyze project structure
            logger.info("Analyzing project organization")
            structure_analysis = self._analyze_project_structure(project_path, structure_info, language_results)
            
            # 7. Generate intelligent configuration
            logger.info("Generating optimal configuration")
            config = self._generate_configuration(
                language_results, framework_results, dependency_results, 
                size_analysis, structure_analysis
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence_score(
                language_results, framework_results, dependency_results
            )
            
            # Get primary and secondary languages
            primary_language = max(language_results, key=lambda x: x.file_count) if language_results else None
            secondary_languages = [lang for lang in language_results if lang != primary_language]
            
            analysis_duration = time.time() - start_time
            
            result = ProjectDetectionResult(
                project_path=str(project_path),
                detection_timestamp=datetime.utcnow(),
                analysis_duration=analysis_duration,
                primary_language=primary_language,
                secondary_languages=secondary_languages,
                detected_frameworks=framework_results,
                dependency_analysis=dependency_results,
                size_analysis=size_analysis,
                structure_analysis=structure_analysis,
                recommended_config=config,
                confidence_score=confidence_score,
                warnings=warnings,
                recommendations=recommendations
            )
            
            logger.info("Project detection completed successfully",
                       duration=f"{analysis_duration:.2f}s",
                       primary_language=primary_language.language if primary_language else None,
                       frameworks=len(framework_results),
                       confidence=confidence_score)
            
            return result
            
        except Exception as e:
            logger.error("Project detection failed", error=str(e), path=str(project_path))
            raise RuntimeError(f"Project detection failed: {e}")
    
    def _scan_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Scan and catalog project directory structure."""
        structure_info = {
            'files': [],
            'directories': [],
            'file_extensions': {},
            'directory_names': set(),
            'total_files': 0,
            'total_size': 0,
            'max_depth': 0
        }
        
        try:
            for root, dirs, files in os.walk(project_path):
                current_path = Path(root)
                depth = len(current_path.relative_to(project_path).parts)
                structure_info['max_depth'] = max(structure_info['max_depth'], depth)
                
                # Stop if we've gone too deep
                if depth > self.max_file_scan_depth:
                    dirs.clear()  # Don't recurse further
                    continue
                
                # Process directories
                for dir_name in dirs:
                    dir_path = current_path / dir_name
                    structure_info['directories'].append(str(dir_path.relative_to(project_path)))
                    structure_info['directory_names'].add(dir_name)
                
                # Process files
                for file_name in files:
                    if structure_info['total_files'] >= self.max_files_to_analyze:
                        break
                    
                    file_path = current_path / file_name
                    try:
                        file_stat = file_path.stat()
                        relative_path = str(file_path.relative_to(project_path))
                        
                        structure_info['files'].append({
                            'path': relative_path,
                            'name': file_name,
                            'size': file_stat.st_size,
                            'extension': file_path.suffix.lower(),
                            'depth': depth
                        })
                        
                        # Track extensions
                        ext = file_path.suffix.lower()
                        if ext:
                            structure_info['file_extensions'][ext] = structure_info['file_extensions'].get(ext, 0) + 1
                        
                        structure_info['total_files'] += 1
                        structure_info['total_size'] += file_stat.st_size
                        
                    except (OSError, PermissionError):
                        continue
                
                if structure_info['total_files'] >= self.max_files_to_analyze:
                    break
        
        except Exception as e:
            logger.warning("Error scanning project structure", error=str(e))
        
        return structure_info
    
    def _detect_languages(self, project_path: Path, structure_info: Dict[str, Any]) -> List[LanguageDetectionResult]:
        """Detect programming languages used in the project."""
        language_stats = {}
        
        # Initialize language tracking
        for lang in self.LANGUAGE_PATTERNS:
            language_stats[lang] = {
                'file_count': 0,
                'total_lines': 0,
                'evidence_score': 0,
                'primary_files': [],
                'evidence': {}
            }
        
        # 1. File extension analysis
        for ext, count in structure_info['file_extensions'].items():
            for lang, patterns in self.LANGUAGE_PATTERNS.items():
                if ext in patterns['extensions']:
                    language_stats[lang]['file_count'] += count
                    language_stats[lang]['evidence_score'] += count * 10
                    language_stats[lang]['evidence']['extensions'] = language_stats[lang]['evidence'].get('extensions', [])
                    language_stats[lang]['evidence']['extensions'].append(ext)
        
        # 2. Special file detection
        for file_info in structure_info['files']:
            file_name = file_info['name'].lower()
            for lang, patterns in self.LANGUAGE_PATTERNS.items():
                for special_file in patterns['files']:
                    if self._matches_pattern(file_name, special_file):
                        language_stats[lang]['evidence_score'] += 50
                        language_stats[lang]['evidence']['special_files'] = language_stats[lang]['evidence'].get('special_files', [])
                        language_stats[lang]['evidence']['special_files'].append(file_info['name'])
        
        # 3. Directory structure analysis
        for dir_name in structure_info['directory_names']:
            for lang, patterns in self.LANGUAGE_PATTERNS.items():
                if dir_name in patterns['directories']:
                    language_stats[lang]['evidence_score'] += 25
                    language_stats[lang]['evidence']['directories'] = language_stats[lang]['evidence'].get('directories', [])
                    language_stats[lang]['evidence']['directories'].append(dir_name)
        
        # 4. Content analysis (sample files)
        self._analyze_file_contents(project_path, structure_info, language_stats)
        
        # 5. Convert to results
        results = []
        for lang, stats in language_stats.items():
            if stats['evidence_score'] > 0:
                confidence = self._determine_language_confidence(stats)
                
                result = LanguageDetectionResult(
                    language=lang,
                    confidence=confidence,
                    file_count=stats['file_count'],
                    total_lines=stats['total_lines'],
                    primary_files=stats['primary_files'][:10],  # Top 10 files
                    evidence=stats['evidence']
                )
                results.append(result)
        
        # Sort by confidence and file count
        results.sort(key=lambda x: (x.confidence.value, x.file_count), reverse=True)
        
        return results
    
    def _analyze_file_contents(self, project_path: Path, structure_info: Dict[str, Any], language_stats: Dict[str, Any]):
        """Analyze file contents to detect language patterns."""
        files_analyzed = 0
        max_content_analysis = 50  # Limit content analysis
        
        for file_info in structure_info['files']:
            if files_analyzed >= max_content_analysis:
                break
            
            file_path = project_path / file_info['path']
            
            # Skip large files
            if file_info['size'] > self.sample_file_size_limit:
                continue
            
            # Skip binary files based on extension
            if file_info['extension'] in ['.jpg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz']:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # Analyze content for language patterns
                    for lang, patterns in self.LANGUAGE_PATTERNS.items():
                        matches = 0
                        for pattern in patterns['patterns']:
                            if re.search(pattern, content, re.MULTILINE):
                                matches += 1
                        
                        if matches > 0:
                            language_stats[lang]['evidence_score'] += matches * 5
                            language_stats[lang]['total_lines'] += len(lines)
                            language_stats[lang]['primary_files'].append(file_info['path'])
                            
                            if 'content_patterns' not in language_stats[lang]['evidence']:
                                language_stats[lang]['evidence']['content_patterns'] = 0
                            language_stats[lang]['evidence']['content_patterns'] += matches
                
                files_analyzed += 1
                
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
    
    def _detect_frameworks(self, project_path: Path, language_results: List[LanguageDetectionResult], structure_info: Dict[str, Any]) -> List[FrameworkDetectionResult]:
        """Detect frameworks and tools used in the project."""
        framework_results = []
        
        # Get detected languages
        detected_languages = {result.language for result in language_results}
        
        for lang in detected_languages:
            if lang not in self.FRAMEWORK_PATTERNS:
                continue
            
            frameworks = self.FRAMEWORK_PATTERNS[lang]
            
            for framework_name, framework_config in frameworks.items():
                evidence_score = 0
                evidence_files = []
                config_files = []
                characteristics = {}
                
                # Check for framework-specific files
                if 'files' in framework_config:
                    for pattern in framework_config['files']:
                        for file_info in structure_info['files']:
                            if self._matches_pattern(file_info['name'], pattern):
                                evidence_score += 30
                                evidence_files.append(file_info['path'])
                                config_files.append(file_info['path'])
                
                # Check for framework patterns in content
                if 'patterns' in framework_config:
                    content_matches = self._scan_for_patterns(
                        project_path, framework_config['patterns'], structure_info
                    )
                    evidence_score += content_matches * 10
                    characteristics['pattern_matches'] = content_matches
                
                # Check dependencies (if dependency analysis available)
                if 'dependencies' in framework_config:
                    dep_matches = self._check_dependencies(
                        project_path, framework_config['dependencies']
                    )
                    evidence_score += dep_matches * 20
                    characteristics['dependency_matches'] = dep_matches
                
                # Create result if evidence found
                if evidence_score > 0:
                    confidence = self._determine_framework_confidence(evidence_score)
                    
                    result = FrameworkDetectionResult(
                        framework=framework_name,
                        version=None,  # Version detection would require more sophisticated analysis
                        confidence=confidence,
                        evidence_files=evidence_files,
                        configuration_files=config_files,
                        characteristics=characteristics
                    )
                    framework_results.append(result)
        
        # Sort by confidence
        framework_results.sort(key=lambda x: x.confidence.value, reverse=True)
        
        return framework_results
    
    def _analyze_dependencies(self, project_path: Path, language_results: List[LanguageDetectionResult]) -> List[DependencyAnalysisResult]:
        """Analyze project dependencies and package management."""
        dependency_results = []
        
        # Define dependency file patterns for each language/package manager
        dependency_patterns = {
            'python': [
                {'file': 'requirements.txt', 'manager': 'pip'},
                {'file': 'pyproject.toml', 'manager': 'poetry'},
                {'file': 'Pipfile', 'manager': 'pipenv'},
                {'file': 'environment.yml', 'manager': 'conda'},
                {'file': 'setup.py', 'manager': 'setuptools'}
            ],
            'javascript': [
                {'file': 'package.json', 'manager': 'npm'},
                {'file': 'yarn.lock', 'manager': 'yarn'},
                {'file': 'pnpm-lock.yaml', 'manager': 'pnpm'}
            ],
            'go': [
                {'file': 'go.mod', 'manager': 'go-modules'},
                {'file': 'Gopkg.toml', 'manager': 'dep'}
            ],
            'rust': [
                {'file': 'Cargo.toml', 'manager': 'cargo'}
            ],
            'java': [
                {'file': 'pom.xml', 'manager': 'maven'},
                {'file': 'build.gradle', 'manager': 'gradle'}
            ],
            'csharp': [
                {'file': '*.csproj', 'manager': 'nuget'},
                {'file': 'packages.config', 'manager': 'nuget'}
            ],
            'php': [
                {'file': 'composer.json', 'manager': 'composer'}
            ]
        }
        
        detected_languages = {result.language for result in language_results}
        
        for lang in detected_languages:
            if lang not in dependency_patterns:
                continue
            
            for dep_config in dependency_patterns[lang]:
                dep_file_pattern = dep_config['file']
                manager = dep_config['manager']
                
                # Find dependency files
                dependency_files = []
                for file_path in project_path.rglob(dep_file_pattern):
                    if file_path.is_file():
                        dependency_files.append(str(file_path.relative_to(project_path)))
                
                if dependency_files:
                    # Analyze first dependency file found
                    first_dep_file = project_path / dependency_files[0]
                    analysis = self._analyze_dependency_file(first_dep_file, manager)
                    
                    if analysis:
                        result = DependencyAnalysisResult(
                            package_manager=manager,
                            total_dependencies=analysis['total'],
                            production_dependencies=analysis['production'],
                            dev_dependencies=analysis['dev'],
                            dependency_files=dependency_files,
                            major_dependencies=analysis['major_deps']
                        )
                        dependency_results.append(result)
        
        return dependency_results
    
    def _analyze_project_size(self, project_path: Path, structure_info: Dict[str, Any], language_results: List[LanguageDetectionResult]) -> ProjectSizeAnalysis:
        """Analyze project size and complexity."""
        file_count = structure_info['total_files']
        total_size = structure_info['total_size']
        directory_count = len(structure_info['directories'])
        
        # Calculate total lines of code
        total_lines = sum(lang.total_lines for lang in language_results)
        
        # Calculate complexity score based on multiple factors
        complexity_factors = {
            'file_count': min(file_count / 1000, 1.0),  # Normalize to 0-1
            'directory_depth': min(structure_info['max_depth'] / 10, 1.0),
            'language_diversity': min(len(language_results) / 5, 1.0),
            'lines_of_code': min(total_lines / 100000, 1.0),
            'directory_count': min(directory_count / 100, 1.0)
        }
        
        complexity_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        # Determine size category
        if file_count < 50 and total_lines < 5000:
            size_category = ProjectSize.SMALL
            estimated_team_size = 1
            development_stage = "prototype"
        elif file_count < 200 and total_lines < 20000:
            size_category = ProjectSize.MEDIUM
            estimated_team_size = 2-3
            development_stage = "development"
        elif file_count < 1000 and total_lines < 100000:
            size_category = ProjectSize.LARGE
            estimated_team_size = 5-10
            development_stage = "production"
        else:
            size_category = ProjectSize.ENTERPRISE
            estimated_team_size = 10
            development_stage = "enterprise"
        
        return ProjectSizeAnalysis(
            size_category=size_category,
            file_count=file_count,
            line_count=total_lines,
            directory_count=directory_count,
            complexity_score=complexity_score,
            estimated_team_size=estimated_team_size,
            development_stage=development_stage
        )
    
    def _analyze_project_structure(self, project_path: Path, structure_info: Dict[str, Any], language_results: List[LanguageDetectionResult]) -> ProjectStructureAnalysis:
        """Analyze project organization and architecture."""
        
        # Detect structure type
        structure_type = self._detect_structure_type(structure_info)
        
        # Find entry points
        entry_points = self._find_entry_points(structure_info, language_results)
        
        # Categorize directories
        core_directories = []
        test_directories = []
        doc_directories = []
        
        for dir_path in structure_info['directories']:
            dir_name = Path(dir_path).name.lower()
            
            if any(test_word in dir_name for test_word in ['test', 'tests', 'spec', 'specs']):
                test_directories.append(dir_path)
            elif any(doc_word in dir_name for doc_word in ['doc', 'docs', 'documentation']):
                doc_directories.append(dir_path)
            elif any(core_word in dir_name for core_word in ['src', 'lib', 'app', 'core', 'main']):
                core_directories.append(dir_path)
        
        # Find important files
        documentation_files = []
        configuration_files = []
        build_files = []
        
        for file_info in structure_info['files']:
            file_name = file_info['name'].lower()
            
            if any(doc_word in file_name for doc_word in ['readme', 'changelog', 'license', 'contributing']):
                documentation_files.append(file_info['path'])
            elif any(config_word in file_name for config_word in ['config', 'settings', '.env', '.yaml', '.toml', '.json']):
                configuration_files.append(file_info['path'])
            elif any(build_word in file_name for build_word in ['makefile', 'dockerfile', 'build', 'webpack', 'rollup']):
                build_files.append(file_info['path'])
        
        return ProjectStructureAnalysis(
            structure_type=structure_type,
            entry_points=entry_points,
            core_directories=core_directories,
            test_directories=test_directories,
            documentation_files=documentation_files,
            configuration_files=configuration_files,
            build_files=build_files
        )
    
    def _generate_configuration(self, language_results: List[LanguageDetectionResult], 
                              framework_results: List[FrameworkDetectionResult],
                              dependency_results: List[DependencyAnalysisResult],
                              size_analysis: ProjectSizeAnalysis,
                              structure_analysis: ProjectStructureAnalysis) -> IntelligentProjectConfig:
        """Generate intelligent Project Index configuration."""
        
        # Base analysis settings
        analysis_settings = {
            'parse_ast': True,
            'extract_dependencies': True,
            'calculate_complexity': True,
            'analyze_docstrings': True,
            'max_file_size_mb': 10,
            'max_line_count': 50000,
            'timeout_seconds': 30
        }
        
        # Adjust based on project size
        if size_analysis.size_category == ProjectSize.ENTERPRISE:
            analysis_settings.update({
                'max_file_size_mb': 20,
                'max_line_count': 100000,
                'timeout_seconds': 60
            })
        elif size_analysis.size_category == ProjectSize.SMALL:
            analysis_settings.update({
                'max_file_size_mb': 5,
                'timeout_seconds': 15
            })
        
        # Generate file patterns based on detected languages
        include_patterns = []
        for lang_result in language_results:
            if lang_result.confidence in [LanguageConfidence.HIGH, LanguageConfidence.VERY_HIGH]:
                lang_patterns = self.LANGUAGE_PATTERNS.get(lang_result.language, {})
                extensions = lang_patterns.get('extensions', [])
                for ext in extensions:
                    include_patterns.append(f"**/*{ext}")
        
        # Default patterns if none detected
        if not include_patterns:
            include_patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.json']
        
        file_patterns = {
            'include': include_patterns,
            'exclude': []
        }
        
        # Generate ignore patterns
        ignore_patterns = [
            '**/__pycache__/**',
            '**/.git/**',
            '**/node_modules/**',
            '**/.venv/**',
            '**/venv/**',
            '**/env/**',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.so',
            '**/*.dylib',
            '**/*.dll',
            '**/target/**',  # Rust
            '**/bin/**',     # Various
            '**/obj/**',     # C#
            '**/vendor/**',  # Go, PHP
            '**/build/**',
            '**/dist/**'
        ]
        
        # Add language-specific ignores
        detected_languages = {result.language for result in language_results}
        if 'java' in detected_languages:
            ignore_patterns.extend(['**/target/**', '**/*.class'])
        if 'csharp' in detected_languages:
            ignore_patterns.extend(['**/bin/**', '**/obj/**'])
        if 'go' in detected_languages:
            ignore_patterns.extend(['**/vendor/**'])
        if 'rust' in detected_languages:
            ignore_patterns.extend(['**/target/**'])
        
        # Optimization settings based on project characteristics
        optimization_settings = {
            'context_optimization_enabled': True,
            'max_context_files': 50,
            'relevance_threshold': 0.7
        }
        
        if size_analysis.size_category == ProjectSize.ENTERPRISE:
            optimization_settings.update({
                'max_context_files': 100,
                'relevance_threshold': 0.8
            })
        elif size_analysis.size_category == ProjectSize.SMALL:
            optimization_settings.update({
                'max_context_files': 25,
                'relevance_threshold': 0.6
            })
        
        # Monitoring configuration
        monitoring_config = {
            'enabled': True,
            'debounce_seconds': 2.0,
            'watch_subdirectories': True,
            'max_file_size_mb': analysis_settings['max_file_size_mb']
        }
        
        # Performance settings based on project size
        performance_settings = {
            'max_concurrent_analyses': 4,
            'analysis_batch_size': 50,
            'cache_enabled': True,
            'batch_insert_size': 100
        }
        
        if size_analysis.size_category == ProjectSize.ENTERPRISE:
            performance_settings.update({
                'max_concurrent_analyses': 8,
                'analysis_batch_size': 100,
                'batch_insert_size': 200
            })
        elif size_analysis.size_category == ProjectSize.SMALL:
            performance_settings.update({
                'max_concurrent_analyses': 2,
                'analysis_batch_size': 25,
                'batch_insert_size': 50
            })
        
        return IntelligentProjectConfig(
            analysis_settings=analysis_settings,
            file_patterns=file_patterns,
            ignore_patterns=ignore_patterns,
            optimization_settings=optimization_settings,
            monitoring_config=monitoring_config,
            performance_settings=performance_settings
        )
    
    # Helper methods
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a pattern (supports wildcards)."""
        if '*' in pattern:
            # Convert shell-style wildcards to regex
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            return bool(re.match(regex_pattern, text, re.IGNORECASE))
        else:
            return text.lower() == pattern.lower()
    
    def _scan_for_patterns(self, project_path: Path, patterns: List[str], structure_info: Dict[str, Any]) -> int:
        """Scan project files for specific patterns."""
        matches = 0
        files_scanned = 0
        max_scan = 20  # Limit pattern scanning
        
        for file_info in structure_info['files']:
            if files_scanned >= max_scan:
                break
            
            file_path = project_path / file_info['path']
            
            # Skip large files and binary files
            if file_info['size'] > self.sample_file_size_limit:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern in patterns:
                        if re.search(pattern, content):
                            matches += 1
                
                files_scanned += 1
                
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
        
        return matches
    
    def _check_dependencies(self, project_path: Path, dependencies: List[str]) -> int:
        """Check for specific dependencies in dependency files."""
        matches = 0
        
        # Check common dependency files
        dep_files = [
            'requirements.txt', 'package.json', 'go.mod', 'Cargo.toml',
            'pom.xml', 'build.gradle', 'composer.json'
        ]
        
        for dep_file in dep_files:
            file_path = project_path / dep_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for dep in dependencies:
                        if dep.lower() in content.lower():
                            matches += 1
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue
        
        return matches
    
    def _analyze_dependency_file(self, file_path: Path, manager: str) -> Optional[Dict[str, Any]]:
        """Analyze a specific dependency file."""
        try:
            if manager == 'npm' and file_path.name == 'package.json':
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                deps = data.get('dependencies', {})
                dev_deps = data.get('devDependencies', {})
                
                major_deps = []
                for name, version in list(deps.items())[:10]:  # Top 10
                    major_deps.append({'name': name, 'version': version, 'type': 'production'})
                
                return {
                    'total': len(deps) + len(dev_deps),
                    'production': len(deps),
                    'dev': len(dev_deps),
                    'major_deps': major_deps
                }
            
            elif manager == 'pip' and file_path.name == 'requirements.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                major_deps = []
                for line in lines[:10]:  # Top 10
                    parts = re.split(r'[>=<]', line)
                    name = parts[0].strip()
                    version = parts[1] if len(parts) > 1 else 'latest'
                    major_deps.append({'name': name, 'version': version, 'type': 'production'})
                
                return {
                    'total': len(lines),
                    'production': len(lines),
                    'dev': 0,
                    'major_deps': major_deps
                }
            
            # Add more parsers for other dependency managers as needed
            
        except Exception as e:
            logger.debug("Failed to parse dependency file", file=str(file_path), error=str(e))
        
        return None
    
    def _determine_language_confidence(self, stats: Dict[str, Any]) -> LanguageConfidence:
        """Determine confidence level for language detection."""
        score = stats['evidence_score']
        file_count = stats['file_count']
        
        if score >= 100 and file_count >= 10:
            return LanguageConfidence.VERY_HIGH
        elif score >= 50 and file_count >= 5:
            return LanguageConfidence.HIGH
        elif score >= 20 and file_count >= 2:
            return LanguageConfidence.MEDIUM
        else:
            return LanguageConfidence.LOW
    
    def _determine_framework_confidence(self, evidence_score: int) -> LanguageConfidence:
        """Determine confidence level for framework detection."""
        if evidence_score >= 80:
            return LanguageConfidence.VERY_HIGH
        elif evidence_score >= 50:
            return LanguageConfidence.HIGH
        elif evidence_score >= 25:
            return LanguageConfidence.MEDIUM
        else:
            return LanguageConfidence.LOW
    
    def _calculate_confidence_score(self, language_results: List[LanguageDetectionResult],
                                  framework_results: List[FrameworkDetectionResult],
                                  dependency_results: List[DependencyAnalysisResult]) -> float:
        """Calculate overall confidence score for the detection."""
        scores = []
        
        # Language confidence
        if language_results:
            lang_confidence = max(result.confidence.value for result in language_results[:3])
            confidence_mapping = {
                'very_high': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
            scores.append(confidence_mapping.get(lang_confidence, 0.5))
        
        # Framework confidence
        if framework_results:
            framework_confidence = max(result.confidence.value for result in framework_results[:2])
            scores.append(confidence_mapping.get(framework_confidence, 0.5))
        
        # Dependency analysis confidence
        if dependency_results:
            scores.append(0.8)  # High confidence if dependencies found
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _detect_structure_type(self, structure_info: Dict[str, Any]) -> str:
        """Detect project structure type."""
        directories = structure_info['directory_names']
        
        # Check for microservice indicators
        if any(name in directories for name in ['services', 'microservices', 'apps']):
            return "microservice"
        
        # Check for monorepo indicators
        if any(name in directories for name in ['packages', 'libs', 'modules']) and len(structure_info['directories']) > 20:
            return "monorepo"
        
        # Check for modular structure
        if any(name in directories for name in ['src', 'lib', 'components', 'modules']):
            return "modular"
        
        return "monolithic"
    
    def _find_entry_points(self, structure_info: Dict[str, Any], language_results: List[LanguageDetectionResult]) -> List[str]:
        """Find likely entry points for the application."""
        entry_points = []
        
        # Common entry point file names
        entry_patterns = [
            'main.py', 'app.py', 'server.py', 'run.py', '__main__.py',
            'index.js', 'app.js', 'server.js', 'main.js',
            'main.go', 'server.go',
            'main.rs', 'lib.rs',
            'Main.java', 'Application.java',
            'Program.cs', 'Startup.cs',
            'index.php', 'app.php'
        ]
        
        for file_info in structure_info['files']:
            if file_info['name'] in entry_patterns:
                entry_points.append(file_info['path'])
        
        return entry_points
    
    def export_to_json(self, result: ProjectDetectionResult, output_path: Union[str, Path]) -> None:
        """Export detection result to JSON file."""
        output_path = Path(output_path)
        
        # Convert result to dictionary with proper serialization
        result_dict = asdict(result)
        
        # Convert enums to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        result_dict = convert_enums(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info("Detection result exported", output_file=str(output_path))