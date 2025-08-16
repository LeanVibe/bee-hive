#!/usr/bin/env python3
"""
Advanced Project Structure Analyzer
===================================

Comprehensive analysis of project organization, architecture patterns,
and code structure to provide intelligent insights for Project Index optimization.

Features:
- Architecture pattern detection (MVC, Microservices, Monorepo, etc.)
- Code organization analysis
- Entry point detection
- Test structure analysis
- Documentation assessment
- Configuration file identification
- Build system detection
- Deployment pipeline analysis

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Common architecture patterns."""
    MONOLITHIC = "monolithic"
    MODULAR = "modular"
    MICROSERVICES = "microservices"
    MONOREPO = "monorepo"
    LAYERED = "layered"
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    HEXAGONAL = "hexagonal"
    ONION = "onion"
    CLEAN = "clean"
    SERVERLESS = "serverless"


class ProjectType(Enum):
    """Project type classifications."""
    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    DESKTOP_APPLICATION = "desktop_application"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    CLI_TOOL = "cli_tool"
    API_SERVICE = "api_service"
    MICROSERVICE = "microservice"
    DATA_PIPELINE = "data_pipeline"
    MACHINE_LEARNING = "machine_learning"
    GAME = "game"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"


class TestingStrategy(Enum):
    """Testing approach classifications."""
    UNIT_ONLY = "unit_only"
    INTEGRATION_FOCUSED = "integration_focused"
    E2E_FOCUSED = "e2e_focused"
    COMPREHENSIVE = "comprehensive"
    MINIMAL = "minimal"
    NONE = "none"


@dataclass
class DirectoryClassification:
    """Classification of a directory's purpose."""
    path: str
    name: str
    classification: str  # 'source', 'test', 'docs', 'config', 'build', 'assets', 'vendor', 'other'
    confidence: float
    sub_classifications: List[str]
    file_count: int
    important_files: List[str]


@dataclass
class EntryPoint:
    """Represents an application entry point."""
    file_path: str
    entry_type: str  # 'main', 'server', 'cli', 'web', 'test', 'build'
    confidence: float
    language: str
    framework: Optional[str] = None
    purpose: Optional[str] = None


@dataclass
class CodeOrganization:
    """Analysis of code organization patterns."""
    structure_depth: int
    namespace_strategy: str  # 'flat', 'hierarchical', 'domain-based', 'feature-based'
    separation_of_concerns: float  # 0-1 score
    modularity_score: float  # 0-1 score
    coupling_analysis: Dict[str, Any]
    common_patterns: List[str]


@dataclass
class TestingAnalysis:
    """Analysis of testing structure and strategy."""
    testing_strategy: TestingStrategy
    test_directories: List[str]
    test_files_count: int
    test_frameworks: List[str]
    coverage_indicators: List[str]
    test_to_code_ratio: float
    testing_patterns: List[str]


@dataclass
class DocumentationAnalysis:
    """Analysis of project documentation."""
    readme_files: List[str]
    documentation_directories: List[str]
    api_documentation: List[str]
    inline_documentation_score: float  # 0-1 based on docstring/comment analysis
    changelog_files: List[str]
    license_files: List[str]
    contributing_guidelines: List[str]
    documentation_completeness: float  # 0-1 score


@dataclass
class BuildSystemAnalysis:
    """Analysis of build and deployment systems."""
    build_tools: List[str]
    build_files: List[str]
    ci_cd_files: List[str]
    deployment_configs: List[str]
    containerization: List[str]  # Docker, etc.
    orchestration: List[str]  # Kubernetes, etc.
    package_configs: List[str]


@dataclass
class ProjectStructureAnalysis:
    """Comprehensive project structure analysis result."""
    project_path: str
    architecture_pattern: ArchitecturePattern
    project_type: ProjectType
    confidence_scores: Dict[str, float]
    
    # Detailed analyses
    directory_classifications: List[DirectoryClassification]
    entry_points: List[EntryPoint]
    code_organization: CodeOrganization
    testing_analysis: TestingAnalysis
    documentation_analysis: DocumentationAnalysis
    build_system_analysis: BuildSystemAnalysis
    
    # Summary statistics
    total_directories: int
    total_files: int
    code_files: int
    test_files: int
    config_files: int
    
    # Recommendations
    structure_recommendations: List[str]
    improvement_suggestions: List[str]


class AdvancedStructureAnalyzer:
    """
    Advanced analyzer for project structure and architecture patterns.
    """
    
    # Directory classification patterns
    DIRECTORY_PATTERNS = {
        'source': {
            'patterns': ['src', 'source', 'app', 'lib', 'library', 'core', 'main', 'code'],
            'weight': 10
        },
        'test': {
            'patterns': ['test', 'tests', 'spec', 'specs', '__tests__', 'testing', 'e2e', 'integration'],
            'weight': 10
        },
        'docs': {
            'patterns': ['doc', 'docs', 'documentation', 'manual', 'guide', 'wiki'],
            'weight': 8
        },
        'config': {
            'patterns': ['config', 'configuration', 'settings', 'conf', 'cfg', '.config'],
            'weight': 7
        },
        'build': {
            'patterns': ['build', 'dist', 'target', 'out', 'output', 'bin', 'release'],
            'weight': 6
        },
        'assets': {
            'patterns': ['assets', 'static', 'public', 'resources', 'res', 'media', 'images'],
            'weight': 5
        },
        'vendor': {
            'patterns': ['vendor', 'node_modules', 'third_party', 'external', 'lib', 'libs'],
            'weight': 4
        }
    }
    
    # Entry point patterns by language
    ENTRY_POINT_PATTERNS = {
        'python': {
            'main': ['main.py', '__main__.py', 'app.py', 'run.py', 'start.py'],
            'server': ['server.py', 'wsgi.py', 'asgi.py', 'manage.py'],
            'cli': ['cli.py', 'console.py', 'command.py'],
            'test': ['test_*.py', '*_test.py', 'tests.py']
        },
        'javascript': {
            'main': ['index.js', 'main.js', 'app.js', 'start.js'],
            'server': ['server.js', 'app.js', 'index.js'],
            'web': ['index.html', 'main.js', 'app.js'],
            'test': ['*.test.js', '*.spec.js', 'test.js']
        },
        'typescript': {
            'main': ['index.ts', 'main.ts', 'app.ts', 'start.ts'],
            'server': ['server.ts', 'app.ts', 'index.ts'],
            'web': ['index.html', 'main.ts', 'app.ts'],
            'test': ['*.test.ts', '*.spec.ts', 'test.ts']
        },
        'java': {
            'main': ['Main.java', 'Application.java', 'App.java'],
            'server': ['Server.java', 'Application.java'],
            'test': ['*Test.java', '*Tests.java']
        },
        'go': {
            'main': ['main.go', 'app.go'],
            'server': ['server.go', 'main.go'],
            'cli': ['cmd.go', 'cli.go'],
            'test': ['*_test.go']
        },
        'rust': {
            'main': ['main.rs', 'lib.rs'],
            'server': ['server.rs', 'main.rs'],
            'cli': ['cli.rs', 'main.rs'],
            'test': ['*_test.rs', 'tests.rs']
        }
    }
    
    # Architecture pattern indicators
    ARCHITECTURE_PATTERNS = {
        ArchitecturePattern.MICROSERVICES: {
            'indicators': ['services/', 'microservices/', 'api/', 'service-', 'docker-compose', 'kubernetes/'],
            'files': ['docker-compose.yml', 'k8s/', 'helm/', 'service.yml'],
            'weight': 8
        },
        ArchitecturePattern.MONOREPO: {
            'indicators': ['packages/', 'libs/', 'apps/', 'modules/', 'workspaces/'],
            'files': ['lerna.json', 'nx.json', 'rush.json', 'workspace.json'],
            'weight': 9
        },
        ArchitecturePattern.MVC: {
            'indicators': ['models/', 'views/', 'controllers/', 'mvc/'],
            'files': ['routes.py', 'urls.py', 'router.js'],
            'weight': 7
        },
        ArchitecturePattern.LAYERED: {
            'indicators': ['layers/', 'presentation/', 'business/', 'data/', 'domain/'],
            'files': ['repository.py', 'service.py', 'dto.py'],
            'weight': 6
        },
        ArchitecturePattern.CLEAN: {
            'indicators': ['domain/', 'application/', 'infrastructure/', 'interfaces/'],
            'files': ['use_case.py', 'entity.py', 'repository.py'],
            'weight': 7
        },
        ArchitecturePattern.SERVERLESS: {
            'indicators': ['functions/', 'lambda/', 'serverless/'],
            'files': ['serverless.yml', 'function.json', 'template.yaml'],
            'weight': 8
        }
    }
    
    # Project type indicators
    PROJECT_TYPE_INDICATORS = {
        ProjectType.WEB_APPLICATION: {
            'files': ['index.html', 'package.json', 'webpack.config.js', 'next.config.js'],
            'directories': ['public/', 'static/', 'assets/', 'components/'],
            'weight': 8
        },
        ProjectType.API_SERVICE: {
            'files': ['api.py', 'server.py', 'routes.py', 'swagger.yml'],
            'directories': ['api/', 'routes/', 'endpoints/'],
            'weight': 8
        },
        ProjectType.LIBRARY: {
            'files': ['setup.py', 'pyproject.toml', 'package.json', '__init__.py'],
            'directories': ['lib/', 'src/'],
            'indicators': ['no main entry', 'distribution files'],
            'weight': 7
        },
        ProjectType.CLI_TOOL: {
            'files': ['cli.py', 'main.py', 'console.py', 'bin/'],
            'directories': ['commands/', 'cli/'],
            'weight': 7
        },
        ProjectType.MOBILE_APPLICATION: {
            'files': ['App.js', 'App.tsx', 'MainActivity.java', 'Info.plist'],
            'directories': ['ios/', 'android/', 'mobile/'],
            'weight': 9
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced structure analyzer."""
        self.config = config or {}
        self.max_depth = self.config.get('max_depth', 10)
        self.max_files_to_analyze = self.config.get('max_files_to_analyze', 5000)
        
        logger.info("Advanced structure analyzer initialized")
    
    def analyze_project_structure(self, project_path: Union[str, Path]) -> ProjectStructureAnalysis:
        """
        Perform comprehensive project structure analysis.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Complete project structure analysis
        """
        project_path = Path(project_path).resolve()
        
        logger.info("Starting advanced structure analysis", project_path=str(project_path))
        
        # Scan project structure
        structure_scan = self._scan_project_structure(project_path)
        
        # Classify directories
        directory_classifications = self._classify_directories(structure_scan)
        
        # Detect entry points
        entry_points = self._detect_entry_points(project_path, structure_scan)
        
        # Analyze code organization
        code_organization = self._analyze_code_organization(project_path, structure_scan)
        
        # Analyze testing structure
        testing_analysis = self._analyze_testing_structure(project_path, structure_scan)
        
        # Analyze documentation
        documentation_analysis = self._analyze_documentation(project_path, structure_scan)
        
        # Analyze build systems
        build_system_analysis = self._analyze_build_systems(project_path, structure_scan)
        
        # Detect architecture pattern
        architecture_pattern = self._detect_architecture_pattern(project_path, structure_scan, directory_classifications)
        
        # Detect project type
        project_type = self._detect_project_type(project_path, structure_scan, entry_points)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            architecture_pattern, project_type, directory_classifications, entry_points
        )
        
        # Generate recommendations
        structure_recommendations, improvement_suggestions = self._generate_recommendations(
            architecture_pattern, project_type, code_organization, testing_analysis
        )
        
        # Calculate summary statistics
        total_directories = len(structure_scan['directories'])
        total_files = len(structure_scan['files'])
        code_files = len([f for f in structure_scan['files'] if self._is_code_file(f['name'])])
        test_files = len([f for f in structure_scan['files'] if self._is_test_file(f['path'])])
        config_files = len([f for f in structure_scan['files'] if self._is_config_file(f['name'])])
        
        return ProjectStructureAnalysis(
            project_path=str(project_path),
            architecture_pattern=architecture_pattern,
            project_type=project_type,
            confidence_scores=confidence_scores,
            directory_classifications=directory_classifications,
            entry_points=entry_points,
            code_organization=code_organization,
            testing_analysis=testing_analysis,
            documentation_analysis=documentation_analysis,
            build_system_analysis=build_system_analysis,
            total_directories=total_directories,
            total_files=total_files,
            code_files=code_files,
            test_files=test_files,
            config_files=config_files,
            structure_recommendations=structure_recommendations,
            improvement_suggestions=improvement_suggestions
        )
    
    def _scan_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Scan and catalog project structure."""
        structure = {
            'directories': [],
            'files': [],
            'directory_tree': {},
            'file_extensions': {},
            'depth_analysis': {}
        }
        
        files_scanned = 0
        
        for root, dirs, files in os.walk(project_path):
            current_path = Path(root)
            depth = len(current_path.relative_to(project_path).parts)
            
            # Skip if too deep
            if depth > self.max_depth:
                dirs.clear()
                continue
            
            # Track depth
            structure['depth_analysis'][depth] = structure['depth_analysis'].get(depth, 0) + len(files)
            
            # Process directories
            for dir_name in dirs:
                dir_path = current_path / dir_name
                relative_path = str(dir_path.relative_to(project_path))
                
                structure['directories'].append({
                    'path': relative_path,
                    'name': dir_name,
                    'depth': depth + 1,
                    'parent': str(current_path.relative_to(project_path)) if current_path != project_path else ""
                })
            
            # Process files
            for file_name in files:
                if files_scanned >= self.max_files_to_analyze:
                    break
                
                file_path = current_path / file_name
                relative_path = str(file_path.relative_to(project_path))
                
                try:
                    file_stat = file_path.stat()
                    extension = file_path.suffix.lower()
                    
                    structure['files'].append({
                        'path': relative_path,
                        'name': file_name,
                        'extension': extension,
                        'size': file_stat.st_size,
                        'depth': depth
                    })
                    
                    # Track extensions
                    if extension:
                        structure['file_extensions'][extension] = structure['file_extensions'].get(extension, 0) + 1
                    
                    files_scanned += 1
                    
                except (OSError, PermissionError):
                    continue
            
            if files_scanned >= self.max_files_to_analyze:
                break
        
        return structure
    
    def _classify_directories(self, structure_scan: Dict[str, Any]) -> List[DirectoryClassification]:
        """Classify directories by their purpose."""
        classifications = []
        
        for dir_info in structure_scan['directories']:
            dir_name = dir_info['name'].lower()
            dir_path = dir_info['path']
            
            # Calculate classification scores
            scores = {}
            for classification, config in self.DIRECTORY_PATTERNS.items():
                score = 0
                for pattern in config['patterns']:
                    if pattern in dir_name:
                        score += config['weight']
                        if dir_name == pattern:  # Exact match bonus
                            score += 5
                scores[classification] = score
            
            # Determine best classification
            if scores:
                best_classification = max(scores, key=scores.get)
                confidence = min(scores[best_classification] / 10.0, 1.0)
            else:
                best_classification = 'other'
                confidence = 0.1
            
            # Count files in directory
            dir_files = [f for f in structure_scan['files'] if f['path'].startswith(dir_path + '/')]
            file_count = len(dir_files)
            
            # Get important files
            important_files = [f['name'] for f in dir_files[:5]]  # First 5 files
            
            # Sub-classifications
            sub_classifications = [k for k, v in scores.items() if v > 0 and k != best_classification]
            
            classification = DirectoryClassification(
                path=dir_path,
                name=dir_info['name'],
                classification=best_classification,
                confidence=confidence,
                sub_classifications=sub_classifications,
                file_count=file_count,
                important_files=important_files
            )
            classifications.append(classification)
        
        return classifications
    
    def _detect_entry_points(self, project_path: Path, structure_scan: Dict[str, Any]) -> List[EntryPoint]:
        """Detect application entry points."""
        entry_points = []
        
        # Get detected languages from file extensions
        languages = self._detect_languages_from_extensions(structure_scan['file_extensions'])
        
        for file_info in structure_scan['files']:
            file_name = file_info['name']
            file_path = file_info['path']
            
            for language in languages:
                if language in self.ENTRY_POINT_PATTERNS:
                    patterns = self.ENTRY_POINT_PATTERNS[language]
                    
                    for entry_type, pattern_list in patterns.items():
                        for pattern in pattern_list:
                            if self._matches_entry_pattern(file_name, pattern):
                                confidence = self._calculate_entry_point_confidence(file_path, entry_type, language)
                                
                                entry_point = EntryPoint(
                                    file_path=file_path,
                                    entry_type=entry_type,
                                    confidence=confidence,
                                    language=language,
                                    purpose=self._infer_entry_point_purpose(file_path, entry_type)
                                )
                                entry_points.append(entry_point)
        
        # Sort by confidence
        entry_points.sort(key=lambda x: x.confidence, reverse=True)
        
        return entry_points
    
    def _analyze_code_organization(self, project_path: Path, structure_scan: Dict[str, Any]) -> CodeOrganization:
        """Analyze code organization patterns."""
        
        # Calculate structure depth
        max_depth = max(structure_scan['depth_analysis'].keys()) if structure_scan['depth_analysis'] else 0
        
        # Determine namespace strategy
        namespace_strategy = self._determine_namespace_strategy(structure_scan)
        
        # Calculate modularity and coupling scores (simplified)
        modularity_score = self._calculate_modularity_score(structure_scan)
        separation_score = self._calculate_separation_score(structure_scan)
        
        # Analyze coupling (placeholder)
        coupling_analysis = {
            'high_coupling_areas': [],
            'loose_coupling_score': 0.7,
            'dependencies_complexity': 'medium'
        }
        
        # Detect common patterns
        common_patterns = self._detect_code_patterns(structure_scan)
        
        return CodeOrganization(
            structure_depth=max_depth,
            namespace_strategy=namespace_strategy,
            separation_of_concerns=separation_score,
            modularity_score=modularity_score,
            coupling_analysis=coupling_analysis,
            common_patterns=common_patterns
        )
    
    def _analyze_testing_structure(self, project_path: Path, structure_scan: Dict[str, Any]) -> TestingAnalysis:
        """Analyze testing structure and strategy."""
        
        # Find test directories
        test_directories = []
        for dir_info in structure_scan['directories']:
            dir_name = dir_info['name'].lower()
            if any(test_word in dir_name for test_word in ['test', 'tests', 'spec', 'specs', '__tests__']):
                test_directories.append(dir_info['path'])
        
        # Count test files
        test_files = [f for f in structure_scan['files'] if self._is_test_file(f['path'])]
        test_files_count = len(test_files)
        
        # Detect test frameworks (simplified)
        test_frameworks = self._detect_test_frameworks(structure_scan)
        
        # Calculate test-to-code ratio
        code_files = [f for f in structure_scan['files'] if self._is_code_file(f['name'])]
        test_to_code_ratio = test_files_count / len(code_files) if code_files else 0
        
        # Determine testing strategy
        testing_strategy = self._determine_testing_strategy(test_to_code_ratio, test_directories, test_frameworks)
        
        # Find coverage indicators
        coverage_indicators = [f['name'] for f in structure_scan['files'] if 'coverage' in f['name'].lower()]
        
        # Detect testing patterns
        testing_patterns = self._detect_testing_patterns(test_directories, test_frameworks)
        
        return TestingAnalysis(
            testing_strategy=testing_strategy,
            test_directories=test_directories,
            test_files_count=test_files_count,
            test_frameworks=test_frameworks,
            coverage_indicators=coverage_indicators,
            test_to_code_ratio=test_to_code_ratio,
            testing_patterns=testing_patterns
        )
    
    def _analyze_documentation(self, project_path: Path, structure_scan: Dict[str, Any]) -> DocumentationAnalysis:
        """Analyze project documentation."""
        
        # Find README files
        readme_files = [f['path'] for f in structure_scan['files'] 
                       if f['name'].lower().startswith('readme')]
        
        # Find documentation directories
        doc_directories = []
        for dir_info in structure_scan['directories']:
            dir_name = dir_info['name'].lower()
            if any(doc_word in dir_name for doc_word in ['doc', 'docs', 'documentation']):
                doc_directories.append(dir_info['path'])
        
        # Find API documentation
        api_docs = [f['path'] for f in structure_scan['files'] 
                   if any(api_word in f['name'].lower() for api_word in ['swagger', 'openapi', 'api'])]
        
        # Find other important files
        changelog_files = [f['path'] for f in structure_scan['files'] 
                          if any(change_word in f['name'].lower() for change_word in ['changelog', 'changes', 'history'])]
        
        license_files = [f['path'] for f in structure_scan['files'] 
                        if f['name'].lower().startswith('license')]
        
        contributing_files = [f['path'] for f in structure_scan['files'] 
                             if f['name'].lower().startswith('contributing')]
        
        # Calculate documentation completeness (simplified)
        completeness_score = self._calculate_documentation_completeness(
            readme_files, doc_directories, api_docs, license_files
        )
        
        # Placeholder for inline documentation analysis
        inline_doc_score = 0.6  # Would require actual code analysis
        
        return DocumentationAnalysis(
            readme_files=readme_files,
            documentation_directories=doc_directories,
            api_documentation=api_docs,
            inline_documentation_score=inline_doc_score,
            changelog_files=changelog_files,
            license_files=license_files,
            contributing_guidelines=contributing_files,
            documentation_completeness=completeness_score
        )
    
    def _analyze_build_systems(self, project_path: Path, structure_scan: Dict[str, Any]) -> BuildSystemAnalysis:
        """Analyze build and deployment systems."""
        
        # Build tools detection
        build_tools = []
        build_files = []
        
        build_indicators = {
            'make': ['Makefile', 'makefile'],
            'cmake': ['CMakeLists.txt'],
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'npm': ['package.json'],
            'webpack': ['webpack.config.js'],
            'rollup': ['rollup.config.js'],
            'vite': ['vite.config.js'],
            'docker': ['Dockerfile'],
            'poetry': ['pyproject.toml'],
            'cargo': ['Cargo.toml']
        }
        
        for tool, files in build_indicators.items():
            for file_pattern in files:
                matching_files = [f['path'] for f in structure_scan['files'] if f['name'] == file_pattern]
                if matching_files:
                    build_tools.append(tool)
                    build_files.extend(matching_files)
        
        # CI/CD detection
        ci_cd_files = []
        ci_cd_patterns = [
            '.github/workflows/', '.gitlab-ci.yml', 'azure-pipelines.yml',
            'Jenkinsfile', '.travis.yml', '.circleci/', 'buildkite.yml'
        ]
        
        for file_info in structure_scan['files']:
            if any(pattern in file_info['path'] for pattern in ci_cd_patterns):
                ci_cd_files.append(file_info['path'])
        
        # Deployment configs
        deployment_configs = []
        deployment_patterns = ['deploy', 'helm/', 'k8s/', 'kubernetes/', 'terraform/']
        
        for file_info in structure_scan['files']:
            if any(pattern in file_info['path'].lower() for pattern in deployment_patterns):
                deployment_configs.append(file_info['path'])
        
        # Containerization
        containerization = []
        if any('docker' in tool.lower() for tool in build_tools):
            containerization.append('Docker')
        
        docker_compose_files = [f['path'] for f in structure_scan['files'] 
                               if 'docker-compose' in f['name'].lower()]
        if docker_compose_files:
            containerization.append('Docker Compose')
        
        # Orchestration
        orchestration = []
        if any('k8s' in path or 'kubernetes' in path for path in deployment_configs):
            orchestration.append('Kubernetes')
        if any('helm' in path for path in deployment_configs):
            orchestration.append('Helm')
        
        # Package configs
        package_configs = []
        package_files = ['setup.py', 'pyproject.toml', 'package.json', 'Cargo.toml', 'composer.json']
        for file_pattern in package_files:
            matching_files = [f['path'] for f in structure_scan['files'] if f['name'] == file_pattern]
            package_configs.extend(matching_files)
        
        return BuildSystemAnalysis(
            build_tools=build_tools,
            build_files=build_files,
            ci_cd_files=ci_cd_files,
            deployment_configs=deployment_configs,
            containerization=containerization,
            orchestration=orchestration,
            package_configs=package_configs
        )
    
    def _detect_architecture_pattern(self, project_path: Path, structure_scan: Dict[str, Any], 
                                   directory_classifications: List[DirectoryClassification]) -> ArchitecturePattern:
        """Detect the project's architecture pattern."""
        
        pattern_scores = {}
        
        for pattern, config in self.ARCHITECTURE_PATTERNS.items():
            score = 0
            
            # Check directory indicators
            for indicator in config.get('indicators', []):
                for dir_info in structure_scan['directories']:
                    if indicator in dir_info['path'].lower():
                        score += config['weight']
            
            # Check file indicators
            for file_indicator in config.get('files', []):
                for file_info in structure_scan['files']:
                    if file_indicator in file_info['path'].lower():
                        score += config['weight']
            
            pattern_scores[pattern] = score
        
        # Default to modular if no clear pattern
        if not pattern_scores or max(pattern_scores.values()) == 0:
            return ArchitecturePattern.MODULAR
        
        return max(pattern_scores, key=pattern_scores.get)
    
    def _detect_project_type(self, project_path: Path, structure_scan: Dict[str, Any], 
                           entry_points: List[EntryPoint]) -> ProjectType:
        """Detect the project type."""
        
        type_scores = {}
        
        for project_type, config in self.PROJECT_TYPE_INDICATORS.items():
            score = 0
            
            # Check file indicators
            for file_indicator in config.get('files', []):
                for file_info in structure_scan['files']:
                    if file_indicator in file_info['name'].lower():
                        score += config['weight']
            
            # Check directory indicators
            for dir_indicator in config.get('directories', []):
                for dir_info in structure_scan['directories']:
                    if dir_indicator in dir_info['path'].lower():
                        score += config['weight']
            
            type_scores[project_type] = score
        
        # Factor in entry points
        if entry_points:
            main_entry = entry_points[0]
            if main_entry.entry_type == 'server':
                type_scores[ProjectType.API_SERVICE] = type_scores.get(ProjectType.API_SERVICE, 0) + 5
            elif main_entry.entry_type == 'cli':
                type_scores[ProjectType.CLI_TOOL] = type_scores.get(ProjectType.CLI_TOOL, 0) + 5
            elif main_entry.entry_type == 'web':
                type_scores[ProjectType.WEB_APPLICATION] = type_scores.get(ProjectType.WEB_APPLICATION, 0) + 5
        
        # Default to unknown if no clear type
        if not type_scores or max(type_scores.values()) == 0:
            return ProjectType.UNKNOWN
        
        return max(type_scores, key=type_scores.get)
    
    # Helper methods
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file."""
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.cs', '.php']
        return any(filename.lower().endswith(ext) for ext in code_extensions)
    
    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file."""
        test_indicators = ['test', 'spec', '__tests__']
        return any(indicator in filepath.lower() for indicator in test_indicators)
    
    def _is_config_file(self, filename: str) -> bool:
        """Check if file is a configuration file."""
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg']
        config_names = ['config', 'settings', 'makefile', 'dockerfile']
        return (any(filename.lower().endswith(ext) for ext in config_extensions) or
                any(name in filename.lower() for name in config_names))
    
    def _detect_languages_from_extensions(self, file_extensions: Dict[str, int]) -> List[str]:
        """Detect primary languages from file extensions."""
        language_mapping = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php'
        }
        
        detected = []
        for ext, count in file_extensions.items():
            if ext in language_mapping and count > 0:
                detected.append(language_mapping[ext])
        
        return detected
    
    def _matches_entry_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches entry point pattern."""
        if '*' in pattern:
            return re.match(pattern.replace('*', '.*'), filename) is not None
        return filename == pattern
    
    def _calculate_entry_point_confidence(self, filepath: str, entry_type: str, language: str) -> float:
        """Calculate confidence score for entry point detection."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for files in root directory
        if '/' not in filepath:
            confidence += 0.3
        
        # Boost confidence for exact name matches
        filename = Path(filepath).name
        if filename in ['main.py', 'app.py', 'index.js', 'main.js']:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _infer_entry_point_purpose(self, filepath: str, entry_type: str) -> Optional[str]:
        """Infer the purpose of an entry point."""
        if entry_type == 'server':
            return 'HTTP/API server'
        elif entry_type == 'cli':
            return 'Command line interface'
        elif entry_type == 'web':
            return 'Web application'
        elif entry_type == 'test':
            return 'Test runner'
        else:
            return 'Main application'
    
    def _determine_namespace_strategy(self, structure_scan: Dict[str, Any]) -> str:
        """Determine the namespace/organization strategy."""
        max_depth = max(structure_scan['depth_analysis'].keys()) if structure_scan['depth_analysis'] else 0
        
        if max_depth <= 2:
            return 'flat'
        elif max_depth <= 4:
            return 'hierarchical'
        else:
            # Check for domain-based vs feature-based organization
            domain_indicators = ['models', 'views', 'controllers', 'services']
            feature_indicators = ['user', 'product', 'order', 'payment']
            
            directories = [d['name'].lower() for d in structure_scan['directories']]
            
            domain_score = sum(1 for indicator in domain_indicators if indicator in directories)
            feature_score = sum(1 for indicator in feature_indicators if indicator in directories)
            
            if domain_score > feature_score:
                return 'domain-based'
            elif feature_score > domain_score:
                return 'feature-based'
            else:
                return 'hierarchical'
    
    def _calculate_modularity_score(self, structure_scan: Dict[str, Any]) -> float:
        """Calculate modularity score based on structure."""
        # Simplified calculation based on directory organization
        total_dirs = len(structure_scan['directories'])
        if total_dirs == 0:
            return 0.1
        
        # More directories generally indicates better modularity
        if total_dirs > 20:
            return 0.9
        elif total_dirs > 10:
            return 0.7
        elif total_dirs > 5:
            return 0.5
        else:
            return 0.3
    
    def _calculate_separation_score(self, structure_scan: Dict[str, Any]) -> float:
        """Calculate separation of concerns score."""
        # Check for separation indicators
        separation_dirs = ['models', 'views', 'controllers', 'services', 'utils', 'config']
        found_separations = 0
        
        for dir_info in structure_scan['directories']:
            if any(sep_dir in dir_info['name'].lower() for sep_dir in separation_dirs):
                found_separations += 1
        
        return min(found_separations / len(separation_dirs), 1.0)
    
    def _detect_code_patterns(self, structure_scan: Dict[str, Any]) -> List[str]:
        """Detect common code organization patterns."""
        patterns = []
        
        directories = [d['name'].lower() for d in structure_scan['directories']]
        
        if 'src' in directories:
            patterns.append('source-separation')
        if 'test' in directories or 'tests' in directories:
            patterns.append('test-separation')
        if 'config' in directories:
            patterns.append('configuration-separation')
        if any(d in directories for d in ['models', 'views', 'controllers']):
            patterns.append('mvc-pattern')
        if any(d in directories for d in ['services', 'repositories']):
            patterns.append('service-layer-pattern')
        
        return patterns
    
    def _detect_test_frameworks(self, structure_scan: Dict[str, Any]) -> List[str]:
        """Detect testing frameworks used."""
        frameworks = []
        
        # Check file names for framework indicators
        for file_info in structure_scan['files']:
            filename = file_info['name'].lower()
            
            if 'pytest' in filename or 'test_' in filename:
                frameworks.append('pytest')
            elif 'jest' in filename or '.test.js' in filename:
                frameworks.append('jest')
            elif 'mocha' in filename:
                frameworks.append('mocha')
            elif 'unittest' in filename:
                frameworks.append('unittest')
            elif 'junit' in filename:
                frameworks.append('junit')
        
        return list(set(frameworks))  # Remove duplicates
    
    def _determine_testing_strategy(self, test_ratio: float, test_dirs: List[str], frameworks: List[str]) -> TestingStrategy:
        """Determine the overall testing strategy."""
        if test_ratio == 0 and not test_dirs:
            return TestingStrategy.NONE
        elif test_ratio < 0.1:
            return TestingStrategy.MINIMAL
        elif test_ratio < 0.3:
            return TestingStrategy.UNIT_ONLY
        elif len(frameworks) > 1 and test_ratio > 0.5:
            return TestingStrategy.COMPREHENSIVE
        elif any('e2e' in d for d in test_dirs):
            return TestingStrategy.E2E_FOCUSED
        else:
            return TestingStrategy.INTEGRATION_FOCUSED
    
    def _detect_testing_patterns(self, test_dirs: List[str], frameworks: List[str]) -> List[str]:
        """Detect testing patterns and conventions."""
        patterns = []
        
        if test_dirs:
            patterns.append('separate-test-directories')
        if 'pytest' in frameworks:
            patterns.append('pytest-conventions')
        if 'jest' in frameworks:
            patterns.append('jest-conventions')
        if any('e2e' in d for d in test_dirs):
            patterns.append('end-to-end-testing')
        if any('integration' in d for d in test_dirs):
            patterns.append('integration-testing')
        
        return patterns
    
    def _calculate_documentation_completeness(self, readme_files: List[str], doc_dirs: List[str], 
                                            api_docs: List[str], license_files: List[str]) -> float:
        """Calculate documentation completeness score."""
        score = 0.0
        
        if readme_files:
            score += 0.4  # README is most important
        if doc_dirs:
            score += 0.3  # Dedicated documentation
        if api_docs:
            score += 0.2  # API documentation
        if license_files:
            score += 0.1  # License file
        
        return min(score, 1.0)
    
    def _calculate_confidence_scores(self, architecture_pattern: ArchitecturePattern, project_type: ProjectType,
                                   directory_classifications: List[DirectoryClassification], 
                                   entry_points: List[EntryPoint]) -> Dict[str, float]:
        """Calculate confidence scores for various detections."""
        
        # Architecture pattern confidence
        arch_confidence = 0.7  # Base confidence
        
        # Project type confidence
        type_confidence = 0.6 if project_type != ProjectType.UNKNOWN else 0.2
        
        # Directory classification confidence (average)
        dir_confidence = sum(d.confidence for d in directory_classifications) / len(directory_classifications) if directory_classifications else 0.5
        
        # Entry point confidence
        entry_confidence = max([e.confidence for e in entry_points]) if entry_points else 0.3
        
        return {
            'architecture_pattern': arch_confidence,
            'project_type': type_confidence,
            'directory_classification': dir_confidence,
            'entry_points': entry_confidence,
            'overall': (arch_confidence + type_confidence + dir_confidence + entry_confidence) / 4
        }
    
    def _generate_recommendations(self, architecture_pattern: ArchitecturePattern, project_type: ProjectType,
                                code_organization: CodeOrganization, testing_analysis: TestingAnalysis) -> Tuple[List[str], List[str]]:
        """Generate structure recommendations and improvement suggestions."""
        
        structure_recommendations = []
        improvement_suggestions = []
        
        # Architecture-specific recommendations
        if architecture_pattern == ArchitecturePattern.MONOLITHIC:
            structure_recommendations.append("Consider modularizing large components for better maintainability")
        elif architecture_pattern == ArchitecturePattern.MICROSERVICES:
            structure_recommendations.append("Ensure proper service boundaries and communication patterns")
        
        # Testing recommendations
        if testing_analysis.testing_strategy == TestingStrategy.NONE:
            improvement_suggestions.append("Add basic unit tests to improve code quality")
        elif testing_analysis.test_to_code_ratio < 0.2:
            improvement_suggestions.append("Increase test coverage for better reliability")
        
        # Code organization improvements
        if code_organization.modularity_score < 0.5:
            improvement_suggestions.append("Improve code modularity by separating concerns")
        if code_organization.separation_of_concerns < 0.6:
            improvement_suggestions.append("Better separate business logic from presentation layer")
        
        # General structure improvements
        if code_organization.structure_depth > 8:
            improvement_suggestions.append("Consider flattening deep directory structures")
        
        return structure_recommendations, improvement_suggestions


# CLI interface for standalone usage
def main():
    """CLI entry point for structure analysis."""
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="Advanced Project Structure Analysis")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--output", "-o", help="Output file for analysis results")
    parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        analyzer = AdvancedStructureAnalyzer()
        analysis = analyzer.analyze_project_structure(args.project_path)
        
        if args.format == "json":
            result = asdict(analysis)
            # Convert enums to strings for JSON serialization
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                else:
                    return obj
            
            result = convert_enums(result)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Analysis exported to: {args.output}")
            else:
                print(json.dumps(result, indent=2))
        
        else:  # summary format
            print(f"🏗️  Project Structure Analysis")
            print(f"Project: {analysis.project_path}")
            print(f"Architecture: {analysis.architecture_pattern.value}")
            print(f"Type: {analysis.project_type.value}")
            print(f"Files: {analysis.total_files} ({analysis.code_files} code, {analysis.test_files} test)")
            print(f"Directories: {analysis.total_directories}")
            print(f"Testing Strategy: {analysis.testing_analysis.testing_strategy.value}")
            print(f"Overall Confidence: {analysis.confidence_scores['overall']:.1%}")
            
            if analysis.structure_recommendations:
                print(f"\n💡 Recommendations:")
                for rec in analysis.structure_recommendations:
                    print(f"  • {rec}")
            
            if analysis.improvement_suggestions:
                print(f"\n🔧 Improvements:")
                for suggestion in analysis.improvement_suggestions:
                    print(f"  • {suggestion}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()