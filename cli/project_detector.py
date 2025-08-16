"""
Project Detection System

Intelligent project analysis and framework detection for automatic configuration
of the Project Index system. This module provides comprehensive detection of
programming languages, frameworks, build systems, and development tools.
"""

import os
import json
import yaml
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class ProjectType(Enum):
    """Project type classification"""
    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    DESKTOP_APPLICATION = "desktop_application"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    API_SERVICE = "api_service"
    MICROSERVICE = "microservice"
    MONOREPO = "monorepo"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"

class Language(Enum):
    """Programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    OBJECTIVE_C = "objective_c"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SCALA = "scala"
    DART = "dart"
    R = "r"
    JULIA = "julia"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    UNKNOWN = "unknown"

@dataclass
class FrameworkInfo:
    """Information about a detected framework"""
    name: str
    category: str
    language: str
    version: Optional[str] = None
    confidence: float = 1.0
    files_evidence: List[str] = None
    dependencies_evidence: List[str] = None

@dataclass
class ProjectAnalysis:
    """Complete project analysis results"""
    project_path: str
    project_name: str
    project_type: ProjectType
    primary_language: Language
    languages: Dict[Language, float]  # Language -> percentage
    frameworks: List[FrameworkInfo]
    build_systems: List[str]
    development_tools: List[str]
    file_count: int
    line_count: int
    estimated_complexity: str  # "low", "medium", "high", "enterprise"
    monorepo_structure: Optional[Dict[str, Any]] = None
    package_managers: List[str] = None
    testing_frameworks: List[str] = None
    ci_cd_tools: List[str] = None
    deployment_targets: List[str] = None
    documentation_tools: List[str] = None

class ProjectDetector:
    """Advanced project detection and analysis system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Language file extensions mapping
        self.language_extensions = {
            Language.PYTHON: ['.py', '.pyx', '.pyi', '.pyw'],
            Language.JAVASCRIPT: ['.js', '.mjs', '.cjs'],
            Language.TYPESCRIPT: ['.ts', '.tsx'],
            Language.JAVA: ['.java'],
            Language.KOTLIN: ['.kt', '.kts'],
            Language.SWIFT: ['.swift'],
            Language.OBJECTIVE_C: ['.m', '.mm'],
            Language.C: ['.c', '.h'],
            Language.CPP: ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'],
            Language.CSHARP: ['.cs'],
            Language.GO: ['.go'],
            Language.RUST: ['.rs'],
            Language.PHP: ['.php', '.phtml', '.php3', '.php4', '.php5'],
            Language.RUBY: ['.rb', '.rbw'],
            Language.SCALA: ['.scala', '.sc'],
            Language.DART: ['.dart'],
            Language.R: ['.r', '.R'],
            Language.JULIA: ['.jl'],
            Language.SHELL: ['.sh', '.bash', '.zsh', '.fish'],
            Language.SQL: ['.sql'],
            Language.HTML: ['.html', '.htm', '.xhtml'],
            Language.CSS: ['.css', '.scss', '.sass', '.less'],
        }
        
        # Framework detection patterns
        self.framework_patterns = {
            # Python frameworks
            'Flask': {
                'files': ['app.py', 'application.py', 'run.py'],
                'dependencies': ['flask'],
                'patterns': [r'from flask import', r'import flask'],
                'category': 'web_framework',
                'language': 'python'
            },
            'Django': {
                'files': ['manage.py', 'django_settings.py', 'settings.py'],
                'dependencies': ['django'],
                'patterns': [r'from django', r'import django'],
                'category': 'web_framework',
                'language': 'python'
            },
            'FastAPI': {
                'files': ['main.py', 'app.py'],
                'dependencies': ['fastapi'],
                'patterns': [r'from fastapi import', r'FastAPI\('],
                'category': 'web_framework',
                'language': 'python'
            },
            'Streamlit': {
                'files': ['streamlit_app.py', 'app.py'],
                'dependencies': ['streamlit'],
                'patterns': [r'import streamlit', r'st\.'],
                'category': 'web_framework',
                'language': 'python'
            },
            'Jupyter': {
                'files': ['*.ipynb'],
                'dependencies': ['jupyter', 'notebook'],
                'patterns': [],
                'category': 'data_science',
                'language': 'python'
            },
            
            # JavaScript/Node.js frameworks
            'React': {
                'files': [],
                'dependencies': ['react'],
                'patterns': [r'import.*React', r'from ["\']react["\']'],
                'category': 'frontend_framework',
                'language': 'javascript'
            },
            'Vue.js': {
                'files': ['vue.config.js'],
                'dependencies': ['vue'],
                'patterns': [r'import.*Vue', r'from ["\']vue["\']'],
                'category': 'frontend_framework',
                'language': 'javascript'
            },
            'Angular': {
                'files': ['angular.json', 'angular-cli.json'],
                'dependencies': ['@angular/core'],
                'patterns': [r'import.*@angular'],
                'category': 'frontend_framework',
                'language': 'typescript'
            },
            'Next.js': {
                'files': ['next.config.js', 'next.config.ts'],
                'dependencies': ['next'],
                'patterns': [r'import.*next'],
                'category': 'frontend_framework',
                'language': 'javascript'
            },
            'Express.js': {
                'files': [],
                'dependencies': ['express'],
                'patterns': [r'require\(["\']express["\']\)', r'import.*express'],
                'category': 'backend_framework',
                'language': 'javascript'
            },
            'Nest.js': {
                'files': ['nest-cli.json'],
                'dependencies': ['@nestjs/core'],
                'patterns': [r'import.*@nestjs'],
                'category': 'backend_framework',
                'language': 'typescript'
            },
            
            # Build tools and bundlers
            'Webpack': {
                'files': ['webpack.config.js', 'webpack.config.ts'],
                'dependencies': ['webpack'],
                'patterns': [],
                'category': 'build_tool',
                'language': 'javascript'
            },
            'Vite': {
                'files': ['vite.config.js', 'vite.config.ts'],
                'dependencies': ['vite'],
                'patterns': [],
                'category': 'build_tool',
                'language': 'javascript'
            },
            'Rollup': {
                'files': ['rollup.config.js'],
                'dependencies': ['rollup'],
                'patterns': [],
                'category': 'build_tool',
                'language': 'javascript'
            },
            
            # Mobile frameworks
            'React Native': {
                'files': [],
                'dependencies': ['react-native'],
                'patterns': [r'from ["\']react-native["\']'],
                'category': 'mobile_framework',
                'language': 'javascript'
            },
            'Flutter': {
                'files': ['pubspec.yaml'],
                'dependencies': ['flutter'],
                'patterns': [r'import ["\']package:flutter'],
                'category': 'mobile_framework',
                'language': 'dart'
            },
            'Ionic': {
                'files': ['ionic.config.json'],
                'dependencies': ['@ionic/angular', '@ionic/react', '@ionic/vue'],
                'patterns': [],
                'category': 'mobile_framework',
                'language': 'javascript'
            },
            
            # Testing frameworks
            'Jest': {
                'files': ['jest.config.js', 'jest.config.json'],
                'dependencies': ['jest'],
                'patterns': [r'describe\(', r'test\(', r'it\('],
                'category': 'testing_framework',
                'language': 'javascript'
            },
            'Pytest': {
                'files': ['pytest.ini', 'conftest.py'],
                'dependencies': ['pytest'],
                'patterns': [r'import pytest', r'def test_'],
                'category': 'testing_framework',
                'language': 'python'
            },
            'Cypress': {
                'files': ['cypress.json', 'cypress.config.js'],
                'dependencies': ['cypress'],
                'patterns': [],
                'category': 'testing_framework',
                'language': 'javascript'
            },
            'Playwright': {
                'files': ['playwright.config.js', 'playwright.config.ts'],
                'dependencies': ['@playwright/test'],
                'patterns': [],
                'category': 'testing_framework',
                'language': 'javascript'
            },
            
            # Infrastructure and DevOps
            'Docker': {
                'files': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'],
                'dependencies': [],
                'patterns': [r'FROM ', r'RUN ', r'COPY '],
                'category': 'infrastructure',
                'language': 'docker'
            },
            'Kubernetes': {
                'files': ['*.yaml', '*.yml'],
                'dependencies': [],
                'patterns': [r'apiVersion:', r'kind:', r'metadata:'],
                'category': 'infrastructure',
                'language': 'yaml'
            },
            'Terraform': {
                'files': ['*.tf', 'terraform.tfvars'],
                'dependencies': [],
                'patterns': [r'resource "', r'provider "', r'variable "'],
                'category': 'infrastructure',
                'language': 'hcl'
            },
            
            # Other frameworks
            'Tailwind CSS': {
                'files': ['tailwind.config.js', 'tailwind.config.ts'],
                'dependencies': ['tailwindcss'],
                'patterns': [r'@tailwind'],
                'category': 'css_framework',
                'language': 'css'
            },
        }
        
        # Package manager files
        self.package_managers = {
            'npm': ['package.json', 'package-lock.json'],
            'yarn': ['yarn.lock', '.yarnrc'],
            'pnpm': ['pnpm-lock.yaml'],
            'pip': ['requirements.txt', 'requirements-dev.txt'],
            'pipenv': ['Pipfile', 'Pipfile.lock'],
            'poetry': ['pyproject.toml'],
            'conda': ['environment.yml', 'environment.yaml'],
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'cargo': ['Cargo.toml', 'Cargo.lock'],
            'composer': ['composer.json', 'composer.lock'],
            'bundler': ['Gemfile', 'Gemfile.lock'],
            'go-mod': ['go.mod', 'go.sum'],
            'pub': ['pubspec.yaml', 'pubspec.lock'],
        }
        
        # CI/CD tools
        self.ci_cd_patterns = {
            'GitHub Actions': ['.github/workflows/'],
            'GitLab CI': ['.gitlab-ci.yml'],
            'Jenkins': ['Jenkinsfile'],
            'CircleCI': ['.circleci/config.yml'],
            'Travis CI': ['.travis.yml'],
            'Azure Pipelines': ['azure-pipelines.yml', '.azure/'],
            'Bitbucket Pipelines': ['bitbucket-pipelines.yml'],
        }
    
    def analyze_project(self, project_path: str) -> ProjectAnalysis:
        """Perform comprehensive project analysis"""
        self.logger.info(f"Starting project analysis for: {project_path}")
        
        project_path = Path(project_path).resolve()
        project_name = project_path.name
        
        # Basic file analysis
        file_count, line_count = self._count_files_and_lines(project_path)
        languages = self._detect_languages(project_path)
        primary_language = max(languages.items(), key=lambda x: x[1])[0] if languages else Language.UNKNOWN
        
        # Framework detection
        frameworks = self._detect_frameworks(project_path)
        
        # Project type classification
        project_type = self._classify_project_type(project_path, frameworks, languages)
        
        # Additional tools and systems
        build_systems = self._detect_build_systems(project_path)
        development_tools = self._detect_development_tools(project_path)
        package_managers = self._detect_package_managers(project_path)
        testing_frameworks = [f.name for f in frameworks if f.category == 'testing_framework']
        ci_cd_tools = self._detect_ci_cd_tools(project_path)
        deployment_targets = self._detect_deployment_targets(project_path)
        documentation_tools = self._detect_documentation_tools(project_path)
        
        # Complexity estimation
        estimated_complexity = self._estimate_complexity(file_count, line_count, frameworks, languages)
        
        # Monorepo detection
        monorepo_structure = self._detect_monorepo_structure(project_path)
        
        analysis = ProjectAnalysis(
            project_path=str(project_path),
            project_name=project_name,
            project_type=project_type,
            primary_language=primary_language,
            languages=languages,
            frameworks=frameworks,
            build_systems=build_systems,
            development_tools=development_tools,
            file_count=file_count,
            line_count=line_count,
            estimated_complexity=estimated_complexity,
            monorepo_structure=monorepo_structure,
            package_managers=package_managers,
            testing_frameworks=testing_frameworks,
            ci_cd_tools=ci_cd_tools,
            deployment_targets=deployment_targets,
            documentation_tools=documentation_tools
        )
        
        self.logger.info(f"Project analysis complete: {len(frameworks)} frameworks detected")
        return analysis
    
    def _count_files_and_lines(self, project_path: Path) -> Tuple[int, int]:
        """Count files and lines of code, excluding common ignore patterns"""
        ignore_patterns = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache', 
            'venv', '.venv', 'env', '.env', 'dist', 'build', 
            'target', '.idea', '.vscode', 'coverage', '.coverage',
            '*.egg-info', '.tox', '.mypy_cache', 'vendor',
            'public', 'static', 'assets', 'uploads'
        }
        
        file_count = 0
        line_count = 0
        
        try:
            for root, dirs, files in os.walk(project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith('.')]
                
                for file in files:
                    if file.startswith('.') or any(pattern in file for pattern in ignore_patterns):
                        continue
                    
                    file_path = Path(root) / file
                    file_count += 1
                    
                    # Count lines for text files
                    try:
                        if file_path.suffix.lower() in {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb', '.scala', '.kt', '.swift', '.dart', '.r', '.jl', '.sh', '.sql', '.html', '.css', '.scss', '.sass', '.less', '.vue', '.jsx', '.tsx'}:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                line_count += len(f.readlines())
                    except:
                        pass  # Skip files that can't be read
                    
                    # Limit scan for performance
                    if file_count > 50000:
                        break
        except:
            pass  # Handle permission errors gracefully
        
        return file_count, line_count
    
    def _detect_languages(self, project_path: Path) -> Dict[Language, float]:
        """Detect programming languages and their relative usage"""
        language_counts = {}
        total_files = 0
        
        for root, dirs, files in os.walk(project_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'dist', 'build'}]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = Path(file)
                extension = file_path.suffix.lower()
                
                # Find matching language
                for language, extensions in self.language_extensions.items():
                    if extension in extensions:
                        language_counts[language] = language_counts.get(language, 0) + 1
                        total_files += 1
                        break
        
        # Convert to percentages
        if total_files == 0:
            return {Language.UNKNOWN: 1.0}
        
        percentages = {}
        for language, count in language_counts.items():
            percentages[language] = count / total_files
        
        return percentages
    
    def _detect_frameworks(self, project_path: Path) -> List[FrameworkInfo]:
        """Detect frameworks and libraries used in the project"""
        frameworks = []
        
        for framework_name, pattern_config in self.framework_patterns.items():
            confidence = 0.0
            files_evidence = []
            dependencies_evidence = []
            version = None
            
            # Check for specific files
            for file_pattern in pattern_config.get('files', []):
                if '*' in file_pattern:
                    # Handle glob patterns
                    import glob
                    matches = glob.glob(str(project_path / file_pattern), recursive=True)
                    if matches:
                        confidence += 0.8
                        files_evidence.extend([str(Path(m).relative_to(project_path)) for m in matches])
                else:
                    file_path = project_path / file_pattern
                    if file_path.exists():
                        confidence += 0.8
                        files_evidence.append(file_pattern)
            
            # Check dependencies in package files
            dependencies = pattern_config.get('dependencies', [])
            if dependencies:
                dep_found, dep_evidence, dep_version = self._check_dependencies(project_path, dependencies)
                if dep_found:
                    confidence += 0.9
                    dependencies_evidence.extend(dep_evidence)
                    if dep_version:
                        version = dep_version
            
            # Check code patterns
            patterns = pattern_config.get('patterns', [])
            if patterns and confidence > 0:  # Only check patterns if we have other evidence
                pattern_found = self._check_code_patterns(project_path, patterns)
                if pattern_found:
                    confidence += 0.3
            
            # Create framework info if we have sufficient confidence
            if confidence >= 0.5:
                framework = FrameworkInfo(
                    name=framework_name,
                    category=pattern_config['category'],
                    language=pattern_config['language'],
                    version=version,
                    confidence=min(confidence, 1.0),
                    files_evidence=files_evidence,
                    dependencies_evidence=dependencies_evidence
                )
                frameworks.append(framework)
        
        # Sort by confidence
        frameworks.sort(key=lambda f: f.confidence, reverse=True)
        return frameworks
    
    def _check_dependencies(self, project_path: Path, dependencies: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Check if dependencies are present in package files"""
        evidence = []
        version = None
        
        # Check package.json
        package_json = project_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                
                for dep in dependencies:
                    for dep_section in ['dependencies', 'devDependencies', 'peerDependencies']:
                        if dep_section in data and dep in data[dep_section]:
                            evidence.append(f"package.json:{dep_section}:{dep}")
                            if not version:
                                version = data[dep_section][dep]
                            return True, evidence, version
            except:
                pass
        
        # Check requirements.txt
        requirements_txt = project_path / 'requirements.txt'
        if requirements_txt.exists():
            try:
                with open(requirements_txt, 'r') as f:
                    content = f.read()
                
                for dep in dependencies:
                    if dep in content:
                        evidence.append(f"requirements.txt:{dep}")
                        # Try to extract version
                        version_match = re.search(f"{dep}[>=<]*([0-9.]+)", content)
                        if version_match and not version:
                            version = version_match.group(1)
                        return True, evidence, version
            except:
                pass
        
        # Check pyproject.toml
        pyproject_toml = project_path / 'pyproject.toml'
        if pyproject_toml.exists():
            try:
                with open(pyproject_toml, 'r') as f:
                    content = f.read()
                
                for dep in dependencies:
                    if dep in content:
                        evidence.append(f"pyproject.toml:{dep}")
                        return True, evidence, version
            except:
                pass
        
        # Check Pipfile
        pipfile = project_path / 'Pipfile'
        if pipfile.exists():
            try:
                with open(pipfile, 'r') as f:
                    content = f.read()
                
                for dep in dependencies:
                    if dep in content:
                        evidence.append(f"Pipfile:{dep}")
                        return True, evidence, version
            except:
                pass
        
        return len(evidence) > 0, evidence, version
    
    def _check_code_patterns(self, project_path: Path, patterns: List[str]) -> bool:
        """Check for specific code patterns in source files"""
        try:
            # Use grep-like search for performance
            for pattern in patterns:
                try:
                    result = subprocess.run([
                        'grep', '-r', '--include=*.py', '--include=*.js', '--include=*.ts', 
                        '--include=*.jsx', '--include=*.tsx', '--include=*.vue',
                        pattern, str(project_path)
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Fallback to Python search if grep is not available
                    return self._python_pattern_search(project_path, patterns)
        except:
            pass
        
        return False
    
    def _python_pattern_search(self, project_path: Path, patterns: List[str]) -> bool:
        """Fallback pattern search using Python"""
        extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.vue'}
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv'}]
            
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for pattern in patterns:
                            if re.search(pattern, content):
                                return True
                    except:
                        continue
        
        return False
    
    def _classify_project_type(self, project_path: Path, frameworks: List[FrameworkInfo], languages: Dict[Language, float]) -> ProjectType:
        """Classify the type of project based on frameworks and structure"""
        framework_names = [f.name.lower() for f in frameworks]
        framework_categories = [f.category for f in frameworks]
        
        # Check for specific project types
        if any('mobile' in cat for cat in framework_categories):
            return ProjectType.MOBILE_APPLICATION
        
        if any(name in framework_names for name in ['react native', 'flutter', 'ionic']):
            return ProjectType.MOBILE_APPLICATION
        
        if any('frontend' in cat for cat in framework_categories):
            return ProjectType.WEB_APPLICATION
        
        if any('backend' in cat or 'web' in cat for cat in framework_categories):
            return ProjectType.WEB_APPLICATION
        
        if 'infrastructure' in framework_categories:
            return ProjectType.INFRASTRUCTURE
        
        # Check for monorepo structure
        if self._detect_monorepo_structure(project_path):
            return ProjectType.MONOREPO
        
        # Check for specific files that indicate project type
        if (project_path / 'setup.py').exists() or (project_path / 'pyproject.toml').exists():
            return ProjectType.LIBRARY
        
        if (project_path / 'main.py').exists() or (project_path / 'cli.py').exists():
            return ProjectType.CLI_TOOL
        
        # Check for Jupyter notebooks
        jupyter_files = list(project_path.rglob('*.ipynb'))
        if jupyter_files:
            return ProjectType.DATA_SCIENCE
        
        # Check for API-specific patterns
        api_indicators = ['app.py', 'main.py', 'server.py', 'api.py']
        if any((project_path / indicator).exists() for indicator in api_indicators):
            return ProjectType.API_SERVICE
        
        # Check for documentation projects
        docs_indicators = ['docs/', 'documentation/', 'mkdocs.yml', 'sphinx/', 'docusaurus.config.js']
        if any((project_path / indicator).exists() for indicator in docs_indicators):
            return ProjectType.DOCUMENTATION
        
        # Default classification based on languages
        if Language.PYTHON in languages and languages[Language.PYTHON] > 0.5:
            return ProjectType.API_SERVICE
        
        if Language.JAVASCRIPT in languages or Language.TYPESCRIPT in languages:
            return ProjectType.WEB_APPLICATION
        
        return ProjectType.UNKNOWN
    
    def _detect_build_systems(self, project_path: Path) -> List[str]:
        """Detect build systems and tools"""
        build_systems = []
        
        build_indicators = {
            'Webpack': ['webpack.config.js', 'webpack.config.ts'],
            'Vite': ['vite.config.js', 'vite.config.ts'],
            'Rollup': ['rollup.config.js'],
            'Parcel': ['.parcelrc'],
            'Gulp': ['gulpfile.js'],
            'Grunt': ['Gruntfile.js'],
            'Make': ['Makefile', 'makefile'],
            'CMake': ['CMakeLists.txt'],
            'Gradle': ['build.gradle', 'build.gradle.kts'],
            'Maven': ['pom.xml'],
            'SBT': ['build.sbt'],
            'Cargo': ['Cargo.toml'],
            'Go Modules': ['go.mod'],
            'Poetry': ['pyproject.toml'],
            'setuptools': ['setup.py'],
        }
        
        for build_system, indicators in build_indicators.items():
            if any((project_path / indicator).exists() for indicator in indicators):
                build_systems.append(build_system)
        
        return build_systems
    
    def _detect_development_tools(self, project_path: Path) -> List[str]:
        """Detect development tools and utilities"""
        tools = []
        
        tool_indicators = {
            'ESLint': ['.eslintrc.json', '.eslintrc.js', '.eslintrc.yml'],
            'Prettier': ['.prettierrc', '.prettier.config.js'],
            'Black': ['pyproject.toml'],  # Will check content separately
            'flake8': ['.flake8', 'setup.cfg'],
            'mypy': ['mypy.ini', '.mypy.ini'],
            'TypeScript': ['tsconfig.json'],
            'Babel': ['.babelrc', 'babel.config.js'],
            'EditorConfig': ['.editorconfig'],
            'Husky': ['.husky/'],
            'pre-commit': ['.pre-commit-config.yaml'],
            'Commitizen': ['.cz.json', '.czrc'],
            'Semantic Release': ['.releaserc', 'release.config.js'],
        }
        
        for tool, indicators in tool_indicators.items():
            if any((project_path / indicator).exists() for indicator in indicators):
                tools.append(tool)
        
        return tools
    
    def _detect_package_managers(self, project_path: Path) -> List[str]:
        """Detect package managers used in the project"""
        managers = []
        
        for manager, files in self.package_managers.items():
            if any((project_path / file).exists() for file in files):
                managers.append(manager)
        
        return managers
    
    def _detect_ci_cd_tools(self, project_path: Path) -> List[str]:
        """Detect CI/CD tools and platforms"""
        tools = []
        
        for tool, patterns in self.ci_cd_patterns.items():
            if any((project_path / pattern).exists() for pattern in patterns):
                tools.append(tool)
        
        return tools
    
    def _detect_deployment_targets(self, project_path: Path) -> List[str]:
        """Detect deployment targets and platforms"""
        targets = []
        
        deployment_indicators = {
            'Docker': ['Dockerfile', 'docker-compose.yml'],
            'Kubernetes': ['k8s/', 'kubernetes/', '*.yaml'],
            'Heroku': ['Procfile', 'app.json'],
            'Vercel': ['vercel.json', '.vercel/'],
            'Netlify': ['netlify.toml', '_redirects'],
            'AWS': ['aws/', '.aws/', 'cloudformation/', 'serverless.yml'],
            'GCP': ['gcp/', '.gcp/', 'app.yaml'],
            'Azure': ['azure/', '.azure/', 'azure-pipelines.yml'],
            'Railway': ['railway.toml'],
            'Fly.io': ['fly.toml'],
        }
        
        for target, indicators in deployment_indicators.items():
            if any((project_path / indicator).exists() for indicator in indicators):
                targets.append(target)
        
        return targets
    
    def _detect_documentation_tools(self, project_path: Path) -> List[str]:
        """Detect documentation tools and generators"""
        tools = []
        
        doc_indicators = {
            'Sphinx': ['docs/conf.py', 'doc/conf.py'],
            'MkDocs': ['mkdocs.yml', 'mkdocs.yaml'],
            'Docusaurus': ['docusaurus.config.js'],
            'GitBook': ['.gitbook.yaml'],
            'VuePress': ['.vuepress/'],
            'Docsify': ['_docsify/'],
            'Jekyll': ['_config.yml', '_site/'],
            'Hugo': ['config.toml', 'config.yaml'],
            'Gatsby': ['gatsby-config.js'],
            'Storybook': ['.storybook/'],
            'JSDoc': ['jsdoc.conf.json'],
            'TypeDoc': ['typedoc.json'],
            'Swagger/OpenAPI': ['swagger.yml', 'openapi.yml', 'api-docs/'],
        }
        
        for tool, indicators in doc_indicators.items():
            if any((project_path / indicator).exists() for indicator in indicators):
                tools.append(tool)
        
        return tools
    
    def _estimate_complexity(self, file_count: int, line_count: int, frameworks: List[FrameworkInfo], languages: Dict[Language, float]) -> str:
        """Estimate project complexity based on various factors"""
        complexity_score = 0
        
        # File count factor
        if file_count > 10000:
            complexity_score += 4
        elif file_count > 1000:
            complexity_score += 3
        elif file_count > 100:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Line count factor
        if line_count > 100000:
            complexity_score += 3
        elif line_count > 10000:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Framework complexity
        complex_frameworks = {'Angular', 'React', 'Vue.js', 'Django', 'Spring Boot', 'Kubernetes'}
        framework_names = {f.name for f in frameworks}
        
        if len(framework_names & complex_frameworks) > 2:
            complexity_score += 3
        elif len(framework_names & complex_frameworks) > 0:
            complexity_score += 2
        
        # Language diversity
        if len(languages) > 5:
            complexity_score += 2
        elif len(languages) > 2:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 10:
            return "enterprise"
        elif complexity_score >= 7:
            return "high"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _detect_monorepo_structure(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Detect if this is a monorepo and analyze its structure"""
        # Look for common monorepo indicators
        monorepo_indicators = [
            'lerna.json',
            'nx.json',
            'rush.json',
            'workspace.json',
            'pnpm-workspace.yaml',
            'yarn.lock'  # With workspaces
        ]
        
        if any((project_path / indicator).exists() for indicator in monorepo_indicators):
            # Analyze structure
            packages = []
            
            # Check for packages/ or apps/ directories
            for subdir in ['packages', 'apps', 'libs', 'modules', 'services']:
                subdir_path = project_path / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    for item in subdir_path.iterdir():
                        if item.is_dir():
                            packages.append(str(item.relative_to(project_path)))
            
            # Check workspace configuration
            workspace_config = None
            package_json = project_path / 'package.json'
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        data = json.load(f)
                        if 'workspaces' in data:
                            workspace_config = data['workspaces']
                except:
                    pass
            
            if packages or workspace_config:
                return {
                    'packages': packages,
                    'workspace_config': workspace_config,
                    'detected_tools': [tool for tool in ['Lerna', 'Nx', 'Rush', 'Yarn Workspaces'] 
                                     if (project_path / f"{tool.lower().replace(' ', '-')}.json").exists()]
                }
        
        return None
    
    def generate_summary_report(self, analysis: ProjectAnalysis) -> str:
        """Generate a human-readable summary report"""
        report = []
        
        report.append(f"📊 Project Analysis Report")
        report.append(f"=" * 50)
        report.append(f"Project: {analysis.project_name}")
        report.append(f"Path: {analysis.project_path}")
        report.append(f"Type: {analysis.project_type.value.replace('_', ' ').title()}")
        report.append(f"Complexity: {analysis.estimated_complexity.title()}")
        report.append("")
        
        # Languages
        report.append("🔤 Languages:")
        for lang, percentage in sorted(analysis.languages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0.05:  # Only show languages > 5%
                report.append(f"  • {lang.value.title()}: {percentage:.1%}")
        report.append("")
        
        # Frameworks
        if analysis.frameworks:
            report.append("🚀 Frameworks & Libraries:")
            for framework in analysis.frameworks:
                confidence_stars = "★" * int(framework.confidence * 5)
                report.append(f"  • {framework.name} ({framework.category}) {confidence_stars}")
        report.append("")
        
        # Statistics
        report.append("📈 Statistics:")
        report.append(f"  • Files: {analysis.file_count:,}")
        report.append(f"  • Lines of Code: {analysis.line_count:,}")
        report.append("")
        
        # Tools
        if analysis.build_systems:
            report.append(f"🔧 Build Systems: {', '.join(analysis.build_systems)}")
        
        if analysis.package_managers:
            report.append(f"📦 Package Managers: {', '.join(analysis.package_managers)}")
        
        if analysis.testing_frameworks:
            report.append(f"🧪 Testing: {', '.join(analysis.testing_frameworks)}")
        
        if analysis.ci_cd_tools:
            report.append(f"🔄 CI/CD: {', '.join(analysis.ci_cd_tools)}")
        
        if analysis.deployment_targets:
            report.append(f"🚀 Deployment: {', '.join(analysis.deployment_targets)}")
        
        # Monorepo info
        if analysis.monorepo_structure:
            report.append("")
            report.append("📁 Monorepo Structure:")
            packages = analysis.monorepo_structure.get('packages', [])
            if packages:
                report.append(f"  • Packages: {len(packages)}")
                for pkg in packages[:10]:  # Show first 10
                    report.append(f"    - {pkg}")
                if len(packages) > 10:
                    report.append(f"    ... and {len(packages) - 10} more")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python project_detector.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    detector = ProjectDetector()
    analysis = detector.analyze_project(project_path)
    
    print(detector.generate_summary_report(analysis))
    
    # Save detailed analysis to JSON
    with open("project_analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2, default=str)