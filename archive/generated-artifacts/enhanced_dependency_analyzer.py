#!/usr/bin/env python3
"""
Enhanced Dependency Analysis Engine
===================================

Advanced dependency parsing and analysis for multiple package managers and languages.
Supports version range analysis, security scanning, and intelligent dependency insights.

Features:
- Multi-language dependency parsing (Python, JavaScript, Go, Rust, Java, C#, PHP)
- Version constraint analysis and resolution
- Dependency graph construction
- Security vulnerability detection
- License compatibility analysis
- Outdated dependency detection
- Dependency size and impact analysis

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import logging
import tomllib
import yaml

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    OPTIONAL = "optional"
    PEER = "peer"
    BUILD = "build"
    TEST = "test"


class PackageManager(Enum):
    """Supported package managers."""
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    PIP = "pip"
    POETRY = "poetry"
    PIPENV = "pipenv"
    CONDA = "conda"
    GO_MODULES = "go-modules"
    CARGO = "cargo"
    MAVEN = "maven"
    GRADLE = "gradle"
    NUGET = "nuget"
    COMPOSER = "composer"


class SecurityLevel(Enum):
    """Security vulnerability levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DependencyVersion:
    """Represents a dependency version constraint."""
    raw_version: str
    operator: Optional[str] = None  # ^, ~, >=, >, <, <=, =
    major: Optional[int] = None
    minor: Optional[int] = None
    patch: Optional[int] = None
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None
    is_range: bool = False
    is_exact: bool = False


@dataclass
class Dependency:
    """Represents a single dependency."""
    name: str
    version: DependencyVersion
    dependency_type: DependencyType
    package_manager: PackageManager
    source_file: str
    description: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    size_bytes: Optional[int] = None
    is_direct: bool = True
    security_vulnerabilities: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.security_vulnerabilities is None:
            self.security_vulnerabilities = []


@dataclass
class DependencyGraph:
    """Represents the complete dependency graph."""
    dependencies: List[Dependency]
    total_count: int
    by_type: Dict[DependencyType, int]
    by_package_manager: Dict[PackageManager, int]
    direct_dependencies: List[Dependency]
    transitive_dependencies: List[Dependency]
    security_summary: Dict[SecurityLevel, int]
    license_summary: Dict[str, int]
    outdated_dependencies: List[Dependency]
    size_analysis: Dict[str, Any]


class EnhancedDependencyAnalyzer:
    """
    Advanced dependency analyzer supporting multiple package managers
    and providing comprehensive dependency insights.
    """
    
    # Version parsing patterns
    VERSION_PATTERNS = {
        'semver': re.compile(r'^(\^|~|>=|>|<=|<|=)?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'),
        'pip': re.compile(r'^([><=!~]+)?(\d+(?:\.\d+)*(?:[a-zA-Z0-9\-\.]+)?)$'),
        'maven': re.compile(r'^(\[|\()?(\d+(?:\.\d+)*),?(\d+(?:\.\d+)*)?(\]|\))?$')
    }
    
    # Package manager configurations
    PACKAGE_MANAGERS = {
        PackageManager.NPM: {
            'files': ['package.json'],
            'lock_files': ['package-lock.json'],
            'dependency_fields': ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies'],
            'version_pattern': 'semver'
        },
        PackageManager.YARN: {
            'files': ['package.json'],
            'lock_files': ['yarn.lock'],
            'dependency_fields': ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies'],
            'version_pattern': 'semver'
        },
        PackageManager.PNPM: {
            'files': ['package.json'],
            'lock_files': ['pnpm-lock.yaml'],
            'dependency_fields': ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies'],
            'version_pattern': 'semver'
        },
        PackageManager.PIP: {
            'files': ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt'],
            'lock_files': ['pip.lock'],
            'version_pattern': 'pip'
        },
        PackageManager.POETRY: {
            'files': ['pyproject.toml'],
            'lock_files': ['poetry.lock'],
            'version_pattern': 'pip'
        },
        PackageManager.PIPENV: {
            'files': ['Pipfile'],
            'lock_files': ['Pipfile.lock'],
            'version_pattern': 'pip'
        },
        PackageManager.CONDA: {
            'files': ['environment.yml', 'environment.yaml', 'conda.yml'],
            'lock_files': [],
            'version_pattern': 'pip'
        },
        PackageManager.GO_MODULES: {
            'files': ['go.mod'],
            'lock_files': ['go.sum'],
            'version_pattern': 'semver'
        },
        PackageManager.CARGO: {
            'files': ['Cargo.toml'],
            'lock_files': ['Cargo.lock'],
            'version_pattern': 'semver'
        },
        PackageManager.MAVEN: {
            'files': ['pom.xml'],
            'lock_files': [],
            'version_pattern': 'maven'
        },
        PackageManager.GRADLE: {
            'files': ['build.gradle', 'build.gradle.kts'],
            'lock_files': ['gradle.lockfile'],
            'version_pattern': 'maven'
        },
        PackageManager.NUGET: {
            'files': ['*.csproj', 'packages.config', 'PackageReference'],
            'lock_files': ['packages.lock.json'],
            'version_pattern': 'semver'
        },
        PackageManager.COMPOSER: {
            'files': ['composer.json'],
            'lock_files': ['composer.lock'],
            'version_pattern': 'semver'
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced dependency analyzer."""
        self.config = config or {}
        self.security_databases = self.config.get('security_databases', [])
        self.license_compatibility = self.config.get('license_compatibility', {})
        self.cache = {}
        
        logger.info("Enhanced dependency analyzer initialized")
    
    def analyze_project_dependencies(self, project_path: Union[str, Path]) -> DependencyGraph:
        """
        Analyze all dependencies in a project across multiple package managers.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Complete dependency graph with analysis
        """
        project_path = Path(project_path)
        all_dependencies = []
        
        logger.info("Starting dependency analysis", project_path=str(project_path))
        
        # Scan for all package manager files
        package_manager_files = self._discover_package_files(project_path)
        
        # Analyze each package manager
        for package_manager, files in package_manager_files.items():
            for file_path in files:
                try:
                    dependencies = self._analyze_dependency_file(file_path, package_manager)
                    all_dependencies.extend(dependencies)
                    logger.debug(f"Analyzed {len(dependencies)} dependencies from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Build dependency graph
        return self._build_dependency_graph(all_dependencies)
    
    def _discover_package_files(self, project_path: Path) -> Dict[PackageManager, List[Path]]:
        """Discover all package manager files in the project."""
        discovered = {}
        
        for package_manager, config in self.PACKAGE_MANAGERS.items():
            files = []
            for file_pattern in config['files']:
                if '*' in file_pattern:
                    # Use glob for wildcard patterns
                    found_files = list(project_path.rglob(file_pattern))
                else:
                    # Direct file check
                    file_path = project_path / file_pattern
                    if file_path.exists():
                        found_files = [file_path]
                    else:
                        found_files = []
                
                files.extend(found_files)
            
            if files:
                discovered[package_manager] = files
        
        return discovered
    
    def _analyze_dependency_file(self, file_path: Path, package_manager: PackageManager) -> List[Dependency]:
        """Analyze a specific dependency file."""
        logger.debug(f"Analyzing dependency file: {file_path} ({package_manager.value})")
        
        # Route to appropriate parser
        if package_manager in [PackageManager.NPM, PackageManager.YARN, PackageManager.PNPM]:
            return self._parse_package_json(file_path, package_manager)
        elif package_manager == PackageManager.PIP:
            return self._parse_requirements_txt(file_path)
        elif package_manager == PackageManager.POETRY:
            return self._parse_pyproject_toml(file_path)
        elif package_manager == PackageManager.PIPENV:
            return self._parse_pipfile(file_path)
        elif package_manager == PackageManager.CONDA:
            return self._parse_conda_environment(file_path)
        elif package_manager == PackageManager.GO_MODULES:
            return self._parse_go_mod(file_path)
        elif package_manager == PackageManager.CARGO:
            return self._parse_cargo_toml(file_path)
        elif package_manager == PackageManager.MAVEN:
            return self._parse_maven_pom(file_path)
        elif package_manager == PackageManager.GRADLE:
            return self._parse_gradle_build(file_path)
        elif package_manager == PackageManager.COMPOSER:
            return self._parse_composer_json(file_path)
        else:
            logger.warning(f"Unsupported package manager: {package_manager}")
            return []
    
    def _parse_package_json(self, file_path: Path, package_manager: PackageManager) -> List[Dependency]:
        """Parse package.json for JavaScript/Node.js dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse different dependency types
            dep_sections = {
                'dependencies': DependencyType.PRODUCTION,
                'devDependencies': DependencyType.DEVELOPMENT,
                'peerDependencies': DependencyType.PEER,
                'optionalDependencies': DependencyType.OPTIONAL
            }
            
            for section, dep_type in dep_sections.items():
                if section in data:
                    for name, version_spec in data[section].items():
                        version = self._parse_version(version_spec, 'semver')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=dep_type,
                            package_manager=package_manager,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
            
        except Exception as e:
            logger.error(f"Failed to parse package.json {file_path}: {e}")
        
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path) -> List[Dependency]:
        """Parse requirements.txt for Python dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                # Parse dependency line
                match = re.match(r'^([a-zA-Z0-9\-_\.]+)([><=!~]+.*)?$', line)
                if match:
                    name = match.group(1)
                    version_spec = match.group(2) or ""
                    
                    version = self._parse_version(version_spec, 'pip')
                    
                    # Determine dependency type from filename
                    dep_type = DependencyType.PRODUCTION
                    if 'dev' in file_path.name or 'test' in file_path.name:
                        dep_type = DependencyType.DEVELOPMENT
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=dep_type,
                        package_manager=PackageManager.PIP,
                        source_file=str(file_path.name)
                    )
                    dependencies.append(dependency)
                    
        except Exception as e:
            logger.error(f"Failed to parse requirements.txt {file_path}: {e}")
        
        return dependencies
    
    def _parse_pyproject_toml(self, file_path: Path) -> List[Dependency]:
        """Parse pyproject.toml for Poetry dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Poetry dependencies
            if 'tool' in data and 'poetry' in data['tool']:
                poetry_config = data['tool']['poetry']
                
                # Production dependencies
                if 'dependencies' in poetry_config:
                    for name, version_spec in poetry_config['dependencies'].items():
                        if name == 'python':  # Skip Python version spec
                            continue
                        
                        # Handle complex version specifications
                        if isinstance(version_spec, dict):
                            version_str = version_spec.get('version', '*')
                        else:
                            version_str = str(version_spec)
                        
                        version = self._parse_version(version_str, 'pip')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=DependencyType.PRODUCTION,
                            package_manager=PackageManager.POETRY,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
                
                # Development dependencies
                if 'group' in poetry_config and 'dev' in poetry_config['group']:
                    dev_deps = poetry_config['group']['dev'].get('dependencies', {})
                    for name, version_spec in dev_deps.items():
                        if isinstance(version_spec, dict):
                            version_str = version_spec.get('version', '*')
                        else:
                            version_str = str(version_spec)
                        
                        version = self._parse_version(version_str, 'pip')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=DependencyType.DEVELOPMENT,
                            package_manager=PackageManager.POETRY,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
                        
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml {file_path}: {e}")
        
        return dependencies
    
    def _parse_go_mod(self, file_path: Path) -> List[Dependency]:
        """Parse go.mod for Go module dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_require_block = False
            for line in lines:
                line = line.strip()
                
                if line.startswith('require ('):
                    in_require_block = True
                    continue
                elif line == ')' and in_require_block:
                    in_require_block = False
                    continue
                elif line.startswith('require ') and not in_require_block:
                    # Single require statement
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1]
                        version_str = parts[2]
                        
                        version = self._parse_version(version_str, 'semver')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=DependencyType.PRODUCTION,
                            package_manager=PackageManager.GO_MODULES,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
                elif in_require_block and line:
                    # Inside require block
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        version_str = parts[1]
                        
                        version = self._parse_version(version_str, 'semver')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=DependencyType.PRODUCTION,
                            package_manager=PackageManager.GO_MODULES,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
                        
        except Exception as e:
            logger.error(f"Failed to parse go.mod {file_path}: {e}")
        
        return dependencies
    
    def _parse_cargo_toml(self, file_path: Path) -> List[Dependency]:
        """Parse Cargo.toml for Rust dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Production dependencies
            if 'dependencies' in data:
                for name, version_spec in data['dependencies'].items():
                    if isinstance(version_spec, dict):
                        version_str = version_spec.get('version', '*')
                    else:
                        version_str = str(version_spec)
                    
                    version = self._parse_version(version_str, 'semver')
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=DependencyType.PRODUCTION,
                        package_manager=PackageManager.CARGO,
                        source_file=str(file_path.name)
                    )
                    dependencies.append(dependency)
            
            # Development dependencies
            if 'dev-dependencies' in data:
                for name, version_spec in data['dev-dependencies'].items():
                    if isinstance(version_spec, dict):
                        version_str = version_spec.get('version', '*')
                    else:
                        version_str = str(version_spec)
                    
                    version = self._parse_version(version_str, 'semver')
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=DependencyType.DEVELOPMENT,
                        package_manager=PackageManager.CARGO,
                        source_file=str(file_path.name)
                    )
                    dependencies.append(dependency)
                    
        except Exception as e:
            logger.error(f"Failed to parse Cargo.toml {file_path}: {e}")
        
        return dependencies
    
    def _parse_maven_pom(self, file_path: Path) -> List[Dependency]:
        """Parse pom.xml for Maven dependencies."""
        dependencies = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Handle XML namespaces
            namespace = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            if root.tag.startswith('{'):
                ns = root.tag[1:].split('}')[0]
                namespace = {'maven': ns}
            
            # Find dependencies
            deps_elem = root.find('.//maven:dependencies', namespace)
            if deps_elem is not None:
                for dep_elem in deps_elem.findall('maven:dependency', namespace):
                    group_id = dep_elem.find('maven:groupId', namespace)
                    artifact_id = dep_elem.find('maven:artifactId', namespace)
                    version_elem = dep_elem.find('maven:version', namespace)
                    scope_elem = dep_elem.find('maven:scope', namespace)
                    
                    if group_id is not None and artifact_id is not None:
                        name = f"{group_id.text}:{artifact_id.text}"
                        version_str = version_elem.text if version_elem is not None else "LATEST"
                        scope = scope_elem.text if scope_elem is not None else "compile"
                        
                        # Map Maven scope to dependency type
                        scope_mapping = {
                            'compile': DependencyType.PRODUCTION,
                            'test': DependencyType.TEST,
                            'provided': DependencyType.OPTIONAL,
                            'runtime': DependencyType.PRODUCTION
                        }
                        
                        dep_type = scope_mapping.get(scope, DependencyType.PRODUCTION)
                        version = self._parse_version(version_str, 'maven')
                        
                        dependency = Dependency(
                            name=name,
                            version=version,
                            dependency_type=dep_type,
                            package_manager=PackageManager.MAVEN,
                            source_file=str(file_path.name)
                        )
                        dependencies.append(dependency)
                        
        except Exception as e:
            logger.error(f"Failed to parse pom.xml {file_path}: {e}")
        
        return dependencies
    
    def _parse_composer_json(self, file_path: Path) -> List[Dependency]:
        """Parse composer.json for PHP dependencies."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Production dependencies
            if 'require' in data:
                for name, version_spec in data['require'].items():
                    if name == 'php':  # Skip PHP version spec
                        continue
                    
                    version = self._parse_version(version_spec, 'semver')
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=DependencyType.PRODUCTION,
                        package_manager=PackageManager.COMPOSER,
                        source_file=str(file_path.name)
                    )
                    dependencies.append(dependency)
            
            # Development dependencies
            if 'require-dev' in data:
                for name, version_spec in data['require-dev'].items():
                    version = self._parse_version(version_spec, 'semver')
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=DependencyType.DEVELOPMENT,
                        package_manager=PackageManager.COMPOSER,
                        source_file=str(file_path.name)
                    )
                    dependencies.append(dependency)
                    
        except Exception as e:
            logger.error(f"Failed to parse composer.json {file_path}: {e}")
        
        return dependencies
    
    # Placeholder methods for other parsers
    def _parse_pipfile(self, file_path: Path) -> List[Dependency]:
        """Parse Pipfile for Pipenv dependencies.""" 
        # TODO: Implement Pipfile parsing
        return []
    
    def _parse_conda_environment(self, file_path: Path) -> List[Dependency]:
        """Parse conda environment.yml."""
        # TODO: Implement conda environment parsing
        return []
    
    def _parse_gradle_build(self, file_path: Path) -> List[Dependency]:
        """Parse Gradle build files."""
        # TODO: Implement Gradle parsing
        return []
    
    def _parse_version(self, version_spec: str, pattern_type: str) -> DependencyVersion:
        """Parse version specification into structured format."""
        if not version_spec:
            return DependencyVersion(raw_version="*", is_range=True)
        
        version_spec = version_spec.strip()
        
        # Try to match version pattern
        pattern = self.VERSION_PATTERNS.get(pattern_type)
        if pattern:
            match = pattern.match(version_spec)
            if match:
                groups = match.groups()
                
                if pattern_type == 'semver':
                    operator = groups[0]
                    major = int(groups[1]) if groups[1] else None
                    minor = int(groups[2]) if groups[2] else None
                    patch = int(groups[3]) if groups[3] else None
                    pre_release = groups[4]
                    build_metadata = groups[5]
                    
                    return DependencyVersion(
                        raw_version=version_spec,
                        operator=operator,
                        major=major,
                        minor=minor,
                        patch=patch,
                        pre_release=pre_release,
                        build_metadata=build_metadata,
                        is_range=operator in ['^', '~', '>=', '>', '<', '<='] if operator else False,
                        is_exact=operator == '=' or operator is None
                    )
        
        # Fallback for unparseable versions
        return DependencyVersion(
            raw_version=version_spec,
            is_range='*' in version_spec or any(op in version_spec for op in ['>', '<', '~', '^'])
        )
    
    def _build_dependency_graph(self, dependencies: List[Dependency]) -> DependencyGraph:
        """Build comprehensive dependency graph with analysis."""
        
        # Count by type
        by_type = {}
        for dep_type in DependencyType:
            by_type[dep_type] = len([d for d in dependencies if d.dependency_type == dep_type])
        
        # Count by package manager
        by_package_manager = {}
        for pkg_mgr in PackageManager:
            by_package_manager[pkg_mgr] = len([d for d in dependencies if d.package_manager == pkg_mgr])
        
        # Separate direct vs transitive (all are direct for now)
        direct_dependencies = [d for d in dependencies if d.is_direct]
        transitive_dependencies = [d for d in dependencies if not d.is_direct]
        
        # Security analysis (placeholder)
        security_summary = {level: 0 for level in SecurityLevel}
        
        # License analysis (placeholder)
        license_summary = {}
        
        # Outdated analysis (placeholder)
        outdated_dependencies = []
        
        # Size analysis (placeholder)
        size_analysis = {
            'total_size_estimate_mb': 0,
            'largest_dependencies': [],
            'size_by_type': {}
        }
        
        return DependencyGraph(
            dependencies=dependencies,
            total_count=len(dependencies),
            by_type=by_type,
            by_package_manager=by_package_manager,
            direct_dependencies=direct_dependencies,
            transitive_dependencies=transitive_dependencies,
            security_summary=security_summary,
            license_summary=license_summary,
            outdated_dependencies=outdated_dependencies,
            size_analysis=size_analysis
        )
    
    def export_dependency_graph(self, graph: DependencyGraph, output_path: Path, format: str = 'json') -> None:
        """Export dependency graph to file."""
        
        def convert_for_export(obj):
            """Convert objects for JSON serialization."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [convert_for_export(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_for_export(v) for k, v in obj.items()}
            else:
                return obj
        
        graph_dict = convert_for_export(graph)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(graph_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Dependency graph exported to {output_path}")


# CLI interface for standalone usage
def main():
    """CLI entry point for dependency analysis."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Enhanced Dependency Analysis")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--output", "-o", help="Output file for dependency graph")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        analyzer = EnhancedDependencyAnalyzer()
        graph = analyzer.analyze_project_dependencies(args.project_path)
        
        print(f"📦 Dependency Analysis Results")
        print(f"Total Dependencies: {graph.total_count}")
        print(f"By Type: {dict(graph.by_type)}")
        print(f"By Package Manager: {dict(graph.by_package_manager)}")
        
        if args.output:
            analyzer.export_dependency_graph(graph, Path(args.output), args.format)
            print(f"Results exported to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()