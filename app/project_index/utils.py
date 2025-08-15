"""
Utility Functions for LeanVibe Agent Hive 2.0 Project Index

Cross-platform file handling, path operations, hashing utilities,
and helper functions for project analysis and code intelligence.
"""

import hashlib
import mimetypes
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from urllib.parse import urlparse

import structlog

from ..models.project_index import FileType

logger = structlog.get_logger()


class PathUtils:
    """Cross-platform path utilities for file operations."""
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """
        Normalize path for cross-platform compatibility.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized Path object
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Resolve symlinks and relative paths
        try:
            return path.resolve()
        except (OSError, RuntimeError):
            # Fallback for broken symlinks or very long paths
            return path.absolute()
    
    @staticmethod
    def get_relative_path(file_path: Union[str, Path], root_path: Union[str, Path]) -> str:
        """
        Get relative path from root to file.
        
        Args:
            file_path: File path
            root_path: Root directory path
            
        Returns:
            Relative path string
        """
        file_path = PathUtils.normalize_path(file_path)
        root_path = PathUtils.normalize_path(root_path)
        
        try:
            return str(file_path.relative_to(root_path))
        except ValueError:
            # File is not under root, return absolute path
            return str(file_path)
    
    @staticmethod
    def is_safe_path(path: Union[str, Path], root_path: Union[str, Path]) -> bool:
        """
        Check if path is safe (within root directory).
        
        Args:
            path: Path to check
            root_path: Root directory
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            file_path = PathUtils.normalize_path(path)
            root_path = PathUtils.normalize_path(root_path)
            
            # Check if file is under root
            file_path.relative_to(root_path)
            return True
        except (ValueError, OSError):
            return False
    
    @staticmethod
    def get_common_prefix(paths: List[Union[str, Path]]) -> Optional[Path]:
        """
        Get common prefix path for a list of paths.
        
        Args:
            paths: List of paths
            
        Returns:
            Common prefix path or None if no common prefix
        """
        if not paths:
            return None
        
        normalized_paths = [PathUtils.normalize_path(p) for p in paths]
        
        try:
            # Use os.path.commonpath for cross-platform compatibility
            common = os.path.commonpath([str(p) for p in normalized_paths])
            return Path(common)
        except ValueError:
            # No common path
            return None
    
    @staticmethod
    def is_hidden_file(path: Union[str, Path]) -> bool:
        """
        Check if file/directory is hidden.
        
        Args:
            path: Path to check
            
        Returns:
            True if hidden, False otherwise
        """
        path = Path(path)
        
        # Unix-style hidden files (start with .)
        if path.name.startswith('.'):
            return True
        
        # Windows hidden files
        if platform.system() == 'Windows':
            try:
                attrs = os.stat(path).st_file_attributes
                return bool(attrs & stat.FILE_ATTRIBUTE_HIDDEN)
            except (OSError, AttributeError):
                pass
        
        return False
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = PathUtils.normalize_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_file_name(name: str, max_length: int = 255) -> str:
        """
        Create safe filename by removing/replacing invalid characters.
        
        Args:
            name: Original filename
            max_length: Maximum filename length
            
        Returns:
            Safe filename
        """
        # Remove or replace invalid characters
        invalid_chars = r'[<>:"/\\|?*\0]'
        safe_name = re.sub(invalid_chars, '_', name)
        
        # Remove control characters
        safe_name = ''.join(c for c in safe_name if ord(c) >= 32)
        
        # Trim whitespace
        safe_name = safe_name.strip()
        
        # Ensure not empty
        if not safe_name:
            safe_name = 'unnamed_file'
        
        # Truncate if too long
        if len(safe_name) > max_length:
            # Try to preserve extension
            if '.' in safe_name:
                name_part, ext = safe_name.rsplit('.', 1)
                max_name_length = max_length - len(ext) - 1
                if max_name_length > 0:
                    safe_name = name_part[:max_name_length] + '.' + ext
                else:
                    safe_name = safe_name[:max_length]
            else:
                safe_name = safe_name[:max_length]
        
        return safe_name


class FileUtils:
    """File system utilities for analysis operations."""
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        file_path = PathUtils.normalize_path(file_path)
        
        try:
            stat_result = file_path.stat()
            
            info = {
                'exists': file_path.exists(),
                'is_file': file_path.is_file(),
                'is_dir': file_path.is_dir(),
                'is_symlink': file_path.is_symlink(),
                'size': stat_result.st_size,
                'mtime': stat_result.st_mtime,
                'ctime': stat_result.st_ctime,
                'mode': stat_result.st_mode,
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK),
                'is_executable': os.access(file_path, os.X_OK),
                'is_hidden': PathUtils.is_hidden_file(file_path)
            }
            
            # Detect if binary
            info['is_binary'] = FileUtils.is_binary_file(file_path)
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            info['mime_type'] = mime_type
            
            # Detect if generated
            info['is_generated'] = FileUtils.is_generated_file(file_path)
            
        except (OSError, PermissionError) as e:
            logger.warning("Failed to get file info", file_path=str(file_path), error=str(e))
            info = {
                'exists': False,
                'error': str(e)
            }
        
        return info
    
    @staticmethod
    def is_binary_file(file_path: Union[str, Path], chunk_size: int = 8192) -> bool:
        """
        Detect if file is binary by checking for null bytes.
        
        Args:
            file_path: Path to file
            chunk_size: Size of chunk to read for detection
            
        Returns:
            True if binary, False if text
        """
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                
            # Check for null bytes (common in binary files)
            if b'\0' in chunk:
                return True
            
            # Check percentage of control characters
            if chunk:
                control_chars = sum(1 for byte in chunk if byte < 32 and byte not in [9, 10, 13])
                control_ratio = control_chars / len(chunk)
                
                # If more than 10% control characters, likely binary
                return control_ratio > 0.1
            
            return False
            
        except (OSError, PermissionError, UnicodeDecodeError):
            # If we can't read it, assume binary
            return True
    
    @staticmethod
    def is_generated_file(file_path: Union[str, Path]) -> bool:
        """
        Detect if file is likely generated/auto-generated.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if likely generated, False otherwise
        """
        file_path = Path(file_path)
        
        # Check filename patterns
        generated_patterns = [
            r'.*\.generated\.',
            r'.*\.g\.',
            r'.*\.min\.',
            r'.*\.bundle\.',
            r'.*\.build\.',
            r'.*\.dist\.',
            r'.*\.compiled\.',
            r'.*_pb2\.py$',  # Protocol buffer generated files
            r'.*\.pb\.go$',   # Protocol buffer Go files
            r'.*\.pb\.h$',    # Protocol buffer C++ headers
            r'.*\.pb\.cc$',   # Protocol buffer C++ source
        ]
        
        filename = file_path.name.lower()
        for pattern in generated_patterns:
            if re.match(pattern, filename):
                return True
        
        # Check for generation markers in content (first few lines)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 10 lines
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    
                    line_lower = line.lower()
                    
                    # Common generation markers
                    generation_markers = [
                        'auto-generated',
                        'automatically generated',
                        'generated by',
                        'do not edit',
                        'autogenerated',
                        '@generated',
                        'this file was generated',
                        'code generated by'
                    ]
                    
                    for marker in generation_markers:
                        if marker in line_lower:
                            return True
            
        except (OSError, PermissionError, UnicodeDecodeError):
            pass
        
        return False
    
    @staticmethod
    def classify_file_type(file_path: Union[str, Path]) -> FileType:
        """
        Classify file into FileType categories.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileType enum value
        """
        file_path = Path(file_path)
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        # Source code files
        source_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx',
            '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.clj', '.hs', '.ml', '.fs', '.vb', '.pas', '.pl', '.r', '.m', '.mm'
        }
        
        # Test files
        test_patterns = [
            r'.*test.*',
            r'.*spec.*',
            r'.*_test\.',
            r'.*\.test\.',
            r'.*_spec\.',
            r'.*\.spec\.',
            r'test_.*',
            r'spec_.*'
        ]
        
        # Configuration files
        config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.properties',
            '.env', '.config', '.settings', '.plist', '.xml'
        }
        config_names = {
            'makefile', 'dockerfile', 'vagrantfile', 'procfile', 'cmakelists.txt',
            '.gitignore', '.gitconfig', '.dockerignore', '.npmignore', '.eslintrc',
            'package.json', 'composer.json', 'cargo.toml', 'pyproject.toml', 'setup.py',
            'requirements.txt', 'pipfile', 'poetry.lock', 'yarn.lock', 'pom.xml',
            'build.gradle', 'webpack.config.js', 'rollup.config.js', 'tsconfig.json'
        }
        
        # Documentation files
        doc_extensions = {
            '.md', '.rst', '.txt', '.adoc', '.tex', '.rtf', '.pdf', '.doc', '.docx',
            '.odt', '.html', '.htm'
        }
        doc_names = {
            'readme', 'license', 'changelog', 'authors', 'contributors', 'todo',
            'news', 'history', 'install', 'copying', 'notice'
        }
        
        # Build files
        build_extensions = {
            '.make', '.mk', '.cmake', '.gradle', '.sbt', '.bazel', '.bzl'
        }
        build_names = {
            'makefile', 'cmakelists.txt', 'build.xml', 'build.gradle', 'sbt',
            'rakefile', 'gulpfile.js', 'gruntfile.js', 'webpack.config.js'
        }
        
        # Check test files first (more specific)
        for pattern in test_patterns:
            if re.match(pattern, filename):
                return FileType.TEST
        
        # Check source files
        if extension in source_extensions:
            return FileType.SOURCE
        
        # Check configuration files
        if extension in config_extensions or filename in config_names:
            return FileType.CONFIG
        
        # Check documentation files
        if extension in doc_extensions or any(name in filename for name in doc_names):
            return FileType.DOCUMENTATION
        
        # Check build files
        if extension in build_extensions or filename in build_names:
            return FileType.BUILD
        
        # Default to other
        return FileType.OTHER
    
    @staticmethod
    def read_file_safe(
        file_path: Union[str, Path], 
        encoding: str = 'utf-8',
        max_size: int = 10 * 1024 * 1024,  # 10MB
        fallback_encodings: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Safely read file content with encoding detection and size limits.
        
        Args:
            file_path: Path to file
            encoding: Primary encoding to try
            max_size: Maximum file size to read
            fallback_encodings: List of fallback encodings to try
            
        Returns:
            File content or None if reading failed
        """
        file_path = Path(file_path)
        
        # Check file size
        try:
            if file_path.stat().st_size > max_size:
                logger.warning("File too large to read", 
                              file_path=str(file_path), 
                              size=file_path.stat().st_size)
                return None
        except OSError:
            return None
        
        # Try encodings
        encodings_to_try = [encoding]
        if fallback_encodings:
            encodings_to_try.extend(fallback_encodings)
        else:
            encodings_to_try.extend(['latin1', 'cp1252', 'iso-8859-1'])
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    return f.read()
            except (OSError, UnicodeDecodeError):
                continue
        
        logger.warning("Failed to read file with any encoding", file_path=str(file_path))
        return None
    
    @staticmethod
    def get_file_lines(
        file_path: Union[str, Path], 
        max_lines: Optional[int] = None,
        encoding: str = 'utf-8'
    ) -> List[str]:
        """
        Get file lines with optional limit.
        
        Args:
            file_path: Path to file
            max_lines: Maximum number of lines to read
            encoding: File encoding
            
        Returns:
            List of file lines
        """
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line.rstrip('\n\r'))
                    return lines
                else:
                    return [line.rstrip('\n\r') for line in f]
        except (OSError, UnicodeDecodeError):
            return []


class HashUtils:
    """Hashing utilities for file content and data integrity."""
    
    @staticmethod
    def hash_file(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """
        Calculate hash of file content.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            
        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except (OSError, ValueError) as e:
            logger.error("Failed to hash file", file_path=str(file_path), error=str(e))
            return ''
    
    @staticmethod
    def hash_string(content: str, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of string content.
        
        Args:
            content: String content to hash
            algorithm: Hash algorithm
            
        Returns:
            Hex digest of content hash
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def hash_directory_structure(
        directory: Union[str, Path], 
        include_content: bool = False
    ) -> str:
        """
        Calculate hash of directory structure.
        
        Args:
            directory: Directory path
            include_content: Whether to include file content in hash
            
        Returns:
            Hash of directory structure
        """
        directory = PathUtils.normalize_path(directory)
        
        if not directory.is_dir():
            return ''
        
        hash_obj = hashlib.sha256()
        
        try:
            # Walk directory in sorted order for consistent hashing
            for root, dirs, files in os.walk(directory):
                # Sort for consistency
                dirs.sort()
                files.sort()
                
                # Add directory path
                rel_root = os.path.relpath(root, directory)
                hash_obj.update(rel_root.encode('utf-8'))
                
                for filename in files:
                    file_path = Path(root) / filename
                    
                    # Add filename
                    hash_obj.update(filename.encode('utf-8'))
                    
                    if include_content and file_path.is_file():
                        # Add file content hash
                        try:
                            file_hash = HashUtils.hash_file(file_path)
                            hash_obj.update(file_hash.encode('utf-8'))
                        except Exception:
                            # Skip files we can't read
                            continue
            
            return hash_obj.hexdigest()
            
        except OSError as e:
            logger.error("Failed to hash directory", directory=str(directory), error=str(e))
            return ''


class GitUtils:
    """Git repository utilities for version control integration."""
    
    @staticmethod
    def is_git_repository(path: Union[str, Path]) -> bool:
        """
        Check if directory is a Git repository.
        
        Args:
            path: Directory path to check
            
        Returns:
            True if Git repository, False otherwise
        """
        path = PathUtils.normalize_path(path)
        
        # Check for .git directory
        git_dir = path / '.git'
        if git_dir.exists():
            return True
        
        # Check if we're in a worktree
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def get_git_info(path: Union[str, Path]) -> Dict[str, Optional[str]]:
        """
        Get Git repository information.
        
        Args:
            path: Repository path
            
        Returns:
            Dictionary with Git information
        """
        path = PathUtils.normalize_path(path)
        
        if not GitUtils.is_git_repository(path):
            return {}
        
        info = {}
        
        def run_git_command(cmd: List[str]) -> Optional[str]:
            try:
                result = subprocess.run(
                    ['git'] + cmd,
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.stdout.strip() if result.returncode == 0 else None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None
        
        # Get repository URL
        info['remote_url'] = run_git_command(['remote', 'get-url', 'origin'])
        
        # Get current branch
        info['branch'] = run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
        
        # Get current commit hash
        info['commit_hash'] = run_git_command(['rev-parse', 'HEAD'])
        
        # Get short commit hash
        info['commit_hash_short'] = run_git_command(['rev-parse', '--short', 'HEAD'])
        
        # Get repository root
        info['root_path'] = run_git_command(['rev-parse', '--show-toplevel'])
        
        # Check if working directory is clean
        status = run_git_command(['status', '--porcelain'])
        info['is_clean'] = status == '' if status is not None else None
        
        return info
    
    @staticmethod
    def get_file_git_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get Git information for a specific file.
        
        Args:
            file_path: File path
            
        Returns:
            Dictionary with file Git information
        """
        file_path = PathUtils.normalize_path(file_path)
        
        if not file_path.exists():
            return {}
        
        # Find repository root
        repo_path = file_path.parent
        while repo_path != repo_path.parent:
            if GitUtils.is_git_repository(repo_path):
                break
            repo_path = repo_path.parent
        else:
            return {}
        
        try:
            relative_path = file_path.relative_to(repo_path)
        except ValueError:
            return {}
        
        def run_git_command(cmd: List[str]) -> Optional[str]:
            try:
                result = subprocess.run(
                    ['git'] + cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.stdout.strip() if result.returncode == 0 else None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None
        
        info = {}
        
        # Get last commit hash for file
        info['last_commit'] = run_git_command(['log', '-1', '--format=%H', '--', str(relative_path)])
        
        # Get last commit date
        info['last_commit_date'] = run_git_command(['log', '-1', '--format=%ci', '--', str(relative_path)])
        
        # Get file status
        status = run_git_command(['status', '--porcelain', '--', str(relative_path)])
        info['status'] = status.strip() if status else 'unmodified'
        
        # Check if file is tracked
        tracked = run_git_command(['ls-files', '--', str(relative_path)])
        info['is_tracked'] = bool(tracked)
        
        return info


class ProjectUtils:
    """Project-level utilities for analysis and management."""
    
    @staticmethod
    def detect_project_type(path: Union[str, Path]) -> Optional[str]:
        """
        Detect project type based on files and structure.
        
        Args:
            path: Project root path
            
        Returns:
            Detected project type or None
        """
        path = PathUtils.normalize_path(path)
        
        if not path.is_dir():
            return None
        
        # Check for specific project files
        project_indicators = {
            'python': ['setup.py', 'pyproject.toml', 'requirements.txt', 'Pipfile', 'poetry.lock'],
            'node': ['package.json', 'yarn.lock', 'package-lock.json'],
            'java': ['pom.xml', 'build.gradle', 'gradle.properties'],
            'go': ['go.mod', 'go.sum'],
            'rust': ['Cargo.toml', 'Cargo.lock'],
            'dotnet': ['*.sln', '*.csproj', '*.fsproj', '*.vbproj'],
            'ruby': ['Gemfile', 'Gemfile.lock', '*.gemspec'],
            'php': ['composer.json', 'composer.lock'],
            'swift': ['Package.swift', '*.xcodeproj', '*.xcworkspace'],
            'docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml']
        }
        
        detected_types = []
        
        for project_type, indicators in project_indicators.items():
            for indicator in indicators:
                if '*' in indicator:
                    # Glob pattern
                    if list(path.glob(indicator)):
                        detected_types.append(project_type)
                        break
                else:
                    # Exact file
                    if (path / indicator).exists():
                        detected_types.append(project_type)
                        break
        
        # Return most likely type (could be enhanced with more logic)
        return detected_types[0] if detected_types else None
    
    @staticmethod
    def get_project_metadata(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract project metadata from common project files.
        
        Args:
            path: Project root path
            
        Returns:
            Dictionary with project metadata
        """
        path = PathUtils.normalize_path(path)
        metadata = {}
        
        # Python projects
        if (path / 'pyproject.toml').exists():
            try:
                import tomllib
                with open(path / 'pyproject.toml', 'rb') as f:
                    data = tomllib.load(f)
                    if 'project' in data:
                        metadata.update({
                            'name': data['project'].get('name'),
                            'version': data['project'].get('version'),
                            'description': data['project'].get('description'),
                            'authors': data['project'].get('authors', [])
                        })
            except (ImportError, Exception):
                pass
        
        elif (path / 'setup.py').exists():
            # Could parse setup.py but it's complex and potentially unsafe
            metadata['has_setup_py'] = True
        
        # Node.js projects
        if (path / 'package.json').exists():
            try:
                import json
                with open(path / 'package.json', 'r') as f:
                    data = json.load(f)
                    metadata.update({
                        'name': data.get('name'),
                        'version': data.get('version'),
                        'description': data.get('description'),
                        'author': data.get('author'),
                        'license': data.get('license')
                    })
            except (json.JSONDecodeError, OSError):
                pass
        
        # Git information
        if GitUtils.is_git_repository(path):
            git_info = GitUtils.get_git_info(path)
            metadata['git'] = git_info
        
        # Project type
        metadata['project_type'] = ProjectUtils.detect_project_type(path)
        
        return metadata
    
    @staticmethod
    def get_project_structure(
        path: Union[str, Path], 
        max_depth: int = 3,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        Get overview of project directory structure.
        
        Args:
            path: Project root path
            max_depth: Maximum depth to traverse
            include_hidden: Whether to include hidden files/directories
            
        Returns:
            Dictionary representing project structure
        """
        path = PathUtils.normalize_path(path)
        
        def build_tree(current_path: Path, current_depth: int) -> Dict[str, Any]:
            if current_depth > max_depth:
                return {}
            
            tree = {
                'type': 'directory' if current_path.is_dir() else 'file',
                'name': current_path.name,
                'path': str(current_path.relative_to(path))
            }
            
            if current_path.is_dir():
                children = {}
                try:
                    for child in current_path.iterdir():
                        if not include_hidden and PathUtils.is_hidden_file(child):
                            continue
                        
                        child_tree = build_tree(child, current_depth + 1)
                        if child_tree:
                            children[child.name] = child_tree
                    
                    tree['children'] = children
                except (OSError, PermissionError):
                    tree['error'] = 'Permission denied'
            else:
                # Add file info
                try:
                    stat_result = current_path.stat()
                    tree['size'] = stat_result.st_size
                    tree['mtime'] = stat_result.st_mtime
                except (OSError, PermissionError):
                    pass
            
            return tree
        
        return build_tree(path, 0)


# Validation utilities
def validate_project_path(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate project path for analysis.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = PathUtils.normalize_path(path)
        
        if not path.exists():
            return False, f"Path does not exist: {path}"
        
        if not path.is_dir():
            return False, f"Path is not a directory: {path}"
        
        if not os.access(path, os.R_OK):
            return False, f"Path is not readable: {path}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem operations."""
    return PathUtils.safe_file_name(filename)


def get_file_extension_info(extension: str) -> Dict[str, Any]:
    """
    Get information about a file extension.
    
    Args:
        extension: File extension (with or without dot)
        
    Returns:
        Dictionary with extension information
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    extension = extension.lower()
    
    # Language mapping
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    # Category mapping
    category_map = {
        '.py': 'source',
        '.js': 'source',
        '.ts': 'source',
        '.json': 'config',
        '.yaml': 'config',
        '.yml': 'config',
        '.toml': 'config',
        '.md': 'documentation',
        '.rst': 'documentation',
        '.txt': 'documentation'
    }
    
    return {
        'extension': extension,
        'language': language_map.get(extension),
        'category': category_map.get(extension, 'other'),
        'is_source': extension in language_map,
        'is_config': category_map.get(extension) == 'config',
        'is_documentation': category_map.get(extension) == 'documentation'
    }