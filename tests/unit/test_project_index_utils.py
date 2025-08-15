"""
Unit tests for Project Index utilities.

Tests for PathUtils, FileUtils, HashUtils, GitUtils, and ProjectUtils
that provide cross-platform file handling and project analysis utilities.
"""

import os
import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, Any

from app.project_index.utils import (
    PathUtils,
    FileUtils, 
    HashUtils,
    GitUtils,
    ProjectUtils,
    validate_project_path,
    sanitize_filename,
    get_file_extension_info
)
from app.models.project_index import FileType


class TestPathUtils:
    """Test PathUtils functionality."""
    
    def test_normalize_path_string(self):
        """Test path normalization with string input."""
        test_path = "/test/path/file.py"
        normalized = PathUtils.normalize_path(test_path)
        assert isinstance(normalized, Path)
        assert str(normalized).endswith("file.py")
    
    def test_normalize_path_pathlib(self):
        """Test path normalization with Path input.""" 
        test_path = Path("/test/path/file.py")
        normalized = PathUtils.normalize_path(test_path)
        assert isinstance(normalized, Path)
        assert str(normalized).endswith("file.py")
    
    def test_get_relative_path(self):
        """Test getting relative path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "subdir" / "file.py"
            
            # Create the file structure
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            
            relative = PathUtils.get_relative_path(file_path, temp_path)
            expected = str(Path("subdir") / "file.py")
            assert relative == expected
    
    def test_get_relative_path_outside_root(self):
        """Test relative path when file is outside root."""
        file_path = "/completely/different/path/file.py"
        root_path = "/test/root"
        
        relative = PathUtils.get_relative_path(file_path, root_path)
        # Should return absolute path when file is not under root
        assert relative == file_path or relative.endswith("file.py")
    
    def test_is_safe_path_safe(self):
        """Test safe path validation for valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            safe_file = temp_path / "safe_file.py"
            safe_file.touch()
            
            assert PathUtils.is_safe_path(safe_file, temp_path) is True
    
    def test_is_safe_path_unsafe(self):
        """Test safe path validation for unsafe path.""" 
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            unsafe_file = "/completely/different/path/file.py"
            
            assert PathUtils.is_safe_path(unsafe_file, temp_path) is False
    
    def test_get_common_prefix(self):
        """Test getting common prefix of paths."""
        paths = [
            "/test/project/src/file1.py",
            "/test/project/src/file2.py", 
            "/test/project/tests/test_file.py"
        ]
        
        common = PathUtils.get_common_prefix(paths)
        assert common is not None
        assert "project" in str(common) or str(common) == "/test" or str(common).endswith("test")
    
    def test_get_common_prefix_no_common(self):
        """Test getting common prefix when there is none."""
        paths = [
            "/completely/different/path1.py",
            "/totally/separate/path2.py"
        ]
        
        # This might return None or a minimal common path depending on OS
        common = PathUtils.get_common_prefix(paths)
        # Just ensure it doesn't crash and returns a Path or None
        assert common is None or isinstance(common, Path)
    
    def test_is_hidden_file_unix_style(self):
        """Test hidden file detection for Unix-style files."""
        assert PathUtils.is_hidden_file(".hidden_file") is True
        assert PathUtils.is_hidden_file("regular_file.py") is False
        assert PathUtils.is_hidden_file(".git") is True
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new" / "nested" / "directory"
            
            created_dir = PathUtils.ensure_directory(new_dir)
            assert created_dir.exists()
            assert created_dir.is_dir()
            assert created_dir == new_dir.resolve()
    
    def test_safe_file_name_basic(self):
        """Test safe filename creation."""
        unsafe_name = "file<>:with|invalid?chars*.py"
        safe_name = PathUtils.safe_file_name(unsafe_name)
        
        assert "<" not in safe_name
        assert ">" not in safe_name
        assert ":" not in safe_name
        assert "|" not in safe_name
        assert "?" not in safe_name
        assert "*" not in safe_name
        assert safe_name.endswith(".py")
    
    def test_safe_file_name_too_long(self):
        """Test safe filename with length limit."""
        long_name = "a" * 300 + ".py"
        safe_name = PathUtils.safe_file_name(long_name, max_length=50)
        
        assert len(safe_name) <= 50
        assert safe_name.endswith(".py")
    
    def test_safe_file_name_empty(self):
        """Test safe filename with empty input."""
        safe_name = PathUtils.safe_file_name("")
        assert safe_name == "unnamed_file"


class TestFileUtils:
    """Test FileUtils functionality."""
    
    def test_get_file_info_existing_file(self):
        """Test getting file info for existing file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test content")
            temp_file.flush()
            
            try:
                info = FileUtils.get_file_info(temp_file.name)
                
                assert info['exists'] is True
                assert info['is_file'] is True
                assert info['is_dir'] is False
                assert info['size'] > 0
                assert 'mtime' in info
                assert 'ctime' in info
                assert 'is_readable' in info
                assert 'is_binary' in info
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_info_nonexistent(self):
        """Test getting file info for non-existent file."""
        info = FileUtils.get_file_info("/nonexistent/file.py")
        assert info['exists'] is False
        assert 'error' in info
    
    def test_is_binary_file_text(self):
        """Test binary file detection for text file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("This is plain text content")
            temp_file.flush()
            
            try:
                is_binary = FileUtils.is_binary_file(temp_file.name)
                assert is_binary is False
            finally:
                os.unlink(temp_file.name)
    
    def test_is_binary_file_with_null_bytes(self):
        """Test binary file detection for file with null bytes."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            temp_file.write(b"Binary content with \x00 null bytes")
            temp_file.flush()
            
            try:
                is_binary = FileUtils.is_binary_file(temp_file.name)
                assert is_binary is True
            finally:
                os.unlink(temp_file.name)
    
    def test_is_generated_file_by_name(self):
        """Test generated file detection by filename."""
        assert FileUtils.is_generated_file("file.generated.py") is True
        assert FileUtils.is_generated_file("test.min.js") is True
        assert FileUtils.is_generated_file("bundle.bundle.css") is True
        assert FileUtils.is_generated_file("regular_file.py") is False
    
    def test_is_generated_file_by_content(self):
        """Test generated file detection by content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("// This file was automatically generated\n")
            temp_file.write("// DO NOT EDIT\n")
            temp_file.write("export const config = {};")
            temp_file.flush()
            
            try:
                is_generated = FileUtils.is_generated_file(temp_file.name)
                assert is_generated is True
            finally:
                os.unlink(temp_file.name)
    
    def test_classify_file_type_source(self):
        """Test file type classification for source files."""
        assert FileUtils.classify_file_type("main.py") == FileType.SOURCE
        assert FileUtils.classify_file_type("app.js") == FileType.SOURCE
        assert FileUtils.classify_file_type("component.tsx") == FileType.SOURCE
    
    def test_classify_file_type_test(self):
        """Test file type classification for test files."""
        assert FileUtils.classify_file_type("test_main.py") == FileType.TEST
        assert FileUtils.classify_file_type("main.test.js") == FileType.TEST
        assert FileUtils.classify_file_type("spec_component.py") == FileType.TEST
    
    def test_classify_file_type_config(self):
        """Test file type classification for config files."""
        assert FileUtils.classify_file_type("config.json") == FileType.CONFIG
        assert FileUtils.classify_file_type("settings.yaml") == FileType.CONFIG
        assert FileUtils.classify_file_type("package.json") == FileType.CONFIG
    
    def test_classify_file_type_documentation(self):
        """Test file type classification for documentation files."""
        assert FileUtils.classify_file_type("README.md") == FileType.DOCUMENTATION
        assert FileUtils.classify_file_type("docs.rst") == FileType.DOCUMENTATION
        assert FileUtils.classify_file_type("manual.txt") == FileType.DOCUMENTATION
    
    def test_classify_file_type_build(self):
        """Test file type classification for build files."""
        assert FileUtils.classify_file_type("Rakefile") == FileType.BUILD       # In build_names only
        assert FileUtils.classify_file_type("sbt") == FileType.BUILD           # In build_names, not in config
        # Note: Many build files overlap with config or source extensions,
        # reflecting the reality that build configuration is often config
    
    def test_read_file_safe_success(self):
        """Test safe file reading success case."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            test_content = "Test file content\nwith multiple lines"
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                content = FileUtils.read_file_safe(temp_file.name)
                assert content == test_content
            finally:
                os.unlink(temp_file.name)
    
    def test_read_file_safe_too_large(self):
        """Test safe file reading with size limit."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("x" * 100)  # Small file
            temp_file.flush()
            
            try:
                # Set very small max size
                content = FileUtils.read_file_safe(temp_file.name, max_size=50)
                assert content is None  # Should be rejected as too large
            finally:
                os.unlink(temp_file.name)
    
    def test_read_file_safe_encoding_fallback(self):
        """Test safe file reading with encoding fallback."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            # Write some content that might have encoding issues
            temp_file.write("Test content".encode('utf-8'))
            temp_file.flush()
            
            try:
                content = FileUtils.read_file_safe(
                    temp_file.name, 
                    encoding='utf-8',
                    fallback_encodings=['latin1']
                )
                assert content is not None
                assert "Test content" in content
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_lines(self):
        """Test getting file lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            test_lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
            temp_file.write("\n".join(test_lines))
            temp_file.flush()
            
            try:
                # Get all lines
                lines = FileUtils.get_file_lines(temp_file.name)
                assert len(lines) == 4
                assert lines[0] == "Line 1"
                assert lines[3] == "Line 4"
                
                # Get limited lines
                lines = FileUtils.get_file_lines(temp_file.name, max_lines=2)
                assert len(lines) == 2
                assert lines[0] == "Line 1"
                assert lines[1] == "Line 2"
            finally:
                os.unlink(temp_file.name)


class TestHashUtils:
    """Test HashUtils functionality."""
    
    def test_hash_file(self):
        """Test file hashing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            test_content = "Test content for hashing"
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                file_hash = HashUtils.hash_file(temp_file.name)
                assert len(file_hash) == 64  # SHA256 hex length
                assert file_hash != ""
                
                # Test with different algorithm
                md5_hash = HashUtils.hash_file(temp_file.name, algorithm='md5')
                assert len(md5_hash) == 32  # MD5 hex length
                assert md5_hash != file_hash
            finally:
                os.unlink(temp_file.name)
    
    def test_hash_string(self):
        """Test string hashing."""
        test_string = "Test string for hashing"
        string_hash = HashUtils.hash_string(test_string)
        
        assert len(string_hash) == 64  # SHA256 hex length
        assert string_hash != ""
        
        # Same string should produce same hash
        same_hash = HashUtils.hash_string(test_string)
        assert string_hash == same_hash
        
        # Different string should produce different hash
        different_hash = HashUtils.hash_string("Different string")
        assert string_hash != different_hash
    
    def test_hash_directory_structure(self):
        """Test directory structure hashing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directory structure
            (temp_path / "subdir").mkdir()
            (temp_path / "file1.txt").write_text("Content 1")
            (temp_path / "subdir" / "file2.txt").write_text("Content 2")
            
            # Hash without content
            hash1 = HashUtils.hash_directory_structure(temp_path, include_content=False)
            assert len(hash1) == 64
            assert hash1 != ""
            
            # Hash with content
            hash2 = HashUtils.hash_directory_structure(temp_path, include_content=True)
            assert len(hash2) == 64
            assert hash2 != ""
            assert hash1 != hash2  # Should be different
    
    def test_hash_directory_nonexistent(self):
        """Test directory hashing for non-existent directory."""
        hash_result = HashUtils.hash_directory_structure("/nonexistent/directory")
        assert hash_result == ""


class TestGitUtils:
    """Test GitUtils functionality."""
    
    @patch('subprocess.run')
    def test_is_git_repository_true(self, mock_run):
        """Test Git repository detection for valid repo."""
        mock_run.return_value.returncode = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake .git directory
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            
            is_git = GitUtils.is_git_repository(temp_dir)
            assert is_git is True
    
    @patch('subprocess.run')
    def test_is_git_repository_false(self, mock_run):
        """Test Git repository detection for non-repo."""
        mock_run.side_effect = FileNotFoundError()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            is_git = GitUtils.is_git_repository(temp_dir)
            assert is_git is False
    
    @patch('subprocess.run')
    def test_get_git_info(self, mock_run):
        """Test getting Git repository info."""
        # Mock successful git commands
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="https://github.com/user/repo.git"),  # remote url
            MagicMock(returncode=0, stdout="main"),  # branch
            MagicMock(returncode=0, stdout="abc123def456"),  # commit hash
            MagicMock(returncode=0, stdout="abc123d"),  # short hash
            MagicMock(returncode=0, stdout="/path/to/repo"),  # root path
            MagicMock(returncode=0, stdout=""),  # status (clean)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake .git directory
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            
            info = GitUtils.get_git_info(temp_dir)
            
            assert info['remote_url'] == "https://github.com/user/repo.git"
            assert info['branch'] == "main"
            assert info['commit_hash'] == "abc123def456"
            assert info['commit_hash_short'] == "abc123d"
            assert info['root_path'] == "/path/to/repo"
            assert info['is_clean'] is True
    
    @patch('subprocess.run')
    def test_get_file_git_info(self, mock_run):
        """Test getting Git info for specific file."""
        # Mock git commands for file info
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123def456"),  # last commit
            MagicMock(returncode=0, stdout="2023-01-01 12:00:00 +0000"),  # commit date
            MagicMock(returncode=0, stdout="M  file.py"),  # status
            MagicMock(returncode=0, stdout="file.py"),  # tracked
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake .git directory and file
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            test_file = Path(temp_dir) / "file.py"
            test_file.touch()
            
            info = GitUtils.get_file_git_info(test_file)
            
            assert info['last_commit'] == "abc123def456"
            assert info['last_commit_date'] == "2023-01-01 12:00:00 +0000"
            assert info['status'] == "M  file.py"
            assert info['is_tracked'] is True


class TestProjectUtils:
    """Test ProjectUtils functionality."""
    
    def test_detect_project_type_python(self):
        """Test Python project type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "setup.py").touch()
            
            project_type = ProjectUtils.detect_project_type(temp_path)
            assert project_type == "python"
    
    def test_detect_project_type_node(self):
        """Test Node.js project type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "package.json").touch()
            
            project_type = ProjectUtils.detect_project_type(temp_path)
            assert project_type == "node"
    
    def test_detect_project_type_go(self):
        """Test Go project type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "go.mod").touch()
            
            project_type = ProjectUtils.detect_project_type(temp_path)
            assert project_type == "go"
    
    def test_detect_project_type_unknown(self):
        """Test project type detection for unknown type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_type = ProjectUtils.detect_project_type(temp_dir)
            assert project_type is None
    
    def test_get_project_metadata_package_json(self):
        """Test extracting metadata from package.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            package_json = temp_path / "package.json"
            
            package_data = {
                "name": "test-project",
                "version": "1.0.0",
                "description": "Test project description",
                "author": "Test Author"
            }
            
            with open(package_json, 'w') as f:
                import json
                json.dump(package_data, f)
            
            metadata = ProjectUtils.get_project_metadata(temp_path)
            
            assert metadata['name'] == "test-project"
            assert metadata['version'] == "1.0.0"
            assert metadata['description'] == "Test project description"
            assert metadata['author'] == "Test Author"
    
    def test_get_project_structure(self):
        """Test getting project structure overview."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test structure
            (temp_path / "src").mkdir()
            (temp_path / "src" / "main.py").touch()
            (temp_path / "tests").mkdir()
            (temp_path / "tests" / "test_main.py").touch()
            (temp_path / "README.md").touch()
            
            structure = ProjectUtils.get_project_structure(temp_path, max_depth=2)
            
            assert structure['type'] == 'directory'
            assert structure['name'] == temp_path.name
            assert 'children' in structure
            
            # Check that src and tests directories are present
            children_names = list(structure['children'].keys())
            assert 'src' in children_names
            assert 'tests' in children_names
            assert 'README.md' in children_names


class TestValidationFunctions:
    """Test validation and utility functions."""
    
    def test_validate_project_path_valid(self):
        """Test project path validation for valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid, error = validate_project_path(temp_dir)
            assert is_valid is True
            assert error is None
    
    def test_validate_project_path_nonexistent(self):
        """Test project path validation for non-existent path."""
        is_valid, error = validate_project_path("/nonexistent/directory")
        assert is_valid is False
        assert error is not None
        assert "does not exist" in error
    
    def test_validate_project_path_file_not_directory(self):
        """Test project path validation for file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            is_valid, error = validate_project_path(temp_file.name)
            assert is_valid is False
            assert error is not None
            assert "not a directory" in error
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        unsafe_name = "file<>:with|invalid?chars*.py"
        safe_name = sanitize_filename(unsafe_name)
        
        assert "<" not in safe_name
        assert ">" not in safe_name
        assert "|" not in safe_name
        assert "?" not in safe_name
        assert "*" not in safe_name
    
    def test_get_file_extension_info_python(self):
        """Test file extension info for Python file."""
        info = get_file_extension_info(".py")
        
        assert info['extension'] == ".py"
        assert info['language'] == "python"
        assert info['category'] == "source"
        assert info['is_source'] is True
        assert info['is_config'] is False
        assert info['is_documentation'] is False
    
    def test_get_file_extension_info_json(self):
        """Test file extension info for JSON file."""
        info = get_file_extension_info("json")  # Without dot
        
        assert info['extension'] == ".json"
        assert info['language'] is None
        assert info['category'] == "config"
        assert info['is_source'] is False
        assert info['is_config'] is True
        assert info['is_documentation'] is False
    
    def test_get_file_extension_info_markdown(self):
        """Test file extension info for Markdown file."""
        info = get_file_extension_info(".md")
        
        assert info['extension'] == ".md"
        assert info['language'] is None
        assert info['category'] == "documentation"
        assert info['is_source'] is False
        assert info['is_config'] is False
        assert info['is_documentation'] is True
    
    def test_get_file_extension_info_unknown(self):
        """Test file extension info for unknown extension."""
        info = get_file_extension_info(".xyz")
        
        assert info['extension'] == ".xyz"
        assert info['language'] is None
        assert info['category'] == "other"
        assert info['is_source'] is False
        assert info['is_config'] is False
        assert info['is_documentation'] is False