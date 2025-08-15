"""
Unit tests for Project Index Pydantic models.

Tests for configuration models, analysis result models, and utility functions
that provide type safety and validation for the project indexing system.
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List

from app.project_index.models import (
    # Configuration models
    AnalysisConfiguration,
    ProjectIndexConfig,
    
    # Analysis result models
    ComplexityMetrics,
    CodeStructure,
    DependencyResult,
    FileAnalysisResult,
    AnalysisResult,
    ProjectStatistics,
    
    # Context optimization models
    ContextRecommendation,
    ContextClusterInfo,
    
    # Monitoring and change models
    FileChangeInfo,
    AnalysisProgress,
    AnalysisError,
    ValidationResult,
    
    # Utility functions
    create_default_analysis_config,
    create_default_project_config,
    validate_file_path,
    calculate_file_metrics
)


class TestAnalysisConfiguration:
    """Test AnalysisConfiguration model."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = AnalysisConfiguration()
        
        assert config.enabled_languages == ['python', 'javascript', 'typescript', 'json']
        assert config.parse_ast is True
        assert config.extract_dependencies is True
        assert config.calculate_complexity is True
        assert config.analyze_docstrings is True
        assert config.max_file_size_mb == 10
        assert config.max_line_count == 50000
        assert config.timeout_seconds == 30
        assert config.confidence_threshold == 0.7
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = AnalysisConfiguration(
            enabled_languages=['python', 'rust'],
            parse_ast=False,
            max_file_size_mb=5,
            confidence_threshold=0.8
        )
        
        assert config.enabled_languages == ['python', 'rust']
        assert config.parse_ast is False
        assert config.max_file_size_mb == 5
        assert config.confidence_threshold == 0.8
    
    def test_file_size_validation(self):
        """Test file size validation."""
        # Valid values
        config = AnalysisConfiguration(max_file_size_mb=5)
        assert config.max_file_size_mb == 5
        
        # Invalid values
        with pytest.raises(ValueError, match="File size must be between 1 and 100 MB"):
            AnalysisConfiguration(max_file_size_mb=0)
        
        with pytest.raises(ValueError, match="File size must be between 1 and 100 MB"):
            AnalysisConfiguration(max_file_size_mb=150)
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Valid values
        config = AnalysisConfiguration(confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5
        
        # Invalid values
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            AnalysisConfiguration(confidence_threshold=-0.1)
        
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            AnalysisConfiguration(confidence_threshold=1.1)


class TestProjectIndexConfig:
    """Test ProjectIndexConfig model."""
    
    def test_default_config(self):
        """Test default project index configuration."""
        config = ProjectIndexConfig()
        
        assert isinstance(config.analysis_config, AnalysisConfiguration)
        assert config.max_concurrent_analyses == 4
        assert config.analysis_batch_size == 50
        assert config.cache_enabled is True
        assert config.context_optimization_enabled is True
        assert config.file_monitoring_enabled is True
    
    def test_custom_config(self):
        """Test custom project index configuration."""
        analysis_config = AnalysisConfiguration(enabled_languages=['python'])
        config = ProjectIndexConfig(
            analysis_config=analysis_config,
            max_concurrent_analyses=8,
            cache_enabled=False
        )
        
        assert config.analysis_config == analysis_config
        assert config.max_concurrent_analyses == 8
        assert config.cache_enabled is False
    
    def test_concurrency_validation(self):
        """Test concurrency validation."""
        # Valid values
        config = ProjectIndexConfig(max_concurrent_analyses=10)
        assert config.max_concurrent_analyses == 10
        
        # Invalid values
        with pytest.raises(ValueError, match="Concurrent analyses must be between 1 and 20"):
            ProjectIndexConfig(max_concurrent_analyses=0)
        
        with pytest.raises(ValueError, match="Concurrent analyses must be between 1 and 20"):
            ProjectIndexConfig(max_concurrent_analyses=25)


class TestComplexityMetrics:
    """Test ComplexityMetrics model."""
    
    def test_default_metrics(self):
        """Test default complexity metrics."""
        metrics = ComplexityMetrics()
        
        assert metrics.cyclomatic_complexity == 1
        assert metrics.cognitive_complexity == 1
        assert metrics.nesting_depth is None
        assert metrics.halstead_volume is None
    
    def test_custom_metrics(self):
        """Test custom complexity metrics."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            nesting_depth=3,
            halstead_volume=42.5
        )
        
        assert metrics.cyclomatic_complexity == 5
        assert metrics.cognitive_complexity == 8
        assert metrics.nesting_depth == 3
        assert metrics.halstead_volume == 42.5
    
    def test_complexity_validation(self):
        """Test complexity validation."""
        # Valid values
        metrics = ComplexityMetrics(cyclomatic_complexity=5)
        assert metrics.cyclomatic_complexity == 5
        
        # Invalid values
        with pytest.raises(ValueError, match="Complexity must be at least 1"):
            ComplexityMetrics(cyclomatic_complexity=0)
        
        with pytest.raises(ValueError, match="Complexity must be at least 1"):
            ComplexityMetrics(cognitive_complexity=-1)


class TestCodeStructure:
    """Test CodeStructure model."""
    
    def test_default_structure(self):
        """Test default code structure."""
        structure = CodeStructure()
        
        assert structure.functions == []
        assert structure.classes == []
        assert structure.imports == []
        assert structure.exports == []
        assert structure.variables == []
        assert structure.constants == []
    
    def test_custom_structure(self):
        """Test custom code structure."""
        functions = [{"name": "test_func", "line": 10}]
        classes = [{"name": "TestClass", "line": 1}]
        imports = [{"module": "os", "line": 1}]
        
        structure = CodeStructure(
            functions=functions,
            classes=classes,
            imports=imports
        )
        
        assert structure.functions == functions
        assert structure.classes == classes
        assert structure.imports == imports


class TestDependencyResult:
    """Test DependencyResult model."""
    
    def test_basic_dependency(self):
        """Test basic dependency result."""
        dep = DependencyResult(
            source_file_path="/path/to/source.py",
            target_name="os",
            dependency_type="import"
        )
        
        assert dep.source_file_path == "/path/to/source.py"
        assert dep.target_name == "os"
        assert dep.dependency_type == "import"
        assert dep.is_external is False
        assert dep.is_dynamic is False
        assert dep.confidence_score == 1.0
    
    def test_full_dependency(self):
        """Test dependency result with all fields."""
        dep = DependencyResult(
            source_file_path="/path/to/source.py",
            source_file_id="file_123",
            target_name="requests",
            target_path="/path/to/requests",
            target_file_id="file_456",
            dependency_type="import",
            line_number=5,
            column_number=1,
            source_text="import requests",
            is_external=True,
            is_dynamic=False,
            confidence_score=0.9,
            metadata={"package": "requests"}
        )
        
        assert dep.source_file_id == "file_123"
        assert dep.target_path == "/path/to/requests"
        assert dep.line_number == 5
        assert dep.is_external is True
        assert dep.confidence_score == 0.9
        assert dep.metadata == {"package": "requests"}
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid values
        dep = DependencyResult(
            source_file_path="/path/to/source.py",
            target_name="os",
            dependency_type="import",
            confidence_score=0.5
        )
        assert dep.confidence_score == 0.5
        
        # Invalid values
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            DependencyResult(
                source_file_path="/path/to/source.py",
                target_name="os",
                dependency_type="import",
                confidence_score=-0.1
            )
        
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            DependencyResult(
                source_file_path="/path/to/source.py",
                target_name="os",
                dependency_type="import",
                confidence_score=1.1
            )
    
    def test_position_validation(self):
        """Test line and column number validation."""
        # Valid values
        dep = DependencyResult(
            source_file_path="/path/to/source.py",
            target_name="os",
            dependency_type="import",
            line_number=5,
            column_number=10
        )
        assert dep.line_number == 5
        assert dep.column_number == 10
        
        # Invalid values
        with pytest.raises(ValueError, match="Line and column numbers must be positive"):
            DependencyResult(
                source_file_path="/path/to/source.py",
                target_name="os",
                dependency_type="import",
                line_number=0
            )
    
    def test_to_dict_and_from_dict(self):
        """Test dictionary conversion."""
        dep = DependencyResult(
            source_file_path="/path/to/source.py",
            target_name="os",
            dependency_type="import",
            line_number=5,
            confidence_score=0.9
        )
        
        # Convert to dict
        dep_dict = dep.to_dict()
        assert isinstance(dep_dict, dict)
        assert dep_dict['source_file_path'] == "/path/to/source.py"
        assert dep_dict['confidence_score'] == 0.9
        
        # Convert back from dict
        dep_restored = DependencyResult.from_dict(dep_dict)
        assert dep_restored.source_file_path == dep.source_file_path
        assert dep_restored.confidence_score == dep.confidence_score


class TestFileAnalysisResult:
    """Test FileAnalysisResult model."""
    
    def test_basic_analysis_result(self):
        """Test basic file analysis result."""
        result = FileAnalysisResult(
            file_path="/path/to/file.py"
        )
        
        assert result.file_path == "/path/to/file.py"
        assert result.is_binary is False
        assert result.is_generated is False
        assert result.analysis_successful is True
        assert result.dependencies == []
        assert result.tags == []
        assert result.metadata == {}
    
    def test_full_analysis_result(self):
        """Test file analysis result with all fields."""
        complexity = ComplexityMetrics(cyclomatic_complexity=5)
        structure = CodeStructure(functions=[{"name": "test", "line": 1}])
        dependencies = [DependencyResult(
            source_file_path="/path/to/file.py",
            target_name="os",
            dependency_type="import"
        )]
        
        result = FileAnalysisResult(
            file_path="/path/to/file.py",
            relative_path="src/file.py",
            file_name="file.py",
            file_extension=".py",
            file_type="source",
            language="python",
            file_size=1024,
            line_count=50,
            sha256_hash="abc123",
            complexity_metrics=complexity,
            code_structure=structure,
            dependencies=dependencies,
            analysis_duration=1.5,
            tags=["utility"],
            metadata={"analyzed": True}
        )
        
        assert result.relative_path == "src/file.py"
        assert result.file_size == 1024
        assert result.line_count == 50
        assert result.complexity_metrics == complexity
        assert result.code_structure == structure
        assert len(result.dependencies) == 1
        assert result.analysis_duration == 1.5
        assert result.tags == ["utility"]
    
    def test_file_size_validation(self):
        """Test file size validation."""
        # Valid values
        result = FileAnalysisResult(
            file_path="/path/to/file.py",
            file_size=1024
        )
        assert result.file_size == 1024
        
        # Invalid values
        with pytest.raises(ValueError, match="File size cannot be negative"):
            FileAnalysisResult(
                file_path="/path/to/file.py",
                file_size=-1
            )
    
    def test_line_count_validation(self):
        """Test line count validation."""
        # Valid values
        result = FileAnalysisResult(
            file_path="/path/to/file.py",
            line_count=100
        )
        assert result.line_count == 100
        
        # Invalid values
        with pytest.raises(ValueError, match="Line count cannot be negative"):
            FileAnalysisResult(
                file_path="/path/to/file.py",
                line_count=-1
            )
    
    def test_to_dict_and_from_dict(self):
        """Test dictionary conversion with datetime."""
        now = datetime.utcnow()
        result = FileAnalysisResult(
            file_path="/path/to/file.py",
            file_size=1024,
            last_modified=now
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['file_path'] == "/path/to/file.py"
        assert isinstance(result_dict['last_modified'], str)  # Should be ISO string
        
        # Convert back from dict
        result_restored = FileAnalysisResult.from_dict(result_dict)
        assert result_restored.file_path == result.file_path
        assert isinstance(result_restored.last_modified, datetime)


class TestAnalysisResult:
    """Test AnalysisResult model."""
    
    def test_basic_analysis_result(self):
        """Test basic analysis result."""
        result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        assert result.project_id == "proj_123"
        assert result.session_id == "sess_456"
        assert result.analysis_type == "full"
        assert result.files_processed == 0
        assert result.files_analyzed == 0
        assert result.dependencies_found == 0
        assert result.analysis_duration == 0.0
        assert result.file_results == []
        assert result.warnings == []
        assert result.errors == []
    
    def test_full_analysis_result(self):
        """Test analysis result with all fields."""
        file_result = FileAnalysisResult(file_path="/path/to/file.py")
        dependency_result = DependencyResult(
            source_file_path="/path/to/file.py",
            target_name="os",
            dependency_type="import"
        )
        
        started_at = datetime.utcnow()
        completed_at = datetime.utcnow()
        
        result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            files_processed=5,
            files_analyzed=4,
            dependencies_found=10,
            analysis_duration=30.5,
            started_at=started_at,
            completed_at=completed_at,
            file_results=[file_result],
            dependency_results=[dependency_result],
            warnings=["Warning 1"],
            errors=["Error 1"]
        )
        
        assert result.files_processed == 5
        assert result.files_analyzed == 4
        assert result.dependencies_found == 10
        assert result.analysis_duration == 30.5
        assert len(result.file_results) == 1
        assert len(result.dependency_results) == 1
        assert result.warnings == ["Warning 1"]
        assert result.errors == ["Error 1"]
    
    def test_count_validation(self):
        """Test count field validation."""
        # Valid values
        result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            files_processed=5
        )
        assert result.files_processed == 5
        
        # Invalid values
        with pytest.raises(ValueError, match="Counts cannot be negative"):
            AnalysisResult(
                project_id="proj_123",
                session_id="sess_456",
                analysis_type="full",
                files_processed=-1
            )
    
    def test_duration_validation(self):
        """Test duration validation."""
        # Valid values
        result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            analysis_duration=30.5
        )
        assert result.analysis_duration == 30.5
        
        # Invalid values
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            AnalysisResult(
                project_id="proj_123",
                session_id="sess_456",
                analysis_type="full",
                analysis_duration=-1.0
            )


class TestProjectStatistics:
    """Test ProjectStatistics model."""
    
    def test_default_statistics(self):
        """Test default project statistics."""
        stats = ProjectStatistics()
        
        assert stats.total_files == 0
        assert stats.files_by_type == {}
        assert stats.files_by_language == {}
        assert stats.total_lines_of_code == 0
        assert stats.total_file_size == 0
        assert stats.average_file_size == 0.0
        assert stats.total_dependencies == 0
        assert stats.average_complexity == 0.0
        assert stats.documentation_coverage == 0.0
    
    def test_custom_statistics(self):
        """Test custom project statistics."""
        stats = ProjectStatistics(
            total_files=100,
            files_by_type={"source": 60, "test": 30, "config": 10},
            files_by_language={"python": 80, "javascript": 20},
            total_lines_of_code=5000,
            total_file_size=1024000,
            average_file_size=10240.0,
            documentation_coverage=75.5
        )
        
        assert stats.total_files == 100
        assert stats.files_by_type["source"] == 60
        assert stats.files_by_language["python"] == 80
        assert stats.total_lines_of_code == 5000
        assert stats.documentation_coverage == 75.5
    
    def test_totals_validation(self):
        """Test total counts validation."""
        # Valid values
        stats = ProjectStatistics(total_files=100)
        assert stats.total_files == 100
        
        # Invalid values
        with pytest.raises(ValueError, match="Total counts cannot be negative"):
            ProjectStatistics(total_files=-1)
    
    def test_percentages_validation(self):
        """Test percentage validation."""
        # Valid values
        stats = ProjectStatistics(documentation_coverage=75.5)
        assert stats.documentation_coverage == 75.5
        
        # Invalid values
        with pytest.raises(ValueError, match="Percentages must be between 0.0 and 100.0"):
            ProjectStatistics(documentation_coverage=-1.0)
        
        with pytest.raises(ValueError, match="Percentages must be between 0.0 and 100.0"):
            ProjectStatistics(documentation_coverage=101.0)


class TestContextModels:
    """Test context optimization models."""
    
    def test_context_recommendation(self):
        """Test ContextRecommendation model."""
        rec = ContextRecommendation(
            file_path="/path/to/file.py",
            relevance_score=0.9,
            confidence=0.8,
            reasoning=["High dependency centrality", "Frequently modified"],
            tags=["core", "api"]
        )
        
        assert rec.file_path == "/path/to/file.py"
        assert rec.relevance_score == 0.9
        assert rec.confidence == 0.8
        assert rec.reasoning == ["High dependency centrality", "Frequently modified"]
        assert rec.tags == ["core", "api"]
    
    def test_context_recommendation_score_validation(self):
        """Test score validation for context recommendation."""
        # Valid values
        rec = ContextRecommendation(
            file_path="/path/to/file.py",
            relevance_score=0.5,
            confidence=0.5
        )
        assert rec.relevance_score == 0.5
        assert rec.confidence == 0.5
        
        # Invalid values
        with pytest.raises(ValueError, match="Scores must be between 0.0 and 1.0"):
            ContextRecommendation(
                file_path="/path/to/file.py",
                relevance_score=-0.1,
                confidence=0.5
            )
    
    def test_context_cluster_info(self):
        """Test ContextClusterInfo model."""
        cluster = ContextClusterInfo(
            cluster_id="cluster_1",
            name="API Layer",
            description="Files related to API endpoints",
            files=["/path/to/api1.py", "/path/to/api2.py"],
            central_files=["/path/to/api1.py"],
            cluster_score=0.85
        )
        
        assert cluster.cluster_id == "cluster_1"
        assert cluster.name == "API Layer"
        assert cluster.description == "Files related to API endpoints"
        assert len(cluster.files) == 2
        assert cluster.central_files == ["/path/to/api1.py"]
        assert cluster.cluster_score == 0.85
    
    def test_context_cluster_score_validation(self):
        """Test cluster score validation."""
        # Valid values
        cluster = ContextClusterInfo(
            cluster_id="cluster_1",
            name="Test Cluster",
            description="Test cluster",
            files=["/path/to/file.py"],
            cluster_score=0.75
        )
        assert cluster.cluster_score == 0.75
        
        # Invalid values
        with pytest.raises(ValueError, match="Cluster score must be between 0.0 and 1.0"):
            ContextClusterInfo(
                cluster_id="cluster_1",
                name="Test Cluster",
                description="Test cluster",
                files=["/path/to/file.py"],
                cluster_score=1.1
            )


class TestMonitoringModels:
    """Test monitoring and change models."""
    
    def test_file_change_info(self):
        """Test FileChangeInfo model."""
        now = datetime.utcnow()
        change = FileChangeInfo(
            file_path="/path/to/file.py",
            change_type="modified",
            timestamp=now,
            old_hash="abc123",
            new_hash="def456",
            metadata={"size_change": 100}
        )
        
        assert change.file_path == "/path/to/file.py"
        assert change.change_type == "modified"
        assert change.timestamp == now
        assert change.old_hash == "abc123"
        assert change.new_hash == "def456"
        assert change.metadata == {"size_change": 100}
    
    def test_analysis_progress(self):
        """Test AnalysisProgress model."""
        now = datetime.utcnow()
        progress = AnalysisProgress(
            session_id="sess_123",
            project_id="proj_456",
            current_phase="analyzing_files",
            progress_percentage=45.5,
            files_processed=45,
            files_total=100,
            estimated_completion=now,
            last_updated=now
        )
        
        assert progress.session_id == "sess_123"
        assert progress.project_id == "proj_456"
        assert progress.current_phase == "analyzing_files"
        assert progress.progress_percentage == 45.5
        assert progress.files_processed == 45
        assert progress.files_total == 100
    
    def test_progress_percentage_validation(self):
        """Test progress percentage validation."""
        # Valid values
        progress = AnalysisProgress(
            session_id="sess_123",
            project_id="proj_456",
            current_phase="analyzing",
            progress_percentage=50.0
        )
        assert progress.progress_percentage == 50.0
        
        # Invalid values
        with pytest.raises(ValueError, match="Progress percentage must be between 0.0 and 100.0"):
            AnalysisProgress(
                session_id="sess_123",
                project_id="proj_456",
                current_phase="analyzing",
                progress_percentage=-1.0
            )
    
    def test_analysis_error(self):
        """Test AnalysisError model."""
        now = datetime.utcnow()
        error = AnalysisError(
            error_type="ParseError",
            error_message="Failed to parse file",
            file_path="/path/to/file.py",
            line_number=42,
            severity="error",
            timestamp=now,
            stack_trace="Traceback...",
            metadata={"parser": "python"}
        )
        
        assert error.error_type == "ParseError"
        assert error.error_message == "Failed to parse file"
        assert error.file_path == "/path/to/file.py"
        assert error.line_number == 42
        assert error.severity == "error"
        assert error.timestamp == now
    
    def test_validation_result(self):
        """Test ValidationResult model."""
        error = AnalysisError(
            error_type="ValidationError",
            error_message="Invalid configuration"
        )
        warning = AnalysisError(
            error_type="ConfigWarning",
            error_message="Deprecated setting",
            severity="warning"
        )
        
        result = ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[warning],
            metadata={"validated_fields": 10}
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.errors[0].error_type == "ValidationError"
        assert result.warnings[0].severity == "warning"
        assert result.metadata == {"validated_fields": 10}


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_default_analysis_config(self):
        """Test default analysis config creation."""
        config = create_default_analysis_config()
        assert isinstance(config, AnalysisConfiguration)
        assert config.enabled_languages == ['python', 'javascript', 'typescript', 'json']
    
    def test_create_default_project_config(self):
        """Test default project config creation."""
        config = create_default_project_config()
        assert isinstance(config, ProjectIndexConfig)
        assert config.max_concurrent_analyses == 4
        assert config.cache_enabled is True
    
    def test_validate_file_path_string(self):
        """Test file path validation with string."""
        path = "/test/path/file.py"
        validated_path = validate_file_path(path)
        assert isinstance(validated_path, str)
        assert validated_path.endswith("file.py")
    
    def test_validate_file_path_pathlib(self):
        """Test file path validation with Path object."""
        from pathlib import Path
        path = Path("/test/path/file.py")
        validated_path = validate_file_path(path)
        assert isinstance(validated_path, str)
        assert validated_path.endswith("file.py")
    
    def test_validate_file_path_empty(self):
        """Test file path validation with empty string."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_path("")
    
    def test_calculate_file_metrics_empty(self):
        """Test calculating metrics with empty file results."""
        metrics = calculate_file_metrics([])
        assert metrics == {}
    
    def test_calculate_file_metrics_with_results(self):
        """Test calculating metrics with file results."""
        results = [
            FileAnalysisResult(
                file_path="/path/to/file1.py",
                file_size=1000,
                line_count=50,
                language="python",
                file_type="source",
                analysis_successful=True
            ),
            FileAnalysisResult(
                file_path="/path/to/file2.js",
                file_size=2000,
                line_count=100,
                language="javascript",
                file_type="source",
                analysis_successful=True
            ),
            FileAnalysisResult(
                file_path="/path/to/test.py",
                file_size=500,
                line_count=25,
                language="python",
                file_type="test",
                analysis_successful=False
            )
        ]
        
        metrics = calculate_file_metrics(results)
        
        assert metrics['total_files'] == 3
        assert metrics['successful_analyses'] == 2
        assert metrics['success_rate'] == 2/3
        assert metrics['total_lines_of_code'] == 175
        assert metrics['total_file_size'] == 3500
        assert metrics['average_file_size'] == 3500/3
        assert metrics['language_distribution'] == {'python': 2, 'javascript': 1}
        assert metrics['file_type_distribution'] == {'source': 2, 'test': 1}