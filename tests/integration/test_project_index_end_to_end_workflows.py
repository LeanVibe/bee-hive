"""
Integration tests for Project Index end-to-end workflows.

Tests complete workflows from project creation through analysis,
file monitoring, dependency tracking, and context optimization.
"""

import pytest
import asyncio
import uuid
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, AnalysisStatus, AnalysisSessionType
)
from tests.project_index_conftest import (
    wait_for_analysis_completion, assert_project_index_valid,
    assert_file_entry_valid, assert_dependency_valid
)


@pytest.mark.project_index_integration
@pytest.mark.integration
@pytest.mark.slow
class TestCompleteProjectLifecycle:
    """Test complete project lifecycle from creation to analysis completion."""
    
    async def test_create_and_analyze_project_workflow(
        self, 
        project_index_session: AsyncSession,
        temp_project_directory: Path,
        mock_redis_client,
        mock_event_publisher
    ):
        """Test complete workflow: create project -> analyze -> verify results."""
        
        # Step 1: Create project index
        project = ProjectIndex(
            name="Integration Test Project",
            description="End-to-end integration test project",
            root_path=str(temp_project_directory),
            git_repository_url="https://github.com/test/integration-project.git",
            git_branch="main",
            status=ProjectStatus.INACTIVE,
            configuration={
                "languages": ["python"],
                "analysis_depth": 3,
                "enable_ai_analysis": True
            },
            analysis_settings={
                "parse_ast": True,
                "extract_dependencies": True,
                "calculate_complexity": True
            },
            file_patterns={
                "include": ["**/*.py", "**/*.md"],
                "exclude": ["__pycache__/**"]
            }
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        assert_project_index_valid(project)
        assert project.status == ProjectStatus.INACTIVE
        initial_project_id = project.id
        
        # Step 2: Create analysis session
        analysis_session = AnalysisSession(
            project_id=project.id,
            session_name="Integration Test Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            status=AnalysisStatus.PENDING,
            files_total=5  # Expected number of files in temp directory
        )
        
        project_index_session.add(analysis_session)
        await project_index_session.commit()
        await project_index_session.refresh(analysis_session)
        
        # Step 3: Simulate analysis process
        # Start the session
        analysis_session.start_session()
        analysis_session.update_progress(10.0, "scanning_files")
        await project_index_session.commit()
        
        # Step 4: Create file entries (simulating file analysis)
        file_entries = []
        
        # Analyze README.md
        readme_entry = FileEntry(
            project_id=project.id,
            file_path=str(temp_project_directory / "README.md"),
            relative_path="README.md",
            file_name="README.md",
            file_extension=".md",
            file_type=FileType.DOCUMENTATION,
            language="markdown",
            file_size=45,
            line_count=3,
            sha256_hash="readme_integration_hash",
            content_preview="# Test Project\n\nA test project...",
            analysis_data={
                "word_count": 8,
                "sections": ["Test Project"]
            },
            tags=["documentation", "readme"],
            indexed_at=datetime.utcnow()
        )
        file_entries.append(readme_entry)
        
        # Analyze main.py
        main_entry = FileEntry(
            project_id=project.id,
            file_path=str(temp_project_directory / "src" / "main.py"),
            relative_path="src/main.py",
            file_name="main.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=156,
            line_count=10,
            sha256_hash="main_integration_hash",
            content_preview="import os\nimport sys\n...",
            analysis_data={
                "functions": [{"name": "main", "line": 5, "complexity": 1}],
                "imports": [
                    {"module": "os", "line": 1},
                    {"module": "sys", "line": 2},
                    {"module": "typing", "line": 3}
                ],
                "complexity_metrics": {
                    "cyclomatic_complexity": 1,
                    "cognitive_complexity": 1
                }
            },
            tags=["main", "entry_point"],
            indexed_at=datetime.utcnow()
        )
        file_entries.append(main_entry)
        
        # Analyze utils.py
        utils_entry = FileEntry(
            project_id=project.id,
            file_path=str(temp_project_directory / "src" / "utils.py"),
            relative_path="src/utils.py",
            file_name="utils.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=289,
            line_count=12,
            sha256_hash="utils_integration_hash",
            content_preview="import json\nfrom pathlib import Path...",
            analysis_data={
                "functions": [
                    {"name": "load_config", "line": 4, "complexity": 2},
                    {"name": "save_data", "line": 8, "complexity": 2}
                ],
                "imports": [
                    {"module": "json", "line": 1},
                    {"module": "pathlib", "line": 2}
                ],
                "complexity_metrics": {
                    "cyclomatic_complexity": 4,
                    "cognitive_complexity": 3
                }
            },
            tags=["utility", "helper"],
            indexed_at=datetime.utcnow()
        )
        file_entries.append(utils_entry)
        
        # Add all file entries
        for entry in file_entries:
            project_index_session.add(entry)
        
        await project_index_session.commit()
        
        # Refresh entries to get IDs
        for entry in file_entries:
            await project_index_session.refresh(entry)
            assert_file_entry_valid(entry)
        
        # Update analysis progress
        analysis_session.update_progress(50.0, "analyzing_files")
        analysis_session.files_processed = len(file_entries)
        await project_index_session.commit()
        
        # Step 5: Create dependency relationships
        dependencies = []
        
        # Dependencies from main.py
        main_deps = [
            DependencyRelationship(
                project_id=project.id,
                source_file_id=main_entry.id,
                target_name="os",
                dependency_type="import",
                line_number=1,
                source_text="import os",
                is_external=True,
                confidence_score=1.0,
                metadata={"standard_library": True}
            ),
            DependencyRelationship(
                project_id=project.id,
                source_file_id=main_entry.id,
                target_name="sys",
                dependency_type="import",
                line_number=2,
                source_text="import sys",
                is_external=True,
                confidence_score=1.0,
                metadata={"standard_library": True}
            )
        ]
        dependencies.extend(main_deps)
        
        # Dependencies from utils.py
        utils_deps = [
            DependencyRelationship(
                project_id=project.id,
                source_file_id=utils_entry.id,
                target_name="json",
                dependency_type="import",
                line_number=1,
                source_text="import json",
                is_external=True,
                confidence_score=1.0,
                metadata={"standard_library": True}
            ),
            DependencyRelationship(
                project_id=project.id,
                source_file_id=utils_entry.id,
                target_name="pathlib.Path",
                dependency_type="import",
                line_number=2,
                source_text="from pathlib import Path",
                is_external=True,
                confidence_score=1.0,
                metadata={"standard_library": True}
            )
        ]
        dependencies.extend(utils_deps)
        
        # Add all dependencies
        for dep in dependencies:
            project_index_session.add(dep)
        
        await project_index_session.commit()
        
        # Refresh dependencies
        for dep in dependencies:
            await project_index_session.refresh(dep)
            assert_dependency_valid(dep)
        
        # Update analysis progress
        analysis_session.update_progress(80.0, "extracting_dependencies")
        analysis_session.dependencies_found = len(dependencies)
        await project_index_session.commit()
        
        # Step 6: Complete analysis
        result_data = {
            "summary": {
                "files_analyzed": len(file_entries),
                "dependencies_extracted": len(dependencies),
                "languages_detected": ["python", "markdown"],
                "complexity_calculated": True
            },
            "statistics": {
                "total_lines_of_code": sum(f.line_count or 0 for f in file_entries),
                "avg_complexity": 2.0,
                "documentation_files": 1,
                "source_files": 2
            },
            "quality_metrics": {
                "test_coverage": 0.0,  # No test files
                "documentation_coverage": 0.33,  # 1 doc file out of 3 total
                "code_duplication": 0.0
            }
        }
        
        analysis_session.complete_session(result_data)
        await project_index_session.commit()
        
        # Step 7: Update project statistics
        project.status = ProjectStatus.ACTIVE
        project.file_count = len(file_entries)
        project.dependency_count = len(dependencies)
        project.last_indexed_at = datetime.utcnow()
        project.last_analysis_at = datetime.utcnow()
        await project_index_session.commit()
        
        # Step 8: Verify final state
        await project_index_session.refresh(project)
        await project_index_session.refresh(analysis_session)
        
        # Verify project state
        assert project.status == ProjectStatus.ACTIVE
        assert project.file_count == 3
        assert project.dependency_count == 4
        assert project.last_indexed_at is not None
        assert project.last_analysis_at is not None
        
        # Verify analysis session state
        assert analysis_session.status == AnalysisStatus.COMPLETED
        assert analysis_session.progress_percentage == 100.0
        assert analysis_session.files_processed == 3
        assert analysis_session.dependencies_found == 4
        assert analysis_session.completed_at is not None
        assert analysis_session.result_data is not None
        
        # Verify relationships work
        stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        result = await project_index_session.execute(stmt)
        project_files = result.scalars().all()
        assert len(project_files) == 3
        
        stmt = select(DependencyRelationship).where(DependencyRelationship.project_id == project.id)
        result = await project_index_session.execute(stmt)
        project_deps = result.scalars().all()
        assert len(project_deps) == 4
        
        # Verify file analysis data
        python_files = [f for f in project_files if f.language == "python"]
        assert len(python_files) == 2
        
        for py_file in python_files:
            assert py_file.analysis_data is not None
            assert "functions" in py_file.analysis_data
            assert "imports" in py_file.analysis_data
            assert "complexity_metrics" in py_file.analysis_data


@pytest.mark.project_index_integration
@pytest.mark.integration
class TestFileChangeWorkflow:
    """Test file change detection and re-analysis workflow."""
    
    async def test_file_modification_detection_workflow(
        self,
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        sample_file_entries: List[FileEntry],
        mock_redis_client
    ):
        """Test workflow when files are modified and need re-analysis."""
        
        # Step 1: Get initial file state
        main_file = next(f for f in sample_file_entries if f.file_name == "main.py")
        initial_hash = main_file.sha256_hash
        initial_modified = main_file.last_modified
        
        # Step 2: Simulate file modification
        # Update file with new content and hash
        new_content = """
import os
import sys
import json  # New import
from typing import List, Dict  # Extended import

def main() -> None:
    print("Hello, Updated World!")
    
def new_function() -> Dict[str, Any]:  # New function
    return {"status": "updated"}

if __name__ == "__main__":
    main()
"""
        
        main_file.sha256_hash = "new_hash_after_modification"
        main_file.last_modified = datetime.utcnow()
        main_file.content_preview = new_content[:200] + "..."
        main_file.line_count = 12  # Updated line count
        main_file.analysis_data = {
            "functions": [
                {"name": "main", "line": 5, "complexity": 1},
                {"name": "new_function", "line": 8, "complexity": 1}
            ],
            "imports": [
                {"module": "os", "line": 1},
                {"module": "sys", "line": 2},
                {"module": "json", "line": 3},  # New import
                {"module": "typing", "line": 4, "names": ["List", "Dict"]}
            ],
            "complexity_metrics": {
                "cyclomatic_complexity": 2,  # Increased complexity
                "cognitive_complexity": 2
            }
        }
        
        await project_index_session.commit()
        await project_index_session.refresh(main_file)
        
        # Step 3: Verify file was updated
        assert main_file.sha256_hash != initial_hash
        assert main_file.last_modified > initial_modified
        assert main_file.line_count == 12
        assert len(main_file.analysis_data["functions"]) == 2
        assert len(main_file.analysis_data["imports"]) == 4
        
        # Step 4: Create incremental analysis session
        incremental_session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="File Modification Analysis",
            session_type=AnalysisSessionType.INCREMENTAL,
            status=AnalysisStatus.RUNNING,
            files_total=1,  # Only re-analyzing the modified file
            session_data={
                "trigger": "file_modification",
                "modified_files": [main_file.relative_path],
                "change_type": "content_update"
            }
        )
        
        project_index_session.add(incremental_session)
        await project_index_session.commit()
        await project_index_session.refresh(incremental_session)
        
        # Step 5: Add new dependency detected from the new import
        new_dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=main_file.id,
            target_name="json",
            dependency_type="import",
            line_number=3,
            source_text="import json",
            is_external=True,
            confidence_score=1.0,
            metadata={"standard_library": True, "added_in_update": True}
        )
        
        project_index_session.add(new_dependency)
        await project_index_session.commit()
        await project_index_session.refresh(new_dependency)
        
        # Step 6: Complete incremental analysis
        incremental_session.files_processed = 1
        incremental_session.dependencies_found = 1
        incremental_session.complete_session({
            "modified_files": [main_file.relative_path],
            "new_dependencies": 1,
            "functions_added": 1,
            "complexity_change": "+1"
        })
        
        await project_index_session.commit()
        
        # Step 7: Update project statistics
        sample_project_index.dependency_count += 1
        sample_project_index.last_indexed_at = datetime.utcnow()
        await project_index_session.commit()
        
        # Step 8: Verify incremental analysis results
        await project_index_session.refresh(incremental_session)
        
        assert incremental_session.status == AnalysisStatus.COMPLETED
        assert incremental_session.files_processed == 1
        assert incremental_session.dependencies_found == 1
        assert "modified_files" in incremental_session.result_data
        
        # Verify project dependency count increased
        await project_index_session.refresh(sample_project_index)
        assert sample_project_index.dependency_count == 4  # Original 3 + 1 new


@pytest.mark.project_index_integration
@pytest.mark.integration
class TestDependencyGraphWorkflow:
    """Test dependency graph analysis and optimization workflow."""
    
    async def test_dependency_graph_analysis_workflow(
        self,
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        sample_file_entries: List[FileEntry],
        sample_dependencies: List[DependencyRelationship]
    ):
        """Test complete dependency graph analysis and relationship mapping."""
        
        # Step 1: Add internal dependency (file-to-file)
        main_file = next(f for f in sample_file_entries if f.file_name == "main.py")
        utils_file = next(f for f in sample_file_entries if f.file_name == "utils.py")
        
        # Create internal dependency: main.py imports utils.py
        internal_dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=main_file.id,
            target_file_id=utils_file.id,
            target_path=utils_file.relative_path,
            target_name="utils",
            dependency_type="import",
            line_number=4,
            source_text="from src import utils",
            is_external=False,  # Internal dependency
            confidence_score=0.95,
            metadata={
                "import_type": "relative",
                "internal_dependency": True
            }
        )
        
        project_index_session.add(internal_dependency)
        await project_index_session.commit()
        await project_index_session.refresh(internal_dependency)
        
        # Step 2: Create dependency mapping analysis session
        dependency_session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Dependency Graph Analysis",
            session_type=AnalysisSessionType.DEPENDENCY_MAPPING,
            status=AnalysisStatus.RUNNING,
            session_data={
                "analysis_scope": "full_project",
                "include_external": True,
                "graph_depth": 2
            }
        )
        
        project_index_session.add(dependency_session)
        await project_index_session.commit()
        
        # Step 3: Analyze dependency graph structure
        all_deps = sample_dependencies + [internal_dependency]
        
        # Calculate graph metrics
        external_deps = [d for d in all_deps if d.is_external]
        internal_deps = [d for d in all_deps if not d.is_external]
        
        # File dependency analysis
        file_in_degrees = {}
        file_out_degrees = {}
        
        for file_entry in sample_file_entries:
            file_in_degrees[file_entry.id] = len([d for d in all_deps if d.target_file_id == file_entry.id])
            file_out_degrees[file_entry.id] = len([d for d in all_deps if d.source_file_id == file_entry.id])
        
        # Step 4: Generate dependency graph analysis results
        graph_analysis = {
            "graph_metrics": {
                "total_dependencies": len(all_deps),
                "external_dependencies": len(external_deps),
                "internal_dependencies": len(internal_deps),
                "dependency_ratio": len(external_deps) / len(all_deps) if all_deps else 0
            },
            "file_metrics": {
                "files_with_outgoing_deps": len([f for f in sample_file_entries if file_out_degrees[f.id] > 0]),
                "files_with_incoming_deps": len([f for f in sample_file_entries if file_in_degrees[f.id] > 0]),
                "avg_outgoing_deps": sum(file_out_degrees.values()) / len(sample_file_entries),
                "avg_incoming_deps": sum(file_in_degrees.values()) / len(sample_file_entries)
            },
            "dependency_types": {
                dep_type: len([d for d in all_deps if d.dependency_type == dep_type])
                for dep_type in set(d.dependency_type for d in all_deps)
            },
            "circular_dependencies": [],  # Would be detected by actual analysis
            "dependency_clusters": [
                {
                    "cluster_id": "python_stdlib",
                    "dependencies": [d.target_name for d in external_deps if d.metadata.get("standard_library")],
                    "cluster_type": "external_library_group"
                }
            ]
        }
        
        # Step 5: Complete dependency analysis
        dependency_session.complete_session({
            "analysis_type": "dependency_mapping",
            "graph_analysis": graph_analysis,
            "dependencies_analyzed": len(all_deps),
            "files_involved": len(sample_file_entries)
        })
        
        dependency_session.dependencies_found = len(all_deps)
        dependency_session.files_processed = len(sample_file_entries)
        
        await project_index_session.commit()
        
        # Step 6: Verify dependency analysis results
        await project_index_session.refresh(dependency_session)
        
        assert dependency_session.status == AnalysisStatus.COMPLETED
        assert dependency_session.dependencies_found == 5  # 4 original + 1 internal
        
        result_data = dependency_session.result_data
        assert "graph_analysis" in result_data
        
        graph_metrics = result_data["graph_analysis"]["graph_metrics"]
        assert graph_metrics["total_dependencies"] == 5
        assert graph_metrics["external_dependencies"] == 4
        assert graph_metrics["internal_dependencies"] == 1
        
        # Verify internal dependency was created correctly
        assert internal_dependency.is_external is False
        assert internal_dependency.target_file_id == utils_file.id
        assert internal_dependency.source_file_id == main_file.id


@pytest.mark.project_index_integration
@pytest.mark.integration
class TestContextOptimizationWorkflow:
    """Test context optimization and intelligent recommendations workflow."""
    
    async def test_context_optimization_workflow(
        self,
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        sample_file_entries: List[FileEntry],
        sample_dependencies: List[DependencyRelationship],
        sample_analysis_session: AnalysisSession
    ):
        """Test AI-powered context optimization and file relevance scoring."""
        
        # Step 1: Create context optimization session
        context_session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Context Optimization Analysis",
            session_type=AnalysisSessionType.CONTEXT_OPTIMIZATION,
            status=AnalysisStatus.RUNNING,
            session_data={
                "optimization_type": "relevance_scoring",
                "context_window": "current_task",
                "ai_model": "claude-3-sonnet"
            }
        )
        
        project_index_session.add(context_session)
        await project_index_session.commit()
        
        # Step 2: Simulate AI analysis of file relevance
        # Calculate relevance scores based on various factors
        file_relevance_scores = {}
        
        for file_entry in sample_file_entries:
            # Base score calculation
            relevance_score = 0.5  # Base relevance
            
            # Boost score for main files
            if "main" in file_entry.file_name.lower():
                relevance_score += 0.3
            
            # Boost score for files with many dependencies
            outgoing_deps = [d for d in sample_dependencies if d.source_file_id == file_entry.id]
            if len(outgoing_deps) > 2:
                relevance_score += 0.2
            
            # Boost score for recently modified files
            if file_entry.last_modified and file_entry.last_modified > datetime.utcnow() - timedelta(days=7):
                relevance_score += 0.1
            
            # Adjust for file type
            if file_entry.file_type == FileType.SOURCE:
                relevance_score += 0.1
            elif file_entry.file_type == FileType.TEST:
                relevance_score += 0.05
            
            # Cap at 1.0
            relevance_score = min(1.0, relevance_score)
            
            file_relevance_scores[file_entry.id] = {
                "file_path": file_entry.relative_path,
                "relevance_score": relevance_score,
                "confidence": 0.85,
                "factors": {
                    "dependency_centrality": len(outgoing_deps) / max(len(sample_dependencies), 1),
                    "recent_activity": file_entry.last_modified is not None,
                    "file_type_weight": 0.8 if file_entry.file_type == FileType.SOURCE else 0.5,
                    "name_significance": "main" in file_entry.file_name.lower()
                }
            }
        
        # Step 3: Generate context recommendations
        sorted_files = sorted(
            file_relevance_scores.items(),
            key=lambda x: x[1]["relevance_score"],
            reverse=True
        )
        
        context_recommendations = []
        for file_id, score_data in sorted_files[:2]:  # Top 2 most relevant files
            file_entry = next(f for f in sample_file_entries if f.id == file_id)
            
            recommendation = {
                "file_id": str(file_id),
                "file_path": score_data["file_path"],
                "relevance_score": score_data["relevance_score"],
                "confidence": score_data["confidence"],
                "reasoning": [
                    f"High dependency centrality ({score_data['factors']['dependency_centrality']:.2f})",
                    f"File type weight: {score_data['factors']['file_type_weight']}",
                    "Recently active" if score_data["factors"]["recent_activity"] else "Stable file"
                ],
                "context_priority": "high" if score_data["relevance_score"] > 0.7 else "medium",
                "suggested_context_size": min(2000, file_entry.file_size or 0)  # Characters to include
            }
            context_recommendations.append(recommendation)
        
        # Step 4: Generate file clustering analysis
        file_clusters = [
            {
                "cluster_id": "core_application",
                "cluster_name": "Core Application Logic",
                "files": [f.relative_path for f in sample_file_entries if f.file_type == FileType.SOURCE],
                "cluster_score": 0.9,
                "central_files": ["src/main.py"],
                "description": "Main application logic and utilities"
            },
            {
                "cluster_id": "documentation",
                "cluster_name": "Project Documentation",
                "files": [f.relative_path for f in sample_file_entries if f.file_type == FileType.DOCUMENTATION],
                "cluster_score": 0.6,
                "central_files": ["README.md"],
                "description": "Project documentation and guides"
            }
        ]
        
        # Step 5: Complete context optimization
        optimization_results = {
            "context_recommendations": context_recommendations,
            "file_clusters": file_clusters,
            "optimization_metrics": {
                "files_analyzed": len(sample_file_entries),
                "relevance_scores_generated": len(file_relevance_scores),
                "high_priority_files": len([r for r in context_recommendations if r["context_priority"] == "high"]),
                "avg_relevance_score": sum(s["relevance_score"] for s in file_relevance_scores.values()) / len(file_relevance_scores),
                "optimization_confidence": 0.82
            },
            "suggested_context_window": {
                "total_files_to_include": len(context_recommendations),
                "estimated_context_size": sum(r["suggested_context_size"] for r in context_recommendations),
                "context_strategy": "relevance_weighted"
            }
        }
        
        context_session.complete_session(optimization_results)
        context_session.files_processed = len(sample_file_entries)
        
        await project_index_session.commit()
        
        # Step 6: Verify context optimization results
        await project_index_session.refresh(context_session)
        
        assert context_session.status == AnalysisStatus.COMPLETED
        assert context_session.files_processed == len(sample_file_entries)
        
        result_data = context_session.result_data
        assert "context_recommendations" in result_data
        assert "file_clusters" in result_data
        assert "optimization_metrics" in result_data
        
        # Verify recommendations structure
        recommendations = result_data["context_recommendations"]
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "file_path" in rec
            assert "relevance_score" in rec
            assert "confidence" in rec
            assert "reasoning" in rec
            assert 0.0 <= rec["relevance_score"] <= 1.0
            assert 0.0 <= rec["confidence"] <= 1.0
        
        # Verify clustering results
        clusters = result_data["file_clusters"]
        assert len(clusters) > 0
        
        for cluster in clusters:
            assert "cluster_id" in cluster
            assert "cluster_name" in cluster
            assert "files" in cluster
            assert "cluster_score" in cluster
            assert isinstance(cluster["files"], list)


@pytest.mark.project_index_integration
@pytest.mark.integration
@pytest.mark.slow
class TestErrorRecoveryWorkflow:
    """Test error handling and recovery workflows."""
    
    async def test_analysis_failure_recovery_workflow(
        self,
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        temp_project_directory: Path
    ):
        """Test analysis failure detection and recovery workflow."""
        
        # Step 1: Create analysis session that will encounter errors
        error_session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Error Recovery Test",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            status=AnalysisStatus.RUNNING,
            files_total=10  # Expect more files than actually exist
        )
        
        project_index_session.add(error_session)
        await project_index_session.commit()
        
        # Step 2: Simulate various errors during analysis
        
        # Parse error for a corrupted file
        error_session.add_error(
            "Failed to parse file: src/corrupted.py - Invalid syntax at line 15",
            "error"
        )
        
        # Permission error for a protected file
        error_session.add_error(
            "Permission denied accessing file: /protected/file.py",
            "error"
        )
        
        # Warning for an unsupported file type
        error_session.add_error(
            "Unsupported file type: .obscure_extension - skipping analysis",
            "warning"
        )
        
        # File not found error
        error_session.add_error(
            "File not found: expected_file.py (referenced in imports)",
            "warning"
        )
        
        await project_index_session.commit()
        
        # Step 3: Simulate partial completion despite errors
        error_session.files_processed = 3  # Some files processed successfully
        error_session.update_progress(30.0, "analysis_with_errors")
        
        # Add performance metrics showing degraded performance
        error_session.performance_metrics = {
            "total_duration": 45.0,  # Longer than normal
            "avg_file_analysis_time": 15.0,  # Much slower than normal
            "errors_per_file": 1.33,  # High error rate
            "success_rate": 0.75,  # 75% success rate
            "retry_attempts": 5
        }
        
        await project_index_session.commit()
        
        # Step 4: Decide on failure vs. partial success
        # If too many errors, mark as failed
        if error_session.errors_count > 2:
            error_session.fail_session("Analysis failed due to excessive errors")
        else:
            # Otherwise, complete with warnings
            error_session.complete_session({
                "status": "completed_with_errors",
                "files_successfully_analyzed": 3,
                "files_failed": 2,
                "total_errors": error_session.errors_count,
                "total_warnings": error_session.warnings_count,
                "recovery_actions": [
                    "Skipped corrupted files",
                    "Used fallback analysis for protected files",
                    "Applied best-effort parsing for partial files"
                ]
            })
        
        await project_index_session.commit()
        
        # Step 5: Create recovery session
        recovery_session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Error Recovery Analysis",
            session_type=AnalysisSessionType.INCREMENTAL,
            status=AnalysisStatus.RUNNING,
            session_data={
                "recovery_mode": True,
                "parent_session_id": str(error_session.id),
                "retry_failed_files": True,
                "error_mitigation": [
                    "Skip corrupted files",
                    "Use restricted permissions",
                    "Apply lenient parsing"
                ]
            }
        )
        
        project_index_session.add(recovery_session)
        await project_index_session.commit()
        
        # Step 6: Simulate successful recovery of some files
        # Successfully analyze files that failed before
        recovery_files = [
            FileEntry(
                project_id=sample_project_index.id,
                file_path=str(temp_project_directory / "recovered_file.py"),
                relative_path="recovered_file.py",
                file_name="recovered_file.py",
                file_extension=".py",
                file_type=FileType.SOURCE,
                language="python",
                file_size=100,
                line_count=5,
                sha256_hash="recovery_hash",
                analysis_data={
                    "functions": [],
                    "imports": [],
                    "recovery_notes": "Successfully analyzed after error recovery"
                },
                tags=["recovered"],
                indexed_at=datetime.utcnow()
            )
        ]
        
        for file_entry in recovery_files:
            project_index_session.add(file_entry)
        
        await project_index_session.commit()
        
        # Complete recovery session
        recovery_session.complete_session({
            "recovery_status": "partial_success",
            "files_recovered": len(recovery_files),
            "remaining_failures": 1,  # Some files still couldn't be recovered
            "recovery_techniques_used": [
                "Lenient parsing mode",
                "Syntax error tolerance",
                "Partial content analysis"
            ]
        })
        
        recovery_session.files_processed = len(recovery_files)
        
        await project_index_session.commit()
        
        # Step 7: Verify error handling and recovery
        await project_index_session.refresh(error_session)
        await project_index_session.refresh(recovery_session)
        
        # Verify error session recorded failures properly
        assert error_session.status == AnalysisStatus.FAILED
        assert error_session.errors_count > 0
        assert error_session.warnings_count > 0
        assert len(error_session.error_log) >= 4  # All errors recorded
        
        # Verify error log structure
        for log_entry in error_session.error_log:
            assert "timestamp" in log_entry
            assert "error" in log_entry
            assert "level" in log_entry
            assert log_entry["level"] in ["error", "warning", "fatal"]
        
        # Verify recovery session succeeded
        assert recovery_session.status == AnalysisStatus.COMPLETED
        assert recovery_session.files_processed > 0
        assert "recovery_status" in recovery_session.result_data
        
        # Verify recovery session data
        recovery_data = recovery_session.result_data
        assert recovery_data["recovery_status"] == "partial_success"
        assert recovery_data["files_recovered"] == 1
        assert "recovery_techniques_used" in recovery_data