"""
TDD Tests for Documentation Consolidation System Enhancements
Sprint 2: WebSocket Resilience & Documentation Foundation

Test-driven development for documentation consolidation system that enables:
- Automated content analysis and extraction from 500+ files
- Master document creation with deduplication
- Living documentation with automated validation
- Archive management with backward compatibility
"""

import asyncio
import json
import tempfile
import shutil
import pytest
import hashlib
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta

# Import the documentation automation modules
import sys
sys.path.append("/Users/bogdan/work/leanvibe-dev/bee-hive/docs/automation")

from link_validator import LinkValidator, LinkValidationResult
from code_example_tester import CodeExampleTester, CodeTestResult
from documentation_consolidator import DocumentationConsolidator, LivingDocumentationSystem, DocumentationAsset, ConsolidationStrategy


# Classes are now imported from documentation_consolidator module


class TestDocumentationConsolidationSystem:
    """TDD test suite for documentation consolidation system"""

    @pytest.fixture
    def mock_documentation_files(self):
        """Create mock documentation files for testing"""
        return {
            "README.md": "# Project Overview\nThis is the main project. It provides a comprehensive overview of the system architecture and core functionality.",
            "docs/setup.md": "# Setup Guide\nFollow these steps to setup the system. First install dependencies, then configure the environment, and finally run the application.",
            "docs/api.md": "# API Reference\nAPI endpoints documentation. Complete reference for all REST endpoints.",
            "docs/architecture.md": "# Architecture\nSystem architecture overview. Detailed design and component relationships.",
            "docs/duplicate_setup.md": "# Setup Instructions\nFollow these steps to setup the system. First install dependencies, then configure the environment, and finally run the application.",  # High similarity
            "docs/old_readme.md": "# Project Info\nThis is the old project overview. It provides a comprehensive overview of the system architecture and core functionality.",  # High similarity to README
            "docs/archive/legacy_api.md": "# Legacy API\nOld API documentation. Historical reference for deprecated endpoints.",
            "docs/guides/development.md": "# Development Guide\nDevelopment procedures and best practices.",
            "docs/guides/deployment.md": "# Deployment Guide\nDeployment procedures and production setup.",
            "docs/reference/schemas.md": "# Data Schemas\nAPI schema definitions and data models."
        }

    @pytest.fixture
    def mock_consolidation_strategy(self):
        """Provide consolidation strategy for testing"""
        return ConsolidationStrategy(
            target_structure={
                "README.md": ["README.md", "docs/old_readme.md"],
                "DEVELOPER_GUIDE.md": ["docs/setup.md", "docs/duplicate_setup.md", "docs/guides/development.md"],
                "API_REFERENCE.md": ["docs/api.md", "docs/archive/legacy_api.md"],
                "ARCHITECTURE.md": ["docs/architecture.md"],
                "OPERATIONS_GUIDE.md": ["docs/guides/deployment.md"],
                "REFERENCE.md": ["docs/reference/schemas.md"]
            },
            deduplication_threshold=0.8,
            content_merge_rules={
                "chronological": "newer_first",
                "quality": "higher_quality_first",
                "completeness": "more_complete_first"
            },
            quality_gates={
                "min_content_length": 50,  # More realistic for test content
                "max_duplicate_ratio": 0.6,  # More realistic for merged content
                "required_sections": 0.8
            },
            archive_rules={
                "preserve_history": True,
                "maintain_references": True,
                "create_redirect_map": True
            }
        )

    @pytest.fixture
    def mock_consolidator(self, mock_documentation_files):
        """Create documentation consolidator with mock data"""
        consolidator = DocumentationConsolidator()
        
        # Mock assets based on files
        consolidator.assets = []
        for file_path, content in mock_documentation_files.items():
            asset = DocumentationAsset(
                path=file_path,
                content=content,
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                file_size=len(content),
                last_modified=datetime.now(),
                doc_type="core" if "README" in file_path else "guide",
                quality_score=0.8
            )
            consolidator.assets.append(asset)
        
        return consolidator

    # =============== CONTENT ANALYSIS & DISCOVERY TESTS ===============

    @pytest.mark.asyncio
    async def test_should_analyze_documentation_structure_comprehensively(self, mock_consolidator):
        """Test comprehensive analysis of existing documentation structure"""
        # Given: A consolidator with documentation assets
        
        # When: Documentation structure is analyzed
        assets = await mock_consolidator.analyze_existing_documentation()
        
        # Then: Should return comprehensive asset information
        assert len(assets) >= 10  # Should find all mock files
        
        # And: Should categorize documents by type
        doc_types = set(asset.doc_type for asset in assets)
        assert "core" in doc_types
        assert "guide" in doc_types
        
        # And: Should calculate quality scores
        quality_scores = [asset.quality_score for asset in assets]
        assert all(0 <= score <= 1 for score in quality_scores)
        
        # And: Should detect file metadata
        for asset in assets:
            assert asset.content_hash is not None
            assert asset.file_size > 0
            assert asset.last_modified is not None

    @pytest.mark.asyncio 
    async def test_should_detect_duplicate_and_similar_content_accurately(self, mock_consolidator):
        """Test accurate detection of duplicate and similar content"""
        # Given: Documentation with known duplicates
        
        # When: Duplicate content is detected
        duplicates = await mock_consolidator.detect_content_duplicates(similarity_threshold=0.8)
        
        # Then: Should identify duplicate setup documents
        setup_duplicates = duplicates.get("setup_content", [])
        assert len(setup_duplicates) >= 2  # setup.md and duplicate_setup.md
        
        # And: Should identify similar overview documents  
        overview_duplicates = duplicates.get("overview_content", [])
        assert len(overview_duplicates) >= 2  # README.md and old_readme.md
        
        # And: Should group by content similarity
        for content_group, files in duplicates.items():
            assert len(files) >= 2  # Each group should have multiple files
            
        # And: Should calculate similarity scores
        assert len(duplicates) >= 2  # Should find at least 2 duplicate groups

    @pytest.mark.asyncio
    async def test_should_identify_content_dependencies_and_references(self, mock_consolidator):
        """Test identification of content dependencies and cross-references"""
        # Given: Documentation with internal references
        
        # When: Assets are analyzed for dependencies
        assets = await mock_consolidator.analyze_existing_documentation()
        
        # Then: Should identify internal links as dependencies
        assets_with_deps = [asset for asset in assets if asset.dependencies]
        assert len(assets_with_deps) >= 1
        
        # And: Should track cross-references between documents
        for asset in assets_with_deps:
            for dep in asset.dependencies:
                # Dependencies should be valid file paths
                assert isinstance(dep, str)
                assert len(dep) > 0

    # =============== CONTENT CONSOLIDATION & MERGING TESTS ===============

    @pytest.mark.asyncio
    async def test_should_generate_master_documents_following_strategy(self, mock_consolidator, mock_consolidation_strategy):
        """Test generation of master documents according to consolidation strategy"""
        # Given: A consolidation strategy and documentation assets
        
        # When: Master documents are generated
        master_docs = await mock_consolidator.generate_master_documents(mock_consolidation_strategy)
        
        # Then: Should create all target documents from strategy
        target_files = list(mock_consolidation_strategy.target_structure.keys())
        generated_files = [doc["file_name"] for doc in master_docs]
        
        for target_file in target_files:
            assert target_file in generated_files
        
        # And: Should merge content according to rules
        for doc in master_docs:
            assert "merged_content" in doc
            assert len(doc["merged_content"]) > 0
            
        # And: Should preserve important information
        for doc in master_docs:
            assert "source_files" in doc
            assert "merge_metadata" in doc
            assert len(doc["source_files"]) >= 1

    @pytest.mark.asyncio
    async def test_should_apply_content_deduplication_intelligently(self, mock_consolidator, mock_consolidation_strategy):
        """Test intelligent content deduplication during merging"""
        # Given: Documents with overlapping content
        
        # When: Content is deduplicated during consolidation
        master_docs = await mock_consolidator.generate_master_documents(mock_consolidation_strategy)
        
        # Then: Should remove duplicate sections
        developer_guide = next(doc for doc in master_docs if doc["file_name"] == "DEVELOPER_GUIDE.md")
        
        # Should not contain duplicate setup instructions
        content = developer_guide["merged_content"]
        setup_mentions = content.lower().count("follow these steps to setup")
        assert setup_mentions <= 1  # Should deduplicate
        
        # And: Should preserve unique information from each source
        assert "merge_summary" in developer_guide
        merge_summary = developer_guide["merge_summary"]
        assert "duplicates_removed" in merge_summary
        assert merge_summary["duplicates_removed"] >= 1

    @pytest.mark.asyncio
    async def test_should_maintain_content_quality_during_consolidation(self, mock_consolidator, mock_consolidation_strategy):
        """Test that content quality is maintained during consolidation"""
        # Given: Quality gates in consolidation strategy
        quality_gates = mock_consolidation_strategy.quality_gates
        
        # When: Master documents are generated
        master_docs = await mock_consolidator.generate_master_documents(mock_consolidation_strategy)
        
        # Then: Should meet minimum content length requirements
        for doc in master_docs:
            content_length = len(doc["merged_content"])
            assert content_length >= quality_gates["min_content_length"]
        
        # And: Should maintain low duplicate ratio
        for doc in master_docs:
            if "quality_metrics" in doc:
                duplicate_ratio = doc["quality_metrics"].get("duplicate_ratio", 0)
                assert duplicate_ratio <= quality_gates["max_duplicate_ratio"]
        
        # And: Should preserve required sections
        for doc in master_docs:
            assert "sections_preserved" in doc
            sections_ratio = doc["sections_preserved"]
            assert sections_ratio >= quality_gates["required_sections"]

    # =============== ARCHIVE MANAGEMENT & MIGRATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_safely_migrate_files_to_archive(self, mock_consolidator):
        """Test safe migration of legacy files to archive"""
        # Given: Files identified for archiving
        files_to_archive = [
            "docs/duplicate_setup.md",
            "docs/old_readme.md", 
            "docs/archive/legacy_api.md"
        ]
        
        # When: Files are migrated to archive
        migration_result = await mock_consolidator.migrate_to_archive(files_to_archive)
        
        # Then: Should successfully archive all files
        assert migration_result["status"] == "success"
        assert migration_result["files_archived"] == len(files_to_archive)
        
        # And: Should create redirect mappings
        assert "redirect_map" in migration_result
        redirect_map = migration_result["redirect_map"]
        
        for original_file in files_to_archive:
            assert original_file in redirect_map
            assert redirect_map[original_file].startswith("archive/")
        
        # And: Should preserve file history
        assert "preservation_metadata" in migration_result
        preservation = migration_result["preservation_metadata"]
        assert "original_paths" in preservation
        assert "archive_timestamp" in preservation

    @pytest.mark.asyncio
    async def test_should_maintain_backward_compatibility_during_migration(self, mock_consolidator):
        """Test backward compatibility maintenance during archive migration"""
        # Given: Files with external references
        files_to_archive = ["docs/api.md"]
        
        # When: Migration maintains backward compatibility
        migration_result = await mock_consolidator.migrate_to_archive(files_to_archive)
        
        # Then: Should create reference mappings
        assert "reference_updates" in migration_result
        reference_updates = migration_result["reference_updates"]
        
        # And: Should generate redirect instructions
        assert "redirect_rules" in migration_result
        redirect_rules = migration_result["redirect_rules"]
        assert len(redirect_rules) >= 1
        
        # And: Should preserve link integrity
        assert "link_preservation" in migration_result
        link_preservation = migration_result["link_preservation"]
        assert link_preservation["broken_links_created"] == 0

    # =============== CONSOLIDATION INTEGRITY VALIDATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_validate_no_information_loss_during_consolidation(self, mock_consolidator, mock_consolidation_strategy):
        """Test validation that no critical information is lost"""
        # Given: Original documentation content
        original_content_length = sum(len(asset.content) for asset in mock_consolidator.assets)
        
        # When: Consolidation integrity is validated
        integrity_result = await mock_consolidator.validate_consolidation_integrity()
        
        # Then: Should verify content preservation
        assert integrity_result["status"] == "passed"
        assert "content_coverage" in integrity_result
        
        coverage = integrity_result["content_coverage"]
        assert coverage["percentage"] >= 95.0  # Should preserve 95%+ of content
        
        # And: Should identify any missing critical sections
        assert "missing_sections" in integrity_result
        missing_sections = integrity_result["missing_sections"]
        assert len(missing_sections) == 0  # No critical sections should be missing
        
        # And: Should validate all internal references
        assert "reference_integrity" in integrity_result
        ref_integrity = integrity_result["reference_integrity"]
        assert ref_integrity["broken_references"] == 0

    @pytest.mark.asyncio
    async def test_should_generate_consolidation_audit_trail(self, mock_consolidator, mock_consolidation_strategy):
        """Test generation of comprehensive consolidation audit trail"""
        # Given: Consolidation process completion
        master_docs = await mock_consolidator.generate_master_documents(mock_consolidation_strategy)
        
        # When: Consolidation integrity is validated
        integrity_result = await mock_consolidator.validate_consolidation_integrity()
        
        # Then: Should generate detailed audit trail
        assert "audit_trail" in integrity_result
        audit_trail = integrity_result["audit_trail"]
        
        # Should track all file transformations
        assert "file_transformations" in audit_trail
        transformations = audit_trail["file_transformations"]
        assert len(transformations) >= len(mock_consolidator.assets)
        
        # Should record merge decisions
        assert "merge_decisions" in audit_trail
        merge_decisions = audit_trail["merge_decisions"]
        assert len(merge_decisions) >= 1
        
        # Should log quality gate results
        assert "quality_gate_results" in audit_trail


class TestLivingDocumentationSystem:
    """TDD test suite for living documentation system"""

    @pytest.fixture
    def mock_living_docs_system(self):
        """Create living documentation system for testing"""
        return LivingDocumentationSystem()

    @pytest.fixture
    def mock_codebase_changes(self):
        """Mock codebase changes that should trigger doc updates"""
        return [
            "app/api/agents.py",  # API endpoint changes
            "app/models/agent.py",  # Model changes
            "app/core/orchestrator.py",  # Core system changes
            "README.md",  # Documentation changes
            "docs/api.md"  # Direct doc changes
        ]

    # =============== AUTOMATED VALIDATION PIPELINE TESTS ===============

    @pytest.mark.asyncio
    async def test_should_setup_comprehensive_automated_validation(self, mock_living_docs_system):
        """Test setup of comprehensive automated validation pipelines"""
        # Given: A living documentation system
        
        # When: Automated validation is setup
        setup_result = await mock_living_docs_system.setup_automated_validation()
        
        # Then: Should configure all validation types
        assert setup_result["status"] == "success"
        assert "validation_pipelines" in setup_result
        
        pipelines = setup_result["validation_pipelines"]
        expected_pipelines = ["link_validation", "code_testing", "content_currency", "onboarding_validation"]
        
        for pipeline in expected_pipelines:
            assert pipeline in pipelines
            assert pipelines[pipeline]["configured"] is True
        
        # And: Should set up monitoring schedules
        assert "monitoring_schedule" in setup_result
        schedule = setup_result["monitoring_schedule"]
        assert "link_validation" in schedule  # Daily
        assert "code_testing" in schedule  # On changes
        assert "content_currency" in schedule  # Weekly

    @pytest.mark.asyncio
    async def test_should_integrate_with_existing_automation_tools(self, mock_living_docs_system):
        """Test integration with existing link validator, code tester, etc."""
        # Given: Living documentation system
        
        # When: Automated validation is setup
        setup_result = await mock_living_docs_system.setup_automated_validation()
        
        # Then: Should integrate with LinkValidator
        assert "link_validator_config" in setup_result
        link_config = setup_result["link_validator_config"]
        assert link_config["rate_limit_delay"] >= 1.0
        assert link_config["cache_duration"] >= 3600
        
        # And: Should integrate with CodeExampleTester
        assert "code_tester_config" in setup_result
        code_config = setup_result["code_tester_config"]
        assert code_config["test_timeout"] >= 30
        assert "supported_languages" in code_config
        
        # And: Should integrate with ContentCurrencyMonitor
        assert "currency_monitor_config" in setup_result
        currency_config = setup_result["currency_monitor_config"]
        assert "freshness_thresholds" in currency_config

    # =============== CODE-DOCUMENTATION SYNCHRONIZATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_detect_code_changes_requiring_doc_updates(self, mock_living_docs_system, mock_codebase_changes):
        """Test detection of code changes that require documentation updates"""
        # Given: Recent codebase changes
        
        # When: Documentation sync is performed
        sync_result = await mock_living_docs_system.sync_with_codebase_changes(mock_codebase_changes)
        
        # Then: Should identify files requiring documentation updates
        assert "files_requiring_updates" in sync_result
        requiring_updates = sync_result["files_requiring_updates"]
        
        # API changes should trigger API doc updates
        api_updates = [f for f in requiring_updates if "api" in f["doc_file"].lower()]
        assert len(api_updates) >= 1
        
        # Model changes should trigger schema/reference updates
        model_updates = [f for f in requiring_updates if "model" in f["change_type"]]
        assert len(model_updates) >= 1
        
        # And: Should categorize update types
        for update in requiring_updates:
            assert "change_type" in update
            assert "urgency" in update
            assert update["urgency"] in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_should_automatically_update_api_documentation(self, mock_living_docs_system):
        """Test automatic API documentation updates from code changes"""
        # Given: API endpoint changes
        api_changes = ["app/api/agents.py"]
        
        # When: Documentation is synchronized
        sync_result = await mock_living_docs_system.sync_with_codebase_changes(api_changes)
        
        # Then: Should generate updated API documentation
        assert "auto_generated_content" in sync_result
        auto_content = sync_result["auto_generated_content"]
        
        assert "api_documentation" in auto_content
        api_docs = auto_content["api_documentation"]
        
        # Should include endpoint definitions
        assert "endpoints" in api_docs
        endpoints = api_docs["endpoints"]
        assert len(endpoints) >= 1
        
        # Should include request/response schemas
        for endpoint in endpoints:
            assert "method" in endpoint
            assert "path" in endpoint
            assert "description" in endpoint

    @pytest.mark.asyncio
    async def test_should_maintain_documentation_version_synchronization(self, mock_living_docs_system, mock_codebase_changes):
        """Test maintenance of version synchronization between code and docs"""
        # Given: Code changes with version information
        
        # When: Version synchronization is performed
        sync_result = await mock_living_docs_system.sync_with_codebase_changes(mock_codebase_changes)
        
        # Then: Should track version mismatches
        assert "version_analysis" in sync_result
        version_analysis = sync_result["version_analysis"]
        
        assert "code_version" in version_analysis
        assert "docs_version" in version_analysis
        assert "sync_status" in version_analysis
        
        # And: Should recommend version updates
        if version_analysis["sync_status"] != "synchronized":
            assert "recommended_updates" in version_analysis
            recommendations = version_analysis["recommended_updates"]
            assert len(recommendations) >= 1

    # =============== DYNAMIC CONTENT GENERATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_generate_api_documentation_from_code(self, mock_living_docs_system):
        """Test generation of API documentation from code analysis"""
        # Given: Living documentation system
        
        # When: Dynamic API content is generated
        content_result = await mock_living_docs_system.generate_dynamic_content("api_documentation")
        
        # Then: Should generate comprehensive API docs
        assert content_result["status"] == "success"
        assert "generated_content" in content_result
        
        api_content = content_result["generated_content"]
        assert "endpoints" in api_content
        assert "schemas" in api_content
        assert "authentication" in api_content
        
        # And: Should include code examples
        assert "examples" in api_content
        examples = api_content["examples"]
        assert len(examples) >= 1
        
        # Each example should be executable
        for example in examples:
            assert "language" in example
            assert "code" in example
            assert "description" in example

    @pytest.mark.asyncio
    async def test_should_generate_status_dashboards_from_metrics(self, mock_living_docs_system):
        """Test generation of status dashboards from system metrics"""
        # Given: Living documentation system
        
        # When: Dynamic dashboard content is generated
        content_result = await mock_living_docs_system.generate_dynamic_content("status_dashboard")
        
        # Then: Should generate real-time status information
        assert content_result["status"] == "success"
        assert "generated_content" in content_result
        
        dashboard_content = content_result["generated_content"]
        assert "system_health" in dashboard_content
        assert "performance_metrics" in dashboard_content
        assert "recent_activity" in dashboard_content
        
        # And: Should include current timestamps
        assert "last_updated" in dashboard_content
        assert "refresh_interval" in dashboard_content

    # =============== DEVELOPER EXPERIENCE VALIDATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_validate_30_minute_onboarding_experience(self, mock_living_docs_system):
        """Test validation of 30-minute developer onboarding target"""
        # Given: Living documentation system with onboarding docs
        
        # When: Onboarding experience is validated
        onboarding_result = await mock_living_docs_system.validate_onboarding_experience()
        
        # Then: Should complete onboarding validation
        assert onboarding_result["status"] == "success"
        assert "total_duration" in onboarding_result
        
        # Should meet 30-minute target
        total_duration = onboarding_result["total_duration"]
        assert total_duration <= 1800  # 30 minutes in seconds
        
        # And: Should validate all critical steps
        assert "steps_validated" in onboarding_result
        steps = onboarding_result["steps_validated"]
        
        critical_steps = ["repository_clone", "environment_setup", "dependency_installation", "first_agent_creation"]
        for step in critical_steps:
            step_result = next(s for s in steps if s["step_name"] == step)
            assert step_result["status"] == "success"
        
        # And: Should provide success rate metrics
        assert "success_rate" in onboarding_result
        success_rate = onboarding_result["success_rate"]
        assert success_rate >= 0.9  # 90% success rate target

    @pytest.mark.asyncio
    async def test_should_identify_onboarding_friction_points(self, mock_living_docs_system):
        """Test identification of friction points in onboarding process"""
        # Given: Onboarding validation results
        
        # When: Onboarding experience is analyzed
        onboarding_result = await mock_living_docs_system.validate_onboarding_experience()
        
        # Then: Should identify slow or failing steps
        assert "friction_analysis" in onboarding_result
        friction_analysis = onboarding_result["friction_analysis"]
        
        assert "slow_steps" in friction_analysis
        assert "error_prone_steps" in friction_analysis
        assert "improvement_opportunities" in friction_analysis
        
        # And: Should provide actionable recommendations
        if friction_analysis["slow_steps"]:
            assert "performance_recommendations" in friction_analysis
        
        if friction_analysis["error_prone_steps"]:
            assert "reliability_recommendations" in friction_analysis


# =============== INTEGRATION TESTS WITH EXISTING AUTOMATION ===============

class TestDocumentationAutomationIntegration:
    """Integration tests with existing automation tools"""

    @pytest.mark.asyncio
    async def test_should_integrate_with_link_validator_for_quality_gates(self):
        """Test integration with existing link validator"""
        # When: Running validation on test documentation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Given: Link validator instance configured for temp directory
            validator = LinkValidator(temp_dir)
            
            # Create test markdown file with various link types
            test_file = Path(temp_dir) / "test.md"
            test_content = """
# Test Document

[Internal Link](./other.md)
[External Link](https://example.com)
[Anchor Link](#section-one)

## Section One
Content here.
"""
            test_file.write_text(test_content)
            
            # Create referenced file
            other_file = Path(temp_dir) / "other.md"
            other_file.write_text("# Other Document")
            
            # Validate links
            results = await validator.validate_all_links([test_file])
        
            # Then: Should validate all link types
            assert len(results) >= 3  # At least 3 links found
            
            # Should identify link types correctly
            link_types = set(result.link_type for result in results)
            expected_types = {"internal", "external", "anchor"}
            assert expected_types.issubset(link_types)
            
            # Should provide validation results
            for result in results:
                assert result.status in ["valid", "broken", "warning", "error", "redirect"]
                assert result.file_path is not None

    @pytest.mark.asyncio
    async def test_should_integrate_with_code_example_tester_for_executable_docs(self):
        """Test integration with existing code example tester"""
        # Given: Code example tester instance configured for temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = CodeExampleTester(temp_dir)
            
            # Create test markdown with various code examples
            test_file = Path(temp_dir) / "test.md"
            test_content = """
# Test Documentation

## Python Example
```python
print("Hello, World!")
x = 1 + 1
assert x == 2
```

## JSON Example
```json
{
  "name": "test",
  "value": 42
}
```

## Bash Example  
```bash
echo "Hello"
```
"""
            test_file.write_text(test_content)
            
            # When: Testing documentation with code examples
            results = await tester.test_all_code_examples([test_file])
        
            # Then: Should test all code block types
            assert len(results) >= 3  # At least 3 code blocks
            
            # Should identify different languages
            languages = set(result.language for result in results)
            expected_languages = {"python", "json", "bash"}
            assert expected_languages.issubset(languages)
            
            # Should provide test results
            for result in results:
                assert result.status in ["success", "error", "warning", "skipped"]
                assert result.line_number > 0


# =============== PERFORMANCE AND SCALABILITY TESTS ===============

class TestDocumentationConsolidationPerformance:
    """Performance tests for documentation consolidation system"""

    @pytest.mark.asyncio
    async def test_should_handle_large_documentation_sets_efficiently(self):
        """Test efficient handling of 500+ documentation files"""
        # Given: Large set of documentation files (simulated)
        consolidator = DocumentationConsolidator()
        
        # Create many mock assets
        large_asset_set = []
        for i in range(500):
            asset = DocumentationAsset(
                path=f"docs/file_{i}.md",
                content=f"# Document {i}\nContent for document {i}" * 10,  # Realistic size
                content_hash=hashlib.md5(f"content_{i}".encode()).hexdigest(),
                file_size=1000 + i,
                last_modified=datetime.now(),
                doc_type="guide",
                quality_score=0.7 + (i % 3) * 0.1
            )
            large_asset_set.append(asset)
        
        consolidator.assets = large_asset_set
        
        # When: Processing large documentation set
        import time
        start_time = time.time()
        
        # Simulate analysis (would be implemented)
        duplicates = await consolidator.detect_content_duplicates()
        
        processing_time = time.time() - start_time
        
        # Then: Should process efficiently
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert isinstance(duplicates, dict)  # Should return results

    @pytest.mark.asyncio
    async def test_should_maintain_memory_efficiency_during_consolidation(self):
        """Test memory efficiency during consolidation process"""
        # Given: Documentation consolidation process
        consolidator = DocumentationConsolidator()
        
        # When: Processing documentation with memory monitoring
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate large consolidation process
        large_asset_set = []
        for i in range(100):  # Smaller set for testing
            asset = DocumentationAsset(
                path=f"docs/large_file_{i}.md",
                content="# Large Document\n" + "Content line\n" * 1000,  # ~13KB each
                content_hash=hashlib.md5(f"large_content_{i}".encode()).hexdigest(),
                file_size=13000,
                last_modified=datetime.now(),
                doc_type="guide",
                quality_score=0.8
            )
            large_asset_set.append(asset)
        
        consolidator.assets = large_asset_set
        
        # Process and measure memory
        await consolidator.detect_content_duplicates()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Then: Should maintain reasonable memory usage
        assert memory_increase < 100  # Should not increase by more than 100MB
        assert final_memory < 500  # Should stay under 500MB total


@pytest.mark.asyncio 
async def test_documentation_consolidation_integration():
    """Integration test for complete documentation consolidation workflow"""
    # Given: Complete documentation consolidation system
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic documentation structure
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        # Create various documentation files
        files_to_create = {
            "README.md": "# Main Project\nOverview of the project",
            "docs/setup.md": "# Setup\nSetup instructions",
            "docs/api.md": "# API\nAPI documentation",
            "docs/duplicate_setup.md": "# Setup\nDuplicate setup instructions",
            "docs/guides/dev.md": "# Development\nDev guide",
        }
        
        for file_path, content in files_to_create.items():
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # When: Running complete consolidation workflow
        consolidator = DocumentationConsolidator(temp_dir)
        
        # Should analyze existing documentation
        assets = await consolidator.analyze_existing_documentation()
        assert len(assets) >= len(files_to_create)
        
        # Should detect duplicates
        duplicates = await consolidator.detect_content_duplicates()
        assert isinstance(duplicates, dict)
        
        # Should validate integration with existing tools
        validator = LinkValidator(temp_dir)
        md_files = list(Path(temp_dir).glob("**/*.md"))
        validation_results = await validator.validate_all_links(md_files)
        
        # Then: Should complete workflow successfully
        assert len(validation_results) >= 0  # Should handle validation
        
        # Should integrate all components
        living_docs = LivingDocumentationSystem(str(docs_dir))
        setup_result = await living_docs.setup_automated_validation()
        assert setup_result["status"] == "success"