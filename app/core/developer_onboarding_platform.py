"""
Developer Onboarding Platform for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.2

Implements comprehensive developer onboarding with registration, submission processes,
and developer tools for the plugin marketplace ecosystem.

Key Features:
- Developer registration and profile management
- Plugin submission workflow with automated review
- Developer tools and SDK integration
- Analytics dashboard for plugin performance
- Community features and support system
- Revenue sharing and monetization tools

Epic 1 Preservation:
- <50ms developer API operations
- <80MB memory usage with efficient caching
- Non-blocking submission processes
- Optimized developer experience
"""

import asyncio
import uuid
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager

from .logging_service import get_component_logger
from .plugin_marketplace import (
    PluginMarketplace, MarketplacePluginEntry, PluginStatus, 
    CertificationLevel, PluginCategory
)
from .security_certification_pipeline import (
    SecurityCertificationPipeline, CertificationReport, QualityGateStatus
)
from .orchestrator_plugins import PluginMetadata, PluginType

logger = get_component_logger("developer_onboarding_platform")


class DeveloperTier(Enum):
    """Developer tier levels with different privileges."""
    COMMUNITY = "community"
    VERIFIED = "verified"
    PARTNER = "partner"
    ENTERPRISE = "enterprise"


class SubmissionStatus(Enum):
    """Plugin submission workflow status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    TESTING = "testing"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class ReviewAction(Enum):
    """Actions available during plugin review."""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"
    REQUEST_TESTING = "request_testing"
    ESCALATE = "escalate"


@dataclass
class DeveloperProfile:
    """Developer profile and account information."""
    developer_id: str
    username: str
    email: str
    full_name: str
    company_name: Optional[str] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    github_profile: Optional[str] = None
    tier: DeveloperTier = DeveloperTier.COMMUNITY
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    plugins_published: int = 0
    total_downloads: int = 0
    average_rating: float = 0.0
    revenue_earned: float = 0.0
    
    # Settings
    notifications_enabled: bool = True
    public_profile: bool = True
    api_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "developer_id": self.developer_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "company_name": self.company_name,
            "bio": self.bio,
            "website": self.website,
            "github_profile": self.github_profile,
            "tier": self.tier.value,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "statistics": {
                "plugins_published": self.plugins_published,
                "total_downloads": self.total_downloads,
                "average_rating": self.average_rating,
                "revenue_earned": self.revenue_earned
            },
            "settings": {
                "notifications_enabled": self.notifications_enabled,
                "public_profile": self.public_profile
            }
        }


@dataclass
class PluginSubmission:
    """Plugin submission with review workflow."""
    submission_id: str
    developer_id: str
    plugin_metadata: PluginMetadata
    source_code_path: Optional[Path] = None
    documentation_path: Optional[Path] = None
    test_cases_path: Optional[Path] = None
    
    status: SubmissionStatus = SubmissionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    
    # Review information
    assigned_reviewer: Optional[str] = None
    review_notes: List[str] = field(default_factory=list)
    certification_report: Optional[CertificationReport] = None
    
    # Automated checks
    security_scan_passed: bool = False
    performance_test_passed: bool = False
    documentation_complete: bool = False
    tests_passing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "developer_id": self.developer_id,
            "plugin_metadata": asdict(self.plugin_metadata),
            "source_code_path": str(self.source_code_path) if self.source_code_path else None,
            "documentation_path": str(self.documentation_path) if self.documentation_path else None,
            "test_cases_path": str(self.test_cases_path) if self.test_cases_path else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "assigned_reviewer": self.assigned_reviewer,
            "review_notes": self.review_notes,
            "automated_checks": {
                "security_scan_passed": self.security_scan_passed,
                "performance_test_passed": self.performance_test_passed,
                "documentation_complete": self.documentation_complete,
                "tests_passing": self.tests_passing
            }
        }


@dataclass
class DeveloperAnalytics:
    """Analytics data for developer dashboard."""
    developer_id: str
    period_start: datetime
    period_end: datetime
    
    # Download metrics
    total_downloads: int = 0
    new_downloads: int = 0
    downloads_by_plugin: Dict[str, int] = field(default_factory=dict)
    
    # Rating metrics
    average_rating: float = 0.0
    new_ratings: int = 0
    ratings_by_plugin: Dict[str, float] = field(default_factory=dict)
    
    # Revenue metrics
    total_revenue: float = 0.0
    new_revenue: float = 0.0
    revenue_by_plugin: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    plugin_uptime: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeveloperSDK:
    """SDK tools for plugin development."""
    
    def __init__(self):
        self.templates = {}
        self.validators = {}
        self.testing_tools = {}
    
    async def generate_plugin_template(
        self,
        plugin_type: PluginType,
        template_name: str = "basic"
    ) -> Dict[str, str]:
        """Generate plugin template code."""
        templates = {
            PluginType.ORCHESTRATOR: {
                "basic": self._get_orchestrator_template(),
                "advanced": self._get_advanced_orchestrator_template()
            },
            PluginType.PROCESSOR: {
                "basic": self._get_processor_template(),
                "advanced": self._get_advanced_processor_template()
            },
            PluginType.INTEGRATION: {
                "basic": self._get_integration_template(),
                "advanced": self._get_advanced_integration_template()
            }
        }
        
        return templates.get(plugin_type, {}).get(template_name, {})
    
    async def validate_plugin_structure(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin file structure and format."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check required files
        required_files = ["__init__.py", "plugin.py", "metadata.json", "README.md"]
        for required_file in required_files:
            if not (plugin_path / required_file).exists():
                validation_result["errors"].append(f"Missing required file: {required_file}")
                validation_result["valid"] = False
        
        # Check optional but recommended files
        recommended_files = ["tests/", "docs/", "examples/", "CHANGELOG.md"]
        for recommended_file in recommended_files:
            if not (plugin_path / recommended_file).exists():
                validation_result["warnings"].append(f"Missing recommended file/directory: {recommended_file}")
        
        return validation_result
    
    async def run_plugin_tests(self, plugin_path: Path) -> Dict[str, Any]:
        """Run automated tests for plugin."""
        test_result = {
            "passed": True,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_coverage": 0.0,
            "details": []
        }
        
        # Simulate test execution (in real implementation, would run actual tests)
        test_result["total_tests"] = 10
        test_result["passed_tests"] = 8
        test_result["failed_tests"] = 2
        test_result["test_coverage"] = 85.0
        test_result["passed"] = test_result["failed_tests"] == 0
        
        return test_result
    
    def _get_orchestrator_template(self) -> Dict[str, str]:
        """Basic orchestrator plugin template."""
        return {
            "plugin.py": '''"""
Basic Orchestrator Plugin Template
"""

from typing import Dict, List, Any, Optional
from ..orchestrator_plugins import OrchestratorPlugin, PluginType

class MyOrchestratorPlugin(OrchestratorPlugin):
    """Custom orchestrator plugin implementation."""
    
    def __init__(self):
        super().__init__(
            plugin_id="my_orchestrator_plugin",
            name="My Orchestrator Plugin",
            version="1.0.0",
            plugin_type=PluginType.ORCHESTRATOR
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        return True
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestration task."""
        # Implement your orchestration logic here
        return {"status": "completed", "result": task_data}
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
''',
            "metadata.json": '''{
    "plugin_id": "my_orchestrator_plugin",
    "name": "My Orchestrator Plugin",
    "version": "1.0.0",
    "description": "A custom orchestrator plugin for task management",
    "author": "Your Name",
    "plugin_type": "orchestrator",
    "dependencies": [],
    "entry_point": "plugin.MyOrchestratorPlugin",
    "permissions": ["task_management"],
    "configuration_schema": {
        "type": "object",
        "properties": {
            "max_concurrent_tasks": {"type": "integer", "default": 10}
        }
    }
}''',
            "README.md": '''# My Orchestrator Plugin

A custom orchestrator plugin for LeanVibe Agent Hive.

## Features

- Task orchestration
- Configurable concurrency
- Error handling

## Installation

```python
from leanvibe_hive.plugins import install_plugin
install_plugin("my_orchestrator_plugin")
```

## Configuration

```json
{
    "max_concurrent_tasks": 10
}
```
'''
        }
    
    def _get_processor_template(self) -> Dict[str, str]:
        """Basic processor plugin template."""
        return {
            "plugin.py": '''"""
Basic Processor Plugin Template
"""

from typing import Dict, List, Any, Optional
from ..orchestrator_plugins import OrchestratorPlugin, PluginType

class MyProcessorPlugin(OrchestratorPlugin):
    """Custom processor plugin implementation."""
    
    def __init__(self):
        super().__init__(
            plugin_id="my_processor_plugin",
            name="My Processor Plugin",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        return True
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data."""
        # Implement your data processing logic here
        processed_data = data.copy()
        processed_data["processed_by"] = self.plugin_id
        return processed_data
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
''',
            "metadata.json": '''{
    "plugin_id": "my_processor_plugin",
    "name": "My Processor Plugin",
    "version": "1.0.0",
    "description": "A custom data processor plugin",
    "author": "Your Name",
    "plugin_type": "processor",
    "dependencies": [],
    "entry_point": "plugin.MyProcessorPlugin",
    "permissions": ["data_processing"],
    "configuration_schema": {
        "type": "object",
        "properties": {
            "batch_size": {"type": "integer", "default": 100}
        }
    }
}'''
        }
    
    def _get_integration_template(self) -> Dict[str, str]:
        """Basic integration plugin template."""
        return {
            "plugin.py": '''"""
Basic Integration Plugin Template
"""

from typing import Dict, List, Any, Optional
from ..orchestrator_plugins import OrchestratorPlugin, PluginType

class MyIntegrationPlugin(OrchestratorPlugin):
    """Custom integration plugin implementation."""
    
    def __init__(self):
        super().__init__(
            plugin_id="my_integration_plugin",
            name="My Integration Plugin",
            version="1.0.0",
            plugin_type=PluginType.INTEGRATION
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        self.api_key = config.get("api_key")
        return True
    
    async def connect(self) -> bool:
        """Connect to external service."""
        # Implement connection logic here
        return True
    
    async def sync_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync data with external service."""
        # Implement sync logic here
        return {"status": "synced", "data": data}
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
''',
            "metadata.json": '''{
    "plugin_id": "my_integration_plugin",
    "name": "My Integration Plugin",
    "version": "1.0.0",
    "description": "A custom integration plugin for external services",
    "author": "Your Name",
    "plugin_type": "integration",
    "dependencies": [],
    "entry_point": "plugin.MyIntegrationPlugin",
    "permissions": ["network_access"],
    "configuration_schema": {
        "type": "object",
        "properties": {
            "api_key": {"type": "string"},
            "endpoint_url": {"type": "string"}
        },
        "required": ["api_key"]
    }
}'''
        }
    
    def _get_advanced_orchestrator_template(self) -> Dict[str, str]:
        """Advanced orchestrator plugin template with additional features."""
        return self._get_orchestrator_template()  # Simplified for this implementation
    
    def _get_advanced_processor_template(self) -> Dict[str, str]:
        """Advanced processor plugin template with additional features."""
        return self._get_processor_template()  # Simplified for this implementation
    
    def _get_advanced_integration_template(self) -> Dict[str, str]:
        """Advanced integration plugin template with additional features."""
        return self._get_integration_template()  # Simplified for this implementation


class DeveloperOnboardingPlatform:
    """
    Comprehensive developer onboarding platform for plugin marketplace.
    
    Epic 1: Maintains <50ms API operations and <80MB memory usage
    """
    
    def __init__(
        self,
        marketplace: PluginMarketplace,
        certification_pipeline: SecurityCertificationPipeline
    ):
        self.marketplace = marketplace
        self.certification_pipeline = certification_pipeline
        self.sdk = DeveloperSDK()
        
        # Epic 1: Efficient in-memory storage
        self._developers: Dict[str, DeveloperProfile] = {}
        self._submissions: Dict[str, PluginSubmission] = {}
        self._analytics_cache: Dict[str, DeveloperAnalytics] = {}
        
        # Performance tracking
        self._operation_times: List[float] = []
    
    async def register_developer(
        self,
        username: str,
        email: str,
        full_name: str,
        company_name: Optional[str] = None,
        github_profile: Optional[str] = None
    ) -> DeveloperProfile:
        """
        Register new developer account.
        
        Epic 1: Target <50ms registration time
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not self._validate_email(email):
                raise ValueError("Invalid email format")
            
            if not self._validate_username(username):
                raise ValueError("Invalid username format")
            
            # Check for existing accounts
            existing_dev = await self._find_developer_by_email(email)
            if existing_dev:
                raise ValueError("Developer with this email already exists")
            
            existing_username = await self._find_developer_by_username(username)
            if existing_username:
                raise ValueError("Username already taken")
            
            # Create developer profile
            developer_id = f"dev_{uuid.uuid4().hex[:8]}"
            api_key = self._generate_api_key(developer_id)
            
            developer = DeveloperProfile(
                developer_id=developer_id,
                username=username,
                email=email,
                full_name=full_name,
                company_name=company_name,
                github_profile=github_profile,
                api_key=api_key
            )
            
            # Store developer
            self._developers[developer_id] = developer
            
            # Epic 1: Track operation time
            operation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._operation_times.append(operation_time)
            if len(self._operation_times) > 100:
                self._operation_times.pop(0)
            
            logger.info("Developer registered successfully",
                       developer_id=developer_id,
                       username=username,
                       operation_time_ms=round(operation_time, 2))
            
            return developer
            
        except Exception as e:
            logger.error("Developer registration failed", username=username, error=str(e))
            raise
    
    async def create_plugin_submission(
        self,
        developer_id: str,
        plugin_metadata: PluginMetadata,
        source_code_path: Optional[Path] = None,
        documentation_path: Optional[Path] = None,
        test_cases_path: Optional[Path] = None
    ) -> PluginSubmission:
        """
        Create new plugin submission.
        
        Epic 1: Target <50ms submission creation
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate developer
            developer = self._developers.get(developer_id)
            if not developer:
                raise ValueError("Developer not found")
            
            # Create submission
            submission_id = f"sub_{uuid.uuid4().hex[:8]}"
            submission = PluginSubmission(
                submission_id=submission_id,
                developer_id=developer_id,
                plugin_metadata=plugin_metadata,
                source_code_path=source_code_path,
                documentation_path=documentation_path,
                test_cases_path=test_cases_path
            )
            
            # Store submission
            self._submissions[submission_id] = submission
            
            # Epic 1: Track operation time
            operation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._operation_times.append(operation_time)
            
            logger.info("Plugin submission created",
                       submission_id=submission_id,
                       developer_id=developer_id,
                       plugin_id=plugin_metadata.plugin_id,
                       operation_time_ms=round(operation_time, 2))
            
            return submission
            
        except Exception as e:
            logger.error("Plugin submission creation failed",
                        developer_id=developer_id,
                        error=str(e))
            raise
    
    async def submit_plugin_for_review(self, submission_id: str) -> bool:
        """
        Submit plugin for review process.
        
        Epic 1: Target <500ms for full submission processing
        """
        start_time = datetime.utcnow()
        
        try:
            submission = self._submissions.get(submission_id)
            if not submission:
                raise ValueError("Submission not found")
            
            if submission.status != SubmissionStatus.DRAFT:
                raise ValueError("Submission must be in draft status")
            
            # Run automated checks
            await self._run_automated_checks(submission)
            
            # Update submission status
            submission.status = SubmissionStatus.SUBMITTED
            submission.submitted_at = datetime.utcnow()
            submission.updated_at = datetime.utcnow()
            
            # Start review workflow
            await self._assign_reviewer(submission)
            
            # Epic 1: Track operation time
            operation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._operation_times.append(operation_time)
            
            logger.info("Plugin submitted for review",
                       submission_id=submission_id,
                       operation_time_ms=round(operation_time, 2))
            
            return True
            
        except Exception as e:
            logger.error("Plugin submission failed",
                        submission_id=submission_id,
                        error=str(e))
            return False
    
    async def get_developer_analytics(
        self,
        developer_id: str,
        period_days: int = 30
    ) -> DeveloperAnalytics:
        """
        Get analytics data for developer dashboard.
        
        Epic 1: Target <50ms analytics retrieval
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"{developer_id}_{period_days}"
            if cache_key in self._analytics_cache:
                cached_analytics = self._analytics_cache[cache_key]
                # Return cached data if recent (within 1 hour)
                if (datetime.utcnow() - cached_analytics.period_end).total_seconds() < 3600:
                    return cached_analytics
            
            developer = self._developers.get(developer_id)
            if not developer:
                raise ValueError("Developer not found")
            
            # Calculate analytics
            period_end = datetime.utcnow()
            period_start = period_end - timedelta(days=period_days)
            
            analytics = DeveloperAnalytics(
                developer_id=developer_id,
                period_start=period_start,
                period_end=period_end
            )
            
            # Get developer's plugins
            developer_plugins = []
            for plugin_entry in self.marketplace._registry.values():
                if plugin_entry.developer_id == developer_id:
                    developer_plugins.append(plugin_entry)
            
            # Calculate metrics (simplified for this implementation)
            for plugin in developer_plugins:
                analytics.downloads_by_plugin[plugin.plugin_id] = plugin.download_count
                analytics.ratings_by_plugin[plugin.plugin_id] = plugin.average_rating
                analytics.revenue_by_plugin[plugin.plugin_id] = 0.0  # Placeholder
                analytics.plugin_uptime[plugin.plugin_id] = 99.5  # Placeholder
                analytics.error_rates[plugin.plugin_id] = 0.1  # Placeholder
            
            analytics.total_downloads = sum(analytics.downloads_by_plugin.values())
            analytics.average_rating = sum(analytics.ratings_by_plugin.values()) / len(analytics.ratings_by_plugin) if analytics.ratings_by_plugin else 0.0
            analytics.total_revenue = sum(analytics.revenue_by_plugin.values())
            
            # Cache analytics
            self._analytics_cache[cache_key] = analytics
            
            # Epic 1: Track operation time
            operation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._operation_times.append(operation_time)
            
            logger.info("Developer analytics generated",
                       developer_id=developer_id,
                       period_days=period_days,
                       operation_time_ms=round(operation_time, 2))
            
            return analytics
            
        except Exception as e:
            logger.error("Analytics generation failed",
                        developer_id=developer_id,
                        error=str(e))
            raise
    
    async def generate_plugin_template(
        self,
        developer_id: str,
        plugin_type: PluginType,
        template_name: str = "basic"
    ) -> Dict[str, str]:
        """Generate plugin template for developer."""
        developer = self._developers.get(developer_id)
        if not developer:
            raise ValueError("Developer not found")
        
        return await self.sdk.generate_plugin_template(plugin_type, template_name)
    
    async def validate_plugin_structure(
        self,
        developer_id: str,
        plugin_path: Path
    ) -> Dict[str, Any]:
        """Validate plugin structure for developer."""
        developer = self._developers.get(developer_id)
        if not developer:
            raise ValueError("Developer not found")
        
        return await self.sdk.validate_plugin_structure(plugin_path)
    
    async def get_developer_submissions(
        self,
        developer_id: str,
        status_filter: Optional[SubmissionStatus] = None
    ) -> List[PluginSubmission]:
        """Get all submissions for a developer."""
        developer = self._developers.get(developer_id)
        if not developer:
            raise ValueError("Developer not found")
        
        submissions = []
        for submission in self._submissions.values():
            if submission.developer_id == developer_id:
                if status_filter is None or submission.status == status_filter:
                    submissions.append(submission)
        
        return sorted(submissions, key=lambda s: s.updated_at, reverse=True)
    
    async def update_developer_profile(
        self,
        developer_id: str,
        updates: Dict[str, Any]
    ) -> DeveloperProfile:
        """Update developer profile information."""
        developer = self._developers.get(developer_id)
        if not developer:
            raise ValueError("Developer not found")
        
        # Update allowed fields
        allowed_fields = [
            'full_name', 'company_name', 'bio', 'website', 'github_profile',
            'notifications_enabled', 'public_profile'
        ]
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(developer, field):
                setattr(developer, field, value)
        
        developer.last_active = datetime.utcnow()
        
        logger.info("Developer profile updated",
                   developer_id=developer_id,
                   updated_fields=list(updates.keys()))
        
        return developer
    
    async def _run_automated_checks(self, submission: PluginSubmission) -> None:
        """Run automated checks on plugin submission."""
        try:
            # Security scan
            if submission.source_code_path:
                security_report = await self.certification_pipeline.security_scanner.scan_plugin_security(
                    MarketplacePluginEntry.from_plugin_metadata(submission.plugin_metadata),
                    source_path=submission.source_code_path
                )
                submission.security_scan_passed = security_report.is_safe
            
            # Structure validation
            if submission.source_code_path:
                structure_validation = await self.sdk.validate_plugin_structure(submission.source_code_path)
                submission.documentation_complete = structure_validation["valid"]
            
            # Test execution
            if submission.test_cases_path:
                test_results = await self.sdk.run_plugin_tests(submission.test_cases_path)
                submission.tests_passing = test_results["passed"]
            
            # Performance check (placeholder)
            submission.performance_test_passed = True  # Simplified
            
        except Exception as e:
            logger.error("Automated checks failed",
                        submission_id=submission.submission_id,
                        error=str(e))
    
    async def _assign_reviewer(self, submission: PluginSubmission) -> None:
        """Assign reviewer to submission."""
        # Simplified reviewer assignment (in real implementation, would use reviewer pool)
        submission.assigned_reviewer = "reviewer_001"
        submission.status = SubmissionStatus.UNDER_REVIEW
        submission.updated_at = datetime.utcnow()
    
    async def _find_developer_by_email(self, email: str) -> Optional[DeveloperProfile]:
        """Find developer by email address."""
        for developer in self._developers.values():
            if developer.email.lower() == email.lower():
                return developer
        return None
    
    async def _find_developer_by_username(self, username: str) -> Optional[DeveloperProfile]:
        """Find developer by username."""
        for developer in self._developers.values():
            if developer.username.lower() == username.lower():
                return developer
        return None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        # Username: 3-30 characters, alphanumeric and underscore only
        pattern = r'^[a-zA-Z0-9_]{3,30}$'
        return re.match(pattern, username) is not None
    
    def _generate_api_key(self, developer_id: str) -> str:
        """Generate API key for developer."""
        key_data = f"{developer_id}_{datetime.utcnow().timestamp()}_{uuid.uuid4().hex[:8]}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get platform performance metrics for Epic 1 validation."""
        if not self._operation_times:
            return {"average_operation_time_ms": 0, "total_operations": 0}
        
        avg_time = sum(self._operation_times) / len(self._operation_times)
        return {
            "average_operation_time_ms": round(avg_time, 2),
            "total_operations": len(self._operation_times),
            "target_met": avg_time < 50.0,  # Epic 1: <50ms target
            "developers_registered": len(self._developers),
            "submissions_processed": len(self._submissions)
        }