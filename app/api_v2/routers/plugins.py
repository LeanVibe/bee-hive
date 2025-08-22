"""
Plugin Marketplace API Endpoints for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.2

RESTful API endpoints for comprehensive plugin marketplace operations including
registry management, AI-powered discovery, security certification, and developer onboarding.

Endpoints:
- Plugin Registry: /api/v2/plugins/
- Discovery: /api/v2/plugins/discover/
- Security: /api/v2/plugins/security/
- Developer: /api/v2/plugins/developer/

Epic 1 Preservation:
- <50ms API response times
- <80MB memory usage with efficient caching
- Non-blocking operations with async/await
- Optimized serialization and pagination
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, Form, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ...core.logging_service import get_component_logger
from ...core.plugin_marketplace import (
    PluginMarketplace, MarketplacePluginEntry, SearchQuery, SearchResult,
    PluginCategory, CertificationLevel, PluginStatus, PluginRating
)
from ...core.ai_plugin_discovery import (
    AIPluginDiscovery, RecommendationType, PluginRecommendation,
    PluginCompatibility, CompatibilityLevel
)
from ...core.security_certification_pipeline import (
    SecurityCertificationPipeline, CertificationReport, QualityGateStatus
)
from ...core.developer_onboarding_platform import (
    DeveloperOnboardingPlatform, DeveloperProfile, PluginSubmission,
    SubmissionStatus, DeveloperAnalytics, DeveloperTier
)
from ...core.orchestrator_plugins import PluginMetadata, PluginType

logger = get_component_logger("plugin_marketplace_api")

# Initialize router
router = APIRouter(prefix="/api/v2/plugins", tags=["plugins"])
security = HTTPBearer()

# Global instances (in production, these would be dependency-injected)
marketplace_instance = None
ai_discovery_instance = None
certification_pipeline_instance = None
developer_platform_instance = None


# Pydantic models for API schemas
class PluginSearchRequest(BaseModel):
    """Request model for plugin search."""
    query: str = Field(..., description="Search query")
    category: Optional[PluginCategory] = Field(None, description="Plugin category filter")
    certification_level: Optional[CertificationLevel] = Field(None, description="Minimum certification level")
    status: Optional[PluginStatus] = Field(None, description="Plugin status filter")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Results offset for pagination")
    sort_by: str = Field("relevance", description="Sort criteria")


class PluginRegistrationRequest(BaseModel):
    """Request model for plugin registration."""
    plugin_id: str = Field(..., description="Unique plugin identifier")
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    plugin_type: PluginType = Field(..., description="Plugin type")
    category: PluginCategory = Field(..., description="Plugin category")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")
    configuration_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")


class PluginDiscoveryRequest(BaseModel):
    """Request model for AI-powered plugin discovery."""
    query: str = Field(..., description="Natural language query")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context for personalization")
    recommendation_types: List[RecommendationType] = Field(
        default_factory=lambda: [RecommendationType.SIMILAR, RecommendationType.COMPLEMENTARY],
        description="Types of recommendations to include"
    )
    limit: int = Field(10, ge=1, le=50, description="Maximum number of recommendations")


class PluginCertificationRequest(BaseModel):
    """Request model for plugin certification."""
    plugin_id: str = Field(..., description="Plugin identifier")
    target_level: CertificationLevel = Field(..., description="Target certification level")
    compliance_standards: List[str] = Field(default_factory=list, description="Required compliance standards")


class DeveloperRegistrationRequest(BaseModel):
    """Request model for developer registration."""
    username: str = Field(..., min_length=3, max_length=30, description="Developer username")
    email: str = Field(..., description="Developer email address")
    full_name: str = Field(..., description="Developer full name")
    company_name: Optional[str] = Field(None, description="Company name")
    github_profile: Optional[str] = Field(None, description="GitHub profile URL")
    
    @validator('email')
    def validate_email(cls, v):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v


class PluginSubmissionRequest(BaseModel):
    """Request model for plugin submission."""
    plugin_metadata: Dict[str, Any] = Field(..., description="Plugin metadata")
    submission_notes: Optional[str] = Field(None, description="Submission notes")


class PluginRatingRequest(BaseModel):
    """Request model for plugin rating."""
    plugin_id: str = Field(..., description="Plugin identifier")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating score (1-5)")
    review_text: Optional[str] = Field(None, max_length=1000, description="Review text")


# Dependency injection functions
async def get_marketplace() -> PluginMarketplace:
    """Get marketplace instance."""
    global marketplace_instance
    if marketplace_instance is None:
        # Initialize marketplace (simplified for this implementation)
        from ...core.advanced_plugin_manager import AdvancedPluginManager
        from ...core.plugin_security_framework import PluginSecurityFramework
        
        plugin_manager = AdvancedPluginManager()
        security_framework = PluginSecurityFramework()
        marketplace_instance = PluginMarketplace(plugin_manager, security_framework)
        await marketplace_instance.initialize()
    
    return marketplace_instance


async def get_ai_discovery(marketplace: PluginMarketplace = Depends(get_marketplace)) -> AIPluginDiscovery:
    """Get AI discovery instance."""
    global ai_discovery_instance
    if ai_discovery_instance is None:
        ai_discovery_instance = AIPluginDiscovery(marketplace)
        await ai_discovery_instance.initialize()
    
    return ai_discovery_instance


async def get_certification_pipeline() -> SecurityCertificationPipeline:
    """Get certification pipeline instance."""
    global certification_pipeline_instance
    if certification_pipeline_instance is None:
        certification_pipeline_instance = SecurityCertificationPipeline()
        await certification_pipeline_instance.initialize()
    
    return certification_pipeline_instance


async def get_developer_platform(
    marketplace: PluginMarketplace = Depends(get_marketplace),
    certification_pipeline: SecurityCertificationPipeline = Depends(get_certification_pipeline)
) -> DeveloperOnboardingPlatform:
    """Get developer platform instance."""
    global developer_platform_instance
    if developer_platform_instance is None:
        developer_platform_instance = DeveloperOnboardingPlatform(marketplace, certification_pipeline)
    
    return developer_platform_instance


async def get_developer_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform)
) -> DeveloperProfile:
    """Get developer profile from authentication token."""
    # Simplified authentication (in production, would verify JWT token)
    api_key = credentials.credentials
    
    # Find developer by API key
    for developer in developer_platform._developers.values():
        if developer.api_key == api_key:
            return developer
    
    raise HTTPException(status_code=401, detail="Invalid authentication token")


# Plugin Registry Endpoints
@router.get("/", response_model=Dict[str, Any])
async def search_plugins(
    q: str = Query(..., description="Search query"),
    category: Optional[PluginCategory] = Query(None, description="Category filter"),
    certification_level: Optional[CertificationLevel] = Query(None, description="Certification filter"),
    status: Optional[PluginStatus] = Query(None, description="Status filter"),
    limit: int = Query(20, ge=1, le=100, description="Results limit"),
    offset: int = Query(0, ge=0, description="Results offset"),
    sort_by: str = Query("relevance", description="Sort criteria"),
    marketplace: PluginMarketplace = Depends(get_marketplace)
):
    """
    Search plugins in the marketplace.
    
    Epic 1: Target <50ms response time
    """
    try:
        start_time = datetime.utcnow()
        
        # Create search query
        search_query = SearchQuery(
            query=q,
            category=category,
            certification_level=certification_level,
            status=status,
            limit=limit,
            offset=offset,
            sort_by=sort_by
        )
        
        # Execute search
        search_result = await marketplace.search_plugins(search_query)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugins": [plugin.to_dict() for plugin in search_result.plugins],
            "total_count": search_result.total_count,
            "has_more": search_result.has_more,
            "query": search_query.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin search failed", query=q, error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def register_plugin(
    request: PluginRegistrationRequest,
    marketplace: PluginMarketplace = Depends(get_marketplace),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Register a new plugin in the marketplace.
    
    Epic 1: Target <50ms registration time
    """
    try:
        start_time = datetime.utcnow()
        
        # Create plugin metadata
        plugin_metadata = PluginMetadata(
            plugin_id=request.plugin_id,
            name=request.name,
            version=request.version,
            description=request.description,
            author=request.author,
            plugin_type=request.plugin_type,
            dependencies=request.dependencies,
            configuration_schema=request.configuration_schema,
            permissions=request.permissions
        )
        
        # Register plugin
        registration_result = await marketplace.register_plugin(
            plugin_metadata,
            developer_id=developer.developer_id,
            category=request.category,
            tags=request.tags
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": registration_result.success,
            "plugin_id": registration_result.plugin_id,
            "message": registration_result.message,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin registration failed", plugin_id=request.plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin_details(
    plugin_id: str,
    marketplace: PluginMarketplace = Depends(get_marketplace)
):
    """
    Get detailed information about a specific plugin.
    
    Epic 1: Target <50ms response time
    """
    try:
        start_time = datetime.utcnow()
        
        # Get plugin details
        plugin_entry = await marketplace.get_plugin_details(plugin_id)
        
        if not plugin_entry:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugin": plugin_entry.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get plugin details failed", plugin_id=plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get plugin details: {str(e)}")


@router.post("/{plugin_id}/install", response_model=Dict[str, Any])
async def install_plugin(
    plugin_id: str,
    target_system: str = "default",
    marketplace: PluginMarketplace = Depends(get_marketplace),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Install a plugin from the marketplace.
    
    Epic 1: Target <500ms installation time
    """
    try:
        start_time = datetime.utcnow()
        
        # Install plugin
        installation_result = await marketplace.install_plugin(plugin_id, target_system)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": installation_result.success,
            "plugin_id": installation_result.plugin_id,
            "installation_id": installation_result.installation_id,
            "message": installation_result.message,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin installation failed", plugin_id=plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Installation failed: {str(e)}")


# AI-Powered Discovery Endpoints
@router.post("/discover", response_model=Dict[str, Any])
async def discover_plugins_ai(
    request: PluginDiscoveryRequest,
    ai_discovery: AIPluginDiscovery = Depends(get_ai_discovery)
):
    """
    AI-powered intelligent plugin discovery and recommendations.
    
    Epic 1: Target <50ms AI inference time
    """
    try:
        start_time = datetime.utcnow()
        
        # Perform AI discovery
        discovery_result = await ai_discovery.discover_plugins_intelligent(
            query=request.query,
            user_context=request.user_context,
            filters={"recommendation_types": [rt.value for rt in request.recommendation_types]},
            limit=request.limit
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "recommendations": discovery_result.get("recommendations", []),
            "similar_plugins": discovery_result.get("similar_plugins", []),
            "complementary_plugins": discovery_result.get("complementary_plugins", []),
            "trending_plugins": discovery_result.get("trending_plugins", []),
            "query_analysis": discovery_result.get("query_analysis", {}),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("AI plugin discovery failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get("/recommendations/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin_recommendations(
    plugin_id: str,
    recommendation_type: RecommendationType = Query(RecommendationType.SIMILAR),
    limit: int = Query(10, ge=1, le=50),
    ai_discovery: AIPluginDiscovery = Depends(get_ai_discovery)
):
    """
    Get AI-powered recommendations for a specific plugin.
    
    Epic 1: Target <50ms recommendation generation
    """
    try:
        start_time = datetime.utcnow()
        
        # Get recommendations
        recommendations = await ai_discovery.get_plugin_recommendations(
            plugin_id=plugin_id,
            recommendation_type=recommendation_type,
            limit=limit
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugin_id": plugin_id,
            "recommendation_type": recommendation_type.value,
            "recommendations": [rec.to_dict() for rec in recommendations],
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin recommendations failed", plugin_id=plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.get("/compatibility/{plugin_a}/{plugin_b}", response_model=Dict[str, Any])
async def check_plugin_compatibility(
    plugin_a: str,
    plugin_b: str,
    ai_discovery: AIPluginDiscovery = Depends(get_ai_discovery)
):
    """
    Check compatibility between two plugins.
    
    Epic 1: Target <50ms compatibility check
    """
    try:
        start_time = datetime.utcnow()
        
        # Check compatibility
        compatibility = await ai_discovery.check_plugin_compatibility(plugin_a, plugin_b)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugin_a": plugin_a,
            "plugin_b": plugin_b,
            "compatibility": compatibility.to_dict() if compatibility else None,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Compatibility check failed", plugin_a=plugin_a, plugin_b=plugin_b, error=str(e))
        raise HTTPException(status_code=500, detail=f"Compatibility check failed: {str(e)}")


# Security Certification Endpoints
@router.post("/security/certify", response_model=Dict[str, Any])
async def certify_plugin_security(
    request: PluginCertificationRequest,
    source_code: Optional[UploadFile] = File(None),
    certification_pipeline: SecurityCertificationPipeline = Depends(get_certification_pipeline),
    marketplace: PluginMarketplace = Depends(get_marketplace),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Initiate security certification for a plugin.
    
    Epic 1: Target <500ms certification pipeline
    """
    try:
        start_time = datetime.utcnow()
        
        # Get plugin entry
        plugin_entry = await marketplace.get_plugin_details(request.plugin_id)
        if not plugin_entry:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        # Check if developer owns this plugin
        if plugin_entry.developer_id != developer.developer_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Process source code if provided
        source_path = None
        if source_code:
            # Save uploaded file (simplified for this implementation)
            source_path = Path(f"/tmp/{plugin_entry.plugin_id}_source.zip")
            with open(source_path, "wb") as f:
                content = await source_code.read()
                f.write(content)
        
        # Run certification
        certification_report = await certification_pipeline.certify_plugin(
            plugin_entry=plugin_entry,
            target_level=request.target_level,
            source_path=source_path
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "certification_report": certification_report.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plugin certification failed", plugin_id=request.plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Certification failed: {str(e)}")


@router.get("/security/status/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin_security_status(
    plugin_id: str,
    certification_pipeline: SecurityCertificationPipeline = Depends(get_certification_pipeline)
):
    """
    Get security certification status for a plugin.
    
    Epic 1: Target <50ms status check
    """
    try:
        start_time = datetime.utcnow()
        
        # Get certification status
        certification_report = await certification_pipeline.get_plugin_certification_status(plugin_id)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugin_id": plugin_id,
            "certification_report": certification_report.to_dict() if certification_report else None,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Security status check failed", plugin_id=plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# Developer Onboarding Endpoints
@router.post("/developer/register", response_model=Dict[str, Any])
async def register_developer(
    request: DeveloperRegistrationRequest,
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform)
):
    """
    Register a new developer account.
    
    Epic 1: Target <50ms registration time
    """
    try:
        start_time = datetime.utcnow()
        
        # Register developer
        developer_profile = await developer_platform.register_developer(
            username=request.username,
            email=request.email,
            full_name=request.full_name,
            company_name=request.company_name,
            github_profile=request.github_profile
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "developer_profile": developer_profile.to_dict(),
            "api_key": developer_profile.api_key,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Developer registration failed", username=request.username, error=str(e))
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/developer/profile", response_model=Dict[str, Any])
async def get_developer_profile(
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Get developer profile information.
    
    Epic 1: Target <50ms profile retrieval
    """
    try:
        start_time = datetime.utcnow()
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "developer_profile": developer.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Get developer profile failed", developer_id=developer.developer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")


@router.put("/developer/profile", response_model=Dict[str, Any])
async def update_developer_profile(
    updates: Dict[str, Any],
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Update developer profile information.
    
    Epic 1: Target <50ms profile update
    """
    try:
        start_time = datetime.utcnow()
        
        # Update profile
        updated_profile = await developer_platform.update_developer_profile(
            developer.developer_id,
            updates
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "developer_profile": updated_profile.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Developer profile update failed", developer_id=developer.developer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@router.get("/developer/analytics", response_model=Dict[str, Any])
async def get_developer_analytics(
    period_days: int = Query(30, ge=1, le=365),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Get developer analytics dashboard data.
    
    Epic 1: Target <50ms analytics retrieval
    """
    try:
        start_time = datetime.utcnow()
        
        # Get analytics
        analytics = await developer_platform.get_developer_analytics(
            developer.developer_id,
            period_days
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "analytics": analytics.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Developer analytics failed", developer_id=developer.developer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.post("/developer/submissions", response_model=Dict[str, Any])
async def create_plugin_submission(
    request: PluginSubmissionRequest,
    source_code: Optional[UploadFile] = File(None),
    documentation: Optional[UploadFile] = File(None),
    test_cases: Optional[UploadFile] = File(None),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Create a new plugin submission for review.
    
    Epic 1: Target <500ms submission processing
    """
    try:
        start_time = datetime.utcnow()
        
        # Process uploaded files (simplified for this implementation)
        source_path = None
        docs_path = None
        tests_path = None
        
        if source_code:
            source_path = Path(f"/tmp/{developer.developer_id}_source.zip")
            with open(source_path, "wb") as f:
                content = await source_code.read()
                f.write(content)
        
        if documentation:
            docs_path = Path(f"/tmp/{developer.developer_id}_docs.zip")
            with open(docs_path, "wb") as f:
                content = await documentation.read()
                f.write(content)
        
        if test_cases:
            tests_path = Path(f"/tmp/{developer.developer_id}_tests.zip")
            with open(tests_path, "wb") as f:
                content = await test_cases.read()
                f.write(content)
        
        # Create plugin metadata from request
        plugin_metadata = PluginMetadata(**request.plugin_metadata)
        
        # Create submission
        submission = await developer_platform.create_plugin_submission(
            developer_id=developer.developer_id,
            plugin_metadata=plugin_metadata,
            source_code_path=source_path,
            documentation_path=docs_path,
            test_cases_path=tests_path
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "submission": submission.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin submission creation failed", developer_id=developer.developer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Submission creation failed: {str(e)}")


@router.get("/developer/submissions", response_model=Dict[str, Any])
async def get_developer_submissions(
    status: Optional[SubmissionStatus] = Query(None),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Get developer's plugin submissions.
    
    Epic 1: Target <50ms submissions retrieval
    """
    try:
        start_time = datetime.utcnow()
        
        # Get submissions
        submissions = await developer_platform.get_developer_submissions(
            developer.developer_id,
            status_filter=status
        )
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "submissions": [submission.to_dict() for submission in submissions],
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Get submissions failed", developer_id=developer.developer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Submissions retrieval failed: {str(e)}")


@router.post("/developer/submit/{submission_id}", response_model=Dict[str, Any])
async def submit_plugin_for_review(
    submission_id: str,
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform),
    developer: DeveloperProfile = Depends(get_developer_from_token)
):
    """
    Submit plugin for marketplace review.
    
    Epic 1: Target <500ms submission processing
    """
    try:
        start_time = datetime.utcnow()
        
        # Submit for review
        success = await developer_platform.submit_plugin_for_review(submission_id)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": success,
            "submission_id": submission_id,
            "message": "Plugin submitted for review" if success else "Submission failed",
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin submission failed", submission_id=submission_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


# Plugin Reviews and Ratings
@router.post("/reviews", response_model=Dict[str, Any])
async def submit_plugin_rating(
    request: PluginRatingRequest,
    developer: DeveloperProfile = Depends(get_developer_from_token),
    marketplace: PluginMarketplace = Depends(get_marketplace)
):
    """
    Submit a rating and review for a plugin.
    
    Epic 1: Target <50ms rating submission
    """
    try:
        start_time = datetime.utcnow()
        
        # Create rating
        rating = PluginRating(
            user_id=developer.developer_id,
            rating=request.rating,
            review_text=request.review_text
        )
        
        # Submit rating
        success = await marketplace.submit_plugin_rating(request.plugin_id, rating)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": success,
            "plugin_id": request.plugin_id,
            "rating": rating.to_dict(),
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Plugin rating submission failed", plugin_id=request.plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Rating submission failed: {str(e)}")


@router.get("/reviews/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin_reviews(
    plugin_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    marketplace: PluginMarketplace = Depends(get_marketplace)
):
    """
    Get reviews and ratings for a plugin.
    
    Epic 1: Target <50ms reviews retrieval
    """
    try:
        start_time = datetime.utcnow()
        
        # Get reviews
        reviews = await marketplace.get_plugin_reviews(plugin_id, limit, offset)
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "plugin_id": plugin_id,
            "reviews": [review.to_dict() for review in reviews],
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Get plugin reviews failed", plugin_id=plugin_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Reviews retrieval failed: {str(e)}")


# Performance and Health Endpoints
@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    marketplace: PluginMarketplace = Depends(get_marketplace),
    ai_discovery: AIPluginDiscovery = Depends(get_ai_discovery),
    certification_pipeline: SecurityCertificationPipeline = Depends(get_certification_pipeline),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform)
):
    """
    Health check for plugin marketplace services.
    
    Epic 1: Target <50ms health check
    """
    try:
        start_time = datetime.utcnow()
        
        # Get performance metrics
        marketplace_metrics = await marketplace.get_performance_metrics()
        ai_metrics = await ai_discovery.get_performance_metrics()
        certification_metrics = await certification_pipeline.get_performance_metrics()
        developer_metrics = await developer_platform.get_performance_metrics()
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "marketplace": {
                    "status": "online",
                    "metrics": marketplace_metrics
                },
                "ai_discovery": {
                    "status": "online",
                    "metrics": ai_metrics
                },
                "certification_pipeline": {
                    "status": "online",
                    "metrics": certification_metrics
                },
                "developer_platform": {
                    "status": "online",
                    "metrics": developer_metrics
                }
            },
            "epic1_validation": {
                "api_performance_target": "<50ms",
                "current_response_time_ms": round(response_time_ms, 2),
                "target_met": response_time_ms < 50.0
            }
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/stats", response_model=Dict[str, Any])
async def get_marketplace_statistics(
    marketplace: PluginMarketplace = Depends(get_marketplace),
    developer_platform: DeveloperOnboardingPlatform = Depends(get_developer_platform)
):
    """
    Get marketplace statistics and metrics.
    
    Epic 1: Target <50ms statistics retrieval
    """
    try:
        start_time = datetime.utcnow()
        
        # Get marketplace statistics
        stats = await marketplace.get_marketplace_statistics()
        
        # Get developer platform statistics
        dev_stats = {
            "total_developers": len(developer_platform._developers),
            "total_submissions": len(developer_platform._submissions),
            "active_developers": len([d for d in developer_platform._developers.values() 
                                    if (datetime.utcnow() - d.last_active).days < 30])
        }
        
        # Epic 1: Track response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "marketplace_stats": stats,
            "developer_stats": dev_stats,
            "response_time_ms": round(response_time_ms, 2)
        }
        
    except Exception as e:
        logger.error("Get marketplace statistics failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")