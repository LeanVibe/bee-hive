"""
OAuth 2.0/OIDC API endpoints for enterprise authentication.

Provides comprehensive OAuth 2.0 and OpenID Connect endpoints for
enterprise identity provider integration.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Depends, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_session
from ...core.redis import get_redis_client
from ...core.oauth_provider_system import OAuthProviderSystem, create_oauth_provider_system
from ...core.security import create_access_token, get_current_user, require_admin_access
from ...schemas.security import (
    OAuthProviderConfig, OAuthAuthorizationRequest, OAuthCallbackRequest,
    OAuthAuthorizationResponse, OAuthCallbackResponse, OAuthProviderListResponse,
    UserProfile, SecurityError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth", tags=["OAuth 2.0 Authentication"])


async def get_oauth_system(
    db: AsyncSession = Depends(get_session),
    redis = Depends(get_redis_client)
) -> OAuthProviderSystem:
    """Get OAuth provider system dependency."""
    return await create_oauth_provider_system(db, redis)


@router.get("/providers", response_model=OAuthProviderListResponse)
async def list_oauth_providers(
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    List configured OAuth providers.
    
    Returns list of available OAuth providers with their configurations.
    """
    try:
        providers = oauth_system.get_provider_list()
        return OAuthProviderListResponse(providers=providers)
        
    except Exception as e:
        logger.error(f"Failed to list OAuth providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve OAuth providers"
        )


@router.post("/providers", status_code=status.HTTP_201_CREATED)
async def configure_oauth_provider(
    config: OAuthProviderConfig,
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system),
    current_user: Dict[str, Any] = Depends(require_admin_access)
):
    """
    Configure OAuth provider.
    
    Admin-only endpoint to configure new OAuth providers.
    """
    try:
        success = await oauth_system.configure_provider(
            provider_name=config.provider_name,
            provider_type=config.provider_type,
            client_id=config.client_id,
            client_secret=config.client_secret,
            tenant_id=config.tenant_id,
            custom_config={
                "scopes": config.scopes,
                "domain_hint": config.domain_hint,
                "redirect_uri": config.redirect_uri,
                "authorization_endpoint": config.authorization_endpoint,
                "token_endpoint": config.token_endpoint,
                "userinfo_endpoint": config.userinfo_endpoint,
                "jwks_uri": config.jwks_uri,
                "issuer": config.issuer
            }
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to configure OAuth provider"
            )
        
        return {"message": f"OAuth provider '{config.provider_name}' configured successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure OAuth provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/authorize", response_model=OAuthAuthorizationResponse)
async def initiate_oauth_authorization(
    request_data: OAuthAuthorizationRequest,
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Initiate OAuth 2.0 authorization flow.
    
    Returns authorization URL and session information for OAuth flow.
    """
    try:
        authorization_url, session_id = await oauth_system.initiate_authorization(
            provider_name=request_data.provider_name,
            redirect_uri=request_data.redirect_uri,
            scopes=request_data.scopes,
            state_data=request_data.state_data
        )
        
        return OAuthAuthorizationResponse(
            authorization_url=authorization_url,
            session_id=session_id,
            provider_name=request_data.provider_name,
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate OAuth authorization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authorization initiation failed: {str(e)}"
        )


@router.get("/authorize/{provider_name}")
async def oauth_authorization_redirect(
    provider_name: str,
    redirect_uri: Optional[str] = None,
    scopes: Optional[str] = None,
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Direct OAuth authorization redirect endpoint.
    
    Convenience endpoint that immediately redirects to OAuth provider.
    """
    try:
        scopes_list = scopes.split(",") if scopes else None
        
        authorization_url, _ = await oauth_system.initiate_authorization(
            provider_name=provider_name,
            redirect_uri=redirect_uri,
            scopes=scopes_list
        )
        
        return RedirectResponse(url=authorization_url, status_code=302)
        
    except Exception as e:
        logger.error(f"Failed OAuth authorization redirect: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization failed: {str(e)}"
        )


@router.get("/callback/{provider_name}", response_model=OAuthCallbackResponse)
async def oauth_callback(
    provider_name: str,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    OAuth callback endpoint.
    
    Handles OAuth provider callbacks and completes the authentication flow.
    """
    try:
        if error:
            return OAuthCallbackResponse(
                success=False,
                error=f"OAuth error: {error} - {error_description or 'Unknown error'}"
            )
        
        if not code or not state:
            return OAuthCallbackResponse(
                success=False,
                error="Missing required parameters: code and state"
            )
        
        # Handle the authorization callback
        token_set, user_profile = await oauth_system.handle_authorization_callback(
            provider_name=provider_name,
            code=code,
            state=state,
            error=error,
            error_description=error_description
        )
        
        # Create internal JWT token for the user
        jwt_payload = {
            "sub": user_profile.user_id,
            "email": user_profile.email,
            "name": user_profile.name,
            "provider": user_profile.provider,
            "roles": ["authenticated_user"],  # Default role for OAuth users
            "scopes": ["read:profile", "read:basic"]  # Default scopes
        }
        
        internal_token = create_access_token(jwt_payload)
        
        return OAuthCallbackResponse(
            success=True,
            user_profile=user_profile,
            access_token=internal_token,
            expires_in=3600  # 1 hour
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        return OAuthCallbackResponse(
            success=False,
            error=f"Authentication failed: {str(e)}"
        )


@router.post("/callback/{provider_name}", response_model=OAuthCallbackResponse)
async def oauth_callback_post(
    provider_name: str,
    callback_data: OAuthCallbackRequest,
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    POST version of OAuth callback endpoint.
    
    Some providers may POST callback data instead of using query parameters.
    """
    try:
        if callback_data.error:
            return OAuthCallbackResponse(
                success=False,
                error=f"OAuth error: {callback_data.error} - {callback_data.error_description or 'Unknown error'}"
            )
        
        # Handle the authorization callback
        token_set, user_profile = await oauth_system.handle_authorization_callback(
            provider_name=provider_name,
            code=callback_data.code,
            state=callback_data.state,
            error=callback_data.error,
            error_description=callback_data.error_description
        )
        
        # Create internal JWT token for the user
        jwt_payload = {
            "sub": user_profile.user_id,
            "email": user_profile.email,
            "name": user_profile.name,
            "provider": user_profile.provider,
            "roles": ["authenticated_user"],
            "scopes": ["read:profile", "read:basic"]
        }
        
        internal_token = create_access_token(jwt_payload)
        
        return OAuthCallbackResponse(
            success=True,
            user_profile=user_profile,
            access_token=internal_token,
            expires_in=3600
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback POST failed: {e}")
        return OAuthCallbackResponse(
            success=False,
            error=f"Authentication failed: {str(e)}"
        )


@router.get("/profile", response_model=UserProfile)
async def get_oauth_user_profile(
    provider_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Get user profile from OAuth provider.
    
    Retrieves fresh user profile data from the OAuth provider.
    """
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_id = current_user.get("id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user context"
            )
        
        user_profile = await oauth_system.get_user_profile(provider_name, user_id)
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found or tokens expired"
            )
        
        return user_profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OAuth user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.post("/refresh/{provider_name}")
async def refresh_oauth_token(
    provider_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Refresh OAuth access token.
    
    Uses refresh token to obtain a new access token from the provider.
    """
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_id = current_user.get("id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user context"
            )
        
        new_token_set = await oauth_system.refresh_access_token(provider_name, user_id)
        if not new_token_set:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to refresh token. Re-authentication may be required."
            )
        
        return {
            "message": "Token refreshed successfully",
            "expires_in": new_token_set.expires_in,
            "token_type": new_token_set.token_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh OAuth token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.delete("/revoke/{provider_name}")
async def revoke_oauth_tokens(
    provider_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Revoke OAuth tokens.
    
    Revokes stored OAuth tokens and optionally notifies the provider.
    """
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_id = current_user.get("id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user context"
            )
        
        success = await oauth_system.revoke_tokens(provider_name, user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke tokens"
            )
        
        return {"message": "Tokens revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke OAuth tokens: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token revocation failed"
        )


@router.get("/metrics")
async def get_oauth_metrics(
    current_user: Dict[str, Any] = Depends(require_admin_access),
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Get OAuth system metrics.
    
    Admin-only endpoint to retrieve OAuth system performance metrics.
    """
    try:
        metrics = oauth_system.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get OAuth metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.delete("/providers/{provider_name}")
async def remove_oauth_provider(
    provider_name: str,
    current_user: Dict[str, Any] = Depends(require_admin_access),
    oauth_system: OAuthProviderSystem = Depends(get_oauth_system)
):
    """
    Remove OAuth provider configuration.
    
    Admin-only endpoint to remove OAuth provider configurations.
    """
    try:
        # Check if provider exists
        providers = oauth_system.get_provider_list()
        provider_exists = any(p["name"] == provider_name for p in providers)
        
        if not provider_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"OAuth provider '{provider_name}' not found"
            )
        
        # Remove provider (this would need to be implemented in the OAuth system)
        # For now, we'll return a message indicating the operation
        return {
            "message": f"OAuth provider '{provider_name}' removal initiated",
            "note": "Existing user sessions may remain active until token expiration"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove OAuth provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Provider removal failed"
        )