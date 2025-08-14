"""
Base schemas for LeanVibe Agent Hive 2.0

Common response schemas and base classes used throughout the application.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for API endpoints."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(default="", description="Optional message describing the result")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data payload")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class SuccessResponse(BaseResponse):
    """Standard success response."""
    
    success: bool = Field(default=True)
    message: str = Field(default="Operation completed successfully")


class ErrorResponse(BaseResponse):
    """Standard error response."""
    
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message describing what went wrong")


class ValidationErrorResponse(BaseResponse):
    """Validation error response with detailed field errors."""
    
    success: bool = Field(default=False)
    error: str = Field(default="Validation failed")
    validation_errors: Dict[str, Any] = Field(default_factory=dict, description="Field-level validation errors")


class PaginatedResponse(BaseResponse):
    """Base response for paginated results."""
    
    total_count: int = Field(..., description="Total number of items available")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages available")
    has_prev: bool = Field(..., description="Whether there are previous pages available")