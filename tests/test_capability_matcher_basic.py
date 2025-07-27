"""
Basic tests for Capability Matcher functionality.
"""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.capability_matcher import CapabilityMatcher, MatchingAlgorithm
from app.models.task import TaskPriority


class TestCapabilityMatcherBasic:
    """Basic test suite for capability matching."""
    
    @pytest.fixture
    def capability_matcher(self):
        """Create capability matcher instance."""
        return CapabilityMatcher()
    
    @pytest.fixture
    def sample_agent_capabilities(self):
        """Sample agent capabilities for testing."""
        return [
            {
                "name": "python_development",
                "description": "Python backend development",
                "confidence_level": 0.9,
                "specialization_areas": ["fastapi", "sqlalchemy", "pytest"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_exact_match_capabilities(self, capability_matcher, sample_agent_capabilities):
        """Test exact capability matching algorithm."""
        requirements = ["python_development"]
        score = await capability_matcher._exact_match_capabilities(requirements, sample_agent_capabilities)
        assert score == 1.0  # Perfect match
        
        requirements = ["javascript"]
        score = await capability_matcher._exact_match_capabilities(requirements, sample_agent_capabilities)
        assert score == 0.0  # No match
    
    @pytest.mark.asyncio
    async def test_weighted_match_capabilities(self, capability_matcher, sample_agent_capabilities):
        """Test weighted capability matching algorithm."""
        requirements = ["python_development"]
        score = await capability_matcher._weighted_match_capabilities(requirements, sample_agent_capabilities)
        assert score > 0.8  # High score for direct match
    
    @pytest.mark.asyncio 
    async def test_match_capabilities_main_method(self, capability_matcher, sample_agent_capabilities):
        """Test main match_capabilities method."""
        requirements = ["python_development"]
        score = await capability_matcher.match_capabilities(requirements, sample_agent_capabilities)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should have high match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])