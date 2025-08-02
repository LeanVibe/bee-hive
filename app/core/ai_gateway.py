"""
AI Model Gateway for LeanVibe Agent Hive 2.0

Unified abstraction layer for multiple AI models (Claude, GPT-4, Gemini)
with reliable error handling, cost management, and autonomous development optimization.

Based on Gemini CLI strategic analysis recommendations.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class AIModel(Enum):
    """Supported AI models for autonomous development."""
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GEMINI_PRO = "gemini-pro"


class TaskType(Enum):
    """Types of autonomous development tasks."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review" 
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"


@dataclass
class AIRequest:
    """Structured request for AI model interaction."""
    task_id: str
    agent_id: str
    task_type: TaskType
    prompt: str
    model: AIModel
    context: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIResponse:
    """Structured response from AI model."""
    request_id: str
    task_id: str
    content: str
    model: AIModel
    usage: Dict[str, int]
    cost_estimate: float
    duration_seconds: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingChunk:
    """Chunk of streaming AI response."""
    request_id: str
    content: str
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIModelClient(ABC):
    """Abstract base class for AI model clients."""
    
    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """Generate a complete response."""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: AIRequest) -> AsyncGenerator[StreamingChunk, None]:
        """Generate a streaming response."""
        pass
    
    @abstractmethod
    def estimate_cost(self, request: AIRequest) -> float:
        """Estimate cost for the request."""
        pass


class ClaudeClient(AIModelClient):
    """Anthropic Claude client for autonomous development."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self.session
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, request: AIRequest) -> AIResponse:
        """Generate a complete response from Claude."""
        start_time = time.time()
        session = await self._get_session()
        
        payload = {
            "model": request.model.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user", 
                    "content": self._format_prompt_for_autonomous_development(request)
                }
            ]
        }
        
        try:
            async with session.post(f"{self.base_url}/messages", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                
                duration = time.time() - start_time
                
                return AIResponse(
                    request_id=request.task_id,
                    task_id=request.task_id,
                    content=data["content"][0]["text"],
                    model=request.model,
                    usage=data.get("usage", {}),
                    cost_estimate=self.estimate_cost(request),
                    duration_seconds=duration,
                    metadata={
                        "stop_reason": data.get("stop_reason"),
                        "stop_sequence": data.get("stop_sequence")
                    }
                )
        
        except Exception as e:
            logger.error("Claude API error", error=str(e), task_id=request.task_id)
            raise
    
    async def generate_stream(self, request: AIRequest) -> AsyncGenerator[StreamingChunk, None]:
        """Generate streaming response from Claude."""
        session = await self._get_session()
        
        payload = {
            "model": request.model.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": self._format_prompt_for_autonomous_development(request)
                }
            ]
        }
        
        try:
            async with session.post(f"{self.base_url}/messages", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                yield StreamingChunk(
                                    request_id=request.task_id,
                                    content="",
                                    is_final=True
                                )
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "content_block_delta":
                                    content = data.get("delta", {}).get("text", "")
                                    if content:
                                        yield StreamingChunk(
                                            request_id=request.task_id,
                                            content=content
                                        )
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error("Claude streaming error", error=str(e), task_id=request.task_id)
            raise
    
    def estimate_cost(self, request: AIRequest) -> float:
        """Estimate cost for Claude request."""
        # Claude 3.5 Sonnet pricing (approximate)
        input_cost_per_token = 0.000003  # $3 per 1M input tokens
        output_cost_per_token = 0.000015  # $15 per 1M output tokens
        
        # Rough estimate based on prompt length and max_tokens
        input_tokens = len(request.prompt.split()) * 1.3  # Rough token estimation
        output_tokens = request.max_tokens * 0.8  # Assume 80% utilization
        
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
    
    def _format_prompt_for_autonomous_development(self, request: AIRequest) -> str:
        """Format prompt with autonomous development context."""
        context_str = ""
        if request.context:
            context_str = f"\n\nContext:\n{json.dumps(request.context, indent=2)}"
        
        task_guidance = {
            TaskType.CODE_GENERATION: "Generate production-ready code with proper error handling, documentation, and following best practices.",
            TaskType.CODE_REVIEW: "Provide detailed code review focusing on correctness, security, performance, and maintainability.",
            TaskType.TESTING: "Create comprehensive tests with good coverage, including edge cases and error scenarios.",
            TaskType.DOCUMENTATION: "Write clear, comprehensive documentation with examples and usage patterns.",
            TaskType.ARCHITECTURE: "Design scalable, maintainable architecture following industry best practices.",
            TaskType.DEBUGGING: "Analyze the issue systematically and provide step-by-step debugging guidance."
        }
        
        guidance = task_guidance.get(request.task_type, "Complete the autonomous development task with high quality.")
        
        return f"""You are an expert autonomous development agent specializing in {request.task_type.value}.

Task: {guidance}

{request.prompt}{context_str}

Please provide a complete, production-ready response."""

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


class AIGateway:
    """Central gateway for AI model interactions with cost management and reliability."""
    
    def __init__(self):
        self.clients: Dict[AIModel, AIModelClient] = {}
        self.request_cache: Dict[str, AIResponse] = {}
        self.cost_tracker: Dict[str, float] = {}  # agent_id -> total_cost
        self.rate_limits: Dict[str, List[float]] = {}  # agent_id -> request_timestamps
        
        # Initialize clients based on available API keys
        if settings.ANTHROPIC_API_KEY:
            self.clients[AIModel.CLAUDE_3_5_SONNET] = ClaudeClient(settings.ANTHROPIC_API_KEY)
            logger.info("✅ Claude client initialized")
        
        if not self.clients:
            logger.warning("⚠️  No AI model clients available. Please configure API keys.")
    
    async def generate(self, request: AIRequest) -> AIResponse:
        """Generate response with automatic client selection and error handling."""
        
        # Check rate limits
        if not self._check_rate_limit(request.agent_id):
            raise Exception(f"Rate limit exceeded for agent {request.agent_id}")
        
        # Check cost limits
        estimated_cost = self._estimate_request_cost(request)
        if not self._check_cost_limit(request.agent_id, estimated_cost):
            raise Exception(f"Cost limit exceeded for agent {request.agent_id}")
        
        # Get appropriate client
        client = self._get_client(request.model)
        
        try:
            # Generate response
            response = await client.generate(request)
            
            # Track costs and usage
            self._track_usage(request.agent_id, response.cost_estimate)
            self._track_rate_limit(request.agent_id)
            
            # Cache response
            self.request_cache[request.task_id] = response
            
            logger.info(
                "AI response generated",
                task_id=request.task_id,
                model=request.model.value,
                duration=response.duration_seconds,
                cost=response.cost_estimate
            )
            
            return response
        
        except Exception as e:
            logger.error(
                "AI generation failed",
                task_id=request.task_id,
                model=request.model.value,
                error=str(e)
            )
            raise
    
    async def generate_stream(self, request: AIRequest) -> AsyncGenerator[StreamingChunk, None]:
        """Generate streaming response."""
        
        if not self._check_rate_limit(request.agent_id):
            raise Exception(f"Rate limit exceeded for agent {request.agent_id}")
        
        client = self._get_client(request.model)
        
        try:
            self._track_rate_limit(request.agent_id)
            
            async for chunk in client.generate_stream(request):
                yield chunk
        
        except Exception as e:
            logger.error(
                "AI streaming failed",
                task_id=request.task_id,
                model=request.model.value,
                error=str(e)
            )
            raise
    
    def _get_client(self, model: AIModel) -> AIModelClient:
        """Get client for specified model."""
        if model not in self.clients:
            # Fallback to available client
            if self.clients:
                fallback_model = next(iter(self.clients.keys()))
                logger.warning(
                    f"Model {model.value} not available, using {fallback_model.value}"
                )
                return self.clients[fallback_model]
            else:
                raise Exception("No AI model clients available")
        
        return self.clients[model]
    
    def _estimate_request_cost(self, request: AIRequest) -> float:
        """Estimate cost for request."""
        client = self._get_client(request.model)
        return client.estimate_cost(request)
    
    def _check_rate_limit(self, agent_id: str, max_requests_per_minute: int = 20) -> bool:
        """Check if agent is within rate limits."""
        now = time.time()
        
        if agent_id not in self.rate_limits:
            self.rate_limits[agent_id] = []
        
        # Clean old timestamps
        self.rate_limits[agent_id] = [
            ts for ts in self.rate_limits[agent_id] 
            if now - ts < 60  # Keep only last minute
        ]
        
        return len(self.rate_limits[agent_id]) < max_requests_per_minute
    
    def _track_rate_limit(self, agent_id: str):
        """Track rate limit usage."""
        now = time.time()
        if agent_id not in self.rate_limits:
            self.rate_limits[agent_id] = []
        self.rate_limits[agent_id].append(now)
    
    def _check_cost_limit(self, agent_id: str, estimated_cost: float, max_cost_per_hour: float = 10.0) -> bool:
        """Check if agent is within cost limits."""
        current_cost = self.cost_tracker.get(agent_id, 0.0)
        return current_cost + estimated_cost <= max_cost_per_hour
    
    def _track_usage(self, agent_id: str, cost: float):
        """Track cost usage."""
        if agent_id not in self.cost_tracker:
            self.cost_tracker[agent_id] = 0.0
        self.cost_tracker[agent_id] += cost
    
    async def close(self):
        """Close all clients."""
        for client in self.clients.values():
            await client.close()


# Global AI Gateway instance
_ai_gateway: Optional[AIGateway] = None


async def get_ai_gateway() -> AIGateway:
    """Get or create AI Gateway instance."""
    global _ai_gateway
    if _ai_gateway is None:
        _ai_gateway = AIGateway()
    return _ai_gateway


async def close_ai_gateway():
    """Close AI Gateway and cleanup resources."""
    global _ai_gateway
    if _ai_gateway:
        await _ai_gateway.close()
        _ai_gateway = None