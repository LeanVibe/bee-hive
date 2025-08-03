"""
AI Model Integration Service for LeanVibe Agent Hive 2.0

Implements Claude API connectivity, context management, and streaming responses
for autonomous development capabilities. This is the core technical foundation
that enables actual AI-powered autonomous development.

CRITICAL COMPONENT: Without this, autonomous development cannot function.
"""

import asyncio
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import httpx
from contextlib import asynccontextmanager

logger = structlog.get_logger()


class ModelType(Enum):
    """Available AI models for different tasks."""
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022" 
    CLAUDE_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_OPUS = "claude-3-opus-20240229"


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"          # Documentation, simple refactoring
    MEDIUM = "medium"          # Feature implementation, bug fixes
    COMPLEX = "complex"        # Architecture design, complex features
    CRITICAL = "critical"      # Security, compliance, critical systems


@dataclass
class AIRequest:
    """AI request with context and requirements."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request content
    prompt: str = ""
    system_prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    model_type: ModelType = ModelType.CLAUDE_SONNET
    max_tokens: int = 4096
    temperature: float = 0.1
    
    # Task metadata
    task_type: str = "development"
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    pilot_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Streaming configuration
    stream: bool = True
    include_context: bool = True
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AIResponse:
    """AI response with metadata and metrics."""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    
    # Response content
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Response metadata
    model_used: ModelType = ModelType.CLAUDE_SONNET
    tokens_used: int = 0
    response_time_ms: float = 0.0
    
    # Quality metrics
    success: bool = True
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    
    # Context management
    context_preserved: bool = True
    context_compression_ratio: float = 0.0
    
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContextWindow:
    """Intelligent context window management."""
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Context content
    system_context: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    code_context: Dict[str, str] = field(default_factory=dict)
    project_context: Dict[str, Any] = field(default_factory=dict)
    
    # Window management
    max_tokens: int = 100000  # Claude's context window
    current_tokens: int = 0
    compression_threshold: float = 0.8  # Compress when 80% full
    
    # Context optimization
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "current_task": 1.0,
        "recent_conversation": 0.8,
        "project_structure": 0.6,
        "historical_context": 0.4
    })
    
    last_compression: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class AIModelIntegrationService:
    """
    Core AI model integration service providing Claude API connectivity.
    
    Implements intelligent context management, model routing, and streaming
    responses for autonomous development capabilities.
    """
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found - AI capabilities will be limited")
        
        self.base_url = "https://api.anthropic.com/v1"
        self.default_headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Model configuration
        self.model_configs = {
            ModelType.CLAUDE_SONNET: {
                "max_tokens": 8192,
                "recommended_temperature": 0.1,
                "best_for": ["complex_reasoning", "code_generation", "architecture"]
            },
            ModelType.CLAUDE_HAIKU: {
                "max_tokens": 4096, 
                "recommended_temperature": 0.0,
                "best_for": ["simple_tasks", "quick_responses", "documentation"]
            },
            ModelType.CLAUDE_OPUS: {
                "max_tokens": 4096,
                "recommended_temperature": 0.2,
                "best_for": ["creative_tasks", "complex_analysis", "strategic_thinking"]
            }
        }
        
        # Performance tracking
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "context_compression_events": 0
        }
        
        # Context management
        self.active_contexts: Dict[str, ContextWindow] = {}
        
    async def select_optimal_model(self, 
                                 task_type: str,
                                 complexity: TaskComplexity,
                                 context_size: int = 0) -> ModelType:
        """Select optimal model based on task requirements."""
        
        # Model selection logic
        if complexity == TaskComplexity.SIMPLE and context_size < 2000:
            return ModelType.CLAUDE_HAIKU
        elif complexity == TaskComplexity.CRITICAL or "security" in task_type.lower():
            return ModelType.CLAUDE_OPUS
        else:
            return ModelType.CLAUDE_SONNET
    
    async def prepare_request_context(self,
                                    request: AIRequest,
                                    context_window: Optional[ContextWindow] = None) -> Tuple[str, str]:
        """Prepare optimized system and user prompts with context."""
        
        # System prompt construction
        system_prompt = request.system_prompt or """
You are an expert AI software engineer specializing in autonomous development.
You have access to a complete development environment and can:
- Analyze code and requirements
- Generate high-quality, production-ready code
- Implement features following best practices
- Debug and fix issues systematically
- Provide detailed explanations and documentation

Focus on:
- Clean, maintainable code
- Security best practices  
- Performance optimization
- Comprehensive testing
- Clear documentation
"""
        
        # Add context from context window
        if context_window:
            if context_window.project_context:
                system_prompt += f"\n\nProject Context:\n{json.dumps(context_window.project_context, indent=2)}"
            
            if context_window.code_context:
                system_prompt += f"\n\nCode Context:\n"
                for file_path, content in context_window.code_context.items():
                    system_prompt += f"\n{file_path}:\n```\n{content[:2000]}...\n```"
        
        # User prompt with request context
        user_prompt = request.prompt
        if request.context:
            user_prompt += f"\n\nAdditional Context:\n{json.dumps(request.context, indent=2)}"
        
        return system_prompt, user_prompt
    
    async def execute_ai_request(self, request: AIRequest) -> AIResponse:
        """Execute AI request with Claude API."""
        
        if not self.api_key:
            return AIResponse(
                request_id=request.request_id,
                success=False,
                error_message="ANTHROPIC_API_KEY not configured"
            )
        
        start_time = datetime.utcnow()
        
        try:
            # Select optimal model
            optimal_model = await self.select_optimal_model(
                request.task_type, 
                request.complexity
            )
            
            # Get or create context window
            context_key = f"{request.pilot_id}_{request.agent_id}" if request.pilot_id and request.agent_id else "default"
            context_window = self.active_contexts.get(context_key)
            
            # Prepare request context
            system_prompt, user_prompt = await self.prepare_request_context(request, context_window)
            
            # Prepare API request
            api_payload = {
                "model": optimal_model.value,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Add conversation history if available
            if context_window and context_window.conversation_history:
                # Add recent conversation history (last 5 exchanges)
                recent_history = context_window.conversation_history[-10:]
                messages = []
                for exchange in recent_history:
                    messages.append({"role": "user", "content": exchange.get("user", "")})
                    if exchange.get("assistant"):
                        messages.append({"role": "assistant", "content": exchange.get("assistant", "")})
                
                # Add current message
                messages.append({"role": "user", "content": user_prompt})
                api_payload["messages"] = messages
            
            # Execute API request
            async with httpx.AsyncClient(timeout=300.0) as client:
                headers = {
                    **self.default_headers,
                    "x-api-key": self.api_key
                }
                
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=api_payload
                )
                
                response.raise_for_status()
                result = response.json()
            
            # Process response
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            ai_response = AIResponse(
                request_id=request.request_id,
                content=result.get("content", [{}])[0].get("text", ""),
                model_used=optimal_model,
                tokens_used=result.get("usage", {}).get("output_tokens", 0),
                response_time_ms=response_time,
                success=True,
                confidence_score=0.95  # High confidence for successful API responses
            )
            
            # Update context window
            if context_window:
                await self._update_context_window(context_window, request, ai_response)
            
            # Update metrics
            self.request_metrics["total_requests"] += 1
            self.request_metrics["successful_requests"] += 1
            self.request_metrics["average_response_time"] = (
                (self.request_metrics["average_response_time"] * (self.request_metrics["total_requests"] - 1) + response_time) / 
                self.request_metrics["total_requests"]
            )
            
            logger.info(
                "AI request completed successfully",
                request_id=request.request_id,
                model=optimal_model.value,
                tokens_used=ai_response.tokens_used,
                response_time_ms=response_time
            )
            
            return ai_response
            
        except Exception as e:
            error_response = AIResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self.request_metrics["total_requests"] += 1
            
            logger.error(
                "AI request failed",
                request_id=request.request_id,
                error=str(e)
            )
            
            return error_response
    
    async def execute_streaming_request(self, request: AIRequest) -> AsyncGenerator[str, None]:
        """Execute streaming AI request for real-time responses."""
        
        if not self.api_key:
            yield "Error: ANTHROPIC_API_KEY not configured"
            return
        
        try:
            # Select optimal model
            optimal_model = await self.select_optimal_model(
                request.task_type,
                request.complexity
            )
            
            # Prepare context
            system_prompt, user_prompt = await self.prepare_request_context(request)
            
            # Prepare streaming API request
            api_payload = {
                "model": optimal_model.value,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "stream": True
            }
            
            # Execute streaming request
            async with httpx.AsyncClient(timeout=300.0) as client:
                headers = {
                    **self.default_headers,
                    "x-api-key": self.api_key
                }
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=api_payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield delta.get("text", "")
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def _update_context_window(self,
                                   context_window: ContextWindow,
                                   request: AIRequest,
                                   response: AIResponse) -> None:
        """Update context window with new conversation."""
        
        # Add to conversation history
        context_window.conversation_history.append({
            "user": request.prompt,
            "assistant": response.content,
            "timestamp": datetime.utcnow().isoformat(),
            "tokens": response.tokens_used
        })
        
        # Estimate token usage (rough approximation)
        estimated_tokens = len(request.prompt) // 4 + len(response.content) // 4
        context_window.current_tokens += estimated_tokens
        
        # Check if compression needed
        if (context_window.current_tokens / context_window.max_tokens) > context_window.compression_threshold:
            await self._compress_context_window(context_window)
    
    async def _compress_context_window(self, context_window: ContextWindow) -> None:
        """Compress context window when approaching token limits."""
        
        # Simple compression: keep most recent and highest priority content
        if len(context_window.conversation_history) > 10:
            # Keep last 5 exchanges
            context_window.conversation_history = context_window.conversation_history[-5:]
        
        # Reset token count estimate
        total_content = ""
        for exchange in context_window.conversation_history:
            total_content += exchange.get("user", "") + exchange.get("assistant", "")
        
        context_window.current_tokens = len(total_content) // 4
        context_window.last_compression = datetime.utcnow()
        
        self.request_metrics["context_compression_events"] += 1
        
        logger.info(
            "Context window compressed",
            window_id=context_window.window_id,
            remaining_exchanges=len(context_window.conversation_history),
            estimated_tokens=context_window.current_tokens
        )
    
    async def create_context_window(self,
                                  pilot_id: str,
                                  agent_id: str,
                                  project_context: Optional[Dict[str, Any]] = None) -> ContextWindow:
        """Create new context window for pilot/agent."""
        
        context_window = ContextWindow(
            project_context=project_context or {}
        )
        
        context_key = f"{pilot_id}_{agent_id}"
        self.active_contexts[context_key] = context_window
        
        logger.info(
            "Context window created",
            pilot_id=pilot_id,
            agent_id=agent_id,
            window_id=context_window.window_id
        )
        
        return context_window
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get AI model integration service status."""
        
        return {
            "service_status": "operational" if self.api_key else "limited_functionality",
            "api_configured": bool(self.api_key),
            "available_models": [model.value for model in ModelType],
            "active_contexts": len(self.active_contexts),
            "performance_metrics": self.request_metrics.copy(),
            "service_capabilities": [
                "claude_api_integration",
                "intelligent_model_selection", 
                "context_window_management",
                "streaming_responses",
                "autonomous_development_support"
            ]
        }


# Global AI model integration service instance
_ai_model_service: Optional[AIModelIntegrationService] = None


async def get_ai_model_service() -> AIModelIntegrationService:
    """Get or create AI model integration service instance."""
    global _ai_model_service
    if _ai_model_service is None:
        _ai_model_service = AIModelIntegrationService()
    return _ai_model_service


# Convenience functions for common use cases
async def execute_development_task(prompt: str,
                                 pilot_id: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> AIResponse:
    """Execute autonomous development task."""
    
    service = await get_ai_model_service()
    
    request = AIRequest(
        prompt=prompt,
        task_type="development",
        complexity=TaskComplexity.MEDIUM,
        pilot_id=pilot_id,
        context=context or {},
        max_tokens=8192,
        temperature=0.1
    )
    
    return await service.execute_ai_request(request)


async def execute_architecture_analysis(prompt: str,
                                      pilot_id: Optional[str] = None,
                                      context: Optional[Dict[str, Any]] = None) -> AIResponse:
    """Execute architecture analysis task."""
    
    service = await get_ai_model_service()
    
    request = AIRequest(
        prompt=prompt,
        task_type="architecture",
        complexity=TaskComplexity.COMPLEX,
        pilot_id=pilot_id,
        context=context or {},
        max_tokens=8192,
        temperature=0.2,
        model_type=ModelType.CLAUDE_OPUS
    )
    
    return await service.execute_ai_request(request)


async def execute_code_review(code: str,
                            requirements: str,
                            pilot_id: Optional[str] = None) -> AIResponse:
    """Execute code review task."""
    
    service = await get_ai_model_service()
    
    prompt = f"""
Please review the following code against the requirements:

Requirements:
{requirements}

Code:
```
{code}
```

Provide feedback on:
1. Code quality and best practices
2. Security considerations
3. Performance optimization opportunities
4. Testing coverage recommendations
5. Documentation completeness
"""
    
    request = AIRequest(
        prompt=prompt,
        task_type="code_review",
        complexity=TaskComplexity.MEDIUM,
        pilot_id=pilot_id,
        max_tokens=4096,
        temperature=0.1
    )
    
    return await service.execute_ai_request(request)