"""
Code execution API endpoints for LeanVibe Agent Hive 2.0

Provides HTTP endpoints for generating, analyzing, and executing code
with comprehensive security validation and quality assessment.
"""

import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from ...core.code_execution import (
    code_executor, CodeGenerationEngine, CodeLanguage, CodeBlock,
    SecurityLevel, CodeQuality
)
from ...core.database import get_session_dependency
from ...models.agent import Agent
from anthropic import AsyncAnthropic
from ...core.config import settings

logger = structlog.get_logger()
router = APIRouter()

# Initialize code generation engine
code_generator = CodeGenerationEngine(AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY))


class CodeGenerationRequest(BaseModel):
    """Request to generate code."""
    agent_id: str = Field(..., description="Agent ID generating the code")
    requirements: str = Field(..., description="Code requirements and specifications")
    language: CodeLanguage = Field(..., description="Programming language")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    existing_code: Optional[str] = Field(None, description="Existing code to build upon")


class CodeExecutionRequest(BaseModel):
    """Request to execute code."""
    agent_id: str = Field(..., description="Agent ID executing the code")
    code: str = Field(..., description="Code to execute")
    language: CodeLanguage = Field(..., description="Programming language")
    description: str = Field(default="Code execution", description="Code description")
    validate_security: bool = Field(default=True, description="Perform security validation")
    validate_quality: bool = Field(default=True, description="Perform quality validation")
    file_path: Optional[str] = Field(None, description="File path for the code")


class CodeAnalysisRequest(BaseModel):
    """Request to analyze code."""
    code: str = Field(..., description="Code to analyze")
    language: CodeLanguage = Field(..., description="Programming language")
    analysis_type: str = Field(default="both", description="Analysis type: security, quality, or both")


class CodeGenerationResponse(BaseModel):
    """Response from code generation."""
    code_id: str
    agent_id: str
    language: str
    code: str
    description: str
    generation_time_ms: int
    created_at: str


class CodeExecutionResponse(BaseModel):
    """Response from code execution."""
    code_id: str
    agent_id: str
    execution_success: bool
    output: str
    error: str
    execution_time_ms: int
    
    # Security analysis
    security_level: Optional[str]
    security_threats: List[str]
    security_warnings: List[str]
    
    # Quality analysis
    quality_level: Optional[str]
    quality_score: Optional[float]
    syntax_errors: List[str]
    style_violations: List[str]
    
    # Performance metrics
    memory_used_mb: float
    files_created: List[str]
    files_modified: List[str]


class CodeAnalysisResponse(BaseModel):
    """Response from code analysis."""
    # Security analysis
    security_level: str
    security_safe: bool
    security_threats: List[str]
    security_warnings: List[str]
    security_confidence: float
    
    # Quality analysis
    quality_level: str
    quality_score: float
    complexity_score: float
    maintainability_score: float
    readability_score: float
    
    # Issues and suggestions
    syntax_errors: List[str]
    style_violations: List[str]
    potential_bugs: List[str]
    improvement_suggestions: List[str]


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> CodeGenerationResponse:
    """Generate code based on requirements."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        import time
        start_time = time.time()
        
        # Generate code
        code_block = await code_generator.generate_code(
            agent_id=request.agent_id,
            requirements=request.requirements,
            language=request.language,
            context=request.context,
            existing_code=request.existing_code
        )
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "Code generated via API",
            agent_id=request.agent_id,
            language=request.language.value,
            code_length=len(code_block.content),
            generation_time_ms=generation_time_ms
        )
        
        return CodeGenerationResponse(
            code_id=code_block.id,
            agent_id=request.agent_id,
            language=request.language.value,
            code=code_block.content,
            description=code_block.description,
            generation_time_ms=generation_time_ms,
            created_at=code_block.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Code generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Code generation failed")


@router.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(
    request: CodeExecutionRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> CodeExecutionResponse:
    """Execute code with security and quality validation."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create code block
        code_block = CodeBlock(
            id=str(uuid.uuid4()),
            language=request.language,
            content=request.code,
            description=request.description,
            file_path=request.file_path,
            agent_id=request.agent_id
        )
        
        # Execute code with validation
        security_result, quality_result, execution_result = await code_executor.execute_code(
            code_block=code_block,
            agent_id=request.agent_id,
            validate_security=request.validate_security,
            validate_quality=request.validate_quality
        )
        
        logger.info(
            "Code executed via API",
            agent_id=request.agent_id,
            language=request.language.value,
            success=execution_result.success,
            execution_time_ms=execution_result.execution_time_ms
        )
        
        return CodeExecutionResponse(
            code_id=code_block.id,
            agent_id=request.agent_id,
            execution_success=execution_result.success,
            output=execution_result.output,
            error=execution_result.error,
            execution_time_ms=execution_result.execution_time_ms,
            security_level=security_result.security_level.value if security_result else None,
            security_threats=security_result.threats_detected if security_result else [],
            security_warnings=security_result.warnings if security_result else [],
            quality_level=quality_result.quality_level.value if quality_result else None,
            quality_score=quality_result.quality_score if quality_result else None,
            syntax_errors=quality_result.syntax_errors if quality_result else [],
            style_violations=quality_result.style_violations if quality_result else [],
            memory_used_mb=execution_result.memory_used_mb,
            files_created=execution_result.files_created,
            files_modified=execution_result.files_modified
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Code execution failed", error=str(e))
        raise HTTPException(status_code=500, detail="Code execution failed")


@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest
) -> CodeAnalysisResponse:
    """Analyze code for security and quality."""
    
    try:
        # Create code block for analysis
        code_block = CodeBlock(
            id=str(uuid.uuid4()),
            language=request.language,
            content=request.code,
            description="Code analysis"
        )
        
        security_result = None
        quality_result = None
        
        # Perform requested analysis
        if request.analysis_type in ["security", "both"]:
            security_result = await code_executor.security_analyzer.analyze_security(code_block)
        
        if request.analysis_type in ["quality", "both"]:
            quality_result = await code_executor.quality_analyzer.analyze_quality(code_block)
        
        logger.info(
            "Code analyzed via API",
            language=request.language.value,
            analysis_type=request.analysis_type,
            code_length=len(request.code)
        )
        
        return CodeAnalysisResponse(
            security_level=security_result.security_level.value if security_result else "unknown",
            security_safe=security_result.safe_to_execute if security_result else True,
            security_threats=security_result.threats_detected if security_result else [],
            security_warnings=security_result.warnings if security_result else [],
            security_confidence=security_result.confidence_score if security_result else 0.0,
            quality_level=quality_result.quality_level.value if quality_result else "unknown",
            quality_score=quality_result.quality_score if quality_result else 0.0,
            complexity_score=quality_result.complexity_score if quality_result else 0.0,
            maintainability_score=quality_result.maintainability_score if quality_result else 0.0,
            readability_score=quality_result.readability_score if quality_result else 0.0,
            syntax_errors=quality_result.syntax_errors if quality_result else [],
            style_violations=quality_result.style_violations if quality_result else [],
            potential_bugs=quality_result.potential_bugs if quality_result else [],
            improvement_suggestions=quality_result.improvement_suggestions if quality_result else []
        )
        
    except Exception as e:
        logger.error("Code analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Code analysis failed")


@router.get("/languages")
async def get_supported_languages() -> Dict[str, List[str]]:
    """Get list of supported programming languages."""
    
    languages = [lang.value for lang in CodeLanguage]
    
    return {
        "supported_languages": languages,
        "language_details": {
            "python": {
                "extensions": [".py"],
                "features": ["execution", "analysis", "generation"],
                "frameworks": ["fastapi", "django", "flask"]
            },
            "typescript": {
                "extensions": [".ts"],
                "features": ["execution", "analysis", "generation"],
                "frameworks": ["react", "angular", "vue"]
            },
            "javascript": {
                "extensions": [".js"],
                "features": ["execution", "analysis", "generation"],
                "frameworks": ["node", "express", "react"]
            },
            "rust": {
                "extensions": [".rs"],
                "features": ["analysis", "generation"],
                "frameworks": ["tokio", "actix-web"]
            },
            "go": {
                "extensions": [".go"],
                "features": ["analysis", "generation"],
                "frameworks": ["gin", "echo"]
            },
            "bash": {
                "extensions": [".sh"],
                "features": ["execution", "analysis"],
                "frameworks": []
            }
        }
    }


@router.get("/security-levels")
async def get_security_levels() -> Dict[str, List[Dict[str, str]]]:
    """Get information about security levels."""
    
    return {
        "security_levels": [
            {
                "level": SecurityLevel.SAFE.value,
                "description": "Read-only operations, basic calculations",
                "allowed_operations": ["data processing", "calculations", "string manipulation"],
                "restrictions": ["no file operations", "no network access", "no system calls"]
            },
            {
                "level": SecurityLevel.MODERATE.value,
                "description": "File operations within workspace",
                "allowed_operations": ["workspace file operations", "package installations"],
                "restrictions": ["no system modifications", "workspace-only file access"]
            },
            {
                "level": SecurityLevel.RESTRICTED.value,
                "description": "Network access and system operations",
                "allowed_operations": ["network requests", "subprocess calls"],
                "restrictions": ["requires monitoring", "limited system access"]
            },
            {
                "level": SecurityLevel.DANGEROUS.value,
                "description": "System modifications, requires approval",
                "allowed_operations": ["system modifications", "unrestricted access"],
                "restrictions": ["requires human approval", "comprehensive auditing"]
            }
        ]
    }


@router.get("/quality-metrics")
async def get_quality_metrics() -> Dict[str, Any]:
    """Get information about code quality metrics."""
    
    return {
        "quality_levels": [
            {
                "level": CodeQuality.EXCELLENT.value,
                "score_range": "0.9 - 1.0",
                "description": "Production-ready code with excellent practices"
            },
            {
                "level": CodeQuality.GOOD.value,
                "score_range": "0.7 - 0.9",
                "description": "Well-written code with minor improvements needed"
            },
            {
                "level": CodeQuality.FAIR.value,
                "score_range": "0.5 - 0.7",
                "description": "Functional code requiring significant improvements"
            },
            {
                "level": CodeQuality.POOR.value,
                "score_range": "0.0 - 0.5",
                "description": "Code with major issues requiring refactoring"
            }
        ],
        "metrics": {
            "complexity_score": "Measures code complexity and nesting levels",
            "maintainability_score": "Assesses how easy the code is to maintain",
            "readability_score": "Evaluates code clarity and documentation",
            "test_coverage_potential": "Estimates how testable the code is"
        },
        "common_issues": [
            "syntax_errors", "style_violations", "potential_bugs",
            "missing_documentation", "high_complexity", "poor_naming"
        ]
    }


@router.post("/{agent_id}/generate-and-execute")
async def generate_and_execute(
    agent_id: str,
    requirements: str,
    language: CodeLanguage,
    context: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_session_dependency)
) -> CodeExecutionResponse:
    """Generate and immediately execute code (convenience endpoint)."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Generate code
        code_block = await code_generator.generate_code(
            agent_id=agent_id,
            requirements=requirements,
            language=language,
            context=context or {}
        )
        
        # Execute code
        security_result, quality_result, execution_result = await code_executor.execute_code(
            code_block=code_block,
            agent_id=agent_id,
            validate_security=True,
            validate_quality=True
        )
        
        logger.info(
            "Code generated and executed via API",
            agent_id=agent_id,
            language=language.value,
            success=execution_result.success
        )
        
        return CodeExecutionResponse(
            code_id=code_block.id,
            agent_id=agent_id,
            execution_success=execution_result.success,
            output=execution_result.output,
            error=execution_result.error,
            execution_time_ms=execution_result.execution_time_ms,
            security_level=security_result.security_level.value if security_result else None,
            security_threats=security_result.threats_detected if security_result else [],
            security_warnings=security_result.warnings if security_result else [],
            quality_level=quality_result.quality_level.value if quality_result else None,
            quality_score=quality_result.quality_score if quality_result else None,
            syntax_errors=quality_result.syntax_errors if quality_result else [],
            style_violations=quality_result.style_violations if quality_result else [],
            memory_used_mb=execution_result.memory_used_mb,
            files_created=execution_result.files_created,
            files_modified=execution_result.files_modified
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Generate and execute failed", error=str(e))
        raise HTTPException(status_code=500, detail="Generate and execute failed")