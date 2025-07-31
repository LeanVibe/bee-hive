"""
FastAPI Demo Endpoints for LeanVibe Agent Hive 2.0
Handles autonomous development demo requests and provides real-time progress updates
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

# Import the autonomous development engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from app.core.autonomous_development_engine import (
        AutonomousDevelopmentEngine, 
        DevelopmentTask, 
        TaskComplexity,
        DevelopmentResult,
        create_autonomous_development_engine
    )
    
    # Import sandbox components
    from app.core.sandbox import (
        is_sandbox_mode,
        get_sandbox_config,
        get_sandbox_status,
        create_sandbox_orchestrator
    )
    from app.core.sandbox.demo_scenarios import get_demo_scenario_engine
    
    SANDBOX_AVAILABLE = True
except ImportError:
    # Fallback for demo-only deployment
    from demo.fallback.autonomous_engine import (
        AutonomousDevelopmentEngine,
        DevelopmentTask,
        TaskComplexity, 
        DevelopmentResult
    )
    SANDBOX_AVAILABLE = False

logger = structlog.get_logger()

# Create demo router
demo_router = APIRouter(prefix="/api/demo", tags=["demo"])

# Initialize sandbox orchestrator if available
sandbox_orchestrator = None
if SANDBOX_AVAILABLE:
    try:
        sandbox_orchestrator = create_sandbox_orchestrator()
    except Exception as e:
        logger.warning("Failed to initialize sandbox orchestrator", error=str(e))

# Global demo session storage (in production, use Redis or database)
demo_sessions: Dict[str, Dict[str, Any]] = {}

# SSE connection storage
sse_connections: Dict[str, List] = {}


class DemoTaskRequest(BaseModel):
    """Request model for demo task submission."""
    session_id: str = Field(..., description="Unique session identifier")
    task: Dict[str, Any] = Field(..., description="Task definition")


class DemoProgressEvent(BaseModel):
    """Progress event model for SSE streaming."""
    type: str = Field(..., description="Event type")
    phase: Optional[str] = Field(None, description="Current development phase")
    overall_progress: Optional[int] = Field(None, description="Overall progress percentage")
    code: Optional[str] = Field(None, description="Generated code")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if any")


@demo_router.post("/autonomous-development")
async def start_autonomous_development(
    request: DemoTaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Start autonomous development for a demo task.
    
    This endpoint triggers the autonomous development process and returns immediately.
    Progress updates are sent via Server-Sent Events to the progress endpoint.
    """
    try:
        session_id = request.session_id
        task_data = request.task
        
        logger.info("Starting autonomous development demo", 
                   session_id=session_id, task=task_data)
        
        # Validate task data
        if not task_data.get("description"):
            raise HTTPException(status_code=400, detail="Task description is required")
        
        # Create development task
        complexity = TaskComplexity(task_data.get("complexity", "simple"))
        requirements = task_data.get("requirements", [])
        
        development_task = DevelopmentTask(
            id=str(uuid.uuid4()),
            description=task_data["description"],
            requirements=requirements,
            complexity=complexity,
            language="python",
            created_at=datetime.utcnow()
        )
        
        # Initialize session
        demo_sessions[session_id] = {
            "task": development_task,
            "status": "started", 
            "start_time": datetime.utcnow(),
            "progress": 0,
            "current_phase": None,
            "result": None
        }
        
        # Start autonomous development in background
        background_tasks.add_task(
            run_autonomous_development,
            session_id,
            development_task
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Autonomous development started",
            "estimated_completion": datetime.utcnow() + timedelta(
                seconds=task_data.get("estimatedTime", 60)
            )
        }
        
    except Exception as e:
        logger.error("Failed to start autonomous development", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@demo_router.get("/progress/{session_id}")
async def stream_progress(session_id: str):
    """
    Stream real-time progress updates for a demo session via Server-Sent Events.
    """
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_stream():
        """Generate Server-Sent Events for progress updates."""
        try:
            # Initialize connection
            if session_id not in sse_connections:
                sse_connections[session_id] = []
            
            connection_id = str(uuid.uuid4())
            
            logger.info("SSE connection established", 
                       session_id=session_id, connection_id=connection_id)
            
            # Send initial status
            session = demo_sessions[session_id]
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"
            
            # Stream progress updates
            last_progress = -1
            last_phase = None
            
            while True:
                session = demo_sessions.get(session_id)
                if not session:
                    break
                
                current_progress = session.get("progress", 0)
                current_phase = session.get("current_phase")
                
                # Send progress updates
                if current_progress != last_progress:
                    event = {
                        "type": "progress_update",
                        "overall_progress": current_progress
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                    last_progress = current_progress
                
                # Send phase updates
                if current_phase != last_phase and current_phase:
                    event = {
                        "type": "phase_start",
                        "phase": current_phase,
                        "overall_progress": current_progress
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                    last_phase = current_phase
                
                # Check if development is complete
                if session.get("status") == "completed":
                    result = session.get("result")
                    if result:
                        event = {
                            "type": "development_complete",
                            "success": result.success,
                            "execution_time": result.execution_time_seconds * 1000,  # Convert to ms
                            "files": [
                                {
                                    "name": artifact.name,
                                    "type": artifact.type,
                                    "content": artifact.content,
                                    "description": artifact.description
                                }
                                for artifact in result.artifacts
                            ],
                            "validation_results": result.validation_results
                        }
                        yield f"data: {json.dumps(event)}\n\n"
                    break
                
                # Check for errors
                if session.get("status") == "error":
                    error_event = {
                        "type": "error",
                        "error": session.get("error", "Unknown error occurred")
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    break
                
                # Wait before next update
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error("SSE stream error", session_id=session_id, error=str(e))
            error_event = {
                "type": "error",
                "error": f"Stream error: {str(e)}"
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        
        finally:
            # Clean up connection
            if session_id in sse_connections:
                sse_connections[session_id] = [
                    conn for conn in sse_connections[session_id] 
                    if conn != connection_id
                ]
            
            logger.info("SSE connection closed", 
                       session_id=session_id, connection_id=connection_id)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@demo_router.get("/download/{session_id}")
async def download_solution(session_id: str):
    """
    Download the complete solution as a ZIP file.
    """
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = demo_sessions[session_id]
    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Development not completed")
    
    result = session.get("result")
    if not result or not result.artifacts:
        raise HTTPException(status_code=404, detail="No artifacts available")
    
    try:
        # Create ZIP file content
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for artifact in result.artifacts:
                zip_file.writestr(artifact.name, artifact.content)
            
            # Add metadata
            metadata = {
                "task_description": session["task"].description,
                "completion_time": result.execution_time_seconds,
                "validation_results": result.validation_results,
                "generated_at": datetime.utcnow().isoformat(),
                "leanvibe_version": "2.0.0"
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=leanvibe-solution-{session_id}.zip"
            }
        )
        
    except Exception as e:
        logger.error("Failed to create download", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create download")


@demo_router.get("/status/{session_id}")
async def get_demo_status(session_id: str):
    """
    Get current status of a demo session.
    """
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = demo_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "progress": session.get("progress", 0),
        "current_phase": session.get("current_phase"),
        "start_time": session.get("start_time"),
        "task_description": session["task"].description,
        "estimated_completion": session.get("estimated_completion")
    }


async def run_autonomous_development(session_id: str, task: DevelopmentTask):
    """
    Background task to run autonomous development and update session progress.
    """
    try:
        logger.info("Starting autonomous development", session_id=session_id)
        
        # Initialize development engine
        engine = AutonomousDevelopmentEngine()
        
        # Update session status
        session = demo_sessions[session_id]
        session["status"] = "running"
        
        # Phases for progress tracking
        phases = [
            "understanding",
            "planning", 
            "implementation",
            "testing",
            "documentation",
            "validation"
        ]
        
        # Simulate phase progression (in real implementation, this would come from the engine)
        total_phases = len(phases)
        
        for i, phase in enumerate(phases):
            # Update current phase
            session["current_phase"] = phase
            session["progress"] = int((i / total_phases) * 100)
            
            logger.info("Development phase started", 
                       session_id=session_id, phase=phase, progress=session["progress"])
            
            # Simulate phase duration
            phase_duration = 2.0 + (i * 1.5)  # Variable duration per phase
            await asyncio.sleep(phase_duration)
            
            # Mark phase complete
            session["progress"] = int(((i + 1) / total_phases) * 100)
        
        # Run actual autonomous development
        result = await engine.develop_autonomously(task)
        
        # Update session with results
        session["status"] = "completed" if result.success else "error"
        session["result"] = result
        session["progress"] = 100
        session["completion_time"] = datetime.utcnow()
        
        if not result.success:
            session["error"] = result.error_message or "Development failed"
        
        logger.info("Autonomous development completed", 
                   session_id=session_id, success=result.success,
                   execution_time=result.execution_time_seconds)
        
    except Exception as e:
        logger.error("Autonomous development failed", 
                    session_id=session_id, error=str(e))
        
        session = demo_sessions.get(session_id, {})
        session["status"] = "error"
        session["error"] = str(e)
        session["completion_time"] = datetime.utcnow()


@demo_router.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up a demo session and its resources.
    """
    if session_id in demo_sessions:
        # Clean up session data
        session = demo_sessions[session_id]
        if "result" in session and session["result"]:
            # Clean up workspace if exists
            try:
                result = session["result"]
                if hasattr(result, 'workspace_dir'):
                    import shutil
                    shutil.rmtree(result.workspace_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to cleanup workspace", error=str(e))
        
        del demo_sessions[session_id]
    
    # Clean up SSE connections
    if session_id in sse_connections:
        del sse_connections[session_id]
    
    return {"success": True, "message": "Session cleaned up"}


# Sandbox-specific endpoints
@demo_router.get("/sandbox/status")
async def sandbox_status():
    """Get sandbox mode status and configuration."""
    if not SANDBOX_AVAILABLE:
        return {"sandbox_available": False, "message": "Sandbox mode not available"}
    
    try:
        status = get_sandbox_status()
        return {
            "sandbox_available": True,
            "status": status,
            "orchestrator_ready": sandbox_orchestrator is not None,
            "scenarios_available": True
        }
    except Exception as e:
        return {
            "sandbox_available": True,
            "error": str(e),
            "message": "Sandbox mode available but configuration failed"
        }


@demo_router.get("/sandbox/scenarios")
async def get_sandbox_scenarios():
    """Get available sandbox demo scenarios."""
    if not SANDBOX_AVAILABLE:
        return {"error": "Sandbox mode not available"}
    
    try:
        scenario_engine = get_demo_scenario_engine()
        scenarios = scenario_engine.get_all_scenarios()
        
        return {
            "sandbox_mode": True,
            "scenarios_count": len(scenarios),
            "scenarios": scenarios,
            "recommended": {
                "beginner": scenario_engine.get_recommended_scenario("beginner").to_dict(),
                "intermediate": scenario_engine.get_recommended_scenario("intermediate").to_dict(),
                "advanced": scenario_engine.get_recommended_scenario("advanced").to_dict()
            }
        }
    except Exception as e:
        logger.error("Failed to get sandbox scenarios", error=str(e))
        return {"error": f"Failed to load scenarios: {str(e)}"}


@demo_router.post("/sandbox/start")
async def start_sandbox_demo(
    request: DemoTaskRequest,
    background_tasks: BackgroundTasks
):
    """Start sandbox autonomous development demo."""
    if not SANDBOX_AVAILABLE or not sandbox_orchestrator:
        return {"error": "Sandbox mode not available"}
    
    try:
        session_id = request.session_id
        task_data = request.task
        
        # Start autonomous development in sandbox mode
        result = await sandbox_orchestrator.start_autonomous_development(
            session_id=session_id,
            task_description=task_data.get("description", ""),
            requirements=task_data.get("requirements", []),
            complexity=task_data.get("complexity", "simple")
        )
        
        return {
            "sandbox_mode": True,
            "demo_started": True,
            **result
        }
        
    except Exception as e:
        logger.error("Failed to start sandbox demo", error=str(e))
        return {"error": f"Failed to start demo: {str(e)}"}


@demo_router.get("/sandbox/session/{session_id}")
async def get_sandbox_session_status(session_id: str):
    """Get sandbox session status and progress."""
    if not SANDBOX_AVAILABLE or not sandbox_orchestrator:
        return {"error": "Sandbox mode not available"}
    
    try:
        status = sandbox_orchestrator.get_session_status(session_id)
        if status is None:
            return {"error": "Session not found"}
        
        return {
            "sandbox_mode": True,
            "session_found": True,
            **status
        }
        
    except Exception as e:
        logger.error("Failed to get sandbox session status", error=str(e))
        return {"error": f"Failed to get session status: {str(e)}"}


@demo_router.get("/health")
async def demo_health_check():
    """
    Health check endpoint for the demo API.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "active_sessions": len(demo_sessions),
        "active_connections": sum(len(conns) for conns in sse_connections.values()),
        "sandbox_mode": SANDBOX_AVAILABLE and is_sandbox_mode() if SANDBOX_AVAILABLE else False,
        "sandbox_orchestrator_ready": sandbox_orchestrator is not None
    }