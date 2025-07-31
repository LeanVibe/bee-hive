"""
Autonomous Development Engine for LeanVibe Agent Hive 2.0

This is the core engine that demonstrates autonomous development capabilities.
It takes a development task, uses AI to understand requirements, generates code,
creates tests, writes documentation, and validates the complete solution.

This is a minimal viable implementation that proves the autonomous development concept.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from anthropic import AsyncAnthropic

from .config import settings

# Import sandbox components
try:
    from .sandbox import is_sandbox_mode, get_sandbox_config, create_mock_anthropic_client
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False

logger = structlog.get_logger()


class DevelopmentPhase(str, Enum):
    """Phases of autonomous development."""
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    VALIDATION = "validation"
    COMPLETION = "completion"


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Single function, basic logic
    MODERATE = "moderate"  # Multiple functions, some complexity
    COMPLEX = "complex"    # Multiple files, advanced logic


@dataclass
class DevelopmentTask:
    """Represents a development task to be completed autonomously."""
    id: str
    description: str
    requirements: List[str]
    complexity: TaskComplexity
    language: str = "python"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class DevelopmentArtifact:
    """Represents an artifact generated during development."""
    name: str
    type: str  # "code", "test", "doc", "config"
    content: str
    file_path: str
    description: str


@dataclass
class DevelopmentResult:
    """Result of autonomous development process."""
    task_id: str
    success: bool
    artifacts: List[DevelopmentArtifact]
    execution_time_seconds: float
    phases_completed: List[str]
    validation_results: Dict[str, bool]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "execution_time_seconds": self.execution_time_seconds,
            "phases_completed": self.phases_completed,
            "validation_results": self.validation_results,
            "error_message": self.error_message
        }


class AutonomousDevelopmentEngine:
    """
    Core engine for autonomous development.
    
    Takes a development task and autonomously:
    1. Understands the requirements
    2. Plans the implementation
    3. Generates working code
    4. Creates comprehensive tests
    5. Writes documentation
    6. Validates the complete solution
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        # Check if sandbox mode is enabled
        self.sandbox_mode = SANDBOX_AVAILABLE and (
            settings.SANDBOX_MODE or 
            is_sandbox_mode() or 
            not (anthropic_api_key or settings.ANTHROPIC_API_KEY)
        )
        
        if self.sandbox_mode:
            # Use mock client for sandbox mode
            self.anthropic_client = create_mock_anthropic_client(anthropic_api_key)
            logger.info("Sandbox mode enabled - using mock Anthropic client")
        else:
            # Use real client for production mode
            self.anthropic_client = AsyncAnthropic(
                api_key=anthropic_api_key or settings.ANTHROPIC_API_KEY
            )
            logger.info("Production mode enabled - using real Anthropic client")
        
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="autonomous_dev_"))
        logger.info("Autonomous Development Engine initialized", 
                   workspace_dir=str(self.workspace_dir),
                   sandbox_mode=self.sandbox_mode)
    
    async def develop_autonomously(self, task: DevelopmentTask) -> DevelopmentResult:
        """
        Main entry point for autonomous development.
        
        Args:
            task: The development task to complete
            
        Returns:
            DevelopmentResult with all generated artifacts and validation results
        """
        start_time = datetime.utcnow()
        phases_completed = []
        artifacts = []
        validation_results = {}
        
        try:
            logger.info("Starting autonomous development", task_id=task.id, 
                       description=task.description)
            
            # Phase 1: Understanding
            logger.info("Phase 1: Understanding requirements")
            understanding = await self._understand_requirements(task)
            phases_completed.append(DevelopmentPhase.UNDERSTANDING.value)
            
            # Phase 2: Planning
            logger.info("Phase 2: Planning implementation")
            plan = await self._create_implementation_plan(task, understanding)
            phases_completed.append(DevelopmentPhase.PLANNING.value)
            
            # Phase 3: Implementation
            logger.info("Phase 3: Generating code")
            code_artifacts = await self._generate_code(task, plan)
            artifacts.extend(code_artifacts)
            phases_completed.append(DevelopmentPhase.IMPLEMENTATION.value)
            
            # Phase 4: Testing
            logger.info("Phase 4: Creating tests")
            test_artifacts = await self._generate_tests(task, code_artifacts)
            artifacts.extend(test_artifacts)
            phases_completed.append(DevelopmentPhase.TESTING.value)
            
            # Phase 5: Documentation
            logger.info("Phase 5: Writing documentation")
            doc_artifacts = await self._generate_documentation(task, code_artifacts)
            artifacts.extend(doc_artifacts)
            phases_completed.append(DevelopmentPhase.DOCUMENTATION.value)
            
            # Phase 6: Validation
            logger.info("Phase 6: Validating solution")
            validation_results = await self._validate_solution(artifacts)
            phases_completed.append(DevelopmentPhase.VALIDATION.value)
            
            phases_completed.append(DevelopmentPhase.COMPLETION.value)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            success = all(validation_results.values())
            
            result = DevelopmentResult(
                task_id=task.id,
                success=success,
                artifacts=artifacts,
                execution_time_seconds=execution_time,
                phases_completed=phases_completed,
                validation_results=validation_results
            )
            
            logger.info("Autonomous development completed", 
                       task_id=task.id, success=success, 
                       execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error("Autonomous development failed", 
                        task_id=task.id, error=str(e))
            
            return DevelopmentResult(
                task_id=task.id,
                success=False,
                artifacts=artifacts,
                execution_time_seconds=execution_time,
                phases_completed=phases_completed,
                validation_results=validation_results,
                error_message=str(e)
            )
    
    async def _understand_requirements(self, task: DevelopmentTask) -> Dict[str, Any]:
        """Phase 1: Understand and analyze the requirements."""
        prompt = f"""
Analyze this development task and provide a structured understanding:

Task Description: {task.description}
Requirements: {', '.join(task.requirements)}
Language: {task.language}
Complexity: {task.complexity.value}

Please provide a JSON response with:
{{
    "core_functionality": "What the code should do",
    "inputs": ["List of expected inputs"],
    "outputs": ["List of expected outputs"],
    "edge_cases": ["Important edge cases to handle"],
    "dependencies": ["Required libraries or modules"],
    "complexity_assessment": "Detailed complexity analysis"
}}
"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            understanding = json.loads(message.content[0].text)
            return understanding
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "core_functionality": task.description,
                "inputs": ["user input"],
                "outputs": ["result"],
                "edge_cases": ["invalid input"],
                "dependencies": [],
                "complexity_assessment": f"Standard {task.complexity.value} task"
            }
    
    async def _create_implementation_plan(self, task: DevelopmentTask, 
                                        understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Create detailed implementation plan."""
        prompt = f"""
Create an implementation plan for this task:

Task: {task.description}
Understanding: {json.dumps(understanding, indent=2)}

Provide a JSON response with:
{{
    "functions_to_create": [
        {{
            "name": "function_name",
            "purpose": "what it does",
            "parameters": ["param1", "param2"],
            "return_type": "return type"
        }}
    ],
    "file_structure": {{
        "main_file": "filename.py",
        "test_file": "test_filename.py",
        "additional_files": []
    }},
    "implementation_steps": ["step 1", "step 2", "step 3"]
}}
"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            plan = json.loads(message.content[0].text)
            return plan
        except json.JSONDecodeError:
            # Fallback plan
            return {
                "functions_to_create": [
                    {
                        "name": "main_function",
                        "purpose": task.description,
                        "parameters": ["input"],
                        "return_type": "output"
                    }
                ],
                "file_structure": {
                    "main_file": "solution.py",
                    "test_file": "test_solution.py",
                    "additional_files": []
                },
                "implementation_steps": ["Implement main logic", "Handle edge cases", "Add validation"]
            }
    
    async def _generate_code(self, task: DevelopmentTask, 
                           plan: Dict[str, Any]) -> List[DevelopmentArtifact]:
        """Phase 3: Generate the actual code implementation."""
        prompt = f"""
Implement the solution for this task:

Task: {task.description}
Plan: {json.dumps(plan, indent=2)}

Requirements:
- Write clean, well-documented Python code
- Include proper error handling
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Handle edge cases appropriately

Generate complete, runnable code that solves the problem.
"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        code_content = message.content[0].text
        
        # Extract Python code from the response
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', code_content, re.DOTALL)
        
        if code_blocks:
            code_content = code_blocks[0]
        else:
            # If no code blocks found, use the entire response
            code_content = code_content
        
        # Create the main implementation file
        main_file = plan.get("file_structure", {}).get("main_file", "solution.py")
        file_path = self.workspace_dir / main_file
        
        with open(file_path, 'w') as f:
            f.write(code_content)
        
        artifact = DevelopmentArtifact(
            name=main_file,
            type="code",
            content=code_content,
            file_path=str(file_path),
            description="Main implementation file"
        )
        
        return [artifact]
    
    async def _generate_tests(self, task: DevelopmentTask, 
                            code_artifacts: List[DevelopmentArtifact]) -> List[DevelopmentArtifact]:
        """Phase 4: Generate comprehensive tests."""
        main_code = code_artifacts[0].content
        
        prompt = f"""
Create comprehensive unit tests for this code:

Code to test:
```python
{main_code}
```

Task description: {task.description}

Requirements:
- Use Python unittest framework
- Test normal cases, edge cases, and error conditions
- Include meaningful test names and assertions
- Aim for high code coverage
- Test all public functions

Generate complete test code that thoroughly validates the implementation.
"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        test_content = message.content[0].text
        
        # Extract Python test code
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', test_content, re.DOTALL)
        
        if code_blocks:
            test_content = code_blocks[0]
        
        # Create test file
        test_file = f"test_{code_artifacts[0].name}"
        file_path = self.workspace_dir / test_file
        
        with open(file_path, 'w') as f:
            f.write(test_content)
        
        artifact = DevelopmentArtifact(
            name=test_file,
            type="test",
            content=test_content,
            file_path=str(file_path),
            description="Comprehensive unit tests"
        )
        
        return [artifact]
    
    async def _generate_documentation(self, task: DevelopmentTask, 
                                    code_artifacts: List[DevelopmentArtifact]) -> List[DevelopmentArtifact]:
        """Phase 5: Generate comprehensive documentation."""
        main_code = code_artifacts[0].content
        
        prompt = f"""
Create comprehensive documentation for this code:

Code:
```python
{main_code}
```

Task: {task.description}

Generate a README.md with:
- Clear description of what the code does
- Usage examples with code snippets
- API documentation for functions
- Installation/setup instructions if needed
- Examples of inputs and outputs

Make it professional and user-friendly.
"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        doc_content = message.content[0].text
        
        # Create documentation file
        doc_file = "README.md"
        file_path = self.workspace_dir / doc_file
        
        with open(file_path, 'w') as f:
            f.write(doc_content)
        
        artifact = DevelopmentArtifact(
            name=doc_file,
            type="doc",
            content=doc_content,
            file_path=str(file_path),
            description="Comprehensive documentation"
        )
        
        return [artifact]
    
    async def _validate_solution(self, artifacts: List[DevelopmentArtifact]) -> Dict[str, bool]:
        """Phase 6: Validate the complete solution."""
        validation_results = {}
        
        # Find code and test artifacts
        code_artifact = next((a for a in artifacts if a.type == "code"), None)
        test_artifact = next((a for a in artifacts if a.type == "test"), None)
        doc_artifact = next((a for a in artifacts if a.type == "doc"), None)
        
        # Validate code syntax
        if code_artifact:
            try:
                import ast
                ast.parse(code_artifact.content)
                validation_results["code_syntax_valid"] = True
                logger.info("Code syntax validation passed")
            except SyntaxError as e:
                validation_results["code_syntax_valid"] = False
                logger.error("Code syntax validation failed", error=str(e))
        
        # Validate test syntax
        if test_artifact:
            try:
                import ast
                ast.parse(test_artifact.content)
                validation_results["test_syntax_valid"] = True
                logger.info("Test syntax validation passed")
            except SyntaxError as e:
                validation_results["test_syntax_valid"] = False
                logger.error("Test syntax validation failed", error=str(e))
        
        # Try to run tests
        if code_artifact and test_artifact:
            try:
                # Write files to workspace
                code_file = Path(code_artifact.file_path)
                test_file = Path(test_artifact.file_path)
                
                # Run tests using subprocess
                import subprocess
                result = subprocess.run(
                    ["python", "-m", "unittest", test_file.stem],
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                validation_results["tests_pass"] = result.returncode == 0
                if result.returncode == 0:
                    logger.info("All tests passed")
                else:
                    logger.error("Tests failed", stdout=result.stdout, stderr=result.stderr)
                    
            except Exception as e:
                validation_results["tests_pass"] = False
                logger.error("Test execution failed", error=str(e))
        
        # Validate documentation exists and has content
        if doc_artifact:
            validation_results["documentation_exists"] = bool(doc_artifact.content.strip())
            validation_results["documentation_comprehensive"] = len(doc_artifact.content) > 200
        
        # Overall solution completeness
        validation_results["solution_complete"] = (
            bool(code_artifact) and 
            bool(test_artifact) and 
            bool(doc_artifact)
        )
        
        return validation_results
    
    def get_workspace_path(self) -> Path:
        """Get the workspace directory path."""
        return self.workspace_dir
    
    def cleanup_workspace(self):
        """Clean up the temporary workspace."""
        import shutil
        try:
            shutil.rmtree(self.workspace_dir)
            logger.info("Workspace cleaned up", workspace_dir=str(self.workspace_dir))
        except Exception as e:
            logger.error("Failed to clean up workspace", error=str(e))


# Factory function for easy instantiation
def create_autonomous_development_engine(anthropic_api_key: Optional[str] = None) -> AutonomousDevelopmentEngine:
    """Create and return an autonomous development engine instance."""
    return AutonomousDevelopmentEngine(anthropic_api_key)