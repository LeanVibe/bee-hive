#!/usr/bin/env python3
"""
Standalone Autonomous Development Demo

This is a simplified, standalone version of the autonomous development demo
that doesn't require the full LeanVibe infrastructure. It demonstrates the
core autonomous development capabilities in a minimal setup.

Usage:
    export ANTHROPIC_API_KEY="your_api_key"
    python scripts/demos/standalone_autonomous_demo.py
"""

import asyncio
import json
import os
import tempfile
import ast
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Simple Anthropic client import
try:
    from anthropic import AsyncAnthropic
except ImportError:
    print("❌ Please install the anthropic package: pip install anthropic")
    exit(1)


@dataclass
class SimpleTask:
    """Simple development task representation."""
    description: str
    requirements: List[str]


@dataclass
class GeneratedFile:
    """Represents a generated file."""
    name: str
    content: str
    file_type: str


class StandaloneAutonomousDeveloper:
    """Simplified autonomous developer for demonstration."""
    
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.workspace = Path(tempfile.mkdtemp(prefix="autonomous_demo_"))
        print(f"🏗️  Workspace created: {self.workspace}")
    
    async def develop_solution(self, task: SimpleTask) -> Dict[str, Any]:
        """Autonomously develop a complete solution."""
        print(f"\n🎯 Task: {task.description}")
        print("🤖 AI Agent starting autonomous development...")
        
        start_time = datetime.now()
        files = []
        
        try:
            # Phase 1: Generate Code
            print("\n📝 Phase 1: Generating code...")
            code_content = await self._generate_code(task)
            code_file = GeneratedFile("solution.py", code_content, "code")
            files.append(code_file)
            
            # Phase 2: Generate Tests
            print("🧪 Phase 2: Creating tests...")
            test_content = await self._generate_tests(task, code_content)
            test_file = GeneratedFile("test_solution.py", test_content, "test")
            files.append(test_file)
            
            # Phase 3: Generate Documentation
            print("📖 Phase 3: Writing documentation...")
            doc_content = await self._generate_docs(task, code_content)
            doc_file = GeneratedFile("README.md", doc_content, "doc")
            files.append(doc_file)
            
            # Phase 4: Write files and validate
            print("✅ Phase 4: Validating solution...")
            validation_results = await self._validate_solution(files)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": all(validation_results.values()),
                "files": files,
                "validation": validation_results,
                "execution_time": execution_time,
                "workspace": str(self.workspace)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files": files,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _generate_code(self, task: SimpleTask) -> str:
        """Generate the main code implementation."""
        prompt = f"""
Create a Python solution for this task:

Task: {task.description}
Requirements: {', '.join(task.requirements)}

Generate clean, well-documented Python code that:
- Solves the problem completely
- Includes proper error handling  
- Has clear docstrings
- Follows Python best practices
- Is ready to run

Return only the Python code, no explanations.
"""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        # Extract code from markdown blocks if present
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        return content.strip()
    
    async def _generate_tests(self, task: SimpleTask, code: str) -> str:
        """Generate comprehensive tests."""
        prompt = f"""
Create comprehensive unit tests for this Python code:

Code to test:
```python
{code}
```

Task: {task.description}

Generate Python unittest code that:
- Tests all functions thoroughly
- Covers normal cases and edge cases
- Uses descriptive test names
- Includes setUp/tearDown if needed
- Aims for high coverage

Return only the test code, no explanations.
"""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        # Extract code from markdown blocks if present
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        return content.strip()
    
    async def _generate_docs(self, task: SimpleTask, code: str) -> str:
        """Generate comprehensive documentation."""
        prompt = f"""
Create comprehensive documentation for this Python code:

Code:
```python
{code}
```

Task: {task.description}

Generate a README.md that includes:
- Clear description of what the code does
- Usage examples with sample code
- Function/class documentation
- Installation/setup instructions
- Input/output examples

Return only the markdown content, no code blocks around it.
"""
        
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    async def _validate_solution(self, files: List[GeneratedFile]) -> Dict[str, bool]:
        """Validate the generated solution."""
        validation = {}
        
        # Write files to workspace
        for file in files:
            file_path = self.workspace / file.name
            with open(file_path, 'w') as f:
                f.write(file.content)
        
        # Validate code syntax
        code_file = next((f for f in files if f.file_type == "code"), None)
        if code_file:
            try:
                ast.parse(code_file.content)
                validation["code_syntax_valid"] = True
            except SyntaxError:
                validation["code_syntax_valid"] = False
        
        # Validate test syntax
        test_file = next((f for f in files if f.file_type == "test"), None)
        if test_file:
            try:
                ast.parse(test_file.content)
                validation["test_syntax_valid"] = True
            except SyntaxError:
                validation["test_syntax_valid"] = False
        
        # Try running tests
        if code_file and test_file:
            try:
                result = subprocess.run(
                    ["python", "-m", "unittest", "test_solution"],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                validation["tests_pass"] = result.returncode == 0
            except Exception:
                validation["tests_pass"] = False
        
        # Check documentation exists
        doc_file = next((f for f in files if f.file_type == "doc"), None)
        validation["documentation_exists"] = bool(doc_file and doc_file.content.strip())
        
        return validation


def display_results(result: Dict[str, Any]):
    """Display the development results."""
    print("\n" + "="*80)
    print("🎉 AUTONOMOUS DEVELOPMENT RESULTS")
    print("="*80)
    
    # Status
    status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f} seconds")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Validation results
    print(f"\n🔍 Validation Results:")
    for check, passed in result["validation"].items():
        icon = "✅" if passed else "❌"
        print(f"   {icon} {check.replace('_', ' ').title()}")
    
    # Generated files
    print(f"\n📁 Generated Files ({len(result['files'])}):")
    for file in result["files"]:
        icons = {"code": "💻", "test": "🧪", "doc": "📖"}
        icon = icons.get(file.file_type, "📄")
        print(f"   {icon} {file.name} ({len(file.content)} chars)")
    
    # Show code preview
    code_file = next((f for f in result["files"] if f.file_type == "code"), None)
    if code_file:
        print(f"\n💻 Generated Code Preview ({code_file.name}):")
        print("-" * 60)
        lines = code_file.content.split('\n')[:25]
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line}")
        if len(code_file.content.split('\n')) > 25:
            print("    ... (truncated)")
    
    print(f"\n📁 Full results available in: {result['workspace']}")


async def main():
    """Run the standalone autonomous development demo."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🚀 LeanVibe Agent Hive 2.0 - Standalone Autonomous Development Demo     ║
║                                                                              ║
║  Demonstrates AI agents autonomously completing development tasks from       ║
║  requirements to working code with tests and documentation.                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Please set your API key: export ANTHROPIC_API_KEY='your_key'")
        return
    
    # Create developer
    developer = StandaloneAutonomousDeveloper(api_key)
    
    # Define demo task
    task = SimpleTask(
        description="Create a function to calculate Fibonacci numbers up to n terms",
        requirements=[
            "Handle positive integers as input",
            "Include input validation for non-positive numbers",
            "Use an efficient iterative approach", 
            "Return a list of Fibonacci numbers",
            "Include proper error handling and documentation"
        ]
    )
    
    try:
        # Run autonomous development
        result = await developer.develop_solution(task)
        
        # Display results
        display_results(result)
        
        if result["success"]:
            print("\n🎉 Autonomous development completed successfully!")
            print("   The AI agent has proven it can autonomously:")
            print("   • Understand requirements")
            print("   • Generate working code")
            print("   • Create comprehensive tests")
            print("   • Write documentation")
            print("   • Validate the complete solution")
        else:
            print("\n⚠️  Development completed with issues - see validation results above")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())