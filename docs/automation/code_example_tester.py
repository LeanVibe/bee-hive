#!/usr/bin/env python3
"""
Code Example Tester for Living Documentation System

Automatically tests all code examples in documentation files to ensure they work correctly.
Supports Python, Bash, JavaScript, and JSON validation with execution testing.
"""

import os
import json
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class CodeTestResult:
    """Result of code example testing"""
    file_path: str
    language: str
    line_number: int
    status: str  # 'success', 'warning', 'error', 'skipped'
    message: str
    execution_time: Optional[float] = None
    output: Optional[str] = None
    error_details: Optional[str] = None


class CodeExampleTester:
    """Advanced code example testing with execution validation"""
    
    def __init__(self, base_path: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.base_path = Path(base_path)
        self.results: List[CodeTestResult] = []
        self.test_timeout = 30  # seconds
        
    async def test_all_code_examples(self, file_paths: List[Path]) -> List[CodeTestResult]:
        """Test all code examples in provided files"""
        self.results.clear()
        
        for file_path in file_paths:
            print(f"🧪 Testing code examples in: {file_path.relative_to(self.base_path)}")
            await self._test_file_code_examples(file_path)
            
        return self.results
    
    async def _test_file_code_examples(self, file_path: Path):
        """Extract and test all code examples in a file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find all code blocks with language and line numbers
            code_blocks = self._extract_code_blocks_with_positions(content)
            
            for language, code, line_number in code_blocks:
                await self._test_code_block(file_path, language, code, line_number)
                
        except Exception as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="unknown",
                line_number=0,
                status="error",
                message=f"Failed to parse file: {e}"
            ))
    
    def _extract_code_blocks_with_positions(self, content: str) -> List[Tuple[str, str, int]]:
        """Extract code blocks with their line positions"""
        code_blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for code block start
            match = re.match(r'^```(\w+)', line)
            if match:
                language = match.group(1).lower()
                code_lines = []
                start_line = i + 1
                i += 1
                
                # Collect code block content
                while i < len(lines) and not lines[i].startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                
                if i < len(lines) and lines[i].startswith('```'):
                    code = '\n'.join(code_lines)
                    code_blocks.append((language, code, start_line))
                    
            i += 1
            
        return code_blocks
    
    async def _test_code_block(self, file_path: Path, language: str, code: str, line_number: int):
        """Test individual code block based on language"""
        
        # Skip empty or trivial code blocks
        if not code.strip() or len(code.strip().split('\n')) == 1 and len(code.strip()) < 10:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language=language,
                line_number=line_number,
                status="skipped",
                message="Empty or trivial code block"
            ))
            return
        
        # Skip examples with placeholders
        placeholders = ['<your-', '<example', '{your-', '{example', 'TODO:', 'FIXME:', '...']
        if any(placeholder in code for placeholder in placeholders):
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language=language,
                line_number=line_number,
                status="skipped",
                message="Contains placeholders or incomplete code"
            ))
            return
        
        # Route to appropriate tester
        if language == 'python':
            await self._test_python_code(file_path, code, line_number)
        elif language == 'bash' or language == 'sh':
            await self._test_bash_code(file_path, code, line_number)
        elif language == 'javascript' or language == 'js':
            await self._test_javascript_code(file_path, code, line_number)
        elif language == 'json':
            self._test_json_code(file_path, code, line_number)
        elif language in ['yaml', 'yml']:
            self._test_yaml_code(file_path, code, line_number)
        elif language == 'http':
            self._test_http_code(file_path, code, line_number)
        else:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language=language,
                line_number=line_number,
                status="skipped",
                message=f"Testing not implemented for language: {language}"
            ))
    
    async def _test_python_code(self, file_path: Path, code: str, line_number: int):
        """Test Python code execution"""
        # First, syntax check
        try:
            compile(code, f"{file_path}:line{line_number}", 'exec')
        except SyntaxError as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="python",
                line_number=line_number,
                status="error",
                message=f"Python syntax error: {e.msg}",
                error_details=str(e)
            ))
            return
        
        # Skip execution for code that imports external libraries not available
        dangerous_imports = ['requests', 'flask', 'fastapi', 'django', 'numpy', 'pandas']
        if any(f'import {lib}' in code or f'from {lib}' in code for lib in dangerous_imports):
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="python",
                line_number=line_number,
                status="warning",
                message="Contains external dependencies - syntax validated only"
            ))
            return
        
        # Skip code that looks like it needs external services
        if any(indicator in code.lower() for indicator in ['localhost:', 'http://', 'database', 'redis', 'postgres']):
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="python",
                line_number=line_number,
                status="warning",
                message="Requires external services - syntax validated only"
            ))
            return
        
        # Execute safe Python code
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                'python', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.test_timeout
                )
                
                if process.returncode == 0:
                    self.results.append(CodeTestResult(
                        file_path=str(file_path),
                        language="python",
                        line_number=line_number,
                        status="success",
                        message="Python code executed successfully",
                        output=stdout.decode('utf-8') if stdout else None
                    ))
                else:
                    self.results.append(CodeTestResult(
                        file_path=str(file_path),
                        language="python",
                        line_number=line_number,
                        status="error",
                        message="Python code execution failed",
                        error_details=stderr.decode('utf-8') if stderr else None
                    ))
                    
            except asyncio.TimeoutError:
                process.kill()
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="python",
                    line_number=line_number,
                    status="error",
                    message="Python code execution timed out"
                ))
                
        except Exception as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="python",
                line_number=line_number,
                status="error",
                message=f"Python execution test failed: {e}"
            ))
        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def _test_bash_code(self, file_path: Path, code: str, line_number: int):
        """Test Bash code (syntax check only for safety)"""
        # Check for dangerous commands
        dangerous_commands = [
            'rm -rf', 'sudo rm', 'format', 'delete', 'dd if=', 
            '>/dev/', 'chmod 777', 'curl | sh', 'wget | sh'
        ]
        
        if any(cmd in code.lower() for cmd in dangerous_commands):
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="bash",
                line_number=line_number,
                status="error",
                message="Contains potentially dangerous bash commands"
            ))
            return
        
        # Only syntax check for bash (safer)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write('#!/bin/bash\n' + code)
                temp_path = f.name
            
            # Bash syntax check with -n flag
            process = await asyncio.create_subprocess_exec(
                'bash', '-n', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="bash",
                    line_number=line_number,
                    status="success",
                    message="Bash syntax validation passed"
                ))
            else:
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="bash",
                    line_number=line_number,
                    status="error",
                    message="Bash syntax error detected",
                    error_details=stderr.decode('utf-8') if stderr else None
                ))
                
        except Exception as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="bash",
                line_number=line_number,
                status="error",
                message=f"Bash syntax check failed: {e}"
            ))
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def _test_javascript_code(self, file_path: Path, code: str, line_number: int):
        """Test JavaScript code with Node.js if available"""
        try:
            # Check if Node.js is available
            node_check = await asyncio.create_subprocess_exec(
                'node', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await node_check.communicate()
            
            if node_check.returncode != 0:
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="javascript",
                    line_number=line_number,
                    status="skipped",
                    message="Node.js not available for JavaScript testing"
                ))
                return
            
            # Skip code that requires browser APIs or external modules
            browser_apis = ['document', 'window', 'fetch', 'localStorage', 'require(']
            if any(api in code for api in browser_apis):
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="javascript",
                    line_number=line_number,
                    status="warning",
                    message="Contains browser APIs or external modules - skipped execution"
                ))
                return
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            process = await asyncio.create_subprocess_exec(
                'node', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.test_timeout
                )
                
                if process.returncode == 0:
                    self.results.append(CodeTestResult(
                        file_path=str(file_path),
                        language="javascript",
                        line_number=line_number,
                        status="success",
                        message="JavaScript code executed successfully",
                        output=stdout.decode('utf-8') if stdout else None
                    ))
                else:
                    self.results.append(CodeTestResult(
                        file_path=str(file_path),
                        language="javascript",
                        line_number=line_number,
                        status="error",
                        message="JavaScript execution failed",
                        error_details=stderr.decode('utf-8') if stderr else None
                    ))
                    
            except asyncio.TimeoutError:
                process.kill()
                self.results.append(CodeTestResult(
                    file_path=str(file_path),
                    language="javascript",
                    line_number=line_number,
                    status="error",
                    message="JavaScript execution timed out"
                ))
                
        except Exception as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="javascript",
                line_number=line_number,
                status="error",
                message=f"JavaScript test failed: {e}"
            ))
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _test_json_code(self, file_path: Path, code: str, line_number: int):
        """Test JSON code validity"""
        try:
            json.loads(code)
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="json",
                line_number=line_number,
                status="success",
                message="JSON syntax validation passed"
            ))
        except json.JSONDecodeError as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="json",
                line_number=line_number,
                status="error",
                message=f"JSON syntax error: {e.msg}",
                error_details=f"Line {e.lineno}, Column {e.colno}"
            ))
    
    def _test_yaml_code(self, file_path: Path, code: str, line_number: int):
        """Test YAML code validity"""
        try:
            import yaml
            yaml.safe_load(code)
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="yaml",
                line_number=line_number,
                status="success",
                message="YAML syntax validation passed"
            ))
        except ImportError:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="yaml",
                line_number=line_number,
                status="skipped",
                message="PyYAML not available for YAML validation"
            ))
        except Exception as e:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="yaml",
                line_number=line_number,
                status="error",
                message=f"YAML syntax error: {e}"
            ))
    
    def _test_http_code(self, file_path: Path, code: str, line_number: int):
        """Test HTTP request examples (basic format validation)"""
        lines = code.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Check for valid HTTP method and path
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if not any(first_line.startswith(method) for method in http_methods):
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="http",
                line_number=line_number,
                status="warning",
                message="HTTP example doesn't start with valid HTTP method"
            ))
        else:
            self.results.append(CodeTestResult(
                file_path=str(file_path),
                language="http",
                line_number=line_number,
                status="success",
                message="HTTP example format looks valid"
            ))
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of code testing results"""
        if not self.results:
            return {}
        
        by_status = {}
        by_language = {}
        
        for result in self.results:
            # Count by status
            if result.status not in by_status:
                by_status[result.status] = 0
            by_status[result.status] += 1
            
            # Count by language
            if result.language not in by_language:
                by_language[result.language] = {'total': 0, 'success': 0, 'error': 0}
            by_language[result.language]['total'] += 1
            if result.status == 'success':
                by_language[result.language]['success'] += 1
            elif result.status == 'error':
                by_language[result.language]['error'] += 1
        
        total_tests = len(self.results)
        success_rate = (by_status.get('success', 0) / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_code_blocks_tested': total_tests,
            'by_status': by_status,
            'by_language': by_language,
            'overall_success_rate': round(success_rate, 1)
        }


async def main():
    """Run code example testing"""
    print("🧪 Code Example Tester for Living Documentation")
    print("=" * 60)
    
    tester = CodeExampleTester()
    
    # Find markdown files
    md_files = []
    for pattern in ["*.md", "**/*.md"]:
        for file in tester.base_path.glob(pattern):
            if not any(exclude in str(file) for exclude in ['node_modules', 'venv', 'archive', '.git']):
                md_files.append(file)
    
    print(f"Found {len(md_files)} documentation files")
    
    # Test all code examples
    results = await tester.test_all_code_examples(md_files)
    
    # Display results
    stats = tester.get_summary_statistics()
    
    print(f"\n📊 CODE TESTING SUMMARY")
    print("=" * 60)
    print(f"Code blocks tested: {stats.get('total_code_blocks_tested', 0)}")
    print(f"Success rate: {stats.get('overall_success_rate', 0)}%")
    
    if 'by_status' in stats:
        for status, count in stats['by_status'].items():
            emoji = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'skipped': '⏭️'}.get(status, '📋')
            print(f"{emoji} {status}: {count}")
    
    if 'by_language' in stats:
        print(f"\n📈 BY LANGUAGE:")
        for lang, counts in stats['by_language'].items():
            success_rate = (counts['success'] / counts['total'] * 100) if counts['total'] > 0 else 0
            print(f"  {lang}: {counts['total']} total, {success_rate:.1f}% success")
    
    # Show some errors
    errors = [r for r in results if r.status == 'error']
    if errors:
        print(f"\n❌ SAMPLE ERRORS:")
        for error in errors[:3]:
            file_name = Path(error.file_path).name
            print(f"  {file_name}:{error.line_number} ({error.language}): {error.message}")
    
    return len(errors)


if __name__ == "__main__":
    import sys
    error_count = asyncio.run(main())
    sys.exit(1 if error_count > 0 else 0)