#!/usr/bin/env python3
"""
Living Documentation System for LeanVibe Agent Hive

This system automatically validates code examples, tests API endpoints,
and ensures documentation accuracy across the consolidated docs.
"""

import os
import re
import json
import asyncio
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ValidationResult:
    """Result of documentation validation"""
    file_path: str
    validation_type: str
    status: str  # 'success', 'warning', 'error'
    message: str
    details: Optional[Dict[str, Any]] = None


class DocumentationValidator:
    """Living documentation validation and maintenance system"""
    
    def __init__(self, base_path: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.base_path = Path(base_path)
        self.api_base_url = "http://localhost:18080"
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks on documentation"""
        self.results.clear()
        
        # Find all markdown files (excluding node_modules and archives)
        md_files = self._find_documentation_files()
        
        print(f"🔍 Found {len(md_files)} documentation files to validate")
        
        for md_file in md_files:
            print(f"📋 Validating: {md_file.relative_to(self.base_path)}")
            
            # Validate code examples
            self._validate_code_examples(md_file)
            
            # Validate internal links
            self._validate_internal_links(md_file)
            
            # Validate API examples (if file contains API docs)
            if self._contains_api_examples(md_file):
                self._validate_api_examples(md_file)
                
            # Check content freshness
            self._check_content_freshness(md_file)
            
        return self.results
    
    def _find_documentation_files(self) -> List[Path]:
        """Find all documentation markdown files"""
        exclude_patterns = [
            "*/node_modules/*", 
            "*/venv/*", 
            "*/archive/*",
            "*/.git/*"
        ]
        
        md_files = []
        for pattern in ["*.md", "**/*.md"]:
            for file in self.base_path.glob(pattern):
                # Skip excluded directories
                if any(file.match(exclude) for exclude in exclude_patterns):
                    continue
                md_files.append(file)
        
        return sorted(set(md_files))
    
    def _validate_code_examples(self, file_path: Path):
        """Extract and validate code examples in markdown"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find code blocks with language specified
            code_blocks = re.findall(r'```(\w+)\n(.*?)\n```', content, re.DOTALL)
            
            for lang, code in code_blocks:
                if lang.lower() == 'bash':
                    self._validate_bash_example(file_path, code)
                elif lang.lower() == 'python':
                    self._validate_python_example(file_path, code)
                elif lang.lower() in ['json', 'javascript', 'http']:
                    self._validate_syntax(file_path, lang, code)
                    
        except Exception as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="code_examples",
                status="error",
                message=f"Failed to validate code examples: {e}"
            ))
    
    def _validate_bash_example(self, file_path: Path, code: str):
        """Validate bash code examples (basic syntax check)"""
        # Skip examples with placeholders or obvious non-executable content
        if any(placeholder in code for placeholder in ['<', '>', '{', '}', 'your-', 'example']):
            return
            
        # Check for dangerous commands
        dangerous_commands = ['rm -rf', 'sudo rm', 'format', 'delete']
        if any(cmd in code.lower() for cmd in dangerous_commands):
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="bash_safety",
                status="warning",
                message="Contains potentially dangerous bash commands"
            ))
            return
            
        # Basic syntax validation (check for obvious syntax errors)
        lines = code.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            # Check for unclosed quotes (basic check)
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    validation_type="bash_syntax",
                    status="warning",
                    message=f"Possible unclosed quotes in bash example line {i+1}: {line[:50]}..."
                ))
    
    def _validate_python_example(self, file_path: Path, code: str):
        """Validate Python code examples"""
        # Skip examples with obvious placeholders
        if any(placeholder in code for placeholder in ['<', '>', 'your-', 'example', '...']):
            return
            
        try:
            # Try to compile the Python code (syntax check)
            compile(code, f"{file_path}:python_example", 'exec')
            
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="python_syntax",
                status="success",
                message="Python syntax validation passed"
            ))
            
        except SyntaxError as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="python_syntax", 
                status="error",
                message=f"Python syntax error: {e.msg} at line {e.lineno}"
            ))
    
    def _validate_syntax(self, file_path: Path, lang: str, code: str):
        """Validate syntax for JSON, JavaScript, HTTP examples"""
        if lang.lower() == 'json':
            try:
                json.loads(code)
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    validation_type="json_syntax",
                    status="success",
                    message="JSON syntax validation passed"
                ))
            except json.JSONDecodeError as e:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    validation_type="json_syntax",
                    status="error",
                    message=f"JSON syntax error: {e.msg} at line {e.lineno}"
                ))
    
    def _validate_internal_links(self, file_path: Path):
        """Validate internal links within the documentation"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find markdown links [text](path)
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            
            for link_text, link_path in links:
                # Skip external links
                if link_path.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue
                    
                # Resolve relative paths
                if not link_path.startswith('/'):
                    target_path = (file_path.parent / link_path).resolve()
                else:
                    target_path = (self.base_path / link_path.lstrip('/')).resolve()
                
                if not target_path.exists():
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        validation_type="internal_links",
                        status="error",
                        message=f"Broken internal link: [{link_text}]({link_path})",
                        details={"target_path": str(target_path)}
                    ))
                else:
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        validation_type="internal_links",
                        status="success", 
                        message=f"Valid internal link: [{link_text}]({link_path})"
                    ))
                    
        except Exception as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="internal_links",
                status="error",
                message=f"Failed to validate internal links: {e}"
            ))
    
    def _contains_api_examples(self, file_path: Path) -> bool:
        """Check if file contains API examples"""
        try:
            content = file_path.read_text(encoding='utf-8').lower()
            api_indicators = [
                'curl', 'http://', 'https://', 
                'get /', 'post /', 'put /', 'delete /',
                'api/', '/api', 'localhost:18080'
            ]
            return any(indicator in content for indicator in api_indicators)
        except:
            return False
    
    def _validate_api_examples(self, file_path: Path):
        """Validate API endpoint examples"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find curl commands
            curl_commands = re.findall(r'curl[^`\n]*', content)
            
            for curl_cmd in curl_commands:
                # Skip examples with obvious placeholders
                if any(placeholder in curl_cmd for placeholder in ['<', '>', 'your-', 'example']):
                    continue
                    
                # Extract URL from curl command
                url_match = re.search(r'https?://[^\s]+', curl_cmd)
                if not url_match:
                    continue
                    
                url = url_match.group()
                
                # Only test localhost URLs to avoid external dependencies
                if 'localhost:18080' in url:
                    self._test_api_endpoint(file_path, url)
                    
        except Exception as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="api_examples",
                status="error",
                message=f"Failed to validate API examples: {e}"
            ))
    
    def _test_api_endpoint(self, file_path: Path, url: str):
        """Test if API endpoint is accessible"""
        try:
            # Quick connectivity test (with short timeout)
            response = requests.get(url, timeout=2)
            
            if response.status_code < 400:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    validation_type="api_endpoint",
                    status="success",
                    message=f"API endpoint accessible: {url} (status: {response.status_code})"
                ))
            else:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    validation_type="api_endpoint",
                    status="warning",
                    message=f"API endpoint returned error: {url} (status: {response.status_code})"
                ))
                
        except requests.RequestException as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="api_endpoint",
                status="warning", 
                message=f"API endpoint not accessible: {url} ({type(e).__name__})"
            ))
    
    def _check_content_freshness(self, file_path: Path):
        """Check if documentation content is fresh/recently updated"""
        try:
            # Get file modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age = datetime.now() - mtime
            
            # Define freshness thresholds
            if age > timedelta(days=90):
                status = "warning"
                message = f"Content may be stale (last modified: {mtime.strftime('%Y-%m-%d')})"
            elif age > timedelta(days=30):
                status = "info"
                message = f"Content moderately old (last modified: {mtime.strftime('%Y-%m-%d')})"
            else:
                status = "success"
                message = f"Content is fresh (last modified: {mtime.strftime('%Y-%m-%d')})"
            
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="content_freshness",
                status=status,
                message=message,
                details={"last_modified": mtime.isoformat(), "age_days": age.days}
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                file_path=str(file_path),
                validation_type="content_freshness",
                status="error",
                message=f"Failed to check content freshness: {e}"
            ))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.results:
            return {"error": "No validation results available"}
        
        # Group results by status and type
        by_status = {}
        by_type = {}
        by_file = {}
        
        for result in self.results:
            # By status
            if result.status not in by_status:
                by_status[result.status] = []
            by_status[result.status].append(result)
            
            # By type
            if result.validation_type not in by_type:
                by_type[result.validation_type] = []
            by_type[result.validation_type].append(result)
            
            # By file
            if result.file_path not in by_file:
                by_file[result.file_path] = []
            by_file[result.file_path].append(result)
        
        # Calculate summary statistics
        total_files = len(by_file)
        total_validations = len(self.results)
        success_count = len(by_status.get('success', []))
        warning_count = len(by_status.get('warning', []))
        error_count = len(by_status.get('error', []))
        
        return {
            "summary": {
                "total_files_validated": total_files,
                "total_validations": total_validations,
                "success_count": success_count,
                "warning_count": warning_count,
                "error_count": error_count,
                "success_rate": round((success_count / total_validations) * 100, 1) if total_validations > 0 else 0
            },
            "by_status": {k: len(v) for k, v in by_status.items()},
            "by_type": {k: len(v) for k, v in by_type.items()},
            "errors": [r for r in self.results if r.status == 'error'],
            "warnings": [r for r in self.results if r.status == 'warning'],
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def save_report(self, output_path: str = "docs/validation_report.json"):
        """Save validation report to JSON file"""
        report = self.generate_report()
        
        output_file = self.base_path / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open('w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Validation report saved to: {output_file}")
        return output_file


def main():
    """Run documentation validation"""
    print("🚀 Starting Living Documentation System Validation")
    print("=" * 60)
    
    validator = DocumentationValidator()
    
    # Run all validations
    results = validator.validate_all()
    
    # Generate and display report
    report = validator.generate_report()
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files validated: {report['summary']['total_files_validated']}")
    print(f"Total validations: {report['summary']['total_validations']}")
    print(f"✅ Success: {report['summary']['success_count']}")
    print(f"⚠️  Warnings: {report['summary']['warning_count']}")  
    print(f"❌ Errors: {report['summary']['error_count']}")
    print(f"Success rate: {report['summary']['success_rate']}%")
    
    # Show errors and warnings
    if report['errors']:
        print(f"\n❌ ERRORS ({len(report['errors'])})")
        print("-" * 40)
        for error in report['errors'][:5]:  # Show first 5 errors
            file_name = Path(error.file_path).name
            print(f"  {file_name}: {error.message}")
        if len(report['errors']) > 5:
            print(f"  ... and {len(report['errors']) - 5} more errors")
    
    if report['warnings']:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])})")
        print("-" * 40)
        for warning in report['warnings'][:5]:  # Show first 5 warnings
            file_name = Path(warning.file_path).name
            print(f"  {file_name}: {warning.message}")
        if len(report['warnings']) > 5:
            print(f"  ... and {len(report['warnings']) - 5} more warnings")
    
    # Save detailed report
    report_path = validator.save_report()
    
    print(f"\n📋 Detailed report saved to: {report_path}")
    print("\n🎯 Living Documentation System validation complete!")
    
    # Return exit code based on results
    return 1 if report['summary']['error_count'] > 0 else 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)