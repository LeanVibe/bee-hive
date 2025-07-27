"""
Code Review Assistant for LeanVibe Agent Hive 2.0

Automated code review with security, performance, and style analysis
providing comprehensive feedback and intelligent suggestions.
"""

import asyncio
import logging
import uuid
import re
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.github_integration import (
    PullRequest, CodeReview, ReviewStatus, GitCommit
)
from ..core.github_api_client import GitHubAPIClient


logger = logging.getLogger(__name__)
settings = get_settings()


class ReviewSeverity(Enum):
    """Review finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(Enum):
    """Review finding categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    BEST_PRACTICES = "best_practices"
    DOCUMENTATION = "documentation"


class CodeReviewError(Exception):
    """Custom exception for code review operations."""
    pass


class SecurityAnalyzer:
    """
    Security-focused code analysis with comprehensive vulnerability detection.
    
    Identifies common security issues including injection vulnerabilities,
    authentication flaws, and insecure coding patterns.
    """
    
    def __init__(self):
        self.security_patterns = {
            "sql_injection": [
                r'(?i)\b(execute|query|cursor\.execute)\s*\(\s*["\'].*?\+.*?["\']',
                r'(?i)\b(sql|query)\s*=\s*["\'].*?\+.*?["\']',
                r'(?i)\.format\(\s*.*?(select|insert|update|delete|drop|alter)',
                r'(?i)f["\'][^"\']*?(select|insert|update|delete|drop|alter)[^"\']*?{.*?}[^"\']*?["\']'
            ],
            "command_injection": [
                r'(?i)\b(os\.system|subprocess\.call|subprocess\.run|subprocess\.Popen)\s*\([^)]*?\+',
                r'(?i)\b(exec|eval)\s*\([^)]*?\+',
                r'(?i)shell\s*=\s*True',
                r'(?i)os\.popen\s*\([^)]*?\+'
            ],
            "path_traversal": [
                r'(?i)open\s*\([^)]*?\+.*?["\'][^"\']*\.\.[^"\']*["\']',
                r'(?i)file\s*=.*?\+.*?["\'][^"\']*\.\.[^"\']*["\']',
                r'(?i)\.\.[\\/]',
                r'(?i)os\.path\.join\([^)]*?\.\..*?\)'
            ],
            "hardcoded_secrets": [
                r'(?i)(password|pwd|pass|secret|key|token|api_key)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)(access_key|secret_key|private_key)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}',
                r'(?i)[A-Za-z0-9_\-\.]{32,}'  # Potential tokens
            ],
            "weak_crypto": [
                r'(?i)\b(md5|sha1)\b',
                r'(?i)DES\b',
                r'(?i)random\.random\(\)',
                r'(?i)insecure.*random',
                r'(?i)ssl.*PROTOCOL_SSLv[23]'
            ],
            "unsafe_deserialization": [
                r'(?i)\bpickle\.loads?\b',
                r'(?i)\beval\s*\(',
                r'(?i)\bexec\s*\(',
                r'(?i)yaml\.load\s*\([^)]*\)',
                r'(?i)json\.loads.*user.*input'
            ],
            "xss_injection": [
                r'(?i)innerHTML\s*=.*?\+',
                r'(?i)\.html\(.*?\+',
                r'(?i)document\.write\s*\(.*?\+',
                r'(?i)eval\s*\(.*?request\.',
                r'(?i)dangerouslySetInnerHTML'
            ],
            "authentication_bypass": [
                r'(?i)auth.*=.*None',
                r'(?i)authentication.*disabled?',
                r'(?i)bypass.*auth',
                r'(?i)skip.*authentication',
                r'(?i)if.*user.*==.*admin'
            ]
        }
        
    def analyze_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Analyze file for security vulnerabilities."""
        
        findings = []
        
        for vulnerability_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    finding = {
                        "category": ReviewCategory.SECURITY.value,
                        "severity": self._get_severity_for_vulnerability(vulnerability_type),
                        "type": vulnerability_type,
                        "file": file_path,
                        "line": line_number,
                        "column": match.start() - content.rfind('\n', 0, match.start()) - 1,
                        "message": self._get_message_for_vulnerability(vulnerability_type),
                        "code_snippet": self._extract_code_snippet(content, match.start(), match.end()),
                        "suggestion": self._get_suggestion_for_vulnerability(vulnerability_type),
                        "confidence": self._get_confidence_for_pattern(pattern, match.group())
                    }
                    
                    findings.append(finding)
                    
        return findings
        
    def _get_severity_for_vulnerability(self, vulnerability_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            "sql_injection": ReviewSeverity.CRITICAL.value,
            "command_injection": ReviewSeverity.CRITICAL.value,
            "path_traversal": ReviewSeverity.HIGH.value,
            "hardcoded_secrets": ReviewSeverity.HIGH.value,
            "weak_crypto": ReviewSeverity.MEDIUM.value,
            "unsafe_deserialization": ReviewSeverity.CRITICAL.value,
            "xss_injection": ReviewSeverity.HIGH.value,
            "authentication_bypass": ReviewSeverity.CRITICAL.value
        }
        return severity_map.get(vulnerability_type, ReviewSeverity.MEDIUM.value)
        
    def _get_message_for_vulnerability(self, vulnerability_type: str) -> str:
        """Get descriptive message for vulnerability type."""
        messages = {
            "sql_injection": "Potential SQL injection vulnerability detected",
            "command_injection": "Potential command injection vulnerability detected",
            "path_traversal": "Potential path traversal vulnerability detected",
            "hardcoded_secrets": "Hardcoded secrets or credentials detected",
            "weak_crypto": "Weak cryptographic algorithm usage detected",
            "unsafe_deserialization": "Unsafe deserialization detected",
            "xss_injection": "Potential XSS vulnerability detected",
            "authentication_bypass": "Potential authentication bypass detected"
        }
        return messages.get(vulnerability_type, "Security issue detected")
        
    def _get_suggestion_for_vulnerability(self, vulnerability_type: str) -> str:
        """Get remediation suggestion for vulnerability type."""
        suggestions = {
            "sql_injection": "Use parameterized queries or ORM methods instead of string concatenation",
            "command_injection": "Use subprocess with shell=False and validate/sanitize inputs",
            "path_traversal": "Validate and sanitize file paths, use os.path.normpath() and check bounds",
            "hardcoded_secrets": "Move secrets to environment variables or secure configuration",
            "weak_crypto": "Use strong algorithms like SHA-256, bcrypt, or AES with proper key management",
            "unsafe_deserialization": "Use safe serialization formats like JSON or validate input data",
            "xss_injection": "Escape user input and use safe DOM manipulation methods",
            "authentication_bypass": "Implement proper authentication and authorization checks"
        }
        return suggestions.get(vulnerability_type, "Review and fix security issue")
        
    def _get_confidence_for_pattern(self, pattern: str, match_text: str) -> float:
        """Calculate confidence level for pattern match."""
        # More specific patterns get higher confidence
        if "+" in pattern and ("select" in pattern.lower() or "insert" in pattern.lower()):
            return 0.9  # High confidence for SQL injection patterns
        elif "shell=True" in match_text:
            return 0.95  # Very high confidence for shell=True
        elif re.search(r'[0-9a-f]{32,}', match_text):
            return 0.7   # Medium-high confidence for potential tokens
        else:
            return 0.6   # Default confidence
            
    def _extract_code_snippet(self, content: str, start: int, end: int) -> str:
        """Extract code snippet around the issue."""
        lines = content.split('\n')
        start_line = content[:start].count('\n')
        
        # Get 2 lines before and after for context
        context_start = max(0, start_line - 2)
        context_end = min(len(lines), start_line + 3)
        
        snippet_lines = []
        for i in range(context_start, context_end):
            prefix = ">>> " if i == start_line else "    "
            snippet_lines.append(f"{prefix}{lines[i]}")
            
        return '\n'.join(snippet_lines)


class PerformanceAnalyzer:
    """
    Performance-focused code analysis for optimization opportunities.
    
    Identifies performance bottlenecks, inefficient algorithms,
    and resource usage issues.
    """
    
    def __init__(self):
        self.performance_patterns = {
            "inefficient_loops": [
                r'for.*in.*range\(len\(',
                r'while.*len\(',
                r'for.*in.*\.keys\(\):',
                r'if.*in.*\[.*\]:'  # Linear search in list
            ],
            "memory_leaks": [
                r'global\s+\w+\s*=\s*\[\]',
                r'cache\s*=\s*{}',
                r'\.append\(.*large.*\)',
                r'while\s+True:.*\.append'
            ],
            "unnecessary_calculations": [
                r'for.*in.*range.*:.*math\.',
                r'for.*in.*range.*:.*len\(',
                r'\.join\(.*for.*in.*\)',
                r'sum\(.*for.*in.*\)'
            ],
            "database_antipatterns": [
                r'for.*in.*\.all\(\):',
                r'\.filter\(.*\)\.filter\(',
                r'\.get\(\).*in.*for',
                r'SELECT.*N\+1'
            ],
            "file_io_inefficiency": [
                r'open\(.*\)\.read\(\)\.split',
                r'for.*line.*in.*open\(',
                r'\.write\(.*\+.*\)',
                r'\.seek\(0\).*\.read\(\)'
            ]
        }
        
    def analyze_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Analyze file for performance issues."""
        
        findings = []
        
        # Pattern-based analysis
        for issue_type, patterns in self.performance_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    finding = {
                        "category": ReviewCategory.PERFORMANCE.value,
                        "severity": self._get_severity_for_issue(issue_type),
                        "type": issue_type,
                        "file": file_path,
                        "line": line_number,
                        "column": match.start() - content.rfind('\n', 0, match.start()) - 1,
                        "message": self._get_message_for_issue(issue_type),
                        "code_snippet": self._extract_code_snippet(content, match.start(), match.end()),
                        "suggestion": self._get_suggestion_for_issue(issue_type),
                        "confidence": 0.7
                    }
                    
                    findings.append(finding)
                    
        # AST-based analysis for Python files
        if file_path.endswith('.py'):
            ast_findings = self._analyze_python_ast(file_path, content)
            findings.extend(ast_findings)
            
        return findings
        
    def _analyze_python_ast(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Analyze Python code using AST for deeper performance insights."""
        
        findings = []
        
        try:
            tree = ast.parse(content)
            
            class PerformanceVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.findings = []
                    
                def visit_ListComp(self, node):
                    # Check for nested list comprehensions
                    nested_comps = [child for child in ast.walk(node) 
                                  if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp))]
                    if len(nested_comps) > 1:
                        self.findings.append({
                            "category": ReviewCategory.PERFORMANCE.value,
                            "severity": ReviewSeverity.MEDIUM.value,
                            "type": "nested_comprehensions",
                            "file": file_path,
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": "Nested comprehensions can be inefficient",
                            "suggestion": "Consider breaking into separate comprehensions or using itertools",
                            "confidence": 0.8
                        })
                    self.generic_visit(node)
                    
                def visit_For(self, node):
                    # Check for enumerate usage
                    if (isinstance(node.iter, ast.Call) and 
                        isinstance(node.iter.func, ast.Name) and 
                        node.iter.func.id == "range"):
                        
                        # Check if range(len(x)) pattern
                        if (len(node.iter.args) == 1 and 
                            isinstance(node.iter.args[0], ast.Call) and
                            isinstance(node.iter.args[0].func, ast.Name) and
                            node.iter.args[0].func.id == "len"):
                            
                            self.findings.append({
                                "category": ReviewCategory.PERFORMANCE.value,
                                "severity": ReviewSeverity.LOW.value,
                                "type": "range_len_antipattern",
                                "file": file_path,
                                "line": node.lineno,
                                "column": node.col_offset,
                                "message": "Use enumerate() instead of range(len())",
                                "suggestion": "Replace 'for i in range(len(x)):' with 'for i, item in enumerate(x):'",
                                "confidence": 0.9
                            })
                    self.generic_visit(node)
                    
            visitor = PerformanceVisitor()
            visitor.visit(tree)
            findings.extend(visitor.findings)
            
        except SyntaxError:
            # File has syntax errors, skip AST analysis
            pass
            
        return findings
        
    def _get_severity_for_issue(self, issue_type: str) -> str:
        """Get severity level for performance issue type."""
        severity_map = {
            "inefficient_loops": ReviewSeverity.MEDIUM.value,
            "memory_leaks": ReviewSeverity.HIGH.value,
            "unnecessary_calculations": ReviewSeverity.MEDIUM.value,
            "database_antipatterns": ReviewSeverity.HIGH.value,
            "file_io_inefficiency": ReviewSeverity.MEDIUM.value
        }
        return severity_map.get(issue_type, ReviewSeverity.LOW.value)
        
    def _get_message_for_issue(self, issue_type: str) -> str:
        """Get descriptive message for performance issue type."""
        messages = {
            "inefficient_loops": "Inefficient loop pattern detected",
            "memory_leaks": "Potential memory leak detected",
            "unnecessary_calculations": "Unnecessary calculation in loop detected",
            "database_antipatterns": "Database anti-pattern detected",
            "file_io_inefficiency": "Inefficient file I/O operation detected"
        }
        return messages.get(issue_type, "Performance issue detected")
        
    def _get_suggestion_for_issue(self, issue_type: str) -> str:
        """Get optimization suggestion for performance issue type."""
        suggestions = {
            "inefficient_loops": "Use enumerate(), direct iteration, or vectorized operations",
            "memory_leaks": "Consider using local variables, clearing references, or generators",
            "unnecessary_calculations": "Move calculations outside loops or use caching",
            "database_antipatterns": "Use select_related(), prefetch_related(), or bulk operations",
            "file_io_inefficiency": "Use context managers, buffered I/O, or streaming operations"
        }
        return suggestions.get(issue_type, "Optimize for better performance")
        
    def _extract_code_snippet(self, content: str, start: int, end: int) -> str:
        """Extract code snippet around the issue."""
        lines = content.split('\n')
        start_line = content[:start].count('\n')
        
        context_start = max(0, start_line - 1)
        context_end = min(len(lines), start_line + 2)
        
        snippet_lines = []
        for i in range(context_start, context_end):
            prefix = ">>> " if i == start_line else "    "
            snippet_lines.append(f"{prefix}{lines[i]}")
            
        return '\n'.join(snippet_lines)


class StyleAnalyzer:
    """
    Code style and best practices analyzer.
    
    Ensures code follows style guidelines, naming conventions,
    and best practices for maintainability.
    """
    
    def __init__(self):
        self.style_patterns = {
            "naming_conventions": [
                r'\bclass\s+[a-z]',  # Class should start with uppercase
                r'\bdef\s+[A-Z]',    # Function should start with lowercase
                r'\b[A-Z_]{2,}\s*=', # Constants should be UPPER_CASE
                r'\bdef\s+\w*[A-Z]\w*[a-z]'  # Functions should be snake_case
            ],
            "line_length": [
                r'.{120,}',  # Lines longer than 120 characters
            ],
            "complexity": [
                r'if\s+.*and\s+.*and\s+.*and',  # Complex conditions
                r'elif\s+.*and\s+.*and\s+.*and',
                r'while\s+.*and\s+.*and\s+.*and'
            ],
            "documentation": [
                r'def\s+\w+\(.*\):\s*\n\s*[^"""\'\'\'#]',  # Missing docstring
                r'class\s+\w+.*:\s*\n\s*[^"""\'\'\'#]'     # Missing class docstring
            ],
            "imports": [
                r'from\s+\w+\s+import\s+\*',  # Wildcard imports
                r'import\s+\w+\s+as\s+\w',     # Import aliasing without clear reason
            ]
        }
        
    def analyze_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Analyze file for style issues."""
        
        findings = []
        
        for issue_type, patterns in self.style_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    finding = {
                        "category": ReviewCategory.STYLE.value,
                        "severity": ReviewSeverity.LOW.value,
                        "type": issue_type,
                        "file": file_path,
                        "line": line_number,
                        "column": match.start() - content.rfind('\n', 0, match.start()) - 1,
                        "message": self._get_message_for_issue(issue_type),
                        "code_snippet": self._extract_code_snippet(content, match.start(), match.end()),
                        "suggestion": self._get_suggestion_for_issue(issue_type),
                        "confidence": 0.8
                    }
                    
                    findings.append(finding)
                    
        return findings
        
    def _get_message_for_issue(self, issue_type: str) -> str:
        """Get descriptive message for style issue type."""
        messages = {
            "naming_conventions": "Naming convention violation",
            "line_length": "Line too long",
            "complexity": "Complex condition detected",
            "documentation": "Missing documentation",
            "imports": "Import style issue"
        }
        return messages.get(issue_type, "Style issue detected")
        
    def _get_suggestion_for_issue(self, issue_type: str) -> str:
        """Get style improvement suggestion."""
        suggestions = {
            "naming_conventions": "Follow PEP 8 naming conventions",
            "line_length": "Break line into multiple lines",
            "complexity": "Simplify condition or extract to separate function",
            "documentation": "Add docstring explaining purpose and parameters",
            "imports": "Use explicit imports and avoid wildcards"
        }
        return suggestions.get(issue_type, "Follow style guidelines")
        
    def _extract_code_snippet(self, content: str, start: int, end: int) -> str:
        """Extract code snippet around the issue."""
        lines = content.split('\n')
        start_line = content[:start].count('\n')
        
        return lines[start_line] if start_line < len(lines) else ""


class CodeReviewAssistant:
    """
    Comprehensive code review assistant with multi-dimensional analysis.
    
    Provides automated code review combining security, performance, and style
    analysis with intelligent prioritization and detailed suggestions.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.security_analyzer = SecurityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.style_analyzer = StyleAnalyzer()
        
        self.review_weights = {
            ReviewCategory.SECURITY.value: 1.0,
            ReviewCategory.PERFORMANCE.value: 0.8,
            ReviewCategory.STYLE.value: 0.3,
            ReviewCategory.MAINTAINABILITY.value: 0.7,
            ReviewCategory.CORRECTNESS.value: 0.9
        }
        
    async def perform_comprehensive_review(
        self,
        pull_request: PullRequest,
        review_types: List[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive automated code review."""
        
        review_types = review_types or ["security", "performance", "style"]
        review_start_time = datetime.utcnow()
        
        try:
            # Create review record
            review = CodeReview(
                pull_request_id=pull_request.id,
                reviewer_type="automated",
                review_type="comprehensive",
                review_status=ReviewStatus.PENDING
            )
            
            async with get_db_session() as session:
                session.add(review)
                await session.commit()
                await session.refresh(review)
                
            # Start review
            review.start_review()
            
            # Get file changes from PR
            file_changes = await self._get_pr_file_changes(pull_request)
            
            # Analyze each file
            all_findings = []
            for file_change in file_changes:
                file_findings = await self._analyze_file_comprehensive(
                    file_change["filename"],
                    file_change["content"],
                    review_types
                )
                all_findings.extend(file_findings)
                
            # Categorize findings
            categorized_findings = self._categorize_findings(all_findings)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(categorized_findings)
            
            # Determine approval status
            approved = self._should_approve(categorized_findings, overall_score)
            
            # Update review with results
            review.findings = all_findings
            review.security_issues = categorized_findings.get("security", [])
            review.performance_issues = categorized_findings.get("performance", [])
            review.style_issues = categorized_findings.get("style", [])
            review.overall_score = overall_score
            review.approved = approved
            review.changes_requested = not approved and len(all_findings) > 0
            
            # Complete review
            review.complete_review(approved)
            
            # Generate suggestions
            suggestions = self._generate_improvement_suggestions(categorized_findings)
            review.suggestions = suggestions
            
            # Save final review
            async with get_db_session() as session:
                await session.merge(review)
                await session.commit()
                
            # Post review to GitHub
            github_posted = await self._post_review_to_github(pull_request, review)
            
            review_duration = (datetime.utcnow() - review_start_time).total_seconds()
            
            return {
                "success": True,
                "review_id": str(review.id),
                "overall_score": overall_score,
                "approved": approved,
                "changes_requested": review.changes_requested,
                "findings_count": len(all_findings),
                "categorized_findings": {
                    category: len(findings) 
                    for category, findings in categorized_findings.items()
                },
                "suggestions_count": len(suggestions),
                "github_posted": github_posted,
                "review_duration": review_duration,
                "files_analyzed": len(file_changes)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive review failed: {e}")
            raise CodeReviewError(f"Review failed: {str(e)}")
            
    async def _get_pr_file_changes(self, pull_request: PullRequest) -> List[Dict[str, Any]]:
        """Get file changes from pull request."""
        
        try:
            # Get repository info
            async with get_db_session() as session:
                result = await session.execute(
                    select(PullRequest).options(
                        selectinload(PullRequest.repository)
                    ).where(PullRequest.id == pull_request.id)
                )
                pr_with_repo = result.scalar_one()
                
            repo_parts = pr_with_repo.repository.repository_full_name.split('/')
            
            # Use GraphQL for efficient file retrieval
            query = """
            query GetPullRequestFiles($owner: String!, $name: String!, $number: Int!) {
              repository(owner: $owner, name: $name) {
                pullRequest(number: $number) {
                  files(first: 100) {
                    nodes {
                      path
                      additions
                      deletions
                      changeType
                    }
                  }
                  commits(last: 1) {
                    nodes {
                      commit {
                        oid
                      }
                    }
                  }
                }
              }
            }
            """
            
            variables = {
                "owner": repo_parts[0],
                "name": repo_parts[1],
                "number": pull_request.github_pr_number
            }
            
            result = await self.github_client.execute_graphql(query, variables)
            
            if "data" in result and "repository" in result["data"]:
                pr_data = result["data"]["repository"]["pullRequest"]
                files = pr_data.get("files", {}).get("nodes", [])
                
                file_changes = []
                for file_info in files:
                    # Get file content
                    content = await self._get_file_content(
                        repo_parts[0], repo_parts[1],
                        file_info["path"],
                        pull_request.source_branch
                    )
                    
                    file_changes.append({
                        "filename": file_info["path"],
                        "content": content,
                        "additions": file_info["additions"],
                        "deletions": file_info["deletions"],
                        "change_type": file_info["changeType"]
                    })
                    
                return file_changes
            else:
                logger.warning("Failed to get PR files via GraphQL")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get PR file changes: {e}")
            return []
            
    async def _get_file_content(self, owner: str, repo: str, path: str, ref: str) -> str:
        """Get file content from GitHub."""
        
        try:
            response = await self.github_client._make_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{path}",
                params={"ref": ref}
            )
            
            if "content" in response:
                import base64
                return base64.b64decode(response["content"]).decode('utf-8')
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Failed to get file content for {path}: {e}")
            return ""
            
    async def _analyze_file_comprehensive(
        self,
        file_path: str,
        content: str,
        review_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Perform comprehensive analysis on a single file."""
        
        all_findings = []
        
        if "security" in review_types:
            security_findings = self.security_analyzer.analyze_file(file_path, content)
            all_findings.extend(security_findings)
            
        if "performance" in review_types:
            performance_findings = self.performance_analyzer.analyze_file(file_path, content)
            all_findings.extend(performance_findings)
            
        if "style" in review_types:
            style_findings = self.style_analyzer.analyze_file(file_path, content)
            all_findings.extend(style_findings)
            
        return all_findings
        
    def _categorize_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize findings by type."""
        
        categorized = {}
        for finding in findings:
            category = finding["category"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(finding)
            
        return categorized
        
    def _calculate_overall_score(self, categorized_findings: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall code quality score (0.0 to 1.0)."""
        
        total_weighted_issues = 0.0
        max_possible_weight = 0.0
        
        for category, findings in categorized_findings.items():
            category_weight = self.review_weights.get(category, 0.5)
            max_possible_weight += category_weight
            
            # Calculate severity-weighted issue count
            category_issue_weight = 0.0
            for finding in findings:
                severity = finding.get("severity", "medium")
                severity_weights = {
                    "critical": 1.0,
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.2,
                    "info": 0.1
                }
                category_issue_weight += severity_weights.get(severity, 0.5)
                
            # Normalize to 0-1 scale (max 10 issues per category)
            normalized_issues = min(1.0, category_issue_weight / 10.0)
            total_weighted_issues += normalized_issues * category_weight
            
        # Calculate score (higher is better)
        if max_possible_weight > 0:
            issue_ratio = total_weighted_issues / max_possible_weight
            score = max(0.0, 1.0 - issue_ratio)
        else:
            score = 1.0  # No issues found
            
        return score
        
    def _should_approve(self, categorized_findings: Dict[str, List[Dict[str, Any]]], overall_score: float) -> bool:
        """Determine if PR should be approved based on findings."""
        
        # Check for critical security issues
        security_findings = categorized_findings.get("security", [])
        critical_security = any(
            finding.get("severity") == "critical" 
            for finding in security_findings
        )
        
        if critical_security:
            return False
            
        # Check overall score threshold
        if overall_score < 0.6:  # Minimum 60% quality score
            return False
            
        # Check total high severity issues
        high_severity_count = sum(
            1 for findings in categorized_findings.values()
            for finding in findings
            if finding.get("severity") in ["critical", "high"]
        )
        
        if high_severity_count > 5:  # Too many high severity issues
            return False
            
        return True
        
    def _generate_improvement_suggestions(self, categorized_findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate prioritized improvement suggestions."""
        
        suggestions = []
        
        # Group similar issues
        issue_groups = {}
        for category, findings in categorized_findings.items():
            for finding in findings:
                issue_type = finding.get("type", "unknown")
                if issue_type not in issue_groups:
                    issue_groups[issue_type] = []
                issue_groups[issue_type].append(finding)
                
        # Generate suggestions for each issue group
        for issue_type, findings in issue_groups.items():
            if len(findings) >= 3:  # Pattern detected
                suggestion = {
                    "type": "pattern",
                    "issue_type": issue_type,
                    "category": findings[0]["category"],
                    "severity": max(finding.get("severity", "low") for finding in findings),
                    "occurrences": len(findings),
                    "files_affected": list(set(finding["file"] for finding in findings)),
                    "description": f"Pattern detected: {issue_type} appears {len(findings)} times",
                    "recommendation": self._get_pattern_recommendation(issue_type, findings),
                    "priority": self._calculate_suggestion_priority(findings)
                }
                suggestions.append(suggestion)
            else:
                # Individual suggestions
                for finding in findings:
                    suggestion = {
                        "type": "individual",
                        "issue_type": finding.get("type"),
                        "category": finding["category"],
                        "severity": finding.get("severity", "medium"),
                        "file": finding["file"],
                        "line": finding.get("line"),
                        "description": finding.get("message"),
                        "recommendation": finding.get("suggestion"),
                        "priority": self._calculate_suggestion_priority([finding])
                    }
                    suggestions.append(suggestion)
                    
        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        
        return suggestions[:20]  # Limit to top 20 suggestions
        
    def _get_pattern_recommendation(self, issue_type: str, findings: List[Dict[str, Any]]) -> str:
        """Get recommendation for repeated issue pattern."""
        
        pattern_recommendations = {
            "sql_injection": "Consider implementing a parameterized query helper function or using an ORM consistently throughout the codebase",
            "hardcoded_secrets": "Implement a centralized configuration management system for secrets and credentials",
            "inefficient_loops": "Review algorithm choices and consider vectorized operations or more efficient data structures",
            "naming_conventions": "Run a code formatter (like Black) and establish pre-commit hooks for consistent style",
            "missing_documentation": "Establish documentation standards and consider automated docstring generation tools"
        }
        
        return pattern_recommendations.get(
            issue_type, 
            f"Multiple instances of {issue_type} detected - consider a systematic approach to resolve"
        )
        
    def _calculate_suggestion_priority(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate priority score for suggestion."""
        
        severity_scores = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
            "info": 0.1
        }
        
        category_scores = {
            "security": 1.0,
            "performance": 0.8,
            "maintainability": 0.7,
            "correctness": 0.9,
            "style": 0.3
        }
        
        # Calculate average severity and category scores
        avg_severity = sum(severity_scores.get(f.get("severity", "medium"), 0.5) for f in findings) / len(findings)
        avg_category = sum(category_scores.get(f.get("category", "style"), 0.5) for f in findings) / len(findings)
        
        # Frequency bonus for patterns
        frequency_bonus = min(0.3, len(findings) * 0.1)
        
        return avg_severity * 0.5 + avg_category * 0.4 + frequency_bonus
        
    async def _post_review_to_github(self, pull_request: PullRequest, review: CodeReview) -> bool:
        """Post review results to GitHub PR."""
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(PullRequest).options(
                        selectinload(PullRequest.repository)
                    ).where(PullRequest.id == pull_request.id)
                )
                pr_with_repo = result.scalar_one()
                
            repo_parts = pr_with_repo.repository.repository_full_name.split('/')
            
            # Generate review summary
            summary = self._generate_review_summary(review)
            
            # Post review
            event = "APPROVE" if review.approved else "REQUEST_CHANGES"
            await self.github_client.create_review(
                repo_parts[0], repo_parts[1], pull_request.github_pr_number,
                event=event,
                body=summary
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to post review to GitHub: {e}")
            return False
            
    def _generate_review_summary(self, review: CodeReview) -> str:
        """Generate human-readable review summary."""
        
        summary_parts = [
            "## 🤖 Automated Code Review",
            "",
            f"**Overall Score**: {review.overall_score:.2f}/1.00",
            f"**Status**: {'✅ Approved' if review.approved else '❌ Changes Requested'}",
            ""
        ]
        
        # Findings summary
        if review.security_issues:
            summary_parts.extend([
                f"### 🔒 Security Issues ({len(review.security_issues)})",
                ""
            ])
            for issue in review.security_issues[:5]:  # Top 5 security issues
                summary_parts.append(f"- **{issue.get('file')}:{issue.get('line')}** - {issue.get('message')}")
            if len(review.security_issues) > 5:
                summary_parts.append(f"- ... and {len(review.security_issues) - 5} more security issues")
            summary_parts.append("")
            
        if review.performance_issues:
            summary_parts.extend([
                f"### ⚡ Performance Issues ({len(review.performance_issues)})",
                ""
            ])
            for issue in review.performance_issues[:3]:  # Top 3 performance issues
                summary_parts.append(f"- **{issue.get('file')}:{issue.get('line')}** - {issue.get('message')}")
            if len(review.performance_issues) > 3:
                summary_parts.append(f"- ... and {len(review.performance_issues) - 3} more performance issues")
            summary_parts.append("")
            
        # Top suggestions
        if review.suggestions:
            summary_parts.extend([
                "### 💡 Top Recommendations",
                ""
            ])
            for suggestion in review.suggestions[:3]:
                summary_parts.append(f"- {suggestion.get('recommendation')}")
            summary_parts.append("")
            
        summary_parts.extend([
            "---",
            "*Generated by LeanVibe Agent Hive Code Review Assistant*"
        ])
        
        return '\n'.join(summary_parts)
        
    async def get_review_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get code review statistics."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with get_db_session() as session:
            # Get reviews in period
            result = await session.execute(
                select(CodeReview).where(
                    and_(
                        CodeReview.created_at >= cutoff_date,
                        CodeReview.review_status == ReviewStatus.COMPLETED
                    )
                )
            )
            reviews = result.scalars().all()
            
            if not reviews:
                return {
                    "period_days": days,
                    "total_reviews": 0,
                    "average_score": 0.0,
                    "approval_rate": 0.0,
                    "average_findings": 0.0,
                    "most_common_issues": []
                }
                
            # Calculate statistics
            total_reviews = len(reviews)
            approved_reviews = len([r for r in reviews if r.approved])
            average_score = sum(r.overall_score or 0.0 for r in reviews) / total_reviews
            approval_rate = approved_reviews / total_reviews
            
            # Count findings
            all_findings = []
            for review in reviews:
                if review.findings:
                    all_findings.extend(review.findings)
                    
            average_findings = len(all_findings) / total_reviews if total_reviews > 0 else 0.0
            
            # Most common issues
            issue_counts = {}
            for finding in all_findings:
                issue_type = finding.get("type", "unknown")
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
            most_common_issues = sorted(
                issue_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                "period_days": days,
                "total_reviews": total_reviews,
                "average_score": average_score,
                "approval_rate": approval_rate,
                "average_findings": average_findings,
                "reviews_per_day": total_reviews / days,
                "most_common_issues": [
                    {"issue_type": issue, "count": count}
                    for issue, count in most_common_issues
                ],
                "category_breakdown": {
                    "security": len([f for f in all_findings if f.get("category") == "security"]),
                    "performance": len([f for f in all_findings if f.get("category") == "performance"]),
                    "style": len([f for f in all_findings if f.get("category") == "style"])
                }
            }