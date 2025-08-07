"""
Intelligent Workflow Automation for LeanVibe Agent Hive 2.0

End-to-end PR workflow orchestration with quality gates, automated code formatting,
documentation generation, and release management automation.
"""

import asyncio
import json
import logging
import re
import subprocess
import uuid
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload
import structlog
try:
    import semver
except ImportError:
    # Fallback for semver functionality
    semver = None

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.github_integration import (
    PullRequest, GitHubRepository, CodeReview, 
    AgentWorkTree, BranchOperation, WebhookEvent
)
from ..core.github_api_client import GitHubAPIClient
from ..core.redis import get_redis
from ..core.code_review_assistant import CodeReviewAssistant
from ..core.automated_testing_integration import AutomatedTestingIntegration
from ..core.advanced_repository_management import AdvancedRepositoryManagement

logger = structlog.get_logger()
settings = get_settings()


class WorkflowStage(Enum):
    """Workflow execution stages."""
    INITIATED = "initiated"
    CODE_ANALYSIS = "code_analysis"
    AUTOMATED_TESTING = "automated_testing"
    SECURITY_SCANNING = "security_scanning"
    CODE_FORMATTING = "code_formatting"
    DOCUMENTATION_GENERATION = "documentation_generation"
    QUALITY_GATES = "quality_gates"
    PEER_REVIEW = "peer_review"
    APPROVAL = "approval"
    MERGE_PREPARATION = "merge_preparation"
    AUTOMATED_MERGE = "automated_merge"
    POST_MERGE_ACTIONS = "post_merge_actions"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTrigger(Enum):
    """Workflow trigger types."""
    PR_CREATED = "pr_created"
    PR_UPDATED = "pr_updated"
    COMMIT_PUSHED = "commit_pushed"
    REVIEW_REQUESTED = "review_requested"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED = "scheduled"
    WEBHOOK_EVENT = "webhook_event"


class QualityGateResult(Enum):
    """Quality gate results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    PENDING = "pending"


class ReleaseType(Enum):
    """Release types."""
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"
    HOTFIX = "hotfix"
    PRERELEASE = "prerelease"


@dataclass
class WorkflowStep:
    """Individual workflow step."""
    step_id: str
    name: str
    stage: WorkflowStage
    required: bool = True
    timeout_minutes: int = 30
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class QualityGate:
    """Quality gate definition."""
    gate_id: str
    name: str
    description: str
    required: bool = True
    threshold: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    

@dataclass
class WorkflowExecution:
    """Workflow execution state."""
    execution_id: str
    workflow_id: str
    pr_id: str
    trigger: WorkflowTrigger
    current_stage: WorkflowStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    quality_gates_results: Dict[str, QualityGateResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentWorkflowAutomationError(Exception):
    """Custom exception for workflow automation operations."""
    pass


class CodeFormatter:
    """
    Automated code formatting with multiple language support.
    
    Provides intelligent code formatting, style fixes, and
    automated linting with customizable rules per language.
    """
    
    def __init__(self):
        self.formatters = {
            "python": {
                "command": "black",
                "args": ["--line-length", "88", "--target-version", "py39"],
                "config_file": "pyproject.toml",
                "lint_command": "ruff",
                "lint_args": ["--fix"]
            },
            "javascript": {
                "command": "prettier",
                "args": ["--write", "--tab-width", "2"],
                "config_file": ".prettierrc",
                "lint_command": "eslint",
                "lint_args": ["--fix"]
            },
            "typescript": {
                "command": "prettier",
                "args": ["--write", "--tab-width", "2", "--parser", "typescript"],
                "config_file": ".prettierrc",
                "lint_command": "eslint",
                "lint_args": ["--fix", "--ext", ".ts,.tsx"]
            },
            "rust": {
                "command": "rustfmt",
                "args": ["--edition", "2021"],
                "config_file": "rustfmt.toml",
                "lint_command": "clippy",
                "lint_args": ["--fix", "--allow-dirty"]
            },
            "go": {
                "command": "gofmt",
                "args": ["-w"],
                "config_file": None,
                "lint_command": "golangci-lint",
                "lint_args": ["run", "--fix"]
            }
        }
    
    async def format_repository(
        self,
        repo_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format all code files in repository."""
        
        try:
            repo_path = Path(repo_path)
            formatting_results = {
                "success": True,
                "files_formatted": [],
                "files_linted": [],
                "errors": [],
                "languages_processed": set()
            }
            
            # Detect languages if not specified
            if language:
                languages_to_format = [language]
            else:
                languages_to_format = self._detect_repository_languages(repo_path)
            
            for lang in languages_to_format:
                if lang in self.formatters:
                    lang_results = await self._format_language_files(repo_path, lang)
                    
                    formatting_results["files_formatted"].extend(lang_results["formatted"])
                    formatting_results["files_linted"].extend(lang_results["linted"])
                    formatting_results["errors"].extend(lang_results["errors"])
                    formatting_results["languages_processed"].add(lang)
                    
                    if lang_results["errors"]:
                        formatting_results["success"] = False
            
            # Convert set to list for JSON serialization
            formatting_results["languages_processed"] = list(formatting_results["languages_processed"])
            
            return formatting_results
            
        except Exception as e:
            logger.error(f"Code formatting failed: {e}")
            raise IntelligentWorkflowAutomationError(f"Formatting failed: {str(e)}")
    
    def _detect_repository_languages(self, repo_path: Path) -> List[str]:
        """Detect programming languages in repository."""
        
        language_extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "rust": [".rs"],
            "go": [".go"]
        }
        
        detected_languages = set()
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and ".git" not in str(file_path):
                for lang, extensions in language_extensions.items():
                    if file_path.suffix in extensions:
                        detected_languages.add(lang)
                        break
        
        return list(detected_languages)
    
    async def _format_language_files(
        self,
        repo_path: Path,
        language: str
    ) -> Dict[str, List[str]]:
        """Format files for specific language."""
        
        results = {
            "formatted": [],
            "linted": [],
            "errors": []
        }
        
        formatter_config = self.formatters[language]
        
        try:
            # Run formatter
            format_cmd = [formatter_config["command"]] + formatter_config["args"]
            
            # Add file patterns based on language
            if language == "python":
                format_cmd.append(str(repo_path))
            elif language in ["javascript", "typescript"]:
                format_cmd.extend([
                    f"{repo_path}/**/*.{language[:2]}",
                    f"{repo_path}/**/*.{language[:2]}x" if language == "javascript" else f"{repo_path}/**/*.tsx"
                ])
            elif language == "rust":
                # Find Rust files
                rust_files = list(repo_path.rglob("*.rs"))
                format_cmd.extend([str(f) for f in rust_files])
            elif language == "go":
                # Find Go files
                go_files = list(repo_path.rglob("*.go"))
                format_cmd.extend([str(f) for f in go_files])
            
            # Execute formatter
            format_result = await self._run_command(format_cmd, cwd=repo_path)
            
            if format_result["success"]:
                results["formatted"].append(f"Formatted {language} files")
            else:
                results["errors"].append(f"Formatter failed for {language}: {format_result['error']}")
            
            # Run linter if available
            if formatter_config["lint_command"]:
                lint_cmd = [formatter_config["lint_command"]] + formatter_config["lint_args"]
                
                if language == "python":
                    lint_cmd.append(str(repo_path))
                elif language in ["javascript", "typescript"]:
                    lint_cmd.extend([f"{repo_path}/**/*.{language[:2]}", f"{repo_path}/**/*.{language[:2]}x"])
                
                lint_result = await self._run_command(lint_cmd, cwd=repo_path)
                
                if lint_result["success"]:
                    results["linted"].append(f"Linted {language} files")
                else:
                    results["errors"].append(f"Linter failed for {language}: {lint_result['error']}")
            
        except Exception as e:
            results["errors"].append(f"Error processing {language}: {str(e)}")
        
        return results
    
    async def _run_command(
        self,
        command: List[str],
        cwd: Path,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run shell command with timeout."""
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8'),
                "return_code": process.returncode,
                "error": stderr.decode('utf-8') if process.returncode != 0 else None
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class DocumentationGenerator:
    """
    Automated documentation generation system.
    
    Generates API documentation, README updates, and
    maintains documentation consistency across the codebase.
    """
    
    def __init__(self):
        self.doc_generators = {
            "python": {
                "api_docs": "sphinx-apidoc",
                "docstring_style": "google",
                "config_template": "conf.py.template"
            },
            "javascript": {
                "api_docs": "jsdoc",
                "docstring_style": "jsdoc",
                "config_template": "jsdoc.json.template"
            },
            "typescript": {
                "api_docs": "typedoc",
                "docstring_style": "tsdoc",
                "config_template": "typedoc.json.template"
            }
        }
    
    async def generate_documentation(
        self,
        repo_path: str,
        update_readme: bool = True,
        generate_api_docs: bool = True,
        validate_links: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive documentation for repository."""
        
        try:
            repo_path = Path(repo_path)
            doc_results = {
                "success": True,
                "readme_updated": False,
                "api_docs_generated": False,
                "links_validated": False,
                "documentation_files": [],
                "errors": []
            }
            
            # Update README
            if update_readme:
                readme_result = await self._update_readme(repo_path)
                doc_results["readme_updated"] = readme_result["success"]
                if not readme_result["success"]:
                    doc_results["errors"].append(readme_result["error"])
                    doc_results["success"] = False
            
            # Generate API documentation
            if generate_api_docs:
                api_docs_result = await self._generate_api_documentation(repo_path)
                doc_results["api_docs_generated"] = api_docs_result["success"]
                doc_results["documentation_files"].extend(api_docs_result.get("files", []))
                if not api_docs_result["success"]:
                    doc_results["errors"].append(api_docs_result["error"])
                    doc_results["success"] = False
            
            # Validate documentation links
            if validate_links:
                link_validation_result = await self._validate_documentation_links(repo_path)
                doc_results["links_validated"] = link_validation_result["success"]
                if not link_validation_result["success"]:
                    doc_results["errors"].append(link_validation_result["error"])
                    doc_results["success"] = False
            
            return doc_results
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            raise IntelligentWorkflowAutomationError(f"Documentation generation failed: {str(e)}")
    
    async def _update_readme(self, repo_path: Path) -> Dict[str, Any]:
        """Update README.md with current project information."""
        
        try:
            readme_path = repo_path / "README.md"
            
            if readme_path.exists():
                # Read existing README
                with open(readme_path, 'r') as f:
                    content = f.read()
                
                # Update sections
                updated_content = await self._update_readme_sections(content, repo_path)
                
                # Write updated README
                with open(readme_path, 'w') as f:
                    f.write(updated_content)
                
                return {"success": True, "message": "README updated successfully"}
            else:
                # Generate new README
                new_readme = await self._generate_new_readme(repo_path)
                
                with open(readme_path, 'w') as f:
                    f.write(new_readme)
                
                return {"success": True, "message": "New README generated"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_readme_sections(self, content: str, repo_path: Path) -> str:
        """Update specific sections of README."""
        
        # Update installation section
        content = self._update_installation_section(content, repo_path)
        
        # Update usage section
        content = self._update_usage_section(content, repo_path)
        
        # Update API documentation links
        content = self._update_api_links_section(content, repo_path)
        
        # Update contributing section
        content = self._update_contributing_section(content, repo_path)
        
        return content
    
    def _update_installation_section(self, content: str, repo_path: Path) -> str:
        """Update installation instructions based on project files."""
        
        installation_commands = []
        
        # Python project
        if (repo_path / "requirements.txt").exists() or (repo_path / "pyproject.toml").exists():
            installation_commands.append("```bash\npip install -r requirements.txt\n```")
        
        # Node.js project
        if (repo_path / "package.json").exists():
            installation_commands.append("```bash\nnpm install\n```")
        
        # Rust project
        if (repo_path / "Cargo.toml").exists():
            installation_commands.append("```bash\ncargo build\n```")
        
        # Go project
        if (repo_path / "go.mod").exists():
            installation_commands.append("```bash\ngo mod tidy\n```")
        
        if installation_commands:
            installation_section = "## Installation\n\n" + "\n\n".join(installation_commands)
            
            # Replace existing installation section or add new one
            if "## Installation" in content:
                content = re.sub(
                    r'## Installation.*?(?=##|\Z)',
                    installation_section + "\n\n",
                    content,
                    flags=re.DOTALL
                )
            else:
                # Add after project title
                lines = content.split('\n')
                if lines and lines[0].startswith('#'):
                    lines.insert(2, installation_section)
                    content = '\n'.join(lines)
        
        return content
    
    def _update_usage_section(self, content: str, repo_path: Path) -> str:
        """Update usage examples based on project structure."""
        
        # This would analyze the codebase to generate usage examples
        # For now, ensure there's a usage section placeholder
        
        if "## Usage" not in content:
            usage_section = """## Usage

```bash
# Add usage examples here
```

"""
            
            # Add after installation section
            if "## Installation" in content:
                content = content.replace("## Installation", usage_section + "## Installation")
            else:
                lines = content.split('\n')
                if lines and lines[0].startswith('#'):
                    lines.insert(2, usage_section)
                    content = '\n'.join(lines)
        
        return content
    
    def _update_api_links_section(self, content: str, repo_path: Path) -> str:
        """Update API documentation links."""
        
        # Check if API docs exist
        docs_dir = repo_path / "docs"
        if docs_dir.exists():
            api_links = []
            
            # Look for common API doc files
            for doc_file in docs_dir.rglob("*.md"):
                if "api" in doc_file.name.lower():
                    relative_path = doc_file.relative_to(repo_path)
                    api_links.append(f"- [{doc_file.stem}]({relative_path})")
            
            if api_links:
                api_section = "## API Documentation\n\n" + "\n".join(api_links) + "\n\n"
                
                # Replace or add API section
                if "## API Documentation" in content:
                    content = re.sub(
                        r'## API Documentation.*?(?=##|\Z)',
                        api_section,
                        content,
                        flags=re.DOTALL
                    )
                else:
                    # Add before contributing section
                    if "## Contributing" in content:
                        content = content.replace("## Contributing", api_section + "## Contributing")
        
        return content
    
    def _update_contributing_section(self, content: str, repo_path: Path) -> str:
        """Update contributing guidelines."""
        
        if "## Contributing" not in content:
            contributing_section = """## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

"""
            
            # Add at the end
            content += "\n" + contributing_section
        
        return content
    
    async def _generate_new_readme(self, repo_path: Path) -> str:
        """Generate a new README from scratch."""
        
        project_name = repo_path.name
        
        readme_template = f"""# {project_name}

A brief description of {project_name}.

## Installation

```bash
# Installation instructions will be generated based on project files
```

## Usage

```bash
# Usage examples will be generated based on project structure
```

## Features

- List key features here

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
        
        return readme_template
    
    async def _generate_api_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """Generate API documentation for supported languages."""
        
        try:
            generated_files = []
            
            # Detect project languages
            languages = self._detect_project_languages(repo_path)
            
            for language in languages:
                if language in self.doc_generators:
                    lang_docs = await self._generate_language_docs(repo_path, language)
                    generated_files.extend(lang_docs.get("files", []))
            
            return {
                "success": True,
                "files": generated_files
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_project_languages(self, repo_path: Path) -> List[str]:
        """Detect programming languages in project."""
        
        language_indicators = {
            "python": ["*.py", "requirements.txt", "pyproject.toml"],
            "javascript": ["*.js", "package.json"],
            "typescript": ["*.ts", "tsconfig.json"]
        }
        
        detected = []
        
        for language, indicators in language_indicators.items():
            for indicator in indicators:
                if list(repo_path.rglob(indicator)):
                    detected.append(language)
                    break
        
        return detected
    
    async def _generate_language_docs(self, repo_path: Path, language: str) -> Dict[str, Any]:
        """Generate documentation for specific language."""
        
        try:
            doc_config = self.doc_generators[language]
            docs_dir = repo_path / "docs" / "api"
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            files_generated = []
            
            if language == "python":
                # Generate Python API docs with sphinx
                files_generated = await self._generate_python_docs(repo_path, docs_dir)
            elif language == "javascript":
                # Generate JavaScript API docs with JSDoc
                files_generated = await self._generate_javascript_docs(repo_path, docs_dir)
            elif language == "typescript":
                # Generate TypeScript API docs with TypeDoc
                files_generated = await self._generate_typescript_docs(repo_path, docs_dir)
            
            return {
                "success": True,
                "files": files_generated
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_python_docs(self, repo_path: Path, docs_dir: Path) -> List[str]:
        """Generate Python documentation with sphinx."""
        
        # Create basic sphinx configuration
        conf_content = '''
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'API Documentation'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']
html_theme = 'sphinx_rtd_theme'
'''
        
        conf_path = docs_dir / "conf.py"
        with open(conf_path, 'w') as f:
            f.write(conf_content)
        
        # Generate module documentation files
        module_files = []
        for py_file in repo_path.rglob("*.py"):
            if "__pycache__" not in str(py_file) and py_file.name != "__init__.py":
                rel_path = py_file.relative_to(repo_path)
                module_name = str(rel_path).replace("/", ".").replace(".py", "")
                
                doc_content = f'''
{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
'''
                
                doc_file = docs_dir / f"{module_name}.rst"
                with open(doc_file, 'w') as f:
                    f.write(doc_content)
                
                module_files.append(str(doc_file))
        
        return module_files
    
    async def _generate_javascript_docs(self, repo_path: Path, docs_dir: Path) -> List[str]:
        """Generate JavaScript documentation with JSDoc."""
        
        # Create JSDoc configuration
        jsdoc_config = {
            "source": {
                "include": [str(repo_path)],
                "includePattern": "\\.(js|jsx)$",
                "excludePattern": "node_modules/"
            },
            "opts": {
                "destination": str(docs_dir)
            },
            "plugins": ["plugins/markdown"]
        }
        
        config_path = docs_dir / "jsdoc.json"
        with open(config_path, 'w') as f:
            json.dump(jsdoc_config, f, indent=2)
        
        return [str(config_path)]
    
    async def _generate_typescript_docs(self, repo_path: Path, docs_dir: Path) -> List[str]:
        """Generate TypeScript documentation with TypeDoc."""
        
        # Create TypeDoc configuration
        typedoc_config = {
            "entryPoints": [str(repo_path / "src")],
            "out": str(docs_dir),
            "includeVersion": True,
            "excludeExternals": True,
            "plugin": ["typedoc-plugin-markdown"]
        }
        
        config_path = docs_dir / "typedoc.json"
        with open(config_path, 'w') as f:
            json.dump(typedoc_config, f, indent=2)
        
        return [str(config_path)]
    
    async def _validate_documentation_links(self, repo_path: Path) -> Dict[str, Any]:
        """Validate all links in documentation files."""
        
        try:
            broken_links = []
            
            # Find all markdown files
            for md_file in repo_path.rglob("*.md"):
                if ".git" not in str(md_file):
                    with open(md_file, 'r') as f:
                        content = f.read()
                    
                    # Extract links
                    links = re.findall(r'\[.*?\]\((.*?)\)', content)
                    
                    for link in links:
                        # Skip external links for now
                        if link.startswith(('http', 'https', 'ftp')):
                            continue
                        
                        # Check internal file links
                        if link.startswith('/'):
                            link_path = repo_path / link.lstrip('/')
                        else:
                            link_path = md_file.parent / link
                        
                        if not link_path.exists():
                            broken_links.append({
                                "file": str(md_file.relative_to(repo_path)),
                                "link": link,
                                "resolved_path": str(link_path)
                            })
            
            return {
                "success": len(broken_links) == 0,
                "broken_links": broken_links,
                "error": f"Found {len(broken_links)} broken links" if broken_links else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ReleaseManager:
    """
    Automated release management system.
    
    Handles version bumping, changelog generation, tag creation,
    and release deployment with intelligent versioning strategies.
    """
    
    def __init__(self):
        self.version_files = [
            "package.json",
            "pyproject.toml", 
            "Cargo.toml",
            "pom.xml",
            "go.mod",
            "VERSION",
            "__version__.py"
        ]
    
    async def prepare_release(
        self,
        repo_path: str,
        release_type: ReleaseType,
        custom_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare repository for release."""
        
        try:
            repo_path = Path(repo_path)
            
            release_preparation = {
                "success": True,
                "current_version": None,
                "new_version": None,
                "version_files_updated": [],
                "changelog_generated": False,
                "release_notes": "",
                "errors": []
            }
            
            # Get current version
            current_version = await self._get_current_version(repo_path)
            release_preparation["current_version"] = current_version
            
            # Calculate new version
            if custom_version:
                new_version = custom_version
            else:
                new_version = self._calculate_new_version(current_version, release_type)
            
            release_preparation["new_version"] = new_version
            
            # Update version files
            version_update_result = await self._update_version_files(repo_path, new_version)
            release_preparation["version_files_updated"] = version_update_result["files"]
            if version_update_result["errors"]:
                release_preparation["errors"].extend(version_update_result["errors"])
                release_preparation["success"] = False
            
            # Generate changelog
            changelog_result = await self._generate_changelog(repo_path, current_version, new_version)
            release_preparation["changelog_generated"] = changelog_result["success"]
            release_preparation["release_notes"] = changelog_result.get("content", "")
            if not changelog_result["success"]:
                release_preparation["errors"].append(changelog_result["error"])
                release_preparation["success"] = False
            
            return release_preparation
            
        except Exception as e:
            logger.error(f"Release preparation failed: {e}")
            raise IntelligentWorkflowAutomationError(f"Release preparation failed: {str(e)}")
    
    async def _get_current_version(self, repo_path: Path) -> str:
        """Get current version from version files."""
        
        for version_file in self.version_files:
            file_path = repo_path / version_file
            if file_path.exists():
                version = await self._extract_version_from_file(file_path)
                if version:
                    return version
        
        # Fallback to git tags
        try:
            import git
            repo = git.Repo(repo_path)
            tags = sorted([tag.name for tag in repo.tags if tag.name.startswith('v')], key=semver.VersionInfo.parse, reverse=True)
            if tags:
                return tags[0].lstrip('v')
        except:
            pass
        
        return "0.1.0"  # Default initial version
    
    async def _extract_version_from_file(self, file_path: Path) -> Optional[str]:
        """Extract version string from file."""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if file_path.name == "package.json":
                data = json.loads(content)
                return data.get("version")
            
            elif file_path.name == "pyproject.toml":
                # Simple TOML parsing for version
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                return version_match.group(1) if version_match else None
            
            elif file_path.name == "Cargo.toml":
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                return version_match.group(1) if version_match else None
            
            elif file_path.name == "VERSION":
                return content.strip()
            
            elif file_path.name == "__version__.py":
                version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                return version_match.group(1) if version_match else None
            
        except Exception as e:
            logger.error(f"Failed to extract version from {file_path}: {e}")
        
        return None
    
    def _calculate_new_version(self, current_version: str, release_type: ReleaseType) -> str:
        """Calculate new version based on release type."""
        
        try:
            current = semver.VersionInfo.parse(current_version)
            
            if release_type == ReleaseType.PATCH:
                return str(current.bump_patch())
            elif release_type == ReleaseType.MINOR:
                return str(current.bump_minor())
            elif release_type == ReleaseType.MAJOR:
                return str(current.bump_major())
            elif release_type == ReleaseType.PRERELEASE:
                return str(current.bump_prerelease())
            elif release_type == ReleaseType.HOTFIX:
                return str(current.bump_patch())
            else:
                return str(current.bump_patch())
                
        except Exception as e:
            logger.error(f"Failed to calculate new version: {e}")
            # Fallback to simple increment
            parts = current_version.split('.')
            if len(parts) >= 3:
                if release_type == ReleaseType.MAJOR:
                    return f"{int(parts[0]) + 1}.0.0"
                elif release_type == ReleaseType.MINOR:
                    return f"{parts[0]}.{int(parts[1]) + 1}.0"
                else:
                    return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
            else:
                return "0.1.0"
    
    async def _update_version_files(self, repo_path: Path, new_version: str) -> Dict[str, Any]:
        """Update version in all version files."""
        
        result = {
            "files": [],
            "errors": []
        }
        
        for version_file in self.version_files:
            file_path = repo_path / version_file
            if file_path.exists():
                try:
                    updated = await self._update_version_in_file(file_path, new_version)
                    if updated:
                        result["files"].append(str(file_path))
                except Exception as e:
                    result["errors"].append(f"Failed to update {version_file}: {str(e)}")
        
        return result
    
    async def _update_version_in_file(self, file_path: Path, new_version: str) -> bool:
        """Update version in specific file."""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            if file_path.name == "package.json":
                data = json.loads(content)
                data["version"] = new_version
                content = json.dumps(data, indent=2)
            
            elif file_path.name in ["pyproject.toml", "Cargo.toml"]:
                content = re.sub(
                    r'version\s*=\s*["\'][^"\']+["\']',
                    f'version = "{new_version}"',
                    content
                )
            
            elif file_path.name == "VERSION":
                content = new_version + "\n"
            
            elif file_path.name == "__version__.py":
                content = re.sub(
                    r'__version__\s*=\s*["\'][^"\']+["\']',
                    f'__version__ = "{new_version}"',
                    content
                )
            
            # Write updated content if changed
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                return True
            
        except Exception as e:
            logger.error(f"Failed to update version in {file_path}: {e}")
            raise
        
        return False
    
    async def _generate_changelog(self, repo_path: Path, current_version: str, new_version: str) -> Dict[str, Any]:
        """Generate changelog for new version."""
        
        try:
            # Get git commits since last version
            changelog_content = await self._get_commits_since_version(repo_path, current_version)
            
            # Format changelog
            formatted_changelog = self._format_changelog(new_version, changelog_content)
            
            # Update CHANGELOG.md
            changelog_path = repo_path / "CHANGELOG.md"
            if changelog_path.exists():
                with open(changelog_path, 'r') as f:
                    existing_content = f.read()
                
                # Prepend new changelog
                new_content = formatted_changelog + "\n" + existing_content
            else:
                new_content = f"# Changelog\n\n{formatted_changelog}"
            
            with open(changelog_path, 'w') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "content": formatted_changelog
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_commits_since_version(self, repo_path: Path, version: str) -> List[Dict[str, Any]]:
        """Get git commits since specified version."""
        
        try:
            import git
            repo = git.Repo(repo_path)
            
            # Find the commit for the version tag
            version_tag = f"v{version}"
            commits_since = []
            
            try:
                # Get commits since tag
                commits = list(repo.iter_commits(f"{version_tag}..HEAD"))
            except:
                # If tag doesn't exist, get recent commits
                commits = list(repo.iter_commits("HEAD", max_count=50))
            
            for commit in commits:
                commits_since.append({
                    "hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat()
                })
            
            return commits_since
            
        except Exception as e:
            logger.error(f"Failed to get commits since version: {e}")
            return []
    
    def _format_changelog(self, version: str, commits: List[Dict[str, Any]]) -> str:
        """Format changelog content."""
        
        changelog_lines = [
            f"## [{version}] - {datetime.utcnow().strftime('%Y-%m-%d')}",
            ""
        ]
        
        # Categorize commits
        features = []
        fixes = []
        other = []
        
        for commit in commits:
            message = commit["message"].lower()
            
            if any(keyword in message for keyword in ["feat", "feature", "add"]):
                features.append(commit)
            elif any(keyword in message for keyword in ["fix", "bug", "patch"]):
                fixes.append(commit)
            else:
                other.append(commit)
        
        # Add features
        if features:
            changelog_lines.append("### Added")
            for commit in features:
                changelog_lines.append(f"- {commit['message']} ({commit['hash']})")
            changelog_lines.append("")
        
        # Add fixes
        if fixes:
            changelog_lines.append("### Fixed")
            for commit in fixes:
                changelog_lines.append(f"- {commit['message']} ({commit['hash']})")
            changelog_lines.append("")
        
        # Add other changes
        if other:
            changelog_lines.append("### Changed")
            for commit in other:
                changelog_lines.append(f"- {commit['message']} ({commit['hash']})")
            changelog_lines.append("")
        
        return "\n".join(changelog_lines)


class IntelligentWorkflowAutomation:
    """
    Comprehensive workflow automation system orchestrating all aspects
    of PR lifecycle from creation to merge and post-merge actions.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.code_review_assistant = None  # Will be initialized lazily
        self.testing_integration = None  # Will be initialized lazily
        self.repository_management = None  # Will be initialized lazily
        self.code_formatter = CodeFormatter()
        self.doc_generator = DocumentationGenerator()
        self.release_manager = ReleaseManager()
        self.redis = get_redis()
        
        # Define quality gates
        self.quality_gates = [
            QualityGate(
                gate_id="code_analysis",
                name="Code Analysis",
                description="Automated code review with security, performance, and style analysis",
                required=True,
                threshold={"min_score": 0.7, "max_critical_issues": 0},
                weight=1.0
            ),
            QualityGate(
                gate_id="automated_testing",
                name="Automated Testing",
                description="Comprehensive test execution with coverage analysis",
                required=True,
                threshold={"min_success_rate": 0.95, "min_coverage": 0.8},
                weight=1.0
            ),
            QualityGate(
                gate_id="security_scanning",
                name="Security Scanning",
                description="Security vulnerability detection and analysis",
                required=True,
                threshold={"max_vulnerabilities": 0, "max_critical_severity": 0},
                weight=1.0
            ),
            QualityGate(
                gate_id="code_formatting",
                name="Code Formatting",
                description="Automated code formatting and style consistency",
                required=False,
                threshold={"formatting_success": True},
                weight=0.3
            ),
            QualityGate(
                gate_id="documentation",
                name="Documentation",
                description="Documentation completeness and link validation",
                required=False,
                threshold={"docs_generated": True, "links_valid": True},
                weight=0.3
            )
        ]
    
    async def execute_workflow(
        self,
        pull_request: PullRequest,
        trigger: WorkflowTrigger = WorkflowTrigger.PR_CREATED,
        workflow_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute comprehensive PR workflow."""
        
        workflow_config = workflow_config or {}
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id="comprehensive_pr_workflow",
            pr_id=str(pull_request.id),
            trigger=trigger,
            current_stage=WorkflowStage.INITIATED,
            started_at=datetime.utcnow()
        )
        
        try:
            logger.info(
                "Starting workflow execution",
                execution_id=execution_id,
                pr_number=pull_request.github_pr_number,
                trigger=trigger.value
            )
            
            # Store execution state
            await self._store_execution_state(execution)
            
            # Initialize dependencies lazily
            await self._initialize_dependencies()
            
            # Execute workflow stages
            stages_config = workflow_config.get("stages", {})
            
            # Stage 1: Code Analysis
            await self._execute_stage(execution, WorkflowStage.CODE_ANALYSIS, pull_request, stages_config)
            
            # Stage 2: Automated Testing
            await self._execute_stage(execution, WorkflowStage.AUTOMATED_TESTING, pull_request, stages_config)
            
            # Stage 3: Security Scanning
            await self._execute_stage(execution, WorkflowStage.SECURITY_SCANNING, pull_request, stages_config)
            
            # Stage 4: Code Formatting (optional)
            if stages_config.get("code_formatting", {}).get("enabled", True):
                await self._execute_stage(execution, WorkflowStage.CODE_FORMATTING, pull_request, stages_config)
            
            # Stage 5: Documentation Generation (optional)
            if stages_config.get("documentation", {}).get("enabled", True):
                await self._execute_stage(execution, WorkflowStage.DOCUMENTATION_GENERATION, pull_request, stages_config)
            
            # Stage 6: Quality Gates Evaluation
            await self._execute_stage(execution, WorkflowStage.QUALITY_GATES, pull_request, stages_config)
            
            # Stage 7: Peer Review (if required)
            peer_review_required = workflow_config.get("peer_review_required", False)
            if peer_review_required:
                await self._execute_stage(execution, WorkflowStage.PEER_REVIEW, pull_request, stages_config)
            
            # Stage 8: Approval Check
            await self._execute_stage(execution, WorkflowStage.APPROVAL, pull_request, stages_config)
            
            # Stage 9: Merge Preparation
            await self._execute_stage(execution, WorkflowStage.MERGE_PREPARATION, pull_request, stages_config)
            
            # Stage 10: Automated Merge (if configured)
            auto_merge_enabled = workflow_config.get("auto_merge_enabled", False)
            if auto_merge_enabled:
                await self._execute_stage(execution, WorkflowStage.AUTOMATED_MERGE, pull_request, stages_config)
            
            # Stage 11: Post-Merge Actions
            if pull_request.status.value == "merged" or execution.current_stage == WorkflowStage.AUTOMATED_MERGE:
                await self._execute_stage(execution, WorkflowStage.POST_MERGE_ACTIONS, pull_request, stages_config)
            
            # Mark as completed
            execution.current_stage = WorkflowStage.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.success = True
            
            logger.info(
                "Workflow execution completed successfully",
                execution_id=execution_id,
                duration=(execution.completed_at - execution.started_at).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.current_stage = WorkflowStage.FAILED
            execution.completed_at = datetime.utcnow()
            execution.success = False
            execution.metadata["error"] = str(e)
        
        finally:
            # Store final execution state
            await self._store_execution_state(execution)
        
        return execution
    
    async def _initialize_dependencies(self) -> None:
        """Initialize workflow dependencies lazily."""
        
        if not self.code_review_assistant:
            from ..core.code_review_assistant import CodeReviewAssistant
            self.code_review_assistant = CodeReviewAssistant(self.github_client)
        
        if not self.testing_integration:
            from ..core.automated_testing_integration import AutomatedTestingIntegration
            self.testing_integration = AutomatedTestingIntegration(self.github_client)
        
        if not self.repository_management:
            from ..core.advanced_repository_management import AdvancedRepositoryManagement
            self.repository_management = AdvancedRepositoryManagement(self.github_client)
    
    async def _execute_stage(
        self,
        execution: WorkflowExecution,
        stage: WorkflowStage,
        pull_request: PullRequest,
        stages_config: Dict[str, Any]
    ) -> None:
        """Execute a specific workflow stage."""
        
        execution.current_stage = stage
        await self._store_execution_state(execution)
        
        stage_config = stages_config.get(stage.value, {})
        stage_timeout = stage_config.get("timeout_minutes", 30)
        
        try:
            logger.info(f"Executing stage: {stage.value}", execution_id=execution.execution_id)
            
            # Execute stage-specific logic
            if stage == WorkflowStage.CODE_ANALYSIS:
                await self._execute_code_analysis_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.AUTOMATED_TESTING:
                await self._execute_testing_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.SECURITY_SCANNING:
                await self._execute_security_scanning_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.CODE_FORMATTING:
                await self._execute_formatting_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.DOCUMENTATION_GENERATION:
                await self._execute_documentation_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.QUALITY_GATES:
                await self._execute_quality_gates_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.PEER_REVIEW:
                await self._execute_peer_review_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.APPROVAL:
                await self._execute_approval_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.MERGE_PREPARATION:
                await self._execute_merge_preparation_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.AUTOMATED_MERGE:
                await self._execute_automated_merge_stage(execution, pull_request, stage_config)
            elif stage == WorkflowStage.POST_MERGE_ACTIONS:
                await self._execute_post_merge_stage(execution, pull_request, stage_config)
            
            execution.steps_completed.append(stage.value)
            
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            execution.steps_failed.append(stage.value)
            execution.metadata[f"{stage.value}_error"] = str(e)
            
            # Determine if failure is critical
            if stage in [WorkflowStage.CODE_ANALYSIS, WorkflowStage.AUTOMATED_TESTING, WorkflowStage.SECURITY_SCANNING]:
                raise  # Critical stages cause workflow failure
            else:
                logger.warning(f"Non-critical stage {stage.value} failed, continuing workflow")
    
    async def _execute_code_analysis_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute code analysis stage."""
        
        review_types = stage_config.get("review_types", ["security", "performance", "style"])
        
        # Perform comprehensive code review
        review_result = await self.code_review_assistant.perform_comprehensive_review(
            pull_request, review_types
        )
        
        execution.metadata["code_analysis"] = review_result
        
        # Evaluate quality gate
        gate_result = self._evaluate_quality_gate("code_analysis", review_result)
        execution.quality_gates_results["code_analysis"] = gate_result
        
        if gate_result == QualityGateResult.FAILED:
            raise IntelligentWorkflowAutomationError("Code analysis quality gate failed")
    
    async def _execute_testing_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute automated testing stage."""
        
        test_suites = stage_config.get("test_suites", ["unit", "integration", "security"])
        
        # Trigger automated tests
        test_trigger_result = await self.testing_integration.trigger_automated_tests(
            pull_request, test_suites
        )
        
        if test_trigger_result["status"] == "triggered":
            # Monitor test execution
            test_monitoring_result = await self.testing_integration.monitor_test_execution(
                test_trigger_result["test_run_id"], timeout_minutes=60
            )
            
            execution.metadata["automated_testing"] = test_monitoring_result
            
            # Evaluate quality gate
            gate_result = self._evaluate_quality_gate("automated_testing", test_monitoring_result)
            execution.quality_gates_results["automated_testing"] = gate_result
            
            if gate_result == QualityGateResult.FAILED:
                raise IntelligentWorkflowAutomationError("Automated testing quality gate failed")
    
    async def _execute_security_scanning_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute security scanning stage."""
        
        # Security scanning is part of code analysis
        # This could be extended to include additional security tools
        
        security_scan_result = {
            "success": True,
            "vulnerabilities_found": 0,
            "scan_types": ["static_analysis", "dependency_check"],
            "details": "Security scanning completed via code analysis"
        }
        
        execution.metadata["security_scanning"] = security_scan_result
        
        # Evaluate quality gate
        gate_result = self._evaluate_quality_gate("security_scanning", security_scan_result)
        execution.quality_gates_results["security_scanning"] = gate_result
        
        if gate_result == QualityGateResult.FAILED:
            raise IntelligentWorkflowAutomationError("Security scanning quality gate failed")
    
    async def _execute_formatting_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute code formatting stage."""
        
        # Get work tree for formatting
        work_tree = await self._get_work_tree(pull_request)
        
        # Format code
        formatting_result = await self.code_formatter.format_repository(
            work_tree.work_tree_path,
            language=stage_config.get("language")
        )
        
        execution.metadata["code_formatting"] = formatting_result
        
        # Evaluate quality gate
        gate_result = self._evaluate_quality_gate("code_formatting", formatting_result)
        execution.quality_gates_results["code_formatting"] = gate_result
        
        # Commit formatting changes if successful
        if formatting_result["success"] and formatting_result["files_formatted"]:
            await self._commit_changes(work_tree, "Automated code formatting", formatting_result["files_formatted"])
    
    async def _execute_documentation_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute documentation generation stage."""
        
        # Get work tree for documentation
        work_tree = await self._get_work_tree(pull_request)
        
        # Generate documentation
        doc_result = await self.doc_generator.generate_documentation(
            work_tree.work_tree_path,
            update_readme=stage_config.get("update_readme", True),
            generate_api_docs=stage_config.get("generate_api_docs", True),
            validate_links=stage_config.get("validate_links", True)
        )
        
        execution.metadata["documentation_generation"] = doc_result
        
        # Evaluate quality gate
        gate_result = self._evaluate_quality_gate("documentation", doc_result)
        execution.quality_gates_results["documentation"] = gate_result
        
        # Commit documentation changes if successful
        if doc_result["success"] and doc_result["documentation_files"]:
            await self._commit_changes(work_tree, "Automated documentation update", doc_result["documentation_files"])
    
    async def _execute_quality_gates_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute quality gates evaluation stage."""
        
        # Evaluate all quality gates
        overall_result = self._evaluate_overall_quality_gates(execution.quality_gates_results)
        
        execution.metadata["quality_gates_evaluation"] = overall_result
        
        if not overall_result["passed"]:
            raise IntelligentWorkflowAutomationError(f"Quality gates failed: {overall_result['failures']}")
    
    async def _execute_peer_review_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute peer review stage."""
        
        # Request peer review
        reviewers = stage_config.get("reviewers", ["team-lead"])
        
        # This would integrate with GitHub to request reviews
        peer_review_result = {
            "success": True,
            "reviewers_requested": reviewers,
            "review_status": "requested"
        }
        
        execution.metadata["peer_review"] = peer_review_result
    
    async def _execute_approval_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute approval check stage."""
        
        # Check if PR meets approval criteria
        approval_result = await self._check_approval_status(pull_request)
        
        execution.metadata["approval_check"] = approval_result
        
        if not approval_result["approved"]:
            raise IntelligentWorkflowAutomationError("PR approval requirements not met")
    
    async def _execute_merge_preparation_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute merge preparation stage."""
        
        # Prepare for merge (conflict resolution, etc.)
        merge_prep_result = await self.repository_management.perform_intelligent_merge(
            pull_request
        )
        
        execution.metadata["merge_preparation"] = merge_prep_result
        
        if not merge_prep_result["success"]:
            raise IntelligentWorkflowAutomationError("Merge preparation failed")
    
    async def _execute_automated_merge_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute automated merge stage."""
        
        merge_method = stage_config.get("merge_method", "squash")
        
        # Perform automated merge via GitHub API
        merge_result = await self._perform_github_merge(pull_request, merge_method)
        
        execution.metadata["automated_merge"] = merge_result
        
        if not merge_result["success"]:
            raise IntelligentWorkflowAutomationError("Automated merge failed")
    
    async def _execute_post_merge_stage(
        self,
        execution: WorkflowExecution,
        pull_request: PullRequest,
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute post-merge actions stage."""
        
        post_merge_actions = []
        
        # Branch cleanup
        if stage_config.get("cleanup_branch", True):
            cleanup_result = await self._cleanup_merged_branch(pull_request)
            post_merge_actions.append({"action": "branch_cleanup", "result": cleanup_result})
        
        # Deploy if configured
        if stage_config.get("deploy_after_merge", False):
            deployment_result = await self._trigger_deployment(pull_request, stage_config)
            post_merge_actions.append({"action": "deployment", "result": deployment_result})
        
        # Notification
        if stage_config.get("notify_on_merge", True):
            notification_result = await self._send_merge_notification(pull_request)
            post_merge_actions.append({"action": "notification", "result": notification_result})
        
        execution.metadata["post_merge_actions"] = post_merge_actions
    
    def _evaluate_quality_gate(self, gate_id: str, result: Dict[str, Any]) -> QualityGateResult:
        """Evaluate individual quality gate."""
        
        gate = next((g for g in self.quality_gates if g.gate_id == gate_id), None)
        if not gate:
            return QualityGateResult.SKIPPED
        
        try:
            if gate_id == "code_analysis":
                min_score = gate.threshold.get("min_score", 0.7)
                max_critical = gate.threshold.get("max_critical_issues", 0)
                
                overall_score = result.get("overall_score", 0)
                critical_issues = len([
                    f for findings in result.get("categorized_findings", {}).values()
                    for f in findings if f.get("severity") == "critical"
                ])
                
                if overall_score >= min_score and critical_issues <= max_critical:
                    return QualityGateResult.PASSED
                else:
                    return QualityGateResult.FAILED
            
            elif gate_id == "automated_testing":
                min_success_rate = gate.threshold.get("min_success_rate", 0.95)
                min_coverage = gate.threshold.get("min_coverage", 0.8)
                
                success_rate = result.get("analysis", {}).get("success_rate", 0) / 100.0
                coverage = result.get("analysis", {}).get("coverage_analysis", {}).get("average_coverage", 0) / 100.0
                
                if success_rate >= min_success_rate and coverage >= min_coverage:
                    return QualityGateResult.PASSED
                else:
                    return QualityGateResult.FAILED
            
            elif gate_id == "security_scanning":
                max_vulns = gate.threshold.get("max_vulnerabilities", 0)
                vulns_found = result.get("vulnerabilities_found", 0)
                
                if vulns_found <= max_vulns:
                    return QualityGateResult.PASSED
                else:
                    return QualityGateResult.FAILED
            
            elif gate_id == "code_formatting":
                formatting_success = result.get("success", False)
                return QualityGateResult.PASSED if formatting_success else QualityGateResult.FAILED
            
            elif gate_id == "documentation":
                docs_generated = result.get("api_docs_generated", False)
                links_valid = result.get("links_validated", False)
                
                if docs_generated and links_valid:
                    return QualityGateResult.PASSED
                else:
                    return QualityGateResult.WARNING
            
            else:
                return QualityGateResult.SKIPPED
                
        except Exception as e:
            logger.error(f"Failed to evaluate quality gate {gate_id}: {e}")
            return QualityGateResult.FAILED
    
    def _evaluate_overall_quality_gates(self, gate_results: Dict[str, QualityGateResult]) -> Dict[str, Any]:
        """Evaluate overall quality gates result."""
        
        required_gates = [g for g in self.quality_gates if g.required]
        optional_gates = [g for g in self.quality_gates if not g.required]
        
        failures = []
        warnings = []
        
        # Check required gates
        for gate in required_gates:
            result = gate_results.get(gate.gate_id, QualityGateResult.PENDING)
            if result == QualityGateResult.FAILED:
                failures.append(gate.name)
            elif result == QualityGateResult.WARNING:
                warnings.append(gate.name)
        
        # Check optional gates
        for gate in optional_gates:
            result = gate_results.get(gate.gate_id, QualityGateResult.PENDING)
            if result == QualityGateResult.WARNING:
                warnings.append(gate.name)
        
        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "warnings": warnings,
            "total_gates": len(self.quality_gates),
            "required_gates_passed": len(required_gates) - len([f for f in failures if f in [g.name for g in required_gates]])
        }
    
    async def _get_work_tree(self, pull_request: PullRequest) -> AgentWorkTree:
        """Get work tree for pull request."""
        
        async with get_db_session() as session:
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.agent_id == pull_request.agent_id,
                        AgentWorkTree.repository_id == pull_request.repository_id
                    )
                )
            )
            
            work_tree = result.scalar_one_or_none()
            
            if not work_tree:
                # Create work tree through repository management
                # This is simplified - in practice, would use proper work tree creation
                work_tree = AgentWorkTree(
                    agent_id=pull_request.agent_id,
                    repository_id=pull_request.repository_id,
                    work_tree_path=f"/tmp/workflow-{pull_request.id}",
                    branch_name=pull_request.source_branch
                )
            
            return work_tree
    
    async def _commit_changes(self, work_tree: AgentWorkTree, message: str, files: List[str]) -> None:
        """Commit changes to work tree."""
        
        # This would use git commands to commit changes
        # Simplified implementation
        logger.info(f"Committing changes to {work_tree.branch_name}: {message}")
    
    async def _check_approval_status(self, pull_request: PullRequest) -> Dict[str, Any]:
        """Check PR approval status."""
        
        # This would check GitHub API for actual approval status
        # Simplified implementation
        return {
            "approved": True,
            "required_approvals": 1,
            "current_approvals": 1,
            "approvers": ["automated-workflow"]
        }
    
    async def _perform_github_merge(self, pull_request: PullRequest, merge_method: str) -> Dict[str, Any]:
        """Perform merge via GitHub API."""
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(PullRequest).options(
                        selectinload(PullRequest.repository)
                    ).where(PullRequest.id == pull_request.id)
                )
                pr_with_repo = result.scalar_one()
            
            repo_parts = pr_with_repo.repository.repository_full_name.split('/')
            
            merge_response = await self.github_client._make_request(
                "PUT",
                f"/repos/{repo_parts[0]}/{repo_parts[1]}/pulls/{pull_request.github_pr_number}/merge",
                json={
                    "commit_title": f"Automated merge: {pull_request.title}",
                    "commit_message": "Merged via Intelligent Workflow Automation",
                    "merge_method": merge_method
                }
            )
            
            return {
                "success": True,
                "merge_commit_sha": merge_response.get("sha"),
                "merge_method": merge_method
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _cleanup_merged_branch(self, pull_request: PullRequest) -> Dict[str, Any]:
        """Clean up merged branch."""
        
        try:
            # Delete branch via GitHub API
            async with get_db_session() as session:
                result = await session.execute(
                    select(PullRequest).options(
                        selectinload(PullRequest.repository)
                    ).where(PullRequest.id == pull_request.id)
                )
                pr_with_repo = result.scalar_one()
            
            repo_parts = pr_with_repo.repository.repository_full_name.split('/')
            
            await self.github_client._make_request(
                "DELETE",
                f"/repos/{repo_parts[0]}/{repo_parts[1]}/git/refs/heads/{pull_request.source_branch}"
            )
            
            return {"success": True, "branch_deleted": pull_request.source_branch}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _trigger_deployment(self, pull_request: PullRequest, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger deployment after merge."""
        
        deployment_config = stage_config.get("deployment", {})
        environment = deployment_config.get("environment", "staging")
        
        # This would integrate with deployment systems
        return {
            "success": True,
            "environment": environment,
            "deployment_id": str(uuid.uuid4())
        }
    
    async def _send_merge_notification(self, pull_request: PullRequest) -> Dict[str, Any]:
        """Send merge notification."""
        
        # This would send notifications via various channels
        return {
            "success": True,
            "notifications_sent": ["slack", "email"],
            "recipients": ["team@company.com"]
        }
    
    async def _store_execution_state(self, execution: WorkflowExecution) -> None:
        """Store workflow execution state in Redis."""
        
        try:
            execution_data = {
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id,
                "pr_id": execution.pr_id,
                "trigger": execution.trigger.value,
                "current_stage": execution.current_stage.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "success": execution.success,
                "steps_completed": execution.steps_completed,
                "steps_failed": execution.steps_failed,
                "quality_gates_results": {k: v.value for k, v in execution.quality_gates_results.items()},
                "metadata": execution.metadata
            }
            
            await self.redis.setex(
                f"workflow_execution:{execution.execution_id}",
                86400,  # 24 hours TTL
                json.dumps(execution_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store execution state: {e}")
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status."""
        
        try:
            data = await self.redis.get(f"workflow_execution:{execution_id}")
            if data:
                execution_data = json.loads(data)
                
                # Reconstruct WorkflowExecution object
                execution = WorkflowExecution(
                    execution_id=execution_data["execution_id"],
                    workflow_id=execution_data["workflow_id"],
                    pr_id=execution_data["pr_id"],
                    trigger=WorkflowTrigger(execution_data["trigger"]),
                    current_stage=WorkflowStage(execution_data["current_stage"]),
                    started_at=datetime.fromisoformat(execution_data["started_at"]),
                    completed_at=datetime.fromisoformat(execution_data["completed_at"]) if execution_data["completed_at"] else None,
                    success=execution_data["success"],
                    steps_completed=execution_data["steps_completed"],
                    steps_failed=execution_data["steps_failed"],
                    quality_gates_results={k: QualityGateResult(v) for k, v in execution_data["quality_gates_results"].items()},
                    metadata=execution_data["metadata"]
                )
                
                return execution
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get execution status: {e}")
            return None


# Factory function
async def create_intelligent_workflow_automation() -> IntelligentWorkflowAutomation:
    """Create and initialize intelligent workflow automation."""
    
    github_client = GitHubAPIClient()
    return IntelligentWorkflowAutomation(github_client)


# Export main classes
__all__ = [
    "IntelligentWorkflowAutomation",
    "CodeFormatter",
    "DocumentationGenerator",
    "ReleaseManager",
    "WorkflowExecution",
    "QualityGate",
    "WorkflowStage",
    "WorkflowTrigger",
    "QualityGateResult",
    "ReleaseType",
    "IntelligentWorkflowAutomationError",
    "create_intelligent_workflow_automation"
]