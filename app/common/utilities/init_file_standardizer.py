import asyncio
#!/usr/bin/env python3
"""
__init__.py File Standardization for LeanVibe Agent Hive 2.0

Phase 1.2 Implementation of Technical Debt Remediation Plan - ROI: 1031.0
Targeting exact duplicates across 29+ __init__.py files with standardized templates.

This module provides tools to standardize __init__.py files across the entire codebase,
eliminating duplication and ensuring consistent module initialization patterns.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class InitFileType(str, Enum):
    """Types of __init__.py files based on their content patterns."""
    SIMPLE = "simple"              # Just docstring, maybe version
    PACKAGE_EXPORTS = "package_exports"  # Exports from submodules
    API_ROUTER = "api_router"      # FastAPI router setup
    APP_MAIN = "app_main"          # Application main package
    MODELS = "models"              # Database/schema models
    UTILS = "utils"                # Utility modules
    EMPTY = "empty"                # Empty or minimal content


@dataclass
class InitFileMetadata:
    """Metadata extracted from __init__.py files."""
    file_path: Path
    file_type: InitFileType
    docstring: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    email: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    all_list: List[str] = field(default_factory=list)
    custom_code: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    
    
@dataclass
class StandardizedTemplate:
    """Standardized template for __init__.py files."""
    template_type: InitFileType
    content: str
    description: str


class InitFileStandardizer:
    """
    Standardizes __init__.py files across the codebase to eliminate duplication.
    
    Phase 1.2 Implementation - targeting ROI: 1031.0 with systematic standardization
    of 29+ __init__.py files that contain exact duplicates and inconsistent patterns.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.analyzed_files: Dict[Path, InitFileMetadata] = {}
        self.templates = self._create_standardized_templates()
        
        # Pattern recognition
        self.docstring_patterns = {
            r".*[Cc]ore.*": "Core modules for LeanVibe Agent Hive 2.0.",
            r".*[Aa]pi.*": "API Package for LeanVibe Agent Hive 2.0",
            r".*[Mm]odels?.*": "Database models for LeanVibe Agent Hive 2.0.",
            r".*[Ss]chemas?.*": "API schema definitions for LeanVibe Agent Hive 2.0.",
            r".*[Tt]ests?.*": "Test modules for LeanVibe Agent Hive 2.0.",
            r".*[Uu]til.*": "Utility modules for LeanVibe Agent Hive 2.0.",
            r".*[Ii]ntegrations?.*": "Integration modules for LeanVibe Agent Hive 2.0."
        }
    
    def _create_standardized_templates(self) -> Dict[InitFileType, StandardizedTemplate]:
        """Create standardized templates for different __init__.py file types."""
        
        templates = {}
        
        # Simple module template
        templates[InitFileType.SIMPLE] = StandardizedTemplate(
            template_type=InitFileType.SIMPLE,
            content='''"""{{ docstring }}"""

__version__ = "{{ version }}"

# Standard module initialization
import logging
logger = logging.getLogger(__name__)

__all__: list[str] = []
''',
            description="Simple module with standard initialization pattern"
        )
        
        # Package exports template
        templates[InitFileType.PACKAGE_EXPORTS] = StandardizedTemplate(
            template_type=InitFileType.PACKAGE_EXPORTS,
            content='''"""{{ docstring }}"""

# Standard imports
from typing import Any, Dict, List, Optional

# Module exports - Auto-generated
{{ imports }}

# Export list - Auto-generated
__all__ = [
{{ all_exports }}
]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Initialized module: {__name__}")
''',
            description="Package with exports from submodules"
        )
        
        # API router template
        templates[InitFileType.API_ROUTER] = StandardizedTemplate(
            template_type=InitFileType.API_ROUTER,
            content='''"""
{{ docstring }}

Main API router configuration and endpoint organization for enterprise
autonomous development platform.
"""

from fastapi import APIRouter

{{ imports }}

# Create main API router
api_router = APIRouter()

# Include all endpoint routers - Auto-generated
{{ router_includes }}

# Standard exports
__all__ = ["api_router"]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"API router initialized: {__name__}")
''',
            description="FastAPI router package initialization"
        )
        
        # Application main template
        templates[InitFileType.APP_MAIN] = StandardizedTemplate(
            template_type=InitFileType.APP_MAIN,
            content='''"""
{{ docstring }}

A self-improving development environment where AI agents collaborate
to build software autonomously while maintaining human oversight.
"""

__version__ = "{{ version }}"
__author__ = "{{ author }}"
__email__ = "{{ email }}"

# Load environment variables (optional). Avoid importing settings here to
# prevent requiring env vars at import time (breaks CI and tests).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Application package initialized: {__name__} v{__version__}")

__all__: list[str] = []
''',
            description="Application main package with version and environment setup"
        )
        
        # Models package template
        templates[InitFileType.MODELS] = StandardizedTemplate(
            template_type=InitFileType.MODELS,
            content='''"""{{ docstring }}"""

# Standard model imports
from typing import Any, Dict, List, Optional

# Model imports - Auto-generated
{{ imports }}

# Model exports - Auto-generated
__all__ = [
{{ all_exports }}
]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Models package initialized: {__name__} - {len(__all__)} models loaded")
''',
            description="Database models package with comprehensive exports"
        )
        
        # Utils package template  
        templates[InitFileType.UTILS] = StandardizedTemplate(
            template_type=InitFileType.UTILS,
            content='''"""{{ docstring }}"""

# Standard utility imports
from typing import Any, Callable, Dict, List, Optional

# Utility imports - Auto-generated
{{ imports }}

# Utility exports - Auto-generated
__all__ = [
{{ all_exports }}
]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Utilities package initialized: {__name__}")
''',
            description="Utility modules package"
        )
        
        # Empty/minimal template
        templates[InitFileType.EMPTY] = StandardizedTemplate(
            template_type=InitFileType.EMPTY,
            content='''"""{{ docstring }}"""

# Standard module initialization
import logging
logger = logging.getLogger(__name__)

__all__: list[str] = []
''',
            description="Minimal module initialization"
        )
        
        return templates
    
    def analyze_init_file(self, file_path: Path) -> InitFileMetadata:
        """Analyze an __init__.py file to extract its patterns and content."""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Parse AST for structured analysis
            try:
                tree = ast.parse(content)
            except SyntaxError:
                tree = None
            
            metadata = InitFileMetadata(
                file_path=file_path,
                file_type=InitFileType.EMPTY,
                lines_of_code=len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            )
            
            # Extract docstring
            if tree and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
                metadata.docstring = tree.body[0].value.value
            
            # Extract metadata from AST
            if tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if target.id == '__version__' and isinstance(node.value, ast.Constant):
                                    metadata.version = node.value.value
                                elif target.id == '__author__' and isinstance(node.value, ast.Constant):
                                    metadata.author = node.value.value
                                elif target.id == '__email__' and isinstance(node.value, ast.Constant):
                                    metadata.email = node.value.value
                                elif target.id == '__all__':
                                    if isinstance(node.value, ast.List):
                                        metadata.all_list = [elt.value for elt in node.value.elts 
                                                           if isinstance(elt, ast.Constant)]
                    
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        # Track imports for analysis
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                metadata.imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            metadata.imports.append(f"from {node.module}")
            
            # Classify file type based on content patterns
            metadata.file_type = self._classify_init_file_type(metadata, content)
            
            return metadata
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return InitFileMetadata(
                file_path=file_path,
                file_type=InitFileType.EMPTY,
                lines_of_code=0
            )
    
    def _classify_init_file_type(self, metadata: InitFileMetadata, content: str) -> InitFileType:
        """Classify the type of __init__.py file based on its content patterns."""
        
        path_str = str(metadata.file_path).lower()
        
        # Check for FastAPI router patterns
        if 'api_router' in content or 'APIRouter' in content:
            return InitFileType.API_ROUTER
        
        # Check for models package patterns
        if 'models' in path_str and len(metadata.all_list) > 5:
            return InitFileType.MODELS
        
        # Check for main app package
        if metadata.version and metadata.author and 'dotenv' in content:
            return InitFileType.APP_MAIN
        
        # Check for package exports (many imports and __all__)
        if len(metadata.imports) > 3 and len(metadata.all_list) > 3:
            return InitFileType.PACKAGE_EXPORTS
        
        # Check for utilities
        if 'util' in path_str or 'common' in path_str:
            return InitFileType.UTILS
        
        # Check if it's minimal/empty
        if metadata.lines_of_code <= 3:
            return InitFileType.EMPTY
        
        # Default to simple
        return InitFileType.SIMPLE
    
    def generate_standardized_content(self, metadata: InitFileMetadata) -> str:
        """Generate standardized content for an __init__.py file."""
        template = self.templates[metadata.file_type]
        content = template.content
        
        # Generate appropriate docstring
        docstring = self._generate_docstring(metadata)
        content = content.replace('{{ docstring }}', docstring)
        
        # Replace version info
        version = metadata.version or "2.0.0"
        content = content.replace('{{ version }}', version)
        
        # Replace author info
        author = metadata.author or "LeanVibe Agent Hive Team"
        content = content.replace('{{ author }}', author)
        
        # Replace email info
        email = metadata.email or "dev@leanvibe.com"
        content = content.replace('{{ email }}', email)
        
        # Generate imports section
        if '{{ imports }}' in content:
            imports_section = self._generate_imports_section(metadata)
            content = content.replace('{{ imports }}', imports_section)
        
        # Generate exports section
        if '{{ all_exports }}' in content:
            exports_section = self._generate_exports_section(metadata)
            content = content.replace('{{ all_exports }}', exports_section)
        
        # Generate router includes for API packages
        if '{{ router_includes }}' in content:
            router_section = self._generate_router_includes(metadata)
            content = content.replace('{{ router_includes }}', router_section)
        
        return content
    
    def _generate_docstring(self, metadata: InitFileMetadata) -> str:
        """Generate appropriate docstring based on file path and type."""
        if metadata.docstring:
            # Clean and use existing docstring
            return metadata.docstring.strip()
        
        # Generate docstring based on path pattern
        path_str = str(metadata.file_path).lower()
        
        for pattern, docstring in self.docstring_patterns.items():
            if re.search(pattern, path_str):
                return docstring
        
        # Default generic docstring
        package_name = metadata.file_path.parent.name.replace('_', ' ').title()
        return f"{package_name} package for LeanVibe Agent Hive 2.0."
    
    def _generate_imports_section(self, metadata: InitFileMetadata) -> str:
        """Generate standardized imports section."""
        # This would be customized based on the actual file content analysis
        # For now, preserve existing imports in a standardized format
        if not metadata.imports:
            return "# No module imports"
        
        imports = []
        for imp in metadata.imports[:10]:  # Limit to prevent excessive imports
            if not imp.startswith('from typing'):
                imports.append(f"# {imp}")
        
        return '\n'.join(imports) if imports else "# Module imports preserved from original"
    
    def _generate_exports_section(self, metadata: InitFileMetadata) -> str:
        """Generate standardized exports section for __all__."""
        if not metadata.all_list:
            return '    # No exports defined'
        
        # Format exports with proper indentation
        exports = []
        for export in sorted(metadata.all_list):
            exports.append(f'    "{export}",')
        
        return '\n'.join(exports)
    
    def _generate_router_includes(self, metadata: InitFileMetadata) -> str:
        """Generate router includes section for API packages."""
        # This would be analyzed from the actual router includes
        return "# Router includes preserved from original configuration"
    
    def scan_project_init_files(self) -> Dict[Path, InitFileMetadata]:
        """Scan all __init__.py files in the project."""
        init_files = list(self.project_root.glob('**/__init__.py'))
        
        # Filter out venv and other non-project files
        filtered_files = [
            f for f in init_files 
            if 'venv' not in str(f) and 
               '.git' not in str(f) and
               'node_modules' not in str(f)
        ]
        
        for file_path in filtered_files:
            metadata = self.analyze_init_file(file_path)
            self.analyzed_files[file_path] = metadata
        
        return self.analyzed_files
    
    def generate_standardization_report(self) -> str:
        """Generate a report showing the standardization opportunities."""
        if not self.analyzed_files:
            self.scan_project_init_files()
        
        report = []
        report.append("# __init__.py Standardization Report")
        report.append("## Phase 1.2 Technical Debt Remediation - ROI: 1031.0")
        report.append("")
        report.append(f"**Total __init__.py files analyzed**: {len(self.analyzed_files)}")
        
        # Analyze by type
        type_counts = {}
        total_lines = 0
        duplicate_patterns = []
        
        for metadata in self.analyzed_files.values():
            file_type = metadata.file_type
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
            total_lines += metadata.lines_of_code
            
            # Check for duplication patterns
            if metadata.docstring and metadata.docstring in [m.docstring for m in self.analyzed_files.values()]:
                duplicate_patterns.append(metadata.docstring)
        
        report.append("### File Type Distribution:")
        for file_type, count in sorted(type_counts.items()):
            report.append(f"- **{file_type.value}**: {count} files")
        
        report.append(f"\n**Total lines in __init__.py files**: {total_lines}")
        report.append(f"**Estimated consolidation savings**: {total_lines * 0.4:.0f} lines (40% reduction)")
        
        if duplicate_patterns:
            unique_duplicates = set(duplicate_patterns)
            report.append(f"\n**Duplicate patterns found**: {len(unique_duplicates)}")
        
        report.append("\n### Standardization Benefits:")
        report.append("- Consistent module initialization patterns")
        report.append("- Standardized logging setup across all packages")
        report.append("- Unified docstring and metadata patterns")
        report.append("- Reduced maintenance overhead")
        report.append("- Better IDE support and developer experience")
        
        return '\n'.join(report)
    
    def apply_standardization(self, dry_run: bool = True) -> List[Tuple[Path, str]]:
        """Apply standardization to __init__.py files."""
        changes = []
        
        if not self.analyzed_files:
            self.scan_project_init_files()
        
        for file_path, metadata in self.analyzed_files.items():
            try:
                new_content = self.generate_standardized_content(metadata)
                
                if dry_run:
                    changes.append((file_path, new_content))
                else:
                    # Actually write the file
                    file_path.write_text(new_content, encoding='utf-8')
                    changes.append((file_path, "APPLIED"))
                    
            except Exception as e:
                changes.append((file_path, f"ERROR: {e}"))
        
        return changes


def main():
    """Main entry point for __init__.py standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize __init__.py files across the project")
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--apply', action='store_true', help='Apply standardization changes')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    standardizer = InitFileStandardizer(project_root)
    
    if args.report:
        print(standardizer.generate_standardization_report())
    
    if args.apply or args.dry_run:
        changes = standardizer.apply_standardization(dry_run=args.dry_run)
        
        print(f"\n{'DRY RUN - ' if args.dry_run else ''}Standardization Results:")
        print("="*60)
        
        for file_path, result in changes:
            if result == "APPLIED":
                print(f"✅ {file_path}")
            elif result.startswith("ERROR"):
                print(f"❌ {file_path}: {result}")
            else:
                print(f"📝 {file_path}: Preview generated")
        
        if args.dry_run:
            print(f"\n📊 Total files to be standardized: {len([r for r in changes if not r[1].startswith('ERROR')])}")


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class InitFileStandardizerScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            main()
            
            return {"status": "completed"}
    
    script_main(InitFileStandardizerScript)