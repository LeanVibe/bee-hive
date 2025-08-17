"""
Language-specific parsers for dependency extraction and code analysis.

Provides AST-based parsing for multiple programming languages to extract:
- Import dependencies
- Function definitions
- Class definitions
- Variable assignments
- Function calls
"""

import ast
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

@dataclass
class Dependency:
    """Represents a code dependency relationship"""
    source_file: str
    target_name: str
    target_file: Optional[str] = None
    dependency_type: str = "import"  # import, function_call, class_reference, etc.
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    source_text: Optional[str] = None
    is_external: bool = False
    is_dynamic: bool = False
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CodeStructure:
    """Represents the structure of a code file"""
    file_path: str
    language: str
    functions: List[Dict[str, Any]] = None
    classes: List[Dict[str, Any]] = None
    imports: List[Dict[str, Any]] = None
    variables: List[Dict[str, Any]] = None
    complexity_score: float = 0.0
    line_count: int = 0
    
    def __post_init__(self):
        if self.functions is None:
            self.functions = []
        if self.classes is None:
            self.classes = []
        if self.imports is None:
            self.imports = []
        if self.variables is None:
            self.variables = []

class LanguageParser(ABC):
    """Abstract base class for language-specific parsers"""
    
    @abstractmethod
    def extract_dependencies(self, file_path: Path, content: str) -> List[Dependency]:
        """Extract dependencies from code content"""
        pass
    
    @abstractmethod
    def extract_structure(self, file_path: Path, content: str) -> CodeStructure:
        """Extract code structure information"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> Set[str]:
        """Get file extensions this parser supports"""
        pass

class PythonParser(LanguageParser):
    """Parser for Python code using AST"""
    
    def get_supported_extensions(self) -> Set[str]:
        return {'.py', '.pyw'}
    
    def extract_dependencies(self, file_path: Path, content: str) -> List[Dependency]:
        """Extract import dependencies from Python code"""
        dependencies = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                # Handle regular imports: import module
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep = Dependency(
                            source_file=str(file_path),
                            target_name=alias.name,
                            dependency_type="import",
                            line_number=node.lineno,
                            column_number=getattr(node, 'col_offset', None),
                            source_text=f"import {alias.name}",
                            is_external=self._is_external_module(alias.name),
                            metadata={"alias": alias.asname if alias.asname else alias.name}
                        )
                        dependencies.append(dep)
                
                # Handle from imports: from module import name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        dep = Dependency(
                            source_file=str(file_path),
                            target_name=f"{module}.{alias.name}" if module else alias.name,
                            dependency_type="from_import",
                            line_number=node.lineno,
                            column_number=getattr(node, 'col_offset', None),
                            source_text=f"from {module} import {alias.name}" if module else f"import {alias.name}",
                            is_external=self._is_external_module(module) if module else False,
                            metadata={
                                "module": module,
                                "imported_name": alias.name,
                                "alias": alias.asname if alias.asname else alias.name,
                                "level": node.level  # for relative imports
                            }
                        )
                        dependencies.append(dep)
                
                # Handle function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dep = Dependency(
                            source_file=str(file_path),
                            target_name=node.func.id,
                            dependency_type="function_call",
                            line_number=node.lineno,
                            column_number=getattr(node, 'col_offset', None),
                            source_text=f"{node.func.id}(...)",
                            is_external=False,
                            confidence_score=0.8,  # Function calls are less certain dependencies
                            metadata={"arg_count": len(node.args)}
                        )
                        dependencies.append(dep)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls like obj.method()
                        obj_name = self._get_attribute_base_name(node.func)
                        if obj_name:
                            dep = Dependency(
                                source_file=str(file_path),
                                target_name=f"{obj_name}.{node.func.attr}",
                                dependency_type="method_call",
                                line_number=node.lineno,
                                column_number=getattr(node, 'col_offset', None),
                                source_text=f"{obj_name}.{node.func.attr}(...)",
                                is_external=False,
                                confidence_score=0.7,
                                metadata={"method": node.func.attr, "object": obj_name}
                            )
                            dependencies.append(dep)
                
                # Handle class inheritance
                elif isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            dep = Dependency(
                                source_file=str(file_path),
                                target_name=base.id,
                                dependency_type="inheritance",
                                line_number=node.lineno,
                                column_number=getattr(node, 'col_offset', None),
                                source_text=f"class {node.name}({base.id}):",
                                is_external=False,
                                metadata={"class_name": node.name, "base_class": base.id}
                            )
                            dependencies.append(dep)
        
        except SyntaxError as e:
            # Handle syntax errors gracefully
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return dependencies
    
    def extract_structure(self, file_path: Path, content: str) -> CodeStructure:
        """Extract Python code structure"""
        structure = CodeStructure(
            file_path=str(file_path),
            language="python",
            line_count=len(content.splitlines())
        )
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "line_number": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "docstring": ast.get_docstring(node),
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
                    }
                    structure.functions.append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line_number": node.lineno,
                        "bases": [self._get_base_class_name(base) for base in node.bases],
                        "methods": [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                        "docstring": ast.get_docstring(node),
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
                    }
                    structure.classes.append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Already handled in extract_dependencies, but we can add summary info
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            structure.imports.append({
                                "type": "import",
                                "module": alias.name,
                                "line_number": node.lineno
                            })
                    else:  # ImportFrom
                        module = node.module or ""
                        for alias in node.names:
                            structure.imports.append({
                                "type": "from_import",
                                "module": module,
                                "name": alias.name,
                                "line_number": node.lineno
                            })
            
            # Calculate complexity score (simplified)
            structure.complexity_score = self._calculate_complexity(tree)
            
        except Exception as e:
            print(f"Error extracting structure from {file_path}: {e}")
        
        return structure
    
    def _is_external_module(self, module_name: str) -> bool:
        """Determine if a module is external (not part of the project)"""
        if not module_name:
            return False
        
        # Common external libraries
        external_modules = {
            'os', 'sys', 'json', 're', 'datetime', 'time', 'math', 'random',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'asyncio', 'threading', 'multiprocessing', 'subprocess',
            'requests', 'fastapi', 'sqlalchemy', 'pydantic', 'numpy',
            'pandas', 'matplotlib', 'pytest', 'django', 'flask'
        }
        
        base_module = module_name.split('.')[0]
        return base_module in external_modules
    
    def _get_attribute_base_name(self, node: ast.Attribute) -> Optional[str]:
        """Get the base name of an attribute chain"""
        if isinstance(node.value, ast.Name):
            return node.value.id
        elif isinstance(node.value, ast.Attribute):
            base = self._get_attribute_base_name(node.value)
            return f"{base}.{node.value.attr}" if base else None
        return None
    
    def _get_decorator_name(self, decorator_node) -> str:
        """Extract decorator name from AST node"""
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        elif isinstance(decorator_node, ast.Attribute):
            return f"{self._get_attribute_base_name(decorator_node)}.{decorator_node.attr}"
        else:
            return str(decorator_node)
    
    def _get_base_class_name(self, base_node) -> str:
        """Extract base class name from AST node"""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return f"{self._get_attribute_base_name(base_node)}.{base_node.attr}"
        else:
            return str(base_node)
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate a simple complexity score based on control structures"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity += 0.5
        
        return complexity

class JavaScriptParser(LanguageParser):
    """Parser for JavaScript/TypeScript code using regex patterns"""
    
    def get_supported_extensions(self) -> Set[str]:
        return {'.js', '.jsx', '.ts', '.tsx'}
    
    def extract_dependencies(self, file_path: Path, content: str) -> List[Dependency]:
        """Extract dependencies from JavaScript/TypeScript code"""
        dependencies = []
        lines = content.splitlines()
        
        # Patterns for different import types
        import_patterns = [
            # import defaultExport from "module-name";
            (r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', 'default_import'),
            # import * as name from "module-name";
            (r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']', 'namespace_import'),
            # import { export1, export2 } from "module-name";
            (r'import\s+\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']', 'named_import'),
            # const module = require('module-name');
            (r'(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*["\']([^"\']+)["\']\s*\)', 'require'),
            # require('module-name');
            (r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', 'require_call')
        ]
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            for pattern, dep_type in import_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    if dep_type in ['default_import', 'namespace_import', 'require']:
                        imported_name = match.group(1)
                        module_name = match.group(2)
                    elif dep_type == 'named_import':
                        imported_names = [name.strip() for name in match.group(1).split(',')]
                        module_name = match.group(2)
                        for imported_name in imported_names:
                            dep = Dependency(
                                source_file=str(file_path),
                                target_name=f"{module_name}.{imported_name}",
                                dependency_type=dep_type,
                                line_number=line_num,
                                source_text=line,
                                is_external=self._is_external_js_module(module_name),
                                metadata={"module": module_name, "imported_name": imported_name}
                            )
                            dependencies.append(dep)
                        continue
                    elif dep_type == 'require_call':
                        module_name = match.group(1)
                        imported_name = module_name
                    else:
                        continue
                    
                    dep = Dependency(
                        source_file=str(file_path),
                        target_name=module_name,
                        dependency_type=dep_type,
                        line_number=line_num,
                        source_text=line,
                        is_external=self._is_external_js_module(module_name),
                        metadata={"module": module_name, "imported_name": imported_name}
                    )
                    dependencies.append(dep)
        
        return dependencies
    
    def extract_structure(self, file_path: Path, content: str) -> CodeStructure:
        """Extract JavaScript/TypeScript code structure"""
        structure = CodeStructure(
            file_path=str(file_path),
            language="javascript" if file_path.suffix in ['.js', '.jsx'] else "typescript",
            line_count=len(content.splitlines())
        )
        
        lines = content.splitlines()
        
        # Extract function definitions
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function',
            r'(?:const|let|var)\s+(\w+)\s*=\s*\(.*?\)\s*=>',
            r'(\w+)\s*:\s*(?:async\s+)?function',  # object method
            r'async\s+(\w+)\s*\('
        ]
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    structure.functions.append({
                        "name": func_name,
                        "line_number": line_num,
                        "is_async": 'async' in line,
                        "is_arrow": '=>' in line
                    })
        
        # Extract class definitions
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for line_num, line in enumerate(lines, 1):
            match = re.search(class_pattern, line.strip())
            if match:
                class_name = match.group(1)
                base_class = match.group(2)
                structure.classes.append({
                    "name": class_name,
                    "line_number": line_num,
                    "extends": base_class
                })
        
        return structure
    
    def _is_external_js_module(self, module_name: str) -> bool:
        """Determine if a JavaScript module is external"""
        if not module_name:
            return False
        
        # Relative imports are internal
        if module_name.startswith('.'):
            return False
        
        # Common external modules
        external_modules = {
            'react', 'vue', 'angular', 'lodash', 'jquery', 'express',
            'axios', 'moment', 'underscore', 'bootstrap', 'd3'
        }
        
        # Node.js built-in modules
        node_modules = {
            'fs', 'path', 'http', 'https', 'url', 'crypto', 'os', 'util'
        }
        
        base_module = module_name.split('/')[0]
        return base_module in external_modules or base_module in node_modules

class LanguageParserFactory:
    """Factory for creating language-specific parsers"""
    
    _parsers = {
        'python': PythonParser(),
        'javascript': JavaScriptParser(),
        'typescript': JavaScriptParser(),
    }
    
    @classmethod
    def get_parser(cls, language: str) -> Optional[LanguageParser]:
        """Get parser for specified language"""
        return cls._parsers.get(language.lower())
    
    @classmethod
    def get_parser_for_file(cls, file_path: Path) -> Optional[LanguageParser]:
        """Get parser based on file extension"""
        extension = file_path.suffix.lower()
        
        for parser in cls._parsers.values():
            if extension in parser.get_supported_extensions():
                return parser
        
        return None
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls._parsers.keys())
    
    @classmethod
    def register_parser(cls, language: str, parser: LanguageParser):
        """Register a new language parser"""
        cls._parsers[language.lower()] = parser