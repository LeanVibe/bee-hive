"""
Multi-language Code Analyzer for LeanVibe Agent Hive 2.0

Language-specific AST parsing and code analysis engine supporting Python, JavaScript,
TypeScript and extensible for additional languages. Provides dependency extraction,
complexity metrics, and code structure analysis.
"""

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union

import structlog

from .models import (
    AnalysisConfiguration, DependencyResult, FileAnalysisResult,
    ComplexityMetrics, CodeStructure
)
from .utils import FileUtils

logger = structlog.get_logger()


class CodeAnalyzer:
    """
    Multi-language code analysis engine with AST parsing capabilities.
    
    Supports Python, JavaScript, TypeScript initially with extensible
    architecture for additional languages.
    """
    
    # Language detection mapping
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.pyi': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
        '.rst': 'restructuredtext',
        '.txt': 'text',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp'
    }
    
    # Import patterns for different languages
    IMPORT_PATTERNS = {
        'python': [
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+',
            r'^\s*from\s+\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+',
            r'^\s*from\s+\.\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+'
        ],
        'javascript': [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'export\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        ],
        'typescript': [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+type\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'export\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'/// <reference path=[\'"]([^\'"]+)[\'"] />'
        ]
    }
    
    def __init__(self, config: Optional[AnalysisConfiguration] = None):
        """Initialize CodeAnalyzer with configuration."""
        self.config = config or AnalysisConfiguration()
        self._ast_cache: Dict[str, Any] = {}
        self._dependency_cache: Dict[str, List[DependencyResult]] = {}
        
        # Language-specific analyzers
        self._analyzers = {
            'python': self._analyze_python,
            'javascript': self._analyze_javascript,
            'typescript': self._analyze_typescript,
            'json': self._analyze_json
        }
    
    def detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect programming language from file extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected language name or None if unknown
        """
        # Check extension first
        extension = file_path.suffix.lower()
        if extension in self.LANGUAGE_EXTENSIONS:
            return self.LANGUAGE_EXTENSIONS[extension]
        
        # Check shebang for scripts
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!'):
                    if 'python' in first_line:
                        return 'python'
                    elif 'node' in first_line or 'js' in first_line:
                        return 'javascript'
                    elif 'bash' in first_line or 'sh' in first_line:
                        return 'shell'
        except Exception:
            pass
        
        # Check file name patterns
        filename = file_path.name.lower()
        if filename in ['makefile', 'dockerfile']:
            return filename
        elif filename.startswith('.'):
            # Configuration files
            if filename in ['.gitignore', '.dockerignore']:
                return 'gitignore'
            elif filename.endswith('rc'):
                return 'config'
        
        return None
    
    async def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse file and extract AST-based analysis data.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Analysis data dictionary or None if parsing failed
        """
        language = self.detect_language(file_path)
        if not language or language not in self._analyzers:
            logger.debug("Unsupported language for AST parsing", 
                        file_path=str(file_path), language=language)
            return None
        
        # Check cache
        file_hash = self._get_file_hash(file_path)
        cache_key = f"{file_path}:{file_hash}"
        if cache_key in self._ast_cache:
            logger.debug("AST cache hit", file_path=str(file_path))
            return self._ast_cache[cache_key]
        
        try:
            # Analyze file with language-specific analyzer
            analyzer = self._analyzers[language]
            result = await analyzer(file_path)
            
            if result:
                # Cache result
                self._ast_cache[cache_key] = result
                logger.debug("File parsed successfully", 
                           file_path=str(file_path), language=language)
            
            return result
            
        except Exception as e:
            logger.error("Failed to parse file", 
                        file_path=str(file_path), 
                        language=language, 
                        error=str(e))
            return None
    
    async def extract_dependencies(self, file_path: Path) -> List[DependencyResult]:
        """
        Extract dependencies from file using language-specific patterns.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of DependencyResult objects
        """
        language = self.detect_language(file_path)
        if not language or language not in self.IMPORT_PATTERNS:
            return []
        
        # Check cache
        file_hash = self._get_file_hash(file_path)
        cache_key = f"deps:{file_path}:{file_hash}"
        if cache_key in self._dependency_cache:
            logger.debug("Dependency cache hit", file_path=str(file_path))
            return self._dependency_cache[cache_key]
        
        try:
            dependencies = []
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract dependencies using patterns and AST
            if language == 'python':
                dependencies = await self._extract_python_dependencies(file_path, content)
            elif language in ['javascript', 'typescript']:
                dependencies = await self._extract_js_dependencies(file_path, content, language)
            else:
                # Fallback to regex patterns
                dependencies = self._extract_dependencies_regex(file_path, content, language)
            
            # Cache result
            self._dependency_cache[cache_key] = dependencies
            
            logger.debug("Dependencies extracted", 
                        file_path=str(file_path), 
                        count=len(dependencies))
            
            return dependencies
            
        except Exception as e:
            logger.error("Failed to extract dependencies", 
                        file_path=str(file_path), error=str(e))
            return []
    
    # ================== PYTHON ANALYSIS ==================
    
    async def _analyze_python(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract information
            analysis = {
                'language': 'python',
                'line_count': len(content.splitlines()),
                'character_count': len(content),
                'functions': self._extract_python_functions(tree),
                'classes': self._extract_python_classes(tree),
                'imports': self._extract_python_imports(tree),
                'complexity': self._calculate_python_complexity(tree),
                'docstrings': self._extract_python_docstrings(tree),
                'ast_nodes': self._count_ast_nodes(tree)
            }
            
            return analysis
            
        except SyntaxError as e:
            logger.warning("Python syntax error", 
                          file_path=str(file_path), 
                          error=str(e))
            return {
                'language': 'python',
                'parse_error': str(e),
                'line_count': len(open(file_path, 'r', encoding='utf-8').read().splitlines())
            }
        except Exception as e:
            logger.error("Python analysis failed", 
                        file_path=str(file_path), 
                        error=str(e))
            return None
    
    def _extract_python_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions from Python AST."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'is_async': False,
                    'docstring': ast.get_docstring(node)
                }
                functions.append(func_info)
            elif isinstance(node, ast.AsyncFunctionDef):
                func_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'is_async': True,
                    'docstring': ast.get_docstring(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_python_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions from Python AST."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'bases': [ast.unparse(base) for base in node.bases],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'methods': [],
                    'docstring': ast.get_docstring(node)
                }
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = {
                            'name': item.name,
                            'line_number': item.lineno,
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'args': [arg.arg for arg in item.args.args],
                            'decorators': [ast.unparse(dec) for dec in item.decorator_list]
                        }
                        class_info['methods'].append(method_info)
                
                classes.append(class_info)
        
        return classes
    
    def _extract_python_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements from Python AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = {
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line_number': node.lineno
                    }
                    imports.append(import_info)
            elif isinstance(node, ast.ImportFrom):
                import_info = {
                    'type': 'from_import',
                    'module': node.module,
                    'level': node.level,
                    'names': [{'name': alias.name, 'alias': alias.asname} for alias in node.names],
                    'line_number': node.lineno
                }
                imports.append(import_info)
        
        return imports
    
    def _calculate_python_complexity(self, tree: ast.AST) -> ComplexityMetrics:
        """Calculate cyclomatic complexity for Python code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                               ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each boolean operator adds complexity
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Comprehensions add complexity
                complexity += 1
                for generator in node.generators:
                    complexity += len(generator.ifs)
        
        return ComplexityMetrics(
            cyclomatic_complexity=complexity,
            cognitive_complexity=complexity  # Simplified - could be more sophisticated
        )
    
    def _extract_python_docstrings(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract docstrings from Python AST."""
        docstrings = []
        
        # Module docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            docstrings.append({
                'type': 'module',
                'content': module_docstring,
                'line_number': 1
            })
        
        # Function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append({
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'content': docstring,
                        'line_number': node.lineno
                    })
        
        return docstrings
    
    async def _extract_python_dependencies(
        self, 
        file_path: Path, 
        content: str
    ) -> List[DependencyResult]:
        """Extract Python dependencies using AST analysis."""
        dependencies = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep = DependencyResult(
                            source_file_path=str(file_path),
                            target_name=alias.name,
                            dependency_type='import',
                            line_number=node.lineno,
                            source_text=f"import {alias.name}",
                            is_external=self._is_external_python_module(alias.name),
                            confidence_score=1.0
                        )
                        dependencies.append(dep)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dep = DependencyResult(
                            source_file_path=str(file_path),
                            target_name=node.module,
                            dependency_type='import',
                            line_number=node.lineno,
                            source_text=f"from {node.module} import ...",
                            is_external=self._is_external_python_module(node.module),
                            confidence_score=1.0
                        )
                        dependencies.append(dep)
        
        except SyntaxError:
            # Fallback to regex for files with syntax errors
            dependencies = self._extract_dependencies_regex(file_path, content, 'python')
        
        return dependencies
    
    def _is_external_python_module(self, module_name: str) -> bool:
        """Check if Python module is external (not standard library or local)."""
        # Standard library modules (simplified list)
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 'math', 'random', 're',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'urllib', 'http', 'email', 'logging', 'unittest', 'sqlite3',
            'csv', 'xml', 'html', 'subprocess', 'threading', 'multiprocessing',
            'asyncio', 'concurrent', 'queue', 'socket', 'ssl', 'hashlib'
        }
        
        # Check if it's a relative import
        if module_name.startswith('.'):
            return False
        
        # Extract top-level module name
        top_level = module_name.split('.')[0]
        
        # Check if it's standard library
        if top_level in stdlib_modules:
            return False
        
        # Assume external if not in standard library
        return True
    
    # ================== JAVASCRIPT/TYPESCRIPT ANALYSIS ==================
    
    async def _analyze_javascript(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JavaScript file using regex patterns (AST parsing would require external tools)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'language': 'javascript',
                'line_count': len(content.splitlines()),
                'character_count': len(content),
                'functions': self._extract_js_functions(content),
                'classes': self._extract_js_classes(content),
                'imports': self._extract_js_imports(content),
                'exports': self._extract_js_exports(content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error("JavaScript analysis failed", 
                        file_path=str(file_path), 
                        error=str(e))
            return None
    
    async def _analyze_typescript(self, file_path: Path) -> Dict[str, Any]:
        """Analyze TypeScript file using regex patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'language': 'typescript',
                'line_count': len(content.splitlines()),
                'character_count': len(content),
                'functions': self._extract_js_functions(content),
                'classes': self._extract_js_classes(content),
                'interfaces': self._extract_ts_interfaces(content),
                'types': self._extract_ts_types(content),
                'imports': self._extract_js_imports(content),
                'exports': self._extract_js_exports(content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error("TypeScript analysis failed", 
                        file_path=str(file_path), 
                        error=str(e))
            return None
    
    def _extract_js_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript function definitions using regex."""
        functions = []
        
        # Function patterns
        patterns = [
            r'function\s+(\w+)\s*\([^)]*\)',  # function name()
            r'(\w+)\s*:\s*function\s*\([^)]*\)',  # name: function()
            r'(\w+)\s*=\s*function\s*\([^)]*\)',  # name = function()
            r'(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',  # name = () => {}
            r'async\s+function\s+(\w+)\s*\([^)]*\)',  # async function name()
            r'(\w+)\s*=\s*async\s*\([^)]*\)\s*=>\s*{',  # name = async () => {}
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                functions.append({
                    'name': match.group(1),
                    'line_number': line_number,
                    'is_async': 'async' in match.group(0)
                })
        
        return functions
    
    def _extract_js_classes(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript class definitions using regex."""
        classes = []
        
        pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_number = content[:match.start()].count('\n') + 1
            classes.append({
                'name': match.group(1),
                'line_number': line_number,
                'extends': match.group(2) if match.group(2) else None
            })
        
        return classes
    
    def _extract_ts_interfaces(self, content: str) -> List[Dict[str, Any]]:
        """Extract TypeScript interface definitions using regex."""
        interfaces = []
        
        pattern = r'interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{'
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_number = content[:match.start()].count('\n') + 1
            interfaces.append({
                'name': match.group(1),
                'line_number': line_number,
                'extends': match.group(2).strip() if match.group(2) else None
            })
        
        return interfaces
    
    def _extract_ts_types(self, content: str) -> List[Dict[str, Any]]:
        """Extract TypeScript type definitions using regex."""
        types = []
        
        pattern = r'type\s+(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_number = content[:match.start()].count('\n') + 1
            types.append({
                'name': match.group(1),
                'line_number': line_number,
                'definition': match.group(2).strip()
            })
        
        return types
    
    def _extract_js_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript import statements using regex."""
        imports = []
        
        patterns = [
            r'import\s+([^}]+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # import ... from '...'
            r'import\s+[\'"]([^\'"]+)[\'"]',  # import '...'
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',  # import('...')
            r'const\s+\w+\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',  # require
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                if len(match.groups()) == 2:
                    imports.append({
                        'type': 'import',
                        'names': match.group(1).strip(),
                        'module': match.group(2),
                        'line_number': line_number
                    })
                else:
                    imports.append({
                        'type': 'require' if 'require' in match.group(0) else 'import',
                        'module': match.group(1),
                        'line_number': line_number
                    })
        
        return imports
    
    def _extract_js_exports(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript export statements using regex."""
        exports = []
        
        patterns = [
            r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)',
            r'export\s+default\s+(\w+)',
            r'export\s*{\s*([^}]+)\s*}',
            r'module\.exports\s*=\s*(\w+)'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                exports.append({
                    'name': match.group(1),
                    'line_number': line_number,
                    'is_default': 'default' in match.group(0)
                })
        
        return exports
    
    async def _extract_js_dependencies(
        self, 
        file_path: Path, 
        content: str, 
        language: str
    ) -> List[DependencyResult]:
        """Extract JavaScript/TypeScript dependencies."""
        dependencies = []
        
        patterns = self.IMPORT_PATTERNS[language]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                target_name = match.group(1)
                
                dep = DependencyResult(
                    source_file_path=str(file_path),
                    target_name=target_name,
                    dependency_type='import',
                    line_number=line_number,
                    source_text=match.group(0).strip(),
                    is_external=self._is_external_js_module(target_name),
                    confidence_score=0.9  # Regex-based, slightly lower confidence
                )
                dependencies.append(dep)
        
        return dependencies
    
    def _is_external_js_module(self, module_name: str) -> bool:
        """Check if JavaScript/TypeScript module is external."""
        # Relative imports
        if module_name.startswith('.'):
            return False
        
        # Node.js built-in modules
        builtin_modules = {
            'fs', 'path', 'os', 'crypto', 'util', 'events', 'stream',
            'http', 'https', 'url', 'querystring', 'zlib', 'buffer',
            'child_process', 'cluster', 'worker_threads', 'readline'
        }
        
        if module_name in builtin_modules:
            return False
        
        # Assume external if not relative or built-in
        return True
    
    # ================== JSON ANALYSIS ==================
    
    async def _analyze_json(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON file structure."""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
            
            analysis = {
                'language': 'json',
                'line_count': len(content.splitlines()),
                'character_count': len(content),
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'structure_type': type(data).__name__,
                'is_package_json': file_path.name == 'package.json',
                'is_tsconfig': file_path.name.startswith('tsconfig'),
                'is_config': True
            }
            
            # Extract package.json dependencies
            if file_path.name == 'package.json' and isinstance(data, dict):
                deps = {}
                for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
                    if dep_type in data:
                        deps[dep_type] = list(data[dep_type].keys())
                analysis['package_dependencies'] = deps
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error", 
                          file_path=str(file_path), 
                          error=str(e))
            return {
                'language': 'json',
                'parse_error': str(e),
                'line_count': len(open(file_path, 'r', encoding='utf-8').read().splitlines())
            }
        except Exception as e:
            logger.error("JSON analysis failed", 
                        file_path=str(file_path), 
                        error=str(e))
            return None
    
    # ================== HELPER METHODS ==================
    
    def _extract_dependencies_regex(
        self, 
        file_path: Path, 
        content: str, 
        language: str
    ) -> List[DependencyResult]:
        """Extract dependencies using regex patterns as fallback."""
        dependencies = []
        
        if language not in self.IMPORT_PATTERNS:
            return dependencies
        
        patterns = self.IMPORT_PATTERNS[language]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                target_name = match.group(1)
                
                dep = DependencyResult(
                    source_file_path=str(file_path),
                    target_name=target_name,
                    dependency_type='import',
                    line_number=line_number,
                    source_text=match.group(0).strip(),
                    is_external=True,  # Conservative assumption
                    confidence_score=0.8  # Lower confidence for regex
                )
                dependencies.append(dep)
        
        return dependencies
    
    def _count_ast_nodes(self, tree: ast.AST) -> Dict[str, int]:
        """Count different types of AST nodes."""
        node_counts = {}
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        return node_counts
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file content hash for caching."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return str(file_path.stat().st_mtime)
    
    def clear_cache(self) -> None:
        """Clear AST and dependency caches."""
        self._ast_cache.clear()
        self._dependency_cache.clear()
        logger.info("Analyzer caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'ast_cache_size': len(self._ast_cache),
            'dependency_cache_size': len(self._dependency_cache)
        }