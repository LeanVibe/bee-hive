"""
Static Analysis Validator - Enhanced Safety for Refactoring
==========================================================

Implements Gemini CLI recommendation for static analysis integration
to catch errors before test execution in large-scale refactoring.

Features:
- MyPy type checking integration
- Ruff linting and formatting validation
- Radon complexity analysis
- Combined analysis reporting
- Integration with refactoring workflow

Usage:
    # Validate single file
    python static_analyzer.py --file app/services/example.py
    
    # Validate entire batch
    python static_analyzer.py --batch batch_001.json
    
    # Pre-refactoring validation
    python static_analyzer.py --validate-batch batch_001.json --pre-refactor
"""

import asyncio
import subprocess
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import structlog
import tempfile

logger = structlog.get_logger(__name__)

@dataclass
class AnalysisResult:
    """Result of static analysis on a file."""
    analyzer: str
    file_path: Path
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class CombinedAnalysisResult:
    """Combined results from all static analyzers."""
    file_path: Path
    overall_success: bool
    analyzer_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        return sum(len(result.errors) for result in self.analyzer_results.values())
    
    @property
    def warning_count(self) -> int:
        return sum(len(result.warnings) for result in self.analyzer_results.values())

class MyPyAnalyzer:
    """MyPy type checking integration."""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Run MyPy type checking on file."""
        import time
        start_time = time.time()
        
        try:
            # Run mypy with appropriate configuration
            cmd = [
                "python", "-m", "mypy",
                str(file_path),
                "--ignore-missing-imports",
                "--show-error-codes",
                "--no-error-summary"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            success = result.returncode == 0
            errors = []
            warnings = []
            
            if stdout:
                output_lines = stdout.decode().strip().split('\n')
                for line in output_lines:
                    if line.strip():
                        if 'error:' in line:
                            errors.append(line.strip())
                        elif 'warning:' in line:
                            warnings.append(line.strip())
            
            if stderr:
                stderr_content = stderr.decode().strip()
                if stderr_content:
                    errors.append(f"MyPy stderr: {stderr_content}")
                    
            return AnalysisResult(
                analyzer="mypy",
                file_path=file_path,
                success=success,
                errors=errors,
                warnings=warnings,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"MyPy analysis failed for {file_path}: {e}")
            return AnalysisResult(
                analyzer="mypy",
                file_path=file_path,
                success=False,
                errors=[f"MyPy execution failed: {e}"],
                execution_time=time.time() - start_time
            )

class RuffAnalyzer:
    """Ruff linting and formatting validation."""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Run Ruff linting on file."""
        import time
        start_time = time.time()
        
        try:
            # Run ruff check
            cmd = [
                "python", "-m", "ruff", "check",
                str(file_path),
                "--output-format=json"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            errors = []
            warnings = []
            
            if stdout:
                try:
                    ruff_output = json.loads(stdout.decode())
                    for issue in ruff_output:
                        message = f"Line {issue.get('location', {}).get('row', '?')}: {issue.get('message', 'Unknown issue')} ({issue.get('code', 'UNKNOWN')})"
                        
                        # Categorize by severity
                        if issue.get('severity') == 'error':
                            errors.append(message)
                        else:
                            warnings.append(message)
                            
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    output_lines = stdout.decode().strip().split('\n')
                    for line in output_lines:
                        if line.strip():
                            warnings.append(line.strip())
            
            if stderr:
                stderr_content = stderr.decode().strip()
                if stderr_content and "warning" not in stderr_content.lower():
                    errors.append(f"Ruff stderr: {stderr_content}")
                    
            # Success if no errors (warnings are acceptable)
            success = len(errors) == 0
            
            return AnalysisResult(
                analyzer="ruff",
                file_path=file_path,
                success=success,
                errors=errors,
                warnings=warnings,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Ruff analysis failed for {file_path}: {e}")
            return AnalysisResult(
                analyzer="ruff",
                file_path=file_path,
                success=False,
                errors=[f"Ruff execution failed: {e}"],
                execution_time=time.time() - start_time
            )

class RadonAnalyzer:
    """Radon complexity analysis."""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Run Radon complexity analysis on file."""
        import time
        start_time = time.time()
        
        try:
            # Run radon for cyclomatic complexity
            cmd = [
                "python", "-m", "radon", "cc",
                str(file_path),
                "--json"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            errors = []
            warnings = []
            metrics = {}
            
            if stdout:
                try:
                    radon_output = json.loads(stdout.decode())
                    file_results = radon_output.get(str(file_path), [])
                    
                    total_complexity = 0
                    function_count = 0
                    high_complexity_functions = []
                    
                    for item in file_results:
                        complexity = item.get('complexity', 0)
                        total_complexity += complexity
                        function_count += 1
                        
                        # Flag high complexity (>10 is considered complex)
                        if complexity > 10:
                            high_complexity_functions.append({
                                'name': item.get('name', 'unknown'),
                                'complexity': complexity,
                                'line': item.get('lineno', 0)
                            })
                            warnings.append(f"High complexity function '{item.get('name')}' (CC: {complexity}) at line {item.get('lineno', '?')}")
                    
                    avg_complexity = total_complexity / function_count if function_count > 0 else 0
                    
                    metrics = {
                        'total_complexity': total_complexity,
                        'function_count': function_count,
                        'average_complexity': avg_complexity,
                        'high_complexity_functions': high_complexity_functions
                    }
                    
                except json.JSONDecodeError:
                    errors.append("Failed to parse Radon JSON output")
                    
            if stderr:
                stderr_content = stderr.decode().strip()
                if stderr_content:
                    warnings.append(f"Radon stderr: {stderr_content}")
            
            # Success if analysis completed (complexity warnings are informational)
            success = len(errors) == 0
            
            return AnalysisResult(
                analyzer="radon",
                file_path=file_path,
                success=success,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Radon analysis failed for {file_path}: {e}")
            return AnalysisResult(
                analyzer="radon",
                file_path=file_path,
                success=False,
                errors=[f"Radon execution failed: {e}"],
                execution_time=time.time() - start_time
            )

class StaticAnalysisValidator:
    """
    Comprehensive static analysis validator implementing Gemini CLI recommendations.
    
    Integrates MyPy, Ruff, and Radon analysis to catch errors before test execution
    in large-scale refactoring operations.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        self.analyzers = {
            'mypy': MyPyAnalyzer(),
            'ruff': RuffAnalyzer(),
            'radon': RadonAnalyzer()
        }
        
    async def analyze_file(self, file_path: Path, analyzers: Optional[List[str]] = None) -> CombinedAnalysisResult:
        """Run comprehensive static analysis on a single file."""
        if analyzers is None:
            analyzers = list(self.analyzers.keys())
            
        self.logger.info(f"Running static analysis on {file_path}", analyzers=analyzers)
        
        results = {}
        tasks = []
        
        # Run analyzers concurrently
        for analyzer_name in analyzers:
            if analyzer_name in self.analyzers:
                analyzer = self.analyzers[analyzer_name]
                task = asyncio.create_task(analyzer.analyze_file(file_path))
                tasks.append((analyzer_name, task))
        
        # Collect results
        for analyzer_name, task in tasks:
            try:
                result = await task
                results[analyzer_name] = result
            except Exception as e:
                self.logger.error(f"Analyzer {analyzer_name} failed: {e}")
                results[analyzer_name] = AnalysisResult(
                    analyzer=analyzer_name,
                    file_path=file_path,
                    success=False,
                    errors=[f"Analyzer failed: {e}"]
                )
        
        # Determine overall success
        overall_success = all(result.success for result in results.values())
        
        combined_result = CombinedAnalysisResult(
            file_path=file_path,
            overall_success=overall_success,
            analyzer_results=results
        )
        
        self.logger.info(
            f"Static analysis complete for {file_path}",
            success=overall_success,
            errors=combined_result.error_count,
            warnings=combined_result.warning_count
        )
        
        return combined_result
        
    async def analyze_batch(self, batch_file: Path) -> Dict[str, CombinedAnalysisResult]:
        """Analyze all files in a refactoring batch."""
        try:
            with open(batch_file) as f:
                batch_data = json.load(f)
                
            file_paths = [Path(plan['file_path']) for plan in batch_data['plans']]
            
            self.logger.info(f"Analyzing batch {batch_data['batch_id']}", file_count=len(file_paths))
            
            results = {}
            
            # Analyze files concurrently (with semaphore to limit concurrent processes)
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent analyses
            
            async def analyze_with_semaphore(file_path: Path) -> Tuple[Path, CombinedAnalysisResult]:
                async with semaphore:
                    result = await self.analyze_file(file_path)
                    return file_path, result
            
            tasks = [analyze_with_semaphore(fp) for fp in file_paths]
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item in completed_results:
                if isinstance(item, Exception):
                    self.logger.error(f"Batch analysis error: {item}")
                else:
                    file_path, result = item
                    results[str(file_path)] = result
                    
            # Summary statistics
            total_files = len(results)
            successful_files = sum(1 for r in results.values() if r.overall_success)
            total_errors = sum(r.error_count for r in results.values())
            total_warnings = sum(r.warning_count for r in results.values())
            
            self.logger.info(
                f"Batch analysis complete",
                batch_id=batch_data['batch_id'],
                total_files=total_files,
                successful_files=successful_files,
                success_rate=f"{successful_files/total_files:.1%}",
                total_errors=total_errors,
                total_warnings=total_warnings
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze batch {batch_file}: {e}")
            return {}
            
    def generate_report(self, results: Dict[str, CombinedAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        total_files = len(results)
        successful_files = sum(1 for r in results.values() if r.overall_success)
        failed_files = [str(fp) for fp, r in results.items() if not r.overall_success]
        
        # Aggregate metrics by analyzer
        analyzer_stats = {}
        for result in results.values():
            for analyzer_name, analyzer_result in result.analyzer_results.items():
                if analyzer_name not in analyzer_stats:
                    analyzer_stats[analyzer_name] = {
                        'total_files': 0,
                        'successful_files': 0,
                        'total_errors': 0,
                        'total_warnings': 0,
                        'total_execution_time': 0.0
                    }
                
                stats = analyzer_stats[analyzer_name]
                stats['total_files'] += 1
                if analyzer_result.success:
                    stats['successful_files'] += 1
                stats['total_errors'] += len(analyzer_result.errors)
                stats['total_warnings'] += len(analyzer_result.warnings)
                stats['total_execution_time'] += analyzer_result.execution_time
        
        # Calculate success rates
        for stats in analyzer_stats.values():
            stats['success_rate'] = stats['successful_files'] / stats['total_files'] if stats['total_files'] > 0 else 0.0
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_files'] if stats['total_files'] > 0 else 0.0
        
        return {
            'summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': len(failed_files),
                'overall_success_rate': successful_files / total_files if total_files > 0 else 0.0
            },
            'failed_files': failed_files,
            'analyzer_statistics': analyzer_stats
        }

# CLI Interface
async def main():
    """Command-line interface for static analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Static analysis validator for refactoring")
    parser.add_argument('--file', type=Path, help='Analyze single file')
    parser.add_argument('--batch', type=Path, help='Analyze batch file')
    parser.add_argument('--analyzers', nargs='+', choices=['mypy', 'ruff', 'radon'], 
                       help='Specific analyzers to run')
    parser.add_argument('--report', type=Path, help='Save report to file')
    parser.add_argument('--pre-refactor', action='store_true', help='Pre-refactoring validation mode')
    
    args = parser.parse_args()
    
    validator = StaticAnalysisValidator()
    
    if args.file:
        # Analyze single file
        result = await validator.analyze_file(args.file, args.analyzers)
        
        print(f"Static Analysis Results for {args.file}")
        print(f"Overall Success: {result.overall_success}")
        print(f"Errors: {result.error_count}")
        print(f"Warnings: {result.warning_count}")
        
        for analyzer_name, analyzer_result in result.analyzer_results.items():
            print(f"\n{analyzer_name.upper()} Results:")
            print(f"  Success: {analyzer_result.success}")
            print(f"  Execution Time: {analyzer_result.execution_time:.2f}s")
            
            if analyzer_result.errors:
                print("  Errors:")
                for error in analyzer_result.errors:
                    print(f"    - {error}")
                    
            if analyzer_result.warnings:
                print("  Warnings:")
                for warning in analyzer_result.warnings:
                    print(f"    - {warning}")
                    
            if analyzer_result.metrics:
                print("  Metrics:")
                for key, value in analyzer_result.metrics.items():
                    print(f"    - {key}: {value}")
                    
    elif args.batch:
        # Analyze batch
        results = await validator.analyze_batch(args.batch)
        report = validator.generate_report(results)
        
        print(f"Batch Analysis Report")
        print(f"Total Files: {report['summary']['total_files']}")
        print(f"Successful Files: {report['summary']['successful_files']}")
        print(f"Success Rate: {report['summary']['overall_success_rate']:.1%}")
        
        if report['failed_files']:
            print(f"\nFailed Files ({len(report['failed_files'])}):")
            for failed_file in report['failed_files'][:10]:  # Show first 10
                print(f"  - {failed_file}")
                
        print(f"\nAnalyzer Statistics:")
        for analyzer, stats in report['analyzer_statistics'].items():
            print(f"  {analyzer.upper()}:")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Avg Execution Time: {stats['avg_execution_time']:.2f}s")
            print(f"    Total Errors: {stats['total_errors']}")
            print(f"    Total Warnings: {stats['total_warnings']}")
        
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {args.report}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())