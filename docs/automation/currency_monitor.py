#!/usr/bin/env python3
"""
Currency Monitor for Living Documentation System

Monitors documentation freshness and content currency against codebase changes.
Tracks documentation age, code-doc synchronization, and automated content updates.
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re


@dataclass
class ContentCurrencyResult:
    """Result of content currency analysis"""
    file_path: str
    currency_type: str  # 'age', 'code_sync', 'api_sync', 'link_health', 'content_quality'
    status: str  # 'fresh', 'stale', 'outdated', 'critical', 'unknown'
    score: float  # 0-100 currency score
    message: str
    last_modified: Optional[str] = None
    related_files: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ContentCurrencyMonitor:
    """Advanced content currency monitoring and analysis"""
    
    def __init__(self, base_path: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.base_path = Path(base_path)
        self.results: List[ContentCurrencyResult] = []
        self.git_available = self._check_git_availability()
        
        # Currency thresholds (days)
        self.thresholds = {
            'fresh': 7,
            'moderate': 30,
            'stale': 90,
            'critical': 180
        }
        
    def analyze_content_currency(self, file_paths: List[Path]) -> List[ContentCurrencyResult]:
        """Analyze content currency for all documentation files"""
        self.results.clear()
        
        print(f"📅 Analyzing content currency for {len(file_paths)} files")
        
        for file_path in file_paths:
            print(f"⏰ Analyzing: {file_path.relative_to(self.base_path)}")
            
            # Multiple currency analysis types
            self._analyze_file_age(file_path)
            self._analyze_code_synchronization(file_path)
            self._analyze_api_documentation_currency(file_path)
            self._analyze_content_quality_indicators(file_path)
            self._analyze_reference_integrity(file_path)
            
        return self.results
    
    def _check_git_availability(self) -> bool:
        """Check if git is available for analysis"""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, cwd=self.base_path)
            return result.returncode == 0
        except:
            return False
    
    def _analyze_file_age(self, file_path: Path):
        """Analyze document age and modification patterns"""
        try:
            stat = file_path.stat()
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            age_days = (datetime.now() - modified_time).days
            
            # Determine freshness status
            if age_days <= self.thresholds['fresh']:
                status = 'fresh'
                score = 100
            elif age_days <= self.thresholds['moderate']:
                status = 'fresh'
                score = 85
            elif age_days <= self.thresholds['stale']:
                status = 'stale'
                score = 60
            elif age_days <= self.thresholds['critical']:
                status = 'outdated'
                score = 30
            else:
                status = 'critical'
                score = 10
            
            # Get git history if available
            git_info = self._get_git_file_history(file_path) if self.git_available else {}
            
            recommendations = []
            if age_days > self.thresholds['moderate']:
                recommendations.append("Review content for accuracy and updates")
            if age_days > self.thresholds['stale']:
                recommendations.append("Check for new features or API changes")
            if age_days > self.thresholds['critical']:
                recommendations.append("Comprehensive review and rewrite needed")
            
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='age',
                status=status,
                score=score,
                message=f"Document age: {age_days} days (last modified: {modified_time.strftime('%Y-%m-%d')})",
                last_modified=modified_time.isoformat(),
                recommendations=recommendations,
                metadata={
                    'age_days': age_days,
                    'git_info': git_info
                }
            ))
            
        except Exception as e:
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='age',
                status='unknown',
                score=0,
                message=f"Error analyzing file age: {e}"
            ))
    
    def _get_git_file_history(self, file_path: Path) -> Dict[str, Any]:
        """Get git history information for a file"""
        try:
            rel_path = file_path.relative_to(self.base_path)
            
            # Get last modification date from git
            result = subprocess.run([
                'git', 'log', '-1', '--format=%ci', '--', str(rel_path)
            ], capture_output=True, text=True, cwd=self.base_path)
            
            if result.returncode == 0 and result.stdout.strip():
                git_modified = result.stdout.strip()
                
                # Get commit count
                result = subprocess.run([
                    'git', 'rev-list', '--count', 'HEAD', '--', str(rel_path)
                ], capture_output=True, text=True, cwd=self.base_path)
                
                commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0
                
                return {
                    'last_git_modified': git_modified,
                    'commit_count': commit_count
                }
            
        except Exception:
            pass
        
        return {}
    
    def _analyze_code_synchronization(self, file_path: Path):
        """Analyze synchronization between documentation and referenced code"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find code file references
            code_references = self._extract_code_references(content)
            
            if not code_references:
                return
            
            sync_score = 100
            out_of_sync_files = []
            recommendations = []
            
            for ref_path in code_references:
                try:
                    if ref_path.startswith('/'):
                        target_file = self.base_path / ref_path.lstrip('/')
                    else:
                        target_file = file_path.parent / ref_path
                    
                    if target_file.exists():
                        # Compare modification times
                        doc_mtime = file_path.stat().st_mtime
                        code_mtime = target_file.stat().st_mtime
                        
                        if code_mtime > doc_mtime:
                            # Code is newer than documentation
                            days_behind = (code_mtime - doc_mtime) / 86400
                            if days_behind > 7:
                                sync_score -= 20
                                out_of_sync_files.append(str(target_file))
                                if days_behind > 30:
                                    sync_score -= 20
                    else:
                        # Referenced file doesn't exist
                        sync_score -= 30
                        out_of_sync_files.append(str(target_file) + " (missing)")
                        
                except Exception:
                    sync_score -= 10
            
            sync_score = max(0, sync_score)
            
            if sync_score < 70:
                status = 'outdated'
                recommendations.append("Update documentation to match current code")
            elif sync_score < 90:
                status = 'stale'
                recommendations.append("Review code references for accuracy")
            else:
                status = 'fresh'
            
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='code_sync',
                status=status,
                score=sync_score,
                message=f"Code synchronization score: {sync_score}%",
                related_files=out_of_sync_files,
                recommendations=recommendations,
                metadata={
                    'referenced_files': code_references,
                    'out_of_sync_count': len(out_of_sync_files)
                }
            ))
            
        except Exception as e:
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='code_sync',
                status='unknown',
                score=0,
                message=f"Error analyzing code synchronization: {e}"
            ))
    
    def _extract_code_references(self, content: str) -> List[str]:
        """Extract references to code files from documentation"""
        references = []
        
        # Find file paths in various formats
        patterns = [
            r'`([^`]+\.(py|js|ts|jsx|tsx|java|cpp|h|cs|rb|go|rs|php))`',
            r'\[([^\]]+\.(py|js|ts|jsx|tsx|java|cpp|h|cs|rb|go|rs|php))\]',
            r'(?:^|\s)([a-zA-Z0-9_/.-]+\.(py|js|ts|jsx|tsx|java|cpp|h|cs|rb|go|rs|php))',
            r'src/([^/\s]+/)*[^/\s]+\.(py|js|ts|jsx|tsx|java|cpp|h|cs|rb|go|rs|php)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    file_path = match[0]
                else:
                    file_path = match
                
                # Clean up the path
                file_path = file_path.strip('`[]()\'\"')
                if file_path and file_path not in references:
                    references.append(file_path)
        
        return references
    
    def _analyze_api_documentation_currency(self, file_path: Path):
        """Analyze API documentation currency against actual API endpoints"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check if this is API documentation
            api_indicators = ['api', 'endpoint', 'curl', 'http://', 'https://', '/api/', 'POST', 'GET', 'PUT', 'DELETE']
            if not any(indicator in content.lower() for indicator in api_indicators):
                return
            
            # Extract API endpoints mentioned in documentation
            api_endpoints = self._extract_api_endpoints(content)
            
            if not api_endpoints:
                return
            
            # Check if endpoints exist in actual code
            existing_endpoints = self._find_existing_endpoints()
            
            currency_score = 100
            missing_endpoints = []
            outdated_examples = []
            recommendations = []
            
            for endpoint in api_endpoints:
                if endpoint not in existing_endpoints:
                    currency_score -= 20
                    missing_endpoints.append(endpoint)
            
            # Check for example responses that might be outdated
            if 'response' in content.lower() and 'json' in content.lower():
                # Heuristic: if there are many hardcoded example values, might be outdated
                hardcoded_patterns = [r'"id":\s*"\w+"', r'"timestamp":\s*"[\d-]+', r'"version":\s*"[\d.]+']
                hardcoded_count = sum(len(re.findall(pattern, content)) for pattern in hardcoded_patterns)
                
                if hardcoded_count > 5:
                    currency_score -= 10
                    recommendations.append("Review example responses for current data formats")
            
            if currency_score < 70:
                status = 'outdated'
                recommendations.append("Update API documentation to match current implementation")
            elif currency_score < 90:
                status = 'stale'
                recommendations.append("Verify API examples and responses")
            else:
                status = 'fresh'
            
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='api_sync',
                status=status,
                score=currency_score,
                message=f"API documentation currency: {currency_score}%",
                recommendations=recommendations,
                metadata={
                    'documented_endpoints': api_endpoints,
                    'missing_endpoints': missing_endpoints,
                    'endpoint_count': len(api_endpoints)
                }
            ))
            
        except Exception as e:
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='api_sync',
                status='unknown',
                score=0,
                message=f"Error analyzing API currency: {e}"
            ))
    
    def _extract_api_endpoints(self, content: str) -> List[str]:
        """Extract API endpoints from documentation content"""
        endpoints = []
        
        # Common API endpoint patterns
        patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[^\s\n]+)',
            r'curl[^`\n]*?(?:https?://[^/\s]+)?(/[^\s`\n]+)',
            r'(?:api|endpoint).*?(/[^\s\n`]+)',
            r'(/api/[^\s\n`]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                endpoint = match.strip()
                if endpoint.startswith('/') and endpoint not in endpoints:
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _find_existing_endpoints(self) -> Set[str]:
        """Find actual API endpoints in the codebase"""
        endpoints = set()
        
        # Look for FastAPI/Flask route definitions
        for pattern in ["**/*.py"]:
            for py_file in self.base_path.glob(pattern):
                if 'venv' in str(py_file) or 'node_modules' in str(py_file):
                    continue
                    
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    # FastAPI patterns
                    fastapi_patterns = [
                        r'@app\.(get|post|put|delete|patch)\([\'"]([^\'\"]+)[\'"]',
                        r'@router\.(get|post|put|delete|patch)\([\'"]([^\'\"]+)[\'"]',
                    ]
                    
                    for pattern in fastapi_patterns:
                        matches = re.findall(pattern, content)
                        for method, endpoint in matches:
                            endpoints.add(endpoint)
                    
                except:
                    continue
        
        return endpoints
    
    def _analyze_content_quality_indicators(self, file_path: Path):
        """Analyze content quality indicators that suggest currency needs"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            quality_score = 100
            issues = []
            recommendations = []
            
            # Check for placeholder content
            placeholders = ['TODO', 'FIXME', 'TBD', 'Coming soon', 'Under construction', 'Work in progress']
            placeholder_count = sum(content.lower().count(placeholder.lower()) for placeholder in placeholders)
            
            if placeholder_count > 0:
                quality_score -= (placeholder_count * 15)
                issues.append(f"Contains {placeholder_count} placeholder(s)")
                recommendations.append("Complete placeholder content")
            
            # Check for broken formatting
            formatting_issues = [
                (r'\[\]\([^)]*\)', "Empty link text"),
                (r'\[([^\]]+)\]\(\)', "Empty link URL"),
                (r'!\[\]\([^)]*\)', "Missing image alt text"),
                (r'#{6,}', "Excessive heading levels"),
            ]
            
            for pattern, issue_name in formatting_issues:
                matches = re.findall(pattern, content)
                if matches:
                    quality_score -= (len(matches) * 5)
                    issues.append(f"{issue_name}: {len(matches)} occurrences")
                    recommendations.append(f"Fix {issue_name.lower()}")
            
            # Check for outdated version references
            version_patterns = [
                r'version\s+[\d.]+',
                r'v[\d.]+',
                r'20\d{2}',  # Years that might be outdated
            ]
            
            current_year = datetime.now().year
            for pattern in version_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if pattern == r'20\d{2}':
                        year = int(match)
                        if year < current_year - 1:  # More than 1 year old
                            quality_score -= 10
                            issues.append(f"References old year: {year}")
                            recommendations.append("Update year references")
            
            quality_score = max(0, quality_score)
            
            if quality_score < 70:
                status = 'outdated'
            elif quality_score < 85:
                status = 'stale'
            else:
                status = 'fresh'
            
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='content_quality',
                status=status,
                score=quality_score,
                message=f"Content quality score: {quality_score}%",
                recommendations=recommendations,
                metadata={
                    'issues': issues,
                    'issue_count': len(issues)
                }
            ))
            
        except Exception as e:
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='content_quality',
                status='unknown',
                score=0,
                message=f"Error analyzing content quality: {e}"
            ))
    
    def _analyze_reference_integrity(self, file_path: Path):
        """Analyze the integrity of references within documentation"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find all referenced documentation files
            doc_references = re.findall(r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)', content)
            
            if not doc_references:
                return
            
            integrity_score = 100
            broken_refs = []
            recommendations = []
            
            for link_text, ref_path in doc_references:
                ref_path = ref_path.split('#')[0]  # Remove anchor
                
                try:
                    if ref_path.startswith('/'):
                        target_file = self.base_path / ref_path.lstrip('/')
                    else:
                        target_file = file_path.parent / ref_path
                    
                    if not target_file.exists():
                        integrity_score -= 25
                        broken_refs.append(f"{link_text} -> {ref_path}")
                        
                except Exception:
                    integrity_score -= 10
            
            integrity_score = max(0, integrity_score)
            
            if broken_refs:
                recommendations.append("Fix broken documentation references")
                recommendations.append("Update file paths after restructuring")
            
            if integrity_score < 70:
                status = 'critical'
            elif integrity_score < 90:
                status = 'stale'
            else:
                status = 'fresh'
            
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='reference_integrity',
                status=status,
                score=integrity_score,
                message=f"Reference integrity: {integrity_score}%",
                recommendations=recommendations,
                metadata={
                    'total_references': len(doc_references),
                    'broken_references': broken_refs,
                    'broken_count': len(broken_refs)
                }
            ))
            
        except Exception as e:
            self.results.append(ContentCurrencyResult(
                file_path=str(file_path),
                currency_type='reference_integrity',
                status='unknown',
                score=0,
                message=f"Error analyzing reference integrity: {e}"
            ))
    
    def get_currency_summary(self) -> Dict[str, Any]:
        """Get summary of content currency analysis"""
        if not self.results:
            return {}
        
        by_status = {}
        by_type = {}
        score_totals = {}
        
        for result in self.results:
            # Count by status
            if result.status not in by_status:
                by_status[result.status] = 0
            by_status[result.status] += 1
            
            # Count by type
            if result.currency_type not in by_type:
                by_type[result.currency_type] = {'count': 0, 'total_score': 0}
            by_type[result.currency_type]['count'] += 1
            by_type[result.currency_type]['total_score'] += result.score
        
        # Calculate average scores by type
        avg_scores = {}
        for curr_type, data in by_type.items():
            avg_scores[curr_type] = round(data['total_score'] / data['count'], 1)
        
        total_results = len(self.results)
        overall_score = sum(r.score for r in self.results) / total_results if total_results > 0 else 0
        
        critical_count = by_status.get('critical', 0)
        outdated_count = by_status.get('outdated', 0)
        
        return {
            'total_analyses': total_results,
            'overall_currency_score': round(overall_score, 1),
            'by_status': by_status,
            'by_type_counts': {k: v['count'] for k, v in by_type.items()},
            'average_scores_by_type': avg_scores,
            'critical_issues': critical_count,
            'outdated_content': outdated_count,
            'health_status': self._determine_overall_health(overall_score, critical_count, outdated_count)
        }
    
    def _determine_overall_health(self, overall_score: float, critical: int, outdated: int) -> str:
        """Determine overall documentation health status"""
        if critical > 0 or overall_score < 50:
            return 'critical'
        elif outdated > 5 or overall_score < 70:
            return 'needs_attention'
        elif overall_score < 85:
            return 'good'
        else:
            return 'excellent'
    
    def generate_currency_report(self) -> str:
        """Generate a detailed currency report"""
        summary = self.get_currency_summary()
        
        if not summary:
            return "No currency analysis results available."
        
        report = "📅 CONTENT CURRENCY ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Overall health
        health = summary['health_status']
        health_emoji = {'excellent': '🟢', 'good': '🟡', 'needs_attention': '🟠', 'critical': '🔴'}
        report += f"Overall Health: {health_emoji.get(health, '⚪')} {health.upper()}\n"
        report += f"Overall Score: {summary['overall_currency_score']}%\n\n"
        
        # Status breakdown
        report += "📊 STATUS BREAKDOWN:\n"
        for status, count in summary['by_status'].items():
            emoji = {'fresh': '✅', 'stale': '⚠️', 'outdated': '❌', 'critical': '🚨'}.get(status, '📋')
            report += f"  {emoji} {status}: {count}\n"
        
        # Type breakdown with scores
        report += f"\n📈 BY ANALYSIS TYPE:\n"
        for curr_type, score in summary['average_scores_by_type'].items():
            count = summary['by_type_counts'][curr_type]
            report += f"  {curr_type}: {count} analyses, avg score {score}%\n"
        
        # Critical issues
        critical_results = [r for r in self.results if r.status == 'critical']
        if critical_results:
            report += f"\n🚨 CRITICAL ISSUES ({len(critical_results)}):\n"
            for result in critical_results[:5]:  # Show top 5
                file_name = Path(result.file_path).name
                report += f"  {file_name}: {result.message}\n"
        
        # Recommendations
        all_recommendations = []
        for result in self.results:
            if result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            # Get most common recommendations
            from collections import Counter
            common_recs = Counter(all_recommendations).most_common(5)
            
            report += f"\n💡 TOP RECOMMENDATIONS:\n"
            for rec, count in common_recs:
                report += f"  • {rec} ({count} files)\n"
        
        return report


def main():
    """Run content currency analysis"""
    print("📅 Content Currency Monitor for Living Documentation")
    print("=" * 60)
    
    monitor = ContentCurrencyMonitor()
    
    # Find markdown files
    md_files = []
    for pattern in ["*.md", "**/*.md"]:
        for file in monitor.base_path.glob(pattern):
            if not any(exclude in str(file) for exclude in ['node_modules', 'venv', 'archive', '.git']):
                md_files.append(file)
    
    print(f"Found {len(md_files)} documentation files")
    
    # Analyze content currency
    results = monitor.analyze_content_currency(md_files)
    
    # Display results
    summary = monitor.get_currency_summary()
    
    print(f"\n📊 CURRENCY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Overall Score: {summary.get('overall_currency_score', 0)}%")
    print(f"Health Status: {summary.get('health_status', 'unknown').upper()}")
    
    if 'by_status' in summary:
        for status, count in summary['by_status'].items():
            emoji = {'fresh': '✅', 'stale': '⚠️', 'outdated': '❌', 'critical': '🚨'}.get(status, '📋')
            print(f"{emoji} {status}: {count}")
    
    # Show critical issues
    critical_issues = summary.get('critical_issues', 0) + summary.get('outdated_content', 0)
    if critical_issues > 0:
        print(f"\n⚠️  {critical_issues} files need immediate attention")
    
    # Generate full report
    full_report = monitor.generate_currency_report()
    
    # Save report
    report_path = monitor.base_path / "docs" / "currency_analysis_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(full_report)
    
    print(f"\n📋 Detailed report saved to: {report_path}")
    
    return critical_issues


if __name__ == "__main__":
    import sys
    critical_count = main()
    sys.exit(1 if critical_count > 5 else 0)  # Exit with error if too many critical issues