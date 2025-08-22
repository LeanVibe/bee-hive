#!/usr/bin/env python3
"""
Link Validator for Living Documentation System

Validates all internal and external links in documentation files.
Handles broken link detection, redirect management, and reference integrity.
"""

import os
import re
import asyncio
import aiohttp
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time


@dataclass
class LinkValidationResult:
    """Result of link validation"""
    file_path: str
    link_text: str
    link_url: str
    link_type: str  # 'internal', 'external', 'anchor'
    status: str  # 'valid', 'broken', 'warning', 'redirect', 'error'
    status_code: Optional[int] = None
    redirect_url: Optional[str] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    line_number: Optional[int] = None


class LinkValidator:
    """Advanced link validation with caching and rate limiting"""
    
    def __init__(self, base_path: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.base_path = Path(base_path)
        self.results: List[LinkValidationResult] = []
        self.external_cache: Dict[str, Tuple[str, int, float]] = {}  # url -> (status, code, timestamp)
        self.cache_duration = 3600  # 1 hour cache for external links
        self.rate_limit_delay = 1.0  # seconds between external requests
        self.request_timeout = 10  # seconds
        
    async def validate_all_links(self, file_paths: List[Path]) -> List[LinkValidationResult]:
        """Validate all links in provided documentation files"""
        self.results.clear()
        
        # First pass: extract all links
        all_links = {}
        for file_path in file_paths:
            file_links = self._extract_links_from_file(file_path)
            all_links[file_path] = file_links
            
        print(f"🔍 Found {sum(len(links) for links in all_links.values())} total links")
        
        # Second pass: validate links
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout)
        ) as session:
            for file_path, links in all_links.items():
                print(f"🔗 Validating links in: {file_path.relative_to(self.base_path)}")
                await self._validate_file_links(session, file_path, links)
                
        return self.results
    
    def _extract_links_from_file(self, file_path: Path) -> List[Tuple[str, str, int]]:
        """Extract all links from a markdown file with line numbers"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            links = []
            
            for line_num, line in enumerate(lines, 1):
                # Find markdown links [text](url)
                markdown_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', line)
                for link_text, link_url in markdown_links:
                    links.append((link_text.strip(), link_url.strip(), line_num))
                
                # Find reference-style links [text]: url
                ref_links = re.findall(r'^\s*\[([^\]]+)\]:\s*(.+)$', line)
                for link_text, link_url in ref_links:
                    links.append((link_text.strip(), link_url.strip(), line_num))
                
                # Find bare URLs (http/https)
                bare_urls = re.findall(r'https?://[^\s<>\[\]()]+', line)
                for url in bare_urls:
                    # Skip URLs that are already part of markdown links
                    if f']({url})' not in line and f': {url}' not in line:
                        links.append(('', url.strip(), line_num))
            
            return links
            
        except Exception as e:
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text="",
                link_url="",
                link_type="error",
                status="error",
                error_message=f"Failed to extract links: {e}"
            ))
            return []
    
    async def _validate_file_links(self, session: aiohttp.ClientSession, file_path: Path, links: List[Tuple[str, str, int]]):
        """Validate all links in a single file"""
        for link_text, link_url, line_number in links:
            await self._validate_single_link(session, file_path, link_text, link_url, line_number)
            
            # Rate limiting for external requests
            if self._is_external_url(link_url):
                await asyncio.sleep(self.rate_limit_delay)
    
    async def _validate_single_link(self, session: aiohttp.ClientSession, file_path: Path, 
                                  link_text: str, link_url: str, line_number: int):
        """Validate a single link"""
        
        # Skip empty or malformed URLs
        if not link_url or link_url in ['#', 'javascript:void(0)', 'mailto:']:
            return
        
        # Determine link type and validate accordingly
        if self._is_anchor_link(link_url):
            await self._validate_anchor_link(file_path, link_text, link_url, line_number)
        elif self._is_internal_link(link_url):
            await self._validate_internal_link(file_path, link_text, link_url, line_number)
        elif self._is_external_url(link_url):
            await self._validate_external_link(session, file_path, link_text, link_url, line_number)
        else:
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text=link_text,
                link_url=link_url,
                link_type="unknown",
                status="warning",
                line_number=line_number,
                error_message="Unknown link type"
            ))
    
    def _is_anchor_link(self, url: str) -> bool:
        """Check if URL is an anchor link"""
        return url.startswith('#')
    
    def _is_internal_link(self, url: str) -> bool:
        """Check if URL is an internal link"""
        return not url.startswith(('http://', 'https://', 'ftp://', 'mailto:', 'tel:')) and not url.startswith('#')
    
    def _is_external_url(self, url: str) -> bool:
        """Check if URL is an external URL"""
        return url.startswith(('http://', 'https://'))
    
    async def _validate_anchor_link(self, file_path: Path, link_text: str, link_url: str, line_number: int):
        """Validate anchor links within the same document"""
        try:
            # Read the file to check if anchor exists
            content = file_path.read_text(encoding='utf-8')
            anchor = link_url[1:]  # Remove the #
            
            # Look for heading that matches the anchor
            # Convert heading text to anchor format (lowercase, spaces to hyphens, remove special chars)
            headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            valid_anchors = []
            
            for heading in headings:
                # Convert heading to anchor format
                anchor_text = re.sub(r'[^\w\s-]', '', heading.strip())
                anchor_text = re.sub(r'\s+', '-', anchor_text.lower())
                valid_anchors.append(anchor_text)
            
            if anchor.lower() in [a.lower() for a in valid_anchors]:
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="anchor",
                    status="valid",
                    line_number=line_number
                ))
            else:
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="anchor",
                    status="warning",
                    line_number=line_number,
                    error_message=f"Anchor '{anchor}' not found in document"
                ))
                
        except Exception as e:
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text=link_text,
                link_url=link_url,
                link_type="anchor",
                status="error",
                line_number=line_number,
                error_message=f"Error validating anchor: {e}"
            ))
    
    async def _validate_internal_link(self, file_path: Path, link_text: str, link_url: str, line_number: int):
        """Validate internal file links"""
        try:
            # Resolve the target path
            if link_url.startswith('/'):
                # Absolute path from project root
                target_path = self.base_path / link_url.lstrip('/')
            else:
                # Relative path from current file
                target_path = (file_path.parent / link_url).resolve()
            
            # Handle anchor fragments in internal links
            clean_url = link_url.split('#')[0]
            if clean_url:
                if link_url.startswith('/'):
                    target_path = self.base_path / clean_url.lstrip('/')
                else:
                    target_path = (file_path.parent / clean_url).resolve()
            
            if target_path.exists():
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="internal",
                    status="valid",
                    line_number=line_number
                ))
            else:
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="internal",
                    status="broken",
                    line_number=line_number,
                    error_message=f"File not found: {target_path}"
                ))
                
        except Exception as e:
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text=link_text,
                link_url=link_url,
                link_type="internal",
                status="error",
                line_number=line_number,
                error_message=f"Error validating internal link: {e}"
            ))
    
    async def _validate_external_link(self, session: aiohttp.ClientSession, file_path: Path, 
                                    link_text: str, link_url: str, line_number: int):
        """Validate external URL with caching and rate limiting"""
        
        # Check cache first
        cache_key = hashlib.md5(link_url.encode()).hexdigest()
        if cache_key in self.external_cache:
            status, status_code, timestamp = self.external_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="external",
                    status=status,
                    status_code=status_code,
                    line_number=line_number,
                    error_message="(cached)" if status != "valid" else None
                ))
                return
        
        # Make HTTP request
        start_time = time.time()
        try:
            async with session.head(link_url, allow_redirects=True) as response:
                response_time = time.time() - start_time
                
                if response.status < 400:
                    status = "valid"
                    if response.url != link_url:
                        status = "redirect"
                        redirect_url = str(response.url)
                    else:
                        redirect_url = None
                else:
                    status = "broken"
                    redirect_url = None
                
                # Cache the result
                self.external_cache[cache_key] = (status, response.status, time.time())
                
                self.results.append(LinkValidationResult(
                    file_path=str(file_path),
                    link_text=link_text,
                    link_url=link_url,
                    link_type="external",
                    status=status,
                    status_code=response.status,
                    redirect_url=redirect_url,
                    response_time=response_time,
                    line_number=line_number
                ))
                
        except asyncio.TimeoutError:
            self.external_cache[cache_key] = ("warning", 0, time.time())
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text=link_text,
                link_url=link_url,
                link_type="external",
                status="warning",
                line_number=line_number,
                error_message="Request timeout"
            ))
            
        except Exception as e:
            self.external_cache[cache_key] = ("error", 0, time.time())
            self.results.append(LinkValidationResult(
                file_path=str(file_path),
                link_text=link_text,
                link_url=link_url,
                link_type="external",
                status="error",
                line_number=line_number,
                error_message=str(e)
            ))
    
    def get_broken_links(self) -> List[LinkValidationResult]:
        """Get all broken links"""
        return [r for r in self.results if r.status == 'broken']
    
    def get_external_redirects(self) -> List[LinkValidationResult]:
        """Get all external links that redirect"""
        return [r for r in self.results if r.status == 'redirect']
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get link validation summary statistics"""
        if not self.results:
            return {}
        
        by_status = {}
        by_type = {}
        response_times = []
        
        for result in self.results:
            # Count by status
            if result.status not in by_status:
                by_status[result.status] = 0
            by_status[result.status] += 1
            
            # Count by type
            if result.link_type not in by_type:
                by_type[result.link_type] = 0
            by_type[result.link_type] += 1
            
            # Collect response times
            if result.response_time:
                response_times.append(result.response_time)
        
        total_links = len(self.results)
        broken_count = by_status.get('broken', 0)
        validity_rate = ((total_links - broken_count) / total_links * 100) if total_links > 0 else 0
        
        stats = {
            'total_links_validated': total_links,
            'by_status': by_status,
            'by_type': by_type,
            'broken_links_count': broken_count,
            'validity_rate': round(validity_rate, 1)
        }
        
        if response_times:
            stats['external_response_time'] = {
                'avg': round(sum(response_times) / len(response_times), 2),
                'min': round(min(response_times), 2),
                'max': round(max(response_times), 2)
            }
        
        return stats
    
    def generate_broken_links_report(self) -> str:
        """Generate a report of all broken links for fixing"""
        broken_links = self.get_broken_links()
        
        if not broken_links:
            return "🎉 No broken links found!"
        
        report = f"❌ BROKEN LINKS REPORT ({len(broken_links)} broken links)\n"
        report += "=" * 60 + "\n\n"
        
        # Group by file
        by_file = {}
        for link in broken_links:
            file_name = Path(link.file_path).name
            if file_name not in by_file:
                by_file[file_name] = []
            by_file[file_name].append(link)
        
        for file_name, file_links in by_file.items():
            report += f"📄 {file_name}:\n"
            for link in file_links:
                line_info = f" (line {link.line_number})" if link.line_number else ""
                report += f"  ❌ [{link.link_text}]({link.link_url}){line_info}\n"
                if link.error_message:
                    report += f"     Error: {link.error_message}\n"
            report += "\n"
        
        return report


async def main():
    """Run link validation"""
    print("🔗 Link Validator for Living Documentation")
    print("=" * 60)
    
    validator = LinkValidator()
    
    # Find markdown files
    md_files = []
    for pattern in ["*.md", "**/*.md"]:
        for file in validator.base_path.glob(pattern):
            if not any(exclude in str(file) for exclude in ['node_modules', 'venv', 'archive', '.git']):
                md_files.append(file)
    
    print(f"Found {len(md_files)} documentation files")
    
    # Validate all links
    results = await validator.validate_all_links(md_files)
    
    # Display results
    stats = validator.get_summary_statistics()
    
    print(f"\n📊 LINK VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Links validated: {stats.get('total_links_validated', 0)}")
    print(f"Validity rate: {stats.get('validity_rate', 0)}%")
    
    if 'by_status' in stats:
        for status, count in stats['by_status'].items():
            emoji = {
                'valid': '✅', 
                'broken': '❌', 
                'warning': '⚠️', 
                'redirect': '↪️', 
                'error': '💥'
            }.get(status, '📋')
            print(f"{emoji} {status}: {count}")
    
    if 'by_type' in stats:
        print(f"\n📈 BY LINK TYPE:")
        for link_type, count in stats['by_type'].items():
            print(f"  {link_type}: {count}")
    
    # Show external link performance
    if 'external_response_time' in stats:
        perf = stats['external_response_time']
        print(f"\n⏱️  EXTERNAL LINK PERFORMANCE:")
        print(f"  Average response time: {perf['avg']}s")
        print(f"  Range: {perf['min']}s - {perf['max']}s")
    
    # Show sample broken links
    broken_links = validator.get_broken_links()
    if broken_links:
        print(f"\n❌ SAMPLE BROKEN LINKS:")
        for broken in broken_links[:3]:
            file_name = Path(broken.file_path).name
            line_info = f":{broken.line_number}" if broken.line_number else ""
            print(f"  {file_name}{line_info}: {broken.link_url}")
    
    return len(broken_links)


if __name__ == "__main__":
    import sys
    broken_count = asyncio.run(main())
    sys.exit(1 if broken_count > 0 else 0)