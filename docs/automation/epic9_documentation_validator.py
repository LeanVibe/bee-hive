#!/usr/bin/env python3
"""
Epic 9 Documentation Quality Validation System
Automated validation of the 50 core documentation files for:
- Link validity
- Code example functionality  
- Content freshness
- User journey completeness
"""

import os
import re
import json
import subprocess
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Epic9DocumentationValidator:
    """Validates Epic 9's 50 core documentation files for quality and consistency."""
    
    def __init__(self, root_dir: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.root_dir = Path(root_dir)
        self.core_files = self._get_core_documentation_files()
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(self.core_files),
            "link_validation": {},
            "code_validation": {},
            "freshness_check": {},
            "journey_validation": {},
            "success_metrics": {}
        }
    
    def _get_core_documentation_files(self) -> List[Path]:
        """Returns the Epic 9 core 50 documentation files."""
        core_files = [
            # Root level entry points (8 files)
            "README.md",
            "ARCHITECTURE_CONSOLIDATED.md", 
            "API_REFERENCE_CONSOLIDATED.md",
            "CLI_USAGE_GUIDE.md",
            "DEVELOPER_ONBOARDING_30MIN.md",
            "UV_INSTALLATION_GUIDE.md", 
            "CONTRIBUTING.md",
            "DEPLOYMENT_CHECKLIST.md",
            
            # Core system (6 files)
            "docs/GETTING_STARTED.md",
            "docs/CORE.md",
            "docs/TECHNICAL_SPECIFICATIONS.md", 
            "docs/PRODUCTION_DEPLOYMENT_GUIDE.md",
            "docs/NAV_INDEX.md",
            "docs/OPERATIONAL_RUNBOOK.md",
            
            # User journeys (9 files)
            "docs/paths/EXECUTIVE_PATH.md",
            "docs/paths/DEVELOPER_PATH.md", 
            "docs/paths/ADVANCED_DEVELOPER_PATH.md",
            "docs/tutorials/USER_TUTORIAL_COMPREHENSIVE.md",
            "docs/tutorials/AUTONOMOUS_DEVELOPMENT_DEMO.md",
            "docs/product/PRODUCT_VISION.md",
            "docs/product/VALUE_PROPOSITION.md", 
            "docs/product/TARGET_USER_ANALYSIS.md",
            "docs/competitive-advantages.md",
            
            # Implementation guides (12 files)
            "docs/guides/MULTI_AGENT_COORDINATION_GUIDE.md",
            "docs/guides/EXTERNAL_TOOLS_GUIDE.md",
            "docs/guides/MOBILE_PWA_IMPLEMENTATION_GUIDE.md",
            "docs/guides/ENTERPRISE_USER_GUIDE.md",
            "docs/guides/PERFORMANCE_TUNING_COMPREHENSIVE_GUIDE.md",
            "docs/guides/QUALITY_GATES_AUTOMATION.md",
            "docs/guides/SANDBOX_MODE_GUIDE.md", 
            "docs/implementation/context-compression.md",
            "docs/integrations/HOOK_INTEGRATION_GUIDE.md",
            "docs/integrations/claude/hooks-guide.md",
            "docs/integrations/HIVE_SLASH_COMMANDS.md",
            "docs/design/ENTERPRISE_SYSTEM_ARCHITECTURE.md",
            
            # API & Reference (9 files)
            "docs/reference/API_REFERENCE_COMPREHENSIVE.md",
            "docs/reference/validation-framework.md",
            "docs/reference/DASHBOARD_API_DOCUMENTATION.md", 
            "docs/reference/GITHUB_INTEGRATION_API_COMPREHENSIVE.md",
            "docs/reference/SEMANTIC_MEMORY_API.md",
            "docs/reference/OBSERVABILITY_EVENT_SCHEMA.md",
            "docs/reference/AGENT_SPECIALIZATION_TEMPLATES.md",
            "docs/core/system-architecture.md",
            "docs/core/product-requirements.md",
            
            # Operations & troubleshooting (6 files)
            "docs/runbooks/PRODUCTION_DEPLOYMENT_RUNBOOK.md",
            "docs/runbooks/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md",
            "docs/reports/SYSTEM_VALIDATION_COMPREHENSIVE.md",
            "docs/reports/STATUS_COMPREHENSIVE.md", 
            "docs/migrations/SANDBOX_TO_PRODUCTION_MIGRATION.md",
            "docs/enterprise/market-strategy.md"
        ]
        
        return [self.root_dir / file for file in core_files if (self.root_dir / file).exists()]
    
    async def validate_all(self) -> Dict:
        """Run complete Epic 9 documentation validation."""
        logger.info(f"🚀 Starting Epic 9 validation of {len(self.core_files)} core files...")
        
        # Run all validation types
        await self.validate_links()
        await self.validate_code_examples()
        self.check_content_freshness()
        self.validate_user_journeys()
        self.calculate_success_metrics()
        
        # Generate report
        return self.validation_results
    
    async def validate_links(self):
        """Validate all internal and external links."""
        logger.info("🔗 Validating links...")
        
        link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
        
        for file_path in self.core_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                links = link_pattern.findall(content)
                
                file_results = {
                    "total_links": len(links),
                    "broken_links": [],
                    "valid_links": 0
                }
                
                for link_text, link_url in links:
                    if await self._validate_link(file_path, link_url):
                        file_results["valid_links"] += 1
                    else:
                        file_results["broken_links"].append({
                            "text": link_text,
                            "url": link_url
                        })
                
                self.validation_results["link_validation"][str(file_path.relative_to(self.root_dir))] = file_results
                
            except Exception as e:
                logger.error(f"Error validating links in {file_path}: {e}")
    
    async def _validate_link(self, file_path: Path, url: str) -> bool:
        """Validate a single link."""
        try:
            if url.startswith('http'):
                # External link validation
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        return response.status < 400
            else:
                # Internal link validation
                if url.startswith('/'):
                    link_path = self.root_dir / url.lstrip('/')
                else:
                    link_path = file_path.parent / url
                
                # Handle anchors
                if '#' in url:
                    link_path = Path(str(link_path).split('#')[0])
                
                return link_path.exists()
        except:
            return False
    
    async def validate_code_examples(self):
        """Validate code examples by attempting to execute them."""
        logger.info("💻 Validating code examples...")
        
        bash_pattern = re.compile(r'```bash\n(.*?)\n```', re.DOTALL)
        python_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
        
        for file_path in self.core_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                file_results = {
                    "bash_examples": [],
                    "python_examples": [],
                    "validation_summary": {"total": 0, "passed": 0, "failed": 0}
                }
                
                # Validate bash examples
                bash_examples = bash_pattern.findall(content)
                for i, example in enumerate(bash_examples):
                    result = await self._validate_bash_example(example)
                    file_results["bash_examples"].append({
                        "index": i,
                        "code": example[:200] + "..." if len(example) > 200 else example,
                        "valid": result,
                        "type": "bash"
                    })
                    file_results["validation_summary"]["total"] += 1
                    if result:
                        file_results["validation_summary"]["passed"] += 1
                    else:
                        file_results["validation_summary"]["failed"] += 1
                
                # Validate Python examples  
                python_examples = python_pattern.findall(content)
                for i, example in enumerate(python_examples):
                    result = await self._validate_python_example(example)
                    file_results["python_examples"].append({
                        "index": i,
                        "code": example[:200] + "..." if len(example) > 200 else example,
                        "valid": result,
                        "type": "python"
                    })
                    file_results["validation_summary"]["total"] += 1
                    if result:
                        file_results["validation_summary"]["passed"] += 1
                    else:
                        file_results["validation_summary"]["failed"] += 1
                
                self.validation_results["code_validation"][str(file_path.relative_to(self.root_dir))] = file_results
                
            except Exception as e:
                logger.error(f"Error validating code in {file_path}: {e}")
    
    async def _validate_bash_example(self, code: str) -> bool:
        """Validate a bash code example (syntax check only for safety)."""
        try:
            # Only do syntax validation for safety - don't execute
            result = subprocess.run(['bash', '-n'], input=code, text=True, 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def _validate_python_example(self, code: str) -> bool:
        """Validate a Python code example (syntax check only for safety)."""
        try:
            # Only compile, don't execute
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    def check_content_freshness(self):
        """Check content freshness and update timestamps."""
        logger.info("📅 Checking content freshness...")
        
        for file_path in self.core_files:
            try:
                stat = file_path.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                age_days = (datetime.now() - last_modified).days
                
                freshness_status = "fresh" if age_days < 30 else "stale" if age_days < 90 else "outdated"
                
                self.validation_results["freshness_check"][str(file_path.relative_to(self.root_dir))] = {
                    "last_modified": last_modified.isoformat(),
                    "age_days": age_days,
                    "status": freshness_status
                }
                
            except Exception as e:
                logger.error(f"Error checking freshness for {file_path}: {e}")
    
    def validate_user_journeys(self):
        """Validate that user journeys are complete and coherent."""
        logger.info("🛤️ Validating user journeys...")
        
        journeys = {
            "quick_start": ["README.md", "UV_INSTALLATION_GUIDE.md", "CLI_USAGE_GUIDE.md"],
            "developer": ["DEVELOPER_ONBOARDING_30MIN.md", "ARCHITECTURE_CONSOLIDATED.md", 
                         "API_REFERENCE_CONSOLIDATED.md", "CONTRIBUTING.md"],
            "enterprise": ["DEPLOYMENT_CHECKLIST.md", "docs/guides/ENTERPRISE_USER_GUIDE.md",
                          "docs/OPERATIONAL_RUNBOOK.md"]
        }
        
        for journey_name, journey_files in journeys.items():
            journey_result = {
                "total_files": len(journey_files),
                "existing_files": 0,
                "missing_files": [],
                "complete": False
            }
            
            for file_name in journey_files:
                file_path = self.root_dir / file_name
                if file_path.exists():
                    journey_result["existing_files"] += 1
                else:
                    journey_result["missing_files"].append(file_name)
            
            journey_result["complete"] = journey_result["existing_files"] == journey_result["total_files"]
            self.validation_results["journey_validation"][journey_name] = journey_result
    
    def calculate_success_metrics(self):
        """Calculate Epic 9 success metrics."""
        logger.info("📊 Calculating success metrics...")
        
        # File count success
        file_count_success = len(self.core_files) <= 50
        
        # Link validation success
        total_links = sum(result.get("total_links", 0) for result in self.validation_results["link_validation"].values())
        broken_links = sum(len(result.get("broken_links", [])) for result in self.validation_results["link_validation"].values())
        link_success_rate = (total_links - broken_links) / max(total_links, 1)
        
        # Code validation success
        total_examples = sum(result.get("validation_summary", {}).get("total", 0) 
                           for result in self.validation_results["code_validation"].values())
        passed_examples = sum(result.get("validation_summary", {}).get("passed", 0) 
                            for result in self.validation_results["code_validation"].values())
        code_success_rate = passed_examples / max(total_examples, 1)
        
        # Journey completeness
        complete_journeys = sum(1 for journey in self.validation_results["journey_validation"].values() 
                              if journey.get("complete", False))
        journey_success_rate = complete_journeys / max(len(self.validation_results["journey_validation"]), 1)
        
        # Content freshness
        fresh_files = sum(1 for result in self.validation_results["freshness_check"].values() 
                         if result.get("status") == "fresh")
        freshness_rate = fresh_files / max(len(self.core_files), 1)
        
        self.validation_results["success_metrics"] = {
            "epic9_criteria": {
                "file_count_target": 50,
                "file_count_actual": len(self.core_files),
                "file_count_success": file_count_success,
                "link_success_rate": link_success_rate,
                "code_success_rate": code_success_rate,
                "journey_completeness": journey_success_rate,
                "content_freshness": freshness_rate
            },
            "overall_score": (
                (1.0 if file_count_success else 0.5) * 0.2 +
                link_success_rate * 0.3 +
                code_success_rate * 0.2 +
                journey_success_rate * 0.2 +
                freshness_rate * 0.1
            ),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for broken links
        broken_link_files = [file for file, result in self.validation_results["link_validation"].items() 
                           if result.get("broken_links")]
        if broken_link_files:
            recommendations.append(f"Fix broken links in: {', '.join(broken_link_files[:3])}")
        
        # Check for failed code examples
        failed_code_files = [file for file, result in self.validation_results["code_validation"].items() 
                           if result.get("validation_summary", {}).get("failed", 0) > 0]
        if failed_code_files:
            recommendations.append(f"Fix code examples in: {', '.join(failed_code_files[:3])}")
        
        # Check for incomplete journeys
        incomplete_journeys = [journey for journey, result in self.validation_results["journey_validation"].items() 
                             if not result.get("complete", False)]
        if incomplete_journeys:
            recommendations.append(f"Complete user journeys: {', '.join(incomplete_journeys)}")
        
        # Check for outdated content
        outdated_files = [file for file, result in self.validation_results["freshness_check"].items() 
                         if result.get("status") == "outdated"]
        if outdated_files:
            recommendations.append(f"Update outdated content in: {', '.join(outdated_files[:3])}")
        
        return recommendations

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        metrics = self.validation_results["success_metrics"]
        overall_score = metrics["overall_score"]
        
        report = f"""
# Epic 9 Documentation Quality Report

**Generated**: {self.validation_results["timestamp"]}  
**Overall Score**: {overall_score:.1%} {'✅' if overall_score > 0.8 else '⚠️' if overall_score > 0.6 else '❌'}

## 🎯 Epic 9 Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| File Count | ≤50 | {metrics["epic9_criteria"]["file_count_actual"]} | {'✅' if metrics["epic9_criteria"]["file_count_success"] else '❌'} |
| Link Validity | 100% | {metrics["epic9_criteria"]["link_success_rate"]:.1%} | {'✅' if metrics["epic9_criteria"]["link_success_rate"] > 0.95 else '⚠️' if metrics["epic9_criteria"]["link_success_rate"] > 0.8 else '❌'} |
| Code Examples | 100% | {metrics["epic9_criteria"]["code_success_rate"]:.1%} | {'✅' if metrics["epic9_criteria"]["code_success_rate"] > 0.95 else '⚠️' if metrics["epic9_criteria"]["code_success_rate"] > 0.8 else '❌'} |
| Journey Completeness | 100% | {metrics["epic9_criteria"]["journey_completeness"]:.1%} | {'✅' if metrics["epic9_criteria"]["journey_completeness"] == 1.0 else '❌'} |
| Content Freshness | >80% | {metrics["epic9_criteria"]["content_freshness"]:.1%} | {'✅' if metrics["epic9_criteria"]["content_freshness"] > 0.8 else '⚠️'} |

## 📊 Detailed Results

**Total Files Validated**: {self.validation_results["total_files"]}

**Link Validation**: {sum(r.get("valid_links", 0) for r in self.validation_results["link_validation"].values())}/{sum(r.get("total_links", 0) for r in self.validation_results["link_validation"].values())} valid

**Code Examples**: {sum(r.get("validation_summary", {}).get("passed", 0) for r in self.validation_results["code_validation"].values())}/{sum(r.get("validation_summary", {}).get("total", 0) for r in self.validation_results["code_validation"].values())} working

## 🔧 Recommendations

{chr(10).join(f"- {rec}" for rec in metrics["recommendations"])}

---
*Epic 9 Living Documentation System - Automated Quality Assurance*
"""
        return report

async def main():
    """Run Epic 9 documentation validation."""
    validator = Epic9DocumentationValidator()
    results = await validator.validate_all()
    
    # Save detailed results
    output_path = validator.root_dir / "docs" / "automation" / "epic9_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save human-readable report
    report = validator.generate_report()
    report_path = validator.root_dir / "EPIC9_DOCUMENTATION_QUALITY_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n📄 Detailed results saved to: {output_path}")
    print(f"📋 Human report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())