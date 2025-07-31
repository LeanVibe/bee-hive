#!/usr/bin/env python3
"""
Phase 2 Milestone Demonstration Runner
======================================

Master script that orchestrates the complete Phase 2 milestone demonstration:

1. Pre-validation checks
2. System preparation
3. Full demonstration execution
4. Results analysis and reporting
5. Cleanup and archival

This script ensures all components are ready before running the demonstration
and provides comprehensive reporting of results.

Usage:
    python run_phase_2_demonstration.py [options]
    
Options:
    --validate-only     Only run validation, skip demonstration
    --demo-only         Skip validation, run demonstration directly
    --performance-mode  Enable high-performance testing
    --verbose           Enable verbose logging
    --output-dir        Directory for output files (default: ./phase2_results)
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

# Configure logging
def setup_logging(verbose: bool = False, output_dir: str = "./phase2_results"):
    """Setup comprehensive logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup loggers
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{output_dir}/phase2_demonstration.log")
        ]
    )
    
    return logging.getLogger(__name__)


class Phase2DemonstrationRunner:
    """Master runner for Phase 2 milestone demonstration."""
    
    def __init__(
        self,
        validate_only: bool = False,
        demo_only: bool = False,
        performance_mode: bool = False,
        verbose: bool = False,
        output_dir: str = "./phase2_results"
    ):
        self.validate_only = validate_only
        self.demo_only = demo_only
        self.performance_mode = performance_mode
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(verbose, str(self.output_dir))
        
        self.results = {
            "execution_id": f"phase2_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.utcnow().isoformat(),
            "configuration": {
                "validate_only": validate_only,
                "demo_only": demo_only,
                "performance_mode": performance_mode,
                "verbose": verbose,
                "output_dir": str(output_dir)
            },
            "stages": {},
            "overall_success": False,
            "summary": {}
        }
    
    async def check_prerequisites(self) -> Dict[str, Any]:
        """Check system prerequisites."""
        self.logger.info("🔍 Checking system prerequisites...")
        
        prerequisites = {
            "python_version": sys.version_info >= (3, 8),
            "redis_available": False,
            "database_accessible": False,
            "required_modules": False
        }
        
        try:
            # Check Redis availability
            import redis
            try:
                r = redis.Redis(host='localhost', port=6379, db=15, socket_connect_timeout=2)
                r.ping()
                prerequisites["redis_available"] = True
                self.logger.info("✅ Redis server is accessible")
            except Exception as e:
                self.logger.warning(f"⚠️ Redis server not accessible: {e}")
            
            # Check database accessibility
            try:
                from app.core.database import get_session
                async with get_session() as session:
                    # Simple query to test connection
                    await session.execute("SELECT 1")
                prerequisites["database_accessible"] = True
                self.logger.info("✅ Database is accessible")
            except Exception as e:
                self.logger.warning(f"⚠️ Database not accessible: {e}")
            
            # Check required modules
            required_modules = [
                'app.core.workflow_engine',
                'app.core.consumer_group_coordinator',
                'app.core.enhanced_redis_streams_manager',
                'app.models.workflow',
                'app.models.task'
            ]
            
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    missing_modules.append(module)
                    self.logger.error(f"❌ Missing module: {module}")
            
            prerequisites["required_modules"] = len(missing_modules) == 0
            
            if missing_modules:
                self.logger.error(f"❌ Missing required modules: {missing_modules}")
            else:
                self.logger.info("✅ All required modules available")
        
        except Exception as e:
            self.logger.error(f"❌ Prerequisites check failed: {e}")
        
        # Overall readiness
        all_good = all(prerequisites.values())
        self.logger.info(f"📋 Prerequisites check: {'PASSED' if all_good else 'FAILED'}")
        
        return {
            "status": "passed" if all_good else "failed",
            "details": prerequisites,
            "ready_for_demonstration": all_good
        }
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run pre-demonstration validation."""
        self.logger.info("🧪 Running Phase 2 validation...")
        
        try:
            # Import and run validator
            from validate_phase_2_demonstration import Phase2Validator
            
            validator = Phase2Validator(
                quick_mode=not self.performance_mode,
                skip_integration=False
            )
            
            validation_results = await validator.run_validation()
            
            # Save validation results
            validation_file = self.output_dir / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            self.logger.info(f"📄 Validation results saved to {validation_file}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {"overall_status": "FAILED", "error": str(e)}
    
    async def run_demonstration(self) -> Dict[str, Any]:
        """Run the full Phase 2 demonstration."""
        self.logger.info("🎬 Running Phase 2 demonstration...")
        
        try:
            # Import and run demonstration
            from phase_2_milestone_demonstration import Phase2MilestoneDemonstration
            
            demonstration = Phase2MilestoneDemonstration(
                verbose=self.verbose,
                performance_mode=self.performance_mode
            )
            
            demo_results = await demonstration.run_demonstration()
            
            # Save demonstration results
            demo_file = self.output_dir / "demonstration_results.json"
            with open(demo_file, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            
            self.logger.info(f"📄 Demonstration results saved to {demo_file}")
            
            return demo_results
            
        except Exception as e:
            self.logger.error(f"❌ Demonstration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive HTML and JSON reports."""
        self.logger.info("📊 Generating comprehensive reports...")
        
        try:
            # Create HTML report
            html_report = self._create_html_report()
            
            html_file = self.output_dir / "phase2_demonstration_report.html"
            with open(html_file, 'w') as f:
                f.write(html_report)
            
            # Create summary JSON
            summary_file = self.output_dir / "phase2_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"📄 HTML report: {html_file}")
            self.logger.info(f"📄 Summary JSON: {summary_file}")
            
            return {
                "html_report": str(html_file),
                "summary_json": str(summary_file),
                "output_directory": str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return {"error": str(e)}
    
    def _create_html_report(self) -> str:
        """Create comprehensive HTML report."""
        
        validation_status = self.results.get("stages", {}).get("validation", {}).get("overall_status", "NOT_RUN")
        demo_status = "PASSED" if self.results.get("stages", {}).get("demonstration", {}).get("success", False) else "FAILED"
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 2 Milestone Demonstration Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 2px solid #e0e0e0; }}
        .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
        .header p {{ color: #7f8c8d; font-size: 1.1em; margin: 10px 0; }}
        
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 0.9em; }}
        .status-passed {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .status-failed {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .status-not-run {{ background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-card .label {{ color: #7f8c8d; font-size: 0.9em; }}
        
        .section {{ margin: 40px 0; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        
        .stage-item {{ background: #f8f9fa; margin: 15px 0; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; }}
        .stage-item.failed {{ border-left-color: #dc3545; }}
        .stage-item h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .stage-item p {{ margin: 5px 0; color: #6c757d; }}
        
        .performance-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .performance-table th, .performance-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .performance-table th {{ background: #f8f9fa; font-weight: 600; color: #495057; }}
        
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; text-align: center; color: #7f8c8d; }}
        
        .target-met {{ color: #28a745; font-weight: bold; }}
        .target-not-met {{ color: #dc3545; font-weight: bold; }}
        
        .code-block {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 3px solid #007bff; font-family: 'Courier New', monospace; font-size: 0.9em; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Phase 2 Milestone Demonstration</h1>
            <p>Multi-step workflow with agent crash recovery via consumer groups</p>
            <p><strong>Execution ID:</strong> {self.results['execution_id']}</p>
            <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Overall Status</h3>
                <div class="value">{'✅' if self.results.get('overall_success', False) else '❌'}</div>
                <div class="label">{'SUCCESS' if self.results.get('overall_success', False) else 'FAILED'}</div>
            </div>
            <div class="metric-card">
                <h3>Validation</h3>
                <div class="value">{'✅' if validation_status == 'PASSED' else '❌' if validation_status == 'FAILED' else '⏭️'}</div>
                <div class="label">{validation_status}</div>
            </div>
            <div class="metric-card">
                <h3>Demonstration</h3>
                <div class="value">{'✅' if demo_status == 'PASSED' else '❌'}</div>
                <div class="label">{demo_status}</div>
            </div>
            <div class="metric-card">
                <h3>Performance Mode</h3>
                <div class="value">{'🚀' if self.performance_mode else '🐌'}</div>
                <div class="label">{'ENABLED' if self.performance_mode else 'DISABLED'}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Execution Summary</h2>
            <div class="stage-item">
                <h3>Configuration</h3>
                <p><strong>Validate Only:</strong> {self.validate_only}</p>
                <p><strong>Demo Only:</strong> {self.demo_only}</p>
                <p><strong>Performance Mode:</strong> {self.performance_mode}</p>
                <p><strong>Verbose Logging:</strong> {self.verbose}</p>
                <p><strong>Output Directory:</strong> {self.output_dir}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🧪 Validation Results</h2>
            {self._render_validation_section()}
        </div>
        
        <div class="section">
            <h2>🎬 Demonstration Results</h2>
            {self._render_demonstration_section()}
        </div>
        
        <div class="section">
            <h2>📊 Performance Analysis</h2>
            {self._render_performance_section()}
        </div>
        
        <div class="section">
            <h2>📁 Output Files</h2>
            <div class="stage-item">
                <h3>Generated Files</h3>
                <ul>
                    <li><code>validation_results.json</code> - Detailed validation results</li>
                    <li><code>demonstration_results.json</code> - Complete demonstration data</li>
                    <li><code>phase2_summary.json</code> - Executive summary</li>
                    <li><code>phase2_demonstration.log</code> - Complete execution log</li>
                    <li><code>phase2_demonstration_report.html</code> - This report</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 Generated by LeanVibe Agent Hive 2.0 - Phase 2 Milestone System</p>
            <p>For technical questions, consult the development team or documentation.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _render_validation_section(self) -> str:
        """Render validation results section."""
        validation_results = self.results.get("stages", {}).get("validation", {})
        
        if not validation_results:
            return '<div class="stage-item"><h3>Validation Skipped</h3><p>Validation was not run in this execution.</p></div>'
        
        status = validation_results.get("overall_status", "UNKNOWN")
        passed_tests = validation_results.get("passed_tests", 0)
        failed_tests = validation_results.get("failed_tests", 0)
        success_rate = validation_results.get("success_rate", 0)
        
        return f"""
        <div class="stage-item {'failed' if status == 'FAILED' else ''}">
            <h3>Validation Status: <span class="status-badge status-{'passed' if status == 'PASSED' else 'failed'}">{status}</span></h3>
            <p><strong>Tests Passed:</strong> {passed_tests}</p>
            <p><strong>Tests Failed:</strong> {failed_tests}</p>
            <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
            <p><strong>Duration:</strong> {validation_results.get('duration_seconds', 'N/A')} seconds</p>
        </div>
        """
    
    def _render_demonstration_section(self) -> str:
        """Render demonstration results section."""
        demo_results = self.results.get("stages", {}).get("demonstration", {})
        
        if not demo_results:
            return '<div class="stage-item"><h3>Demonstration Skipped</h3><p>Demonstration was not run in this execution.</p></div>'
        
        success = demo_results.get("success", False)
        phases_completed = len(demo_results.get("phases_completed", []))
        total_phases = demo_results.get("total_phases", 8)
        errors = len(demo_results.get("errors", []))
        
        return f"""
        <div class="stage-item {'failed' if not success else ''}">
            <h3>Demonstration Status: <span class="status-badge status-{'passed' if success else 'failed'}">{'PASSED' if success else 'FAILED'}</span></h3>
            <p><strong>Phases Completed:</strong> {phases_completed}/{total_phases}</p>
            <p><strong>Errors Encountered:</strong> {errors}</p>
            <p><strong>Start Time:</strong> {demo_results.get('start_time', 'N/A')}</p>
            <p><strong>End Time:</strong> {demo_results.get('end_time', 'N/A')}</p>
        </div>
        """
    
    def _render_performance_section(self) -> str:
        """Render performance analysis section."""
        demo_results = self.results.get("stages", {}).get("demonstration", {})
        performance_metrics = demo_results.get("performance_metrics", {}) if demo_results else {}
        
        if not performance_metrics:
            return '<div class="stage-item"><h3>No Performance Data</h3><p>Performance metrics were not collected.</p></div>'
        
        return """
        <div class="stage-item">
            <h3>Performance Targets Validation</h3>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Target</th>
                        <th>Actual</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Message Throughput</td>
                        <td>&gt;10,000 msgs/sec</td>
                        <td>See results file</td>
                        <td>See validation</td>
                    </tr>
                    <tr>
                        <td>Recovery Time</td>
                        <td>&lt;30 seconds</td>
                        <td>See results file</td>
                        <td>See validation</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    
    async def run_complete_demonstration(self) -> Dict[str, Any]:
        """Run the complete demonstration workflow."""
        self.logger.info("🎯 STARTING COMPLETE PHASE 2 DEMONSTRATION")
        self.logger.info("=" * 80)
        
        execution_start = time.time()
        
        try:
            # Stage 1: Prerequisites check
            self.logger.info("\n📍 STAGE 1: Prerequisites Check")
            prereq_results = await self.check_prerequisites()
            self.results["stages"]["prerequisites"] = prereq_results
            
            if not prereq_results.get("ready_for_demonstration", False) and not self.demo_only:
                self.logger.error("❌ Prerequisites not met, aborting demonstration")
                self.results["overall_success"] = False
                return self.results
            
            # Stage 2: Validation (unless demo-only)
            if not self.demo_only:
                self.logger.info("\n📍 STAGE 2: Validation")
                validation_results = await self.run_validation()
                self.results["stages"]["validation"] = validation_results
                
                if self.validate_only:
                    self.logger.info("✅ Validation-only run completed")
                    self.results["overall_success"] = validation_results.get("overall_status") == "PASSED"
                    return self.results
                
                if validation_results.get("overall_status") != "PASSED":
                    self.logger.warning("⚠️ Validation failed, but continuing with demonstration")
            
            # Stage 3: Demonstration (unless validate-only)
            if not self.validate_only:
                self.logger.info("\n📍 STAGE 3: Demonstration Execution")
                demo_results = await self.run_demonstration()
                self.results["stages"]["demonstration"] = demo_results
                
                self.results["overall_success"] = demo_results.get("success", False)
            
            # Stage 4: Report generation
            self.logger.info("\n📍 STAGE 4: Report Generation")
            report_results = self.generate_comprehensive_report()
            self.results["stages"]["reporting"] = report_results
            
            # Final summary
            execution_duration = time.time() - execution_start
            self.results.update({
                "end_time": datetime.utcnow().isoformat(),
                "total_duration_seconds": execution_duration,
                "stages_completed": len(self.results["stages"])
            })
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("🏁 PHASE 2 DEMONSTRATION COMPLETED")
            self.logger.info(f"✅ Overall Success: {'YES' if self.results['overall_success'] else 'NO'}")
            self.logger.info(f"⏱️ Total Duration: {execution_duration:.2f} seconds")
            self.logger.info(f"📁 Output Directory: {self.output_dir}")
            self.logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"💥 CRITICAL ERROR: {e}")
            self.results["overall_success"] = False
            self.results["error"] = str(e)
            self.results["end_time"] = datetime.utcnow().isoformat()
            return self.results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 2 Milestone Demonstration Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete demonstration with validation
    python run_phase_2_demonstration.py
    
    # Run only validation
    python run_phase_2_demonstration.py --validate-only
    
    # Run only demonstration (skip validation)
    python run_phase_2_demonstration.py --demo-only
    
    # Run with high performance testing
    python run_phase_2_demonstration.py --performance-mode --verbose
        """
    )
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, skip demonstration')
    parser.add_argument('--demo-only', action='store_true',
                       help='Skip validation, run demonstration directly')
    parser.add_argument('--performance-mode', action='store_true',
                       help='Enable high-performance testing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--output-dir', default='./phase2_results',
                       help='Directory for output files (default: ./phase2_results)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.validate_only and args.demo_only:
        print("Error: Cannot specify both --validate-only and --demo-only")
        sys.exit(1)
    
    # Create runner and execute
    runner = Phase2DemonstrationRunner(
        validate_only=args.validate_only,
        demo_only=args.demo_only,
        performance_mode=args.performance_mode,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    try:
        results = await runner.run_complete_demonstration()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print(f"✅ Success: {'YES' if results.get('overall_success', False) else 'NO'}")
        print(f"📁 Results: {runner.output_dir}")
        print(f"⏱️ Duration: {results.get('total_duration_seconds', 0):.2f}s")
        print("=" * 60)
        
        # Exit with appropriate code
        sys.exit(0 if results.get('overall_success', False) else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())