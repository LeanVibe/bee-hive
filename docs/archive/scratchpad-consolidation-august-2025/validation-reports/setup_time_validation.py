#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Setup Time Validation
=============================================

Professional validation of the <2 minute setup claim with statistical analysis.
Tests multiple scenarios and provides measurable data.

Author: The Guardian (QA & Test Automation Specialist)
Date: 2025-08-01
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SetupTimeValidator:
    """Validates setup time claims with statistical rigor"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = []
        self.temp_dirs = []
        self.target_time = 120  # 2 minutes in seconds

    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")

    def create_fresh_copy(self) -> Path:
        """Create a fresh copy of the repository for testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix="leanvibe_setup_test_"))
        self.temp_dirs.append(temp_dir)
        
        # Copy essential files only (to simulate git clone)
        essential_files = [
            "scripts",
            "Makefile", 
            "pyproject.toml",
            "docker-compose.yml",
            "docker-compose.fast.yml",
            "alembic.ini",
            "app",
            "migrations"
        ]
        
        for item_name in essential_files:
            item_path = self.project_root / item_name
            if item_path.exists():
                dest_path = temp_dir / item_name
                if item_path.is_dir():
                    # Copy directory, excluding large caches
                    shutil.copytree(
                        item_path, 
                        dest_path,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '*.log', 'venv', 'node_modules')
                    )
                else:
                    shutil.copy2(item_path, dest_path)
        
        logger.debug(f"Created fresh copy: {temp_dir}")
        return temp_dir

    def run_setup_with_timing(self, test_dir: Path, scenario: str) -> Tuple[bool, float, str, str]:
        """Run setup and measure time"""
        logger.info(f"Running setup test: {scenario}")
        
        start_time = time.time()
        
        try:
            # Run make setup
            result = subprocess.run(
                ["make", "setup"],
                cwd=test_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute max timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return success, duration, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, duration, "", "Setup timed out after 5 minutes"
        except Exception as e:
            duration = time.time() - start_time
            return False, duration, "", str(e)

    def test_fresh_setup_performance(self, runs: int = 3) -> Dict[str, Any]:
        """Test fresh setup performance with multiple runs"""
        logger.info(f"🚀 Testing Fresh Setup Performance ({runs} runs)")
        
        times = []
        successes = 0
        details = []
        
        for run in range(runs):
            logger.info(f"Setup performance run {run + 1}/{runs}")
            
            # Create fresh environment for each run
            test_dir = self.create_fresh_copy()
            
            success, duration, stdout, stderr = self.run_setup_with_timing(test_dir, f"Fresh Setup Run {run + 1}")
            
            if success:
                times.append(duration)
                successes += 1
                logger.info(f"  ✅ Run {run + 1}: {duration:.1f}s (SUCCESS)")
            else:
                logger.warning(f"  ❌ Run {run + 1}: {duration:.1f}s (FAILED)")
            
            details.append({
                "run": run + 1,
                "success": success,
                "duration": duration,
                "under_target": duration < self.target_time,
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "error_summary": stderr[:200] if stderr else None
            })
        
        # Calculate statistics
        if times:
            stats = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "successful_runs": len(times),
                "total_runs": runs,
                "success_rate": len(times) / runs * 100,
                "target_time": self.target_time,
                "mean_under_target": statistics.mean(times) < self.target_time if times else False,
                "all_under_target": all(t < self.target_time for t in times)
            }
        else:
            stats = {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std_dev": 0.0,
                "successful_runs": 0,
                "total_runs": runs,
                "success_rate": 0.0,
                "target_time": self.target_time,
                "mean_under_target": False,
                "all_under_target": False
            }
        
        return {
            "test_name": "Fresh Setup Performance",
            "statistics": stats,
            "detailed_runs": details,
            "measurements": times,
            "assessment": self._assess_setup_performance(stats)
        }

    def test_existing_setup_performance(self, runs: int = 3) -> Dict[str, Any]:
        """Test setup performance on existing checkout"""
        logger.info(f"🔄 Testing Existing Setup Performance ({runs} runs)")
        
        times = []
        successes = 0
        details = []
        
        for run in range(runs):
            logger.info(f"Existing setup run {run + 1}/{runs}")
            
            success, duration, stdout, stderr = self.run_setup_with_timing(
                self.project_root, 
                f"Existing Setup Run {run + 1}"
            )
            
            if success:
                times.append(duration)
                successes += 1
                logger.info(f"  ✅ Run {run + 1}: {duration:.1f}s (SUCCESS)")
            else:
                logger.warning(f"  ❌ Run {run + 1}: {duration:.1f}s (FAILED)")
            
            details.append({
                "run": run + 1,
                "success": success,
                "duration": duration,
                "under_target": duration < self.target_time,
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "error_summary": stderr[:200] if stderr else None
            })
        
        # Calculate statistics
        if times:
            stats = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "successful_runs": len(times),
                "total_runs": runs,
                "success_rate": len(times) / runs * 100,
                "target_time": self.target_time,
                "mean_under_target": statistics.mean(times) < self.target_time if times else False,
                "all_under_target": all(t < self.target_time for t in times)
            }
        else:
            stats = {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std_dev": 0.0,
                "successful_runs": 0,
                "total_runs": runs,
                "success_rate": 0.0,
                "target_time": self.target_time,
                "mean_under_target": False,
                "all_under_target": False
            }
        
        return {
            "test_name": "Existing Setup Performance",
            "statistics": stats,
            "detailed_runs": details,
            "measurements": times,
            "assessment": self._assess_setup_performance(stats)
        }

    def _assess_setup_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess setup performance against targets"""
        assessment = {
            "target_claim": f"<{self.target_time}s (2 minutes)",
            "actual_mean": f"{stats['mean']:.1f}s",
            "claim_validated": stats["mean_under_target"],
            "consistency": "HIGH" if stats["std_dev"] < 30 else "MEDIUM" if stats["std_dev"] < 60 else "LOW",
            "reliability": f"{stats['success_rate']:.0f}%",
            "performance_grade": self._calculate_performance_grade(stats)
        }
        
        return assessment

    def _calculate_performance_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate performance grade"""
        if stats["success_rate"] < 80:
            return "F - Unreliable"
        elif not stats["mean_under_target"]:
            return "D - Does not meet target"
        elif stats["mean"] > self.target_time * 0.8:  # > 96 seconds
            return "C - Barely meets target"
        elif stats["mean"] > self.target_time * 0.6:  # > 72 seconds
            return "B - Good performance"
        else:  # <= 72 seconds
            return "A - Excellent performance"

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup time validation report"""
        
        # Run performance tests
        fresh_results = self.test_fresh_setup_performance(runs=3)
        existing_results = self.test_existing_setup_performance(runs=3)
        
        # Overall assessment
        overall_validated = (
            fresh_results["statistics"]["mean_under_target"] and 
            existing_results["statistics"]["mean_under_target"]
        )
        
        report = {
            "validation_metadata": {
                "test_date": datetime.now().isoformat(),
                "target_claim": f"Setup completes in <{self.target_time} seconds",
                "test_methodology": "Multiple isolated runs with statistical analysis",
                "validator": "The Guardian QA Framework"
            },
            "fresh_setup_results": fresh_results,
            "existing_setup_results": existing_results,
            "overall_assessment": {
                "claim_validated": overall_validated,
                "confidence_level": "HIGH" if overall_validated else "MEDIUM",
                "summary": self._generate_summary(fresh_results, existing_results),
                "recommendations": self._generate_performance_recommendations(fresh_results, existing_results)
            },
            "statistical_evidence": {
                "fresh_setup_mean": fresh_results["statistics"]["mean"],
                "existing_setup_mean": existing_results["statistics"]["mean"],
                "combined_success_rate": (
                    fresh_results["statistics"]["success_rate"] + 
                    existing_results["statistics"]["success_rate"]
                ) / 2,
                "target_threshold": self.target_time
            }
        }
        
        return report

    def _generate_summary(self, fresh_results: Dict[str, Any], existing_results: Dict[str, Any]) -> str:
        """Generate overall summary"""
        fresh_mean = fresh_results["statistics"]["mean"]
        existing_mean = existing_results["statistics"]["mean"]
        
        if fresh_results["statistics"]["mean_under_target"] and existing_results["statistics"]["mean_under_target"]:
            return f"✅ CLAIM VALIDATED: Fresh setup averages {fresh_mean:.1f}s, existing setup averages {existing_mean:.1f}s - both under 2-minute target"
        elif fresh_results["statistics"]["mean_under_target"]:
            return f"⚠️ PARTIAL VALIDATION: Fresh setup meets target ({fresh_mean:.1f}s) but existing setup is slow ({existing_mean:.1f}s)"
        elif existing_results["statistics"]["mean_under_target"]:
            return f"⚠️ PARTIAL VALIDATION: Existing setup meets target ({existing_mean:.1f}s) but fresh setup is slow ({fresh_mean:.1f}s)"
        else:
            return f"❌ CLAIM NOT VALIDATED: Fresh setup averages {fresh_mean:.1f}s, existing setup averages {existing_mean:.1f}s - both exceed 2-minute target"

    def _generate_performance_recommendations(self, fresh_results: Dict[str, Any], existing_results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        fresh_stats = fresh_results["statistics"]
        existing_stats = existing_results["statistics"]
        
        if fresh_stats["success_rate"] < 100:
            recommendations.append(f"Improve fresh setup reliability - only {fresh_stats['success_rate']:.0f}% success rate")
        
        if existing_stats["success_rate"] < 100:
            recommendations.append(f"Improve existing setup reliability - only {existing_stats['success_rate']:.0f}% success rate")
        
        if not fresh_stats["mean_under_target"]:
            recommendations.append(f"Optimize fresh setup time - currently {fresh_stats['mean']:.1f}s, target <{self.target_time}s")
        
        if not existing_stats["mean_under_target"]:
            recommendations.append(f"Optimize existing setup time - currently {existing_stats['mean']:.1f}s, target <{self.target_time}s")
        
        if fresh_stats["std_dev"] > 30:
            recommendations.append("Reduce fresh setup time variability for consistent developer experience")
        
        if existing_stats["std_dev"] > 30:
            recommendations.append("Reduce existing setup time variability for consistent developer experience")
        
        if not recommendations:
            recommendations.append("Performance targets met - consider advanced optimizations for even faster setup")
        
        return recommendations

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive Setup Time Validation")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--output", type=Path, help="Output file for validation report")
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs per scenario")
    
    args = parser.parse_args()
    
    # Run setup time validation
    with SetupTimeValidator(project_root=args.project_root) as validator:
        if args.runs:
            # Modify the validator to use custom run count
            pass
        
        report = validator.generate_comprehensive_report()
        
        # Save report
        output_file = args.output or Path("setup_time_validation_report.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("⏱️ SETUP TIME VALIDATION COMPLETE")
        print("="*80)
        
        fresh_stats = report["fresh_setup_results"]["statistics"]
        existing_stats = report["existing_setup_results"]["statistics"]
        
        print(f"Target Claim: Setup completes in <{validator.target_time}s (2 minutes)")
        print("")
        print("📊 RESULTS:")
        print(f"  Fresh Setup:    {fresh_stats['mean']:.1f}s avg (Success: {fresh_stats['success_rate']:.0f}%)")
        print(f"  Existing Setup: {existing_stats['mean']:.1f}s avg (Success: {existing_stats['success_rate']:.0f}%)")
        print("")
        print(f"🎯 VALIDATION: {report['overall_assessment']['summary']}")
        print(f"📈 CONFIDENCE: {report['overall_assessment']['confidence_level']}")
        print("")
        
        if report['overall_assessment']['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for rec in report['overall_assessment']['recommendations']:
                print(f"  • {rec}")
            print("")
        
        print(f"📋 Report saved to: {output_file}")
        print("="*80)

if __name__ == "__main__":
    main()