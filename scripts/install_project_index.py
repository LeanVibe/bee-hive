#!/usr/bin/env python3
"""
Universal Project Index Installer

Automatically detects project type and installs Project Index system with optimal configuration.

Usage:
    python install_project_index.py /path/to/project
    python install_project_index.py /path/to/project --server-url http://localhost:8000
    python install_project_index.py . --auto-detect --analyze-now
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
import time

class ProjectDetector:
    """Intelligent project type and configuration detection"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.detected_languages = set()
        self.detected_frameworks = set()
        self.project_type = "unknown"
        
    def detect_languages(self) -> List[str]:
        """Detect programming languages in project"""
        language_indicators = {
            "python": ["*.py", "requirements.txt", "pyproject.toml", "setup.py", "Pipfile"],
            "javascript": ["*.js", "package.json", "*.mjs"],
            "typescript": ["*.ts", "*.tsx", "tsconfig.json"],
            "java": ["*.java", "pom.xml", "build.gradle"],
            "go": ["*.go", "go.mod", "go.sum"],
            "rust": ["*.rs", "Cargo.toml"],
            "php": ["*.php", "composer.json"],
            "ruby": ["*.rb", "Gemfile"],
            "csharp": ["*.cs", "*.csproj", "*.sln"],
            "cpp": ["*.cpp", "*.hpp", "*.cc", "*.h", "CMakeLists.txt"],
            "swift": ["*.swift", "Package.swift"],
            "kotlin": ["*.kt", "*.kts"],
            "scala": ["*.scala", "build.sbt"],
            "sql": ["*.sql", "*.ddl"],
            "yaml": ["*.yaml", "*.yml"],
            "json": ["*.json"],
            "markdown": ["*.md", "*.markdown"]
        }
        
        detected = set()
        
        for language, patterns in language_indicators.items():
            for pattern in patterns:
                if list(self.project_path.glob(f"**/{pattern}")):
                    detected.add(language)
                    break
        
        return sorted(detected)
    
    def detect_frameworks(self) -> List[str]:
        """Detect frameworks and libraries"""
        frameworks = []
        
        # Python frameworks
        if "python" in self.detected_languages:
            if self._check_file_content("requirements.txt", "fastapi"):
                frameworks.append("fastapi")
            if self._check_file_content("requirements.txt", "django"):
                frameworks.append("django")
            if self._check_file_content("requirements.txt", "flask"):
                frameworks.append("flask")
            if (self.project_path / "manage.py").exists():
                frameworks.append("django")
                
        # JavaScript/TypeScript frameworks
        if any(lang in self.detected_languages for lang in ["javascript", "typescript"]):
            package_json = self.project_path / "package.json"
            if package_json.exists():
                try:
                    package_data = json.loads(package_json.read_text())
                    dependencies = {**package_data.get("dependencies", {}), 
                                  **package_data.get("devDependencies", {})}
                    
                    if "react" in dependencies:
                        frameworks.append("react")
                    if "vue" in dependencies:
                        frameworks.append("vue")
                    if "angular" in dependencies:
                        frameworks.append("angular")
                    if "next" in dependencies:
                        frameworks.append("nextjs")
                    if "express" in dependencies:
                        frameworks.append("express")
                except:
                    pass
        
        return frameworks
    
    def _check_file_content(self, filename: str, content: str) -> bool:
        """Check if file contains specific content"""
        file_path = self.project_path / filename
        if file_path.exists():
            try:
                return content.lower() in file_path.read_text().lower()
            except:
                return False
        return False
    
    def detect_project_type(self) -> str:
        """Determine overall project type"""
        if "fastapi" in self.detected_frameworks:
            return "fastapi-backend"
        elif "django" in self.detected_frameworks:
            return "django-backend"
        elif "flask" in self.detected_frameworks:
            return "flask-backend"
        elif "react" in self.detected_frameworks:
            return "react-frontend"
        elif "vue" in self.detected_frameworks:
            return "vue-frontend"
        elif "python" in self.detected_languages:
            return "python-package"
        elif any(lang in self.detected_languages for lang in ["javascript", "typescript"]):
            return "nodejs-package"
        else:
            return "multi-language"
    
    def generate_configuration(self) -> Dict:
        """Generate optimal configuration for detected project"""
        self.detected_languages = set(self.detect_languages())
        self.detected_frameworks = set(self.detect_frameworks())
        self.project_type = self.detect_project_type()
        
        # Base configuration
        config = {
            "languages": list(self.detected_languages),
            "project_type": self.project_type,
            "frameworks": list(self.detected_frameworks),
            "exclude_patterns": [
                ".git", "__pycache__", "*.pyc", ".pytest_cache",
                "node_modules", "dist", "build", ".next",
                "target", "bin", "obj", ".idea", ".vscode",
                "*.log", "*.tmp", ".env", ".DS_Store"
            ],
            "include_patterns": [],
            "analysis_depth": 3,
            "max_file_size": 1048576,  # 1MB
            "enable_ai_analysis": True
        }
        
        # Language-specific patterns
        if "python" in self.detected_languages:
            config["include_patterns"].extend([
                "*.py", "requirements*.txt", "pyproject.toml", 
                "setup.py", "setup.cfg", "tox.ini"
            ])
            config["exclude_patterns"].extend([
                "*.egg-info", ".tox", ".coverage", "htmlcov"
            ])
            
        if any(lang in self.detected_languages for lang in ["javascript", "typescript"]):
            config["include_patterns"].extend([
                "*.js", "*.ts", "*.jsx", "*.tsx", "package.json",
                "tsconfig.json", "*.json"
            ])
            config["exclude_patterns"].extend([
                "package-lock.json", "yarn.lock", ".npm"
            ])
            
        if "java" in self.detected_languages:
            config["include_patterns"].extend([
                "*.java", "pom.xml", "build.gradle", "gradle.properties"
            ])
            
        if "go" in self.detected_languages:
            config["include_patterns"].extend([
                "*.go", "go.mod", "go.sum"
            ])
            
        # Framework-specific adjustments
        if "fastapi" in self.detected_frameworks:
            config["analysis_depth"] = 4  # Deeper analysis for API projects
            config["include_patterns"].extend(["*.sql", "alembic/**/*.py"])
            
        if "react" in self.detected_frameworks:
            config["include_patterns"].extend(["*.css", "*.scss", "*.sass"])
            
        # Project size-based adjustments
        total_files = len(list(self.project_path.glob("**/*")))
        if total_files > 5000:
            config["analysis_depth"] = 2  # Reduce depth for large projects
            config["max_file_size"] = 524288  # 512KB for large projects
        elif total_files > 1000:
            config["analysis_depth"] = 3
            
        # Context optimization settings
        config["context_optimization"] = {
            "max_context_files": min(30, max(10, total_files // 100)),
            "relevance_threshold": 0.3,
            "include_test_files": True,
            "prioritize_entry_points": True
        }
        
        return config


class ProjectIndexInstaller:
    """Universal Project Index installer"""
    
    def __init__(self, project_path: Path, server_url: str = "http://localhost:8000"):
        self.project_path = project_path.resolve()
        self.server_url = server_url.rstrip("/")
        self.detector = ProjectDetector(project_path)
        
    async def check_server_health(self) -> bool:
        """Check if Project Index server is running"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                return response.status_code == 200
        except:
            return False
    
    async def create_project_index(self, config: Dict, auth_token: str = None) -> str:
        """Create project index via API"""
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        project_data = {
            "name": self.project_path.name,
            "description": f"Auto-generated index for {self.project_path.name} ({config['project_type']})",
            "root_path": str(self.project_path),
            "configuration": config
        }
        
        # Add git information if available
        git_dir = self.project_path / ".git"
        if git_dir.exists():
            try:
                # Try to get git remote URL
                import subprocess
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    project_data["git_repository"] = result.stdout.strip()
                    
                # Get current branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    project_data["git_branch"] = result.stdout.strip()
            except:
                pass
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.server_url}/api/project-index/create",
                json=project_data,
                headers=headers
            )
            
            if response.status_code == 201:
                return response.json()["id"]
            else:
                raise Exception(f"Failed to create project index: {response.status_code} {response.text}")
    
    async def trigger_analysis(self, project_id: str, auth_token: str = None) -> None:
        """Trigger project analysis"""
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.server_url}/api/project-index/{project_id}/analyze",
                headers=headers
            )
            
            if response.status_code not in [200, 202]:
                raise Exception(f"Failed to trigger analysis: {response.status_code} {response.text}")
    
    async def wait_for_analysis(self, project_id: str, timeout: int = 300, auth_token: str = None) -> bool:
        """Wait for analysis to complete"""
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(
                        f"{self.server_url}/api/project-index/{project_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status", "unknown")
                        
                        if status == "analyzed":
                            return True
                        elif status == "failed":
                            raise Exception("Analysis failed")
                        
                        # Show progress if available
                        stats = data.get("statistics", {})
                        if stats:
                            processed = stats.get("analyzed_files", 0)
                            total = stats.get("total_files", 0)
                            if total > 0:
                                progress = (processed / total) * 100
                                print(f"\rAnalysis progress: {progress:.1f}% ({processed}/{total} files)", end="")
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"Error checking analysis status: {e}")
                    await asyncio.sleep(5)
            
        return False
    
    async def install(self, auth_token: str = None, analyze_now: bool = True, wait_for_completion: bool = False) -> Dict:
        """Complete installation process"""
        print(f"🔍 Analyzing project: {self.project_path}")
        
        # Check server health
        if not await self.check_server_health():
            raise Exception(f"Project Index server is not accessible at {self.server_url}")
        
        print("✅ Server is accessible")
        
        # Detect project configuration
        config = self.detector.generate_configuration()
        
        print(f"📊 Detected project type: {config['project_type']}")
        print(f"🔤 Languages: {', '.join(config['languages'])}")
        if config['frameworks']:
            print(f"🛠️  Frameworks: {', '.join(config['frameworks'])}")
        
        # Create project index
        print("\n📝 Creating project index...")
        project_id = await self.create_project_index(config, auth_token)
        print(f"✅ Project index created: {project_id}")
        
        result = {
            "project_id": project_id,
            "project_path": str(self.project_path),
            "configuration": config,
            "dashboard_url": f"{self.server_url}/dashboard/project-index/{project_id}",
            "api_url": f"{self.server_url}/api/project-index/{project_id}"
        }
        
        if analyze_now:
            print("\n🔍 Triggering initial analysis...")
            await self.trigger_analysis(project_id, auth_token)
            print("✅ Analysis started")
            
            if wait_for_completion:
                print("\n⏳ Waiting for analysis to complete...")
                success = await self.wait_for_analysis(project_id, auth_token=auth_token)
                if success:
                    print("\n✅ Analysis completed successfully!")
                    
                    # Get final statistics
                    async with httpx.AsyncClient() as client:
                        headers = {}
                        if auth_token:
                            headers["Authorization"] = f"Bearer {auth_token}"
                            
                        response = await client.get(
                            f"{self.server_url}/api/project-index/{project_id}",
                            headers=headers
                        )
                        if response.status_code == 200:
                            data = response.json()
                            stats = data.get("statistics", {})
                            print(f"📈 Files analyzed: {stats.get('analyzed_files', 0)}")
                            print(f"🔗 Dependencies found: {stats.get('total_dependencies', 0)}")
                            print(f"⏱️  Analysis time: {stats.get('last_analysis_duration', 0):.1f}s")
                else:
                    print("\n⚠️  Analysis did not complete within timeout (check dashboard for progress)")
        
        return result


async def main():
    parser = argparse.ArgumentParser(
        description="Universal Project Index Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_project_index.py /path/to/my-project
  python install_project_index.py . --analyze-now --wait
  python install_project_index.py ../other-project --server-url http://prod-server:8000
  python install_project_index.py /workspace --auth-token your-token-here
        """
    )
    
    parser.add_argument(
        "project_path",
        help="Path to project directory to index"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Project Index server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--auth-token",
        help="Authentication token for API requests"
    )
    parser.add_argument(
        "--analyze-now",
        action="store_true",
        help="Start analysis immediately after creation"
    )
    parser.add_argument(
        "--wait",
        action="store_true", 
        help="Wait for analysis to complete"
    )
    parser.add_argument(
        "--output-json",
        help="Save installation result to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"❌ Project path does not exist: {project_path}")
            sys.exit(1)
        
        if not project_path.is_dir():
            print(f"❌ Project path is not a directory: {project_path}")
            sys.exit(1)
        
        installer = ProjectIndexInstaller(project_path, args.server_url)
        
        result = await installer.install(
            auth_token=args.auth_token,
            analyze_now=args.analyze_now,
            wait_for_completion=args.wait
        )
        
        print(f"\n🎉 Installation completed successfully!")
        print(f"📊 Dashboard: {result['dashboard_url']}")
        print(f"🔗 API: {result['api_url']}")
        
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Result saved to: {args.output_json}")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())