#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Intelligent Project Detection System
===================================================================

Validates the accuracy and performance of the project detection system
across various project types, languages, and configurations.

Features:
- Language detection accuracy testing
- Framework identification validation
- Dependency analysis verification
- Configuration generation testing
- Performance benchmarking
- Edge case handling

Author: Claude Code Agent for LeanVibe Agent Hive 2.0
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import shutil
import os

# Setup test environment
test_logger = logging.getLogger(__name__)


class TestProjectSamples:
    """Creates sample projects for testing detection accuracy."""
    
    @staticmethod
    def create_python_django_project(base_path: Path) -> Path:
        """Create a sample Django project."""
        project_path = base_path / "django_sample"
        project_path.mkdir(exist_ok=True)
        
        # Create Django project structure
        (project_path / "manage.py").write_text("""#!/usr/bin/env python
import os
import sys
if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
""")
        
        # Create app structure
        mysite_dir = project_path / "mysite"
        mysite_dir.mkdir(exist_ok=True)
        
        (mysite_dir / "__init__.py").write_text("")
        (mysite_dir / "settings.py").write_text("""
import os
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-test-key'
DEBUG = True
ALLOWED_HOSTS = []
INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.auth']
""")
        
        (mysite_dir / "urls.py").write_text("""
from django.contrib import admin
from django.urls import path
urlpatterns = [path('admin/', admin.site.urls)]
""")
        
        # Create requirements.txt
        (project_path / "requirements.txt").write_text("""
Django>=4.2,<5.0
psycopg2-binary>=2.9
redis>=4.5
celery>=5.2
""")
        
        # Create app
        blog_dir = project_path / "blog"
        blog_dir.mkdir(exist_ok=True)
        (blog_dir / "__init__.py").write_text("")
        (blog_dir / "models.py").write_text("""
from django.db import models
class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
""")
        (blog_dir / "views.py").write_text("""
from django.shortcuts import render
def index(request):
    return render(request, 'blog/index.html')
""")
        
        # Create tests
        tests_dir = project_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test_models.py").write_text("""
from django.test import TestCase
from blog.models import Post
class PostTestCase(TestCase):
    def test_post_creation(self):
        post = Post.objects.create(title="Test", content="Content")
        self.assertEqual(post.title, "Test")
""")
        
        return project_path
    
    @staticmethod
    def create_react_project(base_path: Path) -> Path:
        """Create a sample React project."""
        project_path = base_path / "react_sample"
        project_path.mkdir(exist_ok=True)
        
        # Create package.json
        (project_path / "package.json").write_text(json.dumps({
            "name": "react-sample",
            "version": "0.1.0",
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.8.0",
                "axios": "^1.3.0"
            },
            "devDependencies": {
                "jest": "^29.4.0",
                "@testing-library/react": "^13.4.0",
                "@testing-library/jest-dom": "^5.16.0",
                "webpack": "^5.75.0",
                "babel-loader": "^9.1.0"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test"
            }
        }, indent=2))
        
        # Create source structure
        src_dir = project_path / "src"
        src_dir.mkdir(exist_ok=True)
        
        (src_dir / "index.js").write_text("""
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
""")
        
        (src_dir / "App.js").write_text("""
import React, { useState, useEffect } from 'react';
import './App.css';
function App() {
  const [data, setData] = useState([]);
  useEffect(() => {
    fetch('/api/data').then(res => res.json()).then(setData);
  }, []);
  return <div className="App"><h1>Hello React</h1></div>;
}
export default App;
""")
        
        # Create components
        components_dir = src_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        (components_dir / "Header.jsx").write_text("""
import React from 'react';
const Header = ({ title }) => {
  return <header><h1>{title}</h1></header>;
};
export default Header;
""")
        
        # Create tests
        tests_dir = src_dir / "__tests__"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "App.test.js").write_text("""
import { render, screen } from '@testing-library/react';
import App from '../App';
test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/hello react/i);
  expect(linkElement).toBeInTheDocument();
});
""")
        
        # Create public files
        public_dir = project_path / "public"
        public_dir.mkdir(exist_ok=True)
        (public_dir / "index.html").write_text("""
<!DOCTYPE html>
<html><head><title>React Sample</title></head>
<body><div id="root"></div></body></html>
""")
        
        return project_path
    
    @staticmethod
    def create_go_project(base_path: Path) -> Path:
        """Create a sample Go project."""
        project_path = base_path / "go_sample"
        project_path.mkdir(exist_ok=True)
        
        # Create go.mod
        (project_path / "go.mod").write_text("""
module github.com/example/go-sample
go 1.21
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/stretchr/testify v1.8.4
    gorm.io/gorm v1.25.1
    gorm.io/driver/postgres v1.5.2
)
""")
        
        # Create main.go
        (project_path / "main.go").write_text("""
package main
import (
    "github.com/gin-gonic/gin"
    "net/http"
)
func main() {
    r := gin.Default()
    r.GET("/", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{"message": "Hello Go!"})
    })
    r.Run(":8080")
}
""")
        
        # Create package structure
        handlers_dir = project_path / "handlers"
        handlers_dir.mkdir(exist_ok=True)
        (handlers_dir / "user.go").write_text("""
package handlers
import "github.com/gin-gonic/gin"
func GetUser(c *gin.Context) {
    c.JSON(200, gin.H{"user": "example"})
}
""")
        
        models_dir = project_path / "models"
        models_dir.mkdir(exist_ok=True)
        (models_dir / "user.go").write_text("""
package models
type User struct {
    ID   uint   `json:"id"`
    Name string `json:"name"`
}
""")
        
        # Create tests
        (project_path / "main_test.go").write_text("""
package main
import (
    "testing"
    "github.com/stretchr/testify/assert"
)
func TestMain(t *testing.T) {
    assert.True(t, true)
}
""")
        
        return project_path
    
    @staticmethod
    def create_rust_project(base_path: Path) -> Path:
        """Create a sample Rust project."""
        project_path = base_path / "rust_sample"
        project_path.mkdir(exist_ok=True)
        
        # Create Cargo.toml
        (project_path / "Cargo.toml").write_text("""
[package]
name = "rust-sample"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
reqwest = { version = "0.11", features = ["json"] }
axum = "0.6"

[dev-dependencies]
tokio-test = "0.4"
""")
        
        # Create source structure
        src_dir = project_path / "src"
        src_dir.mkdir(exist_ok=True)
        
        (src_dir / "main.rs").write_text("""
use axum::{routing::get, Router};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(hello));
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn hello() -> &'static str {
    "Hello, Rust!"
}
""")
        
        (src_dir / "lib.rs").write_text("""
pub mod models;
pub mod handlers;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
""")
        
        (src_dir / "models.rs").write_text("""
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
}
""")
        
        (src_dir / "handlers.rs").write_text("""
use axum::Json;
use crate::models::User;

pub async fn get_user() -> Json<User> {
    Json(User { id: 1, name: "Rust User".to_string() })
}
""")
        
        return project_path
    
    @staticmethod
    def create_java_spring_project(base_path: Path) -> Path:
        """Create a sample Java Spring project."""
        project_path = base_path / "java_spring_sample"
        project_path.mkdir(exist_ok=True)
        
        # Create pom.xml
        (project_path / "pom.xml").write_text("""
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>spring-sample</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.1.0</version>
    </parent>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
""")
        
        # Create source structure
        src_main_java = project_path / "src" / "main" / "java" / "com" / "example" / "demo"
        src_main_java.mkdir(parents=True, exist_ok=True)
        
        (src_main_java / "Application.java").write_text("""
package com.example.demo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
""")
        
        (src_main_java / "UserController.java").write_text("""
package com.example.demo;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @GetMapping
    public List<User> getUsers() {
        return List.of(new User(1L, "John Doe"));
    }
    
    @PostMapping
    public User createUser(@RequestBody User user) {
        return user;
    }
}
""")
        
        (src_main_java / "User.java").write_text("""
package com.example.demo;
import jakarta.persistence.*;

@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    
    public User() {}
    public User(Long id, String name) {
        this.id = id;
        this.name = name;
    }
    // Getters and setters omitted for brevity
}
""")
        
        # Create test structure
        src_test_java = project_path / "src" / "test" / "java" / "com" / "example" / "demo"
        src_test_java.mkdir(parents=True, exist_ok=True)
        
        (src_test_java / "ApplicationTests.java").write_text("""
package com.example.demo;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class ApplicationTests {
    @Test
    void contextLoads() {
    }
}
""")
        
        return project_path


class ProjectDetectionTestSuite(unittest.TestCase):
    """Comprehensive test suite for project detection system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="detection_test_"))
        self.samples = TestProjectSamples()
        
        # Import detection system components
        try:
            from app.project_index.intelligent_detector import IntelligentProjectDetector
            from enhanced_dependency_analyzer import EnhancedDependencyAnalyzer
            from advanced_structure_analyzer import AdvancedStructureAnalyzer
            from intelligent_config_generator import IntelligentConfigGenerator
            
            self.detector = IntelligentProjectDetector()
            self.dependency_analyzer = EnhancedDependencyAnalyzer()
            self.structure_analyzer = AdvancedStructureAnalyzer()
            self.config_generator = IntelligentConfigGenerator()
        except ImportError as e:
            self.skipTest(f"Could not import detection components: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_python_django_detection(self):
        """Test detection of Python Django project."""
        project_path = self.samples.create_python_django_project(self.test_dir)
        
        # Run detection
        result = self.detector.detect_project(project_path)
        
        # Verify results
        self.assertEqual(result.primary_language.language, 'python')
        self.assertGreaterEqual(result.primary_language.confidence.value, 'high')
        
        # Check for Django framework detection
        framework_names = [f.framework for f in result.detected_frameworks]
        self.assertIn('django', framework_names)
        
        # Verify dependency analysis
        self.assertTrue(len(result.dependency_analysis) > 0)
        dep_names = []
        for dep_analysis in result.dependency_analysis:
            dep_names.extend([d['name'] for d in dep_analysis.major_dependencies])
        self.assertTrue(any('django' in name.lower() for name in dep_names))
        
        # Check structure analysis
        self.assertEqual(result.structure_analysis.structure_type, 'modular')
        self.assertTrue(len(result.structure_analysis.entry_points) > 0)
        
        print(f"✅ Python Django detection test passed")
    
    def test_react_project_detection(self):
        """Test detection of React project."""
        project_path = self.samples.create_react_project(self.test_dir)
        
        # Run detection
        result = self.detector.detect_project(project_path)
        
        # Verify results
        self.assertEqual(result.primary_language.language, 'javascript')
        self.assertGreaterEqual(result.primary_language.confidence.value, 'high')
        
        # Check for React framework detection
        framework_names = [f.framework for f in result.detected_frameworks]
        self.assertIn('react', framework_names)
        
        # Verify project type detection
        structure_analysis = self.structure_analyzer.analyze_project_structure(project_path)
        self.assertEqual(structure_analysis.project_type.value, 'web_application')
        
        print(f"✅ React project detection test passed")
    
    def test_go_project_detection(self):
        """Test detection of Go project."""
        project_path = self.samples.create_go_project(self.test_dir)
        
        # Run detection
        result = self.detector.detect_project(project_path)
        
        # Verify results
        self.assertEqual(result.primary_language.language, 'go')
        
        # Check for Gin framework detection
        framework_names = [f.framework for f in result.detected_frameworks]
        self.assertIn('gin', framework_names)
        
        # Verify dependency analysis
        dep_graph = self.dependency_analyzer.analyze_project_dependencies(project_path)
        self.assertTrue(dep_graph.total_count > 0)
        
        print(f"✅ Go project detection test passed")
    
    def test_rust_project_detection(self):
        """Test detection of Rust project."""
        project_path = self.samples.create_rust_project(self.test_dir)
        
        # Run detection
        result = self.detector.detect_project(project_path)
        
        # Verify results
        self.assertEqual(result.primary_language.language, 'rust')
        
        # Check for Axum framework detection
        framework_names = [f.framework for f in result.detected_frameworks]
        self.assertIn('axum', framework_names)
        
        print(f"✅ Rust project detection test passed")
    
    def test_java_spring_detection(self):
        """Test detection of Java Spring project."""
        project_path = self.samples.create_java_spring_project(self.test_dir)
        
        # Run detection
        result = self.detector.detect_project(project_path)
        
        # Verify results
        self.assertEqual(result.primary_language.language, 'java')
        
        # Check for Spring framework detection
        framework_names = [f.framework for f in result.detected_frameworks]
        self.assertIn('spring', framework_names)
        
        print(f"✅ Java Spring detection test passed")
    
    def test_configuration_generation(self):
        """Test configuration generation for different project types."""
        projects = [
            ('python_django', self.samples.create_python_django_project),
            ('react', self.samples.create_react_project),
            ('go', self.samples.create_go_project),
            ('rust', self.samples.create_rust_project),
            ('java_spring', self.samples.create_java_spring_project)
        ]
        
        for project_name, creator_func in projects:
            with self.subTest(project=project_name):
                project_path = creator_func(self.test_dir / project_name)
                
                # Run detection
                result = self.detector.detect_project(project_path)
                
                # Convert result to dict for config generation
                result_dict = {
                    'project_path': result.project_path,
                    'primary_language': {
                        'language': result.primary_language.language,
                        'confidence': result.primary_language.confidence.value
                    },
                    'detected_frameworks': [
                        {'framework': f.framework} for f in result.detected_frameworks
                    ],
                    'size_analysis': {
                        'size_category': result.size_analysis.size_category.value
                    },
                    'confidence_score': result.confidence_score
                }
                
                # Generate configuration
                config = self.config_generator.generate_configuration(result_dict)
                
                # Verify configuration
                self.assertIsNotNone(config)
                self.assertEqual(config.project_path, str(project_path))
                self.assertTrue(len(config.file_patterns['include']) > 0)
                self.assertTrue(len(config.ignore_patterns) > 0)
                
                print(f"✅ Configuration generation test passed for {project_name}")
    
    def test_detection_performance(self):
        """Test detection performance across multiple projects."""
        projects = [
            self.samples.create_python_django_project(self.test_dir / "perf_python"),
            self.samples.create_react_project(self.test_dir / "perf_react"),
            self.samples.create_go_project(self.test_dir / "perf_go"),
        ]
        
        total_time = 0
        for i, project_path in enumerate(projects):
            start_time = time.time()
            result = self.detector.detect_project(project_path)
            detection_time = time.time() - start_time
            total_time += detection_time
            
            # Verify detection completed successfully
            self.assertIsNotNone(result.primary_language)
            self.assertGreater(result.confidence_score, 0.5)
            
            # Performance assertions (detection should be fast)
            self.assertLess(detection_time, 10.0, f"Detection took too long: {detection_time:.2f}s")
            
            print(f"🚀 Project {i+1} detected in {detection_time:.2f}s")
        
        avg_time = total_time / len(projects)
        self.assertLess(avg_time, 5.0, f"Average detection time too slow: {avg_time:.2f}s")
        
        print(f"✅ Performance test passed - Average: {avg_time:.2f}s")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Test empty directory
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        result = self.detector.detect_project(empty_dir)
        self.assertIsNotNone(result)
        self.assertLess(result.confidence_score, 0.5)  # Low confidence for empty project
        
        # Test directory with only configuration files
        config_only_dir = self.test_dir / "config_only"
        config_only_dir.mkdir()
        (config_only_dir / "package.json").write_text('{"name": "test"}')
        (config_only_dir / "README.md").write_text("# Test Project")
        
        result = self.detector.detect_project(config_only_dir)
        self.assertIsNotNone(result)
        
        print(f"✅ Edge cases test passed")
    
    def test_multilanguage_project(self):
        """Test detection of projects with multiple languages."""
        multi_lang_dir = self.test_dir / "multi_lang"
        multi_lang_dir.mkdir()
        
        # Create Python backend
        backend_dir = multi_lang_dir / "backend"
        backend_dir.mkdir()
        (backend_dir / "app.py").write_text("""
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello from Python!'
""")
        (backend_dir / "requirements.txt").write_text("Flask>=2.0")
        
        # Create JavaScript frontend
        frontend_dir = multi_lang_dir / "frontend"
        frontend_dir.mkdir()
        (frontend_dir / "index.js").write_text("""
import React from 'react';
import ReactDOM from 'react-dom';
ReactDOM.render(<h1>Hello from React!</h1>, document.getElementById('root'));
""")
        (frontend_dir / "package.json").write_text(json.dumps({
            "name": "frontend",
            "dependencies": {"react": "^18.0.0"}
        }))
        
        # Run detection
        result = self.detector.detect_project(multi_lang_dir)
        
        # Should detect both languages
        detected_languages = [result.primary_language.language] + [lang.language for lang in result.secondary_languages]
        self.assertTrue(any(lang == 'python' for lang in detected_languages))
        self.assertTrue(any(lang == 'javascript' for lang in detected_languages))
        
        print(f"✅ Multi-language project test passed")
    
    def test_comprehensive_workflow(self):
        """Test the complete workflow from detection to configuration."""
        # Create a comprehensive project
        project_path = self.samples.create_python_django_project(self.test_dir / "workflow_test")
        
        # Step 1: Run project detection
        detection_result = self.detector.detect_project(project_path)
        self.assertIsNotNone(detection_result)
        
        # Step 2: Run dependency analysis
        dep_graph = self.dependency_analyzer.analyze_project_dependencies(project_path)
        self.assertGreater(dep_graph.total_count, 0)
        
        # Step 3: Run structure analysis
        structure_analysis = self.structure_analyzer.analyze_project_structure(project_path)
        self.assertIsNotNone(structure_analysis)
        
        # Step 4: Generate configuration
        result_dict = {
            'project_path': detection_result.project_path,
            'primary_language': {
                'language': detection_result.primary_language.language,
                'confidence': detection_result.primary_language.confidence.value
            },
            'detected_frameworks': [
                {'framework': f.framework} for f in detection_result.detected_frameworks
            ],
            'size_analysis': {
                'size_category': detection_result.size_analysis.size_category.value
            },
            'confidence_score': detection_result.confidence_score
        }
        
        config = self.config_generator.generate_configuration(result_dict)
        self.assertIsNotNone(config)
        
        # Step 5: Export configuration
        config_path = self.test_dir / "test_config.json"
        self.config_generator.export_configuration(config, config_path)
        self.assertTrue(config_path.exists())
        
        # Verify exported configuration is valid JSON
        with open(config_path) as f:
            exported_config = json.load(f)
        self.assertIn('project_name', exported_config)
        self.assertIn('analysis', exported_config)
        
        print(f"✅ Comprehensive workflow test passed")


def run_detection_benchmark():
    """Run performance benchmarks for the detection system."""
    print("\n🚀 Running Detection System Benchmarks")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
    samples = TestProjectSamples()
    
    try:
        from app.project_index.intelligent_detector import IntelligentProjectDetector
        detector = IntelligentProjectDetector()
    except ImportError:
        print("❌ Could not import detection system")
        return
    
    # Create test projects
    projects = {
        'Python Django': samples.create_python_django_project(test_dir / "django"),
        'React': samples.create_react_project(test_dir / "react"),
        'Go Gin': samples.create_go_project(test_dir / "go"),
        'Rust Axum': samples.create_rust_project(test_dir / "rust"),
        'Java Spring': samples.create_java_spring_project(test_dir / "java")
    }
    
    results = {}
    total_time = 0
    
    for project_name, project_path in projects.items():
        print(f"\n📊 Benchmarking {project_name}...")
        
        # Warm up
        detector.detect_project(project_path)
        
        # Actual benchmark
        times = []
        for i in range(3):  # Run 3 times for average
            start_time = time.time()
            result = detector.detect_project(project_path)
            detection_time = time.time() - start_time
            times.append(detection_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results[project_name] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'confidence': result.confidence_score,
            'languages_detected': len([result.primary_language] + result.secondary_languages),
            'frameworks_detected': len(result.detected_frameworks)
        }
        
        total_time += avg_time
        
        print(f"  ⏱️  Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        print(f"  📈 Confidence: {result.confidence_score:.1%}")
        print(f"  🔍 Languages: {results[project_name]['languages_detected']}, Frameworks: {results[project_name]['frameworks_detected']}")
    
    # Summary
    print(f"\n📋 BENCHMARK SUMMARY")
    print(f"Total projects tested: {len(projects)}")
    print(f"Total detection time: {total_time:.2f}s")
    print(f"Average per project: {total_time/len(projects):.3f}s")
    
    # Performance targets
    avg_per_project = total_time / len(projects)
    if avg_per_project < 2.0:
        print("🎯 EXCELLENT: Detection speed exceeds performance targets!")
    elif avg_per_project < 5.0:
        print("✅ GOOD: Detection speed meets performance targets")
    else:
        print("⚠️  SLOW: Detection speed below performance targets")
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    return results


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Project Detection System")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", help="Run specific test method")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.benchmark:
        run_detection_benchmark()
        return
    
    # Run unit tests
    print("🧪 Running Project Detection Test Suite")
    print("=" * 50)
    
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if args.test:
        # Run specific test
        suite.addTest(ProjectDetectionTestSuite(args.test))
    else:
        # Run all tests
        suite = loader.loadTestsFromTestCase(ProjectDetectionTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
        if not args.benchmark:
            print("\n💡 Tip: Run with --benchmark to test performance")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())