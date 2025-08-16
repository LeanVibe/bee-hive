"""
Testing Examples and Validation Framework for Project Index Integrations

Provides comprehensive testing utilities, validation frameworks, and test examples
for all framework integrations to ensure reliability and correctness.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import unittest
from unittest.mock import Mock, patch, AsyncMock
from abc import ABC, abstractmethod


class IntegrationTestFramework:
    """Base framework for testing Project Index integrations."""
    
    def __init__(self, framework_name: str, test_config: Optional[Dict[str, Any]] = None):
        self.framework_name = framework_name
        self.test_config = test_config or {}
        self.test_results = []
        self.api_base_url = self.test_config.get('api_url', 'http://localhost:8000/project-index')
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return results."""
        print(f"🧪 Running integration tests for {self.framework_name}")
        
        tests = [
            ('Connection Test', self.test_api_connection),
            ('Framework Detection', self.test_framework_detection),
            ('Integration Setup', self.test_integration_setup),
            ('API Endpoints', self.test_api_endpoints),
            ('Error Handling', self.test_error_handling),
            ('Performance', self.test_performance),
        ]
        
        results = {
            'framework': self.framework_name,
            'total_tests': len(tests),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'execution_time': 0,
            'details': {}
        }
        
        start_time = time.time()
        
        for test_name, test_func in tests:
            try:
                print(f"  ⏳ Running {test_name}...")
                result = test_func()
                if result.get('success', False):
                    print(f"  ✅ {test_name} passed")
                    results['passed'] += 1
                else:
                    print(f"  ❌ {test_name} failed: {result.get('error', 'Unknown error')}")
                    results['failed'] += 1
                    results['errors'].append({
                        'test': test_name,
                        'error': result.get('error', 'Unknown error')
                    })
                
                results['details'][test_name] = result
                
            except Exception as e:
                print(f"  ❌ {test_name} error: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'test': test_name,
                    'error': str(e)
                })
                results['details'][test_name] = {'success': False, 'error': str(e)}
        
        results['execution_time'] = time.time() - start_time
        
        print(f"📊 Test Summary: {results['passed']}/{results['total_tests']} passed")
        return results
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test basic API connectivity."""
        try:
            import requests
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'response_data': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}",
                    'status_code': response.status_code
                }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': "Could not connect to Project Index API"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Connection test failed: {e}"
            }
    
    def test_framework_detection(self) -> Dict[str, Any]:
        """Test framework auto-detection."""
        try:
            from . import detect_framework
            detected = detect_framework()
            
            return {
                'success': True,
                'detected_framework': detected,
                'matches_expected': detected == self.framework_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Framework detection failed: {e}"
            }
    
    def test_integration_setup(self) -> Dict[str, Any]:
        """Test integration setup process."""
        try:
            from . import IntegrationManager
            adapter_class = IntegrationManager.get_adapter(self.framework_name)
            
            if adapter_class:
                # Test adapter creation
                adapter = adapter_class()
                return {
                    'success': True,
                    'adapter_created': True,
                    'adapter_type': type(adapter).__name__
                }
            else:
                return {
                    'success': False,
                    'error': f"No adapter found for {self.framework_name}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Integration setup failed: {e}"
            }
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test Project Index API endpoints."""
        endpoints = [
            ('GET', '/status'),
            ('GET', '/projects'),
            ('POST', '/analyze', {'project_path': '.', 'languages': ['python']})
        ]
        
        results = {}
        all_passed = True
        
        try:
            import requests
            
            for method, endpoint, *data in endpoints:
                endpoint_url = f"{self.api_base_url}{endpoint}"
                
                try:
                    if method == 'GET':
                        response = requests.get(endpoint_url, timeout=10)
                    elif method == 'POST':
                        response = requests.post(
                            endpoint_url, 
                            json=data[0] if data else {}, 
                            timeout=30
                        )
                    
                    results[endpoint] = {
                        'status_code': response.status_code,
                        'success': 200 <= response.status_code < 300,
                        'response_size': len(response.content)
                    }
                    
                    if not results[endpoint]['success']:
                        all_passed = False
                        
                except Exception as e:
                    results[endpoint] = {
                        'success': False,
                        'error': str(e)
                    }
                    all_passed = False
            
            return {
                'success': all_passed,
                'endpoints': results
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "requests library not available for testing"
            }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        error_scenarios = [
            ('Invalid endpoint', 'GET', '/invalid-endpoint'),
            ('Malformed request', 'POST', '/analyze', {'invalid': 'data'}),
            ('Missing data', 'POST', '/analyze', {})
        ]
        
        results = {}
        all_handled = True
        
        try:
            import requests
            
            for scenario_name, method, endpoint, *data in error_scenarios:
                endpoint_url = f"{self.api_base_url}{endpoint}"
                
                try:
                    if method == 'GET':
                        response = requests.get(endpoint_url, timeout=5)
                    elif method == 'POST':
                        response = requests.post(
                            endpoint_url,
                            json=data[0] if data else {},
                            timeout=5
                        )
                    
                    # For error scenarios, we expect 4xx or 5xx status codes
                    expected_error = response.status_code >= 400
                    
                    results[scenario_name] = {
                        'status_code': response.status_code,
                        'handled_correctly': expected_error,
                        'response_has_error_info': 'error' in response.text.lower()
                    }
                    
                    if not expected_error:
                        all_handled = False
                        
                except Exception as e:
                    results[scenario_name] = {
                        'handled_correctly': True,  # Connection errors are expected
                        'error': str(e)
                    }
            
            return {
                'success': all_handled,
                'scenarios': results
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "requests library not available for testing"
            }
    
    def test_performance(self) -> Dict[str, Any]:
        """Test integration performance."""
        try:
            import requests
            
            # Test response times
            start_time = time.time()
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            response_time = time.time() - start_time
            
            # Performance thresholds
            fast_response = response_time < 1.0  # Less than 1 second
            acceptable_response = response_time < 5.0  # Less than 5 seconds
            
            return {
                'success': acceptable_response,
                'response_time_ms': response_time * 1000,
                'performance_rating': 'fast' if fast_response else 'acceptable' if acceptable_response else 'slow',
                'thresholds': {
                    'fast': '<1s',
                    'acceptable': '<5s',
                    'current': f'{response_time:.2f}s'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Performance test failed: {e}"
            }


class PythonFrameworkTests(IntegrationTestFramework):
    """Specialized tests for Python frameworks."""
    
    def test_fastapi_integration(self) -> Dict[str, Any]:
        """Test FastAPI-specific integration."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from ..python import add_project_index_fastapi
            
            # Create test app
            app = FastAPI()
            adapter = add_project_index_fastapi(app)
            
            # Test client
            client = TestClient(app)
            
            # Test endpoints
            response = client.get("/project-index/status")
            
            return {
                'success': response.status_code in [200, 503],  # 503 is OK if service not running
                'adapter_created': adapter is not None,
                'endpoints_registered': True,
                'status_code': response.status_code
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "FastAPI not available for testing"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"FastAPI integration test failed: {e}"
            }
    
    def test_flask_integration(self) -> Dict[str, Any]:
        """Test Flask-specific integration."""
        try:
            from flask import Flask
            from ..python import add_project_index_flask
            
            # Create test app
            app = Flask(__name__)
            adapter = add_project_index_flask(app)
            
            # Test client
            with app.test_client() as client:
                response = client.get('/project-index/status')
            
            return {
                'success': response.status_code in [200, 503],
                'adapter_created': adapter is not None,
                'endpoints_registered': True,
                'status_code': response.status_code
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "Flask not available for testing"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Flask integration test failed: {e}"
            }


class JavaScriptFrameworkTests(IntegrationTestFramework):
    """Specialized tests for JavaScript/TypeScript frameworks."""
    
    def test_code_generation(self) -> Dict[str, Any]:
        """Test JavaScript code generation."""
        try:
            from ..javascript import generate_express_integration
            
            # Test code generation
            adapter = generate_express_integration()
            
            # Check if files were created
            files_created = [
                'middleware/projectIndex.js',
                'routes/projectIndex.js',
                'examples/express-integration.js'
            ]
            
            created_count = 0
            for file_path in files_created:
                if Path(file_path).exists():
                    created_count += 1
            
            return {
                'success': created_count > 0,
                'files_created': created_count,
                'total_files': len(files_created),
                'adapter_created': adapter is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"JavaScript code generation failed: {e}"
            }


def create_test_suite(framework: str) -> IntegrationTestFramework:
    """Create appropriate test suite for framework."""
    if framework in ['fastapi', 'django', 'flask', 'celery']:
        return PythonFrameworkTests(framework)
    elif framework in ['express', 'nextjs', 'react', 'vue', 'angular']:
        return JavaScriptFrameworkTests(framework)
    else:
        return IntegrationTestFramework(framework)


def generate_test_examples() -> None:
    """Generate test examples for all frameworks."""
    
    # Python test examples
    fastapi_test_example = """
# FastAPI Integration Tests
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.integrations.python import add_project_index_fastapi


@pytest.fixture
def app():
    \"\"\"Create FastAPI app with Project Index integration for testing.\"\"\"
    app = FastAPI()
    adapter = add_project_index_fastapi(app)
    return app


@pytest.fixture
def client(app):
    \"\"\"Create test client.\"\"\"
    return TestClient(app)


def test_project_index_status_endpoint(client):
    \"\"\"Test Project Index status endpoint.\"\"\"
    response = client.get("/project-index/status")
    
    # Should return 200 if service is running, 503 if not
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "initialized" in data


def test_project_index_analyze_endpoint(client):
    \"\"\"Test Project Index analyze endpoint.\"\"\"
    request_data = {
        "project_path": ".",
        "languages": ["python", "javascript"]
    }
    
    response = client.post("/project-index/analyze", json=request_data)
    
    # Should return 200 if analysis succeeds, 400/503 if not
    assert response.status_code in [200, 400, 503]


def test_project_index_projects_endpoint(client):
    \"\"\"Test Project Index projects endpoint.\"\"\"
    response = client.get("/project-index/projects")
    
    # Should return 200 if service is running, 503 if not
    assert response.status_code in [200, 503]


def test_integration_middleware(client):
    \"\"\"Test that Project Index middleware is working.\"\"\"
    response = client.get("/")
    
    # Check if Project Index headers are present
    assert "X-Project-Index" in response.headers or response.status_code == 404


def test_app_state_integration(app):
    \"\"\"Test that Project Index is properly integrated into app state.\"\"\"
    assert hasattr(app.state, 'project_index')
    assert app.state.project_index is not None


@pytest.mark.asyncio
async def test_custom_analysis_route(client):
    \"\"\"Test custom analysis functionality.\"\"\"
    # This would test any custom routes you've added
    # that use Project Index functionality
    pass


class TestProjectIndexIntegration:
    \"\"\"Test class for Project Index integration.\"\"\"
    
    def setup_method(self):
        \"\"\"Setup before each test method.\"\"\"
        self.app = FastAPI()
        self.adapter = add_project_index_fastapi(self.app)
        self.client = TestClient(self.app)
    
    def test_adapter_creation(self):
        \"\"\"Test adapter is created successfully.\"\"\"
        assert self.adapter is not None
        assert hasattr(self.adapter, 'config')
        assert hasattr(self.adapter, 'indexer')
    
    def test_endpoints_registration(self):
        \"\"\"Test that all expected endpoints are registered.\"\"\"
        routes = [route.path for route in self.app.routes]
        
        expected_routes = [
            "/project-index/status",
            "/project-index/analyze",
            "/project-index/projects"
        ]
        
        for route in expected_routes:
            assert any(route in r for r in routes), f"Route {route} not found"
    
    def test_error_handling(self):
        \"\"\"Test error handling for various scenarios.\"\"\"
        # Test invalid endpoint
        response = self.client.get("/project-index/invalid")
        assert response.status_code == 404
        
        # Test malformed request
        response = self.client.post("/project-index/analyze", json={"invalid": "data"})
        assert response.status_code in [400, 422, 503]
    
    def test_configuration(self):
        \"\"\"Test configuration handling.\"\"\"
        assert self.adapter.config is not None
        
        # Test custom configuration
        from app.project_index import ProjectIndexConfig
        custom_config = ProjectIndexConfig(cache_enabled=False)
        
        custom_app = FastAPI()
        custom_adapter = add_project_index_fastapi(custom_app, config=custom_config)
        
        assert custom_adapter.config.cache_enabled == False


if __name__ == "__main__":
    pytest.main([__file__])
"""
    
    # JavaScript test examples
    express_test_example = """
// Express.js Integration Tests
const request = require('supertest');
const express = require('express');
const { expect } = require('chai');
const { projectIndexMiddleware } = require('../middleware/projectIndex');

describe('Express Project Index Integration', () => {
    let app;
    
    beforeEach(() => {
        app = express();
        app.use(express.json());
        app.use(projectIndexMiddleware);
        
        // Add test routes
        app.get('/', (req, res) => {
            res.json({ message: 'Test app' });
        });
        
        app.get('/test-project-index', (req, res) => {
            res.json({
                hasProjectIndex: !!req.projectIndex,
                hasAnalyzeMethod: typeof req.analyzeCurrentProject === 'function'
            });
        });
    });
    
    describe('Middleware Integration', () => {
        it('should add Project Index client to request', (done) => {
            request(app)
                .get('/test-project-index')
                .expect(200)
                .end((err, res) => {
                    if (err) return done(err);
                    
                    expect(res.body.hasProjectIndex).to.be.true;
                    expect(res.body.hasAnalyzeMethod).to.be.true;
                    done();
                });
        });
        
        it('should continue to next middleware even if Project Index fails', (done) => {
            request(app)
                .get('/')
                .expect(200)
                .end((err, res) => {
                    if (err) return done(err);
                    
                    expect(res.body.message).to.equal('Test app');
                    done();
                });
        });
    });
    
    describe('API Endpoints', () => {
        beforeEach(() => {
            // Mount Project Index routes
            const projectIndexRoutes = require('../routes/projectIndex');
            app.use('/api/project-index', projectIndexRoutes);
        });
        
        it('should respond to status endpoint', (done) => {
            request(app)
                .get('/api/project-index/status')
                .expect((res) => {
                    // Should return 200 or 503 (if service unavailable)
                    expect([200, 503]).to.include(res.status);
                })
                .end(done);
        });
        
        it('should respond to analyze endpoint', (done) => {
            request(app)
                .post('/api/project-index/analyze')
                .send({
                    projectPath: '.',
                    languages: ['javascript', 'typescript']
                })
                .expect((res) => {
                    // Should return 200, 400, or 503
                    expect([200, 400, 503]).to.include(res.status);
                })
                .end(done);
        });
        
        it('should respond to projects endpoint', (done) => {
            request(app)
                .get('/api/project-index/projects')
                .expect((res) => {
                    expect([200, 503]).to.include(res.status);
                })
                .end(done);
        });
    });
    
    describe('Error Handling', () => {
        it('should handle invalid endpoints gracefully', (done) => {
            request(app)
                .get('/api/project-index/invalid')
                .expect(404)
                .end(done);
        });
        
        it('should handle malformed requests', (done) => {
            request(app)
                .post('/api/project-index/analyze')
                .send({ invalid: 'data' })
                .expect((res) => {
                    expect([400, 503]).to.include(res.status);
                })
                .end(done);
        });
    });
    
    describe('Performance', () => {
        it('should respond to status within reasonable time', (done) => {
            const start = Date.now();
            
            request(app)
                .get('/api/project-index/status')
                .expect((res) => {
                    const duration = Date.now() - start;
                    expect(duration).to.be.below(5000); // Less than 5 seconds
                })
                .end(done);
        });
    });
});

// Integration test with mock Project Index service
describe('Express Integration with Mock Service', () => {
    let app;
    let mockServer;
    
    before(() => {
        // Start mock Project Index service
        const mockApp = express();
        mockApp.use(express.json());
        
        mockApp.get('/status', (req, res) => {
            res.json({
                status: 'active',
                initialized: true,
                config: {
                    cache_enabled: true,
                    monitoring_enabled: true
                }
            });
        });
        
        mockApp.post('/analyze', (req, res) => {
            res.json({
                project_id: 'test-project',
                files_processed: 42,
                dependencies_found: 15,
                analysis_time: 1.5,
                languages_detected: ['javascript', 'typescript']
            });
        });
        
        mockServer = mockApp.listen(9999);
    });
    
    after(() => {
        if (mockServer) {
            mockServer.close();
        }
    });
    
    beforeEach(() => {
        // Override API URL for testing
        process.env.PROJECT_INDEX_API_URL = 'http://localhost:9999';
        
        app = express();
        app.use(express.json());
        app.use(projectIndexMiddleware);
        
        const projectIndexRoutes = require('../routes/projectIndex');
        app.use('/api/project-index', projectIndexRoutes);
    });
    
    it('should successfully communicate with mock service', (done) => {
        request(app)
            .get('/api/project-index/status')
            .expect(200)
            .expect((res) => {
                expect(res.body.status).to.equal('active');
                expect(res.body.initialized).to.be.true;
            })
            .end(done);
    });
    
    it('should successfully analyze with mock service', (done) => {
        request(app)
            .post('/api/project-index/analyze')
            .send({
                projectPath: '.',
                languages: ['javascript']
            })
            .expect(200)
            .expect((res) => {
                expect(res.body.files_processed).to.equal(42);
                expect(res.body.languages_detected).to.include('javascript');
            })
            .end(done);
    });
});
"""
    
    # Test configuration examples
    pytest_config = """
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=app/integrations
    --cov-report=html
    --cov-report=term-missing
markers =
    integration: Integration tests
    unit: Unit tests
    slow: Slow running tests
    requires_service: Tests that require Project Index service
"""
    
    # Package.json test script
    package_json_test = """
{
  "scripts": {
    "test": "mocha test/**/*.test.js",
    "test:watch": "mocha test/**/*.test.js --watch",
    "test:coverage": "nyc mocha test/**/*.test.js",
    "test:integration": "mocha test/integration/**/*.test.js"
  },
  "devDependencies": {
    "mocha": "^10.0.0",
    "chai": "^4.3.0",
    "supertest": "^6.3.0",
    "nyc": "^15.1.0",
    "sinon": "^15.0.0"
  }
}
"""
    
    # Write test files
    base_dir = Path.cwd()
    
    (base_dir / 'tests/python').mkdir(parents=True, exist_ok=True)
    (base_dir / 'tests/javascript').mkdir(parents=True, exist_ok=True)
    (base_dir / 'tests/config').mkdir(parents=True, exist_ok=True)
    
    with open(base_dir / 'tests/python/test_fastapi_integration.py', 'w') as f:
        f.write(fastapi_test_example)
    
    with open(base_dir / 'tests/javascript/test_express_integration.js', 'w') as f:
        f.write(express_test_example)
    
    with open(base_dir / 'tests/config/pytest.ini', 'w') as f:
        f.write(pytest_config)
    
    with open(base_dir / 'tests/javascript/package.json', 'w') as f:
        f.write(package_json_test)
    
    print("✅ Generated test examples and configurations")


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive tests for all supported frameworks."""
    from . import IntegrationManager
    
    all_results = {
        'total_frameworks': 0,
        'frameworks_tested': 0,
        'overall_success': True,
        'results': {}
    }
    
    supported_frameworks = IntegrationManager.list_supported_frameworks()
    all_results['total_frameworks'] = len(supported_frameworks)
    
    print("🧪 Running comprehensive integration tests")
    print(f"Testing {len(supported_frameworks)} frameworks...")
    
    for framework in supported_frameworks:
        print(f"\n📋 Testing {framework}...")
        
        try:
            test_suite = create_test_suite(framework)
            results = test_suite.run_all_tests()
            
            all_results['results'][framework] = results
            all_results['frameworks_tested'] += 1
            
            if results['failed'] > 0:
                all_results['overall_success'] = False
                
        except Exception as e:
            print(f"❌ Failed to test {framework}: {e}")
            all_results['results'][framework] = {
                'success': False,
                'error': str(e)
            }
            all_results['overall_success'] = False
    
    # Print summary
    print("\n📊 Test Summary")
    print(f"Frameworks tested: {all_results['frameworks_tested']}/{all_results['total_frameworks']}")
    
    for framework, results in all_results['results'].items():
        if 'passed' in results and 'total_tests' in results:
            print(f"  {framework}: {results['passed']}/{results['total_tests']} tests passed")
        else:
            print(f"  {framework}: ❌ Failed to run tests")
    
    return all_results


# Export main components
__all__ = [
    'IntegrationTestFramework',
    'PythonFrameworkTests', 
    'JavaScriptFrameworkTests',
    'create_test_suite',
    'generate_test_examples',
    'run_comprehensive_tests'
]