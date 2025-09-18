import { FullConfig } from '@playwright/test';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

/**
 * Global setup for PWA E2E tests
 * Ensures clean environment and proper test data
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting PWA E2E Testing Global Setup...');

  // Ensure test results directory exists
  const testResultsDir = path.join(process.cwd(), 'test-results');
  if (!fs.existsSync(testResultsDir)) {
    fs.mkdirSync(testResultsDir, { recursive: true });
  }

  // Clean previous test artifacts
  console.log('🧹 Cleaning previous test artifacts...');
  const artifactDirs = ['screenshots', 'videos', 'traces'];
  
  artifactDirs.forEach(dir => {
    const fullPath = path.join(testResultsDir, dir);
    if (fs.existsSync(fullPath)) {
      fs.rmSync(fullPath, { recursive: true, force: true });
    }
    fs.mkdirSync(fullPath, { recursive: true });
  });

  // Verify PWA build exists or create it
  console.log('🏗️ Verifying PWA build...');
  const distDir = path.join(process.cwd(), 'dist');
  
  if (!fs.existsSync(distDir)) {
    console.log('📦 Building PWA for testing...');
    try {
      execSync('npm run build-no-check', { 
        stdio: 'inherit',
        cwd: process.cwd()
      });
    } catch (error) {
      console.error('❌ PWA build failed:', error);
      throw error;
    }
  }

  // Create test environment configuration
  const testConfig = {
    timestamp: new Date().toISOString(),
    environment: 'e2e-testing',
    pwaBuildExists: fs.existsSync(distDir),
    testSuites: {
      workflows: true,
      pwa: true,
      performance: true,
      accessibility: true
    }
  };

  // Save test configuration
  fs.writeFileSync(
    path.join(testResultsDir, 'test-config.json'),
    JSON.stringify(testConfig, null, 2)
  );

  // Initialize test database/state if needed
  console.log('🗄️ Initializing test state...');
  
  // Create mock data fixtures
  const fixturesDir = path.join(process.cwd(), 'tests/e2e/fixtures');
  if (!fs.existsSync(fixturesDir)) {
    fs.mkdirSync(fixturesDir, { recursive: true });
  }

  // Generate test data
  const testData = {
    users: [
      {
        id: 'test-user-1',
        email: 'admin@leanvibe.test',
        role: 'admin',
        permissions: ['read', 'write', 'admin']
      },
      {
        id: 'test-user-2', 
        email: 'user@leanvibe.test',
        role: 'user',
        permissions: ['read']
      }
    ],
    agents: [
      {
        id: 'test-agent-1',
        name: 'Test Agent Alpha',
        status: 'active',
        type: 'monitoring',
        configuration: {
          interval: 30,
          endpoints: ['http://localhost:8000/health']
        }
      },
      {
        id: 'test-agent-2',
        name: 'Test Agent Beta', 
        status: 'inactive',
        type: 'data-collection',
        configuration: {
          sources: ['database', 'api']
        }
      }
    ],
    tasks: [
      {
        id: 'test-task-1',
        title: 'Monitor System Health',
        description: 'Continuous monitoring of system health metrics',
        assignedAgent: 'test-agent-1',
        status: 'running',
        priority: 'high'
      },
      {
        id: 'test-task-2',
        title: 'Collect User Analytics',
        description: 'Gather user interaction analytics',
        assignedAgent: 'test-agent-2',
        status: 'pending',
        priority: 'medium'
      }
    ]
  };

  fs.writeFileSync(
    path.join(fixturesDir, 'test-data.json'),
    JSON.stringify(testData, null, 2)
  );

  console.log('✅ PWA E2E Testing Global Setup Complete');
  
  return testConfig;
}

export default globalSetup;