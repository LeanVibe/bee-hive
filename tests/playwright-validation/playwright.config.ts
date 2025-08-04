import { defineConfig, devices } from '@playwright/test';

/**
 * Comprehensive Playwright Configuration for LeanVibe Agent Hive 2.0 Validation
 * 
 * This configuration is designed for thorough system validation with evidence collection.
 * Every test is configured to capture maximum evidence for trust-building validation.
 */

export default defineConfig({
  testDir: './tests',
  
  /* Parallel execution disabled for evidence collection reliability */
  fullyParallel: false,
  workers: 1,
  
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  
  /* Retry configuration for reliability */
  retries: process.env.CI ? 2 : 1,
  
  /* Reporter configuration for comprehensive evidence collection */
  reporter: [
    ['html', { 
      outputFolder: 'reports/playwright-report',
      open: 'never'
    }],
    ['json', { 
      outputFile: 'reports/validation-results.json' 
    }],
    ['junit', { 
      outputFile: 'reports/validation-results.xml' 
    }],
    ['line'],
    ['./utils/evidence-reporter.ts']
  ],
  
  /* Shared settings for all tests */
  use: {
    /* Maximum evidence collection */
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
    
    /* Extended timeouts for complex system validation */
    actionTimeout: 30000,
    navigationTimeout: 60000,
    
    /* Viewport for consistent dashboard testing */
    viewport: { width: 1920, height: 1080 },
    
    /* Ignore HTTPS errors for local development */
    ignoreHTTPSErrors: true,
    
    /* Additional context options for evidence collection */
    extraHTTPHeaders: {
      'X-Test-Suite': 'LeanVibe-Validation',
      'X-Evidence-Collection': 'Enabled'
    }
  },

  /* Test configuration by category */
  projects: [
    {
      name: 'infrastructure-validation',
      testDir: './tests/infrastructure',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:8000',
        screenshot: 'only-on-failure',
        video: 'on-first-retry'
      },
      timeout: 120000 // 2 minutes for infrastructure checks
    },
    
    {
      name: 'multi-agent-validation',
      testDir: './tests/multi-agent',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:8000',
        screenshot: 'on-first-retry',
        video: 'on-first-retry'
      },
      timeout: 180000, // 3 minutes for agent spawning and coordination
      dependencies: ['infrastructure-validation']
    },
    
    {
      name: 'dashboard-validation',
      testDir: './tests/dashboard',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:3002',
        screenshot: 'on',
        video: 'on'
      },
      timeout: 90000, // 1.5 minutes for dashboard testing
      dependencies: ['multi-agent-validation']
    },
    
    {
      name: 'api-validation',
      testDir: './tests/api',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:8000',
        screenshot: 'only-on-failure',
        video: 'off' // API tests don't need video
      },
      timeout: 300000, // 5 minutes for comprehensive API testing
      dependencies: ['infrastructure-validation']
    },
    
    {
      name: 'workflow-validation',
      testDir: './tests/workflows',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:8000',
        screenshot: 'on',
        video: 'on'
      },
      timeout: 600000, // 10 minutes for autonomous workflow testing
      dependencies: ['multi-agent-validation', 'api-validation']
    },
    
    {
      name: 'integration-validation',
      testDir: './tests/integration',
      use: { 
        ...devices['Desktop Chrome'],
        baseURL: 'http://localhost:8000',
        screenshot: 'on',
        video: 'on'
      },
      timeout: 900000, // 15 minutes for end-to-end integration testing
      dependencies: ['dashboard-validation', 'workflow-validation']
    }
  ],

  /* Global setup and teardown */
  globalSetup: require.resolve('./utils/global-setup.ts'),
  globalTeardown: require.resolve('./utils/global-teardown.ts'),

  /* Test matching patterns */
  testMatch: ['**/*.spec.ts', '**/*.test.ts'],
  
  /* Output directories */
  outputDir: 'reports/test-artifacts',
  
  /* Web server configuration for local testing */
  webServer: [
    {
      command: 'cd ../.. && docker compose up -d',
      port: 8000,
      timeout: 300000, // 5 minutes for Docker Compose startup
      env: {
        NODE_ENV: 'test',
        LOG_LEVEL: 'DEBUG'
      }
    }
  ],

  /* Test environment variables */
  globalSetupEnv: {
    PLAYWRIGHT_TEST_BASE_URL: 'http://localhost:8000',
    PLAYWRIGHT_DASHBOARD_URL: 'http://localhost:3002',
    EVIDENCE_COLLECTION_ENABLED: 'true',
    TEST_TIMEOUT_MULTIPLIER: '2'
  },

  /* Expect configuration for assertions */
  expect: {
    timeout: 15000,
    toMatchSnapshot: {
      mode: 'chromium',
      threshold: 0.2
    }
  },

  /* Maximum failures before stopping */
  maxFailures: process.env.CI ? 10 : 0,

  /* Update snapshots on local development */
  updateSnapshots: process.env.CI ? 'none' : 'missing'
});