import { defineConfig, devices } from '@playwright/test'

/**
 * Enhanced Playwright Configuration for LeanVibe Agent Hive 2.0
 * Comprehensive E2E Testing Suite with PWA, Mobile, and Performance Testing
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './tests/e2e',
  
  /* Run tests in files in parallel */
  fullyParallel: true,
  
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  
  /* Reporter to use with enhanced reporting */
  reporter: [
    ['html', { 
      outputFolder: 'playwright-report',
      open: 'never' // Don't auto-open in CI
    }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['blob', { outputDir: 'test-results/blob-report' }],
    // Add GitHub Actions reporter for CI
    ...(process.env.CI ? [['github' as const]] : [])
  ],
  
  /* Shared settings for all projects */
  use: {
    /* Base URL for testing */
    baseURL: process.env.TEST_BASE_URL || 'http://localhost:5173',
    
    /* Backend URL for API testing */
    extraHTTPHeaders: {
      'X-Test-Backend-URL': process.env.TEST_BACKEND_URL || 'http://localhost:8000'
    },
    
    /* Enhanced tracing and debugging */
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',
    
    /* Enhanced screenshot settings */
    screenshot: {
      mode: 'only-on-failure',
      fullPage: true
    },
    
    /* Enhanced video recording */
    video: {
      mode: 'retain-on-failure',
      size: { width: 1280, height: 720 }
    },
    
    /* Timeouts optimized for real-world conditions */
    actionTimeout: 15 * 1000, // Increased for real API calls
    navigationTimeout: 30 * 1000,
    
    /* Enhanced browser context */
    viewport: { width: 1280, height: 720 },
    ignoreHTTPSErrors: true,
    
    /* Performance and accessibility testing */
    launchOptions: {
      slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0
    }
  },

  /* Test projects with comprehensive device coverage */
  projects: [
    // Setup project for global configuration
    {
      name: 'setup',
      testMatch: '**/global-setup.ts',
      teardown: 'cleanup'
    },
    
    // Cleanup project
    {
      name: 'cleanup',
      testMatch: '**/global-teardown.ts'
    },

    // Smoke tests - critical path validation
    {
      name: 'smoke',
      testMatch: '**/smoke/**',
      use: { ...devices['Desktop Chrome'] },
      dependencies: ['setup']
    },
    
    // Desktop browsers
    {
      name: 'chromium',
      testIgnore: '**/smoke/**',
      use: { 
        ...devices['Desktop Chrome'],
        channel: 'chrome' // Use stable Chrome
      },
      dependencies: ['setup']
    },
    
    {
      name: 'firefox',
      testIgnore: '**/smoke/**',
      use: { ...devices['Desktop Firefox'] },
      dependencies: ['setup']
    },
    
    {
      name: 'webkit',
      testIgnore: '**/smoke/**',
      use: { ...devices['Desktop Safari'] },
      dependencies: ['setup']
    },
    
    {
      name: 'Microsoft Edge',
      testIgnore: '**/smoke/**',
      use: { 
        ...devices['Desktop Edge'], 
        channel: 'msedge' 
      },
      dependencies: ['setup']
    },
    
    // Mobile devices - PWA focus
    {
      name: 'Mobile Chrome',
      testMatch: ['**/mobile/**', '**/pwa/**', '**/responsive/**'],
      use: { 
        ...devices['Pixel 5'],
        hasTouch: true
      },
      dependencies: ['setup']
    },
    
    {
      name: 'Mobile Safari',
      testMatch: ['**/mobile/**', '**/pwa/**', '**/responsive/**'],
      use: { 
        ...devices['iPhone 12'],
        hasTouch: true
      },
      dependencies: ['setup']
    },
    
    {
      name: 'iPhone 13 Pro',
      testMatch: ['**/mobile/**', '**/pwa/**'],
      use: { 
        ...devices['iPhone 13 Pro'],
        hasTouch: true
      },
      dependencies: ['setup']
    },
    
    // Tablet testing
    {
      name: 'iPad',
      testMatch: ['**/tablet/**', '**/responsive/**'],
      use: { 
        ...devices['iPad Pro'],
        hasTouch: true
      },
      dependencies: ['setup']
    },
    
    {
      name: 'iPad landscape',
      testMatch: ['**/tablet/**', '**/responsive/**'],
      use: { 
        ...devices['iPad Pro landscape'],
        hasTouch: true
      },
      dependencies: ['setup']
    },
    
    // Performance testing project
    {
      name: 'performance',
      testMatch: '**/performance/**',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows'
          ]
        }
      },
      dependencies: ['setup']
    },
    
    // Accessibility testing project
    {
      name: 'accessibility',
      testMatch: '**/accessibility/**',
      use: {
        ...devices['Desktop Chrome'],
        reducedMotion: 'reduce',
        forcedColors: 'active'
      },
      dependencies: ['setup']
    },
    
    // Visual regression testing
    {
      name: 'visual-regression',
      testMatch: '**/visual/**',
      use: {
        ...devices['Desktop Chrome'],
        // Disable animations for consistent screenshots
        reducedMotion: 'reduce'
      },
      dependencies: ['setup']
    }
  ],

  /* Global setup and teardown */
  globalSetup: require.resolve('./tests/fixtures/global-setup.ts'),
  globalTeardown: require.resolve('./tests/fixtures/global-teardown.ts'),

  /* Enhanced web server configuration */
  webServer: [
    // Frontend dev server
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
      env: {
        NODE_ENV: 'test'
      }
    },
    // Backend server (conditional)
    ...(process.env.TEST_WITH_BACKEND ? [{
      command: 'cd .. && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000',
      url: 'http://localhost:8000/health',
      reuseExistingServer: true,
      timeout: 60 * 1000
    }] : [])
  ],
  
  /* Enhanced timeouts for comprehensive testing */
  timeout: process.env.CI ? 90 * 1000 : 60 * 1000,
  
  /* Expect timeout for assertions */
  expect: {
    timeout: 15 * 1000,
    // Enhanced screenshot comparison
    threshold: 0.2,
    // Animation handling
    animations: 'disabled'
  },
  
  /* Output directory for test artifacts */
  outputDir: 'test-results/',
  
  /* Enhanced failure handling */
  maxFailures: process.env.CI ? 10 : undefined,
  
  /* Metadata for test organization */
  metadata: {
    testSuite: 'LeanVibe Agent Hive 2.0 E2E Tests',
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    backend: process.env.TEST_BACKEND_URL || 'http://localhost:8000',
    frontend: process.env.TEST_BASE_URL || 'http://localhost:5173'
  }
})