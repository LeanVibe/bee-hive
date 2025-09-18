import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for comprehensive PWA E2E testing
 * Supporting Level 7 (Final Level) of the Testing Pyramid
 */
export default defineConfig({
  // Test directory configuration
  testDir: './tests/e2e',
  
  // Global test patterns
  testMatch: [
    '**/workflows/**/*.spec.ts',
    '**/pwa/**/*.spec.ts', 
    '**/performance/**/*.spec.ts',
    '**/accessibility/**/*.spec.ts'
  ],

  // Parallel execution for faster test runs
  fullyParallel: true,
  
  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,
  
  // Retry on CI only
  retries: process.env.CI ? 2 : 0,
  
  // Opt out of parallel tests on CI for more stable runs
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration for comprehensive test output
  reporter: [
    ['html'],
    ['junit', { outputFile: 'test-results/e2e-results.xml' }],
    ['json', { outputFile: 'test-results/e2e-results.json' }],
    ['list']
  ],
  
  // Global test configuration
  use: {
    // Base URL of your PWA
    baseURL: 'http://localhost:5173',
    
    // Collect trace when retrying the failed test
    trace: 'on-first-retry',
    
    // Take screenshot on failure
    screenshot: 'only-on-failure',
    
    // Record video on retry
    video: 'retain-on-failure',
    
    // Global timeout for all tests
    actionTimeout: 10000,
    navigationTimeout: 30000,
    
    // PWA testing - ignore HTTPS errors in development
    ignoreHTTPSErrors: true,
    
    // Accept downloads automatically
    acceptDownloads: true,
  },

  // Project configuration for cross-browser testing
  projects: [
    // Desktop Browsers
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // PWA testing specific viewport
        viewport: { width: 1280, height: 720 }
      },
    },
    {
      name: 'firefox',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 1280, height: 720 }
      },
    },
    {
      name: 'webkit',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 1280, height: 720 }
      },
    },

    // Mobile Chrome for PWA mobile testing
    {
      name: 'Mobile Chrome',
      use: { 
        ...devices['Pixel 5'],
        // PWA mobile viewport
        viewport: { width: 393, height: 851 }
      },
    },
    
    // Mobile Safari for iOS PWA testing
    {
      name: 'Mobile Safari',
      use: { 
        ...devices['iPhone 12'],
        viewport: { width: 390, height: 844 }
      },
    },

    // Tablet testing for hybrid experiences
    {
      name: 'Tablet',
      use: {
        ...devices['iPad'],
        viewport: { width: 768, height: 1024 }
      },
    },

    // High DPI testing for PWA display quality
    {
      name: 'High DPI',
      use: {
        ...devices['Desktop Chrome HiDPI'],
        viewport: { width: 1920, height: 1080 }
      },
    },

    // PWA Offline Testing
    {
      name: 'Offline PWA',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        // Will be configured for offline testing in tests
      },
    }
  ],

  // Development server configuration
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000, // 2 minutes
  },

  // Global setup and teardown
  globalSetup: './tests/e2e/utils/global-setup.ts',
  globalTeardown: './tests/e2e/utils/global-teardown.ts',

  // Test timeout
  timeout: 30 * 1000, // 30 seconds per test

  // Expect timeout for assertions
  expect: {
    timeout: 5000
  },

  // Output directory
  outputDir: 'test-results/',
});