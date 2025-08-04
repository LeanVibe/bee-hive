import { chromium, FullConfig } from '@playwright/test'

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global test setup...')
  
  // Launch browser to verify the application is running
  const browser = await chromium.launch()
  const page = await browser.newPage()
  
  try {
    // Wait for the application to be available
    console.log('⏳ Waiting for application to be available...')
    await page.goto(config.projects[0].use.baseURL || 'http://localhost:3001', {
      waitUntil: 'networkidle',
      timeout: 60000
    })
    
    // Verify the application loads properly - look for any content indicating the app loaded
    await page.waitForSelector('body', { timeout: 10000 })
    
    // Check if we can find the main app content or title
    const hasAppContent = await page.locator('h1, [class*="dashboard"], [class*="agent"], main, app-root, #app').count() > 0
    if (!hasAppContent) {
      // Try to wait for any of the expected app elements
      await page.waitForSelector('h1, [class*="dashboard"], [class*="agent"], main, app-root, #app', { timeout: 5000 })
    }
    
    console.log('✅ Application is available and responding')
    
    // Set up any global test data or configuration here
    console.log('🔧 Setting up test environment...')
    
    // Example: Clear any existing test data, set up mock data, etc.
    // This could include API calls to reset test databases or cache
    
    console.log('✅ Global setup completed successfully')
    
  } catch (error) {
    console.error('❌ Global setup failed:', error)
    throw error
  } finally {
    await browser.close()
  }
}

export default globalSetup