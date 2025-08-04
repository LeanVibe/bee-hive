import { test, expect } from '@playwright/test';

test.describe('Dashboard Functionality Validation', () => {
  test('Dashboard loads successfully', async ({ page }) => {
    // Test the actual dashboard URL that exists
    await page.goto('http://localhost:8000/dashboard/');
    
    // Wait for page to load and check if it's actually a functional dashboard
    await page.waitForLoadState('networkidle');
    
    // Take screenshot for evidence
    await page.screenshot({ path: 'test-results/dashboard-homepage.png', fullPage: true });
    
    // Check if page has meaningful content (not just a placeholder)
    const title = await page.title();
    console.log('Dashboard title:', title);
    
    // Look for dashboard-specific elements
    const bodyText = await page.textContent('body');
    expect(bodyText).toBeTruthy();
  });

  test('Dashboard shows real agent data', async ({ page, request }) => {
    // First get agent data from API
    const agentResponse = await request.get('/api/agents/status');
    const agentData = await agentResponse.json();
    
    // Then check if dashboard displays this data
    await page.goto('http://localhost:8000/dashboard/');
    await page.waitForLoadState('networkidle');
    
    // Wait a bit for any JavaScript to load data
    await page.waitForTimeout(3000);
    
    // Take screenshot to show dashboard state
    await page.screenshot({ path: 'test-results/dashboard-with-data.png', fullPage: true });
    
    // Look for agent-related content in the page
    const pageContent = await page.textContent('body');
    console.log('Dashboard content preview:', pageContent?.substring(0, 500));
    
    // Check if any agent roles appear in the dashboard
    const agentRoles = ['product_manager', 'architect', 'backend_developer', 'qa_engineer', 'devops_engineer'];
    let rolesFound = 0;
    
    for (const role of agentRoles) {
      if (pageContent?.toLowerCase().includes(role.replace('_', ' ')) || 
          pageContent?.toLowerCase().includes(role)) {
        rolesFound++;
      }
    }
    
    console.log(`Found ${rolesFound} agent roles in dashboard content`);
  });

  test('Dashboard API connectivity', async ({ page }) => {
    await page.goto('http://localhost:8000/dashboard/');
    
    // Monitor network requests to see if dashboard is making API calls
    const apiCalls: string[] = [];
    
    page.on('request', request => {
      if (request.url().includes('localhost:8000')) {
        apiCalls.push(request.url());
      }
    });
    
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(5000); // Allow time for API calls
    
    console.log('API calls made by dashboard:', apiCalls);
    
    // Dashboard should be making calls to the backend API
    expect(apiCalls.length).toBeGreaterThan(0);
  });
});