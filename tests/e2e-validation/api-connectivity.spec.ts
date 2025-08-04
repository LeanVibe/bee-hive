/**
 * API Connectivity Test - Check if PWA can access backend endpoints
 */

import { test, expect } from '@playwright/test';

test.describe('API Connectivity', () => {
  test('PWA can fetch data from backend API', async ({ page }) => {
    const consoleLogs: string[] = [];
    const networkRequests: string[] = [];
    const networkResponses: { url: string, status: number, ok: boolean }[] = [];
    
    // Capture console logs
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      console.log(`Console:`, text);
    });
    
    // Capture network requests
    page.on('request', request => {
      const url = request.url();
      if (url.includes('/dashboard/api/') || url.includes('/api/')) {
        networkRequests.push(url);
        console.log(`REQUEST: ${request.method()} ${url}`);
      }
    });
    
    // Capture network responses
    page.on('response', response => {
      const url = response.url();
      if (url.includes('/dashboard/api/') || url.includes('/api/')) {
        networkResponses.push({ 
          url, 
          status: response.status(), 
          ok: response.ok() 
        });
        console.log(`RESPONSE: ${response.status()} ${url}`);
      }
    });

    console.log('🔍 Testing PWA API connectivity...');
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for initialization
    await page.waitForTimeout(5000);
    
    // Test direct API call from browser context
    const directAPITest = await page.evaluate(async () => {
      try {
        console.log('Testing direct API call...');
        const response = await fetch('/dashboard/api/live-data');
        const data = await response.json();
        console.log('Direct API call success:', response.ok, response.status);
        return {
          success: response.ok,
          status: response.status,
          hasData: !!data,
          agentCount: data.agent_activities?.length || 0,
          projectCount: data.project_snapshots?.length || 0
        };
      } catch (error) {
        console.log('Direct API call error:', error.message);
        return {
          success: false,
          error: error.message
        };
      }
    });
    
    console.log('\n📊 API Test Results:');
    console.log('  - Direct API call successful:', directAPITest.success);
    console.log('  - Response status:', directAPITest.status);
    console.log('  - Has data:', directAPITest.hasData);
    console.log('  - Agent count:', directAPITest.agentCount);
    console.log('  - Project count:', directAPITest.projectCount);
    
    // Test backend adapter service
    const backendAdapterTest = await page.evaluate(async () => {
      try {
        const backendAdapter = window.appServices?.backendAdapter;
        if (!backendAdapter) {
          return { success: false, error: 'Backend adapter service not found' };
        }
        
        console.log('Testing backend adapter...');
        const liveData = await backendAdapter.getLiveData();
        console.log('Backend adapter success, data:', !!liveData);
        
        return {
          success: true,
          hasData: !!liveData,
          agentCount: liveData.agent_activities?.length || 0
        };
      } catch (error) {
        console.log('Backend adapter error:', error.message);
        return {
          success: false,
          error: error.message
        };
      }
    });
    
    console.log('\n🔗 Backend Adapter Test:');
    console.log('  - Backend adapter accessible:', backendAdapterTest.success);
    console.log('  - Has data:', backendAdapterTest.hasData);
    console.log('  - Agent count via adapter:', backendAdapterTest.agentCount);
    if (backendAdapterTest.error) {
      console.log('  - Error:', backendAdapterTest.error);
    }
    
    // Show network activity
    console.log('\n🌐 Network Activity:');
    console.log('  - API requests made:', networkRequests.length);
    networkRequests.forEach(req => console.log(`    - ${req}`));
    console.log('  - API responses received:', networkResponses.length);
    networkResponses.forEach(res => console.log(`    - ${res.status} ${res.url}`));
    
    // Look for backend adapter logs
    const backendLogs = consoleLogs.filter(log =>
      log.includes('Backend adapter') ||
      log.includes('backend-adapter') ||
      log.includes('getLiveData') ||
      log.includes('Syncing data')
    );
    
    console.log('\n📋 Backend Adapter Logs:');
    if (backendLogs.length > 0) {
      backendLogs.forEach(log => console.log(`  - ${log}`));
    } else {
      console.log('  - No backend adapter logs found');
    }
    
    // Basic connectivity assertions
    expect(directAPITest.success).toBe(true);
    expect(directAPITest.agentCount).toBeGreaterThan(0);
    expect(directAPITest.projectCount).toBeGreaterThan(0);
  });
});