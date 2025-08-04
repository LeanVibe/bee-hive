import { test, expect } from '@playwright/test';

test.describe('Infrastructure Health Validation', () => {
  test('API server is running and healthy', async ({ request }) => {
    // Test 1: Basic API health
    const healthResponse = await request.get('/status');
    expect(healthResponse.ok()).toBeTruthy();
    
    const healthData = await healthResponse.json();
    console.log('Health check response:', JSON.stringify(healthData, null, 2));
    
    // Validate required fields
    expect(healthData).toHaveProperty('timestamp');
    expect(healthData).toHaveProperty('version');
    expect(healthData).toHaveProperty('components');
    
    // Validate database connection
    expect(healthData.components.database.connected).toBe(true);
    expect(healthData.components.database.tables).toBeGreaterThan(0);
    
    // Validate Redis connection
    expect(healthData.components.redis.connected).toBe(true);
  });

  test('API documentation is accessible', async ({ request }) => {
    const docsResponse = await request.get('/docs');
    expect(docsResponse.ok()).toBeTruthy();
    
    const openApiResponse = await request.get('/openapi.json');
    expect(openApiResponse.ok()).toBeTruthy();
    
    const openApiData = await openApiResponse.json();
    expect(openApiData).toHaveProperty('paths');
    
    // Count actual endpoints
    const endpointCount = Object.keys(openApiData.paths).length;
    console.log(`Discovered ${endpointCount} API endpoints`);
  });

  test('Database tables are properly migrated', async ({ request }) => {
    const healthResponse = await request.get('/status');
    const healthData = await healthResponse.json();
    
    // Verify we have the expected number of tables
    expect(healthData.components.database.tables).toBeGreaterThanOrEqual(100);
    expect(healthData.components.database.migrations_current).toBe(true);
  });
});