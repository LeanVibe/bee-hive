import { test, expect, Page } from '@playwright/test';
import { EvidenceCollector } from '../../utils/evidence-collector';
import { HealthChecker } from '../../utils/health-checker';

/**
 * Infrastructure Health Check Validation
 * 
 * Validates the core infrastructure claims:
 * - PostgreSQL database with pgvector extension
 * - Redis with streams functionality  
 * - FastAPI application health
 * - Docker Compose service orchestration
 */

test.describe('Infrastructure Health Validation', () => {
  let evidenceCollector: EvidenceCollector;
  let healthChecker: HealthChecker;
  
  test.beforeEach(async ({ page }) => {
    evidenceCollector = new EvidenceCollector(page, 'infrastructure');
    healthChecker = new HealthChecker();
    
    await evidenceCollector.startCollection('infrastructure-health-check');
  });

  test.afterEach(async () => {
    await evidenceCollector.finishCollection();
  });

  test('FastAPI Health Endpoint Returns Complete Status', async ({ page }) => {
    // Navigate to health endpoint
    const response = await page.goto('/health');
    expect(response?.status()).toBe(200);
    
    // Get health check response
    const healthData = await page.evaluate(async () => {
      const response = await fetch('/health');
      return response.json();
    });
    
    // Validate health response structure
    expect(healthData).toHaveProperty('status');
    expect(healthData).toHaveProperty('version', '2.0.0');
    expect(healthData).toHaveProperty('components');
    expect(healthData).toHaveProperty('summary');
    
    // Validate component health
    const components = healthData.components;
    expect(components).toHaveProperty('database');
    expect(components).toHaveProperty('redis');
    expect(components).toHaveProperty('orchestrator');
    expect(components).toHaveProperty('observability');
    
    // Validate database component
    expect(components.database.status).toBe('healthy');
    expect(components.database).toHaveProperty('details');
    expect(components.database).toHaveProperty('response_time_ms');
    
    // Validate Redis component  
    expect(components.redis.status).toBe('healthy');
    expect(components.redis).toHaveProperty('details');
    expect(components.redis).toHaveProperty('response_time_ms');
    
    // Validate orchestrator component
    expect(components.orchestrator.status).toBe('healthy');
    expect(components.orchestrator).toHaveProperty('active_agents');
    expect(components.orchestrator.active_agents).toBeGreaterThanOrEqual(0);
    
    // Collect evidence
    await evidenceCollector.captureApiResponse('/health', healthData);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('health-endpoint') });
    
    // Log validation success
    console.log('✅ Health check validation passed:', {
      status: healthData.status,
      componentsHealthy: healthData.summary.healthy,
      componentsUnhealthy: healthData.summary.unhealthy,
      totalComponents: healthData.summary.total
    });
  });

  test('PostgreSQL Database Connection and Schema Validation', async ({ page }) => {
    // Test database connectivity through system status endpoint
    const statusResponse = await page.evaluate(async () => {
      const response = await fetch('/status');
      return response.json();
    });
    
    expect(statusResponse.components.database.connected).toBe(true);
    expect(statusResponse.components.database.tables).toBeGreaterThan(0);
    expect(statusResponse.components.database.migrations_current).toBe(true);
    
    // Validate database tables exist by checking API endpoints that depend on them
    const agentsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/agents');
      return { status: response.status, ok: response.ok };
    });
    
    expect(agentsResponse.status).toBe(200);
    
    // Collect evidence
    await evidenceCollector.captureApiResponse('/status', statusResponse);
    await evidenceCollector.captureApiResponse('/api/v1/agents', agentsResponse);
    
    console.log('✅ PostgreSQL validation passed:', {
      connected: statusResponse.components.database.connected,
      tables: statusResponse.components.database.tables,
      migrationsCurrent: statusResponse.components.database.migrations_current
    });
  });

  test('Redis Connection and Streams Functionality', async ({ page }) => {
    // Test Redis connectivity through system status endpoint
    const statusResponse = await page.evaluate(async () => {
      const response = await fetch('/status');
      return response.json();
    });
    
    expect(statusResponse.components.redis.connected).toBe(true);
    expect(statusResponse.components.redis).toHaveProperty('memory_used');
    expect(statusResponse.components.redis.streams_active).toBe(true);
    
    // Test Redis streams by checking agent events endpoint
    const eventsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/events/agent');
      return { status: response.status, ok: response.ok };
    });
    
    // Redis streams should be accessible (even if no events yet)
    expect([200, 404].includes(eventsResponse.status)).toBe(true);
    
    // Collect evidence
    await evidenceCollector.captureApiResponse('/status-redis', statusResponse.components.redis);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('redis-status') });
    
    console.log('✅ Redis validation passed:', {
      connected: statusResponse.components.redis.connected,
      memoryUsed: statusResponse.components.redis.memory_used,
      streamsActive: statusResponse.components.redis.streams_active
    });
  });

  test('Docker Compose Services Running Validation', async ({ page }) => {
    // Test that all expected services are accessible
    const serviceChecks = [
      { name: 'FastAPI', url: '/health', expectedStatus: 200 },
      { name: 'Prometheus Metrics', url: '/metrics', expectedStatus: 200 },
      { name: 'API Documentation', url: '/docs', expectedStatus: 200 }
    ];
    
    const serviceResults = [];
    
    for (const service of serviceChecks) {
      const response = await page.goto(service.url);
      const status = response?.status() || 0;
      
      serviceResults.push({
        service: service.name,
        url: service.url,
        status: status,
        healthy: status === service.expectedStatus
      });
      
      expect(status).toBe(service.expectedStatus);
    }
    
    // Collect evidence
    await evidenceCollector.captureData('docker-services-status', serviceResults);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('services-validation') });
    
    console.log('✅ Docker services validation passed:', serviceResults);
  });

  test('System Metrics and Monitoring Validation', async ({ page }) => {
    // Test Prometheus metrics endpoint
    const metricsResponse = await page.goto('/metrics');
    expect(metricsResponse?.status()).toBe(200);
    
    const metricsText = await page.textContent('body');
    expect(metricsText).toContain('leanvibe_health_status');
    expect(metricsText).toContain('leanvibe_uptime_seconds');
    
    // Test system status with detailed information
    const systemStatus = await page.evaluate(async () => {
      const response = await fetch('/status');
      return response.json();
    });
    
    expect(systemStatus).toHaveProperty('timestamp');
    expect(systemStatus).toHaveProperty('version', '2.0.0');
    expect(systemStatus).toHaveProperty('environment');
    expect(systemStatus).toHaveProperty('components');
    
    // Collect evidence
    await evidenceCollector.captureData('prometheus-metrics', { 
      available: true, 
      containsHealthMetrics: metricsText?.includes('leanvibe_health_status'),
      containsUptimeMetrics: metricsText?.includes('leanvibe_uptime_seconds')
    });
    await evidenceCollector.captureApiResponse('/status', systemStatus);
    
    console.log('✅ Monitoring validation passed:', {
      metricsEndpoint: metricsResponse?.status() === 200,
      systemStatusAvailable: true,
      version: systemStatus.version
    });
  });

  test('Infrastructure Performance Benchmarks', async ({ page }) => {
    const performanceResults = [];
    
    // Benchmark health endpoint response time
    const healthStartTime = Date.now();
    const healthResponse = await page.goto('/health');
    const healthEndTime = Date.now();
    const healthResponseTime = healthEndTime - healthStartTime;
    
    expect(healthResponse?.status()).toBe(200);
    expect(healthResponseTime).toBeLessThan(5000); // Less than 5 seconds
    
    performanceResults.push({
      endpoint: '/health',
      responseTime: healthResponseTime,
      status: healthResponse?.status()
    });
    
    // Benchmark status endpoint response time
    const statusStartTime = Date.now();
    const statusResponse = await page.evaluate(async () => {
      const startTime = Date.now();
      const response = await fetch('/status');
      const data = await response.json();
      const endTime = Date.now();
      return { data, responseTime: endTime - startTime, status: response.status };
    });
    
    expect(statusResponse.status).toBe(200);
    expect(statusResponse.responseTime).toBeLessThan(3000); // Less than 3 seconds
    
    performanceResults.push({
      endpoint: '/status',
      responseTime: statusResponse.responseTime,
      status: statusResponse.status
    });
    
    // Benchmark metrics endpoint response time
    const metricsStartTime = Date.now();
    const metricsResponse = await page.goto('/metrics');
    const metricsEndTime = Date.now();
    const metricsResponseTime = metricsEndTime - metricsStartTime;
    
    expect(metricsResponse?.status()).toBe(200);
    expect(metricsResponseTime).toBeLessThan(2000); // Less than 2 seconds
    
    performanceResults.push({
      endpoint: '/metrics',
      responseTime: metricsResponseTime,
      status: metricsResponse?.status()
    });
    
    // Collect evidence
    await evidenceCollector.captureData('performance-benchmarks', performanceResults);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('performance-validation') });
    
    console.log('✅ Performance validation passed:', performanceResults);
  });
});