import { Page } from '@playwright/test';

/**
 * Health Checker Utility
 * 
 * Validates infrastructure health and system component status.
 * Provides comprehensive health monitoring for database, Redis, APIs, and services.
 */

export interface ComponentHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'degraded' | 'unknown';
  responseTime?: number;
  details: any;
  lastCheck: string;
  error?: string;
}

export interface SystemHealthReport {
  overallStatus: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components: ComponentHealth[];
  summary: {
    healthy: number;
    unhealthy: number;
    degraded: number;
    total: number;
  };
  performance: {
    averageResponseTime: number;
    slowestComponent: string;
    fastestComponent: string;
  };
}

export class HealthChecker {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async checkSystemHealth(page: Page): Promise<SystemHealthReport> {
    const components: ComponentHealth[] = [];
    const timestamp = new Date().toISOString();

    // Check all system components
    components.push(await this.checkApiHealth(page));
    components.push(await this.checkDatabaseHealth(page));
    components.push(await this.checkRedisHealth(page));
    components.push(await this.checkOrchestratorHealth(page));
    components.push(await this.checkObservabilityHealth(page));
    components.push(await this.checkMetricsHealth(page));

    // Calculate summary
    const summary = {
      healthy: components.filter(c => c.status === 'healthy').length,
      unhealthy: components.filter(c => c.status === 'unhealthy').length,
      degraded: components.filter(c => c.status === 'degraded').length,
      total: components.length
    };

    // Calculate performance metrics
    const responseTimes = components
      .filter(c => c.responseTime !== undefined)
      .map(c => ({ name: c.name, time: c.responseTime! }));

    const averageResponseTime = responseTimes.length > 0
      ? responseTimes.reduce((sum, rt) => sum + rt.time, 0) / responseTimes.length
      : 0;

    const slowestComponent = responseTimes.length > 0
      ? responseTimes.reduce((prev, current) => prev.time > current.time ? prev : current).name
      : 'unknown';

    const fastestComponent = responseTimes.length > 0
      ? responseTimes.reduce((prev, current) => prev.time < current.time ? prev : current).name
      : 'unknown';

    // Determine overall status
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (summary.unhealthy > 0) {
      overallStatus = summary.unhealthy > summary.healthy ? 'unhealthy' : 'degraded';
    } else if (summary.degraded > 0) {
      overallStatus = 'degraded';
    }

    return {
      overallStatus,
      timestamp,
      components,
      summary,
      performance: {
        averageResponseTime: Math.round(averageResponseTime),
        slowestComponent,
        fastestComponent
      }
    };
  }

  private async checkApiHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const healthData = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/health`);
        const data = await response.json();
        return {
          status: response.status,
          ok: response.ok,
          data: data
        };
      }, this.baseUrl);

      const responseTime = Date.now() - startTime;

      if (!healthData.ok) {
        return {
          name: 'API Health',
          status: 'unhealthy',
          responseTime,
          details: { status: healthData.status },
          lastCheck: new Date().toISOString(),
          error: `API health check failed with status ${healthData.status}`
        };
      }

      const apiStatus = healthData.data.status === 'healthy' ? 'healthy' : 
                       healthData.data.status === 'degraded' ? 'degraded' : 'unhealthy';

      return {
        name: 'API Health',
        status: apiStatus,
        responseTime,
        details: healthData.data,
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'API Health',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  private async checkDatabaseHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const dbHealth = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/status`);
        const data = await response.json();
        return {
          status: response.status,
          ok: response.ok,
          database: data.components?.database || {}
        };
      }, this.baseUrl);

      const responseTime = Date.now() - startTime;

      if (!dbHealth.ok) {
        return {
          name: 'PostgreSQL Database',
          status: 'unhealthy',
          responseTime,
          details: { status: dbHealth.status },
          lastCheck: new Date().toISOString(),
          error: 'Database health check endpoint failed'
        };
      }

      const dbConnected = dbHealth.database.connected === true;
      const hasExpectedTables = (dbHealth.database.tables || 0) > 0;

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      if (!dbConnected) {
        status = 'unhealthy';
      } else if (!hasExpectedTables) {
        status = 'degraded';
      }

      return {
        name: 'PostgreSQL Database',
        status,
        responseTime,
        details: {
          connected: dbConnected,
          tables: dbHealth.database.tables,
          migrations: dbHealth.database.migrations_current
        },
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'PostgreSQL Database',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  private async checkRedisHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const redisHealth = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/status`);
        const data = await response.json();
        return {
          status: response.status,
          ok: response.ok,
          redis: data.components?.redis || {}
        };
      }, this.baseUrl);

      const responseTime = Date.now() - startTime;

      if (!redisHealth.ok) {
        return {
          name: 'Redis Cache',
          status: 'unhealthy',
          responseTime,
          details: { status: redisHealth.status },
          lastCheck: new Date().toISOString(),
          error: 'Redis health check endpoint failed'
        };
      }

      const redisConnected = redisHealth.redis.connected === true;
      const streamsActive = redisHealth.redis.streams_active === true;

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      if (!redisConnected) {
        status = 'unhealthy';
      } else if (!streamsActive) {
        status = 'degraded';
      }

      return {
        name: 'Redis Cache',
        status,
        responseTime,
        details: {
          connected: redisConnected,
          memoryUsed: redisHealth.redis.memory_used,
          streamsActive: streamsActive
        },
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'Redis Cache',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  private async checkOrchestratorHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const orchestratorHealth = await page.evaluate(async (url) => {
        const agentResponse = await fetch(`${url}/debug-agents`);
        const agentData = await agentResponse.json();
        
        const healthResponse = await fetch(`${url}/health`);
        const healthData = await healthResponse.json();
        
        return {
          agentStatus: agentResponse.status,
          agentOk: agentResponse.ok,
          agentData: agentData,
          orchestratorHealth: healthData.components?.orchestrator || {}
        };
      }, this.baseUrl);

      const responseTime = Date.now() - startTime;

      const agentSystemWorking = orchestratorHealth.agentOk && 
                                orchestratorHealth.agentData.status === 'debug_working';
      const activeAgents = orchestratorHealth.agentData.agent_count || 0;
      const orchestratorHealthy = orchestratorHealth.orchestratorHealth.status === 'healthy';

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      if (!agentSystemWorking || !orchestratorHealthy) {
        status = 'unhealthy';
      } else if (activeAgents < 5) { // Expecting at least 5 agents
        status = 'degraded';
      }

      return {
        name: 'Agent Orchestrator',
        status,
        responseTime,
        details: {
          agentSystemWorking,
          activeAgents,
          orchestratorHealthy,
          agentDetails: orchestratorHealth.orchestratorHealth
        },
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'Agent Orchestrator',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  private async checkObservabilityHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const observabilityHealth = await page.evaluate(async (url) => {
        const healthResponse = await fetch(`${url}/health`);
        const healthData = await healthResponse.json();
        
        return {
          status: healthResponse.status,
          ok: healthResponse.ok,
          observability: healthData.components?.observability || {}
        };
      }, this.baseUrl);

      const responseTime = Date.now() - startTime;

      if (!observabilityHealth.ok) {
        return {
          name: 'Observability System',
          status: 'unhealthy',
          responseTime,
          details: { status: observabilityHealth.status },
          lastCheck: new Date().toISOString(),
          error: 'Observability health check failed'
        };
      }

      const observabilityActive = observabilityHealth.observability.status === 'healthy';

      return {
        name: 'Observability System',
        status: observabilityActive ? 'healthy' : 'degraded',
        responseTime,
        details: observabilityHealth.observability,
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'Observability System',
        status: 'degraded', // Observability is not critical for core functionality
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  private async checkMetricsHealth(page: Page): Promise<ComponentHealth> {
    const startTime = Date.now();
    
    try {
      const metricsResponse = await page.goto(`${this.baseUrl}/metrics`);
      const responseTime = Date.now() - startTime;

      if (!metricsResponse || !metricsResponse.ok()) {
        return {
          name: 'Metrics Endpoint',
          status: 'degraded', // Metrics are helpful but not critical
          responseTime,
          details: { status: metricsResponse?.status() || 0 },
          lastCheck: new Date().toISOString(),
          error: 'Metrics endpoint not accessible'
        };
      }

      const metricsContent = await page.textContent('body');
      const hasMetrics = metricsContent?.includes('leanvibe_') || false;

      return {
        name: 'Metrics Endpoint',
        status: hasMetrics ? 'healthy' : 'degraded',
        responseTime,
        details: {
          accessible: true,
          hasLeanVibeMetrics: hasMetrics,
          contentLength: metricsContent?.length || 0
        },
        lastCheck: new Date().toISOString()
      };

    } catch (error) {
      return {
        name: 'Metrics Endpoint',
        status: 'degraded',
        responseTime: Date.now() - startTime,
        details: {},
        lastCheck: new Date().toISOString(),
        error: String(error)
      };
    }
  }

  async performDeepHealthCheck(page: Page): Promise<{
    basicHealth: SystemHealthReport;
    loadTest: any;
    connectivityTest: any;
    dataIntegrityTest: any;
  }> {
    const basicHealth = await this.checkSystemHealth(page);
    
    // Perform load test
    const loadTest = await this.performLoadTest(page);
    
    // Test connectivity between components
    const connectivityTest = await this.testComponentConnectivity(page);
    
    // Test data integrity
    const dataIntegrityTest = await this.testDataIntegrity(page);

    return {
      basicHealth,
      loadTest,
      connectivityTest,
      dataIntegrityTest
    };
  }

  private async performLoadTest(page: Page): Promise<any> {
    const results = [];
    const endpoints = ['/health', '/status', '/debug-agents'];
    
    // Test each endpoint multiple times
    for (const endpoint of endpoints) {
      const endpointResults = [];
      
      for (let i = 0; i < 5; i++) {
        const startTime = Date.now();
        
        try {
          const response = await page.evaluate(async ({ baseUrl, endpoint }) => {
            const res = await fetch(`${baseUrl}${endpoint}`);
            return { status: res.status, ok: res.ok };
          }, { baseUrl: this.baseUrl, endpoint });
          
          const responseTime = Date.now() - startTime;
          
          endpointResults.push({
            attempt: i + 1,
            success: response.ok,
            responseTime,
            status: response.status
          });
          
        } catch (error) {
          endpointResults.push({
            attempt: i + 1,
            success: false,
            responseTime: Date.now() - startTime,
            error: String(error)
          });
        }
        
        // Small delay between requests
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      const successfulRequests = endpointResults.filter(r => r.success).length;
      const averageResponseTime = endpointResults
        .filter(r => r.success)
        .reduce((sum, r) => sum + r.responseTime, 0) / successfulRequests || 0;
      
      results.push({
        endpoint,
        totalRequests: endpointResults.length,
        successfulRequests,
        successRate: successfulRequests / endpointResults.length,
        averageResponseTime: Math.round(averageResponseTime),
        results: endpointResults
      });
    }

    return {
      timestamp: new Date().toISOString(),
      endpointResults: results,
      overallSuccessRate: results.reduce((sum, r) => sum + r.successRate, 0) / results.length
    };
  }

  private async testComponentConnectivity(page: Page): Promise<any> {
    // Test that components can communicate with each other
    const connectivityResults = [];

    // Test API to Database connectivity
    try {
      const dbTest = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/api/v1/agents`);
        return { status: response.status, ok: response.ok };
      }, this.baseUrl);

      connectivityResults.push({
        connection: 'API to Database',
        success: [200, 404].includes(dbTest.status), // 404 is OK for empty collections
        details: dbTest
      });
    } catch (error) {
      connectivityResults.push({
        connection: 'API to Database',
        success: false,
        error: String(error)
      });
    }

    // Test API to Redis connectivity (through agent status)
    try {
      const redisTest = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/debug-agents`);
        return { status: response.status, ok: response.ok };
      }, this.baseUrl);

      connectivityResults.push({
        connection: 'API to Redis',
        success: redisTest.ok,
        details: redisTest
      });
    } catch (error) {
      connectivityResults.push({
        connection: 'API to Redis',
        success: false,
        error: String(error)
      });
    }

    return {
      timestamp: new Date().toISOString(),
      connectivityResults,
      allConnectionsWorking: connectivityResults.every(r => r.success)
    };
  }

  private async testDataIntegrity(page: Page): Promise<any> {
    // Test basic data operations to ensure system integrity
    const integrityTests = [];

    // Test health data consistency
    try {
      const healthResponse1 = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/health`);
        return response.json();
      }, this.baseUrl);

      await new Promise(resolve => setTimeout(resolve, 1000));

      const healthResponse2 = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/health`);
        return response.json();
      }, this.baseUrl);

      const dataConsistent = healthResponse1.version === healthResponse2.version &&
                             healthResponse1.summary.total === healthResponse2.summary.total;

      integrityTests.push({
        test: 'Health Data Consistency',
        success: dataConsistent,
        details: {
          firstCheck: healthResponse1.summary,
          secondCheck: healthResponse2.summary
        }
      });
    } catch (error) {
      integrityTests.push({
        test: 'Health Data Consistency',
        success: false,
        error: String(error)
      });
    }

    // Test agent data consistency
    try {
      const agentResponse1 = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/debug-agents`);
        return response.json();
      }, this.baseUrl);

      await new Promise(resolve => setTimeout(resolve, 2000));

      const agentResponse2 = await page.evaluate(async (url) => {
        const response = await fetch(`${url}/debug-agents`);
        return response.json();
      }, this.baseUrl);

      const agentCountConsistent = agentResponse1.agent_count === agentResponse2.agent_count;

      integrityTests.push({
        test: 'Agent Data Consistency',
        success: agentCountConsistent,
        details: {
          firstCount: agentResponse1.agent_count,
          secondCount: agentResponse2.agent_count
        }
      });
    } catch (error) {
      integrityTests.push({
        test: 'Agent Data Consistency',
        success: false,
        error: String(error)
      });
    }

    return {
      timestamp: new Date().toISOString(),
      integrityTests,
      dataIntegrityPassed: integrityTests.every(t => t.success)
    };
  }
}