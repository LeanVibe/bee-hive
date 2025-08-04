import { Page } from '@playwright/test';

/**
 * API Discoverer Utility
 * 
 * Automatically discovers and validates API endpoints from OpenAPI schema.
 * Provides comprehensive endpoint testing and validation capabilities.
 */

export interface ApiEndpoint {
  path: string;
  method: string;
  operationId?: string;
  summary?: string;
  description?: string;
  parameters?: any[];
  responses?: Record<string, any>;
  tags?: string[];
}

export interface ApiDiscoveryResult {
  totalEndpoints: number;
  endpointsByMethod: Record<string, number>;
  endpointsByTag: Record<string, number>;
  endpoints: ApiEndpoint[];
  validationResults: EndpointValidationResult[];
}

export interface EndpointValidationResult {
  endpoint: ApiEndpoint;
  tested: boolean;
  status: number;
  responseTime: number;
  success: boolean;
  error?: string;
  responseData?: any;
}

export class ApiDiscoverer {
  private page: Page;
  private baseUrl: string;

  constructor(page: Page, baseUrl: string = '') {
    this.page = page;
    this.baseUrl = baseUrl || 'http://localhost:8000';
  }

  async discoverEndpoints(): Promise<ApiEndpoint[]> {
    try {
      // Fetch OpenAPI schema
      const openApiSchema = await this.page.evaluate(async (url) => {
        const response = await fetch(`${url}/openapi.json`);
        if (!response.ok) {
          throw new Error(`Failed to fetch OpenAPI schema: ${response.status}`);
        }
        return response.json();
      }, this.baseUrl);

      return this.parseOpenApiSchema(openApiSchema);
    } catch (error) {
      console.error('Failed to discover endpoints:', error);
      return [];
    }
  }

  private parseOpenApiSchema(schema: any): ApiEndpoint[] {
    const endpoints: ApiEndpoint[] = [];

    if (!schema.paths) {
      return endpoints;
    }

    Object.entries(schema.paths).forEach(([path, pathObj]: [string, any]) => {
      Object.entries(pathObj).forEach(([method, methodObj]: [string, any]) => {
        if (this.isValidHttpMethod(method.toUpperCase())) {
          endpoints.push({
            path,
            method: method.toUpperCase(),
            operationId: methodObj.operationId,
            summary: methodObj.summary,
            description: methodObj.description,
            parameters: methodObj.parameters || [],
            responses: methodObj.responses || {},
            tags: methodObj.tags || []
          });
        }
      });
    });

    return endpoints;
  }

  private isValidHttpMethod(method: string): boolean {
    const validMethods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'];
    return validMethods.includes(method);
  }

  async validateEndpoints(endpoints: ApiEndpoint[]): Promise<EndpointValidationResult[]> {
    const results: EndpointValidationResult[] = [];

    for (const endpoint of endpoints) {
      const result = await this.validateSingleEndpoint(endpoint);
      results.push(result);
      
      // Add small delay to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    return results;
  }

  private async validateSingleEndpoint(endpoint: ApiEndpoint): Promise<EndpointValidationResult> {
    const startTime = Date.now();
    
    try {
      let testData = {};
      let queryParams = '';
      
      // Build test data based on endpoint requirements
      if (endpoint.method === 'POST' || endpoint.method === 'PUT' || endpoint.method === 'PATCH') {
        testData = this.generateTestData(endpoint);
      }
      
      // Build query parameters for GET requests
      if (endpoint.method === 'GET' && endpoint.parameters) {
        const queryParametersData = this.generateQueryParameters(endpoint.parameters);
        if (Object.keys(queryParametersData).length > 0) {
          queryParams = '?' + new URLSearchParams(queryParametersData).toString();
        }
      }

      const response = await this.page.evaluate(async ({ baseUrl, path, method, data, query }) => {
        const url = `${baseUrl}${path}${query}`;
        const options: RequestInit = {
          method: method,
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        };

        if (method !== 'GET' && method !== 'HEAD' && Object.keys(data).length > 0) {
          options.body = JSON.stringify(data);
        }

        try {
          const response = await fetch(url, options);
          let responseData;
          
          try {
            responseData = await response.json();
          } catch {
            responseData = await response.text();
          }

          return {
            status: response.status,
            ok: response.ok,
            statusText: response.statusText,
            data: responseData,
            headers: Object.fromEntries(response.headers.entries())
          };
        } catch (error) {
          return {
            status: 0,
            ok: false,
            error: error.message,
            data: null
          };
        }
      }, {
        baseUrl: this.baseUrl,
        path: endpoint.path,
        method: endpoint.method,
        data: testData,
        query: queryParams
      });

      const responseTime = Date.now() - startTime;

      return {
        endpoint,
        tested: true,
        status: response.status,
        responseTime,
        success: this.isSuccessfulResponse(response.status, endpoint.method),
        responseData: response.data,
        error: response.error
      };

    } catch (error) {
      const responseTime = Date.now() - startTime;

      return {
        endpoint,
        tested: true,
        status: 0,
        responseTime,
        success: false,
        error: String(error)
      };
    }
  }

  private generateTestData(endpoint: ApiEndpoint): any {
    const data: any = {};

    // Generate basic test data based on common patterns
    if (endpoint.path.includes('agents')) {
      return {
        name: 'Test Agent',
        agent_type: 'CLAUDE',
        capabilities: ['test_capability']
      };
    }

    if (endpoint.path.includes('tasks')) {
      return {
        title: 'Test Task',
        description: 'Automated test task',
        priority: 'MEDIUM'
      };
    }

    if (endpoint.path.includes('workflows')) {
      return {
        name: 'Test Workflow',
        description: 'Automated test workflow',
        steps: []
      };
    }

    if (endpoint.path.includes('contexts')) {
      return {
        content: 'Test context content',
        type: 'test'
      };
    }

    // Default minimal test data
    return {
      test: true,
      timestamp: new Date().toISOString()
    };
  }

  private generateQueryParameters(parameters: any[]): any {
    const queryParams: any = {};

    parameters.forEach(param => {
      if (param.in === 'query' && !param.required) {
        // Only add optional query parameters to avoid breaking required ones
        switch (param.name) {
          case 'limit':
            queryParams.limit = '10';
            break;
          case 'offset':
            queryParams.offset = '0';
            break;
          case 'page':
            queryParams.page = '1';
            break;
          case 'search':
            queryParams.search = 'test';
            break;
          default:
            // Skip unknown required parameters
            break;
        }
      }
    });

    return queryParams;
  }

  private isSuccessfulResponse(status: number, method: string): boolean {
    // Define success criteria based on HTTP method
    switch (method) {
      case 'GET':
        return status === 200 || status === 404; // 404 is acceptable for empty collections
      case 'POST':
        return status >= 200 && status < 300; // 201 Created, 200 OK
      case 'PUT':
      case 'PATCH':
        return status >= 200 && status < 300; // 200 OK, 204 No Content
      case 'DELETE':
        return status >= 200 && status < 300 || status === 404; // 204 No Content, 404 acceptable
      case 'HEAD':
        return status >= 200 && status < 300;
      case 'OPTIONS':
        return status >= 200 && status < 300;
      default:
        return status >= 200 && status < 400;
    }
  }

  async generateDiscoveryReport(endpoints: ApiEndpoint[], validationResults: EndpointValidationResult[]): Promise<ApiDiscoveryResult> {
    // Aggregate statistics
    const endpointsByMethod = endpoints.reduce((acc, endpoint) => {
      acc[endpoint.method] = (acc[endpoint.method] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const endpointsByTag = endpoints.reduce((acc, endpoint) => {
      endpoint.tags?.forEach(tag => {
        acc[tag] = (acc[tag] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    return {
      totalEndpoints: endpoints.length,
      endpointsByMethod,
      endpointsByTag,
      endpoints,
      validationResults
    };
  }

  async testCrudOperations(baseEndpoint: string, resourceName: string): Promise<any> {
    const crudResults = {
      create: null,
      read: null,
      update: null,
      delete: null,
      list: null
    };

    try {
      // Test LIST (GET /)
      crudResults.list = await this.page.evaluate(async ({ baseUrl, endpoint }) => {
        const response = await fetch(`${baseUrl}${endpoint}`);
        return {
          status: response.status,
          ok: response.ok,
          data: response.ok ? await response.json() : null
        };
      }, { baseUrl: this.baseUrl, endpoint: baseEndpoint });

      // Test CREATE (POST /)
      const createData = this.generateTestData({ path: baseEndpoint, method: 'POST' } as ApiEndpoint);
      crudResults.create = await this.page.evaluate(async ({ baseUrl, endpoint, data }) => {
        const response = await fetch(`${baseUrl}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        return {
          status: response.status,
          ok: response.ok,
          data: response.ok ? await response.json() : null
        };
      }, { baseUrl: this.baseUrl, endpoint: baseEndpoint, data: createData });

      // If CREATE was successful, test READ, UPDATE, DELETE with the created resource
      if (crudResults.create?.ok && crudResults.create?.data?.id) {
        const resourceId = crudResults.create.data.id;
        const resourceEndpoint = `${baseEndpoint}/${resourceId}`;

        // Test READ (GET /:id)
        crudResults.read = await this.page.evaluate(async ({ baseUrl, endpoint }) => {
          const response = await fetch(`${baseUrl}${endpoint}`);
          return {
            status: response.status,
            ok: response.ok,
            data: response.ok ? await response.json() : null
          };
        }, { baseUrl: this.baseUrl, endpoint: resourceEndpoint });

        // Test UPDATE (PUT /:id)
        const updateData = { ...createData, updated: true };
        crudResults.update = await this.page.evaluate(async ({ baseUrl, endpoint, data }) => {
          const response = await fetch(`${baseUrl}${endpoint}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
          });
          return {
            status: response.status,
            ok: response.ok,
            data: response.ok ? await response.json() : null
          };
        }, { baseUrl: this.baseUrl, endpoint: resourceEndpoint, data: updateData });

        // Test DELETE (DELETE /:id)
        crudResults.delete = await this.page.evaluate(async ({ baseUrl, endpoint }) => {
          const response = await fetch(`${baseUrl}${endpoint}`, {
            method: 'DELETE'
          });
          return {
            status: response.status,
            ok: response.ok,
            data: response.ok ? await response.json() : null
          };
        }, { baseUrl: this.baseUrl, endpoint: resourceEndpoint });
      }

    } catch (error) {
      console.error(`CRUD testing error for ${baseEndpoint}:`, error);
    }

    return crudResults;
  }
}