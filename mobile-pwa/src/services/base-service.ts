/**
 * Base Service Class for LeanVibe Agent Hive API Integration
 * 
 * Provides common functionality for all API services including:
 * - HTTP client with retry logic
 * - Caching mechanisms
 * - Error handling and transformation
 * - Loading state management
 * - Event emission for real-time updates
 */

import { EventEmitter } from '../utils/event-emitter';
import type {
  ApiResponse,
  ApiError,
  ServiceConfig,
  RetryOptions,
  LoadingState,
  EventListener,
  Subscription
} from '../types/api';

export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

export class BaseService extends EventEmitter {
  protected config: ServiceConfig;
  private cache = new Map<string, CacheEntry<any>>();
  private loadingStates = new Map<string, LoadingState>();

  constructor(config: Partial<ServiceConfig> = {}) {
    super();
    
    this.config = {
      baseUrl: 'http://localhost:8000',
      timeout: 10000,
      retryAttempts: 3,
      retryDelay: 1000,
      cacheTimeout: 60000, // 1 minute
      pollingInterval: 5000, // 5 seconds
      ...config
    };
  }

  // ===== HTTP CLIENT METHODS =====

  protected async get<T>(
    endpoint: string, 
    options: RequestInit = {},
    cacheKey?: string
  ): Promise<T> {
    return this.makeRequest<T>('GET', endpoint, undefined, options, cacheKey);
  }

  protected async post<T>(
    endpoint: string, 
    data?: any, 
    options: RequestInit = {}
  ): Promise<T> {
    return this.makeRequest<T>('POST', endpoint, data, options);
  }

  protected async put<T>(
    endpoint: string, 
    data?: any, 
    options: RequestInit = {}
  ): Promise<T> {
    return this.makeRequest<T>('PUT', endpoint, data, options);
  }

  protected async delete<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    return this.makeRequest<T>('DELETE', endpoint, undefined, options);
  }

  private async makeRequest<T>(
    method: string,
    endpoint: string,
    data?: any,
    options: RequestInit = {},
    cacheKey?: string
  ): Promise<T> {
    const operationKey = `${method}:${endpoint}`;
    
    // Check cache first for GET requests
    if (method === 'GET' && cacheKey) {
      const cached = this.getFromCache<T>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    // Set loading state
    this.setLoadingState(operationKey, { isLoading: true });

    try {
      const url = `${this.config.baseUrl}${endpoint}`;
      const requestOptions: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        signal: AbortSignal.timeout(this.config.timeout),
        ...options
      };

      if (data) {
        requestOptions.body = JSON.stringify(data);
      }

      const response = await this.retryRequest(() => fetch(url, requestOptions));
      
      if (!response.ok) {
        throw await this.createApiError(response);
      }

      const result = await response.json();
      
      // Cache successful GET requests
      if (method === 'GET' && cacheKey) {
        this.setCache(cacheKey, result);
      }

      // Update loading state
      this.setLoadingState(operationKey, { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });

      return result;

    } catch (error) {
      const apiError = this.transformError(error);
      
      // Update loading state with error
      this.setLoadingState(operationKey, { 
        isLoading: false, 
        error: apiError,
        lastUpdated: new Date().toISOString()
      });

      // Emit error event
      this.emit('error', apiError);
      
      throw apiError;
    }
  }

  // ===== RETRY LOGIC =====

  private async retryRequest(
    requestFn: () => Promise<Response>,
    options: Partial<RetryOptions> = {}
  ): Promise<Response> {
    const retryOptions: RetryOptions = {
      maxAttempts: this.config.retryAttempts,
      delay: this.config.retryDelay,
      backoffMultiplier: 2,
      shouldRetry: (error) => this.shouldRetryRequest(error),
      ...options
    };

    let lastError: Error;
    
    for (let attempt = 1; attempt <= retryOptions.maxAttempts; attempt++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on last attempt or if shouldn't retry
        if (attempt === retryOptions.maxAttempts || !retryOptions.shouldRetry!(lastError)) {
          throw error;
        }

        // Calculate delay with exponential backoff
        const delay = retryOptions.delay * Math.pow(retryOptions.backoffMultiplier, attempt - 1);
        await this.sleep(delay);
      }
    }

    throw lastError!;
  }

  private shouldRetryRequest(error: Error): boolean {
    // Retry on network errors, timeouts, and 5xx server errors
    if (error.name === 'TypeError' || error.name === 'AbortError') {
      return true;
    }
    
    if ('status' in error) {
      const status = (error as any).status;
      return status >= 500 || status === 429; // Server errors or rate limiting
    }
    
    return false;
  }

  // ===== CACHING METHODS =====

  protected getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }

    // Check if cache entry has expired
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  protected setCache<T>(key: string, data: T, ttl?: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttl || this.config.cacheTimeout
    });
  }

  protected clearCache(pattern?: string): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    // Clear cache entries matching pattern
    for (const [key] of this.cache) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
      }
    }
  }

  // ===== LOADING STATE MANAGEMENT =====

  protected getLoadingState(key: string): LoadingState {
    return this.loadingStates.get(key) || { isLoading: false };
  }

  protected setLoadingState(key: string, state: LoadingState): void {
    this.loadingStates.set(key, state);
    this.emit('loadingStateChanged', { key, state });
  }

  // ===== POLLING SUPPORT =====

  protected startPolling(
    pollFn: () => Promise<void>,
    interval: number = this.config.pollingInterval
  ): () => void {
    let timeoutId: number | null = null;
    let isRunning = true;

    const poll = async () => {
      if (!isRunning) return;

      try {
        await pollFn();
      } catch (error) {
        this.emit('pollingError', error);
      }

      if (isRunning) {
        timeoutId = window.setTimeout(poll, interval);
      }
    };

    // Start polling immediately
    poll();

    // Return stop function
    return () => {
      isRunning = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    };
  }

  // ===== ERROR HANDLING =====

  private async createApiError(response: Response): Promise<ApiError> {
    let errorData: any = {};
    
    try {
      errorData = await response.json();
    } catch {
      // Response doesn't contain JSON
    }

    return {
      code: errorData.code || `HTTP_${response.status}`,
      message: errorData.message || errorData.detail || response.statusText || 'Unknown error',
      details: errorData.details || {},
      timestamp: new Date().toISOString()
    };
  }

  private transformError(error: any): ApiError {
    if (error.name === 'AbortError') {
      return {
        code: 'TIMEOUT',
        message: 'Request timed out',
        timestamp: new Date().toISOString()
      };
    }

    if (error.name === 'TypeError') {
      return {
        code: 'NETWORK_ERROR',
        message: 'Network connection failed',
        timestamp: new Date().toISOString()
      };
    }

    if (error.code && error.message) {
      return error as ApiError;
    }

    return {
      code: 'UNKNOWN_ERROR',
      message: error.message || 'An unknown error occurred',
      details: { originalError: error },
      timestamp: new Date().toISOString()
    };
  }

  // ===== UTILITY METHODS =====

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  protected formatEndpoint(template: string, params: Record<string, any>): string {
    return template.replace(/\{(\w+)\}/g, (match, key) => {
      if (key in params) {
        return encodeURIComponent(params[key]);
      }
      return match;
    });
  }

  protected buildQueryString(params: Record<string, any>): string {
    const searchParams = new URLSearchParams();
    
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    }
    
    const query = searchParams.toString();
    return query ? `?${query}` : '';
  }

  // ===== EVENT SUBSCRIPTION HELPERS =====

  public onLoadingStateChange(listener: EventListener<{ key: string; state: LoadingState }>): Subscription {
    return this.subscribe('loadingStateChanged', listener);
  }

  public onError(listener: EventListener<ApiError>): Subscription {
    return this.subscribe('error', listener);
  }

  public onPollingError(listener: EventListener<Error>): Subscription {
    return this.subscribe('pollingError', listener);
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.cache.clear();
    this.loadingStates.clear();
    this.removeAllListeners();
  }
}