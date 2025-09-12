import { useNotificationStore } from '@/stores/notifications'

export interface ApiResponse<T = any> {
  data?: T
  status: number
  statusText: string
  headers: Headers
}

export interface ApiError extends Error {
  status?: number
  response?: any
}

class ApiClient {
  private baseURL: string
  private defaultHeaders: Record<string, string>
  
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }
  }
  
  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }
    
    try {
      const response = await fetch(url, config)
      
      // Handle non-JSON responses (like Prometheus metrics)
      const contentType = response.headers.get('content-type')
      let data: any
      
      if (contentType?.includes('application/json')) {
        data = await response.json()
      } else {
        data = await response.text()
      }
      
      if (!response.ok) {
        const error: ApiError = new Error(
          data?.detail || data?.message || `HTTP ${response.status}: ${response.statusText}`
        )
        error.status = response.status
        error.response = data
        throw error
      }
      
      return data
      
    } catch (error) {
      // Handle network errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        const networkError: ApiError = new Error('Network error: Unable to connect to server')
        networkError.status = 0
        throw networkError
      }
      
      throw error
    }
  }
  
  async get<T = any>(endpoint: string, params?: URLSearchParams): Promise<T> {
    const url = params ? `${endpoint}?${params}` : endpoint
    return this.request<T>(url, { method: 'GET' })
  }
  
  async post<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }
  
  async put<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }
  
  async patch<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    })
  }
  
  async delete<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
  
  // Convenience method for file uploads
  async upload<T = any>(endpoint: string, file: File, additionalData?: Record<string, any>): Promise<T> {
    const formData = new FormData()
    formData.append('file', file)
    
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, String(value))
      })
    }
    
    return this.request<T>(endpoint, {
      method: 'POST',
      body: formData,
      headers: {
        // Don't set Content-Type for FormData, let browser set it with boundary
        'Accept': 'application/json',
      },
    })
  }
  
  // Set authorization header
  setAuthToken(token: string) {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`
  }
  
  // Remove authorization header
  clearAuthToken() {
    delete this.defaultHeaders['Authorization']
  }
  
  // Update base URL
  setBaseURL(url: string) {
    this.baseURL = url
  }
}

// Create singleton instance
export const apiClient = new ApiClient()

// Default export for backward compatibility
export const api = apiClient
export const apiService = apiClient

// Global error interceptor
const originalRequest = (apiClient as any).request.bind(apiClient)
;(apiClient as any).request = async function<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  try {
    return await originalRequest(endpoint, options)
  } catch (error) {
    // Handle global errors
    const apiError = error as ApiError
    
    // Don't show notifications for certain endpoints or error codes
    const silentEndpoints = ['/health', '/metrics', '/status']
    const silentErrors = [401, 403] // Authentication/authorization errors
    
    const shouldShowNotification = !silentEndpoints.some(endpoint => endpoint.includes(endpoint)) &&
                                  !silentErrors.includes(apiError.status || 0)
    
    if (shouldShowNotification && typeof window !== 'undefined') {
      // Only show notifications in browser environment
      try {
        const notificationStore = useNotificationStore()
        
        if (apiError.status === 0) {
          notificationStore.error(
            'Connection Error',
            'Unable to connect to the server. Please check your connection.'
          )
        } else if (apiError.status && apiError.status >= 500) {
          notificationStore.error(
            'Server Error',
            `Server error (${apiError.status}): ${apiError.message}`
          )
        } else if (apiError.status && apiError.status >= 400) {
          notificationStore.warning(
            'Request Error',
            `Request failed (${apiError.status}): ${apiError.message}`
          )
        }
      } catch (notificationError) {
        // Fallback if notification store is not available
        console.error('Failed to show notification:', notificationError)
      }
    }
    
    throw error
  }
}

// Health check function
export async function checkServerHealth(): Promise<boolean> {
  try {
    await apiClient.get('/health')
    return true
  } catch {
    return false
  }
}

// Retry utility
export async function retryRequest<T>(
  requestFn: () => Promise<T>,
  maxRetries = 3,
  delay = 1000
): Promise<T> {
  let lastError: Error
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn()
    } catch (error) {
      lastError = error as Error
      
      if (attempt === maxRetries) {
        break
      }
      
      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt - 1)))
    }
  }
  
  throw lastError!
}