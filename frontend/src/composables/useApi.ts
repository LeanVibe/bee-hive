/**
 * API Composable
 * Provides a simple interface for making API requests
 */

import { apiClient } from '@/services/api'

interface ApiOptions {
  params?: Record<string, any>
}

export function useApi() {
  const get = async <T = any>(endpoint: string, options: ApiOptions = {}): Promise<{ data: T }> => {
    const params = options.params ? new URLSearchParams(options.params) : undefined
    const data = await apiClient.get<T>(endpoint, params)
    return { data }
  }

  const post = async <T = any>(endpoint: string, body?: any): Promise<{ data: T }> => {
    const data = await apiClient.post<T>(endpoint, body)
    return { data }
  }

  const put = async <T = any>(endpoint: string, body?: any): Promise<{ data: T }> => {
    const data = await apiClient.put<T>(endpoint, body)
    return { data }
  }

  const patch = async <T = any>(endpoint: string, body?: any): Promise<{ data: T }> => {
    const data = await apiClient.patch<T>(endpoint, body)
    return { data }
  }

  const del = async <T = any>(endpoint: string): Promise<{ data: T }> => {
    const data = await apiClient.delete<T>(endpoint)
    return { data }
  }

  return {
    get,
    post,
    put,
    patch,
    delete: del
  }
}