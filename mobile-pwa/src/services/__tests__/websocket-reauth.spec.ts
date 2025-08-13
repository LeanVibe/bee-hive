import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AuthService } from '../../services/auth'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  url: string
  readyState = 1 // OPEN
  onopen: ((ev: any) => any) | null = null
  onmessage: ((ev: any) => any) | null = null
  onclose: ((ev: any) => any) | null = null
  onerror: ((ev: any) => any) | null = null

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
    // auto-open
    setTimeout(() => this.onopen && this.onopen({}), 0)
  }
  send(){}
  close(){ this.readyState = 3; this.onclose && this.onclose({ code: 1000, reason: 'test' }) }
}

// @ts-expect-error override global
global.WebSocket = FakeWebSocket as any

const getAuth = () => AuthService.getInstance() as any

describe('WebSocketService re-auth on token refresh', () => {
  beforeEach(() => {
    FakeWebSocket.instances.length = 0
  })

  it('reconnects with new token when token-refreshed fires', async () => {
    // Stable window env
    ;(global as any).window = {
      location: { protocol: 'http:', hostname: 'localhost' },
      addEventListener: () => {},
      removeEventListener: () => {},
      history: { replaceState: () => {}, pushState: () => {} },
      setInterval,
      clearInterval,
      setTimeout,
      clearTimeout,
      document: { addEventListener: () => {}, visibilityState: 'visible' }
    }

    const auth = getAuth()
    auth.state = {
      user: { id: 'u', email: 'e', name: 'n', full_name: 'n', role: 'viewer', permissions: [], pilot_ids: [], is_active: true, auth_method: 'password' },
      token: 'old-token',
      refreshToken: 'r',
      isAuthenticated: true,
      lastActivity: Date.now(),
      sessionId: 's',
      biometricEnabled: false,
    }
    const mod = await import('../../services/websocket')
    const ws: any = new (mod as any).WebSocketService()
    await (ws as any).connect()

    // First connection uses old token
    expect(FakeWebSocket.instances[0].url).toContain('access_token=old-token')

    // Emit token-refreshed
    auth.state.token = 'new-token'
    auth.emit('token-refreshed', { accessToken: 'new-token', refreshToken: 'r2' })

    // Allow reconnect microtask to schedule new WebSocket
    await new Promise(r => setTimeout(r, 5))

    // Cleanup to avoid hanging timers
    ws.disconnect()

    // Second connection should exist with new token
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1)
    const last = FakeWebSocket.instances.at(-1)!
    expect(last.url).toContain('access_token=new-token')
  })
})
