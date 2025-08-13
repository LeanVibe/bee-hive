import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AuthService } from '../../services/auth'

// Helper to access private state for test
const getAuth = () => AuthService.getInstance() as any

describe('AuthService token refresh', () => {
  beforeEach(() => {
    // Reset fetch mock
    // @ts-expect-error override global
    global.fetch = vi.fn()
  })

  it('emits token-refreshed and updates tokens on successful refresh', async () => {
    const auth = getAuth()
    // Seed a refresh token
    auth.state = {
      user: { id: 'u', email: 'e', name: 'n', full_name: 'n', role: 'viewer', permissions: [], pilot_ids: [], is_active: true, auth_method: 'password' },
      token: 'old-access',
      refreshToken: 'r1',
      isAuthenticated: true,
      lastActivity: Date.now(),
      sessionId: 's',
      biometricEnabled: false,
    }

    const refreshed = { access_token: 'new-access', refresh_token: 'r2' }
    ;(global.fetch as any).mockResolvedValue({ ok: true, json: async () => refreshed })

    const emitted: any[] = []
    auth.on('token-refreshed', (payload: any) => emitted.push(payload))

    await auth.refreshToken()

    expect(auth.getToken()).toBe('new-access')
    expect(emitted.length).toBe(1)
    expect(emitted[0].accessToken).toBe('new-access')
    expect(emitted[0].refreshToken).toBe('r2')
  })
})
