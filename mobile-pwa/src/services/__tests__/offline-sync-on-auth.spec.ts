import { describe, it, expect, vi } from 'vitest'
import { OfflineService } from '../../services/offline'
import { AuthService } from '../../services/auth'

const getAuth = () => AuthService.getInstance() as any

describe('OfflineService resumes sync on auth events', () => {
  it('starts background sync on authenticated and token-refreshed when online', async () => {
    const offline = OfflineService.getInstance() as any
    // Force online
    ;(offline as any).isOnline = true
    ;(offline as any).syncInProgress = false

    const startSpy = vi.spyOn(offline, 'startBackgroundSync' as any)

    const auth = getAuth()
    auth.emit('authenticated')
    auth.emit('token-refreshed')

    expect(startSpy).toHaveBeenCalled()
  })
})
