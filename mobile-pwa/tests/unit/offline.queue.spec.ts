import { describe, it, expect, vi, beforeEach } from 'vitest'
import { OfflineService } from '../../src/services/offline'

describe('OfflineService queue correlation id', () => {
  it('adds correlation_id on create/update operations', async () => {
    const offline: any = new (OfflineService as any)()
    offline.db = { put: vi.fn().mockResolvedValue(undefined) }
    offline.syncQueue = []

    const payload: any = { id: 't1', title: 'T', status: 'pending', priority: 'low', created_at: Date.now(), updated_at: Date.now(), synced: false }
    await offline.queueSync('update', 'tasks', payload)

    expect(payload.correlation_id).toBeTruthy()
    expect(offline.syncQueue[0].data.correlation_id).toBe(payload.correlation_id)
  })
})
