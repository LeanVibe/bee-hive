import { describe, it, expect, vi, beforeEach } from 'vitest'
import { OfflineService } from '../../src/services/offline'

describe('OfflineService backoff and scheduling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  it('computes exponential backoff with cap', async () => {
    const compute = (OfflineService as any).computeBackoffMs as (retry: number, base?: number, cap?: number) => number
    expect(compute(0)).toBe(1000)
    expect(compute(1)).toBe(2000)
    expect(compute(2)).toBe(4000)
    expect(compute(3)).toBe(8000)
    expect(compute(4)).toBe(16000)
    expect(compute(5)).toBe(30000)
    expect(compute(6)).toBe(30000)
  })

  it('schedules retry after failure with computed backoff', async () => {
    const offline: any = new (OfflineService as any)()
    offline.isOnline = true
    offline.db = { put: vi.fn().mockResolvedValue(undefined), delete: vi.fn().mockResolvedValue(undefined) }

    offline.syncQueue = [
      { id: 'op-1', type: 'update', resource: 'tasks', data: { id: 't1' }, timestamp: Date.now(), retryCount: 0, maxRetries: 3 },
    ]

    offline.executeSync = vi.fn().mockRejectedValue(new Error('fail'))

    const scheduledSpy = vi.fn()
    offline.on('sync_scheduled_retry', scheduledSpy)

    await offline.startBackgroundSync()
    expect(scheduledSpy).toHaveBeenCalled()
    const { ms } = scheduledSpy.mock.calls[0][0]
    expect(ms).toBe(1000)

    const startSpy = vi.spyOn(offline, 'startBackgroundSync')
    vi.advanceTimersByTime(1000)
    expect(startSpy).toHaveBeenCalled()
  })
})
