import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { BackendAdapter } from '../backend-adapter'

// Simple fetch mock
const originalFetch = global.fetch

describe('BackendAdapter', () => {
  let adapter: BackendAdapter

  beforeEach(() => {
    adapter = new BackendAdapter()
    // @ts-expect-error override private interval for fast tests
    adapter['fetchInterval'] = 10
    global.fetch = vi.fn()
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  it('fetches and returns live data from /dashboard/api/live-data', async () => {
    const payload = {
      metrics: {
        active_projects: 1,
        active_agents: 2,
        agent_utilization: 50,
        completed_tasks: 3,
        active_conflicts: 0,
        system_efficiency: 90,
        system_status: 'healthy',
        last_updated: new Date().toISOString()
      },
      agent_activities: [],
      project_snapshots: [],
      conflict_snapshots: []
    }

    ;(global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => payload
    })

    const data = await adapter.getLiveData(true)
    expect(data.metrics.active_agents).toBe(2)
  })

  it('uses cache within fetchInterval when not forced', async () => {
    const payload = {
      metrics: {
        active_projects: 0,
        active_agents: 1,
        agent_utilization: 10,
        completed_tasks: 0,
        active_conflicts: 0,
        system_efficiency: 80,
        system_status: 'healthy',
        last_updated: new Date().toISOString()
      },
      agent_activities: [],
      project_snapshots: [],
      conflict_snapshots: []
    }

    ;(global.fetch as any).mockResolvedValue({ ok: true, json: async () => payload })

    const first = await adapter.getLiveData(true)
    expect(first.metrics.active_agents).toBe(1)

    // Change mock to ensure no additional fetch is used
    ;(global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({ ...payload, metrics: { ...payload.metrics, active_agents: 99 } }) })

    const second = await adapter.getLiveData(false)
    expect(second.metrics.active_agents).toBe(1)
    expect((global.fetch as any).mock.calls.length).toBe(1)
  })

  it('falls back to mock data when backend fails', async () => {
    // Fail fast: one attempt, no exponential backoff delays
    ;(global.fetch as any).mockRejectedValue(new TypeError('Network error'))
    // @ts-expect-error override retry attempts for fast test
    adapter['fetchWithRetry'] = async () => { throw new TypeError('Network error') }

    const data = await adapter.getLiveData(true)
    expect(data.metrics.system_status).toBeDefined()
    expect(['healthy', 'degraded', 'critical']).toContain(data.metrics.system_status)
  })
})
