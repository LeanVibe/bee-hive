import { describe, it, expect, vi, beforeEach } from 'vitest'
import { OfflineService } from '../../src/services/offline'

function makeTask(id: string, updated: number, corr?: string) {
  return { id, title: 't', status: 'pending', priority: 'low', created_at: updated - 100, updated_at: updated, synced: false, correlation_id: corr }
}

describe('OfflineService reconciliation', () => {
  let offline: any
  beforeEach(async () => {
    offline = new (OfflineService as any)()
    offline.db = {
      data: new Map<string, any>(),
      get: vi.fn(async (store: string, id: string) => offline.db.data.get(`${store}:${id}`)),
      put: vi.fn(async (store: string, val: any) => { offline.db.data.set(`${store}:${val.id}` , val) }),
    }
  })

  it('applies server task when no local exists', async () => {
    const server = makeTask('a', 2000, 'c1')
    await offline.reconcileTaskFromServer(server)
    const saved = await offline.db.get('tasks', 'a')
    expect(saved.updated_at).toBe(2000)
    expect(saved.synced).toBe(true)
  })

  it('keeps local if newer (last-write-wins)', async () => {
    const local = makeTask('b', 3000, 'c2')
    await offline.db.put('tasks', local)
    const server = makeTask('b', 2000)
    await offline.reconcileTaskFromServer(server)
    const kept = await offline.db.get('tasks', 'b')
    expect(kept.updated_at).toBe(3000)
    expect(kept.synced).toBe(false)
  })

  it('applies server if same correlation_id (server accepted our change)', async () => {
    const local = makeTask('c', 1000, 'c3')
    await offline.db.put('tasks', local)
    const server = makeTask('c', 1500, 'c3')
    await offline.reconcileTaskFromServer(server)
    const reconciled = await offline.db.get('tasks', 'c')
    expect(reconciled.updated_at).toBe(1500)
    expect(reconciled.synced).toBe(true)
  })
})
