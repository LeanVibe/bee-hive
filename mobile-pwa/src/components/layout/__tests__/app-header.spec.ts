import { describe, it, expect, vi, beforeEach } from 'vitest'

import '../app-header'

// Lightweight mock for system health service used by app-header
class MockSystemHealthService {
  private listeners = new Map<string, Function[]>()
  async getHealthSummary() {
    return {
      overall: 'healthy',
      components: { healthy: 3, degraded: 0, unhealthy: 0 },
      alerts: []
    }
  }
  addEventListener(event: string, cb: any) {
    const arr = this.listeners.get(event) || []
    arr.push(cb)
    this.listeners.set(event, arr)
  }
  removeEventListener(event: string, cb: any) {
    const arr = (this.listeners.get(event) || []).filter(fn => fn !== cb)
    this.listeners.set(event, arr)
  }
  emit(event: string, detail: any) {
    const arr = this.listeners.get(event) || []
    arr.forEach(fn => fn({ detail }))
  }
}

// Mock module that exports getSystemHealthService
vi.mock('../../services', () => {
  const instance = new MockSystemHealthService()
  return {
    getSystemHealthService: () => instance
  }
})

describe('app-header', () => {
  const mount = async (attrs: Partial<{ currentRoute: string; isOnline: boolean; showMenuButton: boolean }> = {}) => {
    const el = document.createElement('app-header') as any
    if (attrs.currentRoute !== undefined) el.currentRoute = attrs.currentRoute
    if (attrs.isOnline !== undefined) el.isOnline = attrs.isOnline
    if (attrs.showMenuButton !== undefined) el.showMenuButton = attrs.showMenuButton
    document.body.appendChild(el)
    await el.updateComplete
    return el as HTMLElement
  }

  beforeEach(() => {
    document.body.innerHTML = ''
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  it('renders online/offline status and title', async () => {
    const el = await mount({ currentRoute: '/dashboard', isOnline: true })
    const title = (el.shadowRoot!.querySelector('.title') as HTMLElement)!.textContent || ''
    expect(title).toContain('Dashboard')
    const online = el.shadowRoot!.querySelector('.status-online')
    expect(online).toBeTruthy()
  })

  it('emits menu-toggle on menu click', async () => {
    const el = await mount({ showMenuButton: true })
    const spy = vi.fn()
    el.addEventListener('menu-toggle', spy)
    ;(el.shadowRoot!.querySelector('.menu-button') as HTMLButtonElement).click()
    expect(spy).toHaveBeenCalled()
  })
})
