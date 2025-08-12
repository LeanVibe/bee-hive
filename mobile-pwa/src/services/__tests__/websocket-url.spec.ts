import { describe, it, expect, vi } from 'vitest'

vi.stubGlobal('window', {
  location: {
    protocol: 'http:',
    hostname: 'localhost'
  }
})

// Lazy import to use stubbed window
describe('WebSocket URL', () => {
  it('uses /api/dashboard/ws/dashboard in development', async () => {
    const mod = await import('../websocket')
    // access private via any
    const svc: any = new (mod as any).WebSocketService()
    // monkeypatch connect to capture URL computation
    const urls: string[] = []
    ;(svc as any).cleanup = () => {}
    ;(svc as any).handleOpen = () => {}
    ;(svc as any).handleMessage = () => {}
    ;(svc as any).handleClose = () => {}
    ;(svc as any).handleError = () => {}

    const orig = (global as any).WebSocket
    ;(global as any).WebSocket = class {
      url: string
      readyState = 1
      onopen: any; onmessage: any; onclose: any; onerror: any
      constructor(url: string) {
        this.url = url
        urls.push(url)
      }
      close() {}
      send() {}
    }

    await (svc as any).connect()

    expect(urls[0]).toBe('ws://localhost:8000/api/dashboard/ws/dashboard')

    ;(global as any).WebSocket = orig
  })
})
