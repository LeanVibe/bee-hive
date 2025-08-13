import { describe, it, expect, vi } from 'vitest'
import '../../../app'

// Minimal custom element harness

describe('Version mismatch banner', () => {
  it('renders persistent banner when version-mismatch event is received', async () => {
    // jsdom stubs
    // Provide missing DOM APIs for jsdom
    ;(global as any).window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      onchange: null,
      media: ''
    })
    ;(global as any).window.addEventListener = (global as any).window.addEventListener || (() => {})
    ;(global as any).window.removeEventListener = (global as any).window.removeEventListener || (() => {})

    const app = document.createElement('agent-hive-app') as any
    // Avoid heavy initializeApp during test
    app.initializeApp = async () => {}
    document.body.appendChild(app)

    // simulate event from ws service by toggling state directly
    app.versionMismatchActive = true
    app.versionMismatchMessage = 'Version mismatch: server 2.0.0 · supported: 1.x'
    await app.updateComplete

    const banners = Array.from(app.shadowRoot!.querySelectorAll('.error-banner'))
    const hasVersionBanner = banners.some((b: Element) => b.textContent?.includes('Version mismatch'))
    expect(hasVersionBanner).toBe(true)

    // Ensure it does not auto-hide (no timer involved); still present after a tick
    await new Promise(r => setTimeout(r, 10))
    const stillThere = Array.from(app.shadowRoot!.querySelectorAll('.error-banner')).some((b: Element) => b.textContent?.includes('Version mismatch'))
    expect(stillThere).toBe(true)
  })
})
