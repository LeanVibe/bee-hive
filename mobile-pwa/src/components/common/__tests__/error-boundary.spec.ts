import { describe, it, expect, vi } from 'vitest'

import '../error-boundary'

describe('error-boundary', () => {
  const mount = async () => {
    const el = document.createElement('error-boundary') as any
    document.body.appendChild(el)
    await el.updateComplete
    return el as HTMLElement & { showError: (e: Error) => void }
  }

  it('renders slot content when no error', async () => {
    const el = await mount()
    el.innerHTML = '<div id="content">hello</div>'
    await (el as any).updateComplete
    const content = el.shadowRoot!.querySelector('slot')
    expect(content).toBeTruthy()
  })

  it('shows error UI when showError is called', async () => {
    const el = await mount()
    ;(el as any).showError(new Error('Boom'))
    await (el as any).updateComplete
    const title = el.shadowRoot!.querySelector('.error-title') as HTMLElement
    expect(title?.textContent).toContain('Something went wrong')
    const message = el.shadowRoot!.querySelector('.error-message') as HTMLElement
    expect(message?.textContent).toContain('Boom')
  })

  it('reloads on button click', async () => {
    const el = await mount()
    ;(el as any).showError(new Error('Boom'))
    await (el as any).updateComplete

    const originalLocation = window.location
    Object.defineProperty(window, 'location', {
      value: { ...originalLocation, reload: vi.fn() },
      configurable: true,
    })

    const btn = el.shadowRoot!.querySelector('.btn-primary') as HTMLButtonElement
    btn.click()

    expect((window.location.reload as unknown as jest.Mock | any)).toHaveBeenCalled()

    Object.defineProperty(window, 'location', { value: originalLocation, configurable: true })
  })
})
