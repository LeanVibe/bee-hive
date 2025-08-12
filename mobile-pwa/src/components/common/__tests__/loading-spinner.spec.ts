import { describe, it, expect } from 'vitest'

import '../loading-spinner'

describe('loading-spinner', () => {
  const mount = async (attrs: Partial<{ size: 'small'|'medium'|'large'; color: string; text: string }> = {}) => {
    const el = document.createElement('loading-spinner') as any
    if (attrs.size) el.size = attrs.size
    if (attrs.color) el.color = attrs.color
    if (attrs.text) el.text = attrs.text
    document.body.appendChild(el)
    await (el as any).updateComplete
    return el as HTMLElement & { size: 'small'|'medium'|'large'; color: string; text: string }
  }

  it('renders with default props', async () => {
    const el = await mount()
    const spinner = el.shadowRoot!.querySelector('.spinner') as HTMLElement
    expect(spinner).toBeTruthy()
    expect(spinner.classList.contains('medium')).toBe(true)
  })

  it('applies size, color and text', async () => {
    const el = await mount({ size: 'large', color: '#ff0000', text: 'Loading data…' })
    const spinner = el.shadowRoot!.querySelector('.spinner') as HTMLElement
    const text = el.shadowRoot!.querySelector('.text') as HTMLElement
    expect(spinner.classList.contains('large')).toBe(true)
    expect(spinner.getAttribute('style')).toContain('--spinner-color: #ff0000')
    expect(text.textContent).toContain('Loading data…')
  })
})
