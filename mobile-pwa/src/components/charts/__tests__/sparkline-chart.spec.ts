import { describe, it, expect, beforeEach } from 'vitest'

import '../sparkline-chart'

describe('sparkline-chart', () => {
  const mount = async (data: Array<{ value: number; timestamp?: string }> = [], opts: Partial<{ width: number; height: number; interactive: boolean; showArea: boolean; showDots: boolean }> = {}) => {
    const el = document.createElement('sparkline-chart') as any
    el.data = data
    if (opts.width) el.width = opts.width
    if (opts.height) el.height = opts.height
    if (opts.interactive !== undefined) el.interactive = opts.interactive
    if (opts.showArea !== undefined) el.showArea = opts.showArea
    if (opts.showDots !== undefined) el.showDots = opts.showDots
    document.body.appendChild(el)
    await el.updateComplete
    return el as HTMLElement
  }

  beforeEach(() => {
    document.body.innerHTML = ''
  })

  it('renders baseline when no data', async () => {
    const el = await mount([])
    const svg = el.shadowRoot!.querySelector('svg')
    expect(svg).toBeTruthy()
    const value = (el.shadowRoot!.querySelector('.sparkline-value') as HTMLElement).textContent || ''
    expect(value.trim()).toBe('--')
  })

  it('renders path for data and shows formatted value', async () => {
    const el = await mount([{ value: 10 }, { value: 20 }, { value: 15 }], { width: 100, height: 40 })
    const path = el.shadowRoot!.querySelector('.sparkline-path') as SVGPathElement
    expect(path).toBeTruthy()
    const value = (el.shadowRoot!.querySelector('.sparkline-value') as HTMLElement).textContent || ''
    expect(value).toContain('15')
  })
})
