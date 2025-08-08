import { describe, it, expect } from 'vitest'
import { BaseService } from '../base-service'

class TestService extends BaseService {
  public format(template: string, params: Record<string, any>) {
    return this.formatEndpoint(template, params)
  }
  public query(params: Record<string, any>) {
    return this.buildQueryString(params)
  }
}

describe('BaseService helpers', () => {
  const svc = new TestService({ baseUrl: '' })

  it('formatEndpoint replaces tokens with encoded values', () => {
    const out = svc.format('/api/items/{id}/name/{name}', { id: 42, name: 'a b' })
    expect(out).toBe('/api/items/42/name/a%20b')
  })

  it('buildQueryString serializes only defined params', () => {
    const out = svc.query({ a: 1, b: null, c: undefined, d: 'x y' })
    // URLSearchParams encodes spaces as + by default in URL query context
    expect(out === '?a=1&d=x%20y' || out === '?a=1&d=x+y').toBe(true)
  })
})
