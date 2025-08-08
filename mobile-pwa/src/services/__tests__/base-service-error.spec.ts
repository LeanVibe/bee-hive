import { describe, it, expect, vi } from 'vitest'
import { BaseService } from '../base-service'

class TService extends BaseService {
  public async getWrap(url: string) {
    // expose protected get for testing
    // @ts-expect-error access protected for test
    return this.get(url)
  }
}

describe('BaseService error mapping', () => {
  it('maps TypeError to NETWORK_ERROR ApiError', async () => {
    const svc = new TService({ baseUrl: 'http://localhost:8000', timeout: 10000 })

    const err = new TypeError('failed to fetch')
    const fetchMock = vi.spyOn(global, 'fetch' as any).mockRejectedValueOnce(err)

    await expect(svc.getWrap('/never')).rejects.toMatchObject({ code: 'NETWORK_ERROR' })

    fetchMock.mockRestore()
  })
})
