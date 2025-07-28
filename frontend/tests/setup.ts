/**
 * Vitest Test Setup
 * 
 * Global configuration and setup for all tests
 */

import { beforeAll, afterAll, afterEach, vi } from 'vitest'
import { config } from '@vue/test-utils'

// Mock global objects that might not be available in JSDOM
beforeAll(() => {
  // Mock ResizeObserver
  global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // Mock IntersectionObserver
  global.IntersectionObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // Mock requestAnimationFrame
  global.requestAnimationFrame = vi.fn().mockImplementation((cb) => {
    return setTimeout(cb, 16)
  })

  // Mock cancelAnimationFrame
  global.cancelAnimationFrame = vi.fn().mockImplementation((id) => {
    clearTimeout(id)
  })

  // Mock performance.now
  Object.defineProperty(global.performance, 'now', {
    value: vi.fn(() => Date.now()),
  })

  // Mock performance.memory (used by performance optimization)
  Object.defineProperty(global.performance, 'memory', {
    value: {
      usedJSHeapSize: 50 * 1024 * 1024, // 50MB
      totalJSHeapSize: 100 * 1024 * 1024, // 100MB
      jsHeapSizeLimit: 2 * 1024 * 1024 * 1024, // 2GB
    },
  })

  // Mock WebSocket
  global.WebSocket = vi.fn().mockImplementation(() => ({
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    send: vi.fn(),
    close: vi.fn(),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  }))

  // Mock console methods to reduce noise in tests
  console.warn = vi.fn()
  console.error = vi.fn()
})

// Clean up after each test
afterEach(() => {
  vi.clearAllMocks()
  vi.clearAllTimers()
})

// Global teardown
afterAll(() => {
  vi.restoreAllMocks()
})

// Configure Vue Test Utils globally
config.global.mocks = {
  $router: {
    push: vi.fn(),
    replace: vi.fn(),
    go: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
  },
  $route: {
    path: '/',
    params: {},
    query: {},
    meta: {},
  },
}

// Mock D3 modules that might cause issues in tests
vi.mock('d3', () => ({
  select: vi.fn(() => ({
    selectAll: vi.fn(() => ({
      data: vi.fn(() => ({
        enter: vi.fn(() => ({
          append: vi.fn(() => ({
            attr: vi.fn(() => ({ attr: vi.fn() })),
          })),
        })),
        exit: vi.fn(() => ({ remove: vi.fn() })),
        attr: vi.fn(),
      })),
      attr: vi.fn(),
      style: vi.fn(),
      on: vi.fn(),
    })),
    append: vi.fn(() => ({
      attr: vi.fn(() => ({ attr: vi.fn() })),
      style: vi.fn(() => ({ style: vi.fn() })),
    })),
    attr: vi.fn(),
    style: vi.fn(),
    on: vi.fn(),
    call: vi.fn(),
  })),
  scaleOrdinal: vi.fn(() => ({
    domain: vi.fn(() => ({ range: vi.fn() })),
  })),
  schemeCategory10: ['#1f77b4', '#ff7f0e', '#2ca02c'],
  forceSimulation: vi.fn(() => ({
    nodes: vi.fn(() => ({ links: vi.fn() })),
    force: vi.fn(),
    on: vi.fn(),
    alpha: vi.fn(),
    alphaTarget: vi.fn(),
    restart: vi.fn(),
    stop: vi.fn(),
  })),
  forceLink: vi.fn(() => ({
    id: vi.fn(),
    distance: vi.fn(),
  })),
  forceManyBody: vi.fn(() => ({
    strength: vi.fn(),
  })),
  forceCenter: vi.fn(),
  drag: vi.fn(() => ({
    on: vi.fn(),
  })),
  zoom: vi.fn(() => ({
    scaleExtent: vi.fn(() => ({ on: vi.fn() })),
  })),
  zoomTransform: vi.fn(() => ({
    k: 1,
    x: 0,
    y: 0,
  })),
}))

export {}