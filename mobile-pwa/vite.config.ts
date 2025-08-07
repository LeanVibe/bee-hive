import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'prompt',
      injectRegister: 'auto',
      devOptions: {
        enabled: false  // Disable in development to fix the error
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        maximumFileSizeToCacheInBytes: 5000000, // 5MB
        cleanupOutdatedCaches: true,
        skipWaiting: true,
        clientsClaim: true,
        runtimeCaching: [
          // API Cache - Stale While Revalidate for dynamic data
          {
            urlPattern: /^https?:\/\/.*\/dashboard\/api\/(live-data).*$/,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-dynamic-cache',
              expiration: {
                maxEntries: 500,
                maxAgeSeconds: 60 * 60 * 2 // 2 hours for dynamic data
              },
              cacheKeyWillBeUsed: async ({ request }) => {
                // Remove auth headers from cache key
                const url = new URL(request.url)
                return url.pathname + url.search
              }
            }
          },
          // WebSocket endpoints - Network Only
          {
            urlPattern: /^wss?:\/\/.*$/,
            handler: 'NetworkOnly'
          },
          // Static API resources - Cache First
          {
            urlPattern: /^https?:\/\/.*\/dashboard\/api\/(health|status|version).*$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'api-static-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60 * 24 // 24 hours for static resources
              }
            }
          },
          // Images - Cache First with fallback
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'images-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24 * 7 // 7 days
              }
            }
          },
          // Fonts - Cache First
          {
            urlPattern: /\.(?:woff|woff2|eot|ttf|otf)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'fonts-cache',
              expiration: {
                maxEntries: 20,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
              }
            }
          }
        ],
        // Background sync for offline operations
        offlineGoogleAnalytics: false,
        // Add notification handlers
        additionalManifestEntries: [
          { url: 'offline.html', revision: '1' }
        ]
      },
      manifest: {
        name: 'LeanVibe Agent Hive - Executive Command Center',
        short_name: 'Agent Hive',
        description: 'Executive mobile command center for multi-agent development operations. Monitor builds, manage tasks, and control AI agents on-the-go.',
        theme_color: '#1e40af',
        background_color: '#0f172a',
        display: 'standalone',
        display_override: ['window-controls-overlay', 'standalone', 'minimal-ui'],
        orientation: 'portrait-primary',
        scope: '/',
        start_url: '/?source=pwa',
        lang: 'en',
        dir: 'ltr',
        categories: ['productivity', 'developer', 'business', 'utilities'],
        icons: [
          {
            src: '/icons/icon-72x72.png',
            sizes: '72x72',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-96x96.png',
            sizes: '96x96',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-128x128.png',
            sizes: '128x128',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-144x144.png',
            sizes: '144x144',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-152x152.png',
            sizes: '152x152',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-384x384.png',
            sizes: '384x384',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable any'
          }
        ],
        shortcuts: [
          {
            name: 'Dashboard',
            short_name: 'Dashboard',
            description: 'View system overview',
            url: '/',
            icons: [{ src: '/icons/icon-96x96.png', sizes: '96x96' }]
          },
          {
            name: 'Tasks',
            short_name: 'Tasks',
            description: 'Manage development tasks',
            url: '/tasks',
            icons: [{ src: '/icons/icon-96x96.png', sizes: '96x96' }]
          }
        ]
      }
    })
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    open: false,
    cors: true,
    // Allow access from Tailscale network
    allowedHosts: [
      'code-mb16-1.taild7760a.ts.net',
      '100.107.91.78',
      'localhost',
      '127.0.0.1'
    ],
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    },
    proxy: {
      '/dashboard/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        headers: {
          'Host': 'localhost:8000'
        }
      },
      '/dashboard/simple-ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
      // Proxy API calls to backend for Tailscale access
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        headers: {
          'Host': 'localhost:8000'
        }
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        headers: {
          'Host': 'localhost:8000'
        }
      }
    },
  },
  build: {
    target: 'es2020',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      },
      output: {
        manualChunks: {
          vendor: ['lit', 'zustand'],
          utils: ['date-fns', 'sortablejs', 'idb']
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },
  optimizeDeps: {
    include: ['lit', 'zustand', 'date-fns', 'sortablejs', 'idb'],
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
  },
  esbuild: {
    target: 'es2020',
    format: 'esm'
  },
  clearScreen: false
})