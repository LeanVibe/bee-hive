import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'
import { resolve } from 'path'
import autoprefixer from 'autoprefixer'
import cssnano from 'cssnano'

export default defineConfig({
  // Production build optimizations
  build: {
    target: ['es2020', 'edge88', 'firefox78', 'chrome87', 'safari13.1'],
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    
    // Performance budgets enforcement
    chunkSizeWarningLimit: 200,
    
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      },
      output: {
        // Optimize chunk splitting for mobile
        manualChunks: {
          // Vendor libraries
          vendor: ['lit', 'firebase/messaging', 'firebase/app'],
          // Dashboard components
          dashboard: [
            'src/components/dashboard/realtime-agent-status-panel',
            'src/components/dashboard/performance-analytics-panel',
            'src/components/dashboard/security-monitoring-panel'
          ],
          // Services
          services: [
            'src/services/enhanced-coordination',
            'src/services/security-monitoring',
            'src/services/performance-analytics'
          ]
        },
        
        // Asset naming for caching
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.')
          const extType = info[info.length - 1]
          
          // Organize assets by type
          if (/\.(png|jpe?g|gif|svg|webp|avif)$/i.test(assetInfo.name)) {
            return `assets/images/[name].[hash][extname]`
          }
          if (/\.(woff2?|ttf|eot)$/i.test(assetInfo.name)) {
            return `assets/fonts/[name].[hash][extname]`
          }
          if (/\.css$/i.test(assetInfo.name)) {
            return `assets/styles/[name].[hash][extname]`
          }
          
          return `assets/[name].[hash][extname]`
        }
      },
      
      // External dependencies (CDN)
      external: process.env.USE_CDN ? [
        'firebase/messaging',
        'firebase/app'
      ] : []
    },
    
    // Terser optimization for mobile
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.debug'],
        passes: 2
      },
      mangle: {
        safari10: true
      },
      format: {
        safari10: true
      }
    }
  },
  
  // CSS processing
  css: {
    postcss: {
      plugins: [
        autoprefixer({
          overrideBrowserslist: [
            'iOS >= 13',
            'Android >= 6',
            'last 2 versions'
          ]
        }),
        cssnano({
          preset: ['default', {
            discardComments: { removeAll: true },
            normalizeWhitespace: true,
            mergeIdents: true,
            minifySelectors: true
          }]
        })
      ]
    }
  },
  
  // PWA configuration for production
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        // Service worker configuration
        globPatterns: ['**/*.{js,css,html,ico,png,svg,webp,avif}'],
        maximumFileSizeToCacheInBytes: 3 * 1024 * 1024, // 3MB
        
        // Runtime caching for mobile optimization
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\./,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 5 * 60 // 5 minutes
              },
              cacheableResponse: {
                statuses: [0, 200]
              },
              networkTimeoutSeconds: 10
            }
          },
          {
            urlPattern: /^https:\/\/ws\./,
            handler: 'NetworkOnly'
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'images-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 24 * 60 * 60 // 24 hours
              }
            }
          },
          {
            urlPattern: /\.(?:js|css|woff2|woff)$/,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'static-resources',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 24 * 60 * 60
              }
            }
          }
        ],
        
        // Skip waiting and claim clients immediately
        skipWaiting: true,
        clientsClaim: true
      },
      
      // Manifest configuration
      manifest: {
        name: 'LeanVibe Agent Hive',
        short_name: 'LeanVibe',
        description: 'Autonomous Development Platform Dashboard',
        theme_color: '#1f2937',
        background_color: '#ffffff',
        display: 'standalone',
        orientation: 'portrait-primary',
        scope: '/',
        start_url: '/',
        
        // Mobile-optimized icons
        icons: [
          {
            src: '/icons/icon-72.png',
            sizes: '72x72',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-96.png',
            sizes: '96x96',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-128.png',
            sizes: '128x128',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-144.png',
            sizes: '144x144',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-152.png',
            sizes: '152x152',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-384.png',
            sizes: '384x384',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: '/icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable any'
          }
        ],
        
        // Mobile-specific features
        categories: ['productivity', 'business', 'developer'],
        screenshots: [
          {
            src: '/screenshots/mobile-dashboard.png',
            sizes: '540x720',
            type: 'image/png',
            form_factor: 'narrow'
          },
          {
            src: '/screenshots/desktop-dashboard.png',
            sizes: '1280x720',
            type: 'image/png',
            form_factor: 'wide'
          }
        ]
      },
      
      // Development options
      devOptions: {
        enabled: false
      }
    })
  ],
  
  // Development server configuration for mobile testing
  server: {
    host: true,
    port: 3000,
    strictPort: true,
    
    // HTTPS for service worker testing
    https: process.env.HTTPS_DEV === 'true',
    
    // Proxy for mobile development
    proxy: {
      '/api': {
        target: process.env.API_BASE_URL || 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: process.env.WS_BASE_URL || 'ws://localhost:8000',
        ws: true,
        changeOrigin: true
      }
    }
  },
  
  // Optimization for mobile devices
  optimizeDeps: {
    include: [
      'lit',
      'firebase/messaging',
      'firebase/app'
    ],
    
    // Pre-bundle for faster mobile loading
    force: true
  },
  
  // Define global constants
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    __BUILD_DATE__: JSON.stringify(new Date().toISOString()),
    __PRODUCTION__: JSON.stringify(true)
  }
})