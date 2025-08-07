/**
 * Mobile Performance Optimization and CDN Configuration
 * For LeanVibe Agent Hive Mobile Dashboard
 */

const CDNOptimization = {
    // CDN configuration for mobile assets
    cdn: {
        baseUrl: process.env.CDN_URL || 'https://cdn.leanvibe.com',
        staticFiles: process.env.STATIC_FILES_CDN || 'https://static.leanvibe.com',
        apiCache: process.env.API_CACHE_CDN || 'https://api-cache.leanvibe.com',
        
        // Mobile-specific optimizations
        mobileOptimizations: {
            webp: true,
            avif: true,
            responsive: true,
            lazy: true,
            compression: 'gzip',
            minification: true
        },
        
        // Cache control policies
        cacheControl: {
            static: 'public, max-age=31536000, immutable',
            api: 'public, max-age=300, s-maxage=600',
            dynamic: 'public, max-age=60',
            private: 'private, no-cache'
        }
    },
    
    // Asset optimization for mobile
    assets: {
        // Critical path resources
        critical: [
            '/app.js',
            '/app.css',
            '/manifest.json',
            '/sw.js'
        ],
        
        // Preload resources for mobile
        preload: [
            { href: '/fonts/inter.woff2', as: 'font', type: 'font/woff2', crossorigin: 'anonymous' },
            { href: '/api/dashboard/status', as: 'fetch', crossorigin: 'anonymous' }
        ],
        
        // Resource hints
        prefetch: [
            '/api/agents/list',
            '/api/coordination/status',
            '/components/dashboard-grid.js'
        ],
        
        // Image optimization
        images: {
            formats: ['avif', 'webp', 'jpg'],
            sizes: [320, 640, 768, 1024, 1440],
            quality: {
                avif: 50,
                webp: 75,
                jpg: 85
            }
        }
    },
    
    // Service Worker configuration for mobile
    serviceWorker: {
        cacheStrategies: {
            // Network first for API calls
            api: 'networkFirst',
            // Cache first for static assets
            static: 'cacheFirst',
            // Stale while revalidate for dashboard data
            dashboard: 'staleWhileRevalidate'
        },
        
        cacheNames: {
            static: 'leanvibe-static-v1',
            api: 'leanvibe-api-v1',
            runtime: 'leanvibe-runtime-v1'
        },
        
        runtimeCaching: [
            {
                urlPattern: /^https:\/\/api\./,
                handler: 'NetworkFirst',
                options: {
                    cacheName: 'api-cache',
                    expiration: {
                        maxEntries: 100,
                        maxAgeSeconds: 300
                    },
                    cacheableResponse: {
                        statuses: [0, 200]
                    }
                }
            },
            {
                urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/,
                handler: 'CacheFirst',
                options: {
                    cacheName: 'images-cache',
                    expiration: {
                        maxEntries: 50,
                        maxAgeSeconds: 86400
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
                        maxAgeSeconds: 86400
                    }
                }
            }
        ]
    },
    
    // Performance budgets for mobile
    performanceBudgets: {
        // Core Web Vitals targets
        lcp: 2500,  // Largest Contentful Paint (ms)
        fid: 100,   // First Input Delay (ms)
        cls: 0.1,   // Cumulative Layout Shift
        
        // Loading performance
        ttfb: 600,  // Time to First Byte (ms)
        fcp: 1800,  // First Contentful Paint (ms)
        tti: 3800,  // Time to Interactive (ms)
        
        // Resource budgets
        totalSize: 500,     // Total page weight (KB)
        jsSize: 200,        // JavaScript bundle size (KB)
        cssSize: 50,        // CSS bundle size (KB)
        imageSize: 200,     // Images total size (KB)
        requestCount: 20    // Total HTTP requests
    },
    
    // Mobile-specific optimizations
    mobile: {
        // Touch optimization
        touchTarget: {
            minSize: 44,  // Minimum touch target size (px)
            spacing: 8    // Minimum spacing between targets (px)
        },
        
        // Viewport optimization
        viewport: {
            width: 'device-width',
            initialScale: 1,
            maximumScale: 5,
            userScalable: 'yes'
        },
        
        // Connection awareness
        networkAdaptation: {
            '2g': {
                reducedMotion: true,
                lowQualityImages: true,
                prefetchDisabled: true
            },
            '3g': {
                reducedAnimations: true,
                standardQualityImages: true,
                limitedPrefetch: true
            },
            '4g': {
                fullAnimations: true,
                highQualityImages: true,
                fullPrefetch: true
            }
        }
    },
    
    // Critical resource inlining
    inline: {
        css: {
            critical: true,
            maxSize: 14000  // 14KB inline CSS limit
        },
        js: {
            critical: false,
            maxSize: 5000   // 5KB inline JS limit
        }
    },
    
    // Compression configuration
    compression: {
        gzip: {
            enabled: true,
            level: 6,
            threshold: 1024
        },
        brotli: {
            enabled: true,
            quality: 6,
            threshold: 1024
        }
    }
};

// Performance monitoring configuration
const PerformanceMonitoring = {
    // Real User Monitoring (RUM)
    rum: {
        enabled: true,
        sampleRate: 0.1,  // 10% of users
        
        // Metrics to collect
        metrics: [
            'navigation-timing',
            'resource-timing',
            'paint-timing',
            'largest-contentful-paint',
            'first-input-delay',
            'cumulative-layout-shift'
        ],
        
        // Custom metrics
        custom: {
            dashboardLoadTime: true,
            apiResponseTime: true,
            websocketLatency: true,
            offlineFunctionality: true
        }
    },
    
    // Performance API usage
    performanceObserver: {
        entryTypes: [
            'navigation',
            'resource',
            'paint',
            'largest-contentful-paint',
            'first-input',
            'layout-shift'
        ]
    },
    
    // Reporting endpoints
    reporting: {
        endpoint: '/api/performance/reports',
        batchSize: 10,
        flushInterval: 30000  // 30 seconds
    }
};

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CDNOptimization, PerformanceMonitoring };
} else if (typeof window !== 'undefined') {
    window.CDNOptimization = CDNOptimization;
    window.PerformanceMonitoring = PerformanceMonitoring;
}