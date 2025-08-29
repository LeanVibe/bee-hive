/**
 * Advanced Dynamic Imports for PWA Performance
 * Implements intelligent code splitting and lazy loading
 */

// Lazy load heavy dependencies
export const loadChartJS = () => 
  import('chart.js').then(module => module.default);

export const loadSortableJS = () => 
  import('sortablejs').then(module => module.default);

export const loadDateFns = () => 
  import('date-fns');

// Lazy load components with error boundaries
export const loadDashboardComponents = () => 
  Promise.all([
    import('../views/dashboard-view.ts'),
    import('../components/dashboard/performance-metrics-panel.ts'),
    import('../components/charts/sparkline-chart.ts')
  ]);

export const loadTaskComponents = () => 
  Promise.all([
    import('../views/tasks-view.ts'),
    import('../components/kanban/kanban-board.ts'),
    import('../components/kanban/task-card.ts')
  ]);

export const loadAgentComponents = () => 
  Promise.all([
    import('../views/agents-view.ts'),
    import('../components/dashboard/agent-health-panel.ts'),
    import('../components/dashboard/live-agent-monitor.ts')
  ]);

// Route-based component lazy loading
export const routeComponents = new Map([
  ['/', () => loadDashboardComponents()],
  ['/dashboard', () => loadDashboardComponents()],
  ['/tasks', () => loadTaskComponents()],
  ['/agents', () => loadAgentComponents()]
]);

// Intelligent preloading based on user behavior
export class IntelligentPreloader {
  private static instance: IntelligentPreloader;
  private loadedRoutes = new Set<string>();
  private preloadQueue: string[] = [];

  static getInstance() {
    if (!this.instance) {
      this.instance = new IntelligentPreloader();
    }
    return this.instance;
  }

  preloadRoute(route: string) {
    if (this.loadedRoutes.has(route) || this.preloadQueue.includes(route)) {
      return;
    }

    this.preloadQueue.push(route);
    
    // Use requestIdleCallback for non-blocking preloading
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => this.executePreload(route));
    } else {
      setTimeout(() => this.executePreload(route), 100);
    }
  }

  private async executePreload(route: string) {
    try {
      const loader = routeComponents.get(route);
      if (loader) {
        await loader();
        this.loadedRoutes.add(route);
        this.preloadQueue = this.preloadQueue.filter(r => r !== route);
        console.log(`Preloaded components for route: ${route}`);
      }
    } catch (error) {
      console.error(`Failed to preload route ${route}:`, error);
    }
  }

  // Preload based on user interaction patterns
  onUserInteraction(targetRoute: string) {
    // Preload likely next routes based on current route
    const currentRoute = window.location.pathname;
    
    const routePatterns = {
      '/': ['/dashboard', '/tasks'],
      '/dashboard': ['/tasks', '/agents'],
      '/tasks': ['/agents', '/dashboard'],
      '/agents': ['/dashboard', '/tasks']
    };

    const nextRoutes = routePatterns[currentRoute as keyof typeof routePatterns];
    if (nextRoutes) {
      nextRoutes.forEach(route => this.preloadRoute(route));
    }
  }
}

// Web Vitals optimization
export const optimizeWebVitals = async () => {
  try {
    const { getCLS, getFID, getFCP, getLCP, getTTFB } = await import('web-vitals');
    
    // Track Core Web Vitals with mobile-optimized thresholds
    getCLS((metric) => {
      if (metric.value > 0.1) { // Good CLS threshold
        console.warn('CLS needs improvement:', metric.value);
      }
    });

    getFID((metric) => {
      if (metric.value > 100) { // Good FID threshold
        console.warn('FID needs improvement:', metric.value);
      }
    });

    getFCP((metric) => {
      if (metric.value > 1800) { // Good FCP threshold for mobile
        console.warn('FCP needs improvement:', metric.value);
      }
    });

    getLCP((metric) => {
      if (metric.value > 2500) { // Good LCP threshold for mobile
        console.warn('LCP needs improvement:', metric.value);
      }
    });

    getTTFB((metric) => {
      if (metric.value > 800) { // Good TTFB threshold for mobile
        console.warn('TTFB needs improvement:', metric.value);
      }
    });

  } catch (error) {
    console.error('Failed to load web-vitals:', error);
  }
};