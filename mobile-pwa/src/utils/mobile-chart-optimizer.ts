/**
 * Mobile-Specific Chart Optimizations
 * Touch interactions, responsive design, and mobile performance optimizations
 */

export class MobileChartOptimizer {
  private static instance: MobileChartOptimizer;
  private touchStartTime: number = 0;
  private lastTouchEnd: number = 0;
  private isDoubleTap: boolean = false;

  static getInstance() {
    if (!this.instance) {
      this.instance = new MobileChartOptimizer();
    }
    return this.instance;
  }

  // Optimize chart for mobile devices
  static optimizeForMobile(chart: Chart) {
    const instance = MobileChartOptimizer.getInstance();
    
    // Add touch event handlers
    instance.addTouchHandlers(chart);
    
    // Optimize for different screen sizes
    instance.optimizeForScreenSize(chart);
    
    // Add gesture support
    instance.addGestureSupport(chart);
    
    // Optimize performance for mobile
    instance.optimizePerformance(chart);
  }

  private addTouchHandlers(chart: Chart) {
    const canvas = chart.canvas;
    
    // Prevent default touch behaviors that interfere with chart interaction
    canvas.addEventListener('touchstart', (e) => {
      this.touchStartTime = Date.now();
      
      // Check for double tap
      const timeSinceLastTap = this.touchStartTime - this.lastTouchEnd;
      if (timeSinceLastTap < 300 && timeSinceLastTap > 0) {
        this.isDoubleTap = true;
        this.handleDoubleTap(chart, e);
      }
      
      // Prevent scrolling when touching the chart
      e.preventDefault();
    }, { passive: false });
    
    canvas.addEventListener('touchend', (e) => {
      this.lastTouchEnd = Date.now();
      const touchDuration = this.lastTouchEnd - this.touchStartTime;
      
      if (!this.isDoubleTap && touchDuration < 200) {
        // Short tap - show tooltip
        this.handleTap(chart, e);
      }
      
      this.isDoubleTap = false;
    });
    
    // Add long press for context menu
    let longPressTimer: number;
    
    canvas.addEventListener('touchstart', (e) => {
      longPressTimer = setTimeout(() => {
        this.handleLongPress(chart, e);
      }, 800);
    });
    
    canvas.addEventListener('touchend', () => {
      clearTimeout(longPressTimer);
    });
    
    canvas.addEventListener('touchmove', () => {
      clearTimeout(longPressTimer);
    });
  }

  private handleTap(chart: Chart, event: TouchEvent) {
    const points = chart.getElementsAtEventForMode(
      event,
      'nearest',
      { intersect: true },
      true
    );
    
    if (points.length) {
      const point = points[0];
      const datasetIndex = point.datasetIndex;
      const index = point.index;
      
      // Show mobile-optimized tooltip
      this.showMobileTooltip(chart, datasetIndex, index, event);
    }
  }

  private handleDoubleTap(chart: Chart, event: TouchEvent) {
    // Reset zoom on double tap
    if (chart.options.scales?.x?.min || chart.options.scales?.y?.min) {
      chart.resetZoom();
    } else {
      // Zoom to data if not zoomed
      this.zoomToData(chart);
    }
  }

  private handleLongPress(chart: Chart, event: TouchEvent) {
    // Show context menu for data export or other actions
    this.showContextMenu(chart, event);
  }

  private showMobileTooltip(chart: Chart, datasetIndex: number, index: number, event: TouchEvent) {
    const dataset = chart.data.datasets[datasetIndex];
    const dataPoint = dataset.data[index];
    
    // Create or update mobile tooltip
    let tooltip = document.getElementById('mobile-chart-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'mobile-chart-tooltip';
      tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        pointer-events: none;
        z-index: 1000;
        max-width: 200px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
      `;
      document.body.appendChild(tooltip);
    }
    
    const value = typeof dataPoint === 'object' ? (dataPoint as any).y : dataPoint;
    tooltip.innerHTML = `
      <strong>${dataset.label}</strong><br>
      Value: ${value}<br>
      <small>Tap outside to close</small>
    `;
    
    // Position tooltip near touch point
    const touch = event.changedTouches[0];
    const rect = chart.canvas.getBoundingClientRect();
    
    tooltip.style.left = `${touch.clientX - rect.left + 10}px`;
    tooltip.style.top = `${touch.clientY - rect.top - 50}px`;
    tooltip.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
      if (tooltip) {
        tooltip.style.display = 'none';
      }
    }, 3000);
    
    // Hide on next touch outside
    const hideTooltip = (e: TouchEvent) => {
      if (!chart.canvas.contains(e.target as Node)) {
        if (tooltip) {
          tooltip.style.display = 'none';
        }
        document.removeEventListener('touchstart', hideTooltip);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('touchstart', hideTooltip, { once: true });
    }, 100);
  }

  private showContextMenu(chart: Chart, event: TouchEvent) {
    const touch = event.changedTouches[0];
    
    // Create context menu for mobile
    const menu = document.createElement('div');
    menu.style.cssText = `
      position: fixed;
      background: white;
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 8px 0;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 1001;
      min-width: 150px;
    `;
    
    const options = [
      { label: 'Export Data', action: () => this.exportChartData(chart) },
      { label: 'Reset Zoom', action: () => chart.resetZoom() },
      { label: 'Toggle Animation', action: () => this.toggleAnimation(chart) }
    ];
    
    options.forEach(option => {
      const item = document.createElement('div');
      item.style.cssText = `
        padding: 12px 16px;
        cursor: pointer;
        font-size: 16px;
        border-bottom: 1px solid #eee;
      `;
      item.textContent = option.label;
      
      item.addEventListener('touchend', (e) => {
        e.preventDefault();
        option.action();
        document.body.removeChild(menu);
      });
      
      menu.appendChild(item);
    });
    
    // Position menu
    menu.style.left = `${Math.min(touch.clientX, window.innerWidth - 150)}px`;
    menu.style.top = `${Math.min(touch.clientY, window.innerHeight - 120)}px`;
    
    document.body.appendChild(menu);
    
    // Remove menu on outside touch
    const removeMenu = (e: TouchEvent) => {
      if (!menu.contains(e.target as Node)) {
        document.body.removeChild(menu);
        document.removeEventListener('touchstart', removeMenu);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('touchstart', removeMenu);
    }, 100);
  }

  private zoomToData(chart: Chart) {
    // Find data bounds
    const datasets = chart.data.datasets;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    datasets.forEach(dataset => {
      if (Array.isArray(dataset.data)) {
        dataset.data.forEach(point => {
          const x = typeof point === 'object' ? (point as any).x : 0;
          const y = typeof point === 'object' ? (point as any).y : point;
          
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y as number);
          maxY = Math.max(maxY, y as number);
        });
      }
    });
    
    // Add 10% padding
    const xPadding = (maxX - minX) * 0.1;
    const yPadding = (maxY - minY) * 0.1;
    
    chart.zoomScale('x', { min: minX - xPadding, max: maxX + xPadding });
    chart.zoomScale('y', { min: minY - yPadding, max: maxY + yPadding });
  }

  private exportChartData(chart: Chart) {
    const data = chart.data;
    const csvContent = this.convertToCSV(data);
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chart-data.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  private convertToCSV(data: any): string {
    const headers = ['Dataset', 'Index', 'Value'];
    const rows = [headers.join(',')];
    
    data.datasets.forEach((dataset: any, datasetIndex: number) => {
      if (Array.isArray(dataset.data)) {
        dataset.data.forEach((point: any, index: number) => {
          const value = typeof point === 'object' ? point.y : point;
          rows.push([dataset.label || `Dataset ${datasetIndex}`, index, value].join(','));
        });
      }
    });
    
    return rows.join('\n');
  }

  private toggleAnimation(chart: Chart) {
    const currentDuration = chart.options.animation?.duration || 0;
    
    if (typeof chart.options.animation === 'object') {
      chart.options.animation.duration = currentDuration > 0 ? 0 : 300;
    } else {
      chart.options.animation = { duration: currentDuration > 0 ? 0 : 300 };
    }
    
    chart.update();
  }

  private addGestureSupport(chart: Chart) {
    // Add pinch-to-zoom support
    let initialDistance = 0;
    let initialScale = { x: 1, y: 1 };
    
    const canvas = chart.canvas;
    
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 2) {
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        
        initialDistance = Math.sqrt(
          Math.pow(touch2.clientX - touch1.clientX, 2) +
          Math.pow(touch2.clientY - touch1.clientY, 2)
        );
        
        // Store initial scale
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;
        
        initialScale = {
          x: xScale ? (xScale.max - xScale.min) : 1,
          y: yScale ? (yScale.max - yScale.min) : 1
        };
        
        e.preventDefault();
      }
    }, { passive: false });
    
    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2) {
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        
        const currentDistance = Math.sqrt(
          Math.pow(touch2.clientX - touch1.clientX, 2) +
          Math.pow(touch2.clientY - touch1.clientY, 2)
        );
        
        const scale = currentDistance / initialDistance;
        
        // Apply zoom
        if (scale !== 1) {
          const xScale = chart.scales.x;
          const yScale = chart.scales.y;
          
          if (xScale && yScale) {
            const newXRange = initialScale.x / scale;
            const newYRange = initialScale.y / scale;
            
            const centerX = (xScale.max + xScale.min) / 2;
            const centerY = (yScale.max + yScale.min) / 2;
            
            chart.zoomScale('x', {
              min: centerX - newXRange / 2,
              max: centerX + newXRange / 2
            });
            
            chart.zoomScale('y', {
              min: centerY - newYRange / 2,
              max: centerY + newYRange / 2
            });
          }
        }
        
        e.preventDefault();
      }
    }, { passive: false });
  }

  private optimizeForScreenSize(chart: Chart) {
    const canvas = chart.canvas;
    const rect = canvas.getBoundingClientRect();
    
    // Adjust chart options based on screen size
    if (window.innerWidth < 768) { // Mobile
      // Reduce padding for small screens
      if (chart.options.layout?.padding) {
        const padding = chart.options.layout.padding as any;
        chart.options.layout.padding = {
          top: Math.min(padding.top || 0, 10),
          right: Math.min(padding.right || 0, 10),
          bottom: Math.min(padding.bottom || 0, 10),
          left: Math.min(padding.left || 0, 10)
        };
      }
      
      // Adjust legend position for mobile
      if (chart.options.plugins?.legend) {
        chart.options.plugins.legend.position = 'bottom';
      }
      
      // Reduce tick count for mobile
      if (chart.options.scales?.x?.ticks) {
        chart.options.scales.x.ticks.maxTicksLimit = 6;
      }
      
      if (chart.options.scales?.y?.ticks) {
        chart.options.scales.y.ticks.maxTicksLimit = 6;
      }
    }
    
    chart.update();
  }

  private optimizePerformance(chart: Chart) {
    // Disable animations on low-end devices
    if (this.isLowEndDevice()) {
      if (typeof chart.options.animation === 'object') {
        chart.options.animation.duration = 0;
      }
      
      // Reduce point radius for better performance
      if (chart.options.elements?.point) {
        chart.options.elements.point.radius = 0;
      }
    }
    
    // Add performance monitoring
    let lastUpdateTime = 0;
    const originalUpdate = chart.update.bind(chart);
    
    chart.update = (mode?: any) => {
      const startTime = performance.now();
      const result = originalUpdate(mode);
      const updateTime = performance.now() - startTime;
      
      if (updateTime > 16 && Date.now() - lastUpdateTime > 5000) {
        console.warn(`Chart update took ${updateTime.toFixed(2)}ms (target: <16ms)`);
        lastUpdateTime = Date.now();
      }
      
      return result;
    };
  }

  private isLowEndDevice(): boolean {
    // Simple heuristic for low-end device detection
    if ('hardwareConcurrency' in navigator) {
      return navigator.hardwareConcurrency <= 2;
    }
    
    // Fallback: assume devices with small screens and mobile user agents are low-end
    return window.innerWidth < 768 && /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  }
}

// Auto-optimize all charts for mobile when they're created
if (typeof window !== 'undefined') {
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  
  HTMLCanvasElement.prototype.getContext = function(this: HTMLCanvasElement, ...args: any[]) {
    const context = originalGetContext.apply(this, args as any);
    
    // If this is a Chart.js canvas, optimize it for mobile
    setTimeout(() => {
      const chart = (this as any).chart;
      if (chart && chart.constructor.name === 'Chart') {
        MobileChartOptimizer.optimizeForMobile(chart);
      }
    }, 100);
    
    return context;
  };
}

export { MobileChartOptimizer };