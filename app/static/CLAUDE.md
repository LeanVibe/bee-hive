# CLAUDE.md - Mobile PWA & Frontend Assets

## 🎯 **Context: Mobile PWA Real-time Dashboard**

You are working in the **mobile PWA and frontend asset layer** of LeanVibe Agent Hive 2.0. This directory contains Lit-based web components, real-time dashboard interfaces, and progressive web application features that provide mobile-optimized monitoring and control.

## ✅ **Existing PWA Capabilities (DO NOT REBUILD)**

### **Core Mobile PWA Features Already Implemented**
- **Real-time Dashboard**: Live agent activity monitoring with WebSocket connections
- **Lit Components**: Modern web component architecture with TypeScript support
- **Progressive Web App**: Service worker, offline capabilities, mobile optimization
- **Responsive Design**: Tablet and mobile-optimized layouts
- **Touch Gestures**: Native mobile interactions and gestures
- **WebSocket Integration**: Real-time updates from `/ws/` endpoints

### **Component Library Already Available**
- Dashboard widgets for agent status and metrics
- Real-time data visualization components
- Mobile-optimized navigation and controls
- Notification and alert components
- Touch-friendly interaction elements

## 🔧 **Development Guidelines**

### **Enhancement Strategy (NOT Replacement)**
When improving PWA functionality:

1. **FIRST**: Review existing Lit components and PWA features
2. **ENHANCE** existing components with advanced capabilities
3. **INTEGRATE** with enhanced command ecosystem from `/app/core/`
4. **MAINTAIN** mobile-first design and performance standards

### **Integration with Enhanced Systems**
```typescript
// Pattern for enhancing existing components
import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { EnhancedCommandIntegration } from '../core/command-integration.js';

@customElement('enhanced-agent-dashboard')
export class EnhancedAgentDashboard extends LitElement {
  @property({ type: Boolean }) mobile = false;
  @property({ type: Boolean }) enhanced = false;
  @state() private agentData: AgentStatus[] = [];
  
  private commandIntegration: EnhancedCommandIntegration;
  
  constructor() {
    super();
    this.commandIntegration = new EnhancedCommandIntegration();
  }
  
  async connectedCallback() {
    super.connectedCallback();
    if (this.enhanced) {
      await this.initializeEnhancedFeatures();
    }
    this.setupWebSocketConnection();
  }
  
  private async initializeEnhancedFeatures() {
    // Integrate with enhanced command ecosystem
    const ecosystem = await this.commandIntegration.getEcosystem();
    this.setupEnhancedCommands(ecosystem);
  }
  
  render() {
    return html`
      <div class="dashboard ${this.mobile ? 'mobile' : 'desktop'}">
        ${this.enhanced ? this.renderEnhancedView() : this.renderStandardView()}
      </div>
    `;
  }
  
  static styles = css`
    .dashboard.mobile {
      /* Mobile-first responsive design */
      padding: 8px;
      font-size: 14px;
      touch-action: manipulation;
    }
    
    .dashboard.desktop {
      /* Desktop optimizations */
      padding: 16px;
      font-size: 16px;
    }
  `;
}
```

### **Mobile Optimization Standards**
```typescript
// Touch gesture handling
class TouchGestureManager {
  private startY: number = 0;
  private startTime: number = 0;
  
  handleTouchStart(event: TouchEvent) {
    this.startY = event.touches[0].clientY;
    this.startTime = Date.now();
  }
  
  handleTouchEnd(event: TouchEvent) {
    const endY = event.changedTouches[0].clientY;
    const deltaY = this.startY - endY;
    const deltaTime = Date.now() - this.startTime;
    
    // Pull-to-refresh gesture
    if (deltaY < -100 && deltaTime < 500) {
      this.triggerRefresh();
    }
    
    // Swipe navigation
    if (Math.abs(deltaY) > 50 && deltaTime < 300) {
      this.handleSwipeNavigation(deltaY > 0 ? 'up' : 'down');
    }
  }
}
```

### **Real-time Data Handling**
```typescript
class RealTimeDataManager {
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  async connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
    
    this.websocket = new WebSocket(wsUrl);
    
    this.websocket.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.subscribeToUpdates();
    };
    
    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleRealTimeUpdate(data);
    };
    
    this.websocket.onclose = () => {
      this.handleReconnection();
    };
    
    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
  
  private handleRealTimeUpdate(data: any) {
    // Update dashboard components with real-time data
    this.updateComponents(data);
    
    // Show notifications for critical events
    if (data.type === 'agent_failure') {
      this.showMobileNotification('Agent Alert', data.message);
    }
  }
}
```

## 📱 **PWA Standards**

### **Service Worker Implementation**
```javascript
// service-worker.js
const CACHE_NAME = 'leanvibe-hive-v1';
const STATIC_ASSETS = [
  '/',
  '/static/css/dashboard.css',
  '/static/js/components.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  // Network-first strategy for API calls
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .catch(() => caches.match(event.request))
    );
  } else {
    // Cache-first strategy for static assets
    event.respondWith(
      caches.match(event.request)
        .then(response => response || fetch(event.request))
    );
  }
});
```

### **Manifest Configuration**
```json
{
  "name": "LeanVibe Agent Hive",
  "short_name": "AgentHive",
  "description": "Real-time agent monitoring and control",
  "start_url": "/",
  "display": "standalone",
  "orientation": "portrait-primary",
  "theme_color": "#1f2937",
  "background_color": "#111827",
  "icons": [
    {
      "src": "/static/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

## 🧪 **Testing Requirements**

### **Component Testing**
```typescript
// tests/components/test-enhanced-dashboard.ts
import { fixture, expect, html } from '@open-wc/testing';
import '../src/enhanced-agent-dashboard.js';
import type { EnhancedAgentDashboard } from '../src/enhanced-agent-dashboard.js';

describe('EnhancedAgentDashboard', () => {
  let element: EnhancedAgentDashboard;
  
  beforeEach(async () => {
    element = await fixture(html`
      <enhanced-agent-dashboard 
        .mobile=${true} 
        .enhanced=${true}>
      </enhanced-agent-dashboard>
    `);
  });
  
  it('renders mobile-optimized layout', async () => {
    const dashboard = element.shadowRoot!.querySelector('.dashboard');
    expect(dashboard).to.have.class('mobile');
  });
  
  it('integrates with enhanced command ecosystem', async () => {
    await element.updateComplete;
    const enhancedFeatures = element.shadowRoot!.querySelector('.enhanced-controls');
    expect(enhancedFeatures).to.exist;
  });
  
  it('handles real-time WebSocket updates', async () => {
    const mockData = { type: 'agent_update', agentId: 'test-123' };
    element.handleWebSocketMessage(mockData);
    await element.updateComplete;
    
    const agentElement = element.shadowRoot!.querySelector('[data-agent-id="test-123"]');
    expect(agentElement).to.exist;
  });
});
```

### **PWA Testing**
```javascript
// tests/pwa/test-service-worker.js
describe('Service Worker', () => {
  it('caches static assets on install', async () => {
    const registration = await navigator.serviceWorker.register('/service-worker.js');
    await registration.installing?.postMessage({ type: 'SKIP_WAITING' });
    
    const cache = await caches.open('leanvibe-hive-v1');
    const cachedUrls = await cache.keys();
    expect(cachedUrls.length).to.be.greaterThan(0);
  });
  
  it('provides offline functionality', async () => {
    // Simulate offline condition
    navigator.serviceWorker.controller?.postMessage({ type: 'SIMULATE_OFFLINE' });
    
    const response = await fetch('/');
    expect(response.ok).to.be.true;
  });
});
```

## 🔗 **Integration Points**

### **API Integration** (`/app/api/`)
- WebSocket endpoints: `/ws/dashboard`, `/ws/agents/{id}`
- REST API: `/api/v1/agents`, `/api/v1/tasks`, `/api/v1/metrics`
- Real-time data synchronization

### **Core System Integration** (`/app/core/`)
- Enhanced command ecosystem integration
- Quality gates for mobile performance
- Mobile-optimized command discovery

### **CLI Integration** (`/app/cli/`)
- Shared configuration and authentication
- Mobile QR code generation for CLI access
- Cross-platform command synchronization

## ⚠️ **Critical Guidelines**

### **DO NOT Rebuild Existing PWA**
- All basic PWA functionality exists and works well
- Focus on **enhancement** and **mobile optimization**
- Add AI-powered features to existing components
- Improve real-time capabilities and gesture handling

### **Mobile-First Performance**
- Components load in <2 seconds on 3G networks
- Touch targets are minimum 44px for accessibility
- Smooth 60fps animations and transitions
- Efficient memory usage <100MB for PWA process

### **Progressive Enhancement**
- Core functionality works without JavaScript
- Enhanced features activate progressively
- Graceful degradation for older browsers
- Offline functionality for critical features

## 📋 **Enhancement Priorities**

### **High Priority**
1. **AI-powered insights** in existing dashboard components
2. **Advanced touch gestures** and mobile interactions
3. **Offline synchronization** improvements
4. **Real-time performance** optimization

### **Medium Priority**
5. **Voice command** integration for mobile
6. **Advanced notification** system with native APIs
7. **Multi-device sync** and cross-platform consistency
8. **Enhanced accessibility** features

### **Low Priority**
9. **AR/VR dashboard** views for advanced devices
10. **Machine learning** for predictive mobile UX
11. **Advanced PWA** features (background sync, push notifications)
12. **Custom mobile themes** and personalization

## 🎯 **Success Criteria**

Your PWA enhancements are successful when:
- **Existing functionality** is preserved and enhanced
- **Mobile performance** meets 60fps and <2s load targets
- **Real-time capabilities** provide instant updates
- **Touch interactions** are intuitive and responsive
- **Progressive enhancement** works across all devices
- **Integration** with core systems is seamless

Focus on **enhancing the existing PWA foundation** with AI-powered insights and advanced mobile capabilities.