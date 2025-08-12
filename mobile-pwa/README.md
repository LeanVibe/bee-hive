# 📱 LeanVibe Agent Hive 2.0 - Mobile PWA Dashboard

**Professional mobile-first dashboard for autonomous development platform management**

[![PWA Ready](https://img.shields.io/badge/PWA-Ready-success.svg)]()
[![Mobile First](https://img.shields.io/badge/Mobile-First-blue.svg)]()
[![Real-time](https://img.shields.io/badge/Real--time-Updates-green.svg)]()
[![Testing](https://img.shields.io/badge/Testing-Comprehensive-success.svg)]()

---

## 🚀 Overview

The Mobile PWA Dashboard is the command center for the LeanVibe Agent Hive 2.0 autonomous development platform. Built with modern web technologies, it provides real-time visibility and control over AI agents, tasks, and system health from any device.

### ✨ Key Features

- **📱 Mobile-First Design**: Responsive interface optimized for mobile, tablet, and desktop
- **⚡ Real-time Updates**: Live WebSocket integration with polling fallback
- **🤖 Agent Management**: Complete agent activation, configuration, and monitoring
- **📋 Task Management**: Kanban board with drag-and-drop task orchestration
- **📊 System Health**: Live performance metrics and monitoring
- **🔄 Offline Support**: Offline-first architecture with sync capabilities
- **🎯 PWA Features**: Installable, push notifications, offline functionality

---

## 🏗️ Architecture

### Tech Stack
- **Framework**: Lit (Lightweight, fast web components)
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS for utility-first design
- **Build Tool**: Vite for fast development and production builds
- **Testing**: Playwright for comprehensive end-to-end testing
- **PWA**: Service Worker, Web App Manifest, push notifications

### Project Structure
```
mobile-pwa/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── charts/       # Data visualization components
│   │   ├── common/       # Shared components (error, loading)
│   │   ├── dashboard/    # Dashboard-specific components
│   │   ├── kanban/       # Task management components
│   │   ├── layout/       # Layout and navigation components
│   │   ├── mobile/       # Mobile-specific interactions
│   │   └── modals/       # Modal dialogs
│   ├── services/         # API integration and business logic
│   ├── views/           # Page-level components
│   ├── types/           # TypeScript type definitions
│   ├── utils/           # Utility functions
│   └── styles/          # Global styles
├── tests/               # Comprehensive test suite
│   ├── e2e/            # End-to-end tests
│   ├── fixtures/       # Test fixtures and page objects
│   └── utils/          # Test utilities
└── public/             # Static assets and PWA files
```

---

## 🚀 Quick Start

Ensure the backend is running per `../docs/GETTING_STARTED.md`. Then:

### Prerequisites
- Node.js 18+
- LeanVibe Agent Hive 2.0 backend running on port 8000

### Development Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3001
```

### Production Build
```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Deploy (static files in dist/)
npm run deploy
```

---

## 📱 Features Deep Dive

### Agent Management
- **Team Activation**: One-click activation of 5-agent development teams
- **Individual Control**: Spawn, configure, and monitor individual agents
- **Performance Metrics**: Real-time CPU, memory, and task performance tracking
- **Status Monitoring**: Live agent status with health indicators

### Task Management  
- **Kanban Board**: 4-column workflow (Pending → In Progress → Review → Done)
- **Drag & Drop**: Intuitive task management with optimistic updates
- **Real-time Sync**: Live task updates across all connected clients
- **Filtering & Search**: Advanced task filtering by status, agent, priority

### System Health
- **Live Metrics**: CPU, memory, task completion rates
- **Event Timeline**: Real-time system events with filtering
- **Sync Status**: Connection health and data sync indicators
- **Performance Dashboards**: Historical trends and system analytics

### Mobile Experience
- **Touch-Optimized**: Large touch targets, gesture support
- **Pull-to-Refresh**: Native mobile refresh patterns
- **Offline Mode**: Full functionality when disconnected
- **Push Notifications**: Real-time alerts for critical events

---

## 🔧 Development

### Available Scripts
```bash
npm run dev              # Start development server
npm run build           # Production build
npm run preview         # Preview production build
npm run lint           # Lint TypeScript and CSS
npm run type-check     # TypeScript type checking
npm run test:e2e       # Run end-to-end tests
npm run test:e2e:ui    # Run tests with UI
```

### Component Development
The dashboard uses Lit web components for maximum performance and browser compatibility:

```typescript
// Example component
import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

@customElement('agent-card')
export class AgentCard extends LitElement {
  @property({ type: Object }) agent!: Agent;

  render() {
    return html`
      <div class="agent-card">
        <h3>${this.agent.name}</h3>
        <span class="status ${this.agent.status}">${this.agent.status}</span>
      </div>
    `;
  }
}
```

### Service Integration
All API integration happens through service classes with error handling and caching:

```typescript
// Example service usage
import { agentService } from './services';

// Activate agent team
const result = await agentService.activateAgentTeam({ teamSize: 5 });

// Listen for real-time updates
agentService.on('agentStatusChanged', (data) => {
  // Update UI
});
```

---

## 🧪 Testing

### Comprehensive Test Suite
The dashboard includes extensive Playwright testing covering:

- **Navigation & UI**: All dashboard views and components
- **Agent Management**: Team activation, individual controls
- **Task Management**: Kanban board, drag-and-drop, CRUD operations
- **Real-time Updates**: WebSocket integration, polling fallback
- **Responsive Design**: Mobile, tablet, desktop breakpoints
- **Error Handling**: Network failures, API errors, recovery
- **Visual Regression**: Pixel-perfect UI validation

### Running Tests
```bash
# Run all tests
npm run test:e2e

# Run specific test suite
npm run test:e2e tests/e2e/agent-management.spec.ts

# Run tests in debug mode
npm run test:e2e:debug

# Generate test report
npm run test:e2e:report
```

### Test Documentation
- **[Testing Guide](README-TESTING.md)**: Comprehensive testing documentation
- **[Test Implementation](TESTING_IMPLEMENTATION_SUMMARY.md)**: Complete test suite overview

---

## 📊 Performance

### Metrics & Benchmarks
- **First Paint**: < 1.5s on 3G networks
- **Interactive**: < 3s on mobile devices
- **Bundle Size**: < 500KB gzipped
- **Lighthouse Score**: 95+ across all categories
- **Memory Usage**: < 50MB for dashboard operations

### PWA Features
- **Install Prompt**: Native app-like installation
- **Offline Mode**: Full functionality without network
- **Background Sync**: Queue operations, sync when online
- **Push Notifications**: Real-time alerts and updates

---

## 🔧 Configuration

### Environment Variables (dev defaults)
```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/api/dashboard/ws/dashboard
VITE_ENABLE_NOTIFICATIONS=true
VITE_OFFLINE_CACHE_DURATION=3600000
```

### Build Configuration
- **Vite Config**: [`vite.config.ts`](vite.config.ts)
- **TypeScript**: [`tsconfig.json`](tsconfig.json)
- **Tailwind**: [`tailwind.config.js`](tailwind.config.js)
- **Playwright**: [`playwright.config.ts`](playwright.config.ts)

---

## 📱 PWA Installation

### Manual Installation
1. Visit dashboard URL in supported browser
2. Look for "Install App" prompt or menu option
3. Click "Install" to add to home screen
4. Launch as native app from home screen

### Browser Support
- ✅ Chrome/Chromium 90+
- ✅ Firefox 90+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

---

## 🚨 Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend development server
npm run dev
```

#### Real-time Updates Not Working
- Verify WebSocket connection to backend
- Check browser console for connection errors
- Ensure firewall allows WebSocket connections

#### PWA Installation Issues
- Ensure HTTPS in production
- Verify service worker registration
- Check Web App Manifest is valid

### Debug Mode
```bash
# Enable debug logging
VITE_DEBUG=true npm run dev

# Run with verbose logging
DEBUG=* npm run dev
```

---

## 🔗 Integration

### Backend API Integration
The dashboard integrates with the backend API/WS:

- **Health**: `GET /health`
- **Live data (compat)**: `GET /dashboard/api/live-data`
- **WebSocket**: `GET /api/dashboard/ws/dashboard` (multi-subscription)

### Authentication (Postponed)
Authentication features are implemented but currently disabled per project requirements. When enabled:
- JWT token-based authentication
- WebAuthn biometric login support
- Session management and refresh

---

## 📚 Related Documentation

- **[Main Project](../README.md)**: LeanVibe Agent Hive 2.0 overview
- **Navigation index](../docs/NAV_INDEX.md)**: Repo navigation
- **[Testing Guide](README-TESTING.md)**: Complete testing documentation
- **[CLAUDE.md](../CLAUDE.md)**: Development guidelines and instructions

---

## 🤝 Contributing

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with tests
3. Run full test suite: `npm run test:e2e`
4. Submit PR with test coverage

### Code Standards
- TypeScript strict mode enabled
- ESLint + Prettier for code formatting
- Lit component patterns
- Comprehensive test coverage required

---

## 📈 Roadmap

### Current Status: ✅ Production Ready
- Complete agent and task management
- Real-time updates and monitoring
- Comprehensive test coverage
- PWA features fully implemented

### Future Enhancements
- Advanced analytics and reporting
- Custom dashboard layouts
- Multi-language support
- Enhanced mobile gestures

---

**🚀 Ready to manage your autonomous development platform from anywhere!**

The Mobile PWA Dashboard provides professional, real-time control over your AI agent teams with enterprise-grade reliability and mobile-first user experience.