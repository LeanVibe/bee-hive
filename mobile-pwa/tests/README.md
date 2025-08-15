# E2E Testing Suite - LeanVibe Agent Hive 2.0

Comprehensive end-to-end testing for the LeanVibe Agent Hive dashboard with focus on mobile-first PWA functionality, autonomous agent workflows, and production-grade quality assurance.

## 🎯 Testing Strategy

Our E2E testing suite validates the complete user experience across multiple dimensions:

### Test Categories

#### 🚀 **Smoke Tests** (`smoke/`)
Quick validation of critical functionality for fast feedback loops.
- Dashboard loads and core elements are visible
- Navigation works correctly  
- Basic agent status is displayed
- Performance meets baseline requirements (<2s load time)
- WebSocket connections establish

#### 📱 **PWA Functionality** (`pwa/`)
Progressive Web App features and offline capabilities.
- Manifest validation and installability
- Service worker registration and caching
- Offline mode and functionality
- App-like behavior and native feel
- Storage management and updates

#### 📲 **Mobile Responsiveness** (`mobile/`)
Touch interactions and mobile-optimized UI.
- Touch target sizes (44px minimum compliance)
- Gesture support (tap, swipe, scroll)
- Mobile navigation patterns
- Orientation change handling
- Performance on mobile devices

#### 🎨 **Visual Regression** (`visual/`)
Theme switching and visual consistency.
- Dark/light mode functionality
- Theme persistence across sessions
- Color contrast validation (WCAG AA)
- Component rendering across themes
- High contrast mode compatibility

#### ⚡ **Performance Validation** (`performance/`)
Core Web Vitals and optimization verification.
- Dashboard loads <2s (PRD requirement)
- FCP <1.8s, LCP <2.5s, CLS <0.1
- Bundle size optimization (<2MB total)
- Memory usage monitoring
- Network condition tolerance

#### ♿ **Accessibility Compliance** (`accessibility/`)
WCAG 2.1 AA standards validation.
- Semantic HTML structure
- Keyboard navigation
- Screen reader compatibility
- ARIA attributes and roles
- Focus management
- Color contrast ratios

#### 🤖 **Agent Workflows** (`agent-workflows/`)
End-to-end autonomous development scenarios.
- Task creation and assignment
- Agent spawning and management
- Real-time progress monitoring
- Multi-agent coordination
- Error handling and recovery

#### 🔄 **Real-Time Features** (`websocket-realtime.spec.ts`)
WebSocket and live data functionality.
- Connection establishment and resilience
- Real-time UI updates
- Message handling robustness
- Performance under load
- Connection state management

## 🛠️ Setup & Usage

### Quick Start

```bash
# Install dependencies
npm install

# Install Playwright browsers
npm run test:e2e:install

# Run smoke tests (recommended for development)
npm run test:e2e:smoke

# Run with visible browser
npm run test:e2e:smoke --headed
```

### Test Suites

```bash
# Comprehensive test suites
npm run test:e2e:smoke         # Quick critical path validation
npm run test:e2e:full          # Complete cross-browser testing
npm run test:e2e:mobile        # Mobile and tablet testing
npm run test:e2e:pwa           # PWA functionality
npm run test:e2e:performance   # Performance validation
npm run test:e2e:accessibility # WCAG compliance
npm run test:e2e:visual        # Visual regression testing
npm run test:e2e:agents        # Agent workflow testing
npm run test:e2e:realtime      # WebSocket and real-time features

# Development helpers
npm run test:e2e:debug         # Debug mode (headed + verbose)
npm run test:e2e:update        # Update visual snapshots
npm run test:e2e:report        # View last test report
npm run test:e2e:help          # Show all options
```

### Advanced Usage

```bash
# Custom configurations
npm run test:e2e -- --suite mobile --headed --retries 3
npm run test:e2e -- --debug --no-parallel
npm run test:e2e -- --suite performance --workers 2 --verbose

# Direct Playwright commands
npm run playwright test --project="Mobile Chrome"
npm run playwright test --grep "websocket"
npm run playwright test --update-snapshots
```

## 📊 Test Configuration

### Browser Matrix

| Project | Browser | Viewport | Focus |
|---------|---------|----------|-------|
| `chromium` | Chrome Desktop | 1280x720 | General functionality |
| `firefox` | Firefox Desktop | 1280x720 | Cross-browser compatibility |
| `webkit` | Safari Desktop | 1280x720 | Safari-specific issues |
| `Mobile Chrome` | Chrome Mobile | Pixel 5 | Mobile touch interactions |
| `Mobile Safari` | Safari Mobile | iPhone 12 | iOS-specific behavior |
| `iPad` | Safari Tablet | iPad Pro | Tablet responsiveness |
| `performance` | Chrome Optimized | 1280x720 | Performance benchmarks |
| `accessibility` | Chrome A11y | 1280x720 | Accessibility validation |

### Test Environments

- **Local Development**: `http://localhost:5173` (Vite dev server)
- **CI/CD**: Production build with backend integration
- **Performance Testing**: Optimized production build
- **Visual Regression**: Stable Chrome with animations disabled

## 🏗️ Architecture

### Test Structure

```
tests/
├── e2e/                          # End-to-end test specifications
│   ├── smoke/                    # Critical path smoke tests
│   ├── pwa/                      # PWA functionality tests
│   ├── mobile/                   # Mobile responsiveness tests
│   ├── visual/                   # Visual regression tests
│   ├── performance/              # Performance validation tests
│   ├── accessibility/            # WCAG compliance tests
│   ├── agent-workflows/          # Agent workflow tests
│   └── websocket-realtime.spec.ts # Real-time feature tests
├── fixtures/                     # Test data and setup
├── utils/                        # Shared test utilities
└── README.md                     # This documentation
```

### Test Utilities (`utils/test-helpers.ts`)

Comprehensive helper functions for common testing operations:

- **WebSocket Testing**: Connection monitoring and message validation
- **Performance Metrics**: Core Web Vitals measurement
- **PWA Validation**: Manifest and service worker checks
- **Accessibility Checks**: WCAG compliance validation
- **Mobile Testing**: Touch target validation and gesture simulation
- **Visual Testing**: Screenshot comparison and theme validation
- **Memory Monitoring**: JavaScript heap usage tracking

## 📈 Quality Gates

### Success Criteria

- **Smoke Tests**: 100% pass rate (blocking for deployment)
- **Performance**: Dashboard loads <2s, Core Web Vitals in "Good" range
- **Accessibility**: WCAG AA compliance, 90%+ elements properly labeled
- **Mobile**: 80%+ touch targets meet 44px minimum size
- **PWA**: Valid manifest, service worker functional, offline mode works
- **Visual**: No unexpected UI regressions, themes work correctly

### Performance Targets

| Metric | Target | Validation |
|--------|--------|------------|
| Dashboard Load Time | <2s | PRD requirement |
| First Contentful Paint | <1.8s | Core Web Vitals |
| Largest Contentful Paint | <2.5s | Core Web Vitals |
| Cumulative Layout Shift | <0.1 | Core Web Vitals |
| Bundle Size | <2MB | Resource optimization |
| Memory Usage | <50% heap | Runtime efficiency |

## 🔧 Configuration

### Playwright Config (`playwright.config.ts`)

Enhanced configuration with:
- Multiple browser projects and device emulation
- Environment-specific settings (CI vs local)
- Performance optimization flags
- Comprehensive reporting (HTML, JSON, JUnit)
- Visual regression baseline management
- Retry strategies and timeout optimization

### Environment Variables

```bash
# Test execution
TEST_BASE_URL=http://localhost:5173    # Frontend URL
TEST_BACKEND_URL=http://localhost:8000 # Backend URL for integration tests
TEST_WITH_BACKEND=true                 # Enable backend integration

# Debugging
PWDEBUG=1                             # Playwright debug mode
DEBUG=pw:api                          # Playwright API debugging
SLOW_MO=1000                          # Slow motion for observation

# CI/CD
CI=true                               # CI environment detection
GITHUB_ACTIONS=true                   # GitHub Actions specific
```

## 🚀 CI/CD Integration

### GitHub Actions Workflow

Automated testing across multiple scenarios:

- **PR Validation**: Smoke tests on every pull request
- **Main Branch**: Full test suite on merge to main
- **Nightly Builds**: Comprehensive testing including visual regression
- **Manual Triggers**: On-demand test execution with suite selection

### Test Artifacts

- **HTML Reports**: Interactive test results with screenshots and videos
- **JUnit XML**: Integration with CI/CD platforms and reporting tools
- **Screenshots**: Failure investigation and visual regression baselines
- **Performance Metrics**: Trending and regression detection

## 🐛 Debugging & Troubleshooting

### Common Issues

1. **Test Timeouts**
   ```bash
   # Increase timeout for slow environments
   npm run test:e2e -- --timeout 60000
   ```

2. **Flaky Tests**
   ```bash
   # Run with retries
   npm run test:e2e -- --retries 3
   ```

3. **Visual Regression Failures**
   ```bash
   # Update baselines after intentional changes
   npm run test:e2e:update
   ```

4. **WebSocket Connection Issues**
   ```bash
   # Test with backend integration
   TEST_WITH_BACKEND=true npm run test:e2e:realtime
   ```

### Debug Mode

```bash
# Full debug experience
npm run test:e2e:debug

# Selective debugging
npm run test:e2e -- --debug --grep "specific test"
```

### Test Data

All tests use deterministic mock data and avoid external dependencies to ensure reliability and consistency across environments.

## 📋 Best Practices

### Writing Tests

1. **Use Page Object Pattern**: Encapsulate UI interactions in reusable components
2. **Implement Proper Waits**: Use `waitForSelector` and `waitForLoadState` instead of arbitrary timeouts
3. **Add Meaningful Assertions**: Test behavior, not implementation details
4. **Handle Flakiness**: Use proper selectors and wait for stable states
5. **Document Test Intent**: Clear test names and comments explaining complex scenarios

### Performance

1. **Parallel Execution**: Tests run in parallel by default for speed
2. **Selective Testing**: Use `--grep` patterns for focused test runs
3. **Resource Cleanup**: Properly close browsers and clean up test data
4. **Screenshot Strategy**: Only capture on failures to reduce overhead

### Maintenance

1. **Regular Updates**: Keep Playwright and browsers updated
2. **Baseline Management**: Update visual regression baselines when UI changes
3. **Test Coverage**: Monitor test coverage and add tests for new features
4. **Performance Monitoring**: Track test execution times and optimize slow tests

---

**For questions or contributions to the testing suite, see the [Contributing Guide](../CONTRIBUTING.md) or contact the QA team.**