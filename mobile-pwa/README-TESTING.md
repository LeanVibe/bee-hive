# LeanVibe Agent Hive 2.0 - Comprehensive Testing Guide

This guide covers the comprehensive Playwright end-to-end testing suite for the LeanVibe Agent Hive 2.0 dashboard.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn
- Running development server on `http://localhost:3001`

### Installation
```bash
# Install dependencies (if not already done)
npm install

# Install Playwright browsers
npm run test:e2e:install
```

### Running Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Run tests with UI mode (interactive)
npm run test:e2e:ui

# Debug specific test
npm run test:e2e:debug tests/e2e/dashboard-navigation.spec.ts

# Run specific test file
npm run test:e2e tests/e2e/agent-management.spec.ts

# Generate and view test report
npm run test:e2e:report
```

## 📋 Test Coverage

### Core Test Suites

#### 1. Dashboard Navigation Tests (`dashboard-navigation.spec.ts`)
- ✅ Header and navigation functionality
- ✅ Tab switching and state management
- ✅ Overview dashboard layout
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Refresh and sync functionality
- ✅ Error handling and accessibility
- ✅ Performance benchmarks

#### 2. Agent Management Tests (`agent-management.spec.ts`)
- ✅ Agent display and status indicators
- ✅ Agent team activation/deactivation
- ✅ Individual agent controls
- ✅ Agent configuration modals
- ✅ Bulk agent operations
- ✅ Real-time agent updates
- ✅ Performance monitoring
- ✅ Error handling for agent operations

#### 3. Task Management Tests (`task-management.spec.ts`)
- ✅ Kanban board display and layout
- ✅ Task filtering and search
- ✅ Drag and drop functionality
- ✅ Task creation and editing
- ✅ Task priority and assignment
- ✅ Offline mode support
- ✅ Real-time task updates
- ✅ Performance with large datasets

#### 4. Real-time Updates Tests (`real-time-updates.spec.ts`)
- ✅ System health real-time updates
- ✅ Task updates via WebSocket
- ✅ Agent status real-time changes
- ✅ Event timeline updates
- ✅ Polling functionality fallback
- ✅ Performance and resource management

#### 5. Responsive Design Tests (`responsive-design.spec.ts`)
- ✅ Desktop breakpoints (1920px, 1366px, ultrawide)
- ✅ Tablet breakpoints (iPad Pro, standard iPad, landscape)
- ✅ Mobile breakpoints (iPhone 14 Pro, iPhone SE, Android)
- ✅ Touch interactions and gestures
- ✅ PWA mobile features
- ✅ Accessibility on mobile
- ✅ Performance on mobile devices

#### 6. Error Handling Tests (`error-handling.spec.ts`)
- ✅ Network error handling
- ✅ API error responses (500, 404, 401, 403)
- ✅ Partial service failures
- ✅ User action error handling
- ✅ Data validation and integrity
- ✅ Error recovery and retry mechanisms
- ✅ User experience during errors

#### 7. Visual Regression Tests (`visual-regression.spec.ts`)
- ✅ Dashboard view screenshots
- ✅ Component screenshots
- ✅ Responsive visual testing
- ✅ State-based visual testing
- ✅ Modal and overlay screenshots
- ✅ Theme variations
- ✅ Data variation screenshots

## 🧪 Test Architecture

### Page Object Model
Tests use the Page Object Model pattern for maintainability:

```typescript
// Example usage
const dashboardPage = new DashboardPage(page)
await dashboardPage.navigateToTasks()
await dashboardPage.verifyTaskCount(5)
```

Available Page Objects:
- `DashboardPage` - Main dashboard functionality
- `KanbanBoardPage` - Task management and Kanban
- `AgentHealthPanelPage` - Agent management
- `EventTimelinePage` - Event monitoring
- `TaskEditModalPage` - Task editing
- `AgentConfigModalPage` - Agent configuration

### Test Utilities

#### TestHelpers
Common utilities for test operations:
- `waitForElement()` - Wait for elements with custom timeout
- `simulateSlowNetwork()` - Network condition simulation
- `verifyLoadingState()` - Loading state validation
- `testResponsiveBreakpoint()` - Responsive design testing
- `testTouchInteraction()` - Touch gesture simulation

#### APIMocks
Comprehensive API mocking for consistent testing:
- `setupStandardMocks()` - Standard mock data
- `mockErrorResponses()` - Error condition simulation
- `mockSlowNetwork()` - Performance testing
- `mockPartialFailures()` - Degraded service testing

### Test Data
Realistic mock data in `test-data.ts`:
- Tasks with various statuses and priorities
- Agents with different states and metrics
- Events with different severity levels
- System health and performance data

## 📊 Test Execution

### Browser Support
Tests run on multiple browsers:
- ✅ Chromium (Desktop)
- ✅ Firefox (Desktop)  
- ✅ WebKit/Safari (Desktop)
- ✅ Mobile Chrome (Pixel 5)
- ✅ Mobile Safari (iPhone 12)
- ✅ iPad Pro
- ✅ Microsoft Edge

### Viewport Testing
Responsive testing across:
- 📱 Mobile: 375x667 (iPhone SE), 393x852 (iPhone 14 Pro)
- 📱 Android: 360x640
- 📱 Tablet: 768x1024 (iPad), 1024x1366 (iPad Pro)
- 💻 Desktop: 1366x768, 1920x1080, 2560x1440

### Test Reports
Comprehensive reporting includes:
- HTML report with screenshots
- JSON results for CI/CD integration
- JUnit XML for test management systems
- Visual regression comparisons

## 🔧 Configuration

### Playwright Configuration (`playwright.config.ts`)
- Parallel execution for faster testing
- Retry logic for flaky test handling
- Screenshot and video capture on failure
- Global setup and teardown
- Development server auto-start

### Environment Variables
```bash
# Optional configuration
CI=true                    # Enable CI mode
HEADED=true               # Run in headed mode
DEBUG=true                # Enable debug mode
SLOW_MO=1000             # Slow down actions (ms)
```

## 🚨 Quality Gates

### Pre-commit Testing
```bash
# Run core tests before committing
npm run test:e2e tests/e2e/dashboard-navigation.spec.ts
npm run test:e2e tests/e2e/task-management.spec.ts
```

### CI/CD Integration
Tests are configured for:
- GitHub Actions
- Jenkins
- Azure DevOps
- Other CI systems supporting Playwright

### Performance Benchmarks
- Dashboard load time: < 5 seconds
- Kanban drag operations: < 3 seconds
- Large dataset handling: < 10 seconds
- Mobile responsiveness: All breakpoints tested

## 🐛 Debugging Tests

### Debug Mode
```bash
# Run single test in debug mode
npm run test:e2e:debug tests/e2e/dashboard-navigation.spec.ts

# Run with browser visible
npm run test:e2e:headed tests/e2e/task-management.spec.ts
```

### Screenshots and Videos
Failed tests automatically capture:
- Screenshots at failure point
- Video recordings of test execution
- Network activity logs
- Console error messages

### Common Issues
1. **Timing Issues**: Use `waitForCondition()` helper
2. **Element Not Found**: Verify selectors with page object methods
3. **Network Errors**: Check API mock setup
4. **Flaky Tests**: Add appropriate waits and conditions

## 📈 Metrics and Monitoring

### Test Metrics
- Test execution time: ~15 minutes for full suite
- Success rate: Target 99%+ 
- Coverage: 90%+ of user interactions
- Visual regression: Pixel-perfect UI validation

### Performance Testing
- Load time benchmarks
- Memory usage monitoring
- Network efficiency validation
- Mobile performance testing

## 🎯 Best Practices

### Writing New Tests
1. Use Page Object Model pattern
2. Include data-testid attributes for stable selectors
3. Mock external dependencies consistently
4. Test happy path and error conditions
5. Verify accessibility requirements

### Test Maintenance
1. Update page objects when UI changes
2. Refresh mock data periodically
3. Review and update visual baselines
4. Monitor for flaky tests and fix promptly

### Performance Optimization
1. Run tests in parallel where possible
2. Use focused test execution for development
3. Implement smart test selection based on changes
4. Optimize test data and mock responses

## 🔗 Integration with Development

### Local Development
```bash
# Start dev server and run tests
npm run dev &
npm run test:e2e

# Or use the built-in web server (automatic)
npm run test:e2e
```

### Continuous Integration
Tests integrate with:
- Pull request validation
- Nightly regression testing
- Release candidate validation
- Production deployment gates

## 📚 Additional Resources

- [Playwright Documentation](https://playwright.dev/)
- [Page Object Model Guide](https://playwright.dev/docs/pom)
- [Visual Testing Best Practices](https://playwright.dev/docs/test-snapshots)
- [Mobile Testing Guide](https://playwright.dev/docs/emulation)

---

**The Ultimate Testing Goal**: Ensure the LeanVibe Agent Hive 2.0 dashboard provides a flawless, professional experience that validates our platform as enterprise-ready autonomous development infrastructure. 🚀