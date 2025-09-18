import { test, expect, Page } from '@playwright/test';
import { 
  AuthHelpers, 
  NavigationHelpers, 
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle
} from '../utils/test-helpers';

/**
 * Authentication Workflow E2E Tests
 * Tests complete user authentication journey for PWA
 */

test.describe('Authentication Workflows', () => {
  let authHelpers: AuthHelpers;
  let navHelpers: NavigationHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    authHelpers = new AuthHelpers(page);
    navHelpers = new NavigationHelpers(page);
    testData = loadTestData();
    
    // Start each test from the home page
    await page.goto('/');
  });

  test.describe('Login Flow', () => {
    test('should successfully log in with valid credentials', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Navigate to login page
      await page.goto('/login');
      await expect(page.locator('[data-testid="login-form"]')).toBeVisible();

      // Fill in credentials
      await page.fill('[data-testid="email-input"]', adminUser.email);
      await page.fill('[data-testid="password-input"]', 'test-password');

      // Submit login form
      await page.click('[data-testid="login-submit"]');

      // Wait for redirect to dashboard
      await page.waitForURL('**/dashboard', { timeout: 10000 });
      
      // Verify successful login
      await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
      await expect(page.locator('[data-testid="dashboard-title"]')).toBeVisible();
      
      // Verify user information displayed
      const userInfo = page.locator('[data-testid="user-info"]');
      await expect(userInfo).toContainText(adminUser.email);

      await takeTimestampedScreenshot(page, 'successful-login');
    });

    test('should reject invalid credentials', async ({ page }) => {
      await page.goto('/login');

      // Try with invalid credentials
      await page.fill('[data-testid="email-input"]', 'invalid@example.com');
      await page.fill('[data-testid="password-input"]', 'wrongpassword');
      await page.click('[data-testid="login-submit"]');

      // Should stay on login page with error message
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials');
      
      // Should not redirect to dashboard
      expect(page.url()).toContain('/login');

      await takeTimestampedScreenshot(page, 'invalid-login');
    });

    test('should validate required fields', async ({ page }) => {
      await page.goto('/login');

      // Try to submit without filling fields
      await page.click('[data-testid="login-submit"]');

      // Should show validation errors
      await expect(page.locator('[data-testid="email-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="password-error"]')).toBeVisible();

      // Test individual field validation
      await page.fill('[data-testid="email-input"]', 'invalid-email');
      await page.blur('[data-testid="email-input"]');
      await expect(page.locator('[data-testid="email-error"]')).toContainText('valid email');

      await takeTimestampedScreenshot(page, 'validation-errors');
    });

    test('should handle forgot password flow', async ({ page }) => {
      await page.goto('/login');

      // Click forgot password link
      await page.click('[data-testid="forgot-password-link"]');
      await page.waitForURL('**/forgot-password');

      // Fill email for password reset
      await page.fill('[data-testid="reset-email-input"]', 'user@leanvibe.test');
      await page.click('[data-testid="reset-submit"]');

      // Should show success message
      await expect(page.locator('[data-testid="reset-success"]')).toBeVisible();
      await expect(page.locator('[data-testid="reset-success"]')).toContainText('reset instructions');

      await takeTimestampedScreenshot(page, 'forgot-password');
    });
  });

  test.describe('Session Management', () => {
    test('should maintain session across page refreshes', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in
      await authHelpers.login(adminUser);
      
      // Refresh page
      await page.reload();
      await waitForNetworkIdle(page);

      // Should still be logged in
      await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
      expect(page.url()).toContain('/dashboard');
    });

    test('should handle session timeout', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in
      await authHelpers.login(adminUser);

      // Simulate session timeout by clearing localStorage
      await page.evaluate(() => {
        localStorage.removeItem('authToken');
        sessionStorage.clear();
      });

      // Navigate to a protected route
      await page.goto('/agents');

      // Should redirect to login
      await page.waitForURL('**/login', { timeout: 5000 });
      await expect(page.locator('[data-testid="session-expired-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'session-timeout');
    });

    test('should handle multiple tabs/windows', async ({ context, page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in on first tab
      await authHelpers.login(adminUser);

      // Open new tab
      const newPage = await context.newPage();
      await newPage.goto('/dashboard');

      // Should be logged in on new tab
      await expect(newPage.locator('[data-testid="user-menu"]')).toBeVisible();

      // Log out from first tab
      await authHelpers.logout();

      // New tab should also be logged out (if using shared session)
      await newPage.reload();
      await newPage.waitForURL('**/login', { timeout: 5000 });

      await newPage.close();
    });
  });

  test.describe('Logout Flow', () => {
    test('should successfully log out', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in first
      await authHelpers.login(adminUser);

      // Perform logout
      await authHelpers.logout();

      // Should be redirected to login page
      expect(page.url()).toContain('/login');
      
      // User menu should not be visible
      await expect(page.locator('[data-testid="user-menu"]')).not.toBeVisible();

      // Trying to access protected route should redirect to login
      await page.goto('/dashboard');
      await page.waitForURL('**/login', { timeout: 5000 });

      await takeTimestampedScreenshot(page, 'successful-logout');
    });

    test('should clear user data on logout', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in
      await authHelpers.login(adminUser);

      // Check that user data is present
      const userDataBefore = await page.evaluate(() => ({
        localStorage: { ...localStorage },
        sessionStorage: { ...sessionStorage }
      }));

      expect(Object.keys(userDataBefore.localStorage)).toContain('authToken');

      // Log out
      await authHelpers.logout();

      // Check that user data is cleared
      const userDataAfter = await page.evaluate(() => ({
        localStorage: { ...localStorage },
        sessionStorage: { ...sessionStorage }
      }));

      expect(Object.keys(userDataAfter.localStorage)).not.toContain('authToken');
      expect(Object.keys(userDataAfter.sessionStorage)).toHaveLength(0);
    });
  });

  test.describe('Role-Based Access', () => {
    test('should enforce admin-only routes', async ({ page }) => {
      const regularUser = testData.users.find((u: any) => u.role === 'user');
      
      // Log in as regular user
      await authHelpers.login(regularUser);

      // Try to access admin-only route
      await page.goto('/admin/settings');

      // Should be redirected or show access denied
      await expect(
        page.locator('[data-testid="access-denied"]')
      ).toBeVisible();

      await takeTimestampedScreenshot(page, 'access-denied');
    });

    test('should allow admin access to all routes', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Log in as admin
      await authHelpers.login(adminUser);

      // Test access to admin routes
      const adminRoutes = ['/admin/settings', '/admin/users', '/admin/system'];
      
      for (const route of adminRoutes) {
        await page.goto(route);
        
        // Should not show access denied
        await expect(
          page.locator('[data-testid="access-denied"]')
        ).not.toBeVisible();
        
        // Should show admin content
        await expect(
          page.locator('[data-testid="admin-content"]')
        ).toBeVisible();
      }
    });

    test('should display role-appropriate navigation', async ({ page }) => {
      // Test regular user navigation
      const regularUser = testData.users.find((u: any) => u.role === 'user');
      await authHelpers.login(regularUser);

      // Should not see admin navigation items
      await expect(page.locator('[data-testid="nav-admin"]')).not.toBeVisible();
      
      // Should see regular user navigation
      await expect(page.locator('[data-testid="nav-dashboard"]')).toBeVisible();
      await expect(page.locator('[data-testid="nav-agents"]')).toBeVisible();

      await authHelpers.logout();

      // Test admin navigation
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);

      // Should see admin navigation items
      await expect(page.locator('[data-testid="nav-admin"]')).toBeVisible();
      await expect(page.locator('[data-testid="nav-dashboard"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'role-based-navigation');
    });
  });

  test.describe('Security Features', () => {
    test('should implement CSRF protection', async ({ page }) => {
      // This test would verify CSRF token handling
      // Implementation depends on your CSRF strategy
      await page.goto('/login');
      
      // Check for CSRF token in form
      const csrfToken = await page.getAttribute('[name="csrf-token"]', 'content');
      expect(csrfToken).toBeTruthy();
    });

    test('should rate limit login attempts', async ({ page }) => {
      await page.goto('/login');

      // Attempt multiple failed logins
      for (let i = 0; i < 5; i++) {
        await page.fill('[data-testid="email-input"]', 'test@example.com');
        await page.fill('[data-testid="password-input"]', 'wrongpassword');
        await page.click('[data-testid="login-submit"]');
        await page.waitForTimeout(500);
      }

      // Should show rate limit message
      await expect(
        page.locator('[data-testid="rate-limit-message"]')
      ).toBeVisible();

      await takeTimestampedScreenshot(page, 'rate-limiting');
    });

    test('should validate password strength on registration', async ({ page }) => {
      await page.goto('/register');

      // Test weak password
      await page.fill('[data-testid="password-input"]', '123');
      await page.blur('[data-testid="password-input"]');
      
      await expect(
        page.locator('[data-testid="password-strength-weak"]')
      ).toBeVisible();

      // Test strong password
      await page.fill('[data-testid="password-input"]', 'StrongPassword123!');
      await page.blur('[data-testid="password-input"]');
      
      await expect(
        page.locator('[data-testid="password-strength-strong"]')
      ).toBeVisible();

      await takeTimestampedScreenshot(page, 'password-validation');
    });
  });
});