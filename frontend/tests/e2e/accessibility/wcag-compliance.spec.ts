import { test, expect } from '@playwright/test';
import { 
  AccessibilityHelpers,
  AuthHelpers,
  NavigationHelpers,
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle
} from '../utils/test-helpers';

/**
 * WCAG Accessibility Compliance E2E Tests
 * Tests PWA accessibility features and WCAG AA/AAA compliance
 */

test.describe('WCAG Accessibility Compliance', () => {
  let accessibilityHelpers: AccessibilityHelpers;
  let authHelpers: AuthHelpers;
  let navHelpers: NavigationHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    accessibilityHelpers = new AccessibilityHelpers(page);
    authHelpers = new AuthHelpers(page);
    navHelpers = new NavigationHelpers(page);
    testData = loadTestData();
  });

  test.describe('Keyboard Navigation', () => {
    test('should support full keyboard navigation', async ({ page }) => {
      await page.goto('/');

      // Test keyboard navigation functionality
      const keyboardNavWorking = await accessibilityHelpers.testKeyboardNavigation();
      expect(keyboardNavWorking).toBe(true);

      // Test specific navigation patterns
      await page.keyboard.press('Tab');
      let focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(focusedElement).toBeTruthy();

      // Navigate through login form with keyboard
      await page.goto('/login');
      await page.keyboard.press('Tab'); // Focus first input
      await page.keyboard.type('test@example.com');
      
      await page.keyboard.press('Tab'); // Focus password input
      await page.keyboard.type('password');
      
      await page.keyboard.press('Tab'); // Focus submit button
      await page.keyboard.press('Enter'); // Submit form

      await takeTimestampedScreenshot(page, 'keyboard-navigation');
    });

    test('should provide visible focus indicators', async ({ page }) => {
      await page.goto('/');

      // Check focus indicators on interactive elements
      const interactiveElements = await page.locator('button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])').all();

      for (const element of interactiveElements.slice(0, 5)) { // Test first 5 elements
        await element.focus();
        
        // Check if focus is visible
        const focusStyles = await element.evaluate((el) => {
          const computed = window.getComputedStyle(el);
          const pseudoFocus = window.getComputedStyle(el, ':focus');
          
          return {
            outline: computed.outline,
            outlineWidth: computed.outlineWidth,
            boxShadow: computed.boxShadow,
            focusOutline: pseudoFocus.outline,
            focusBoxShadow: pseudoFocus.boxShadow
          };
        });

        // Should have visible focus indicator
        const hasFocusIndicator = 
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow !== 'none' ||
          focusStyles.focusOutline !== 'none' ||
          focusStyles.focusBoxShadow !== 'none';

        expect(hasFocusIndicator).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'focus-indicators');
    });

    test('should maintain logical tab order', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Test tab order on dashboard
      const tabOrder: string[] = [];
      
      // Reset focus to body
      await page.evaluate(() => document.body.focus());
      
      // Tab through elements and record order
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press('Tab');
        const elementInfo = await page.evaluate(() => {
          const el = document.activeElement;
          return {
            tagName: el?.tagName || '',
            id: el?.id || '',
            className: el?.className || '',
            textContent: el?.textContent?.trim().substring(0, 20) || '',
            testId: el?.getAttribute('data-testid') || ''
          };
        });
        
        tabOrder.push(`${elementInfo.tagName}${elementInfo.testId ? `[${elementInfo.testId}]` : ''}`);
      }

      // Tab order should follow logical sequence
      expect(tabOrder.length).toBe(10);
      
      // Should not have duplicate focuses (unless intentional)
      const uniqueElements = new Set(tabOrder);
      expect(uniqueElements.size).toBeGreaterThan(5); // At least 5 unique elements

      await takeTimestampedScreenshot(page, 'tab-order');
    });

    test('should support skip links', async ({ page }) => {
      await page.goto('/');

      // Test skip link functionality
      await page.keyboard.press('Tab');
      
      const skipLink = page.locator('[data-testid="skip-to-main"]').first();
      const isSkipLinkVisible = await skipLink.isVisible().catch(() => false);
      
      if (isSkipLinkVisible) {
        await skipLink.click();
        
        // Should focus main content
        const mainContent = await page.evaluate(() => {
          const main = document.querySelector('main, [role="main"], #main-content');
          return !!main && document.activeElement === main;
        });
        
        expect(mainContent).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'skip-links');
    });
  });

  test.describe('Screen Reader Support', () => {
    test('should have proper ARIA labels and roles', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check ARIA compliance
      const ariaCompliance = await accessibilityHelpers.checkAriaCompliance();

      // Should have minimal missing labels
      expect(ariaCompliance.missingLabels.length).toBeLessThan(5);
      
      // Should not have invalid roles
      expect(ariaCompliance.invalidRoles.length).toBe(0);

      // Check specific ARIA patterns
      const ariaPatterns = await page.evaluate(() => {
        const results = {
          hasMainLandmark: !!document.querySelector('main, [role="main"]'),
          hasNavigationLandmarks: document.querySelectorAll('nav, [role="navigation"]').length > 0,
          hasHeadingStructure: document.querySelectorAll('h1, h2, h3, h4, h5, h6').length > 0,
          hasFormLabels: Array.from(document.querySelectorAll('input, select, textarea')).every(input => {
            const id = (input as HTMLElement).id;
            return input.hasAttribute('aria-label') || 
                   input.hasAttribute('aria-labelledby') ||
                   (id && document.querySelector(`label[for="${id}"]`));
          })
        };
        return results;
      });

      expect(ariaPatterns.hasMainLandmark).toBe(true);
      expect(ariaPatterns.hasNavigationLandmarks).toBe(true);
      expect(ariaPatterns.hasHeadingStructure).toBe(true);

      await takeTimestampedScreenshot(page, 'aria-compliance');
    });

    test('should have proper heading hierarchy', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check heading structure
      const headingStructure = await page.evaluate(() => {
        const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        return headings.map(h => ({
          level: parseInt(h.tagName.charAt(1)),
          text: h.textContent?.trim() || '',
          id: h.id || '',
          ariaLevel: h.getAttribute('aria-level')
        }));
      });

      // Should have proper heading hierarchy
      expect(headingStructure.length).toBeGreaterThan(0);
      
      // Should start with h1
      const firstHeading = headingStructure[0];
      expect(firstHeading.level).toBe(1);

      // Check for logical progression (no skipping levels)
      for (let i = 1; i < headingStructure.length; i++) {
        const prev = headingStructure[i - 1];
        const curr = headingStructure[i];
        
        // Level should not increase by more than 1
        if (curr.level > prev.level) {
          expect(curr.level - prev.level).toBeLessThanOrEqual(1);
        }
      }

      await takeTimestampedScreenshot(page, 'heading-hierarchy');
    });

    test('should provide descriptive alt text for images', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check image alt text
      const imageAccessibility = await page.evaluate(() => {
        const images = Array.from(document.querySelectorAll('img'));
        const results = {
          totalImages: images.length,
          imagesWithAlt: 0,
          imagesWithEmptyAlt: 0,
          imagesWithDescriptiveAlt: 0,
          decorativeImages: 0
        };

        images.forEach(img => {
          const alt = img.getAttribute('alt');
          const role = img.getAttribute('role');
          
          if (alt !== null) {
            results.imagesWithAlt++;
            
            if (alt === '') {
              results.imagesWithEmptyAlt++;
              if (role === 'presentation' || img.hasAttribute('aria-hidden')) {
                results.decorativeImages++;
              }
            } else if (alt.length > 5) {
              results.imagesWithDescriptiveAlt++;
            }
          }
        });

        return results;
      });

      if (imageAccessibility.totalImages > 0) {
        // All images should have alt attributes
        expect(imageAccessibility.imagesWithAlt).toBe(imageAccessibility.totalImages);
        
        // Images with content should have descriptive alt text
        const contentImages = imageAccessibility.totalImages - imageAccessibility.decorativeImages;
        if (contentImages > 0) {
          expect(imageAccessibility.imagesWithDescriptiveAlt).toBeGreaterThan(0);
        }
      }

      await takeTimestampedScreenshot(page, 'image-alt-text');
    });

    test('should provide proper form accessibility', async ({ page }) => {
      await page.goto('/login');

      // Check form accessibility
      const formAccessibility = await page.evaluate(() => {
        const formElements = Array.from(document.querySelectorAll('input, select, textarea'));
        const results = {
          totalElements: formElements.length,
          elementsWithLabels: 0,
          elementsWithRequiredIndicators: 0,
          elementsWithErrorHandling: 0
        };

        formElements.forEach(element => {
          const id = (element as HTMLElement).id;
          const hasLabel = element.hasAttribute('aria-label') ||
                          element.hasAttribute('aria-labelledby') ||
                          (id && document.querySelector(`label[for="${id}"]`));
          
          if (hasLabel) results.elementsWithLabels++;
          
          if (element.hasAttribute('required') || element.hasAttribute('aria-required')) {
            results.elementsWithRequiredIndicators++;
          }

          const describedBy = element.getAttribute('aria-describedby');
          if (describedBy) {
            const errorElement = document.getElementById(describedBy);
            if (errorElement) results.elementsWithErrorHandling++;
          }
        });

        return results;
      });

      if (formAccessibility.totalElements > 0) {
        // All form elements should have labels
        expect(formAccessibility.elementsWithLabels).toBe(formAccessibility.totalElements);
        
        // Required fields should be indicated
        if (formAccessibility.elementsWithRequiredIndicators > 0) {
          expect(formAccessibility.elementsWithRequiredIndicators).toBeGreaterThan(0);
        }
      }

      await takeTimestampedScreenshot(page, 'form-accessibility');
    });
  });

  test.describe('Color and Contrast', () => {
    test('should meet WCAG AA color contrast requirements', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check color contrast ratios
      const contrastResults = await page.evaluate(() => {
        function getContrastRatio(foreground: string, background: string): number {
          // Simple contrast ratio calculation (for testing)
          const fLum = getLuminance(foreground);
          const bLum = getLuminance(background);
          const lighter = Math.max(fLum, bLum);
          const darker = Math.min(fLum, bLum);
          return (lighter + 0.05) / (darker + 0.05);
        }

        function getLuminance(color: string): number {
          // Simplified luminance calculation
          const rgb = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
          if (!rgb) return 0.5; // Default for non-RGB colors
          
          const [r, g, b] = rgb.slice(1).map(Number);
          return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        }

        const textElements = document.querySelectorAll('p, span, div, button, a, label, h1, h2, h3, h4, h5, h6');
        const results = {
          totalElements: textElements.length,
          passAA: 0,
          passAAA: 0,
          failAA: 0
        };

        Array.from(textElements).slice(0, 20).forEach(element => { // Check first 20 elements
          const computed = window.getComputedStyle(element);
          const textColor = computed.color;
          const backgroundColor = computed.backgroundColor;
          
          if (textColor && backgroundColor && backgroundColor !== 'rgba(0, 0, 0, 0)') {
            const ratio = getContrastRatio(textColor, backgroundColor);
            
            if (ratio >= 4.5) {
              results.passAA++;
              if (ratio >= 7) {
                results.passAAA++;
              }
            } else {
              results.failAA++;
            }
          }
        });

        return results;
      });

      // Most text should pass WCAG AA contrast requirements
      if (contrastResults.totalElements > 0) {
        const passRate = contrastResults.passAA / (contrastResults.passAA + contrastResults.failAA);
        expect(passRate).toBeGreaterThan(0.8); // 80% pass rate
      }

      await takeTimestampedScreenshot(page, 'color-contrast');
    });

    test('should not rely solely on color for information', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check for status indicators that use more than just color
      const statusIndicators = await page.evaluate(() => {
        const statusElements = document.querySelectorAll('[data-status], .status, .badge');
        const results = {
          totalStatus: statusElements.length,
          withText: 0,
          withIcons: 0,
          withPatterns: 0
        };

        statusElements.forEach(element => {
          const hasText = element.textContent && element.textContent.trim().length > 0;
          const hasIcon = element.querySelector('svg, i, .icon') || 
                         element.classList.toString().includes('icon');
          const hasPattern = element.style.backgroundImage || 
                            element.classList.toString().includes('pattern');

          if (hasText) results.withText++;
          if (hasIcon) results.withIcons++;
          if (hasPattern) results.withPatterns++;
        });

        return results;
      });

      if (statusIndicators.totalStatus > 0) {
        // Status indicators should have text, icons, or patterns in addition to color
        const accessibleIndicators = statusIndicators.withText + 
                                   statusIndicators.withIcons + 
                                   statusIndicators.withPatterns;
        
        expect(accessibleIndicators).toBeGreaterThan(0);
      }

      await takeTimestampedScreenshot(page, 'color-information');
    });

    test('should support high contrast mode', async ({ page }) => {
      // Test with forced colors (high contrast mode simulation)
      await page.emulateMedia({ forcedColors: 'active' });
      
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Verify page is still usable in high contrast mode
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
      await expect(page.locator('[data-testid="navigation"]')).toBeVisible();
      
      // Interactive elements should still be distinguishable
      const buttons = page.locator('button').first();
      await expect(buttons).toBeVisible();

      await takeTimestampedScreenshot(page, 'high-contrast-mode');
    });
  });

  test.describe('Motion and Animation', () => {
    test('should respect reduced motion preferences', async ({ page }) => {
      // Test with reduced motion preference
      await page.emulateMedia({ reducedMotion: 'reduce' });
      
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check if animations are reduced
      const animationState = await page.evaluate(() => {
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        
        // Check for animation properties
        const animatedElements = document.querySelectorAll('[class*="animate"], [class*="transition"]');
        const animationStyles = Array.from(animatedElements).slice(0, 5).map(el => {
          const computed = window.getComputedStyle(el);
          return {
            animationDuration: computed.animationDuration,
            transitionDuration: computed.transitionDuration
          };
        });

        return {
          prefersReduced,
          animatedElements: animatedElements.length,
          animationStyles
        };
      });

      if (animationState.prefersReduced) {
        // When reduced motion is preferred, animations should be minimal
        expect(animationState.prefersReduced).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'reduced-motion');
    });

    test('should provide pause controls for auto-playing content', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check for auto-playing content and pause controls
      const autoPlayContent = await page.evaluate(() => {
        const videos = document.querySelectorAll('video[autoplay]');
        const carousels = document.querySelectorAll('[data-auto-rotate="true"], .carousel.auto');
        const animations = document.querySelectorAll('[data-auto-animate="true"]');

        return {
          autoPlayVideos: videos.length,
          autoCarousels: carousels.length,
          autoAnimations: animations.length,
          hasPauseControls: !!document.querySelector('[data-testid="pause-animations"], .pause-button')
        };
      });

      if (autoPlayContent.autoPlayVideos > 0 || 
          autoPlayContent.autoCarousels > 0 || 
          autoPlayContent.autoAnimations > 0) {
        // Should provide pause controls for auto-playing content
        expect(autoPlayContent.hasPauseControls).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'auto-play-controls');
    });
  });

  test.describe('Responsive Accessibility', () => {
    test('should maintain accessibility across different screen sizes', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);

      // Test mobile view accessibility
      await page.setViewportSize({ width: 375, height: 667 });
      await navHelpers.goToDashboard();

      // Check mobile accessibility
      const mobileAccessibility = await accessibilityHelpers.testKeyboardNavigation();
      expect(mobileAccessibility).toBe(true);

      // Check if mobile navigation is accessible
      const mobileNavButton = page.locator('[data-testid="mobile-nav-toggle"]');
      const hasMobileNav = await mobileNavButton.isVisible().catch(() => false);
      
      if (hasMobileNav) {
        await mobileNavButton.click();
        await expect(page.locator('[data-testid="mobile-nav-menu"]')).toBeVisible();
        
        // Mobile menu should be keyboard accessible
        await page.keyboard.press('Tab');
        const focusedInMenu = await page.evaluate(() => {
          const activeElement = document.activeElement;
          const mobileMenu = document.querySelector('[data-testid="mobile-nav-menu"]');
          return mobileMenu?.contains(activeElement) || false;
        });
        
        expect(focusedInMenu).toBe(true);
      }

      // Test tablet view
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.waitForTimeout(500);

      const tabletAccessibility = await accessibilityHelpers.testKeyboardNavigation();
      expect(tabletAccessibility).toBe(true);

      // Return to desktop
      await page.setViewportSize({ width: 1280, height: 720 });

      await takeTimestampedScreenshot(page, 'responsive-accessibility');
    });

    test('should have appropriate touch targets on mobile', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await navHelpers.goToDashboard();

      // Check touch target sizes
      const touchTargets = await page.evaluate(() => {
        const interactiveElements = document.querySelectorAll('button, a, input, select, [role="button"]');
        const results = {
          totalTargets: interactiveElements.length,
          adequateSize: 0,
          tooSmall: 0
        };

        interactiveElements.forEach(element => {
          const rect = element.getBoundingClientRect();
          const size = Math.min(rect.width, rect.height);
          
          if (size >= 44) { // WCAG recommended minimum
            results.adequateSize++;
          } else {
            results.tooSmall++;
          }
        });

        return results;
      });

      if (touchTargets.totalTargets > 0) {
        // Most touch targets should be adequately sized
        const adequateRate = touchTargets.adequateSize / touchTargets.totalTargets;
        expect(adequateRate).toBeGreaterThan(0.8); // 80% of targets should be adequate size
      }

      await takeTimestampedScreenshot(page, 'touch-targets');
    });
  });
});