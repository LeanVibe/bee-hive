import { FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';

/**
 * Global teardown for PWA E2E tests
 * Cleans up test environment and generates reports
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting PWA E2E Testing Global Teardown...');

  const testResultsDir = path.join(process.cwd(), 'test-results');

  // Generate test summary report
  console.log('📊 Generating test summary report...');
  
  try {
    // Read test results if they exist
    const resultsFiles = fs.readdirSync(testResultsDir)
      .filter(file => file.endsWith('-results.json'));

    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;
    let skippedTests = 0;
    let testDuration = 0;

    const summaryReport = {
      timestamp: new Date().toISOString(),
      environment: 'e2e-testing',
      testSuites: {
        workflows: { status: 'unknown', tests: 0, passed: 0, failed: 0 },
        pwa: { status: 'unknown', tests: 0, passed: 0, failed: 0 },
        performance: { status: 'unknown', tests: 0, passed: 0, failed: 0 },
        accessibility: { status: 'unknown', tests: 0, passed: 0, failed: 0 }
      },
      browsers: {
        chromium: { status: 'unknown', tests: 0 },
        firefox: { status: 'unknown', tests: 0 },
        webkit: { status: 'unknown', tests: 0 },
        mobile: { status: 'unknown', tests: 0 }
      },
      metrics: {
        totalTests,
        passedTests,
        failedTests,
        skippedTests,
        testDuration,
        passRate: totalTests > 0 ? (passedTests / totalTests) * 100 : 0
      },
      qualityGates: {
        passRateThreshold: 90,
        passRateMet: false,
        performanceThreshold: 90,
        accessibilityThreshold: 95
      }
    };

    // Calculate metrics from results
    if (resultsFiles.length > 0) {
      const resultsPath = path.join(testResultsDir, resultsFiles[0]);
      if (fs.existsSync(resultsPath)) {
        const results = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
        
        if (results.stats) {
          totalTests = results.stats.total || 0;
          passedTests = results.stats.passed || 0;
          failedTests = results.stats.failed || 0;
          skippedTests = results.stats.skipped || 0;
          testDuration = results.stats.duration || 0;
          
          summaryReport.metrics = {
            totalTests,
            passedTests,
            failedTests,
            skippedTests,
            testDuration,
            passRate: totalTests > 0 ? (passedTests / totalTests) * 100 : 0
          };

          summaryReport.qualityGates.passRateMet = 
            summaryReport.metrics.passRate >= summaryReport.qualityGates.passRateThreshold;
        }
      }
    }

    // Save summary report
    fs.writeFileSync(
      path.join(testResultsDir, 'e2e-summary-report.json'),
      JSON.stringify(summaryReport, null, 2)
    );

    // Generate quality gate status
    console.log('🎯 Evaluating Quality Gates...');
    
    const qualityGateStatus = {
      overall: summaryReport.qualityGates.passRateMet ? 'PASSED' : 'FAILED',
      details: {
        passRate: {
          status: summaryReport.qualityGates.passRateMet ? 'PASSED' : 'FAILED',
          actual: summaryReport.metrics.passRate,
          threshold: summaryReport.qualityGates.passRateThreshold
        }
      },
      recommendations: []
    };

    if (!summaryReport.qualityGates.passRateMet) {
      qualityGateStatus.recommendations.push(
        'Test pass rate below threshold. Review failed tests and fix issues.'
      );
    }

    if (summaryReport.metrics.failedTests > 0) {
      qualityGateStatus.recommendations.push(
        `${summaryReport.metrics.failedTests} tests failed. Check test reports for details.`
      );
    }

    fs.writeFileSync(
      path.join(testResultsDir, 'quality-gate-status.json'),
      JSON.stringify(qualityGateStatus, null, 2)
    );

    // Display summary
    console.log('📈 Test Execution Summary:');
    console.log(`   Total Tests: ${totalTests}`);
    console.log(`   Passed: ${passedTests}`);
    console.log(`   Failed: ${failedTests}`);
    console.log(`   Skipped: ${skippedTests}`);
    console.log(`   Pass Rate: ${summaryReport.metrics.passRate.toFixed(1)}%`);
    console.log(`   Duration: ${testDuration}ms`);
    
    if (qualityGateStatus.overall === 'PASSED') {
      console.log('✅ Quality Gates: PASSED');
    } else {
      console.log('❌ Quality Gates: FAILED');
      qualityGateStatus.recommendations.forEach(rec => {
        console.log(`   ⚠️ ${rec}`);
      });
    }

  } catch (error) {
    console.error('❌ Error generating test summary:', error);
  }

  // Clean up temporary test data
  console.log('🗄️ Cleaning up test data...');
  
  const fixturesDir = path.join(process.cwd(), 'tests/e2e/fixtures');
  if (fs.existsSync(fixturesDir)) {
    // Keep fixtures but clean any temporary test state files
    const tempFiles = ['temp-*.json', 'session-*.json'];
    tempFiles.forEach(pattern => {
      // Simple cleanup - in production you might use a proper glob library
      if (pattern.includes('*')) {
        const prefix = pattern.split('*')[0];
        const files = fs.readdirSync(fixturesDir)
          .filter(file => file.startsWith(prefix));
        files.forEach(file => {
          fs.unlinkSync(path.join(fixturesDir, file));
        });
      }
    });
  }

  console.log('✅ PWA E2E Testing Global Teardown Complete');
}

export default globalTeardown;