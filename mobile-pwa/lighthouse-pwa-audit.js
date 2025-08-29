import lighthouse from 'lighthouse';
import * as chromeLauncher from 'chrome-launcher';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Lighthouse PWA Audit Script
 * Tests PWA compliance and performance for Agent Hive Mobile PWA
 */

const PWA_REQUIREMENTS = {
  minScore: 90,
  categories: ['pwa', 'performance', 'accessibility', 'best-practices'],
  requiredAudits: [
    'installable-manifest',
    'splash-screen',
    'themed-omnibox',
    'content-width',
    'viewport',
    'without-javascript',
    'service-worker',
    'offline-start-url',
    'apple-touch-icon',
    'maskable-icon'
  ]
};

const URLS_TO_TEST = [
  'http://localhost:5001',
  'http://localhost:5001/dashboard',
  'http://localhost:5001/tasks',
  'http://localhost:5001/agents'
];

async function launchChromeAndRunLighthouse(url, opts = {}) {
  const chrome = await chromeLauncher.launch({
    chromeFlags: [
      '--headless',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-dev-shm-usage'
    ]
  });
  
  opts.port = chrome.port;
  
  const config = {
    extends: 'lighthouse:default',
    settings: {
      formFactor: 'mobile',
      throttling: {
        rttMs: 40,
        throughputKbps: 10240,
        cpuSlowdownMultiplier: 1,
        requestLatencyMs: 0,
        downloadThroughputKbps: 0,
        uploadThroughputKbps: 0
      },
      screenEmulation: {
        mobile: true,
        width: 390,
        height: 844,
        deviceScaleFactor: 3,
        disabled: false
      },
      emulatedUserAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
    }
  };
  
  try {
    const runnerResult = await lighthouse(url, opts, config);
    await chrome.kill();
    return runnerResult;
  } catch (error) {
    await chrome.kill();
    throw error;
  }
}

function analyzePWACompliance(lhr) {
  const pwaCategory = lhr.categories.pwa;
  const audits = lhr.audits;
  
  const results = {
    score: Math.round(pwaCategory.score * 100),
    passed: pwaCategory.score >= PWA_REQUIREMENTS.minScore / 100,
    audits: {},
    recommendations: []
  };
  
  // Check required PWA audits
  PWA_REQUIREMENTS.requiredAudits.forEach(auditId => {
    if (audits[auditId]) {
      const audit = audits[auditId];
      results.audits[auditId] = {
        passed: audit.score === 1,
        score: audit.score,
        title: audit.title,
        description: audit.description,
        displayValue: audit.displayValue
      };
      
      if (audit.score < 1) {
        results.recommendations.push({
          audit: auditId,
          title: audit.title,
          recommendation: getRecommendation(auditId, audit)
        });
      }
    }
  });
  
  return results;
}

function getRecommendation(auditId, audit) {
  const recommendations = {
    'installable-manifest': 'Ensure your web app manifest is properly configured with all required fields (name, short_name, start_url, display, icons)',
    'splash-screen': 'Add icons with sizes 192x192 and 512x512 to your manifest for splash screen support',
    'themed-omnibox': 'Set a theme_color in your manifest to customize the browser address bar',
    'content-width': 'Ensure your viewport meta tag is properly configured: <meta name="viewport" content="width=device-width, initial-scale=1">',
    'viewport': 'Add a viewport meta tag to optimize for mobile devices',
    'without-javascript': 'Ensure your app displays some content when JavaScript is disabled',
    'service-worker': 'Register a service worker to enable offline functionality',
    'offline-start-url': 'Ensure your start_url is cached by the service worker for offline access',
    'apple-touch-icon': 'Add apple-touch-icon links for iOS home screen support',
    'maskable-icon': 'Include maskable icons in your manifest for better Android home screen appearance'
  };
  
  return recommendations[auditId] || 'Review the audit details for specific recommendations';
}

function analyzePerformance(lhr) {
  const performanceCategory = lhr.categories.performance;
  const audits = lhr.audits;
  
  return {
    score: Math.round(performanceCategory.score * 100),
    passed: performanceCategory.score >= 0.9,
    metrics: {
      fcp: audits['first-contentful-paint'].numericValue,
      lcp: audits['largest-contentful-paint'].numericValue,
      cls: audits['cumulative-layout-shift'].numericValue,
      fid: audits['max-potential-fid']?.numericValue || 0,
      tti: audits['interactive'].numericValue
    },
    opportunities: Object.keys(audits)
      .filter(key => audits[key].details?.type === 'opportunity')
      .map(key => ({
        audit: key,
        title: audits[key].title,
        savings: audits[key].details.overallSavingsMs || 0
      }))
      .sort((a, b) => b.savings - a.savings)
      .slice(0, 5)
  };
}

async function runAudit(url) {
  console.log(`🔍 Auditing ${url}...`);
  
  try {
    const runnerResult = await launchChromeAndRunLighthouse(url);
    const lhr = runnerResult.lhr;
    
    const pwaAnalysis = analyzePWACompliance(lhr);
    const performanceAnalysis = analyzePerformance(lhr);
    
    const result = {
      url,
      timestamp: new Date().toISOString(),
      pwa: pwaAnalysis,
      performance: performanceAnalysis,
      categories: {
        pwa: Math.round(lhr.categories.pwa.score * 100),
        performance: Math.round(lhr.categories.performance.score * 100),
        accessibility: Math.round(lhr.categories.accessibility.score * 100),
        bestPractices: Math.round(lhr.categories['best-practices'].score * 100),
        seo: Math.round(lhr.categories.seo.score * 100)
      }
    };
    
    return result;
  } catch (error) {
    console.error(`❌ Failed to audit ${url}:`, error.message);
    return {
      url,
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
}

function generateReport(results) {
  const report = {
    summary: {
      timestamp: new Date().toISOString(),
      totalUrls: results.length,
      passedPWA: results.filter(r => r.pwa?.passed).length,
      averageScores: {}
    },
    results: results,
    recommendations: []
  };
  
  // Calculate average scores
  const validResults = results.filter(r => !r.error);
  if (validResults.length > 0) {
    report.summary.averageScores = {
      pwa: Math.round(validResults.reduce((sum, r) => sum + r.categories.pwa, 0) / validResults.length),
      performance: Math.round(validResults.reduce((sum, r) => sum + r.categories.performance, 0) / validResults.length),
      accessibility: Math.round(validResults.reduce((sum, r) => sum + r.categories.accessibility, 0) / validResults.length),
      bestPractices: Math.round(validResults.reduce((sum, r) => sum + r.categories.bestPractices, 0) / validResults.length),
      seo: Math.round(validResults.reduce((sum, r) => sum + r.categories.seo, 0) / validResults.length)
    };
  }
  
  // Collect all recommendations
  const allRecommendations = new Set();
  results.forEach(result => {
    if (result.pwa?.recommendations) {
      result.pwa.recommendations.forEach(rec => {
        allRecommendations.add(JSON.stringify(rec));
      });
    }
  });
  
  report.recommendations = Array.from(allRecommendations).map(r => JSON.parse(r));
  
  return report;
}

function printResults(report) {
  console.log('\n📊 PWA AUDIT RESULTS\n');
  console.log('='.repeat(50));
  
  console.log(`\n📈 OVERALL SCORES:`);
  console.log(`PWA Score: ${report.summary.averageScores.pwa || 0}/100 ${report.summary.averageScores.pwa >= 90 ? '✅' : '❌'}`);
  console.log(`Performance: ${report.summary.averageScores.performance || 0}/100 ${report.summary.averageScores.performance >= 90 ? '✅' : '⚠️'}`);
  console.log(`Accessibility: ${report.summary.averageScores.accessibility || 0}/100 ${report.summary.averageScores.accessibility >= 90 ? '✅' : '⚠️'}`);
  console.log(`Best Practices: ${report.summary.averageScores.bestPractices || 0}/100 ${report.summary.averageScores.bestPractices >= 90 ? '✅' : '⚠️'}`);
  console.log(`SEO: ${report.summary.averageScores.seo || 0}/100 ${report.summary.averageScores.seo >= 90 ? '✅' : '⚠️'}`);
  
  console.log(`\n📱 PWA COMPLIANCE:`);
  console.log(`URLs Passed: ${report.summary.passedPWA}/${report.summary.totalUrls}`);
  
  if (report.recommendations.length > 0) {
    console.log(`\n🔧 RECOMMENDATIONS:`);
    report.recommendations.forEach((rec, index) => {
      console.log(`${index + 1}. ${rec.title}`);
      console.log(`   ${rec.recommendation}\n`);
    });
  }
  
  console.log('\n📋 DETAILED RESULTS:');
  report.results.forEach(result => {
    if (result.error) {
      console.log(`❌ ${result.url}: ${result.error}`);
    } else {
      console.log(`\n🌐 ${result.url}:`);
      console.log(`   PWA: ${result.categories.pwa}/100 ${result.categories.pwa >= 90 ? '✅' : '❌'}`);
      console.log(`   Performance: ${result.categories.performance}/100`);
      console.log(`   Accessibility: ${result.categories.accessibility}/100`);
    }
  });
  
  console.log('\n='.repeat(50));
  
  if (report.summary.averageScores.pwa >= 90) {
    console.log('🎉 CONGRATULATIONS! Your PWA meets the >90 score requirement!');
  } else {
    console.log('⚠️  PWA score is below 90. Please address the recommendations above.');
  }
}

async function main() {
  console.log('🚀 Starting PWA Lighthouse Audit...');
  console.log(`Testing ${URLS_TO_TEST.length} URLs...`);
  
  const results = [];
  
  for (const url of URLS_TO_TEST) {
    const result = await runAudit(url);
    results.push(result);
  }
  
  const report = generateReport(results);
  
  // Save detailed report
  const reportPath = path.join(__dirname, 'lighthouse-pwa-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n💾 Detailed report saved to: ${reportPath}`);
  
  // Print summary
  printResults(report);
  
  // Exit with appropriate code
  const success = report.summary.averageScores.pwa >= 90;
  process.exit(success ? 0 : 1);
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('❌ Audit failed:', error);
    process.exit(1);
  });
}

export {
  runAudit,
  analyzePWACompliance,
  analyzePerformance,
  generateReport
};