#!/usr/bin/env node

/**
 * Comprehensive E2E Test Runner for LeanVibe Agent Hive 2.0
 * 
 * This script provides a unified interface for running different types of E2E tests
 * with proper setup, configuration, and reporting.
 */

import { execSync, spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Test configurations
const TEST_CONFIGS = {
  smoke: {
    description: 'Quick smoke tests for critical functionality',
    projects: ['smoke'],
    timeout: 30000,
    maxFailures: 3
  },
  
  full: {
    description: 'Complete E2E test suite across all browsers',
    projects: ['chromium', 'firefox', 'webkit'],
    timeout: 60000,
    maxFailures: 10
  },
  
  mobile: {
    description: 'Mobile and tablet responsiveness tests',
    projects: ['Mobile Chrome', 'Mobile Safari', 'iPad'],
    timeout: 45000,
    maxFailures: 5
  },
  
  pwa: {
    description: 'Progressive Web App functionality tests',
    projects: ['Mobile Chrome', 'iPad'],
    grep: 'PWA|offline|manifest|service worker',
    timeout: 60000,
    maxFailures: 5
  },
  
  performance: {
    description: 'Performance and Core Web Vitals validation',
    projects: ['performance'],
    timeout: 90000,
    maxFailures: 2
  },
  
  accessibility: {
    description: 'WCAG AA accessibility compliance tests',
    projects: ['accessibility'],
    timeout: 45000,
    maxFailures: 3
  },
  
  visual: {
    description: 'Visual regression testing',
    projects: ['visual-regression'],
    timeout: 60000,
    maxFailures: 1
  },
  
  agents: {
    description: 'Agent workflow and coordination tests',
    projects: ['chromium'],
    grep: 'agent|workflow|autonomous',
    timeout: 120000,
    maxFailures: 5
  },
  
  realtime: {
    description: 'WebSocket and real-time functionality tests',
    projects: ['chromium'],
    grep: 'websocket|real-time|live',
    timeout: 60000,
    maxFailures: 3
  }
}

class E2ETestRunner {
  constructor() {
    this.startTime = Date.now()
    this.results = {}
  }

  /**
   * Parse command line arguments
   */
  parseArgs() {
    const args = process.argv.slice(2)
    const config = {
      suite: 'smoke',
      headed: false,
      debug: false,
      update: false,
      parallel: true,
      retries: process.env.CI ? 2 : 0,
      workers: process.env.CI ? 1 : undefined,
      reporter: 'html,github',
      verbose: false,
      help: false
    }

    for (let i = 0; i < args.length; i++) {
      const arg = args[i]
      
      switch (arg) {
        case '--suite':
        case '-s':
          config.suite = args[++i]
          break
        case '--headed':
          config.headed = true
          break
        case '--debug':
          config.debug = true
          config.headed = true
          config.parallel = false
          break
        case '--update-snapshots':
          config.update = true
          break
        case '--no-parallel':
          config.parallel = false
          break
        case '--retries':
          config.retries = parseInt(args[++i])
          break
        case '--workers':
          config.workers = parseInt(args[++i])
          break
        case '--reporter':
          config.reporter = args[++i]
          break
        case '--verbose':
        case '-v':
          config.verbose = true
          break
        case '--help':
        case '-h':
          config.help = true
          break
        default:
          if (TEST_CONFIGS[arg]) {
            config.suite = arg
          }
      }
    }

    return config
  }

  /**
   * Display help information
   */
  showHelp() {
    console.log(`
🎭 LeanVibe Agent Hive 2.0 - E2E Test Runner

Usage: npm run test:e2e [options] [suite]

Test Suites:`)

    Object.entries(TEST_CONFIGS).forEach(([name, config]) => {
      console.log(`  ${name.padEnd(12)} - ${config.description}`)
    })

    console.log(`
Options:
  --suite, -s <name>     Test suite to run (default: smoke)
  --headed               Run tests in headed mode (visible browser)
  --debug                Debug mode (headed + serial + verbose)
  --update-snapshots     Update visual test snapshots
  --no-parallel          Run tests serially
  --retries <number>     Number of retries (default: 0 local, 2 CI)
  --workers <number>     Number of parallel workers
  --reporter <reporter>  Test reporter (default: html,github)
  --verbose, -v          Verbose output
  --help, -h             Show this help

Examples:
  npm run test:e2e                    # Run smoke tests
  npm run test:e2e smoke              # Run smoke tests
  npm run test:e2e full --headed      # Run full suite with visible browser
  npm run test:e2e mobile --debug     # Debug mobile tests
  npm run test:e2e visual --update-snapshots  # Update visual snapshots
  npm run test:e2e performance --verbose      # Performance tests with verbose output
`)
  }

  /**
   * Check prerequisites
   */
  checkPrerequisites() {
    console.log('🔍 Checking prerequisites...')
    
    // Check if Playwright is installed
    try {
      execSync('npx playwright --version', { stdio: 'pipe' })
      console.log('✅ Playwright is installed')
    } catch (error) {
      console.error('❌ Playwright not found. Please run: npx playwright install')
      process.exit(1)
    }

    // Check if browsers are installed
    try {
      execSync('npx playwright install-deps', { stdio: 'pipe' })
      console.log('✅ Browser dependencies are installed')
    } catch (error) {
      console.warn('⚠️  Some browser dependencies may be missing')
    }

    // Check if build exists
    const distPath = path.join(__dirname, '..', 'dist')
    if (!fs.existsSync(distPath)) {
      console.log('📦 Building application...')
      try {
        execSync('npm run build', { stdio: 'inherit' })
        console.log('✅ Application built successfully')
      } catch (error) {
        console.error('❌ Build failed')
        process.exit(1)
      }
    }
  }

  /**
   * Setup test environment
   */
  setupEnvironment(config) {
    console.log('🛠️  Setting up test environment...')
    
    // Set environment variables
    process.env.NODE_ENV = 'test'
    process.env.CI = process.env.CI || 'false'
    
    if (config.debug) {
      process.env.PWDEBUG = '1'
      process.env.DEBUG = 'pw:api'
    }
    
    if (config.verbose) {
      process.env.DEBUG = 'pw:*'
    }

    // Create test results directory
    const resultsDir = path.join(__dirname, '..', 'test-results')
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true })
    }

    console.log('✅ Environment configured')
  }

  /**
   * Build Playwright command
   */
  buildPlaywrightCommand(config) {
    const testConfig = TEST_CONFIGS[config.suite]
    if (!testConfig) {
      throw new Error(`Unknown test suite: ${config.suite}`)
    }

    const command = ['npx', 'playwright', 'test']
    
    // Add projects
    testConfig.projects.forEach(project => {
      command.push('--project', project)
    })
    
    // Add grep pattern if specified
    if (testConfig.grep) {
      command.push('--grep', testConfig.grep)
    }
    
    // Add configuration options
    if (config.headed) {
      command.push('--headed')
    }
    
    if (!config.parallel) {
      command.push('--workers', '1')
    } else if (config.workers) {
      command.push('--workers', config.workers.toString())
    }
    
    if (config.retries) {
      command.push('--retries', config.retries.toString())
    }
    
    if (testConfig.timeout) {
      command.push('--timeout', testConfig.timeout.toString())
    }
    
    if (testConfig.maxFailures) {
      command.push('--max-failures', testConfig.maxFailures.toString())
    }
    
    if (config.update) {
      command.push('--update-snapshots')
    }
    
    if (config.reporter) {
      command.push('--reporter', config.reporter)
    }

    return command
  }

  /**
   * Run tests
   */
  async runTests(config) {
    const testConfig = TEST_CONFIGS[config.suite]
    console.log(`\n🚀 Running ${testConfig.description}...`)
    console.log(`📋 Projects: ${testConfig.projects.join(', ')}`)
    
    if (testConfig.grep) {
      console.log(`🔍 Filter: ${testConfig.grep}`)
    }
    
    console.log('')

    const command = this.buildPlaywrightCommand(config)
    console.log(`💻 Command: ${command.join(' ')}`)
    console.log('')

    return new Promise((resolve, reject) => {
      const process = spawn(command[0], command.slice(1), {
        stdio: 'inherit',
        cwd: path.join(__dirname, '..')
      })

      process.on('close', (code) => {
        this.results[config.suite] = {
          code,
          success: code === 0,
          duration: Date.now() - this.startTime
        }
        
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`Tests failed with exit code ${code}`))
        }
      })

      process.on('error', (error) => {
        reject(error)
      })
    })
  }

  /**
   * Generate test report
   */
  generateReport(config) {
    const result = this.results[config.suite]
    const duration = Math.round(result.duration / 1000)
    
    console.log('\n📊 Test Results Summary')
    console.log('=' .repeat(50))
    console.log(`Suite: ${config.suite}`)
    console.log(`Description: ${TEST_CONFIGS[config.suite].description}`)
    console.log(`Duration: ${duration}s`)
    console.log(`Status: ${result.success ? '✅ PASSED' : '❌ FAILED'}`)
    
    if (result.success) {
      console.log('\n🎉 All tests passed successfully!')
    } else {
      console.log('\n💥 Some tests failed. Check the detailed report.')
    }
    
    // Show report locations
    console.log('\n📋 Reports Available:')
    console.log(`- HTML Report: playwright-report/index.html`)
    console.log(`- JSON Results: test-results/results.json`)
    console.log(`- JUnit XML: test-results/junit.xml`)
    
    if (fs.existsSync(path.join(__dirname, '..', 'test-results'))) {
      console.log(`- Screenshots: test-results/`)
    }
  }

  /**
   * Main execution function
   */
  async run() {
    const config = this.parseArgs()
    
    if (config.help) {
      this.showHelp()
      return
    }

    try {
      console.log('🎭 LeanVibe Agent Hive 2.0 - E2E Test Runner\n')
      
      this.checkPrerequisites()
      this.setupEnvironment(config)
      await this.runTests(config)
      this.generateReport(config)
      
    } catch (error) {
      console.error('\n💥 Test execution failed:', error.message)
      
      if (this.results[config.suite]) {
        this.generateReport(config)
      }
      
      process.exit(1)
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new E2ETestRunner()
  runner.run()
}

export default E2ETestRunner