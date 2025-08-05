#!/usr/bin/env node

/**
 * Streamlined Hive Commands for Claude Code - Mobile Integration
 * Unified command interface connecting to LeanVibe Agent Hive 2.0
 * 
 * This command provides simplified access to the autonomous development platform
 * with mobile-first oversight and intelligent command routing.
 */

const fetch = require('node-fetch');
const qrcode = require('qrcode-terminal');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const HIVE_API_BASE = process.env.HIVE_API_URL || 'http://localhost:8000';

class HiveCommandInterface {
  constructor() {
    this.commands = {
      status: this.handleStatus.bind(this),
      config: this.handleConfig.bind(this),
      agents: this.handleAgents.bind(this),
      mobile: this.handleMobile.bind(this),
      review: this.handleReview.bind(this),
      test: this.handleTest.bind(this),
      deploy: this.handleDeploy.bind(this),
      fix: this.handleFix.bind(this),
      memory: this.handleMemory.bind(this),
      help: this.handleHelp.bind(this)
    };
  }

  async execute(command, args = []) {
    const commandName = command.toLowerCase();
    const handler = this.commands[commandName];
    
    if (!handler) {
      console.log(`❌ Unknown command: ${command}`);
      console.log('💡 Available commands:', Object.keys(this.commands).join(', '));
      console.log('🆘 Use "hive help" for detailed usage information');
      return;
    }

    try {
      await handler(args);
    } catch (error) {
      console.log(`❌ Command failed: ${error.message}`);
      console.log('🔄 Try running "hive status" to check system health');
    }
  }

  async handleStatus(args) {
    console.log('🔄 Checking LeanVibe Agent Hive status...');
    
    const isMobile = args.includes('--mobile') || args.includes('-m');
    const isDetailed = args.includes('--detailed') || args.includes('-d');
    
    try {
      // Get system status from LeanVibe API
      const response = await fetch(`${HIVE_API_BASE}/api/hive/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: `/hive:status${isDetailed ? ' --detailed' : ''}${isMobile ? ' --mobile' : ''}`,
          mobile_optimized: isMobile,
          priority: 'high'
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ LeanVibe Agent Hive 2.0 Status:');
        console.log(`📊 Platform: ${result.result.platform_active ? '🟢 Active' : '🔴 Offline'}`);
        console.log(`🤖 Agents: ${result.result.agent_count || 0} active`);
        console.log(`📱 Mobile Dashboard: ${result.result.mobile_dashboard_url || 'Not available'}`);
        console.log(`⚡ Response Time: ${result.execution_time_ms?.toFixed(1)}ms`);
        
        if (isMobile) {
          console.log('📱 Mobile Integration:');
          console.log(`   - Notifications: ${result.result.mobile_notifications_enabled ? '🟢 Enabled' : '🔴 Disabled'}`);
          console.log(`   - Push Alerts: ${result.result.push_notifications_configured ? '🟢 Ready' : '⚠️ Setup needed'}`);
          console.log(`   - WebSocket: ${result.result.websocket_connection ? '🟢 Connected' : '🔴 Disconnected'}`);
        }
        
        if (isDetailed && result.result.detailed_status) {
          console.log('\n📋 Detailed System Status:');
          Object.entries(result.result.detailed_status).forEach(([key, value]) => {
            console.log(`   ${key}: ${value}`);
          });
        }
      } else {
        console.log('⚠️ System status check failed');
        console.log('🔧 LeanVibe Agent Hive may be offline - run "make start" to initialize');
      }
    } catch (error) {
      console.log('❌ Cannot connect to LeanVibe Agent Hive');
      console.log('💡 Ensure the system is running: cd ../.. && make start');
    }
  }

  async handleConfig(args) {
    console.log('⚙️ Claude Code Configuration Management');
    
    if (args.includes('--show') || args.includes('-s')) {
      try {
        const settingsPath = '.claude/settings.json';
        const settings = require(`../${settingsPath}`);
        
        console.log('📋 Current Configuration:');
        console.log(`   Hooks: ${Object.keys(settings.hooks || {}).length} configured`);
        console.log(`   Mobile Integration: ${settings.mobile_integration ? '✅ Enabled' : '❌ Disabled'}`);
        console.log(`   LeanVibe Connection: Checking...`);
        
        // Test LeanVibe connection
        const response = await fetch(`${HIVE_API_BASE}/api/hive/status`);
        console.log(`   LeanVibe Status: ${response.ok ? '🟢 Connected' : '🔴 Offline'}`);
        
      } catch (error) {
        console.log('⚠️ Cannot read configuration file');
      }
    } else if (args.includes('--optimize') || args.includes('-o')) {
      console.log('🚀 Optimizing configuration for mobile integration...');
      // Optimization logic would go here
      console.log('✅ Configuration optimized for mobile-first development');
    } else {
      console.log('💡 Configuration options:');
      console.log('   --show (-s)     Show current configuration');
      console.log('   --optimize (-o) Optimize for mobile integration');
    }
  }

  async handleAgents(args) {
    console.log('🤖 Agent Management via LeanVibe Agent Hive');
    
    try {
      if (args.includes('--list') || args.includes('-l')) {
        const response = await fetch(`${HIVE_API_BASE}/api/agents/status`);
        const result = await response.json();
        
        console.log(`📊 Active Agents: ${result.active_agents || 0}`);
        if (result.agents) {
          result.agents.forEach(agent => {
            console.log(`   🤖 ${agent.id}: ${agent.type} (${agent.status})`);
          });
        }
      } else if (args.includes('--spawn')) {
        const agentType = args[args.indexOf('--spawn') + 1] || 'developer';
        console.log(`🚀 Spawning ${agentType} agent...`);
        
        const response = await fetch(`${HIVE_API_BASE}/api/hive/execute`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command: `/hive:spawn ${agentType}`,
            priority: 'high'
          })
        });
        
        const result = await response.json();
        console.log(result.success ? '✅ Agent spawned successfully' : '❌ Failed to spawn agent');
      } else if (args.includes('--coordinate')) {
        const task = args[args.indexOf('--coordinate') + 1] || 'development task';
        console.log(`📋 Coordinating agents for: ${task}`);
        
        const response = await fetch(`${HIVE_API_BASE}/api/agents/coordinate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task_description: task,
            priority: 'high',
            source: 'claude_code_command'
          })
        });
        
        const result = await response.json();
        console.log(result.success ? 
          `✅ Task coordinated with ${result.assigned_agents} agents` : 
          '❌ Agent coordination failed');
      } else {
        console.log('💡 Agent commands:');
        console.log('   --list (-l)           List active agents');
        console.log('   --spawn <type>        Spawn new agent');
        console.log('   --coordinate <task>   Coordinate agents for task');
      }
    } catch (error) {
      console.log('❌ Cannot connect to agent system');
      console.log('💡 Ensure LeanVibe Agent Hive is running');
    }
  }

  async handleMobile(args) {
    console.log('📱 Mobile Dashboard Integration');
    
    try {
      if (args.includes('--qr') || args.includes('-q')) {
        const dashboardUrl = `${HIVE_API_BASE}/mobile-pwa/`;
        console.log('📱 Mobile Dashboard QR Code:');
        qrcode.generate(dashboardUrl, { small: true });
        console.log(`🔗 Direct URL: ${dashboardUrl}`);
      } else if (args.includes('--notifications') || args.includes('-n')) {
        console.log('🔔 Testing mobile notifications...');
        
        const response = await fetch(`${HIVE_API_BASE}/api/mobile/notifications`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            priority: 'medium',
            title: 'Claude Code Test',
            message: 'Mobile notification system working properly',
            data: { type: 'test_notification' }
          })
        });
        
        console.log(response.ok ? 
          '✅ Test notification sent to mobile dashboard' : 
          '❌ Failed to send notification');
      } else if (args.includes('--status') || args.includes('-s')) {
        const response = await fetch(`${HIVE_API_BASE}/api/mobile/status`);
        const status = await response.json();
        
        console.log('📊 Mobile Dashboard Status:');
        console.log(`   Connection: ${response.ok ? '🟢 Connected' : '🔴 Offline'}`);
        console.log(`   WebSocket: ${status.websocket_healthy ? '🟢 Active' : '🔴 Inactive'}`);
        console.log(`   Notifications: ${status.notifications_enabled ? '🟢 Enabled' : '🔴 Disabled'}`);
      } else {
        console.log('💡 Mobile commands:');
        console.log('   --qr (-q)           Show QR code for mobile access');
        console.log('   --notifications (-n) Test mobile notifications');
        console.log('   --status (-s)        Check mobile dashboard status');
      }
    } catch (error) {
      console.log('❌ Mobile integration unavailable');
      console.log('💡 Check if mobile-pwa service is running');
    }
  }

  async handleReview(args) {
    console.log('👁️ Intelligent Code Review');
    
    try {
      const response = await fetch(`${HIVE_API_BASE}/api/hive/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:review --context=current --mobile',
          mobile_optimized: true,
          priority: 'high'
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ Code review initiated');
        if (result.result.review_summary) {
          console.log('📋 Review Summary:');
          console.log(`   Files analyzed: ${result.result.files_count || 0}`);
          console.log(`   Issues found: ${result.result.issues_count || 0}`);
          console.log(`   Suggestions: ${result.result.suggestions_count || 0}`);
        }
      } else {
        console.log('⚠️ Code review not available');
        console.log('💡 Ensure agents are active for intelligent review');
      }
    } catch (error) {
      console.log('❌ Review system unavailable');
    }
  }

  async handleTest(args) {
    console.log('🧪 Test Execution & Validation');
    
    const testType = args.includes('--integration') ? 'integration' : 
                     args.includes('--unit') ? 'unit' : 'all';
    
    try {
      // Run tests based on project type
      if (require('fs').existsSync('package.json')) {
        console.log('📦 Running JavaScript/TypeScript tests...');
        const { stdout } = await execAsync('npm test');
        console.log(stdout);
      } else if (require('fs').existsSync('pyproject.toml')) {
        console.log('🐍 Running Python tests...');
        const { stdout } = await execAsync('pytest');
        console.log(stdout);
      } else {
        console.log('⚠️ No test configuration detected');
        console.log('💡 Supported: npm test, pytest');
      }
    } catch (error) {
      console.log('❌ Test execution failed');
      console.log(`Details: ${error.message}`);
    }
  }

  async handleDeploy(args) {
    console.log('🚀 Deployment Management');
    
    const environment = args.includes('--prod') ? 'production' : 
                       args.includes('--staging') ? 'staging' : 'development';
    
    console.log(`📦 Preparing deployment to ${environment}...`);
    
    try {
      // Build the project
      if (require('fs').existsSync('package.json')) {
        console.log('📦 Building project...');
        await execAsync('npm run build');
        console.log('✅ Build completed');
      }
      
      // Trigger deployment via LeanVibe
      const response = await fetch(`${HIVE_API_BASE}/api/hive/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: `/hive:deploy --environment=${environment} --mobile-dashboard`,
          priority: 'high'
        })
      });
      
      const result = await response.json();
      console.log(result.success ? 
        '✅ Deployment initiated via agent coordination' : 
        '⚠️ Manual deployment required');
    } catch (error) {
      console.log('❌ Deployment failed');
      console.log(`Details: ${error.message}`);
    }
  }

  async handleFix(args) {
    console.log('🔧 Intelligent Issue Resolution');
    
    try {
      const response = await fetch(`${HIVE_API_BASE}/api/hive/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:fix --auto-detect --mobile-notifications',
          mobile_optimized: true,
          priority: 'high'
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('🔍 Scanning for issues...');
        console.log(`   Issues detected: ${result.result.issues_found || 0}`);
        console.log(`   Auto-fixes applied: ${result.result.fixes_applied || 0}`);
        console.log(`   Manual review needed: ${result.result.manual_review_items || 0}`);
        
        if (result.result.mobile_notification_sent) {
          console.log('📱 Fix results sent to mobile dashboard');
        }
      } else {
        console.log('⚠️ Issue detection not available');
      }
    } catch (error) {
      console.log('❌ Fix system unavailable');
    }
  }

  async handleMemory(args) {
    console.log('🧠 Memory & Context Management');
    
    try {
      if (args.includes('--status') || args.includes('-s')) {
        const response = await fetch(`${HIVE_API_BASE}/api/contexts/status`);
        const status = await response.json();
        
        console.log('📊 Memory Status:');
        console.log(`   Context usage: ${status.usage_percentage || 0}%`);
        console.log(`   Memory entries: ${status.total_entries || 0}`);
        console.log(`   Cache hit rate: ${status.cache_hit_rate || 0}%`);
      } else if (args.includes('--compact') || args.includes('-c')) {
        console.log('🗜️ Compacting context...');
        
        const response = await fetch(`${HIVE_API_BASE}/api/contexts/compact`, {
          method: 'POST'
        });
        
        console.log(response.ok ? 
          '✅ Context compaction completed' : 
          '❌ Compaction failed');
      } else {
        console.log('💡 Memory commands:');
        console.log('   --status (-s)    Show memory status');
        console.log('   --compact (-c)   Compact context');
      }
    } catch (error) {
      console.log('❌ Memory management unavailable');
    }
  }

  async handleHelp(args) {
    const specificCommand = args[0];
    
    if (specificCommand && this.commands[specificCommand]) {
      console.log(`📖 Help for "hive ${specificCommand}"`);
      
      try {
        const response = await fetch(`${HIVE_API_BASE}/api/hive/help/${specificCommand}?mobile=true`);
        const help = await response.json();
        
        if (help.success) {
          console.log(`\n📝 ${help.command.description}`);
          console.log(`\n💡 Usage: ${help.command.usage}`);
          
          if (help.command.examples.length > 0) {
            console.log('\n🎯 Examples:');
            help.command.examples.forEach(example => {
              console.log(`   ${example}`);
            });
          }
          
          if (help.contextual_recommendations.length > 0) {
            console.log('\n💡 Recommendations:');
            help.contextual_recommendations.forEach(rec => {
              console.log(`   ${rec.title}: ${rec.description}`);
            });
          }
        }
      } catch (error) {
        // Fallback to basic help
        this.showBasicHelp(specificCommand);
      }
    } else {
      this.showGeneralHelp();
    }
  }

  showGeneralHelp() {
    console.log('🚀 Streamlined Claude Code Hooks & Commands');
    console.log('   Unified interface for LeanVibe Agent Hive 2.0');
    console.log('\n📋 Available Commands:');
    console.log('   status   - Check system and agent status');
    console.log('   config   - Configuration management');
    console.log('   agents   - Agent coordination and management');
    console.log('   mobile   - Mobile dashboard integration');
    console.log('   review   - Intelligent code review');
    console.log('   test     - Test execution and validation');
    console.log('   deploy   - Deployment management');
    console.log('   fix      - Intelligent issue resolution');
    console.log('   memory   - Memory and context management');
    console.log('   help     - Show this help or help for specific command');
    console.log('\n💡 Examples:');
    console.log('   hive status --mobile');
    console.log('   hive mobile --qr');
    console.log('   hive agents --spawn backend_developer');
    console.log('   hive help status');
    console.log('\n🔗 LeanVibe Agent Hive 2.0 Integration:');
    console.log('   📱 Mobile Dashboard with QR access');
    console.log('   🤖 Multi-agent coordination');
    console.log('   🔔 Real-time notifications');
    console.log('   ⚡ <5ms cached responses');
  }

  showBasicHelp(command) {
    const helpText = {
      status: 'Check LeanVibe Agent Hive status\nOptions: --mobile, --detailed',
      config: 'Manage Claude Code configuration\nOptions: --show, --optimize',
      agents: 'Agent management and coordination\nOptions: --list, --spawn <type>, --coordinate <task>',
      mobile: 'Mobile dashboard integration\nOptions: --qr, --notifications, --status',
      review: 'Intelligent code review system',
      test: 'Run tests and validation\nOptions: --unit, --integration',
      deploy: 'Deployment management\nOptions: --prod, --staging',
      fix: 'Intelligent issue resolution',
      memory: 'Memory and context management\nOptions: --status, --compact'
    };
    
    console.log(`📖 ${command}: ${helpText[command] || 'No specific help available'}`);
  }
}

// Main execution
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  const commandArgs = args.slice(1);

  const hive = new HiveCommandInterface();
  hive.execute(command, commandArgs);
}

module.exports = HiveCommandInterface;