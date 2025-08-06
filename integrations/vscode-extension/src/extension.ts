import * as vscode from 'vscode';
import { HiveManager } from './hive/HiveManager';
import { AgentTreeProvider } from './providers/AgentTreeProvider';
import { TaskTreeProvider } from './providers/TaskTreeProvider';
import { LogsTreeProvider } from './providers/LogsTreeProvider';
import { SettingsTreeProvider } from './providers/SettingsTreeProvider';
import { DashboardWebviewProvider } from './providers/DashboardWebviewProvider';
import { CodeSuggestionProvider } from './providers/CodeSuggestionProvider';
import { HiveStatusBar } from './ui/HiveStatusBar';
import { HiveCommands } from './commands/HiveCommands';
import { ConfigurationManager } from './utils/ConfigurationManager';
import { Logger } from './utils/Logger';

export function activate(context: vscode.ExtensionContext) {
    Logger.info('Activating LeanVibe Agent Hive extension');

    // Initialize core components
    const configManager = new ConfigurationManager();
    const hiveManager = new HiveManager(configManager);
    const statusBar = new HiveStatusBar();
    
    // Initialize tree data providers
    const agentProvider = new AgentTreeProvider(hiveManager);
    const taskProvider = new TaskTreeProvider(hiveManager);
    const logsProvider = new LogsTreeProvider(hiveManager);
    const settingsProvider = new SettingsTreeProvider(configManager);
    
    // Initialize webview providers
    const dashboardProvider = new DashboardWebviewProvider(context.extensionUri, hiveManager);
    
    // Initialize code enhancement providers
    const codeSuggestionProvider = new CodeSuggestionProvider(hiveManager, configManager);
    
    // Register tree views
    const agentTreeView = vscode.window.createTreeView('leanvibe.agents', {
        treeDataProvider: agentProvider,
        showCollapseAll: true
    });
    
    const taskTreeView = vscode.window.createTreeView('leanvibe.tasks', {
        treeDataProvider: taskProvider,
        showCollapseAll: true
    });
    
    const logsTreeView = vscode.window.createTreeView('leanvibe.logs', {
        treeDataProvider: logsProvider,
        showCollapseAll: true
    });
    
    const settingsTreeView = vscode.window.createTreeView('leanvibe.settings', {
        treeDataProvider: settingsProvider
    });

    // Register webview provider
    const dashboardPanel = vscode.window.registerWebviewViewProvider(
        'leanvibe.dashboard',
        dashboardProvider
    );

    // Initialize commands
    const commands = new HiveCommands(hiveManager, dashboardProvider, configManager);
    
    // Register commands
    const disposables = [
        vscode.commands.registerCommand('leanvibe.startHive', () => commands.startHive()),
        vscode.commands.registerCommand('leanvibe.stopHive', () => commands.stopHive()),
        vscode.commands.registerCommand('leanvibe.openDashboard', () => commands.openDashboard()),
        vscode.commands.registerCommand('leanvibe.createAgent', () => commands.createAgent()),
        vscode.commands.registerCommand('leanvibe.deployProject', () => commands.deployProject()),
        vscode.commands.registerCommand('leanvibe.runTests', () => commands.runTests()),
        vscode.commands.registerCommand('leanvibe.optimizeCode', (uri?: vscode.Uri) => commands.optimizeCode(uri)),
        vscode.commands.registerCommand('leanvibe.generateDocumentation', (uri?: vscode.Uri) => commands.generateDocumentation(uri)),
        
        // Tree view item commands
        vscode.commands.registerCommand('leanvibe.agent.start', (agentId: string) => commands.startAgent(agentId)),
        vscode.commands.registerCommand('leanvibe.agent.stop', (agentId: string) => commands.stopAgent(agentId)),
        vscode.commands.registerCommand('leanvibe.agent.restart', (agentId: string) => commands.restartAgent(agentId)),
        vscode.commands.registerCommand('leanvibe.task.cancel', (taskId: string) => commands.cancelTask(taskId)),
        vscode.commands.registerCommand('leanvibe.task.retry', (taskId: string) => commands.retryTask(taskId)),
    ];

    // Register code suggestion provider if enabled
    if (configManager.getConfig().enableCodeSuggestions) {
        const suggestionDisposable = vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**/*' },
            codeSuggestionProvider
        );
        disposables.push(suggestionDisposable);
    }

    // Set up event listeners
    hiveManager.onStatusChanged((status) => {
        statusBar.updateStatus(status);
        vscode.commands.executeCommand('setContext', 'leanvibe.connected', status.connected);
        
        // Refresh all tree views
        agentProvider.refresh();
        taskProvider.refresh();
        logsProvider.refresh();
    });

    // Configuration change listener
    vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration('leanvibe')) {
            configManager.reloadConfig();
            settingsProvider.refresh();
            
            // Restart code suggestions if setting changed
            if (e.affectsConfiguration('leanvibe.enableCodeSuggestions')) {
                const newConfig = configManager.getConfig();
                if (newConfig.enableCodeSuggestions && !codeSuggestionProvider.isEnabled()) {
                    const suggestionDisposable = vscode.languages.registerInlineCompletionItemProvider(
                        { pattern: '**/*' },
                        codeSuggestionProvider
                    );
                    disposables.push(suggestionDisposable);
                }
            }
        }
    });

    // Auto-start if enabled and compatible project detected
    if (configManager.getConfig().autoStart) {
        checkAndAutoStart(hiveManager);
    }

    // Store disposables for cleanup
    context.subscriptions.push(
        ...disposables,
        agentTreeView,
        taskTreeView,
        logsTreeView,
        settingsTreeView,
        dashboardPanel,
        statusBar,
        hiveManager
    );

    Logger.info('LeanVibe Agent Hive extension activated successfully');
}

export function deactivate() {
    Logger.info('Deactivating LeanVibe Agent Hive extension');
}

async function checkAndAutoStart(hiveManager: HiveManager) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return;

    // Check for LeanVibe project indicators
    for (const folder of workspaceFolders) {
        const indicators = [
            vscode.Uri.joinPath(folder.uri, '.leanvibe'),
            vscode.Uri.joinPath(folder.uri, 'pyproject.toml'),
            vscode.Uri.joinPath(folder.uri, 'docker-compose.yml'),
            vscode.Uri.joinPath(folder.uri, 'CLAUDE.md')
        ];

        for (const indicator of indicators) {
            try {
                await vscode.workspace.fs.stat(indicator);
                Logger.info(`Compatible project detected: ${indicator.fsPath}`);
                
                // Small delay to ensure VS Code is fully loaded
                setTimeout(() => {
                    hiveManager.start();
                }, 2000);
                return;
            } catch {
                // File doesn't exist, continue checking
            }
        }
    }
}