import * as vscode from 'vscode';
import { EventEmitter } from 'events';

export interface HiveConfig {
    apiUrl: string;
    apiKey: string;
    autoStart: boolean;
    maxAgents: number;
    logLevel: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
    enableCodeSuggestions: boolean;
    enableAutonomousMode: boolean;
}

export class ConfigurationManager extends EventEmitter {
    private config: HiveConfig;

    constructor() {
        super();
        this.config = this.loadConfig();
        
        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('leanvibe')) {
                const newConfig = this.loadConfig();
                this.config = newConfig;
                this.emit('configChanged', newConfig);
            }
        });
    }

    private loadConfig(): HiveConfig {
        const config = vscode.workspace.getConfiguration('leanvibe');
        
        return {
            apiUrl: config.get('apiUrl', 'http://localhost:8000'),
            apiKey: config.get('apiKey', ''),
            autoStart: config.get('autoStart', false),
            maxAgents: config.get('maxAgents', 5),
            logLevel: config.get('logLevel', 'INFO') as 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR',
            enableCodeSuggestions: config.get('enableCodeSuggestions', true),
            enableAutonomousMode: config.get('enableAutonomousMode', false)
        };
    }

    getConfig(): HiveConfig {
        return this.config;
    }

    reloadConfig(): void {
        this.config = this.loadConfig();
        this.emit('configChanged', this.config);
    }

    async updateConfig(key: keyof HiveConfig, value: any): Promise<void> {
        const config = vscode.workspace.getConfiguration('leanvibe');
        await config.update(key, value, vscode.ConfigurationTarget.Global);
    }
}