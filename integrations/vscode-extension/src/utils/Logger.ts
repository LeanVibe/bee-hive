import * as vscode from 'vscode';

export class Logger {
    private static outputChannel: vscode.OutputChannel;

    static initialize(): void {
        this.outputChannel = vscode.window.createOutputChannel('LeanVibe Agent Hive');
    }

    static debug(message: string, ...args: any[]): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] DEBUG: ${message}`;
        
        if (args.length > 0) {
            console.debug(logMessage, ...args);
        } else {
            console.debug(logMessage);
        }
        
        this.outputChannel?.appendLine(logMessage + (args.length > 0 ? ' ' + JSON.stringify(args) : ''));
    }

    static info(message: string, ...args: any[]): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] INFO: ${message}`;
        
        if (args.length > 0) {
            console.info(logMessage, ...args);
        } else {
            console.info(logMessage);
        }
        
        this.outputChannel?.appendLine(logMessage + (args.length > 0 ? ' ' + JSON.stringify(args) : ''));
    }

    static warn(message: string, ...args: any[]): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] WARN: ${message}`;
        
        if (args.length > 0) {
            console.warn(logMessage, ...args);
        } else {
            console.warn(logMessage);
        }
        
        this.outputChannel?.appendLine(logMessage + (args.length > 0 ? ' ' + JSON.stringify(args) : ''));
    }

    static error(message: string, error?: any): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] ERROR: ${message}`;
        
        if (error) {
            console.error(logMessage, error);
            this.outputChannel?.appendLine(logMessage + ' ' + (error.stack || error.toString()));
        } else {
            console.error(logMessage);
            this.outputChannel?.appendLine(logMessage);
        }
    }

    static show(): void {
        this.outputChannel?.show();
    }
}

// Initialize logger when module is loaded
Logger.initialize();