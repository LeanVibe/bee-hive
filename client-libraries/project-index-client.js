/**
 * Universal Project Index Client Library - JavaScript/Node.js
 * Provides easy integration with Project Index API for any project
 */

class ProjectIndexClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || 'http://localhost:8100';
        this.apiKey = options.apiKey || null;
        this.timeout = options.timeout || 30000;
        this.retries = options.retries || 3;
        this.debug = options.debug || false;
        
        // WebSocket connection for real-time updates
        this.websocket = null;
        this.subscribers = new Map();
        
        this.log('Initialized Project Index client', { baseUrl: this.baseUrl });
    }

    log(message, data = null) {
        if (this.debug) {
            console.log(`[ProjectIndexClient] ${message}`, data || '');
        }
    }

    /**
     * Make HTTP request with retry logic
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}/api${endpoint}`;
        const config = {
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'ProjectIndexClient/1.0.0',
                ...options.headers
            },
            ...options
        };

        if (this.apiKey) {
            config.headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                this.log(`Request attempt ${attempt}`, { url, method: config.method || 'GET' });
                
                const response = await this.fetch(url, config);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.log('Request successful', { status: response.status });
                return data;
                
            } catch (error) {
                this.log(`Request attempt ${attempt} failed`, { error: error.message });
                
                if (attempt === this.retries) {
                    throw new Error(`Request failed after ${this.retries} attempts: ${error.message}`);
                }
                
                // Exponential backoff
                await this.sleep(Math.pow(2, attempt) * 1000);
            }
        }
    }

    /**
     * Fetch implementation (works in both browser and Node.js)
     */
    async fetch(url, options) {
        if (typeof window !== 'undefined' && window.fetch) {
            // Browser environment
            return window.fetch(url, options);
        } else if (typeof global !== 'undefined') {
            // Node.js environment
            const { default: fetch } = await import('node-fetch');
            return fetch(url, options);
        } else {
            throw new Error('No fetch implementation available');
        }
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // =============================================================================
    // PROJECT MANAGEMENT
    // =============================================================================

    /**
     * Get current project information
     */
    async getProject() {
        return this.request('/projects');
    }

    /**
     * Create or update project configuration
     */
    async createProject(projectData) {
        return this.request('/projects', {
            method: 'POST',
            body: JSON.stringify(projectData)
        });
    }

    /**
     * Update project configuration
     */
    async updateProject(projectId, updateData) {
        return this.request(`/projects/${projectId}`, {
            method: 'PATCH',
            body: JSON.stringify(updateData)
        });
    }

    /**
     * Get project statistics
     */
    async getProjectStats() {
        return this.request('/projects/stats');
    }

    // =============================================================================
    // ANALYSIS OPERATIONS
    // =============================================================================

    /**
     * Trigger project analysis
     */
    async analyzeProject(options = {}) {
        const analysisOptions = {
            analysis_type: options.type || 'smart',
            force: options.force || false,
            include_dependencies: options.includeDependencies !== false,
            include_complexity: options.includeComplexity !== false,
            max_depth: options.maxDepth || 3,
            ...options
        };

        return this.request('/projects/analyze', {
            method: 'POST',
            body: JSON.stringify(analysisOptions)
        });
    }

    /**
     * Get analysis status
     */
    async getAnalysisStatus(analysisId) {
        return this.request(`/analysis/${analysisId}/status`);
    }

    /**
     * Get analysis results
     */
    async getAnalysisResults(analysisId) {
        return this.request(`/analysis/${analysisId}/results`);
    }

    /**
     * Get latest analysis results
     */
    async getLatestAnalysis() {
        return this.request('/analysis/latest');
    }

    /**
     * Cancel running analysis
     */
    async cancelAnalysis(analysisId) {
        return this.request(`/analysis/${analysisId}/cancel`, {
            method: 'POST'
        });
    }

    // =============================================================================
    // FILE OPERATIONS
    // =============================================================================

    /**
     * Get file analysis
     */
    async getFileAnalysis(filePath) {
        return this.request(`/files/analyze?path=${encodeURIComponent(filePath)}`);
    }

    /**
     * Get file dependencies
     */
    async getFileDependencies(filePath) {
        return this.request(`/files/dependencies?path=${encodeURIComponent(filePath)}`);
    }

    /**
     * Get file history
     */
    async getFileHistory(filePath, options = {}) {
        const params = new URLSearchParams({
            path: filePath,
            limit: options.limit || 10,
            ...options
        });
        
        return this.request(`/files/history?${params}`);
    }

    /**
     * Search files by content
     */
    async searchFiles(query, options = {}) {
        const searchOptions = {
            query,
            language: options.language,
            file_type: options.fileType,
            include_content: options.includeContent || false,
            limit: options.limit || 50,
            ...options
        };

        return this.request('/files/search', {
            method: 'POST',
            body: JSON.stringify(searchOptions)
        });
    }

    // =============================================================================
    // DEPENDENCY ANALYSIS
    // =============================================================================

    /**
     * Get dependency graph
     */
    async getDependencyGraph(options = {}) {
        const params = new URLSearchParams({
            format: options.format || 'json',
            include_external: options.includeExternal || false,
            max_depth: options.maxDepth || 5,
            ...options
        });

        return this.request(`/dependencies/graph?${params}`);
    }

    /**
     * Get dependency cycles
     */
    async getDependencyCycles() {
        return this.request('/dependencies/cycles');
    }

    /**
     * Get external dependencies
     */
    async getExternalDependencies() {
        return this.request('/dependencies/external');
    }

    /**
     * Get dependency impact analysis
     */
    async getDependencyImpact(filePath) {
        return this.request(`/dependencies/impact?path=${encodeURIComponent(filePath)}`);
    }

    // =============================================================================
    // REAL-TIME MONITORING
    // =============================================================================

    /**
     * Subscribe to real-time updates
     */
    connectWebSocket(options = {}) {
        if (this.websocket) {
            this.websocket.close();
        }

        const wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/api/ws';
        const url = this.apiKey ? `${wsUrl}?token=${this.apiKey}` : wsUrl;

        this.log('Connecting to WebSocket', { url });

        if (typeof WebSocket !== 'undefined') {
            this.websocket = new WebSocket(url);
        } else if (typeof global !== 'undefined') {
            // Node.js environment
            const WebSocket = require('ws');
            this.websocket = new WebSocket(url);
        } else {
            throw new Error('WebSocket not available in this environment');
        }

        this.websocket.onopen = () => {
            this.log('WebSocket connected');
            
            // Subscribe to events
            if (options.events) {
                this.subscribeToEvents(options.events, options.filters);
            }
        };

        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                this.log('Failed to parse WebSocket message', { error: error.message });
            }
        };

        this.websocket.onclose = (event) => {
            this.log('WebSocket disconnected', { code: event.code, reason: event.reason });
            
            // Auto-reconnect if not intentionally closed
            if (event.code !== 1000 && options.autoReconnect !== false) {
                setTimeout(() => {
                    this.log('Attempting to reconnect...');
                    this.connectWebSocket(options);
                }, 5000);
            }
        };

        this.websocket.onerror = (error) => {
            this.log('WebSocket error', { error });
        };

        return this.websocket;
    }

    /**
     * Subscribe to specific event types
     */
    subscribeToEvents(eventTypes, filters = {}) {
        if (!this.websocket || this.websocket.readyState !== 1) {
            throw new Error('WebSocket not connected');
        }

        const subscription = {
            action: 'subscribe',
            event_types: Array.isArray(eventTypes) ? eventTypes : [eventTypes],
            filters
        };

        this.websocket.send(JSON.stringify(subscription));
        this.log('Subscribed to events', subscription);
    }

    /**
     * Add event listener for specific event types
     */
    on(eventType, callback) {
        if (!this.subscribers.has(eventType)) {
            this.subscribers.set(eventType, new Set());
        }
        this.subscribers.get(eventType).add(callback);
    }

    /**
     * Remove event listener
     */
    off(eventType, callback) {
        if (this.subscribers.has(eventType)) {
            this.subscribers.get(eventType).delete(callback);
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(data) {
        this.log('WebSocket message received', { type: data.type });

        // Emit to specific event type subscribers
        if (this.subscribers.has(data.type)) {
            this.subscribers.get(data.type).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.log('Error in event callback', { error: error.message });
                }
            });
        }

        // Emit to wildcard subscribers
        if (this.subscribers.has('*')) {
            this.subscribers.get('*').forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.log('Error in wildcard callback', { error: error.message });
                }
            });
        }
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnect');
            this.websocket = null;
        }
        this.subscribers.clear();
        this.log('Disconnected and cleaned up');
    }

    // =============================================================================
    // UTILITY METHODS
    // =============================================================================

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await this.request('/health');
            return { healthy: true, ...response };
        } catch (error) {
            return { healthy: false, error: error.message };
        }
    }

    /**
     * Get API information
     */
    async getApiInfo() {
        return this.request('/info');
    }

    /**
     * Export project data
     */
    async exportProject(format = 'json') {
        return this.request(`/projects/export?format=${format}`);
    }

    /**
     * Get configuration
     */
    async getConfiguration() {
        return this.request('/config');
    }

    /**
     * Update configuration
     */
    async updateConfiguration(config) {
        return this.request('/config', {
            method: 'PATCH',
            body: JSON.stringify(config)
        });
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Quick project analysis with sensible defaults
 */
async function analyzeProject(projectPath = process.cwd(), options = {}) {
    const client = new ProjectIndexClient(options);
    
    try {
        // Create project if it doesn't exist
        await client.createProject({
            name: require('path').basename(projectPath),
            root_path: projectPath,
            description: `Auto-analyzed project at ${projectPath}`
        });
        
        // Trigger analysis
        const analysis = await client.analyzeProject({
            type: 'smart',
            includeDependencies: true,
            includeComplexity: true
        });
        
        // Wait for completion and return results
        const results = await client.getAnalysisResults(analysis.analysis_id);
        return results;
        
    } catch (error) {
        throw new Error(`Project analysis failed: ${error.message}`);
    }
}

/**
 * Monitor project for real-time changes
 */
function monitorProject(callback, options = {}) {
    const client = new ProjectIndexClient(options);
    
    // Connect to WebSocket
    client.connectWebSocket({
        events: ['file_change', 'analysis_progress', 'dependency_changed'],
        autoReconnect: true,
        ...options
    });
    
    // Subscribe to all events
    client.on('*', callback);
    
    // Return cleanup function
    return () => client.disconnect();
}

// =============================================================================
// EXPORTS
// =============================================================================

// Support both CommonJS and ES modules
if (typeof module !== 'undefined' && module.exports) {
    // Node.js CommonJS
    module.exports = {
        ProjectIndexClient,
        analyzeProject,
        monitorProject
    };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.ProjectIndexClient = ProjectIndexClient;
    window.analyzeProject = analyzeProject;
    window.monitorProject = monitorProject;
} else {
    // ES modules
    export {
        ProjectIndexClient,
        analyzeProject,
        monitorProject
    };
}

// =============================================================================
// USAGE EXAMPLES
// =============================================================================

/*
// Basic usage
const client = new ProjectIndexClient({
    baseUrl: 'http://localhost:8100',
    debug: true
});

// Get project information
const project = await client.getProject();
console.log('Project:', project.name);

// Trigger analysis
const analysis = await client.analyzeProject({
    type: 'full',
    includeDependencies: true
});
console.log('Analysis started:', analysis.analysis_id);

// Get dependency graph
const dependencies = await client.getDependencyGraph();
console.log('Dependencies:', dependencies.nodes.length);

// Real-time monitoring
client.connectWebSocket({ events: ['file_change', 'analysis_progress'] });
client.on('file_change', (event) => {
    console.log('File changed:', event.file_path);
});

// Quick analysis
const results = await analyzeProject('/path/to/project');
console.log('Analysis complete:', results.files_analyzed);

// Monitor with callback
const stopMonitoring = monitorProject((event) => {
    console.log('Project event:', event.type, event.data);
});

// Stop monitoring later
stopMonitoring();
*/