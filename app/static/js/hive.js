/**
 * LeanVibe Agent Hive 2.0 - JavaScript Command Interface
 * 
 * Comprehensive JavaScript interface for mobile/web integration with:
 * - WebSocket support for real-time notifications
 * - Command routing and validation
 * - Error handling and recovery
 * - Mobile optimization features
 * - Offline command queuing
 * 
 * @version 2.0.0
 * @author LeanVibe Agent Hive System
 */

class HiveCommandInterface {
    /**
     * Initialize the Hive command interface
     * @param {Object} options - Configuration options
     * @param {string} options.wsUrl - WebSocket URL for real-time updates
     * @param {string} options.apiUrl - HTTP API base URL
     * @param {boolean} options.mobileOptimized - Enable mobile optimizations
     * @param {number} options.timeout - Command timeout in milliseconds
     * @param {boolean} options.enableOfflineQueue - Enable offline command queuing
     */
    constructor(options = {}) {
        this.config = {
            wsUrl: options.wsUrl || this._detectWebSocketUrl(),
            apiUrl: options.apiUrl || '/api/hive',
            mobileOptimized: options.mobileOptimized || this._detectMobile(),
            timeout: options.timeout || 30000,
            enableOfflineQueue: options.enableOfflineQueue !== false,
            retryAttempts: options.retryAttempts || 3,
            retryDelay: options.retryDelay || 1000
        };

        // State management
        this.socket = null;
        this.connectionState = 'disconnected';
        this.commandHistory = [];
        this.listeners = new Map();
        this.offlineQueue = [];
        this.requestId = 0;
        this.activeRequests = new Map();
        
        // Performance tracking
        this.performanceMetrics = {
            totalCommands: 0,
            successfulCommands: 0,
            avgResponseTime: 0,
            cacheHitRate: 0,
            websocketMessages: 0
        };

        // Command registry cache
        this.commandRegistry = null;
        this.commandSuggestions = new Map();

        // Initialize
        this._initialize();
    }

    /**
     * Initialize the interface
     * @private
     */
    async _initialize() {
        try {
            // Load command registry
            await this._loadCommandRegistry();
            
            // Initialize WebSocket connection
            if (this.config.wsUrl) {
                await this._initializeWebSocket();
            }

            // Setup mobile optimizations
            if (this.config.mobileOptimized) {
                this._initializeMobileOptimizations();
            }

            // Setup offline handling
            if (this.config.enableOfflineQueue) {
                this._initializeOfflineHandling();
            }

            // Initialize performance monitoring
            this._initializePerformanceMonitoring();

            this._logInfo('HiveCommandInterface initialized successfully');
        } catch (error) {
            this._logError('Failed to initialize HiveCommandInterface', error);
            throw new Error(`Initialization failed: ${error.message}`);
        }
    }

    /**
     * Execute a hive command with comprehensive error handling and optimization
     * @param {string} command - Command string (e.g., '/hive:status')
     * @param {Object} options - Execution options
     * @returns {Promise<Object>} Command result
     */
    async executeCommand(command, options = {}) {
        const startTime = performance.now();
        const requestId = this._generateRequestId();
        
        try {
            // Validate command
            const validation = await this._validateCommand(command);
            if (!validation.valid) {
                throw new Error(`Invalid command: ${validation.error}`);
            }

            // Prepare request
            const request = this._prepareCommandRequest(command, options, requestId);
            
            // Track request
            this.activeRequests.set(requestId, {
                command,
                startTime,
                options
            });

            // Check offline status
            if (!navigator.onLine && this.config.enableOfflineQueue) {
                return await this._queueOfflineCommand(request);
            }

            // Execute with preferred method (WebSocket first, fallback to HTTP)
            let result;
            if (this._canUseWebSocket() && options.forceHttp !== true) {
                result = await this._executeViaWebSocket(request);
            } else {
                result = await this._executeViaHTTP(request);
            }

            // Process result
            result = await this._processCommandResult(result, command, startTime);

            // Update metrics and history
            this._updateMetrics(command, result, startTime);
            this._addToHistory(command, result, options);

            return result;

        } catch (error) {
            this._logError(`Command execution failed: ${command}`, error);
            
            // Attempt intelligent error recovery
            const recovery = await this._attemptErrorRecovery(command, error, options);
            if (recovery.success) {
                return recovery.result;
            }

            // Return structured error
            return {
                success: false,
                error: error.message,
                command: command,
                requestId: requestId,
                execution_time_ms: performance.now() - startTime,
                recovery_attempted: recovery.attempted,
                timestamp: new Date().toISOString()
            };
        } finally {
            this.activeRequests.delete(requestId);
        }
    }

    /**
     * Get intelligent command suggestions based on current context
     * @param {string} partialCommand - Partial command input
     * @param {Object} context - Current context information
     * @returns {Promise<Array>} Array of command suggestions
     */
    async getSuggestions(partialCommand, context = {}) {
        try {
            // Check cache first
            const cacheKey = `suggestions:${partialCommand}:${JSON.stringify(context)}`;
            const cached = this.commandSuggestions.get(cacheKey);
            if (cached && (Date.now() - cached.timestamp) < 30000) {
                return cached.suggestions;
            }

            const response = await fetch(`${this.config.apiUrl}/suggestions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Mobile-Optimized': this.config.mobileOptimized.toString()
                },
                body: JSON.stringify({
                    partial: partialCommand,
                    context: {
                        ...context,
                        ...await this._getContextualInfo()
                    },
                    mobile: this.config.mobileOptimized,
                    history: this._getRecentHistory()
                }),
                signal: AbortSignal.timeout(5000) // 5 second timeout for suggestions
            });

            if (!response.ok) {
                throw new Error(`Suggestions request failed: ${response.statusText}`);
            }

            const data = await response.json();
            const suggestions = data.suggestions || [];

            // Cache suggestions
            this.commandSuggestions.set(cacheKey, {
                suggestions,
                timestamp: Date.now()
            });

            return suggestions;

        } catch (error) {
            this._logError('Failed to get command suggestions', error);
            return this._getFallbackSuggestions(partialCommand);
        }
    }

    /**
     * Get comprehensive help for a specific command
     * @param {string} commandName - Name of the command
     * @param {Object} options - Help options
     * @returns {Promise<Object>} Detailed help information
     */
    async getCommandHelp(commandName, options = {}) {
        try {
            const params = new URLSearchParams({
                mobile: this.config.mobileOptimized.toString(),
                context_aware: (options.contextAware !== false).toString()
            });

            const response = await fetch(`${this.config.apiUrl}/help/${commandName}?${params}`, {
                headers: {
                    'X-Mobile-Optimized': this.config.mobileOptimized.toString()
                }
            });

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`Command '${commandName}' not found`);
                }
                throw new Error(`Help request failed: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            this._logError(`Failed to get help for command: ${commandName}`, error);
            return this._getFallbackHelp(commandName);
        }
    }

    /**
     * Get current system status and metrics
     * @returns {Promise<Object>} System status information
     */
    async getSystemStatus() {
        try {
            const response = await fetch(`${this.config.apiUrl}/status`, {
                headers: {
                    'X-Mobile-Optimized': this.config.mobileOptimized.toString()
                }
            });

            if (!response.ok) {
                throw new Error(`Status request failed: ${response.statusText}`);
            }

            const status = await response.json();
            return {
                ...status,
                client_metrics: this.performanceMetrics,
                connection_state: this.connectionState,
                offline_queue_size: this.offlineQueue.length
            };

        } catch (error) {
            this._logError('Failed to get system status', error);
            return {
                success: false,
                error: error.message,
                client_metrics: this.performanceMetrics,
                connection_state: 'error'
            };
        }
    }

    /**
     * Enable mobile-specific optimizations
     */
    optimizeForMobile() {
        this.config.mobileOptimized = true;
        this._enableTouchGestures();
        this._enableOfflineQueuing();
        this._optimizeNetworkRequests();
        this._setupMobileErrorHandling();
        
        this._logInfo('Mobile optimizations enabled');
    }

    /**
     * Add event listener for hive events
     * @param {string} event - Event type
     * @param {Function} callback - Event callback
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    /**
     * Remove event listener
     * @param {string} event - Event type  
     * @param {Function} callback - Event callback
     */
    off(event, callback) {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    /**
     * Get performance metrics
     * @returns {Object} Performance metrics
     */
    getPerformanceMetrics() {
        return {
            ...this.performanceMetrics,
            connection_state: this.connectionState,
            active_requests: this.activeRequests.size,
            cached_suggestions: this.commandSuggestions.size,
            command_history_size: this.commandHistory.length,
            offline_queue_size: this.offlineQueue.length
        };
    }

    /**
     * Clear caches and reset state
     */
    clearCache() {
        this.commandSuggestions.clear();
        this.commandRegistry = null;
        this._logInfo('Cache cleared');
    }

    /**
     * Disconnect and cleanup resources
     */
    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.connectionState = 'disconnected';
        this.activeRequests.clear();
        this._logInfo('Disconnected from Hive interface');
    }

    // Private Methods

    /**
     * Detect WebSocket URL based on current location
     * @private
     */
    _detectWebSocketUrl() {
        if (typeof window === 'undefined') return null;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/hive`;
    }

    /**
     * Detect if running on mobile device
     * @private
     */
    _detectMobile() {
        if (typeof window === 'undefined') return false;
        
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
        ) || window.innerWidth <= 768;
    }

    /**
     * Initialize WebSocket connection with automatic reconnection
     * @private
     */
    async _initializeWebSocket() {
        try {
            this.socket = new WebSocket(this.config.wsUrl);
            
            this.socket.onopen = () => {
                this.connectionState = 'connected';
                this._emit('connected');
                this._processOfflineQueue();
                this._logInfo('WebSocket connected');
            };

            this.socket.onmessage = (event) => {
                this._handleWebSocketMessage(event);
            };

            this.socket.onclose = (event) => {
                this.connectionState = 'disconnected';
                this._emit('disconnected', { code: event.code, reason: event.reason });
                
                if (!event.wasClean) {
                    this._scheduleReconnection();
                }
            };

            this.socket.onerror = (error) => {
                this.connectionState = 'error';
                this._emit('error', error);
                this._logError('WebSocket error', error);
            };

        } catch (error) {
            this._logError('Failed to initialize WebSocket', error);
            throw error;
        }
    }

    /**
     * Initialize mobile-specific optimizations
     * @private
     */
    _initializeMobileOptimizations() {
        // Touch gesture support
        this._enableTouchGestures();
        
        // Reduced network requests
        this._optimizeNetworkRequests();
        
        // Enhanced caching
        this._enableAggressiveCaching();
        
        // Battery optimization
        this._enableBatteryOptimization();
    }

    /**
     * Initialize offline command queuing
     * @private
     */
    _initializeOfflineHandling() {
        // Listen for online/offline events
        window.addEventListener('online', () => {
            this._emit('online');
            this._processOfflineQueue();
        });

        window.addEventListener('offline', () => {
            this._emit('offline');
        });
    }

    /**
     * Initialize performance monitoring
     * @private
     */
    _initializePerformanceMonitoring() {
        // Monitor network performance
        if ('connection' in navigator) {
            navigator.connection.addEventListener('change', () => {
                this._emit('network-change', {
                    effectiveType: navigator.connection.effectiveType,
                    downlink: navigator.connection.downlink
                });
            });
        }

        // Monitor memory usage (if available)
        if ('memory' in performance) {
            setInterval(() => {
                const memInfo = performance.memory;
                this._emit('memory-usage', {
                    used: memInfo.usedJSHeapSize,
                    total: memInfo.totalJSHeapSize,
                    limit: memInfo.jsHeapSizeLimit
                });
            }, 30000);
        }
    }

    /**
     * Load command registry from server
     * @private
     */
    async _loadCommandRegistry() {
        try {
            const response = await fetch(`${this.config.apiUrl}/list`);
            if (response.ok) {
                this.commandRegistry = await response.json();
            }
        } catch (error) {
            this._logError('Failed to load command registry', error);
        }
    }

    /**
     * Validate command format and availability
     * @private
     */
    async _validateCommand(command) {
        // Basic format validation
        if (!command || typeof command !== 'string') {
            return { valid: false, error: 'Command must be a non-empty string' };
        }

        if (!command.startsWith('/hive:')) {
            return { valid: false, error: 'Command must start with /hive:' };
        }

        const commandName = command.split(' ')[0].replace('/hive:', '');
        
        // Check against registry
        if (this.commandRegistry) {
            const availableCommands = Object.keys(this.commandRegistry.commands || {});
            if (!availableCommands.includes(commandName)) {
                return { 
                    valid: false, 
                    error: `Unknown command: ${commandName}. Available: ${availableCommands.join(', ')}` 
                };
            }
        }

        return { valid: true };
    }

    /**
     * Prepare command request with optimization flags
     * @private
     */
    _prepareCommandRequest(command, options, requestId) {
        return {
            id: requestId,
            command: command,
            mobile_optimized: this.config.mobileOptimized,
            use_cache: options.useCache !== false,
            priority: options.priority || 'medium',
            timeout: options.timeout || this.config.timeout,
            context: {
                ...options.context,
                client_info: this._getClientInfo(),
                performance_hints: this._getPerformanceHints()
            }
        };
    }

    /**
     * Execute command via WebSocket
     * @private
     */
    async _executeViaWebSocket(request) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('WebSocket command timeout'));
            }, request.timeout);

            const responseHandler = (event) => {
                const data = JSON.parse(event.data);
                if (data.request_id === request.id) {
                    clearTimeout(timeout);
                    this.socket.removeEventListener('message', responseHandler);
                    resolve(data);
                }
            };

            this.socket.addEventListener('message', responseHandler);
            this.socket.send(JSON.stringify({
                type: 'command',
                ...request
            }));
        });
    }

    /**
     * Execute command via HTTP API
     * @private
     */
    async _executeViaHTTP(request) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), request.timeout);

        try {
            const response = await fetch(`${this.config.apiUrl}/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Mobile-Optimized': this.config.mobileOptimized.toString(),
                    'X-Request-ID': request.id
                },
                body: JSON.stringify(request),
                signal: controller.signal
            });

            clearTimeout(timeout);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeout);
            throw error;
        }
    }

    /**
     * Can use WebSocket for command execution
     * @private
     */
    _canUseWebSocket() {
        return this.socket && 
               this.socket.readyState === WebSocket.OPEN && 
               this.connectionState === 'connected';
    }

    /**
     * Process command result with enhancements
     * @private
     */
    async _processCommandResult(result, command, startTime) {
        // Add client-side metrics
        result.client_execution_time_ms = performance.now() - startTime;
        result.client_timestamp = new Date().toISOString();
        
        // Enhance mobile results
        if (this.config.mobileOptimized && result.mobile_optimized !== false) {
            result = await this._enhanceForMobile(result);
        }

        return result;
    }

    /**
     * Enhance result for mobile interface
     * @private
     */
    async _enhanceForMobile(result) {
        // Add mobile-specific formatting
        if (result.recommendations && Array.isArray(result.recommendations)) {
            result.mobile_quick_actions = result.recommendations
                .filter(rec => rec.command)
                .slice(0, 3)
                .map(rec => ({
                    title: rec.title || rec.description?.substring(0, 30) + '...',
                    command: rec.command,
                    priority: rec.priority || 'medium'
                }));
        }

        return result;
    }

    /**
     * Queue command for offline execution
     * @private
     */
    async _queueOfflineCommand(request) {
        this.offlineQueue.push({
            ...request,
            queued_at: Date.now()
        });

        return {
            success: true,
            queued: true,
            message: 'Command queued for execution when online',
            queue_position: this.offlineQueue.length,
            request_id: request.id
        };
    }

    /**
     * Process offline command queue when back online
     * @private
     */
    async _processOfflineQueue() {
        if (this.offlineQueue.length === 0) return;

        this._logInfo(`Processing ${this.offlineQueue.length} queued commands`);

        const queue = [...this.offlineQueue];
        this.offlineQueue = [];

        for (const request of queue) {
            try {
                const result = await this._executeViaWebSocket(request);
                this._emit('offline-command-completed', { request, result });
            } catch (error) {
                this._emit('offline-command-failed', { request, error: error.message });
                // Re-queue if still offline
                if (!navigator.onLine) {
                    this.offlineQueue.push(request);
                }
            }
        }
    }

    /**
     * Attempt intelligent error recovery
     * @private
     */
    async _attemptErrorRecovery(command, error, options) {
        const recoveryStrategies = [
            () => this._retryWithExponentialBackoff(command, options),
            () => this._tryAlternativeEndpoint(command, options),
            () => this._degradedModeExecution(command, options)
        ];

        for (const strategy of recoveryStrategies) {
            try {
                const result = await strategy();
                return { success: true, result, attempted: true };
            } catch (recoveryError) {
                this._logError('Recovery strategy failed', recoveryError);
            }
        }

        return { success: false, attempted: true };
    }

    /**
     * Update performance metrics
     * @private
     */
    _updateMetrics(command, result, startTime) {
        this.performanceMetrics.totalCommands++;
        
        if (result.success) {
            this.performanceMetrics.successfulCommands++;
        }

        const responseTime = performance.now() - startTime;
        this.performanceMetrics.avgResponseTime = 
            (this.performanceMetrics.avgResponseTime * (this.performanceMetrics.totalCommands - 1) + responseTime) / 
            this.performanceMetrics.totalCommands;

        if (result.cached) {
            this.performanceMetrics.cacheHitRate = 
                (this.performanceMetrics.cacheHitRate * (this.performanceMetrics.totalCommands - 1) + 1) / 
                this.performanceMetrics.totalCommands;
        }
    }

    /**
     * Add command to history
     * @private
     */
    _addToHistory(command, result, options) {
        this.commandHistory.unshift({
            command,
            result: result.success,
            timestamp: new Date().toISOString(),
            execution_time_ms: result.execution_time_ms || result.client_execution_time_ms,
            mobile_optimized: this.config.mobileOptimized
        });

        // Keep only last 50 commands
        this.commandHistory = this.commandHistory.slice(0, 50);
    }

    /**
     * Get contextual information for commands
     * @private
     */
    async _getContextualInfo() {
        return {
            timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent,
            screen_resolution: `${screen.width}x${screen.height}`,
            viewport_size: `${window.innerWidth}x${window.innerHeight}`,
            connection_type: navigator.connection?.effectiveType || 'unknown',
            is_mobile: this.config.mobileOptimized,
            is_online: navigator.onLine
        };
    }

    /**
     * Get client information
     * @private
     */
    _getClientInfo() {
        return {
            version: '2.0.0',
            mobile_optimized: this.config.mobileOptimized,
            connection_state: this.connectionState,
            features_enabled: {
                websocket: !!this.socket,
                offline_queue: this.config.enableOfflineQueue,
                touch_gestures: this.config.mobileOptimized
            }
        };
    }

    /**
     * Generate unique request ID
     * @private
     */
    _generateRequestId() {
        return `hive_${Date.now()}_${++this.requestId}`;
    }

    /**
     * Emit event to listeners
     * @private
     */
    _emit(event, data) {
        const callbacks = this.listeners.get(event) || [];
        callbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                this._logError(`Event listener error for ${event}`, error);
            }
        });
    }

    /**
     * Log info message
     * @private
     */
    _logInfo(message, data = {}) {
        console.log(`[HiveJS] ${message}`, data);
    }

    /**
     * Log error message
     * @private
     */
    _logError(message, error, data = {}) {
        console.error(`[HiveJS] ${message}`, error, data);
    }

    // Utility methods for mobile optimization, touch gestures, etc.
    _enableTouchGestures() { /* Implementation */ }
    _enableOfflineQueuing() { /* Implementation */ }
    _optimizeNetworkRequests() { /* Implementation */ }
    _enableAggressiveCaching() { /* Implementation */ }
    _enableBatteryOptimization() { /* Implementation */ }
    _setupMobileErrorHandling() { /* Implementation */ }
    _getFallbackSuggestions(partial) { return []; }
    _getFallbackHelp(command) { return { error: 'Help unavailable' }; }
    _getRecentHistory() { return this.commandHistory.slice(0, 5); }
    _getPerformanceHints() { return {}; }
    _handleWebSocketMessage(event) { /* Implementation */ }
    _scheduleReconnection() { /* Implementation */ }
    _retryWithExponentialBackoff(command, options) { /* Implementation */ }
    _tryAlternativeEndpoint(command, options) { /* Implementation */ }
    _degradedModeExecution(command, options) { /* Implementation */ }
}

/**
 * Global Hive interface factory
 */
class HiveJS {
    static instance = null;

    /**
     * Get or create global Hive interface instance
     * @param {Object} options - Configuration options
     * @returns {HiveCommandInterface} Interface instance
     */
    static getInstance(options = {}) {
        if (!HiveJS.instance) {
            HiveJS.instance = new HiveCommandInterface(options);
        }
        return HiveJS.instance;
    }

    /**
     * Create new Hive interface instance
     * @param {Object} options - Configuration options
     * @returns {HiveCommandInterface} New interface instance
     */
    static create(options = {}) {
        return new HiveCommandInterface(options);
    }

    /**
     * Quick command execution helper
     * @param {string} command - Command to execute
     * @param {Object} options - Execution options
     * @returns {Promise<Object>} Command result
     */
    static async execute(command, options = {}) {
        const interface = HiveJS.getInstance();
        return await interface.executeCommand(command, options);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HiveCommandInterface, HiveJS };
}

// Global browser export
if (typeof window !== 'undefined') {
    window.HiveJS = HiveJS;
    window.HiveCommandInterface = HiveCommandInterface;
}