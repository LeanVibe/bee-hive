/**
 * Lightweight Syntax Highlighter for Demo
 * Provides basic Python syntax highlighting without external dependencies
 */

class SyntaxHighlighter {
    constructor() {
        this.pythonKeywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
            'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'with',
            'yield', 'True', 'False', 'None', 'async', 'await'
        ];
        
        this.pythonBuiltins = [
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'callable', 'chr',
            'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod',
            'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset',
            'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input',
            'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals',
            'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
            'ord', 'pow', 'property', 'range', 'repr', 'reversed', 'round', 'set',
            'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
            'tuple', 'type', 'vars', 'zip', '__import__', 'Exception', 'ValueError',
            'TypeError', 'AttributeError', 'IndexError', 'KeyError'
        ];
    }

    /**
     * Highlight Python code
     * @param {string} code - The code to highlight
     * @returns {string} - HTML with syntax highlighting
     */
    highlightPython(code) {
        if (!code) return '';
        
        // Escape HTML entities first
        code = this.escapeHtml(code);
        
        // Apply syntax highlighting
        code = this.highlightComments(code);
        code = this.highlightStrings(code);
        code = this.highlightNumbers(code);
        code = this.highlightKeywords(code);
        code = this.highlightBuiltins(code);
        code = this.highlightFunctions(code);
        code = this.highlightDecorators(code);
        
        return code;
    }

    /**
     * Escape HTML entities
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Highlight comments
     */
    highlightComments(code) {
        // Single line comments
        code = code.replace(/(^|\n)([ \t]*#.*)$/gm, '$1<span class="syntax-comment">$2</span>');
        
        // Multi-line strings used as comments (docstrings)
        code = code.replace(/("""[\s\S]*?""")/g, '<span class="syntax-docstring">$1</span>');
        code = code.replace(/('''[\s\S]*?''')/g, '<span class="syntax-docstring">$1</span>');
        
        return code;
    }

    /**
     * Highlight strings
     */
    highlightStrings(code) {
        // Triple quoted strings (but not docstrings already highlighted)
        code = code.replace(/(?<!<span class="syntax-docstring">)("""[\s\S]*?""")/g, '<span class="syntax-string">$1</span>');
        code = code.replace(/(?<!<span class="syntax-docstring">)('''[\s\S]*?''')/g, '<span class="syntax-string">$1</span>');
        
        // Double quoted strings
        code = code.replace(/"([^"\\]|\\.)*"/g, '<span class="syntax-string">$&</span>');
        
        // Single quoted strings
        code = code.replace(/'([^'\\]|\\.)*'/g, '<span class="syntax-string">$&</span>');
        
        // f-strings
        code = code.replace(/f"([^"\\]|\\.)*"/g, '<span class="syntax-fstring">$&</span>');
        code = code.replace(/f'([^'\\]|\\.)*'/g, '<span class="syntax-fstring">$&</span>');
        
        return code;
    }

    /**
     * Highlight numbers
     */
    highlightNumbers(code) {
        // Integers and floats
        code = code.replace(/\b\d+\.?\d*\b/g, '<span class="syntax-number">$&</span>');
        
        // Hex numbers
        code = code.replace(/\b0x[0-9a-fA-F]+\b/g, '<span class="syntax-number">$&</span>');
        
        // Binary numbers
        code = code.replace(/\b0b[01]+\b/g, '<span class="syntax-number">$&</span>');
        
        return code;
    }

    /**
     * Highlight keywords
     */
    highlightKeywords(code) {
        const keywordPattern = new RegExp(`\\b(${this.pythonKeywords.join('|')})\\b`, 'g');
        return code.replace(keywordPattern, '<span class="syntax-keyword">$1</span>');
    }

    /**
     * Highlight built-in functions
     */
    highlightBuiltins(code) {
        const builtinPattern = new RegExp(`\\b(${this.pythonBuiltins.join('|')})\\b`, 'g');
        return code.replace(builtinPattern, '<span class="syntax-builtin">$1</span>');
    }

    /**
     * Highlight function definitions and calls
     */
    highlightFunctions(code) {
        // Function definitions
        code = code.replace(/def\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, 'def <span class="syntax-function-def">$1</span>');
        
        // Class definitions
        code = code.replace(/class\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, 'class <span class="syntax-class-def">$1</span>');
        
        // Function calls (basic pattern)
        code = code.replace(/([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span class="syntax-function-call">$1</span>');
        
        return code;
    }

    /**
     * Highlight decorators
     */
    highlightDecorators(code) {
        return code.replace(/@[a-zA-Z_][a-zA-Z0-9_.]*/g, '<span class="syntax-decorator">$&</span>');
    }

    /**
     * Add line numbers to code
     */
    addLineNumbers(code) {
        const lines = code.split('\n');
        const numberedLines = lines.map((line, index) => {
            const lineNumber = index + 1;
            return `<span class="line-number">${lineNumber.toString().padStart(3)}</span>${line}`;
        });
        return numberedLines.join('\n');
    }

    /**
     * Highlight markdown
     */
    highlightMarkdown(markdown) {
        if (!markdown) return '';
        
        // Escape HTML first
        let html = this.escapeHtml(markdown);
        
        // Headers
        html = html.replace(/^(#{1,6})\s+(.+)$/gm, (match, hashes, text) => {
            const level = hashes.length;
            return `<h${level} class="md-header">${text}</h${level}>`;
        });
        
        // Bold text
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="md-bold">$1</strong>');
        html = html.replace(/__(.*?)__/g, '<strong class="md-bold">$1</strong>');
        
        // Italic text
        html = html.replace(/\*(.*?)\*/g, '<em class="md-italic">$1</em>');
        html = html.replace(/_(.*?)_/g, '<em class="md-italic">$1</em>');
        
        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const highlightedCode = lang === 'python' ? this.highlightPython(code) : this.escapeHtml(code);
            return `<pre class="md-code-block"><code class="language-${lang || 'text'}">${highlightedCode}</code></pre>`;
        });
        
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');
        
        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="md-link">$1</a>');
        
        // Lists
        html = html.replace(/^[\s]*[-*+]\s+(.+)$/gm, '<li class="md-list-item">$1</li>');
        html = html.replace(/(<li class="md-list-item">.*<\/li>)/gs, '<ul class="md-list">$1</ul>');
        
        // Paragraphs
        html = html.replace(/^(?!<[hul]|```)(.*?)$/gm, '<p class="md-paragraph">$1</p>');
        
        return html;
    }
}

// Global instance
window.syntaxHighlighter = new SyntaxHighlighter();

// Add CSS for syntax highlighting
const syntaxCSS = `
<style id="syntax-highlighting-styles">
.syntax-keyword { color: #ff79c6; font-weight: bold; }
.syntax-builtin { color: #50fa7b; }
.syntax-string { color: #f1fa8c; }
.syntax-fstring { color: #f1fa8c; font-weight: bold; }
.syntax-docstring { color: #6272a4; font-style: italic; }
.syntax-comment { color: #6272a4; font-style: italic; }
.syntax-number { color: #bd93f9; }
.syntax-function-def { color: #50fa7b; font-weight: bold; }
.syntax-class-def { color: #8be9fd; font-weight: bold; }
.syntax-function-call { color: #50fa7b; }
.syntax-decorator { color: #ff79c6; }

.line-number {
    color: #6272a4;
    margin-right: 1rem;
    user-select: none;
    display: inline-block;
    text-align: right;
    width: 3em;
}

.md-header { color: #f8f8f2; margin: 1rem 0 0.5rem 0; }
.md-bold { color: #ff79c6; }
.md-italic { color: #50fa7b; }
.md-code-block { 
    background: #282a36; 
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin: 1rem 0;
    overflow-x: auto;
}
.md-inline-code { 
    background: rgba(255, 255, 255, 0.1); 
    padding: 0.2rem 0.4rem; 
    border-radius: 0.25rem; 
    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
}
.md-link { color: #8be9fd; }
.md-list { margin: 0.5rem 0; padding-left: 1.5rem; }
.md-list-item { margin: 0.25rem 0; }
.md-paragraph { margin: 0.5rem 0; line-height: 1.6; }
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', syntaxCSS);