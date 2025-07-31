#!/usr/bin/env python3
"""
Standalone Demo Server for LeanVibe Agent Hive 2.0
Minimal server to run the browser demo independently
"""

import os
import sys
from pathlib import Path

# Add demo directory to Python path
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

try:
    from fastapi import FastAPI, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("❌ Missing dependencies. Please install:")
    print("   pip install fastapi uvicorn")
    sys.exit(1)

# Import demo endpoints
try:
    from api.demo_endpoint import demo_router
except ImportError as e:
    print(f"❌ Failed to import demo endpoints: {e}")
    print("   Make sure all demo files are present")
    sys.exit(1)

# Create FastAPI application
app = FastAPI(
    title="LeanVibe Agent Hive 2.0 - Browser Demo",
    description="Autonomous AI Development Demo Platform",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include demo API routes
app.include_router(demo_router)

# Serve static files
static_dir = demo_dir / "assets"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

# Serve demo files
@app.get("/", response_class=HTMLResponse)
async def serve_demo():
    """Serve the main demo page."""
    html_file = demo_dir / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse(
            content="<h1>Demo files not found</h1><p>Please ensure index.html exists in the demo directory.</p>",
            status_code=404
        )

@app.get("/manifest.json")
async def serve_manifest():
    """Serve PWA manifest."""
    manifest_file = demo_dir / "manifest.json"
    if manifest_file.exists():
        return FileResponse(manifest_file)
    else:
        # Return basic manifest
        return {
            "name": "LeanVibe Agent Hive 2.0 Demo",
            "short_name": "LeanVibe Demo",
            "description": "Autonomous AI Development Demo",
            "start_url": "/",
            "display": "standalone",
            "theme_color": "#0066cc",
            "background_color": "#ffffff",
            "icons": []
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LeanVibe Demo Server",
        "version": "2.0.0"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return HTMLResponse(
        content=f"""
        <html>
            <head><title>Page Not Found</title></head>
            <body>
                <h1>404 - Page Not Found</h1>
                <p>The requested path <code>{request.url.path}</code> was not found.</p>
                <p><a href="/">← Return to Demo</a></p>
            </body>
        </html>
        """,
        status_code=404
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle server errors."""
    return HTMLResponse(
        content="""
        <html>
            <head><title>Server Error</title></head>
            <body>
                <h1>500 - Server Error</h1>
                <p>An internal server error occurred.</p>
                <p><a href="/">← Return to Demo</a></p>
            </body>
        </html>
        """,
        status_code=500
    )

def main():
    """Main entry point for the demo server."""
    print("🚀 Starting LeanVibe Agent Hive 2.0 Browser Demo")
    print("=" * 60)
    
    # Get configuration from environment
    host = os.getenv("DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("DEMO_PORT", "8080"))
    reload = os.getenv("DEMO_RELOAD", "true").lower() == "true"
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("✅ Anthropic API key configured - AI generation enabled")
    else:
        print("⚠️  No Anthropic API key found - using fallback templates")
        print("   Set ANTHROPIC_API_KEY environment variable for full AI features")
    
    print(f"\n🌐 Demo will be available at:")
    print(f"   http://{host}:{port}")
    
    if host == "127.0.0.1":
        print(f"   http://localhost:{port}")
    
    print(f"\n📚 API Documentation:")
    print(f"   http://{host}:{port}/api/docs")
    
    print(f"\n🔧 Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   Demo Directory: {demo_dir}")
    
    # Validate demo files
    required_files = ["index.html", "assets/styles.css", "assets/demo.js"]
    missing_files = []
    
    for file_path in required_files:
        full_path = demo_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n   Please ensure all demo files are present.")
        return False
    
    print("\n✅ All required files found")
    print("\n" + "=" * 60)
    print("🎯 Ready to demonstrate autonomous AI development!")
    print("   Open your browser to the URL above to start the demo")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            access_log=True,
            log_level="info"
        )
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Failed to start demo server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)