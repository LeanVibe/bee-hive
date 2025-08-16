"""
Other Language Framework Integration Templates for Project Index

Provides code generation and integration patterns for:
- Go: HTTP handler integration, middleware (Gin, Echo, standard library)
- Rust: Axum/Rocket integration patterns, middleware
- Java: Spring Boot integration, annotation-based configuration

Note: These generate code templates that integrate with the Project Index API
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, List

from . import BaseFrameworkAdapter, IntegrationManager
from ..project_index import ProjectIndexConfig


class GoAdapter(BaseFrameworkAdapter):
    """
    Go integration adapter for various Go web frameworks.
    
    Generates Go code for HTTP handlers and middleware that integrate
    with Project Index API.
    """
    
    def __init__(self, config: Optional[ProjectIndexConfig] = None):
        super().__init__(config)
        self.project_root = Path.cwd()
        self.api_base_url = "http://localhost:8000/project-index"
    
    def set_api_url(self, url: str) -> None:
        """Set the Project Index API base URL."""
        self.api_base_url = url
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Generate Go integration code.
        
        Args:
            app: Not used for code generation
            **kwargs: Go-specific options
                - framework: Go framework ('gin', 'echo', 'stdlib', 'fiber')
                - package_name: Go package name (default: 'main')
        """
        framework = kwargs.get('framework', 'gin')
        package_name = kwargs.get('package_name', 'main')
        
        self._generate_go_client(package_name)
        
        if framework == 'gin':
            self._generate_gin_integration(package_name)
        elif framework == 'echo':
            self._generate_echo_integration(package_name)
        elif framework == 'fiber':
            self._generate_fiber_integration(package_name)
        else:
            self._generate_stdlib_integration(package_name)
    
    def _generate_go_client(self, package_name: str) -> None:
        """Generate Go client for Project Index API."""
        client_code = f"""
// Project Index Go Client
package {package_name}

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type ProjectIndexClient struct {{
    BaseURL    string
    HTTPClient *http.Client
}}

type ProjectStatus struct {{
    Status      string `json:"status"`
    Initialized bool   `json:"initialized"`
    Config      struct {{
        CacheEnabled              bool `json:"cache_enabled"`
        MonitoringEnabled         bool `json:"monitoring_enabled"`
        MaxConcurrentAnalyses     int  `json:"max_concurrent_analyses"`
    }} `json:"config"`
}}

type AnalysisRequest struct {{
    ProjectPath string   `json:"project_path"`
    Languages   []string `json:"languages,omitempty"`
}}

type AnalysisResult struct {{
    ProjectID         string   `json:"project_id"`
    FilesProcessed    int      `json:"files_processed"`
    DependenciesFound int      `json:"dependencies_found"`
    AnalysisTime      float64  `json:"analysis_time"`
    LanguagesDetected []string `json:"languages_detected"`
}}

func NewProjectIndexClient(baseURL string) *ProjectIndexClient {{
    return &ProjectIndexClient{{
        BaseURL: baseURL,
        HTTPClient: &http.Client{{
            Timeout: 30 * time.Second,
        }},
    }}
}}

func (c *ProjectIndexClient) GetStatus() (*ProjectStatus, error) {{
    resp, err := c.HTTPClient.Get(c.BaseURL + "/status")
    if err != nil {{
        return nil, fmt.Errorf("failed to get status: %w", err)
    }}
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {{
        return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
    }}

    var status ProjectStatus
    if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {{
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }}

    return &status, nil
}}

func (c *ProjectIndexClient) AnalyzeProject(req AnalysisRequest) (*AnalysisResult, error) {{
    jsonData, err := json.Marshal(req)
    if err != nil {{
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }}

    resp, err := c.HTTPClient.Post(
        c.BaseURL+"/analyze",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {{
        return nil, fmt.Errorf("failed to analyze project: %w", err)
    }}
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {{
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
    }}

    var result AnalysisResult
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {{
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }}

    return &result, nil
}}

func (c *ProjectIndexClient) ListProjects() (map[string]interface{{}}, error) {{
    resp, err := c.HTTPClient.Get(c.BaseURL + "/projects")
    if err != nil {{
        return nil, fmt.Errorf("failed to list projects: %w", err)
    }}
    defer resp.Body.Close()

    var result map[string]interface{{}}
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {{
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }}

    return result, nil
}}
"""
        
        self._write_file('project_index_client.go', client_code)
    
    def _generate_gin_integration(self, package_name: str) -> None:
        """Generate Gin framework integration."""
        gin_code = f"""
// Gin Framework Integration for Project Index
package {package_name}

import (
    "net/http"
    "os"
    
    "github.com/gin-gonic/gin"
)

var projectIndexClient = NewProjectIndexClient("{self.api_base_url}")

// ProjectIndexMiddleware adds Project Index client to Gin context
func ProjectIndexMiddleware() gin.HandlerFunc {{
    return func(c *gin.Context) {{
        c.Set("projectIndex", projectIndexClient)
        c.Next()
    }}
}}

// SetupProjectIndexRoutes adds Project Index routes to Gin router
func SetupProjectIndexRoutes(r *gin.Engine) {{
    api := r.Group("/api/project-index")
    {{
        api.GET("/status", getProjectIndexStatus)
        api.POST("/analyze", analyzeProject)
        api.GET("/projects", listProjects)
    }}
}}

func getProjectIndexStatus(c *gin.Context) {{
    status, err := projectIndexClient.GetStatus()
    if err != nil {{
        c.JSON(http.StatusServiceUnavailable, gin.H{{"error": "Project Index unavailable"}})
        return
    }}
    
    c.JSON(http.StatusOK, status)
}}

func analyzeProject(c *gin.Context) {{
    var req AnalysisRequest
    if err := c.ShouldBindJSON(&req); err != nil {{
        c.JSON(http.StatusBadRequest, gin.H{{"error": err.Error()}})
        return
    }}
    
    // Default to current directory if no path provided
    if req.ProjectPath == "" {{
        if cwd, err := os.Getwd(); err == nil {{
            req.ProjectPath = cwd
        }}
    }}
    
    result, err := projectIndexClient.AnalyzeProject(req)
    if err != nil {{
        c.JSON(http.StatusBadRequest, gin.H{{"error": err.Error()}})
        return
    }}
    
    c.JSON(http.StatusOK, result)
}}

func listProjects(c *gin.Context) {{
    projects, err := projectIndexClient.ListProjects()
    if err != nil {{
        c.JSON(http.StatusServiceUnavailable, gin.H{{"error": "Project Index unavailable"}})
        return
    }}
    
    c.JSON(http.StatusOK, projects)
}}

// Example Gin integration
func ExampleGinIntegration() {{
    r := gin.Default()
    
    // Add Project Index middleware
    r.Use(ProjectIndexMiddleware())
    
    // Setup Project Index routes
    SetupProjectIndexRoutes(r)
    
    // Example route using Project Index
    r.GET("/analyze-current", func(c *gin.Context) {{
        req := AnalysisRequest{{
            ProjectPath: ".", // Current directory
            Languages:   []string{{"go", "javascript", "python"}},
        }}
        
        result, err := projectIndexClient.AnalyzeProject(req)
        if err != nil {{
            c.JSON(http.StatusInternalServerError, gin.H{{"error": err.Error()}})
            return
        }}
        
        c.JSON(http.StatusOK, result)
    }})
    
    r.Run(":8080") // Start server on port 8080
}}
"""
        
        self._write_file('gin_integration.go', gin_code)
    
    def _generate_echo_integration(self, package_name: str) -> None:
        """Generate Echo framework integration."""
        echo_code = f"""
// Echo Framework Integration for Project Index
package {package_name}

import (
    "net/http"
    "os"
    
    "github.com/labstack/echo/v4"
    "github.com/labstack/echo/v4/middleware"
)

var projectIndexClient = NewProjectIndexClient("{self.api_base_url}")

// ProjectIndexMiddleware adds Project Index client to Echo context
func ProjectIndexMiddleware() echo.MiddlewareFunc {{
    return func(next echo.HandlerFunc) echo.HandlerFunc {{
        return func(c echo.Context) error {{
            c.Set("projectIndex", projectIndexClient)
            return next(c)
        }}
    }}
}}

// SetupProjectIndexRoutes adds Project Index routes to Echo
func SetupProjectIndexRoutes(e *echo.Echo) {{
    api := e.Group("/api/project-index")
    api.GET("/status", getProjectIndexStatusEcho)
    api.POST("/analyze", analyzeProjectEcho)
    api.GET("/projects", listProjectsEcho)
}}

func getProjectIndexStatusEcho(c echo.Context) error {{
    status, err := projectIndexClient.GetStatus()
    if err != nil {{
        return c.JSON(http.StatusServiceUnavailable, map[string]string{{"error": "Project Index unavailable"}})
    }}
    
    return c.JSON(http.StatusOK, status)
}}

func analyzeProjectEcho(c echo.Context) error {{
    var req AnalysisRequest
    if err := c.Bind(&req); err != nil {{
        return c.JSON(http.StatusBadRequest, map[string]string{{"error": err.Error()}})
    }}
    
    // Default to current directory if no path provided
    if req.ProjectPath == "" {{
        if cwd, err := os.Getwd(); err == nil {{
            req.ProjectPath = cwd
        }}
    }}
    
    result, err := projectIndexClient.AnalyzeProject(req)
    if err != nil {{
        return c.JSON(http.StatusBadRequest, map[string]string{{"error": err.Error()}})
    }}
    
    return c.JSON(http.StatusOK, result)
}}

func listProjectsEcho(c echo.Context) error {{
    projects, err := projectIndexClient.ListProjects()
    if err != nil {{
        return c.JSON(http.StatusServiceUnavailable, map[string]string{{"error": "Project Index unavailable"}})
    }}
    
    return c.JSON(http.StatusOK, projects)
}}

// Example Echo integration
func ExampleEchoIntegration() {{
    e := echo.New()
    
    // Middleware
    e.Use(middleware.Logger())
    e.Use(middleware.Recover())
    e.Use(ProjectIndexMiddleware())
    
    // Setup Project Index routes
    SetupProjectIndexRoutes(e)
    
    // Example route
    e.GET("/analyze-current", func(c echo.Context) error {{
        req := AnalysisRequest{{
            ProjectPath: ".",
            Languages:   []string{{"go"}},
        }}
        
        result, err := projectIndexClient.AnalyzeProject(req)
        if err != nil {{
            return c.JSON(http.StatusInternalServerError, map[string]string{{"error": err.Error()}})
        }}
        
        return c.JSON(http.StatusOK, result)
    }})
    
    e.Logger.Fatal(e.Start(":8080"))
}}
"""
        
        self._write_file('echo_integration.go', echo_code)
    
    def _generate_fiber_integration(self, package_name: str) -> None:
        """Generate Fiber framework integration."""
        fiber_code = f"""
// Fiber Framework Integration for Project Index
package {package_name}

import (
    "os"
    
    "github.com/gofiber/fiber/v2"
    "github.com/gofiber/fiber/v2/middleware/cors"
    "github.com/gofiber/fiber/v2/middleware/logger"
)

var projectIndexClient = NewProjectIndexClient("{self.api_base_url}")

// ProjectIndexMiddleware adds Project Index client to Fiber context
func ProjectIndexMiddleware() fiber.Handler {{
    return func(c *fiber.Ctx) error {{
        c.Locals("projectIndex", projectIndexClient)
        return c.Next()
    }}
}}

// SetupProjectIndexRoutes adds Project Index routes to Fiber
func SetupProjectIndexRoutes(app *fiber.App) {{
    api := app.Group("/api/project-index")
    api.Get("/status", getProjectIndexStatusFiber)
    api.Post("/analyze", analyzeProjectFiber)
    api.Get("/projects", listProjectsFiber)
}}

func getProjectIndexStatusFiber(c *fiber.Ctx) error {{
    status, err := projectIndexClient.GetStatus()
    if err != nil {{
        return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{{"error": "Project Index unavailable"}})
    }}
    
    return c.JSON(status)
}}

func analyzeProjectFiber(c *fiber.Ctx) error {{
    var req AnalysisRequest
    if err := c.BodyParser(&req); err != nil {{
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{{"error": err.Error()}})
    }}
    
    // Default to current directory if no path provided
    if req.ProjectPath == "" {{
        if cwd, err := os.Getwd(); err == nil {{
            req.ProjectPath = cwd
        }}
    }}
    
    result, err := projectIndexClient.AnalyzeProject(req)
    if err != nil {{
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{{"error": err.Error()}})
    }}
    
    return c.JSON(result)
}}

func listProjectsFiber(c *fiber.Ctx) error {{
    projects, err := projectIndexClient.ListProjects()
    if err != nil {{
        return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{{"error": "Project Index unavailable"}})
    }}
    
    return c.JSON(projects)
}}

// Example Fiber integration
func ExampleFiberIntegration() {{
    app := fiber.New()
    
    // Middleware
    app.Use(logger.New())
    app.Use(cors.New())
    app.Use(ProjectIndexMiddleware())
    
    // Setup Project Index routes
    SetupProjectIndexRoutes(app)
    
    // Example route
    app.Get("/analyze-current", func(c *fiber.Ctx) error {{
        req := AnalysisRequest{{
            ProjectPath: ".",
            Languages:   []string{{"go"}},
        }}
        
        result, err := projectIndexClient.AnalyzeProject(req)
        if err != nil {{
            return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{{"error": err.Error()}})
        }}
        
        return c.JSON(result)
    }})
    
    app.Listen(":8080")
}}
"""
        
        self._write_file('fiber_integration.go', fiber_code)
    
    def _generate_stdlib_integration(self, package_name: str) -> None:
        """Generate standard library HTTP integration."""
        stdlib_code = f"""
// Standard Library HTTP Integration for Project Index
package {package_name}

import (
    "encoding/json"
    "log"
    "net/http"
    "os"
)

var projectIndexClient = NewProjectIndexClient("{self.api_base_url}")

// ProjectIndexMiddleware wraps handlers with Project Index context
func ProjectIndexMiddleware(next http.HandlerFunc) http.HandlerFunc {{
    return func(w http.ResponseWriter, r *http.Request) {{
        // Add Project Index client to request context
        // In a real implementation, you'd use context.WithValue
        next(w, r)
    }}
}}

// SetupProjectIndexRoutes registers Project Index routes
func SetupProjectIndexRoutes() {{
    http.HandleFunc("/api/project-index/status", getProjectIndexStatusStdLib)
    http.HandleFunc("/api/project-index/analyze", analyzeProjectStdLib)
    http.HandleFunc("/api/project-index/projects", listProjectsStdLib)
}}

func getProjectIndexStatusStdLib(w http.ResponseWriter, r *http.Request) {{
    if r.Method != http.MethodGet {{
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }}
    
    status, err := projectIndexClient.GetStatus()
    if err != nil {{
        http.Error(w, "Project Index unavailable", http.StatusServiceUnavailable)
        return
    }}
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(status)
}}

func analyzeProjectStdLib(w http.ResponseWriter, r *http.Request) {{
    if r.Method != http.MethodPost {{
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }}
    
    var req AnalysisRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}
    
    // Default to current directory if no path provided
    if req.ProjectPath == "" {{
        if cwd, err := os.Getwd(); err == nil {{
            req.ProjectPath = cwd
        }}
    }}
    
    result, err := projectIndexClient.AnalyzeProject(req)
    if err != nil {{
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(result)
}}

func listProjectsStdLib(w http.ResponseWriter, r *http.Request) {{
    if r.Method != http.MethodGet {{
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }}
    
    projects, err := projectIndexClient.ListProjects()
    if err != nil {{
        http.Error(w, "Project Index unavailable", http.StatusServiceUnavailable)
        return
    }}
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(projects)
}}

// Example standard library integration
func ExampleStdLibIntegration() {{
    // Setup routes
    SetupProjectIndexRoutes()
    
    // Example route
    http.HandleFunc("/analyze-current", func(w http.ResponseWriter, r *http.Request) {{
        req := AnalysisRequest{{
            ProjectPath: ".",
            Languages:   []string{{"go"}},
        }}
        
        result, err := projectIndexClient.AnalyzeProject(req)
        if err != nil {{
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }}
        
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(result)
    }})
    
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}}
"""
        
        self._write_file('stdlib_integration.go', stdlib_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for Go code generation."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for Go code generation."""
        pass


class RustAdapter(BaseFrameworkAdapter):
    """
    Rust integration adapter for Axum and Rocket frameworks.
    
    Generates Rust code for HTTP handlers and middleware.
    """
    
    def __init__(self, config: Optional[ProjectIndexConfig] = None):
        super().__init__(config)
        self.project_root = Path.cwd()
        self.api_base_url = "http://localhost:8000/project-index"
    
    def set_api_url(self, url: str) -> None:
        """Set the Project Index API base URL."""
        self.api_base_url = url
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Generate Rust integration code.
        
        Args:
            app: Not used for code generation
            **kwargs: Rust-specific options
                - framework: Rust framework ('axum', 'rocket')
        """
        framework = kwargs.get('framework', 'axum')
        
        self._generate_rust_client()
        
        if framework == 'axum':
            self._generate_axum_integration()
        elif framework == 'rocket':
            self._generate_rocket_integration()
    
    def _generate_rust_client(self) -> None:
        """Generate Rust client for Project Index API."""
        client_code = f"""
// Project Index Rust Client
use reqwest::Client;
use serde::{{Deserialize, Serialize}};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ProjectIndexClient {{
    client: Client,
    base_url: String,
}}

#[derive(Debug, Deserialize)]
pub struct ProjectStatus {{
    pub status: String,
    pub initialized: bool,
    pub config: ProjectConfig,
}}

#[derive(Debug, Deserialize)]
pub struct ProjectConfig {{
    pub cache_enabled: bool,
    pub monitoring_enabled: bool,
    pub max_concurrent_analyses: u32,
}}

#[derive(Debug, Serialize)]
pub struct AnalysisRequest {{
    pub project_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
}}

#[derive(Debug, Deserialize)]
pub struct AnalysisResult {{
    pub project_id: String,
    pub files_processed: u32,
    pub dependencies_found: u32,
    pub analysis_time: f64,
    pub languages_detected: Vec<String>,
}}

impl ProjectIndexClient {{
    pub fn new(base_url: impl Into<String>) -> Self {{
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {{
            client,
            base_url: base_url.into(),
        }}
    }}

    pub async fn get_status(&self) -> Result<ProjectStatus, reqwest::Error> {{
        let url = format!("{{}}/status", self.base_url);
        let response = self.client.get(&url).send().await?;
        response.json().await
    }}

    pub async fn analyze_project(&self, request: AnalysisRequest) -> Result<AnalysisResult, reqwest::Error> {{
        let url = format!("{{}}/analyze", self.base_url);
        let response = self.client.post(&url).json(&request).send().await?;
        response.json().await
    }}

    pub async fn list_projects(&self) -> Result<HashMap<String, serde_json::Value>, reqwest::Error> {{
        let url = format!("{{}}/projects", self.base_url);
        let response = self.client.get(&url).send().await?;
        response.json().await
    }}
}}

impl Default for ProjectIndexClient {{
    fn default() -> Self {{
        Self::new("{self.api_base_url}")
    }}
}}
"""
        
        self._write_file('src/project_index_client.rs', client_code)
    
    def _generate_axum_integration(self) -> None:
        """Generate Axum framework integration."""
        axum_code = f"""
// Axum Framework Integration for Project Index
use axum::{{
    extract::{{Extension, Json}},
    http::StatusCode,
    response::Json as ResponseJson,
    routing::{{get, post}},
    Router,
}};
use serde_json::{{json, Value}};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

use crate::project_index_client::{{ProjectIndexClient, AnalysisRequest}};

// Shared state
pub type SharedClient = Arc<ProjectIndexClient>;

// Create router with Project Index routes
pub fn create_project_index_router() -> Router {{
    Router::new()
        .route("/status", get(get_status_handler))
        .route("/analyze", post(analyze_project_handler))
        .route("/projects", get(list_projects_handler))
}}

// Handlers
async fn get_status_handler(Extension(client): Extension<SharedClient>) -> Result<ResponseJson<Value>, StatusCode> {{
    match client.get_status().await {{
        Ok(status) => Ok(ResponseJson(json!(status))),
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }}
}}

async fn analyze_project_handler(
    Extension(client): Extension<SharedClient>,
    Json(request): Json<AnalysisRequest>,
) -> Result<ResponseJson<Value>, StatusCode> {{
    match client.analyze_project(request).await {{
        Ok(result) => Ok(ResponseJson(json!(result))),
        Err(_) => Err(StatusCode::BAD_REQUEST),
    }}
}}

async fn list_projects_handler(Extension(client): Extension<SharedClient>) -> Result<ResponseJson<Value>, StatusCode> {{
    match client.list_projects().await {{
        Ok(projects) => Ok(ResponseJson(json!(projects))),
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }}
}}

// Complete Axum app setup
pub fn create_app() -> Router {{
    let client = Arc::new(ProjectIndexClient::default());
    
    Router::new()
        .nest("/api/project-index", create_project_index_router())
        .route("/analyze-current", get(analyze_current_handler))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .layer(Extension(client))
        )
}}

// Example handler using Project Index
async fn analyze_current_handler(Extension(client): Extension<SharedClient>) -> Result<ResponseJson<Value>, StatusCode> {{
    let request = AnalysisRequest {{
        project_path: ".".to_string(),
        languages: Some(vec!["rust".to_string()]),
    }};
    
    match client.analyze_project(request).await {{
        Ok(result) => Ok(ResponseJson(json!(result))),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }}
}}

// Example main function
#[tokio::main]
async fn example_main() {{
    let app = create_app();
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .unwrap();
    
    println!("🚀 Axum server with Project Index integration running on http://0.0.0.0:8080");
    
    axum::serve(listener, app).await.unwrap();
}}
"""
        
        self._write_file('src/axum_integration.rs', axum_code)
    
    def _generate_rocket_integration(self) -> None:
        """Generate Rocket framework integration."""
        rocket_code = f"""
// Rocket Framework Integration for Project Index
use rocket::{{get, post, routes, launch, State, serde::json::Json}};
use rocket::serde::{{Deserialize, Serialize}};
use serde_json::{{json, Value}};
use std::sync::Arc;

use crate::project_index_client::{{ProjectIndexClient, AnalysisRequest}};

// Shared state
pub type SharedClient = Arc<ProjectIndexClient>;

#[derive(Responder)]
#[response(status = 503)]
pub struct ServiceUnavailable {{
    message: String,
}}

#[derive(Responder)]
#[response(status = 400)]
pub struct BadRequest {{
    message: String,
}}

// Routes
#[get("/status")]
async fn get_status(client: &State<SharedClient>) -> Result<Json<Value>, ServiceUnavailable> {{
    match client.get_status().await {{
        Ok(status) => Ok(Json(json!(status))),
        Err(e) => Err(ServiceUnavailable {{
            message: format!("Project Index unavailable: {{}}", e),
        }}),
    }}
}}

#[post("/analyze", data = "<request>")]
async fn analyze_project(
    client: &State<SharedClient>,
    request: Json<AnalysisRequest>,
) -> Result<Json<Value>, BadRequest> {{
    match client.analyze_project(request.into_inner()).await {{
        Ok(result) => Ok(Json(json!(result))),
        Err(e) => Err(BadRequest {{
            message: format!("Analysis failed: {{}}", e),
        }}),
    }}
}}

#[get("/projects")]
async fn list_projects(client: &State<SharedClient>) -> Result<Json<Value>, ServiceUnavailable> {{
    match client.list_projects().await {{
        Ok(projects) => Ok(Json(json!(projects))),
        Err(e) => Err(ServiceUnavailable {{
            message: format!("Project Index unavailable: {{}}", e),
        }}),
    }}
}}

#[get("/analyze-current")]
async fn analyze_current(client: &State<SharedClient>) -> Result<Json<Value>, BadRequest> {{
    let request = AnalysisRequest {{
        project_path: ".".to_string(),
        languages: Some(vec!["rust".to_string()]),
    }};
    
    match client.analyze_project(request).await {{
        Ok(result) => Ok(Json(json!(result))),
        Err(e) => Err(BadRequest {{
            message: format!("Analysis failed: {{}}", e),
        }}),
    }}
}}

#[launch]
fn rocket() -> _ {{
    let client = Arc::new(ProjectIndexClient::default());
    
    rocket::build()
        .manage(client)
        .mount("/api/project-index", routes![get_status, analyze_project, list_projects])
        .mount("/", routes![analyze_current])
}}

// Example of using Project Index in Rocket
pub fn example_rocket_setup() {{
    println!("🚀 Rocket server with Project Index integration");
    // The #[launch] macro handles the rest
}}
"""
        
        self._write_file('src/rocket_integration.rs', rocket_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for Rust code generation."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for Rust code generation."""
        pass


class JavaAdapter(BaseFrameworkAdapter):
    """
    Java Spring Boot integration adapter.
    
    Generates Java code for Spring Boot integration with annotations.
    """
    
    def __init__(self, config: Optional[ProjectIndexConfig] = None):
        super().__init__(config)
        self.project_root = Path.cwd()
        self.api_base_url = "http://localhost:8000/project-index"
        self.package_name = "com.example.projectindex"
    
    def set_api_url(self, url: str) -> None:
        """Set the Project Index API base URL."""
        self.api_base_url = url
    
    def set_package_name(self, package_name: str) -> None:
        """Set the Java package name."""
        self.package_name = package_name
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Generate Java Spring Boot integration code.
        
        Args:
            app: Not used for code generation
            **kwargs: Java-specific options
                - package_name: Java package name
        """
        package_name = kwargs.get('package_name', self.package_name)
        self.package_name = package_name
        
        self._generate_java_client()
        self._generate_spring_boot_integration()
        self._generate_configuration()
    
    def _generate_java_client(self) -> None:
        """Generate Java client for Project Index API."""
        client_code = f"""
// Project Index Java Client
package {self.package_name}.client;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.RestClientException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProjectIndexClient {{
    
    private final RestTemplate restTemplate;
    private final String baseUrl;
    private final ObjectMapper objectMapper;
    
    public ProjectIndexClient(String baseUrl) {{
        this.baseUrl = baseUrl;
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }}
    
    // Data classes
    public static class ProjectStatus {{
        public String status;
        public boolean initialized;
        public ProjectConfig config;
        
        // Getters and setters
        public String getStatus() {{ return status; }}
        public void setStatus(String status) {{ this.status = status; }}
        
        public boolean isInitialized() {{ return initialized; }}
        public void setInitialized(boolean initialized) {{ this.initialized = initialized; }}
        
        public ProjectConfig getConfig() {{ return config; }}
        public void setConfig(ProjectConfig config) {{ this.config = config; }}
    }}
    
    public static class ProjectConfig {{
        @JsonProperty("cache_enabled")
        public boolean cacheEnabled;
        
        @JsonProperty("monitoring_enabled")
        public boolean monitoringEnabled;
        
        @JsonProperty("max_concurrent_analyses")
        public int maxConcurrentAnalyses;
        
        // Getters and setters
        public boolean isCacheEnabled() {{ return cacheEnabled; }}
        public void setCacheEnabled(boolean cacheEnabled) {{ this.cacheEnabled = cacheEnabled; }}
        
        public boolean isMonitoringEnabled() {{ return monitoringEnabled; }}
        public void setMonitoringEnabled(boolean monitoringEnabled) {{ this.monitoringEnabled = monitoringEnabled; }}
        
        public int getMaxConcurrentAnalyses() {{ return maxConcurrentAnalyses; }}
        public void setMaxConcurrentAnalyses(int maxConcurrentAnalyses) {{ this.maxConcurrentAnalyses = maxConcurrentAnalyses; }}
    }}
    
    public static class AnalysisRequest {{
        @JsonProperty("project_path")
        public String projectPath;
        
        public List<String> languages;
        
        public AnalysisRequest(String projectPath, List<String> languages) {{
            this.projectPath = projectPath;
            this.languages = languages;
        }}
        
        // Getters and setters
        public String getProjectPath() {{ return projectPath; }}
        public void setProjectPath(String projectPath) {{ this.projectPath = projectPath; }}
        
        public List<String> getLanguages() {{ return languages; }}
        public void setLanguages(List<String> languages) {{ this.languages = languages; }}
    }}
    
    public static class AnalysisResult {{
        @JsonProperty("project_id")
        public String projectId;
        
        @JsonProperty("files_processed")
        public int filesProcessed;
        
        @JsonProperty("dependencies_found")
        public int dependenciesFound;
        
        @JsonProperty("analysis_time")
        public double analysisTime;
        
        @JsonProperty("languages_detected")
        public List<String> languagesDetected;
        
        // Getters and setters
        public String getProjectId() {{ return projectId; }}
        public void setProjectId(String projectId) {{ this.projectId = projectId; }}
        
        public int getFilesProcessed() {{ return filesProcessed; }}
        public void setFilesProcessed(int filesProcessed) {{ this.filesProcessed = filesProcessed; }}
        
        public int getDependenciesFound() {{ return dependenciesFound; }}
        public void setDependenciesFound(int dependenciesFound) {{ this.dependenciesFound = dependenciesFound; }}
        
        public double getAnalysisTime() {{ return analysisTime; }}
        public void setAnalysisTime(double analysisTime) {{ this.analysisTime = analysisTime; }}
        
        public List<String> getLanguagesDetected() {{ return languagesDetected; }}
        public void setLanguagesDetected(List<String> languagesDetected) {{ this.languagesDetected = languagesDetected; }}
    }}
    
    // API methods
    public ProjectStatus getStatus() throws RestClientException {{
        String url = baseUrl + "/status";
        ResponseEntity<ProjectStatus> response = restTemplate.getForEntity(url, ProjectStatus.class);
        return response.getBody();
    }}
    
    public AnalysisResult analyzeProject(AnalysisRequest request) throws RestClientException {{
        String url = baseUrl + "/analyze";
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<AnalysisRequest> entity = new HttpEntity<>(request, headers);
        ResponseEntity<AnalysisResult> response = restTemplate.postForEntity(url, entity, AnalysisResult.class);
        
        return response.getBody();
    }}
    
    public Map<String, Object> listProjects() throws RestClientException {{
        String url = baseUrl + "/projects";
        ResponseEntity<HashMap> response = restTemplate.getForEntity(url, HashMap.class);
        return response.getBody();
    }}
}}
"""
        
        package_dir = self.package_name.replace('.', '/')
        self._write_file(f'src/main/java/{package_dir}/client/ProjectIndexClient.java', client_code)
    
    def _generate_spring_boot_integration(self) -> None:
        """Generate Spring Boot controller and service."""
        controller_code = f"""
// Spring Boot Controller for Project Index
package {self.package_name}.controller;

import {self.package_name}.client.ProjectIndexClient;
import {self.package_name}.client.ProjectIndexClient.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientException;

import java.util.Arrays;
import java.util.Map;

@RestController
@RequestMapping("/api/project-index")
@CrossOrigin(origins = "*")
public class ProjectIndexController {{
    
    @Autowired
    private ProjectIndexClient projectIndexClient;
    
    @GetMapping("/status")
    public ResponseEntity<?> getStatus() {{
        try {{
            ProjectStatus status = projectIndexClient.getStatus();
            return ResponseEntity.ok(status);
        }} catch (RestClientException e) {{
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Project Index unavailable"));
        }}
    }}
    
    @PostMapping("/analyze")
    public ResponseEntity<?> analyzeProject(@RequestBody AnalysisRequest request) {{
        try {{
            // Default to current directory if no path provided
            if (request.getProjectPath() == null || request.getProjectPath().isEmpty()) {{
                request.setProjectPath(System.getProperty("user.dir"));
            }}
            
            AnalysisResult result = projectIndexClient.analyzeProject(request);
            return ResponseEntity.ok(result);
        }} catch (RestClientException e) {{
            return ResponseEntity.badRequest()
                    .body(Map.of("error", e.getMessage()));
        }}
    }}
    
    @GetMapping("/projects")
    public ResponseEntity<?> listProjects() {{
        try {{
            Map<String, Object> projects = projectIndexClient.listProjects();
            return ResponseEntity.ok(projects);
        }} catch (RestClientException e) {{
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Project Index unavailable"));
        }}
    }}
    
    @GetMapping("/analyze-current")
    public ResponseEntity<?> analyzeCurrent() {{
        try {{
            AnalysisRequest request = new AnalysisRequest(
                System.getProperty("user.dir"),
                Arrays.asList("java", "javascript", "python")
            );
            
            AnalysisResult result = projectIndexClient.analyzeProject(request);
            return ResponseEntity.ok(result);
        }} catch (RestClientException e) {{
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", e.getMessage()));
        }}
    }}
}}
"""
        
        package_dir = self.package_name.replace('.', '/')
        self._write_file(f'src/main/java/{package_dir}/controller/ProjectIndexController.java', controller_code)
    
    def _generate_configuration(self) -> None:
        """Generate Spring Boot configuration."""
        config_code = f"""
// Spring Boot Configuration for Project Index
package {self.package_name}.config;

import {self.package_name}.client.ProjectIndexClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ProjectIndexConfiguration {{
    
    @Value("${{project-index.api.url:{self.api_base_url}}}")
    private String projectIndexApiUrl;
    
    @Bean
    public ProjectIndexClient projectIndexClient() {{
        return new ProjectIndexClient(projectIndexApiUrl);
    }}
}}
"""
        
        package_dir = self.package_name.replace('.', '/')
        self._write_file(f'src/main/java/{package_dir}/config/ProjectIndexConfiguration.java', config_code)
        
        # Generate application.properties
        properties_code = f"""
# Project Index Configuration
project-index.api.url={self.api_base_url}

# Spring Boot Configuration
server.port=8080
spring.application.name=project-index-integration

# Logging
logging.level.{self.package_name}=DEBUG
logging.level.org.springframework.web=DEBUG
"""
        
        self._write_file('src/main/resources/application.properties', properties_code)
        
        # Generate main application class
        main_app_code = f"""
// Spring Boot Main Application
package {self.package_name};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ProjectIndexApplication {{
    
    public static void main(String[] args) {{
        System.out.println("🚀 Starting Spring Boot with Project Index integration...");
        SpringApplication.run(ProjectIndexApplication.class, args);
    }}
}}
"""
        
        package_dir = self.package_name.replace('.', '/')
        self._write_file(f'src/main/java/{package_dir}/ProjectIndexApplication.java', main_app_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for Java code generation."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for Java code generation."""
        pass
    
    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to file, creating directories as needed."""
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            full_path.write_text(content)
            print(f"✅ Generated {file_path}")
        except Exception as e:
            print(f"❌ Failed to generate {file_path}: {e}")


# Add _write_file method to other adapters
GoAdapter._write_file = JavaAdapter._write_file
RustAdapter._write_file = JavaAdapter._write_file

# Register other language adapters
IntegrationManager.register_adapter('go', GoAdapter)
IntegrationManager.register_adapter('rust', RustAdapter)
IntegrationManager.register_adapter('java', JavaAdapter)


# Convenience functions for code generation
def generate_go_integration(framework: str = 'gin', api_url: str = "http://localhost:8000/project-index", **kwargs) -> GoAdapter:
    """Generate Go integration code."""
    adapter = GoAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(framework=framework, **kwargs)
    return adapter


def generate_rust_integration(framework: str = 'axum', api_url: str = "http://localhost:8000/project-index", **kwargs) -> RustAdapter:
    """Generate Rust integration code."""
    adapter = RustAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(framework=framework, **kwargs)
    return adapter


def generate_java_integration(api_url: str = "http://localhost:8000/project-index", package_name: str = "com.example.projectindex", **kwargs) -> JavaAdapter:
    """Generate Java Spring Boot integration code."""
    adapter = JavaAdapter()
    adapter.set_api_url(api_url)
    adapter.set_package_name(package_name)
    adapter.integrate(package_name=package_name, **kwargs)
    return adapter