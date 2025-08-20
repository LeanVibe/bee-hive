#!/usr/bin/env python3
"""
Production Deployment Validator for LeanVibe Agent Hive Mobile Dashboard
Comprehensive validation of production readiness with mobile-specific checks
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import websockets
import ssl
import socket
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionValidator:
    """Validates production deployment readiness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://localhost')
        self.ws_url = config.get('ws_url', 'wss://localhost')
        self.timeout = config.get('timeout', 30)
        self.mobile_user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'unknown',
            'validations': {},
            'performance_metrics': {},
            'mobile_specific': {},
            'security_checks': {},
            'compliance': {}
        }
    
    async def validate_deployment(self) -> Dict[str, Any]:
        """Run complete production deployment validation"""
        logger.info("Starting production deployment validation...")
        
        try:
            # Core infrastructure validation
            await self.validate_infrastructure()
            
            # API and service validation
            await self.validate_api_services()
            
            # Mobile-specific validation
            await self.validate_mobile_features()
            
            # WebSocket validation
            await self.validate_websocket_functionality()
            
            # Security validation
            await self.validate_security_measures()
            
            # Performance validation
            await self.validate_performance_targets()
            
            # Firebase FCM validation
            await self.validate_fcm_functionality()
            
            # PWA validation
            await self.validate_pwa_features()
            
            # Calculate overall status
            self.results['overall_status'] = self._calculate_overall_status()
            
            logger.info(f"Validation completed with status: {self.results['overall_status']}")
            return self.results
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            self.results['overall_status'] = 'failed'
            self.results['error'] = str(e)
            return self.results
    
    async def validate_infrastructure(self):
        """Validate core infrastructure components"""
        logger.info("Validating infrastructure...")
        
        infrastructure_checks = {
            'ssl_certificate': await self._check_ssl_certificate(),
            'dns_resolution': await self._check_dns_resolution(),
            'load_balancer': await self._check_load_balancer(),
            'cdn_performance': await self._check_cdn_performance(),
            'database_connectivity': await self._check_database_health(),
            'redis_connectivity': await self._check_redis_health(),
            'monitoring_systems': await self._check_monitoring_systems()
        }
        
        self.results['validations']['infrastructure'] = infrastructure_checks
    
    async def validate_api_services(self):
        """Validate API services and endpoints"""
        logger.info("Validating API services...")
        
        api_endpoints = [
            ('/health', 'GET'),
            ('/api/dashboard/status', 'GET'),
            ('/api/agents/list', 'GET'),
            ('/api/coordination/status', 'GET'),
            ('/api/mobile/health', 'GET')
        ]
        
        api_checks = {}
        
        for endpoint, method in api_endpoints:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    headers = {'User-Agent': self.mobile_user_agent}
                    url = urljoin(self.base_url, endpoint)
                    
                    async with session.request(method, url, headers=headers, timeout=self.timeout) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        api_checks[endpoint] = {
                            'status': 'pass' if response.status == 200 else 'fail',
                            'response_code': response.status,
                            'response_time_ms': response_time,
                            'content_type': response.headers.get('Content-Type'),
                            'mobile_optimized': self._check_mobile_optimization(response.headers)
                        }
                        
                        # Check response content for critical endpoints
                        if endpoint == '/health' and response.status == 200:
                            health_data = await response.json()
                            api_checks[endpoint]['health_details'] = health_data
                            
            except Exception as e:
                api_checks[endpoint] = {
                    'status': 'fail',
                    'error': str(e)
                }
        
        self.results['validations']['api_services'] = api_checks
    
    async def validate_mobile_features(self):
        """Validate mobile-specific features"""
        logger.info("Validating mobile features...")
        
        mobile_checks = {
            'responsive_design': await self._check_responsive_design(),
            'touch_optimization': await self._check_touch_optimization(),
            'viewport_configuration': await self._check_viewport_config(),
            'mobile_performance': await self._check_mobile_performance(),
            'offline_functionality': await self._check_offline_functionality()
        }
        
        self.results['mobile_specific'] = mobile_checks
    
    async def validate_websocket_functionality(self):
        """Validate WebSocket functionality for mobile"""
        logger.info("Validating WebSocket functionality...")
        
        ws_checks = {
            'connection': await self._check_websocket_connection(),
            'mobile_compatibility': await self._check_mobile_websocket(),
            'message_throughput': await self._check_websocket_performance(),
            'reconnection_logic': await self._check_websocket_resilience()
        }
        
        self.results['validations']['websocket'] = ws_checks
    
    async def validate_security_measures(self):
        """Validate security configuration"""
        logger.info("Validating security measures...")
        
        security_checks = {
            'https_enforcement': await self._check_https_enforcement(),
            'security_headers': await self._check_security_headers(),
            'cors_configuration': await self._check_cors_config(),
            'rate_limiting': await self._check_rate_limiting(),
            'content_security_policy': await self._check_csp_policy(),
            'mobile_security_headers': await self._check_mobile_security()
        }
        
        self.results['security_checks'] = security_checks
    
    async def validate_performance_targets(self):
        """Validate performance targets are met"""
        logger.info("Validating performance targets...")
        
        performance_metrics = {
            'page_load_time': await self._measure_page_load_time(),
            'api_response_times': await self._measure_api_performance(),
            'resource_loading': await self._measure_resource_performance(),
            'mobile_specific_metrics': await self._measure_mobile_performance()
        }
        
        self.results['performance_metrics'] = performance_metrics
    
    async def validate_fcm_functionality(self):
        """Validate Firebase FCM functionality"""
        logger.info("Validating FCM functionality...")
        
        fcm_checks = {
            'firebase_config': await self._check_firebase_config(),
            'service_worker': await self._check_service_worker(),
            'notification_permissions': await self._check_notification_setup(),
            'push_endpoint': await self._check_push_endpoint()
        }
        
        self.results['validations']['fcm'] = fcm_checks
    
    async def validate_pwa_features(self):
        """Validate PWA features"""
        logger.info("Validating PWA features...")
        
        pwa_checks = {
            'manifest': await self._check_pwa_manifest(),
            'service_worker_registration': await self._check_sw_registration(),
            'offline_support': await self._check_offline_support(),
            'installability': await self._check_pwa_installability(),
            'icon_configuration': await self._check_pwa_icons()
        }
        
        self.results['validations']['pwa'] = pwa_checks
    
    # Infrastructure check methods
    async def _check_ssl_certificate(self) -> Dict[str, Any]:
        """Check SSL certificate validity"""
        try:
            hostname = self.base_url.replace('https://', '').replace('http://', '')
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.utcnow()).days
                    
                    return {
                        'status': 'pass' if days_until_expiry > 30 else 'warning',
                        'days_until_expiry': days_until_expiry,
                        'issuer': cert.get('issuer'),
                        'subject': cert.get('subject')
                    }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_dns_resolution(self) -> Dict[str, Any]:
        """Check DNS resolution"""
        try:
            hostname = self.base_url.replace('https://', '').replace('http://', '')
            start_time = time.time()
            
            # Use asyncio's getaddrinfo for async DNS resolution
            loop = asyncio.get_event_loop()
            result = await loop.getaddrinfo(hostname, None)
            
            resolution_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'pass',
                'resolution_time_ms': resolution_time,
                'addresses': [addr[4][0] for addr in result if addr[0] == socket.AF_INET]
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_load_balancer(self) -> Dict[str, Any]:
        """Check load balancer health"""
        try:
            async with aiohttp.ClientSession() as session:
                # Make multiple requests to check load balancing
                response_times = []
                server_headers = set()
                
                for _ in range(5):
                    start_time = time.time()
                    async with session.get(f"{self.base_url}/health") as response:
                        response_times.append((time.time() - start_time) * 1000)
                        server_headers.add(response.headers.get('Server', 'unknown'))
                
                avg_response_time = sum(response_times) / len(response_times)
                
                return {
                    'status': 'pass',
                    'average_response_time_ms': avg_response_time,
                    'response_time_consistency': max(response_times) - min(response_times) < 100,
                    'unique_servers': len(server_headers)
                }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_cdn_performance(self) -> Dict[str, Any]:
        """Check CDN performance"""
        try:
            # Check static asset loading with CDN headers
            static_urls = [
                f"{self.base_url}/assets/app.js",
                f"{self.base_url}/assets/app.css",
                f"{self.base_url}/manifest.json"
            ]
            
            cdn_metrics = {}
            
            for url in static_urls:
                try:
                    start_time = time.time()
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            load_time = (time.time() - start_time) * 1000
                            
                            cdn_metrics[url] = {
                                'load_time_ms': load_time,
                                'cache_status': response.headers.get('X-Cache-Status', 'unknown'),
                                'cdn_headers': {
                                    'cache_control': response.headers.get('Cache-Control'),
                                    'expires': response.headers.get('Expires'),
                                    'etag': response.headers.get('ETag')
                                }
                            }
                except Exception as e:
                    cdn_metrics[url] = {'status': 'fail', 'error': str(e)}
            
            return {
                'status': 'pass',
                'metrics': cdn_metrics
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity through API"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/dashboard/status") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'pass',
                            'response_time_ms': response_time,
                            'database_status': data.get('database', 'unknown')
                        }
                    else:
                        return {
                            'status': 'fail',
                            'response_code': response.status
                        }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity through API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/coordination/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'pass',
                            'redis_status': data.get('redis', 'unknown')
                        }
                    else:
                        return {
                            'status': 'fail',
                            'response_code': response.status
                        }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_monitoring_systems(self) -> Dict[str, Any]:
        """Check monitoring system health"""
        monitoring_endpoints = {
            'prometheus': f"{self.base_url}:9090/-/healthy",
            'grafana': f"{self.base_url}:3001/api/health"
        }
        
        monitoring_status = {}
        
        for system, endpoint in monitoring_endpoints.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=10) as response:
                        monitoring_status[system] = {
                            'status': 'pass' if response.status == 200 else 'fail',
                            'response_code': response.status
                        }
            except Exception as e:
                monitoring_status[system] = {
                    'status': 'fail',
                    'error': str(e)
                }
        
        return monitoring_status
    
    # Mobile-specific check methods
    async def _check_responsive_design(self) -> Dict[str, Any]:
        """Check responsive design implementation"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': self.mobile_user_agent,
                    'Viewport-Width': '375'  # iPhone viewport width
                }
                
                async with session.get(self.base_url, headers=headers) as response:
                    content = await response.text()
                    
                    # Check for viewport meta tag
                    has_viewport = 'viewport' in content and 'device-width' in content
                    
                    # Check for responsive CSS
                    has_media_queries = '@media' in content or 'responsive' in content
                    
                    return {
                        'status': 'pass' if has_viewport and has_media_queries else 'fail',
                        'viewport_meta': has_viewport,
                        'responsive_css': has_media_queries
                    }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_touch_optimization(self) -> Dict[str, Any]:
        """Check touch optimization for mobile"""
        # This would require browser automation for full validation
        # For now, we'll check if touch-related CSS/JS is present
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/assets/app.css") as response:
                    css_content = await response.text()
                    
                    touch_optimized = any(keyword in css_content for keyword in [
                        'touch-action',
                        'min-height: 44px',  # Minimum touch target size
                        'tap-highlight',
                        'user-select'
                    ])
                    
                    return {
                        'status': 'pass' if touch_optimized else 'warning',
                        'touch_optimizations_found': touch_optimized
                    }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_viewport_config(self) -> Dict[str, Any]:
        """Check viewport configuration"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    content = await response.text()
                    
                    # Extract viewport content
                    import re
                    viewport_match = re.search(r'<meta\s+name="viewport"\s+content="([^"]*)"', content, re.IGNORECASE)
                    
                    if viewport_match:
                        viewport_content = viewport_match.group(1)
                        has_device_width = 'device-width' in viewport_content
                        has_initial_scale = 'initial-scale=1' in viewport_content
                        
                        return {
                            'status': 'pass' if has_device_width and has_initial_scale else 'warning',
                            'viewport_content': viewport_content,
                            'device_width': has_device_width,
                            'initial_scale': has_initial_scale
                        }
                    else:
                        return {
                            'status': 'fail',
                            'error': 'No viewport meta tag found'
                        }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    # WebSocket check methods
    async def _check_websocket_connection(self) -> Dict[str, Any]:
        """Check WebSocket connectivity"""
        try:
            ws_url = self.ws_url.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws/mobile/test'
            
            start_time = time.time()
            async with websockets.connect(ws_url, timeout=self.timeout) as websocket:
                connection_time = (time.time() - start_time) * 1000
                
                # Send test message
                test_message = json.dumps({'type': 'ping', 'timestamp': time.time()})
                await websocket.send(test_message)
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                return {
                    'status': 'pass',
                    'connection_time_ms': connection_time,
                    'ping_response': response_data.get('type') == 'pong'
                }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    async def _check_mobile_websocket(self) -> Dict[str, Any]:
        """Check mobile-specific WebSocket features"""
        try:
            ws_url = self.ws_url.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws/mobile/dashboard'
            
            headers = {
                'User-Agent': self.mobile_user_agent,
                'X-Mobile-Client': 'true'
            }
            
            async with websockets.connect(ws_url, timeout=self.timeout, extra_headers=headers) as websocket:
                # Test mobile-specific messages
                mobile_message = json.dumps({
                    'type': 'mobile_dashboard_subscribe',
                    'client_info': {
                        'platform': 'mobile',
                        'user_agent': self.mobile_user_agent
                    }
                })
                
                await websocket.send(mobile_message)
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                
                return {
                    'status': 'pass',
                    'mobile_support': True,
                    'response_received': bool(response)
                }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    def _check_mobile_optimization(self, headers: Dict[str, str]) -> bool:
        """Check if response is optimized for mobile"""
        mobile_optimizations = [
            headers.get('Content-Encoding') in ['gzip', 'br'],  # Compression
            'no-cache' not in headers.get('Cache-Control', ''),  # Appropriate caching
            int(headers.get('Content-Length', '0')) < 1000000,  # Reasonable size
        ]
        
        return sum(mobile_optimizations) >= 2
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall deployment status"""
        def extract_statuses(obj):
            statuses = []
            if isinstance(obj, dict):
                if 'status' in obj:
                    statuses.append(obj['status'])
                for value in obj.values():
                    statuses.extend(extract_statuses(value))
            elif isinstance(obj, list):
                for item in obj:
                    statuses.extend(extract_statuses(item))
            return statuses
        
        all_statuses = extract_statuses(self.results)
        
        if not all_statuses:
            return 'unknown'
        
        fail_count = all_statuses.count('fail')
        warning_count = all_statuses.count('warning')
        pass_count = all_statuses.count('pass')
        total_count = len(all_statuses)
        
        if fail_count > total_count * 0.1:  # More than 10% failures
            return 'not_ready'
        elif fail_count > 0 or warning_count > total_count * 0.2:  # Some failures or many warnings
            return 'needs_attention'
        elif pass_count >= total_count * 0.9:  # 90% or more pass
            return 'ready'
        else:
            return 'partial'


# Additional check methods for completeness
async def _check_mobile_performance(self) -> Dict[str, Any]:
    """Check mobile performance metrics"""
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            headers = {'User-Agent': self.mobile_user_agent}
            async with session.get(self.base_url, headers=headers) as response:
                load_time = (time.time() - start_time) * 1000
                content_size = len(await response.read())
                
                # Performance targets for mobile
                performance_targets = {
                    'load_time_under_3s': load_time < 3000,
                    'content_size_under_500kb': content_size < 512000,
                    'gzip_compression': response.headers.get('Content-Encoding') == 'gzip'
                }
                
                return {
                    'status': 'pass' if all(performance_targets.values()) else 'warning',
                    'load_time_ms': load_time,
                    'content_size_bytes': content_size,
                    'targets_met': performance_targets
                }
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


async def main():
    """Main CLI entry point"""
    config = {
        'base_url': os.getenv('BASE_URL', 'https://localhost'),
        'ws_url': os.getenv('WS_URL', 'wss://localhost'),
        'timeout': int(os.getenv('TIMEOUT', '30'))
    }
    
    validator = ProductionValidator(config)
    results = await validator.validate_deployment()
    
    # Output results
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    status = results['overall_status']
    if status == 'ready':
        sys.exit(0)
    elif status in ['partial', 'needs_attention']:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class Production-deployment-validatorScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(Production-deployment-validatorScript)