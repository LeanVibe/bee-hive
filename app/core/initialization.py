"""
LeanVibe Agent Hive 2.0 Initialization Module

Common initialization routines for Redis, Database, and other critical infrastructure
components. This module ensures consistent initialization across API servers, CLI commands,
and other entry points.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class InitializationError(Exception):
    """Raised when initialization of a critical component fails."""
    pass


class SystemInitializer:
    """Manages initialization and lifecycle of system components."""
    
    def __init__(self):
        self.initialized_components = set()
        self.redis_client = None
        self.database_engine = None
        self.initialization_errors = []
    
    async def initialize_all(self, components: Optional[list] = None) -> Dict[str, bool]:
        """Initialize all system components or specified subset."""
        components = components or ['redis', 'database']
        results = {}
        
        logger.info("Starting LeanVibe Agent Hive 2.0 system initialization...")
        
        for component in components:
            try:
                if component == 'redis':
                    results['redis'] = await self.initialize_redis()
                elif component == 'database':
                    results['database'] = await self.initialize_database()
                else:
                    logger.warning(f"Unknown component: {component}")
                    results[component] = False
            except Exception as e:
                logger.error(f"Failed to initialize {component}: {e}")
                results[component] = False
                self.initialization_errors.append((component, str(e)))
        
        success_count = sum(results.values())
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("✅ All system components initialized successfully")
        else:
            logger.warning(f"⚠️  {success_count}/{total_count} components initialized successfully")
        
        return results
    
    async def initialize_redis(self, retry_count: int = 3) -> bool:
        """Initialize Redis with retry logic."""
        if 'redis' in self.initialized_components:
            logger.debug("Redis already initialized")
            return True
        
        logger.info("🔗 Initializing Redis connection...")
        
        for attempt in range(retry_count):
            try:
                from .redis import init_redis, get_redis
                
                await init_redis()
                
                # Test connection
                redis_client = get_redis()
                await redis_client.ping()
                
                self.redis_client = redis_client
                self.initialized_components.add('redis')
                
                logger.info("✅ Redis initialized successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Redis initialization attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ Redis initialization failed after {retry_count} attempts")
                    return False
        
        return False
    
    async def initialize_database(self, retry_count: int = 3) -> bool:
        """Initialize database with retry logic."""
        if 'database' in self.initialized_components:
            logger.debug("Database already initialized")
            return True
        
        logger.info("🔌 Initializing database connection...")
        
        for attempt in range(retry_count):
            try:
                from .database import init_database, get_session
                from sqlalchemy import text
                
                await init_database()
                
                # Test connection
                async with get_session() as session:
                    await session.execute(text("SELECT 1"))
                
                self.initialized_components.add('database')
                
                logger.info("✅ Database initialized successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Database initialization attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ Database initialization failed after {retry_count} attempts")
                    return False
        
        return False
    
    async def shutdown_all(self) -> Dict[str, bool]:
        """Gracefully shutdown all initialized components."""
        results = {}
        
        logger.info("Shutting down system components...")
        
        if 'redis' in self.initialized_components:
            results['redis'] = await self.shutdown_redis()
        
        if 'database' in self.initialized_components:
            results['database'] = await self.shutdown_database()
        
        self.initialized_components.clear()
        
        return results
    
    async def shutdown_redis(self) -> bool:
        """Shutdown Redis connections."""
        try:
            from .redis import close_redis
            await close_redis()
            
            logger.info("✅ Redis connections closed")
            return True
        except Exception as e:
            logger.error(f"❌ Redis shutdown error: {e}")
            return False
    
    async def shutdown_database(self) -> bool:
        """Shutdown database connections."""
        try:
            from .database import close_database
            await close_database()
            
            logger.info("✅ Database connections closed")
            return True
        except Exception as e:
            logger.error(f"❌ Database shutdown error: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        return {
            'initialized_components': list(self.initialized_components),
            'redis_available': 'redis' in self.initialized_components,
            'database_available': 'database' in self.initialized_components,
            'initialization_errors': self.initialization_errors
        }


# Global initializer instance
_global_initializer: Optional[SystemInitializer] = None


def get_system_initializer() -> SystemInitializer:
    """Get or create the global system initializer."""
    global _global_initializer
    if _global_initializer is None:
        _global_initializer = SystemInitializer()
    return _global_initializer


async def initialize_system(components: Optional[list] = None) -> Dict[str, bool]:
    """Convenience function to initialize system components."""
    initializer = get_system_initializer()
    return await initializer.initialize_all(components)


async def shutdown_system() -> Dict[str, bool]:
    """Convenience function to shutdown system components."""
    initializer = get_system_initializer()
    return await initializer.shutdown_all()


@asynccontextmanager
async def managed_system(components: Optional[list] = None):
    """Context manager for system initialization and cleanup."""
    initializer = get_system_initializer()
    
    try:
        # Initialize
        init_results = await initializer.initialize_all(components)
        yield init_results
    finally:
        # Cleanup
        await initializer.shutdown_all()


async def ensure_initialized(components: Optional[list] = None) -> bool:
    """Ensure system components are initialized, initializing if needed."""
    initializer = get_system_initializer()
    
    components = components or ['redis', 'database']
    missing_components = [c for c in components if c not in initializer.initialized_components]
    
    if not missing_components:
        return True
    
    results = await initializer.initialize_all(missing_components)
    return all(results.values())


def get_initialization_status() -> Dict[str, Any]:
    """Get current initialization status."""
    initializer = get_system_initializer()
    return initializer.get_health_status()


class RequiresInitialization:
    """Decorator to ensure components are initialized before function execution."""
    
    def __init__(self, components: Optional[list] = None):
        self.components = components or ['redis', 'database']
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            success = await ensure_initialized(self.components)
            if not success:
                raise InitializationError(f"Failed to initialize required components: {self.components}")
            return await func(*args, **kwargs)
        return wrapper


# Convenience decorators
requires_redis = RequiresInitialization(['redis'])
requires_database = RequiresInitialization(['database'])
requires_all = RequiresInitialization(['redis', 'database'])