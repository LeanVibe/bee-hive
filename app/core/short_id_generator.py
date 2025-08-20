"""
Human-Friendly Short ID Generation System for LeanVibe Agent Hive 2.0

A collision-resistant, hierarchical ID system that provides both human-friendly
short IDs and backing UUID system for global uniqueness. Designed for CLI commands
like `hive task show TSK-A7B2` while maintaining database performance.

Features:
- Hierarchical prefixes for different entity types
- Base32 encoding for human readability (excludes 0, 1, I, O)
- Collision detection with automatic retry
- Partial ID matching (like Git commits)
- Database-agnostic UUID backing
- Efficient caching and indexing
"""

import uuid
import base64
import hashlib
import secrets
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

import logging
from sqlalchemy import Column, String, DateTime, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Human-friendly Base32 alphabet (removes 0, 1, I, O for clarity)
# Uses Crockford's Base32: 0123456789ABCDEFGHJKMNPQRSTVWXYZ (but we exclude 0,1)
HUMAN_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
ALPHABET_SIZE = len(HUMAN_ALPHABET)

class EntityType(Enum):
    """Entity types with hierarchical prefixes for different Hive entities."""
    PROJECT = "PRJ"      # Projects
    EPIC = "EPC"         # Epics  
    PRD = "PRD"          # Product Requirements Documents
    TASK = "TSK"         # Individual tasks
    AGENT = "AGT"        # Agents
    WORKFLOW = "WFL"     # Workflows
    FILE = "FIL"         # File entries in project index
    DEPENDENCY = "DEP"   # Dependency relationships
    SNAPSHOT = "SNP"     # Index snapshots
    SESSION = "SES"      # Analysis sessions
    DEBT = "DBT"         # Technical debt items
    PLAN = "PLN"         # Remediation plans


@dataclass
class ShortIdConfig:
    """Configuration for short ID generation."""
    
    # ID format: PREFIX-XXXX (e.g., TSK-A7B2)
    prefix_length: int = 3          # Length of entity prefix
    code_length: int = 4            # Length of random code
    separator: str = "-"            # Separator between prefix and code
    max_retries: int = 5            # Max collision retry attempts
    
    # Caching and performance
    cache_size: int = 10000         # Size of ID cache
    batch_generate_size: int = 100  # Batch generation size
    
    # Database settings
    index_prefix: str = "short_id"  # Database index prefix


@dataclass  
class IdGenerationStats:
    """Statistics for ID generation performance tracking."""
    
    generated_count: int = 0
    collision_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_generation_time_ms: float = 0.0
    last_generation_time: Optional[datetime] = None


class ShortIdGenerator:
    """
    Human-friendly short ID generator with collision detection.
    
    This class generates collision-resistant short IDs while maintaining
    a backing UUID system for global uniqueness and database performance.
    """
    
    def __init__(self, config: ShortIdConfig = None, db_session: Session = None):
        """
        Initialize the short ID generator.
        
        Args:
            config: Configuration for ID generation behavior
            db_session: Database session for collision checking
        """
        self.config = config or ShortIdConfig()
        self.db_session = db_session
        
        # Statistics and caching
        self.stats = IdGenerationStats()
        self._cache: Dict[str, uuid.UUID] = {}
        self._reverse_cache: Dict[uuid.UUID, str] = {}
        self._collision_cache: Set[str] = set()
        
        # Performance optimization
        self._last_prefix_count: Dict[str, int] = defaultdict(int)
        self._generation_times: List[float] = []
        
        logger.info(f"ShortIdGenerator initialized with config: {self.config}")

    def generate_id(self, entity_type: EntityType, backing_uuid: uuid.UUID = None) -> Tuple[str, uuid.UUID]:
        """
        Generate a human-friendly short ID with UUID backing.
        
        Args:
            entity_type: The type of entity for prefix selection
            backing_uuid: Optional existing UUID to use
            
        Returns:
            Tuple of (short_id, uuid) like ("TSK-A7B2", uuid.UUID(...))
            
        Raises:
            ValueError: If unable to generate unique ID after max retries
        """
        start_time = time.time()
        
        if backing_uuid is None:
            backing_uuid = uuid.uuid4()
            
        prefix = entity_type.value
        short_id = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Generate candidate ID
                code = self._generate_code(backing_uuid, attempt)
                candidate_id = f"{prefix}{self.config.separator}{code}"
                
                # Check for collisions
                if not self._has_collision(candidate_id):
                    short_id = candidate_id
                    break
                    
                self.stats.collision_count += 1
                logger.debug(f"Collision detected for {candidate_id}, attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Error generating ID on attempt {attempt + 1}: {e}")
                continue
        
        if short_id is None:
            raise ValueError(f"Unable to generate unique ID for {entity_type} after {self.config.max_retries} attempts")
        
        # Update caches and statistics
        self._cache[short_id] = backing_uuid
        self._reverse_cache[backing_uuid] = short_id
        self.stats.generated_count += 1
        self.stats.last_generation_time = datetime.now(timezone.utc)
        
        # Track timing
        generation_time = (time.time() - start_time) * 1000
        self._generation_times.append(generation_time)
        if len(self._generation_times) > 100:  # Keep last 100 times
            self._generation_times.pop(0)
        
        self.stats.average_generation_time_ms = sum(self._generation_times) / len(self._generation_times)
        
        logger.debug(f"Generated short ID: {short_id} -> {backing_uuid} in {generation_time:.2f}ms")
        return short_id, backing_uuid

    def _generate_code(self, backing_uuid: uuid.UUID, attempt: int) -> str:
        """
        Generate the random code portion of the short ID.
        
        Uses a combination of UUID bytes, current timestamp, and attempt counter
        to ensure uniqueness while maintaining deterministic properties.
        
        Args:
            backing_uuid: The UUID to base the code on
            attempt: Current collision retry attempt
            
        Returns:
            Human-friendly code string (e.g., "A7B2")
        """
        # Create deterministic but unique seed
        uuid_bytes = backing_uuid.bytes
        timestamp_bytes = int(time.time() * 1000000).to_bytes(8, 'big')  # microsecond precision
        attempt_bytes = attempt.to_bytes(2, 'big')
        
        # Combine all sources
        combined = uuid_bytes + timestamp_bytes + attempt_bytes
        hash_digest = hashlib.sha256(combined).digest()
        
        # Convert to human-friendly alphabet
        code = ""
        value = int.from_bytes(hash_digest[:8], 'big')  # Use first 8 bytes
        
        for _ in range(self.config.code_length):
            code = HUMAN_ALPHABET[value % ALPHABET_SIZE] + code
            value //= ALPHABET_SIZE
            
        return code

    def _has_collision(self, candidate_id: str) -> bool:
        """
        Check if a candidate ID collides with existing IDs.
        
        Args:
            candidate_id: The ID to check for collision
            
        Returns:
            True if collision detected, False otherwise
        """
        # Check local collision cache first
        if candidate_id in self._collision_cache:
            return True
            
        # Check in-memory cache
        if candidate_id in self._cache:
            return True
            
        # Check database if session available
        if self.db_session:
            # This would need to be implemented based on your specific table structure
            # For now, we'll assume no database collision
            # In real implementation, you'd query the short_id_mappings table
            pass
            
        return False

    def resolve_id(self, short_or_uuid: Union[str, uuid.UUID]) -> Tuple[str, uuid.UUID]:
        """
        Resolve either a short ID or UUID to both formats.
        
        Args:
            short_or_uuid: Either a short ID string or UUID
            
        Returns:
            Tuple of (short_id, uuid)
            
        Raises:
            ValueError: If ID cannot be resolved
        """
        if isinstance(short_or_uuid, str):
            # Resolve short ID to UUID
            if short_or_uuid in self._cache:
                self.stats.cache_hits += 1
                return short_or_uuid, self._cache[short_or_uuid]
            
            self.stats.cache_misses += 1
            # Would query database here in real implementation
            raise ValueError(f"Short ID not found: {short_or_uuid}")
            
        elif isinstance(short_or_uuid, uuid.UUID):
            # Resolve UUID to short ID
            if short_or_uuid in self._reverse_cache:
                self.stats.cache_hits += 1
                return self._reverse_cache[short_or_uuid], short_or_uuid
            
            self.stats.cache_misses += 1
            # Would query database here in real implementation
            raise ValueError(f"UUID not found: {short_or_uuid}")
            
        else:
            raise ValueError(f"Invalid ID type: {type(short_or_uuid)}")

    def partial_match(self, partial_id: str, entity_type: EntityType = None) -> List[Tuple[str, uuid.UUID]]:
        """
        Find IDs that match a partial short ID (like Git commit matching).
        
        Args:
            partial_id: Partial ID to match (e.g., "TSK-A7" matches "TSK-A7B2")
            entity_type: Optional entity type to filter by
            
        Returns:
            List of matching (short_id, uuid) tuples
        """
        matches = []
        
        # Search in cache
        for short_id, backing_uuid in self._cache.items():
            if short_id.startswith(partial_id):
                if entity_type is None or short_id.startswith(entity_type.value):
                    matches.append((short_id, backing_uuid))
        
        # In real implementation, would also search database
        
        return matches

    def extract_entity_type(self, short_id: str) -> Optional[EntityType]:
        """
        Extract entity type from a short ID.
        
        Args:
            short_id: The short ID to analyze
            
        Returns:
            EntityType if found, None otherwise
        """
        if self.config.separator not in short_id:
            return None
            
        prefix = short_id.split(self.config.separator)[0]
        
        for entity_type in EntityType:
            if entity_type.value == prefix:
                return entity_type
                
        return None

    def batch_generate(self, entity_type: EntityType, count: int) -> List[Tuple[str, uuid.UUID]]:
        """
        Generate multiple IDs efficiently in batch.
        
        Args:
            entity_type: Entity type for all generated IDs
            count: Number of IDs to generate
            
        Returns:
            List of (short_id, uuid) tuples
        """
        results = []
        
        for _ in range(count):
            short_id, backing_uuid = self.generate_id(entity_type)
            results.append((short_id, backing_uuid))
            
        logger.info(f"Batch generated {count} IDs for {entity_type}")
        return results

    def get_stats(self) -> IdGenerationStats:
        """Get current generation statistics."""
        return self.stats

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._reverse_cache.clear() 
        self._collision_cache.clear()
        logger.info("Caches cleared")

    def validate_id_format(self, short_id: str) -> bool:
        """
        Validate that a short ID follows the correct format.
        
        Args:
            short_id: The ID to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        if not short_id:
            return False
            
        parts = short_id.split(self.config.separator)
        if len(parts) != 2:
            return False
            
        prefix, code = parts
        
        # Validate prefix
        if len(prefix) != self.config.prefix_length:
            return False
            
        # Validate code length and characters
        if len(code) != self.config.code_length:
            return False
            
        if not all(c in HUMAN_ALPHABET for c in code):
            return False
            
        # Validate prefix is a known entity type
        if not any(et.value == prefix for et in EntityType):
            return False
            
        return True


class ShortIdDatabaseMixin:
    """
    Database mixin to add short ID support to existing models.
    
    This mixin can be added to any SQLAlchemy model to provide
    short ID functionality alongside existing UUID primary keys.
    """
    
    # Short ID column (indexed for fast lookups)
    short_id = Column(
        String(20),
        nullable=False,
        unique=True,
        index=True,
        comment="Human-friendly short identifier"
    )
    
    # Metadata for tracking
    short_id_generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When the short ID was generated"
    )
    
    def generate_short_id(self, entity_type: EntityType, generator: ShortIdGenerator = None) -> str:
        """
        Generate and assign a short ID for this entity.
        
        Args:
            entity_type: The type of entity for prefix selection
            generator: Optional generator instance
            
        Returns:
            The generated short ID
        """
        if generator is None:
            generator = ShortIdGenerator()
            
        short_id, _ = generator.generate_id(entity_type, self.id)
        self.short_id = short_id
        self.short_id_generated_at = datetime.now(timezone.utc)
        
        return short_id
    
    @classmethod
    def find_by_short_id(cls, short_id: str, session: Session):
        """
        Find entity by short ID.
        
        Args:
            short_id: The short ID to search for
            session: Database session
            
        Returns:
            Entity instance if found, None otherwise
        """
        return session.query(cls).filter(cls.short_id == short_id).first()
    
    @classmethod  
    def find_by_partial_short_id(cls, partial_id: str, session: Session, limit: int = 10):
        """
        Find entities by partial short ID match.
        
        Args:
            partial_id: Partial short ID to match
            session: Database session
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        return session.query(cls).filter(
            cls.short_id.like(f"{partial_id}%")
        ).limit(limit).all()


def create_short_id_mapping_table():
    """
    SQL for creating the short_id_mappings table for database storage.
    
    This table provides a central mapping between short IDs and UUIDs
    across all entity types in the system.
    """
    return """
    CREATE TABLE IF NOT EXISTS short_id_mappings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        short_id VARCHAR(20) NOT NULL UNIQUE,
        entity_uuid UUID NOT NULL,
        entity_type VARCHAR(10) NOT NULL,
        entity_table VARCHAR(50) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        
        -- Indexes for fast lookups
        CONSTRAINT unique_short_id UNIQUE (short_id),
        CONSTRAINT unique_entity_uuid UNIQUE (entity_uuid, entity_type)
    );
    
    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_short_id_mappings_short_id ON short_id_mappings (short_id);
    CREATE INDEX IF NOT EXISTS idx_short_id_mappings_entity_uuid ON short_id_mappings (entity_uuid);
    CREATE INDEX IF NOT EXISTS idx_short_id_mappings_entity_type ON short_id_mappings (entity_type);
    CREATE INDEX IF NOT EXISTS idx_short_id_mappings_created_at ON short_id_mappings (created_at);
    
    -- Trigger to update updated_at timestamp
    CREATE OR REPLACE FUNCTION update_short_id_mappings_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    CREATE TRIGGER update_short_id_mappings_updated_at
        BEFORE UPDATE ON short_id_mappings
        FOR EACH ROW
        EXECUTE FUNCTION update_short_id_mappings_updated_at();
    """


# Global generator instance for easy access
_global_generator: Optional[ShortIdGenerator] = None


def get_generator(db_session: Session = None) -> ShortIdGenerator:
    """
    Get the global short ID generator instance.
    
    Args:
        db_session: Optional database session for collision checking
        
    Returns:
        ShortIdGenerator instance
    """
    global _global_generator
    
    if _global_generator is None:
        _global_generator = ShortIdGenerator(db_session=db_session)
        
    return _global_generator


def set_global_generator(generator: ShortIdGenerator) -> None:
    """
    Set the global short ID generator instance.
    
    Args:
        generator: The generator instance to use globally
    """
    global _global_generator
    _global_generator = generator


# Convenience functions for common operations
def generate_short_id(entity_type: EntityType, backing_uuid: uuid.UUID = None, 
                     db_session: Session = None) -> Tuple[str, uuid.UUID]:
    """
    Convenience function to generate a short ID.
    
    Args:
        entity_type: The type of entity
        backing_uuid: Optional UUID to use
        db_session: Optional database session
        
    Returns:
        Tuple of (short_id, uuid)
    """
    generator = get_generator(db_session)
    return generator.generate_id(entity_type, backing_uuid)


def resolve_short_id(short_or_uuid: Union[str, uuid.UUID], 
                    db_session: Session = None) -> Tuple[str, uuid.UUID]:
    """
    Convenience function to resolve a short ID or UUID.
    
    Args:
        short_or_uuid: Either short ID or UUID
        db_session: Optional database session
        
    Returns:
        Tuple of (short_id, uuid)
    """
    generator = get_generator(db_session)
    return generator.resolve_id(short_or_uuid)


def validate_short_id_format(short_id: str) -> bool:
    """
    Convenience function to validate short ID format.
    
    Args:
        short_id: The ID to validate
        
    Returns:
        True if format is valid
    """
    generator = get_generator()
    return generator.validate_id_format(short_id)