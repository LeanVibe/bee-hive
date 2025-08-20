"""
Short ID Model Mixin for LeanVibe Agent Hive 2.0

This module provides a mixin class that can be added to any existing
SQLAlchemy model to provide short ID functionality alongside UUIDs.

The mixin automatically:
- Adds short_id and metadata columns
- Provides resolution methods
- Handles automatic generation via database triggers
- Supports partial matching and validation
"""

import uuid
from datetime import datetime
from typing import List, Optional, Union, Tuple, Dict, Any

from sqlalchemy import Column, String, DateTime, event, inspect
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..core.short_id_generator import (
    ShortIdGenerator, EntityType, get_generator, 
    validate_short_id_format, resolve_short_id
)

import logging

logger = logging.getLogger(__name__)


class ShortIdMixin:
    """
    Mixin to add short ID support to any SQLAlchemy model.
    
    This mixin adds:
    - short_id column with unique constraint
    - short_id_generated_at timestamp
    - Methods for ID resolution and validation
    - Automatic generation support
    
    Usage:
        class Task(Base, ShortIdMixin):
            __tablename__ = 'tasks'
            ENTITY_TYPE = EntityType.TASK
            
            id = Column(UUID, primary_key=True)
            # ... other columns
    """
    
    # Short ID columns - these will be added to the inheriting model
    short_id = Column(
        String(20),
        nullable=True,  # Nullable to support migration from existing data
        unique=True,
        index=True,
        comment="Human-friendly short identifier"
    )
    
    short_id_generated_at = Column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
        comment="When the short ID was generated"
    )
    
    # Entity type must be defined in inheriting classes
    ENTITY_TYPE: EntityType = None
    
    def generate_short_id(self, generator: ShortIdGenerator = None, force: bool = False) -> str:
        """
        Generate and assign a short ID for this entity.
        
        Args:
            generator: Optional generator instance
            force: Force regeneration even if short_id exists
            
        Returns:
            The generated short ID
            
        Raises:
            ValueError: If ENTITY_TYPE not defined or generation fails
        """
        if not force and self.short_id:
            return self.short_id
            
        if not self.ENTITY_TYPE:
            raise ValueError(f"ENTITY_TYPE must be defined in {self.__class__.__name__}")
        
        if generator is None:
            generator = get_generator()
            
        try:
            short_id, _ = generator.generate_id(self.ENTITY_TYPE, self.id)
            self.short_id = short_id
            self.short_id_generated_at = datetime.utcnow()
            
            logger.debug(f"Generated short ID {short_id} for {self.__class__.__name__} {self.id}")
            return short_id
            
        except Exception as e:
            logger.error(f"Failed to generate short ID for {self.__class__.__name__} {self.id}: {e}")
            raise
    
    def ensure_short_id(self, session: Session = None, generator: ShortIdGenerator = None) -> str:
        """
        Ensure this entity has a short ID, generating one if needed.
        
        Args:
            session: Optional database session for persistence
            generator: Optional generator instance
            
        Returns:
            The short ID (existing or newly generated)
        """
        if not self.short_id:
            self.generate_short_id(generator)
            
            if session:
                try:
                    session.flush()  # Flush to get any database errors early
                except IntegrityError as e:
                    logger.warning(f"Short ID collision detected, regenerating: {e}")
                    session.rollback()
                    self.generate_short_id(generator, force=True)
                    session.flush()
        
        return self.short_id
    
    @classmethod
    def find_by_short_id(cls, short_id: str, session: Session) -> Optional['ShortIdMixin']:
        """
        Find entity by exact short ID match.
        
        Args:
            short_id: The short ID to search for
            session: Database session
            
        Returns:
            Entity instance if found, None otherwise
        """
        if not validate_short_id_format(short_id):
            return None
            
        return session.query(cls).filter(cls.short_id == short_id).first()
    
    @classmethod
    def find_by_partial_short_id(cls, partial_id: str, session: Session, 
                                 limit: int = 10) -> List['ShortIdMixin']:
        """
        Find entities by partial short ID match.
        
        Args:
            partial_id: Partial short ID to match
            session: Database session
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if not partial_id:
            return []
        
        # Clean and uppercase the input
        partial_id = partial_id.strip().upper()
        
        # Direct prefix match
        query = session.query(cls).filter(
            cls.short_id.like(f"{partial_id}%")
        )
        
        # If we have an entity type and the partial doesn't include prefix,
        # also try matching with the prefix
        if hasattr(cls, 'ENTITY_TYPE') and cls.ENTITY_TYPE and '-' not in partial_id:
            prefix = cls.ENTITY_TYPE.value
            prefixed_partial = f"{prefix}-{partial_id}"
            query = query.union(
                session.query(cls).filter(
                    cls.short_id.like(f"{prefixed_partial}%")
                )
            )
        
        return query.limit(limit).all()
    
    @classmethod
    def resolve_id_input(cls, id_input: Union[str, uuid.UUID], session: Session) -> Optional['ShortIdMixin']:
        """
        Resolve various ID input formats to entity instance.
        
        Args:
            id_input: Short ID, partial short ID, or UUID
            session: Database session
            
        Returns:
            Entity instance if found, None if not found
            
        Raises:
            ValueError: If input is ambiguous (multiple matches)
        """
        if not id_input:
            return None
        
        # Handle UUID input
        if isinstance(id_input, uuid.UUID):
            return session.query(cls).filter(cls.id == id_input).first()
        
        # Handle string input
        id_input = str(id_input).strip()
        
        # Try UUID parsing
        try:
            uuid_obj = uuid.UUID(id_input)
            return session.query(cls).filter(cls.id == uuid_obj).first()
        except ValueError:
            pass
        
        # Try exact short ID match
        if validate_short_id_format(id_input):
            entity = cls.find_by_short_id(id_input, session)
            if entity:
                return entity
        
        # Try partial matching
        matches = cls.find_by_partial_short_id(id_input, session)
        
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            # Multiple matches - this is ambiguous
            match_ids = [m.short_id for m in matches]
            raise ValueError(f"Ambiguous ID '{id_input}' matches multiple entities: {match_ids}")
    
    def to_dict_with_short_id(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary including short ID information.
        
        Returns:
            Dictionary with short ID fields included
        """
        # Get base dictionary (assumes model has to_dict method)
        if hasattr(super(), 'to_dict'):
            data = super().to_dict()
        else:
            # Fallback to basic serialization
            data = {}
            for column in inspect(self.__class__).columns:
                value = getattr(self, column.name)
                if isinstance(value, (datetime, uuid.UUID)):
                    data[column.name] = str(value) if value else None
                else:
                    data[column.name] = value
        
        # Add short ID information
        data.update({
            'short_id': self.short_id,
            'short_id_generated_at': self.short_id_generated_at.isoformat() if self.short_id_generated_at else None,
            'entity_type': self.ENTITY_TYPE.name if self.ENTITY_TYPE else None
        })
        
        return data
    
    def get_display_id(self) -> str:
        """
        Get the preferred display ID (short ID if available, UUID otherwise).
        
        Returns:
            Short ID if available, otherwise UUID string
        """
        return self.short_id or str(self.id)
    
    @property
    def has_short_id(self) -> bool:
        """Check if this entity has a short ID."""
        return self.short_id is not None
    
    def validate_short_id(self) -> bool:
        """
        Validate that this entity's short ID follows correct format.
        
        Returns:
            True if short ID is valid or None, False if invalid
        """
        if not self.short_id:
            return True  # None/empty is considered valid
            
        return validate_short_id_format(self.short_id)
    
    def get_entity_type_from_short_id(self) -> Optional[EntityType]:
        """
        Extract entity type from this entity's short ID.
        
        Returns:
            EntityType if extractable, None otherwise
        """
        if not self.short_id:
            return None
            
        generator = get_generator()
        return generator.extract_entity_type(self.short_id)
    
    def __str__(self) -> str:
        """String representation using short ID when available."""
        class_name = self.__class__.__name__
        display_id = self.get_display_id()
        return f"{class_name}({display_id})"
    
    def __repr__(self) -> str:
        """Detailed representation including both IDs."""
        class_name = self.__class__.__name__
        short_id = self.short_id or "None"
        uuid_str = str(self.id) if hasattr(self, 'id') else "None"
        return f"{class_name}(short_id='{short_id}', id='{uuid_str}')"


# Event listeners for automatic short ID generation

@event.listens_for(ShortIdMixin, 'before_insert', propagate=True)
def generate_short_id_before_insert(mapper, connection, target):
    """
    Automatically generate short ID before insert if not already set.
    
    This is a fallback for when database triggers are not available
    or when working with in-memory operations.
    """
    if not hasattr(target, 'short_id') or target.short_id:
        return  # Skip if no short_id column or already set
    
    if not hasattr(target, 'ENTITY_TYPE') or not target.ENTITY_TYPE:
        logger.warning(f"No ENTITY_TYPE defined for {target.__class__.__name__}, skipping auto-generation")
        return
    
    try:
        target.generate_short_id()
        logger.debug(f"Auto-generated short ID {target.short_id} for {target.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to auto-generate short ID for {target.__class__.__name__}: {e}")
        # Don't raise here as it would break the insert


def add_short_id_to_existing_model(model_class, entity_type: EntityType):
    """
    Utility function to add short ID support to an existing model class.
    
    This is useful for models that can't directly inherit from ShortIdMixin.
    
    Args:
        model_class: The SQLAlchemy model class to enhance
        entity_type: The EntityType for this model
        
    Usage:
        # For existing Task model
        add_short_id_to_existing_model(Task, EntityType.TASK)
    """
    # Add the entity type
    model_class.ENTITY_TYPE = entity_type
    
    # Add the mixin methods
    for attr_name in dir(ShortIdMixin):
        if not attr_name.startswith('_') and callable(getattr(ShortIdMixin, attr_name)):
            attr = getattr(ShortIdMixin, attr_name)
            setattr(model_class, attr_name, attr)
    
    # Add the columns if they don't exist
    if not hasattr(model_class, 'short_id'):
        model_class.short_id = ShortIdMixin.short_id
        model_class.short_id_generated_at = ShortIdMixin.short_id_generated_at
    
    logger.info(f"Added short ID support to {model_class.__name__} with entity type {entity_type.name}")


# Utility functions for working with short ID models

def bulk_generate_short_ids(model_class, session: Session, 
                           batch_size: int = 100, generator: ShortIdGenerator = None) -> int:
    """
    Generate short IDs for all entities of a model that don't have them.
    
    Args:
        model_class: The model class with ShortIdMixin
        session: Database session
        batch_size: Number of entities to process per batch
        generator: Optional generator instance
        
    Returns:
        Number of short IDs generated
    """
    if not hasattr(model_class, 'ENTITY_TYPE') or not model_class.ENTITY_TYPE:
        raise ValueError(f"Model {model_class.__name__} must have ENTITY_TYPE defined")
    
    if generator is None:
        generator = get_generator()
    
    # Count entities without short IDs
    total_count = session.query(model_class).filter(
        model_class.short_id.is_(None)
    ).count()
    
    if total_count == 0:
        logger.info(f"All {model_class.__name__} entities already have short IDs")
        return 0
    
    logger.info(f"Generating short IDs for {total_count} {model_class.__name__} entities")
    
    generated_count = 0
    offset = 0
    
    while offset < total_count:
        # Get batch
        entities = session.query(model_class).filter(
            model_class.short_id.is_(None)
        ).offset(offset).limit(batch_size).all()
        
        if not entities:
            break
        
        # Generate short IDs for batch
        for entity in entities:
            try:
                entity.generate_short_id(generator)
                generated_count += 1
            except Exception as e:
                logger.error(f"Failed to generate short ID for {entity.id}: {e}")
                continue
        
        # Commit batch
        try:
            session.commit()
            logger.info(f"Generated short IDs for batch {offset//batch_size + 1}, "
                       f"total: {generated_count}/{total_count}")
        except Exception as e:
            logger.error(f"Failed to commit batch: {e}")
            session.rollback()
            # Skip this batch and continue
        
        offset += batch_size
    
    logger.info(f"Completed bulk generation: {generated_count} short IDs generated")
    return generated_count


def validate_model_short_ids(model_class, session: Session) -> Dict[str, Any]:
    """
    Validate all short IDs for a model class.
    
    Args:
        model_class: The model class to validate
        session: Database session
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_entities': 0,
        'entities_with_short_id': 0,
        'valid_short_ids': 0,
        'invalid_short_ids': 0,
        'invalid_entities': [],
        'duplicate_short_ids': []
    }
    
    # Get all entities
    entities = session.query(model_class).all()
    results['total_entities'] = len(entities)
    
    short_id_counts = {}
    
    for entity in entities:
        if entity.short_id:
            results['entities_with_short_id'] += 1
            
            # Check format validity
            if entity.validate_short_id():
                results['valid_short_ids'] += 1
            else:
                results['invalid_short_ids'] += 1
                results['invalid_entities'].append({
                    'id': str(entity.id),
                    'short_id': entity.short_id,
                    'error': 'Invalid format'
                })
            
            # Check for duplicates
            if entity.short_id in short_id_counts:
                short_id_counts[entity.short_id] += 1
            else:
                short_id_counts[entity.short_id] = 1
    
    # Find duplicates
    for short_id, count in short_id_counts.items():
        if count > 1:
            duplicate_entities = session.query(model_class).filter(
                model_class.short_id == short_id
            ).all()
            
            results['duplicate_short_ids'].append({
                'short_id': short_id,
                'count': count,
                'entity_ids': [str(e.id) for e in duplicate_entities]
            })
    
    return results