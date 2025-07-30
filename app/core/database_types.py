"""
Database type compatibility layer for LeanVibe Agent Hive 2.0.

Provides database-agnostic types that work with both SQLite and PostgreSQL.
"""

import json
from typing import Any, List, Optional, Union
from sqlalchemy import TypeDecorator, Text, String
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import sqltypes
from sqlalchemy.engine import Dialect


class JSONEncodedList(TypeDecorator):
    """
    A type that stores a list as JSON in SQLite and uses native ARRAY in PostgreSQL.
    
    This provides cross-database compatibility for array fields.
    """
    
    impl = Text
    cache_ok = True
    
    def __init__(self, item_type=None):
        """
        Initialize the JSONEncodedList type.
        
        Args:
            item_type: The type of items in the list (e.g., String, UUID)
        """
        self.item_type = item_type
        super().__init__()
    
    def load_dialect_impl(self, dialect: Dialect):
        """Load the appropriate implementation for the current dialect."""
        if dialect.name == 'postgresql':
            # Use native PostgreSQL ARRAY type
            if self.item_type:
                return dialect.type_descriptor(PG_ARRAY(self.item_type))
            else:
                return dialect.type_descriptor(PG_ARRAY(String))
        else:
            # Use JSON storage for SQLite and other databases
            return dialect.type_descriptor(Text())
    
    def process_bind_param(self, value: Optional[List[Any]], dialect: Dialect) -> Optional[Union[List[Any], str]]:
        """Process the value before sending to the database."""
        if value is None:
            return None
        
        if dialect.name == 'postgresql':
            # PostgreSQL can handle lists directly
            return value
        else:
            # Convert to JSON string for SQLite
            return json.dumps(value)
    
    def process_result_value(self, value: Optional[Union[List[Any], str]], dialect: Dialect) -> Optional[List[Any]]:
        """Process the value after receiving from the database."""
        if value is None:
            return []
        
        if dialect.name == 'postgresql':
            # PostgreSQL returns the list directly
            return value if isinstance(value, list) else []
        else:
            # Parse JSON string for SQLite
            try:
                if isinstance(value, str):
                    return json.loads(value)
                else:
                    return value if isinstance(value, list) else []
            except (json.JSONDecodeError, TypeError):
                return []


class DatabaseAgnosticUUID(TypeDecorator):
    """
    A UUID type that works with both SQLite and PostgreSQL.
    
    Uses native UUID type in PostgreSQL and String(36) in SQLite.
    """
    
    impl = String
    cache_ok = True
    
    def load_dialect_impl(self, dialect: Dialect):
        """Load the appropriate implementation for the current dialect."""
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))
    
    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[str]:
        """Process the value before sending to the database."""
        if value is None:
            return None
        
        if dialect.name == 'postgresql':
            return value  # PostgreSQL handles UUID objects
        else:
            return str(value)  # Convert to string for SQLite
    
    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        """Process the value after receiving from the database."""
        if value is None:
            return None
        
        if dialect.name == 'postgresql':
            return value  # PostgreSQL returns UUID objects
        else:
            # Return string for SQLite (can be converted to UUID later if needed)
            return value


# Convenience functions for creating array types
def StringArray() -> JSONEncodedList:
    """Create a string array type that works across databases."""
    return JSONEncodedList(item_type=String)


def UUIDArray() -> JSONEncodedList:
    """Create a UUID array type that works across databases."""
    return JSONEncodedList(item_type=DatabaseAgnosticUUID())


def IntegerArray() -> JSONEncodedList:
    """Create an integer array type that works across databases."""
    return JSONEncodedList(item_type=sqltypes.Integer)


def TextArray() -> JSONEncodedList:
    """Create a text array type that works across databases."""
    return JSONEncodedList(item_type=Text)


class DatabaseAgnosticIPAddress(TypeDecorator):
    """
    An IP address type that works with both SQLite and PostgreSQL.
    
    Uses native INET type in PostgreSQL and String(45) in SQLite.
    String(45) is sufficient for both IPv4 and IPv6 addresses.
    """
    
    impl = String
    cache_ok = True
    
    def load_dialect_impl(self, dialect: Dialect):
        """Load the appropriate implementation for the current dialect."""
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import INET
            return dialect.type_descriptor(INET)
        else:
            return dialect.type_descriptor(String(45))
    
    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[str]:
        """Process the value before sending to the database."""
        if value is None:
            return None
        
        # Both PostgreSQL and SQLite can handle string IP addresses
        return str(value)
    
    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        """Process the value after receiving from the database."""
        if value is None:
            return None
        
        # Return as string for compatibility
        return str(value)