#!/usr/bin/env python3
"""
Script to fix SQLite/PostgreSQL compatibility issues in model files.

Converts PostgreSQL-specific ARRAY and UUID types to database-agnostic types.
"""

import os
import re
from pathlib import Path

def fix_model_file(file_path: Path):
    """Fix a single model file for database compatibility."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Add import for database_types if not present
    if 'from ..core.database_types import' not in content and 'from app.core.database_types import' not in content:
        # Find the import section - try different patterns
        if 'from ..core.database import Base' in content:
            import_pattern = r'(from \.\.core\.database import Base)'
            replacement = r'\1\nfrom ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray'
            content = re.sub(import_pattern, replacement, content)
        elif 'from app.core.database import Base' in content:
            import_pattern = r'(from app\.core\.database import Base)'
            replacement = r'\1\nfrom app.core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray'
            content = re.sub(import_pattern, replacement, content)
    
    # Remove PostgreSQL-specific imports that we're replacing
    content = re.sub(r'from sqlalchemy\.dialects\.postgresql import[^\n]*ARRAY[^\n]*\n', '', content)
    content = re.sub(r'from sqlalchemy\.dialects\.postgresql import[^\n]*UUID[^\n]*\n', '', content)
    
    # Replace old-style Column syntax
    content = re.sub(r'Column\(UUID\([^)]*\)', 'Column(DatabaseAgnosticUUID()', content)
    content = re.sub(r'Column\(ARRAY\(String\)', 'Column(StringArray()', content)
    content = re.sub(r'Column\(ARRAY\(UUID\([^)]*\)\)', 'Column(UUIDArray()', content)
    
    # Replace new-style mapped_column syntax
    content = re.sub(r'mapped_column\(\s*UUID\([^)]*\)', 'mapped_column(DatabaseAgnosticUUID()', content)
    content = re.sub(r'mapped_column\(\s*ARRAY\(String\)', 'mapped_column(StringArray()', content)  
    content = re.sub(r'mapped_column\(\s*ARRAY\(UUID\([^)]*\)\)', 'mapped_column(UUIDArray()', content)
    
    # Fix any remaining bare UUID references in mapped_column
    content = re.sub(r'mapped_column\(\s*UUID\)', 'mapped_column(DatabaseAgnosticUUID())', content)
    
    # Fix multiline mapped_column patterns
    content = re.sub(r'mapped_column\(\s*\n\s*UUID\([^)]*\)', 'mapped_column(\n        DatabaseAgnosticUUID()', content)
    
    # Fix standalone UUID references (commonly found in multiline contexts)
    content = re.sub(r'\n\s+UUID\(as_uuid=True\),?\s*\n', '\n        DatabaseAgnosticUUID(),\n', content)
    
    # Clean up any remaining PostgreSQL imports that might be empty
    content = re.sub(r'from sqlalchemy\.dialects\.postgresql import\s*\n', '', content)
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {file_path}")
        return True
    else:
        print(f"  - No changes needed for {file_path}")
        return False

def main():
    """Fix all model files."""
    models_dir = Path("app/models")
    
    if not models_dir.exists():
        print("Models directory not found!")
        return
    
    model_files = list(models_dir.glob("*.py"))
    if "__init__.py" in [f.name for f in model_files]:
        model_files = [f for f in model_files if f.name != "__init__.py"]
    
    print(f"Found {len(model_files)} model files to process...")
    
    updated_count = 0
    for model_file in model_files:
        if fix_model_file(model_file):
            updated_count += 1
    
    print(f"\nCompleted! Updated {updated_count} out of {len(model_files)} files.")

if __name__ == "__main__":
    main()