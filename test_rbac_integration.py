#!/usr/bin/env python3
"""
RBAC Integration Test for Epic 6 Phase 2

Tests the RBAC API endpoints to ensure they work correctly
with the LeanVibe Agent Hive 2.0 system.
"""

import asyncio
import httpx
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# Test user credentials (should match the default admin from auth.py)
TEST_ADMIN = {
    "email": "admin@leanvibe.com",
    "password": "AdminPassword123!"
}

async def test_rbac_endpoints():
    """Test RBAC API endpoints integration."""
    print("🔐 Testing RBAC Integration for Epic 6 Phase 2...")
    
    async with httpx.AsyncClient() as client:
        # Step 1: Login to get authentication token
        print("\n1️⃣ Testing authentication...")
        
        try:
            login_response = await client.post(
                f"{BASE_URL}/api/v1/auth/login",
                json=TEST_ADMIN
            )
            
            if login_response.status_code != 200:
                print(f"❌ Login failed: {login_response.status_code} - {login_response.text}")
                return
            
            auth_data = login_response.json()
            access_token = auth_data["access_token"]
            print(f"✅ Login successful - Token: {access_token[:20]}...")
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return
        
        # Step 2: Test RBAC Roles endpoint
        print("\n2️⃣ Testing RBAC roles endpoint...")
        
        try:
            roles_response = await client.get(
                f"{BASE_URL}/api/rbac/roles",
                headers=headers
            )
            
            print(f"Status: {roles_response.status_code}")
            
            if roles_response.status_code == 200:
                roles_data = roles_response.json()
                print(f"✅ Retrieved {len(roles_data)} roles")
                
                for role in roles_data[:3]:  # Show first 3 roles
                    print(f"   - {role['name']}: {len(role['permissions'])} permissions, {role['user_count']} users")
                    
            else:
                print(f"❌ Roles endpoint failed: {roles_response.text}")
                
        except Exception as e:
            print(f"❌ Roles endpoint error: {e}")
        
        # Step 3: Test Permissions endpoint
        print("\n3️⃣ Testing permissions endpoint...")
        
        try:
            permissions_response = await client.get(
                f"{BASE_URL}/api/rbac/permissions",
                headers=headers
            )
            
            print(f"Status: {permissions_response.status_code}")
            
            if permissions_response.status_code == 200:
                permissions_data = permissions_response.json()
                print(f"✅ Retrieved {len(permissions_data)} permissions")
                
                # Group by category
                categories = {}
                for perm in permissions_data:
                    cat = perm['category']
                    if cat not in categories:
                        categories[cat] = 0
                    categories[cat] += 1
                
                for category, count in categories.items():
                    print(f"   - {category}: {count} permissions")
                    
            else:
                print(f"❌ Permissions endpoint failed: {permissions_response.text}")
                
        except Exception as e:
            print(f"❌ Permissions endpoint error: {e}")
        
        # Step 4: Test Permission Matrix endpoint
        print("\n4️⃣ Testing permission matrix endpoint...")
        
        try:
            matrix_response = await client.get(
                f"{BASE_URL}/api/rbac/permission-matrix",
                headers=headers
            )
            
            print(f"Status: {matrix_response.status_code}")
            
            if matrix_response.status_code == 200:
                matrix_data = matrix_response.json()
                print(f"✅ Permission matrix: {len(matrix_data['roles'])} roles × {len(matrix_data['permissions'])} permissions")
                print(f"   Matrix entries: {len(matrix_data['matrix'])}")
                
                # Count granted permissions
                granted = sum(1 for entry in matrix_data['matrix'] if entry['granted'])
                total = len(matrix_data['matrix'])
                percentage = (granted / total * 100) if total > 0 else 0
                print(f"   Granted permissions: {granted}/{total} ({percentage:.1f}%)")
                
            else:
                print(f"❌ Permission matrix endpoint failed: {matrix_response.text}")
                
        except Exception as e:
            print(f"❌ Permission matrix endpoint error: {e}")
        
        # Step 5: Test Role Creation endpoint
        print("\n5️⃣ Testing role creation endpoint...")
        
        test_role = {
            "name": "Test Manager",
            "description": "Test role for RBAC integration testing",
            "permissions": ["view_pilot", "create_development_task"],
            "is_system_role": False
        }
        
        try:
            create_response = await client.post(
                f"{BASE_URL}/api/rbac/roles",
                headers=headers,
                json=test_role
            )
            
            print(f"Status: {create_response.status_code}")
            
            if create_response.status_code == 200:
                created_role = create_response.json()
                print(f"✅ Created role: {created_role['name']} (ID: {created_role['id']})")
                test_role_id = created_role['id']
                
                # Step 6: Test Role Update endpoint
                print("\n6️⃣ Testing role update endpoint...")
                
                update_data = {
                    "description": "Updated test role description"
                }
                
                update_response = await client.put(
                    f"{BASE_URL}/api/rbac/roles/{test_role_id}",
                    headers=headers,
                    json=update_data
                )
                
                if update_response.status_code == 200:
                    updated_role = update_response.json()
                    print(f"✅ Updated role description: {updated_role['description']}")
                else:
                    print(f"❌ Role update failed: {update_response.text}")
                
                # Step 7: Test Role Deletion endpoint
                print("\n7️⃣ Testing role deletion endpoint...")
                
                delete_response = await client.delete(
                    f"{BASE_URL}/api/rbac/roles/{test_role_id}",
                    headers=headers
                )
                
                if delete_response.status_code == 200:
                    print("✅ Role deleted successfully")
                else:
                    print(f"❌ Role deletion failed: {delete_response.text}")
                    
            else:
                print(f"❌ Role creation failed: {create_response.text}")
                
        except Exception as e:
            print(f"❌ Role creation error: {e}")
        
        print("\n🎉 RBAC Integration Test Complete!")
        print("\nNext Steps:")
        print("1. Open http://localhost:8000/docs to explore all RBAC endpoints")
        print("2. Test the frontend components at http://localhost:8080")
        print("3. Check the permission matrix and role assignments")

if __name__ == "__main__":
    asyncio.run(test_rbac_endpoints())