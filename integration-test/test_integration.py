#!/usr/bin/env python3
"""
Test script to verify AI Portal <-> CopyShark integration
"""

import asyncio
import aiohttp
import json

# Configuration
AI_PORTAL_BASE_URL = "http://localhost:8000"
COPYSHARK_BASE_URL = "http://localhost:3000"
AI_PORTAL_API_KEY = "your_ai_portal_api_key_here_for_authentication"

async def test_copyshark_direct():
    """Test direct calls to CopyShark API"""
    print("🧪 Testing CopyShark Direct API Calls...")
    
    async with aiohttp.ClientSession() as session:
        # Test health check
        try:
            async with session.get(f"{COPYSHARK_BASE_URL}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ CopyShark Health: {data['status']}")
                else:
                    print(f"❌ CopyShark Health Check Failed: {response.status}")
        except Exception as e:
            print(f"❌ CopyShark Connection Error: {e}")
            return False

        # Test function definitions
        try:
            async with session.get(f"{COPYSHARK_BASE_URL}/api/functions") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Available Functions: {len(data['functions'])}")
                    for func in data['functions']:
                        print(f"   - {func['name']}: {func['description']}")
        except Exception as e:
            print(f"❌ Function Definitions Error: {e}")

        # Test function call
        try:
            headers = {"X-API-Key": AI_PORTAL_API_KEY}
            payload = {
                "function_name": "generateAdCopy",
                "arguments": {
                    "productName": "SuperRunners Athletic Shoes",
                    "audience": "fitness enthusiasts",
                    "niche": "fitness",
                    "framework": "AIDA",
                    "tone": "energetic"
                }
            }
            
            async with session.post(
                f"{COPYSHARK_BASE_URL}/api/ai-portal/function-call",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Function Call Success:")
                    print(f"   Headline: {data['result']['copy']['headline']}")
                    print(f"   CTA: {data['result']['copy']['cta']}")
                else:
                    print(f"❌ Function Call Failed: {response.status}")
                    print(await response.text())
        except Exception as e:
            print(f"❌ Function Call Error: {e}")

    return True

async def test_ai_portal_integration():
    """Test AI Portal's integrated function calling"""
    print("\n🤖 Testing AI Portal Integration...")
    
    async with aiohttp.ClientSession() as session:
        # Test direct function call endpoint
        try:
            payload = {
                "function_name": "generateAdCopy",
                "arguments": {
                    "productName": "Premium Coffee Beans",
                    "audience": "coffee lovers",
                    "tone": "professional"
                },
                "user_id": "test-user"
            }
            
            async with session.post(
                f"{AI_PORTAL_BASE_URL}/functions/call",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ AI Portal Function Call Success:")
                    print(f"   Function: {data['function_name']}")
                    print(f"   Execution Time: {data['execution_time']:.3f}s")
                    if 'copy' in data['result']:
                        print(f"   Generated: {data['result']['copy']['headline']}")
                else:
                    print(f"❌ AI Portal Function Call Failed: {response.status}")
        except Exception as e:
            print(f"❌ AI Portal Connection Error: {e}")

        # Test chat with automatic function calling
        try:
            payload = {
                "message": "Create an ad for wireless headphones targeting gamers with an urgent tone",
                "user_id": "test-user",
                "task_type": "auto",
                "user_tier": "free"
            }
            
            async with session.post(
                f"{AI_PORTAL_BASE_URL}/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ AI Portal Chat with Functions Success:")
                    print(f"   Model: {data['model']}")
                    print(f"   Reasoning: {data['reasoning']}")
                    print(f"   Response Preview: {data['response'][:100]}...")
                else:
                    print(f"❌ AI Portal Chat Failed: {response.status}")
        except Exception as e:
            print(f"❌ AI Portal Chat Error: {e}")

async def test_available_functions():
    """Test getting available functions from AI Portal"""
    print("\n📋 Testing Available Functions...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{AI_PORTAL_BASE_URL}/functions/available") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Available Functions from AI Portal:")
                    for func in data['functions']:
                        print(f"   - {func['name']}")
                        required_params = func['parameters'].get('required', [])
                        if required_params:
                            print(f"     Required: {', '.join(required_params)}")
                else:
                    print(f"❌ Get Functions Failed: {response.status}")
        except Exception as e:
            print(f"❌ Get Functions Error: {e}")

async def main():
    """Run all tests"""
    print("🚀 Starting AI Portal <-> CopyShark Integration Tests\n")
    
    # Test CopyShark direct API
    copyshark_ok = await test_copyshark_direct()
    
    if copyshark_ok:
        # Test AI Portal integration
        await test_ai_portal_integration()
        await test_available_functions()
    
    print("\n✨ Integration tests complete!")

if __name__ == "__main__":
    asyncio.run(main())