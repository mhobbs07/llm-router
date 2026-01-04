#!/usr/bin/env python3

import asyncio
import os
from src.main import get_available_models
from src.config import load_config_from_env, get_provider_config, CURATED_MODELS
from returns.result import Success, Failure

async def test_models_endpoint():
    print("=== TESTING /models ENDPOINT ===")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test config loading
    print("\n1. Testing config loading...")
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        print(f"❌ Config loading failed: {config_result.failure()}")
        return
    
    config = config_result.unwrap()
    print("✅ Config loaded successfully")
    
    # Test provider configurations
    print("\n2. Testing provider configurations...")
    providers = ["openai", "anthropic", "google_gemini", "xai"]
    
    for provider in providers:
        provider_result = get_provider_config(provider, config)
        if isinstance(provider_result, Success):
            print(f"✅ {provider}: configured")
        else:
            print(f"❌ {provider}: {provider_result.failure()}")
    
    # Test curated models
    print(f"\n3. Curated models available: {len(CURATED_MODELS)}")
    for model_id, metadata in CURATED_MODELS.items():
        print(f"   {model_id} ({metadata.provider}) - {metadata.tier}")
    
    # Test models endpoint
    print("\n4. Testing /models endpoint...")
    try:
        models_response = await get_available_models()
        print(f"✅ Models endpoint returned {len(models_response['data'])} models")
        
        print("\n5. Available models:")
        for model in models_response['data']:
            print(f"   {model['id']} ({model['owned_by']}) - {model['tier']} - ${model['pricing']['input_per_1m']:.2f}/${model['pricing']['output_per_1m']:.2f}")
            
        # Check for missing models
        print("\n6. Checking for missing models...")
        returned_model_ids = {model['id'] for model in models_response['data']}
        curated_model_ids = set(CURATED_MODELS.keys())
        missing_models = curated_model_ids - returned_model_ids
        
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            for missing in missing_models:
                metadata = CURATED_MODELS[missing]
                provider_name = {
                    "openai": "openai",
                    "anthropic": "anthropic", 
                    "google": "google_gemini",
                    "xai": "xai"
                }[metadata.provider]
                provider_result = get_provider_config(provider_name, config)
                if isinstance(provider_result, Failure):
                    print(f"   {missing}: Provider {provider_name} not configured - {provider_result.failure()}")
                else:
                    print(f"   {missing}: Provider {provider_name} is configured but model still missing")
        else:
            print("✅ All curated models are present")
            
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_models_endpoint())