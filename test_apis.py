#!/usr/bin/env python3
"""Test LLM APIs"""
from dotenv import load_dotenv
load_dotenv('/root/prediction_oracle/.env')

import os
import requests

print("="*60)
print("TESTING LLM APIS")
print("="*60)

xai_key = os.getenv('XAI_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

print(f"\nXAI_API_KEY: {xai_key[:20] if xai_key else 'NOT SET'}...")
print(f"OPENAI_API_KEY: {openai_key[:20] if openai_key else 'NOT SET'}...")

# Test Grok
print("\n" + "-"*40)
print("Testing Grok (X.AI)...")
try:
    response = requests.post(
        'https://api.x.ai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {xai_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'grok-beta',
            'messages': [{'role': 'user', 'content': 'Say hello in 5 words'}],
            'max_tokens': 50
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text[:200]}")
except Exception as e:
    print(f"Exception: {e}")

# Test OpenAI
print("\n" + "-"*40)
print("Testing OpenAI...")
try:
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'gpt-4o-mini',
            'messages': [{'role': 'user', 'content': 'Say hello in 5 words'}],
            'max_tokens': 50
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text[:200]}")
except Exception as e:
    print(f"Exception: {e}")

print("\n" + "="*60)
print("DONE")
