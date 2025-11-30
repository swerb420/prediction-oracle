#!/usr/bin/env python3
"""
Re-analyze open trades with category-specific prompts
This uses optimized prompts for each category to improve LLM prediction accuracy
"""

import sqlite3
import subprocess
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/root/prediction_oracle/.env')

# Add path to import category prompts directly without package imports
sys.path.insert(0, '/root/prediction_oracle/llm')

# Import the functions we need directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location("category_prompts", "/root/prediction_oracle/llm/category_prompts.py")
category_prompts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(category_prompts_module)

get_category_prompt = category_prompts_module.get_category_prompt
adjust_for_grok = category_prompts_module.adjust_for_grok
adjust_for_gpt = category_prompts_module.adjust_for_gpt
parse_llm_response = category_prompts_module.parse_llm_response

DB_PATH = "/root/prediction_oracle/paper_trades.db"

def call_grok_api(prompt: str) -> str:
    """Call Grok API with the prompt."""
    api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
    if not api_key:
        return None
    
    try:
        import requests
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  Grok API error: {e}")
    return None


def call_gpt_api(prompt: str) -> str:
    """Call OpenAI GPT API with the prompt."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    
    try:
        import requests
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  GPT API error: {e}")
    return None


def main():
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    print("=" * 80)
    print("🧠 RE-ANALYZING TRADES WITH CATEGORY-SPECIFIC PROMPTS")
    print("=" * 80)
    
    # Get open trades
    cursor.execute("""
        SELECT trade_id, source, market_id, question, direction, entry_price, 
               category, closes_at, grok_dir, gpt_dir
        FROM trades 
        WHERE outcome IS NULL
        ORDER BY category, closes_at
    """)
    trades = cursor.fetchall()
    
    print(f"\n📊 Found {len(trades)} open trades to re-analyze\n")
    
    # Group by category
    by_category = {}
    for t in trades:
        cat = t[6] or "OTHER"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(t)
    
    print("📁 Distribution by category:")
    for cat, cat_trades in sorted(by_category.items()):
        print(f"   {cat}: {len(cat_trades)} trades")
    
    print("\n" + "=" * 80)
    print("🔄 Starting analysis (this will call LLM APIs)...")
    print("=" * 80)
    
    updated_count = 0
    
    for category, cat_trades in by_category.items():
        print(f"\n📂 {category} ({len(cat_trades)} trades)")
        print("-" * 60)
        
        for trade in cat_trades[:5]:  # Limit to 5 per category for now
            trade_id, source, market_id, question, direction, entry_price, cat, closes_at, old_grok, old_gpt = trade
            
            print(f"\n  📌 {question[:50]}...")
            print(f"     Current: Grok={old_grok or 'N/A'}, GPT={old_gpt or 'N/A'}")
            
            # Generate category-specific prompt
            base_prompt = get_category_prompt(
                category=category,
                question=question,
                market_price=entry_price,
                close_time=closes_at or "Unknown"
            )
            
            # Get Grok prediction with Grok-specific adjustments
            grok_prompt = adjust_for_grok(base_prompt)
            grok_response = call_grok_api(grok_prompt)
            
            new_grok_dir = None
            grok_reason = None
            if grok_response:
                parsed = parse_llm_response(grok_response)
                new_grok_dir = parsed.get('direction')
                grok_reason = parsed.get('reason')
                print(f"     🤖 Grok: {new_grok_dir} ({parsed.get('confidence', 'N/A')}) - {grok_reason[:50] if grok_reason else 'No reason'}...")
            
            # Get GPT prediction with GPT-specific adjustments
            gpt_prompt = adjust_for_gpt(base_prompt)
            gpt_response = call_gpt_api(gpt_prompt)
            
            new_gpt_dir = None
            gpt_reason = None
            if gpt_response:
                parsed = parse_llm_response(gpt_response)
                new_gpt_dir = parsed.get('direction')
                gpt_reason = parsed.get('reason')
                print(f"     🧠 GPT:  {new_gpt_dir} ({parsed.get('confidence', 'N/A')}) - {gpt_reason[:50] if gpt_reason else 'No reason'}...")
            
            # Update if we got new predictions
            if new_grok_dir or new_gpt_dir:
                # Combine reasons
                combined_reason = f"Grok: {grok_reason or 'N/A'} | GPT: {gpt_reason or 'N/A'}"[:200]
                
                cursor.execute("""
                    UPDATE trades SET
                        grok_dir = COALESCE(?, grok_dir),
                        gpt_dir = COALESCE(?, gpt_dir),
                        llm_reason = ?
                    WHERE trade_id = ?
                """, (new_grok_dir, new_gpt_dir, combined_reason, trade_id))
                
                updated_count += 1
                
                # Show if prediction changed
                if new_grok_dir and new_grok_dir != old_grok:
                    print(f"     ⚠️ Grok changed: {old_grok} → {new_grok_dir}")
                if new_gpt_dir and new_gpt_dir != old_gpt:
                    print(f"     ⚠️ GPT changed: {old_gpt} → {new_gpt_dir}")
    
    db.commit()
    
    print("\n" + "=" * 80)
    print(f"✅ Updated {updated_count} trades with new category-specific analysis")
    print("=" * 80)
    
    # Show summary of changes
    print("\n📊 CATEGORY PROMPT DIFFERENCES:")
    print("-" * 60)
    print("""
SPORTS:    Focuses on ATS records, injuries, home/away splits, sharp money
POLITICS:  Emphasizes polls, base rates, market overreaction detection  
CRYPTO:    Analyzes price thresholds, volatility, time horizon probabilities
ECONOMICS: Fed guidance, dot plots, funds futures, economic data
GEOPOLITICS: Base rates, escalation signals, expert consensus
ENTERTAINMENT: Industry patterns, expert consensus, social sentiment
AI_TECH:   Official announcements, technical feasibility, hype detection
    """)
    
    db.close()


if __name__ == "__main__":
    main()
