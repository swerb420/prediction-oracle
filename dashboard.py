#!/usr/bin/env python3
"""
🎯 PREDICTION ORACLE MONITORING DASHBOARD
Beautiful terminal dashboard showing all trades, P&L, and LLM performance
"""

import sqlite3
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import sys
import os

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'
    BG_BLUE = '\033[44m'
    WHITE = '\033[97m'

DB_PATH = "/root/prediction_oracle/paper_trades.db"

def get_db():
    return sqlite3.connect(DB_PATH)

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header():
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    🎯 PREDICTION ORACLE DASHBOARD                                ║
║                         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")

def print_section(title, icon="📊"):
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─'*85}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}{icon} {title}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─'*85}{Colors.RESET}")

def format_pnl(pnl):
    if pnl > 0:
        return f"{Colors.GREEN}+${pnl:.2f}{Colors.RESET}"
    elif pnl < 0:
        return f"{Colors.RED}-${abs(pnl):.2f}{Colors.RESET}"
    else:
        return f"${pnl:.2f}"

def format_percent(val):
    if val >= 60:
        return f"{Colors.GREEN}{val:.1f}%{Colors.RESET}"
    elif val >= 40:
        return f"{Colors.YELLOW}{val:.1f}%{Colors.RESET}"
    else:
        return f"{Colors.RED}{val:.1f}%{Colors.RESET}"

def get_summary_stats(cursor):
    """Get overall portfolio statistics"""
    # Resolved trades
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(AVG(edge), 0) as avg_edge
        FROM trades WHERE outcome IS NOT NULL
    """)
    resolved = cursor.fetchone()
    
    # Open trades
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COALESCE(SUM(bet_size), 0) as at_risk,
            COALESCE(SUM(potential_profit), 0) as potential_profit
        FROM trades WHERE outcome IS NULL
    """)
    open_trades = cursor.fetchone()
    
    return {
        'resolved_total': resolved[0] or 0,
        'wins': resolved[1] or 0,
        'losses': resolved[2] or 0,
        'total_pnl': resolved[3] or 0,
        'avg_edge': resolved[4] or 0,
        'open_total': open_trades[0] or 0,
        'at_risk': open_trades[1] or 0,
        'potential_profit': open_trades[2] or 0
    }

def print_portfolio_summary(stats):
    """Print the main portfolio summary box"""
    print_section("PORTFOLIO SUMMARY", "💰")
    
    win_rate = (stats['wins'] / stats['resolved_total'] * 100) if stats['resolved_total'] > 0 else 0
    
    # Main stats in a nice box
    print(f"""
    {Colors.BOLD}┌─────────────────────────────────────────────────────────────────────────────┐{Colors.RESET}
    │  {Colors.CYAN}RESOLVED TRADES{Colors.RESET}                     │  {Colors.CYAN}OPEN POSITIONS{Colors.RESET}                   │
    │  Total: {stats['resolved_total']:<5}                          │  Total: {stats['open_total']:<5}                          │
    │  Wins:  {Colors.GREEN}{stats['wins']:<5}{Colors.RESET}  Losses: {Colors.RED}{stats['losses']:<5}{Colors.RESET}        │  At Risk:  {Colors.YELLOW}${stats['at_risk']:<8.2f}{Colors.RESET}               │
    │  Win Rate: {format_percent(win_rate):<15}            │  Potential: {Colors.GREEN}${stats['potential_profit']:<8.2f}{Colors.RESET}              │
    │                                        │                                       │
    │  {Colors.BOLD}Total P&L: {format_pnl(stats['total_pnl']):<20}{Colors.RESET}     │  Avg Edge: {stats['avg_edge']:.1f}%                        │
    {Colors.BOLD}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}
""")

def get_resolved_trades(cursor, limit=15):
    """Get recently resolved trades"""
    cursor.execute("""
        SELECT 
            trade_id, question, direction, outcome, entry_price, bet_size, pnl,
            edge, grok_dir, grok_correct, gpt_dir, gpt_correct, 
            category, resolved_at, llm_reason
        FROM trades 
        WHERE outcome IS NOT NULL 
        ORDER BY resolved_at DESC 
        LIMIT ?
    """, (limit,))
    return cursor.fetchall()

def print_resolved_trades(trades):
    """Print resolved trades table"""
    print_section("RECENT RESOLVED TRADES", "✅")
    
    if not trades:
        print(f"  {Colors.DIM}No resolved trades yet{Colors.RESET}")
        return
    
    print(f"  {Colors.DIM}{'Result':<8} {'P&L':<12} {'Bet':<8} {'Dir':<5} {'Edge':<7} {'Category':<12} {'Question':<35}{Colors.RESET}")
    print(f"  {Colors.DIM}{'-'*80}{Colors.RESET}")
    
    for trade in trades:
        trade_id, question, direction, outcome, entry_price, bet_size, pnl, \
        edge, grok_dir, grok_correct, gpt_dir, gpt_correct, category, resolved_at, llm_reason = trade
        
        # Result indicator
        if pnl and pnl > 0:
            result = f"{Colors.GREEN}✅ WIN{Colors.RESET}"
            pnl_str = f"{Colors.GREEN}+${pnl:.2f}{Colors.RESET}"
        else:
            result = f"{Colors.RED}❌ LOSS{Colors.RESET}"
            pnl_str = f"{Colors.RED}-${abs(pnl or 0):.2f}{Colors.RESET}"
        
        # Truncate question
        q_short = (question[:32] + "...") if len(question) > 35 else question
        
        # LLM accuracy indicators
        grok_icon = f"{Colors.GREEN}✓{Colors.RESET}" if grok_correct else f"{Colors.RED}✗{Colors.RESET}"
        gpt_icon = f"{Colors.GREEN}✓{Colors.RESET}" if gpt_correct else f"{Colors.RED}✗{Colors.RESET}"
        
        print(f"  {result:<17} {pnl_str:<20} ${bet_size:<6.0f} {direction:<5} {edge or 0:>5.1f}%  {category:<12} {q_short}")
        print(f"  {Colors.DIM}   LLM: Grok={grok_dir or 'N/A'}{grok_icon} GPT={gpt_dir or 'N/A'}{gpt_icon} | Outcome: {outcome}{Colors.RESET}")

def get_open_trades(cursor):
    """Get all open trades sorted by close time"""
    cursor.execute("""
        SELECT 
            trade_id, question, direction, entry_price, bet_size, 
            potential_profit, edge, hours_left, closes_at, category,
            grok_dir, gpt_dir, llm_reason, llm_confidence
        FROM trades 
        WHERE outcome IS NULL 
        ORDER BY closes_at ASC
    """)
    return cursor.fetchall()

def print_open_trades(trades):
    """Print open positions table"""
    print_section(f"OPEN POSITIONS ({len(trades)} total)", "🔓")
    
    if not trades:
        print(f"  {Colors.DIM}No open positions{Colors.RESET}")
        return
    
    # Group by urgency
    urgent = []
    today = []
    later = []
    
    now = datetime.now(timezone.utc)
    
    for trade in trades:
        trade_id, question, direction, entry_price, bet_size, \
        potential_profit, edge, hours_left, closes_at, category, \
        grok_dir, gpt_dir, llm_reason, llm_confidence = trade
        
        try:
            close_dt = datetime.fromisoformat(closes_at.replace('Z', '+00:00'))
            hours_to_close = (close_dt - now).total_seconds() / 3600
        except:
            hours_to_close = float(hours_left or 999)
        
        trade_data = {
            'question': question,
            'direction': direction,
            'bet_size': bet_size,
            'potential_profit': potential_profit or 0,
            'edge': edge or 0,
            'hours_left': hours_to_close,
            'category': category,
            'grok_dir': grok_dir,
            'gpt_dir': gpt_dir,
            'closes_at': closes_at
        }
        
        if hours_to_close < 0:
            urgent.append(trade_data)
        elif hours_to_close < 24:
            today.append(trade_data)
        else:
            later.append(trade_data)
    
    def print_trades_group(group, title, color):
        if not group:
            return
        print(f"\n  {color}{Colors.BOLD}{title}{Colors.RESET}")
        print(f"  {Colors.DIM}{'Time':<10} {'$Risk':<8} {'Edge':<7} {'Dir':<5} {'Cat':<10} {'Question':<40}{Colors.RESET}")
        
        for t in group[:15]:  # Limit display
            hours = t['hours_left']
            if hours < 0:
                time_str = f"{Colors.RED}EXPIRED{Colors.RESET}"
            elif hours < 1:
                time_str = f"{Colors.RED}{hours*60:.0f}m{Colors.RESET}"
            elif hours < 24:
                time_str = f"{Colors.YELLOW}{hours:.1f}h{Colors.RESET}"
            else:
                time_str = f"{hours/24:.1f}d"
            
            q_short = (t['question'][:37] + "...") if len(t['question']) > 40 else t['question']
            
            print(f"  {time_str:<18} ${t['bet_size']:<6.0f} {t['edge']:>5.1f}%  {t['direction']:<5} {t['category']:<10} {q_short}")
    
    print_trades_group(urgent, f"⚠️  PAST CLOSE TIME ({len(urgent)})", Colors.RED)
    print_trades_group(today, f"⏰ CLOSING TODAY ({len(today)})", Colors.YELLOW)
    print_trades_group(later, f"📅 CLOSING LATER ({len(later)})", Colors.CYAN)

def get_llm_performance(cursor):
    """Calculate LLM accuracy by category"""
    cursor.execute("""
        SELECT 
            category,
            COUNT(*) as total,
            SUM(CASE WHEN grok_correct = 1 THEN 1 ELSE 0 END) as grok_wins,
            SUM(CASE WHEN gpt_correct = 1 THEN 1 ELSE 0 END) as gpt_wins,
            SUM(pnl) as total_pnl
        FROM trades 
        WHERE outcome IS NOT NULL
        GROUP BY category
        ORDER BY total DESC
    """)
    return cursor.fetchall()

def print_llm_performance(data):
    """Print LLM performance breakdown"""
    print_section("LLM PERFORMANCE BY CATEGORY", "🤖")
    
    if not data:
        print(f"  {Colors.DIM}No data yet - waiting for trades to resolve{Colors.RESET}")
        return
    
    print(f"  {Colors.DIM}{'Category':<15} {'Trades':<8} {'Grok Acc':<12} {'GPT Acc':<12} {'P&L':<12}{Colors.RESET}")
    print(f"  {Colors.DIM}{'-'*60}{Colors.RESET}")
    
    total_grok_correct = 0
    total_gpt_correct = 0
    total_trades = 0
    
    for row in data:
        category, total, grok_wins, gpt_wins, pnl = row
        grok_wins = grok_wins or 0
        gpt_wins = gpt_wins or 0
        pnl = pnl or 0
        
        grok_acc = (grok_wins / total * 100) if total > 0 else 0
        gpt_acc = (gpt_wins / total * 100) if total > 0 else 0
        
        total_grok_correct += grok_wins
        total_gpt_correct += gpt_wins
        total_trades += total
        
        print(f"  {category:<15} {total:<8} {format_percent(grok_acc):<20} {format_percent(gpt_acc):<20} {format_pnl(pnl)}")
    
    if total_trades > 0:
        print(f"  {Colors.DIM}{'-'*60}{Colors.RESET}")
        overall_grok = total_grok_correct / total_trades * 100
        overall_gpt = total_gpt_correct / total_trades * 100
        print(f"  {Colors.BOLD}{'OVERALL':<15} {total_trades:<8} {format_percent(overall_grok):<20} {format_percent(overall_gpt):<20}{Colors.RESET}")

def get_source_performance(cursor):
    """Get performance by source (Manifold, Polymarket, ESPN, etc)"""
    cursor.execute("""
        SELECT 
            source,
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl) as total_pnl,
            AVG(edge) as avg_edge
        FROM trades 
        WHERE outcome IS NOT NULL
        GROUP BY source
        ORDER BY total DESC
    """)
    return cursor.fetchall()

def print_source_performance(data):
    """Print performance by source"""
    print_section("PERFORMANCE BY SOURCE", "📡")
    
    if not data:
        print(f"  {Colors.DIM}No data yet{Colors.RESET}")
        return
    
    print(f"  {Colors.DIM}{'Source':<20} {'Trades':<8} {'Win Rate':<12} {'Avg Edge':<10} {'P&L':<12}{Colors.RESET}")
    print(f"  {Colors.DIM}{'-'*65}{Colors.RESET}")
    
    for row in data:
        source, total, wins, pnl, avg_edge = row
        wins = wins or 0
        pnl = pnl or 0
        avg_edge = avg_edge or 0
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Clean up source name
        source_name = source.split('-')[0] if source else 'UNKNOWN'
        
        print(f"  {source_name:<20} {total:<8} {format_percent(win_rate):<20} {avg_edge:>6.1f}%    {format_pnl(pnl)}")

async def check_pending_resolutions(cursor):
    """Check and resolve any pending trades"""
    print_section("CHECKING PENDING RESOLUTIONS", "🔄")
    
    cursor.execute("""
        SELECT trade_id, source, market_id, question, direction, entry_price, bet_size,
               grok_dir, gpt_dir, category
        FROM trades 
        WHERE outcome IS NULL
        ORDER BY closes_at ASC
        LIMIT 50
    """)
    trades = cursor.fetchall()
    
    resolved_count = 0
    checked_count = 0
    
    async with httpx.AsyncClient(timeout=15) as client:
        for trade in trades:
            trade_id, source, market_id, question, direction, entry_price, bet_size, grok_dir, gpt_dir, category = trade
            
            if not market_id:
                continue
                
            try:
                if "MANIFOLD" in (source or "").upper():
                    resp = await client.get(f"https://api.manifold.markets/v0/market/{market_id}")
                    if resp.status_code == 200:
                        data = resp.json()
                        checked_count += 1
                        
                        if data.get("isResolved"):
                            outcome = "YES" if data.get("resolution") == "YES" else "NO"
                            final_prob = data.get("probability", 0.5)
                            
                            # Calculate PnL
                            if direction == outcome:
                                if direction == "YES":
                                    payout = bet_size / entry_price
                                else:
                                    payout = bet_size / (1 - entry_price)
                                pnl = payout - bet_size
                            else:
                                pnl = -bet_size
                            
                            grok_correct = 1 if grok_dir and grok_dir.upper() == outcome else 0
                            gpt_correct = 1 if gpt_dir and gpt_dir.upper() == outcome else 0
                            
                            cursor.execute("""
                                UPDATE trades SET
                                    resolved_at = ?,
                                    outcome = ?,
                                    actual_price = ?,
                                    pnl = ?,
                                    grok_correct = ?,
                                    gpt_correct = ?
                                WHERE trade_id = ?
                            """, (
                                datetime.now(timezone.utc).isoformat(),
                                outcome,
                                final_prob,
                                pnl,
                                grok_correct,
                                gpt_correct,
                                trade_id
                            ))
                            
                            resolved_count += 1
                            result = f"{Colors.GREEN}WIN{Colors.RESET}" if pnl > 0 else f"{Colors.RED}LOSS{Colors.RESET}"
                            print(f"  {result} | {format_pnl(pnl)} | {question[:50]}...")
                            
                elif "ESPN" in (source or "").upper():
                    # ESPN trades resolve automatically when game ends
                    checked_count += 1
                    
            except Exception as e:
                pass  # Silently skip errors during dashboard display
    
    if resolved_count > 0:
        cursor.connection.commit()
        print(f"\n  {Colors.GREEN}✅ Resolved {resolved_count} trades!{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}Checked {checked_count} markets - no new resolutions{Colors.RESET}")

def main():
    """Main dashboard display"""
    clear_screen()
    print_header()
    
    db = get_db()
    cursor = db.cursor()
    
    # Skip slow resolution check if --fast flag
    if "--fast" not in sys.argv and "-f" not in sys.argv:
        try:
            asyncio.run(asyncio.wait_for(check_pending_resolutions(cursor), timeout=30))
        except asyncio.TimeoutError:
            print(f"  {Colors.YELLOW}⚠️  Resolution check timed out - showing cached data{Colors.RESET}")
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠️  Resolution check error: {e}{Colors.RESET}")
    else:
        print_section("SKIPPING RESOLUTION CHECK (--fast mode)", "⚡")
    
    # Get and display stats
    stats = get_summary_stats(cursor)
    print_portfolio_summary(stats)
    
    # Resolved trades
    resolved = get_resolved_trades(cursor)
    print_resolved_trades(resolved)
    
    # Open positions
    open_trades = get_open_trades(cursor)
    print_open_trades(open_trades)
    
    # LLM Performance
    llm_data = get_llm_performance(cursor)
    print_llm_performance(llm_data)
    
    # Source performance
    source_data = get_source_performance(cursor)
    print_source_performance(source_data)
    
    # Footer
    print(f"""
{Colors.CYAN}{'═'*85}{Colors.RESET}
{Colors.DIM}  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Run: python dashboard.py
  Auto-refresh: watch -n 60 python dashboard.py{Colors.RESET}
""")
    
    db.close()

if __name__ == "__main__":
    main()
