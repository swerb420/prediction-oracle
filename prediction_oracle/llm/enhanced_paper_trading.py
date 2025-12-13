"""
Enhanced Paper Trading Engine with Whale/Venue Tracking
========================================================

Extends base paper trading with:
1. Whale influence logging per trade
2. Multi-venue slippage simulation
3. Enhanced trade rationale tracking
4. Win rate analysis by whale alignment

Target: Track correlation between whale signals and trade outcomes.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .crypto_data import CryptoSymbol, get_fetcher
from .enhanced_hybrid_oracle import EnhancedHybridPrediction, TradeSignal

logger = logging.getLogger(__name__)


class EnhancedPosition(BaseModel):
    """Position with whale/venue context."""
    id: str
    symbol: CryptoSymbol
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    entry_time: datetime
    size_usd: float
    confidence: float
    target_candles: int
    
    # Enhanced context
    whale_consensus: float
    venue_consensus: float
    signal_alignment: float
    clean_score: float
    grok_regime: str
    trade_rationale: str
    
    # Tracking
    current_price: float | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class EnhancedClosedTrade(BaseModel):
    """Closed trade with full context."""
    id: str
    symbol: CryptoSymbol
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size_usd: float
    confidence: float
    
    # Results
    pnl_usd: float
    pnl_pct: float
    is_win: bool
    duration_minutes: int
    
    # Enhanced context
    whale_consensus: float
    venue_consensus: float
    signal_alignment: float
    clean_score: float
    grok_regime: str
    trade_rationale: str
    
    # Correlation tracking
    whale_aligned: bool  # Did whale consensus match our direction?
    venue_aligned: bool  # Did venue consensus match?


class EnhancedTradingStats(BaseModel):
    """Statistics with whale/venue analysis."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    # By symbol
    stats_by_symbol: dict[str, dict] = {}
    
    # Whale analysis
    whale_aligned_trades: int = 0
    whale_aligned_wins: int = 0
    whale_aligned_win_rate: float = 0.0
    
    whale_opposed_trades: int = 0
    whale_opposed_wins: int = 0
    whale_opposed_win_rate: float = 0.0
    
    # Venue analysis
    venue_aligned_trades: int = 0
    venue_aligned_wins: int = 0
    venue_aligned_win_rate: float = 0.0
    
    # Quality score analysis
    high_quality_trades: int = 0  # clean_score > 0.7
    high_quality_wins: int = 0
    high_quality_win_rate: float = 0.0
    
    # Regime analysis
    stats_by_regime: dict[str, dict] = {}


class EnhancedPaperTradingEngine:
    """
    Enhanced paper trading with whale/venue tracking.
    
    Features:
    - Track whale influence on trades
    - Simulate multi-venue slippage
    - Analyze win rate by signal alignment
    - Store full context for post-trade analysis
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.05,
        max_positions: int = 3,
        hold_candles: int = 1,
        state_file: str = "./enhanced_paper_trading_state.json",
        slippage_bps: float = 5.0,  # Default slippage
    ):
        """
        Initialize enhanced paper trading.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            max_positions: Max concurrent positions
            hold_candles: Candles to hold before close
            state_file: State persistence path
            slippage_bps: Default slippage in basis points
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.hold_candles = hold_candles
        self.state_file = Path(state_file)
        self.slippage_bps = slippage_bps
        
        self.open_positions: dict[str, EnhancedPosition] = {}
        self.closed_trades: list[EnhancedClosedTrade] = []
        self.stats = EnhancedTradingStats()
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                
                self.capital = data.get("capital", self.initial_capital)
                
                self.open_positions = {
                    k: EnhancedPosition(**v) 
                    for k, v in data.get("open_positions", {}).items()
                }
                
                self.closed_trades = [
                    EnhancedClosedTrade(**t) 
                    for t in data.get("closed_trades", [])
                ]
                
                self._update_stats()
                logger.info(
                    f"Loaded state: {len(self.open_positions)} open, "
                    f"{len(self.closed_trades)} closed"
                )
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            data = {
                "capital": self.capital,
                "open_positions": {
                    k: v.model_dump(mode="json") 
                    for k, v in self.open_positions.items()
                },
                "closed_trades": [
                    t.model_dump(mode="json") for t in self.closed_trades
                ]
            }
            
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _update_stats(self):
        """Recalculate statistics with whale/venue analysis."""
        if not self.closed_trades:
            self.stats = EnhancedTradingStats()
            return
        
        total = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.is_win]
        losses = [t for t in self.closed_trades if not t.is_win]
        
        win_rate = len(wins) / total if total > 0 else 0
        
        total_pnl = sum(t.pnl_usd for t in self.closed_trades)
        total_pnl_pct = sum(t.pnl_pct for t in self.closed_trades)
        
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        
        # By symbol
        stats_by_symbol = {}
        for sym in ["BTC", "ETH", "SOL"]:
            sym_trades = [t for t in self.closed_trades if t.symbol == sym]
            if sym_trades:
                sym_wins = len([t for t in sym_trades if t.is_win])
                stats_by_symbol[sym] = {
                    "trades": len(sym_trades),
                    "wins": sym_wins,
                    "win_rate": sym_wins / len(sym_trades),
                    "pnl_usd": sum(t.pnl_usd for t in sym_trades)
                }
        
        # Whale analysis
        whale_aligned = [t for t in self.closed_trades if t.whale_aligned]
        whale_opposed = [t for t in self.closed_trades if not t.whale_aligned]
        
        whale_aligned_wins = len([t for t in whale_aligned if t.is_win])
        whale_opposed_wins = len([t for t in whale_opposed if t.is_win])
        
        # Venue analysis
        venue_aligned = [t for t in self.closed_trades if t.venue_aligned]
        venue_aligned_wins = len([t for t in venue_aligned if t.is_win])
        
        # Quality analysis
        high_quality = [t for t in self.closed_trades if t.clean_score > 0.7]
        high_quality_wins = len([t for t in high_quality if t.is_win])
        
        # Regime analysis
        stats_by_regime = {}
        for regime in ["trending", "ranging", "volatile", "news_driven"]:
            regime_trades = [t for t in self.closed_trades if t.grok_regime == regime]
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t.is_win])
                stats_by_regime[regime] = {
                    "trades": len(regime_trades),
                    "wins": regime_wins,
                    "win_rate": regime_wins / len(regime_trades),
                    "pnl_usd": sum(t.pnl_usd for t in regime_trades)
                }
        
        self.stats = EnhancedTradingStats(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl_usd=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            stats_by_symbol=stats_by_symbol,
            
            whale_aligned_trades=len(whale_aligned),
            whale_aligned_wins=whale_aligned_wins,
            whale_aligned_win_rate=(
                whale_aligned_wins / len(whale_aligned) if whale_aligned else 0
            ),
            
            whale_opposed_trades=len(whale_opposed),
            whale_opposed_wins=whale_opposed_wins,
            whale_opposed_win_rate=(
                whale_opposed_wins / len(whale_opposed) if whale_opposed else 0
            ),
            
            venue_aligned_trades=len(venue_aligned),
            venue_aligned_wins=venue_aligned_wins,
            venue_aligned_win_rate=(
                venue_aligned_wins / len(venue_aligned) if venue_aligned else 0
            ),
            
            high_quality_trades=len(high_quality),
            high_quality_wins=high_quality_wins,
            high_quality_win_rate=(
                high_quality_wins / len(high_quality) if high_quality else 0
            ),
            
            stats_by_regime=stats_by_regime,
        )
    
    async def open_position(
        self,
        prediction: EnhancedHybridPrediction,
    ) -> EnhancedPosition | None:
        """
        Open a new position based on prediction.
        
        Args:
            prediction: Enhanced hybrid prediction
            
        Returns:
            New position or None if not opened
        """
        if not prediction.should_trade:
            return None
        
        if prediction.final_signal == "HOLD":
            return None
        
        if len(self.open_positions) >= self.max_positions:
            logger.debug("Max positions reached")
            return None
        
        # Check if we already have position for this symbol
        for pos in self.open_positions.values():
            if pos.symbol == prediction.symbol:
                logger.debug(f"Already have position for {prediction.symbol}")
                return None
        
        # Calculate position size
        position_size = self.capital * self.position_size_pct
        
        # Apply slippage to entry
        slippage_mult = 1 + (self.slippage_bps / 10000)
        if prediction.final_signal == "LONG":
            entry_price = prediction.current_price * slippage_mult
        else:
            entry_price = prediction.current_price / slippage_mult
        
        # Create position
        position_id = f"{prediction.symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        position = EnhancedPosition(
            id=position_id,
            symbol=prediction.symbol,
            direction="LONG" if prediction.final_signal == "LONG" else "SHORT",
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            size_usd=position_size,
            confidence=prediction.final_confidence,
            target_candles=self.hold_candles,
            whale_consensus=prediction.whale_consensus,
            venue_consensus=prediction.venue_consensus,
            signal_alignment=prediction.signal_alignment,
            clean_score=prediction.clean_score,
            grok_regime=prediction.grok_regime,
            trade_rationale=prediction.trade_rationale,
        )
        
        self.open_positions[position_id] = position
        self._save_state()
        
        logger.info(
            f"Opened {position.direction} {position.symbol} @ ${entry_price:,.2f} "
            f"(whale: {prediction.whale_consensus:+.2f}, venue: {prediction.venue_consensus:+.2f})"
        )
        
        return position
    
    async def update_positions(self) -> list[EnhancedClosedTrade]:
        """
        Update all open positions and close expired ones.
        
        Returns:
            List of closed trades
        """
        fetcher = get_fetcher()
        closed = []
        
        for pos_id, pos in list(self.open_positions.items()):
            try:
                # Get current price
                current_price = await fetcher.get_current_price(pos.symbol)
                pos.current_price = current_price
                
                # Calculate unrealized PnL
                if pos.direction == "LONG":
                    pnl_pct = (current_price / pos.entry_price) - 1
                else:
                    pnl_pct = (pos.entry_price / current_price) - 1
                
                pos.unrealized_pnl_pct = pnl_pct
                pos.unrealized_pnl = pos.size_usd * pnl_pct
                
                # Check if we should close
                # Time-based close (after target candles)
                elapsed = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
                candles_elapsed = elapsed / (15 * 60)
                
                should_close = candles_elapsed >= pos.target_candles
                
                if should_close:
                    trade = await self._close_position(pos)
                    closed.append(trade)
                    
            except Exception as e:
                logger.error(f"Error updating position {pos_id}: {e}")
        
        if closed:
            self._update_stats()
            self._save_state()
        
        return closed
    
    async def _close_position(
        self,
        pos: EnhancedPosition,
    ) -> EnhancedClosedTrade:
        """Close a position and record the trade."""
        fetcher = get_fetcher()
        
        exit_price = await fetcher.get_current_price(pos.symbol)
        
        # Apply exit slippage
        slippage_mult = 1 + (self.slippage_bps / 10000)
        if pos.direction == "LONG":
            exit_price = exit_price / slippage_mult  # Worse exit for long
        else:
            exit_price = exit_price * slippage_mult  # Worse exit for short
        
        # Calculate PnL
        if pos.direction == "LONG":
            pnl_pct = (exit_price / pos.entry_price) - 1
        else:
            pnl_pct = (pos.entry_price / exit_price) - 1
        
        pnl_usd = pos.size_usd * pnl_pct
        is_win = pnl_pct > 0
        
        exit_time = datetime.now(timezone.utc)
        duration = int((exit_time - pos.entry_time).total_seconds() / 60)
        
        # Determine alignment
        expected_consensus = 1.0 if pos.direction == "LONG" else -1.0
        whale_aligned = (pos.whale_consensus * expected_consensus) > 0
        venue_aligned = (pos.venue_consensus * expected_consensus) > 0
        
        trade = EnhancedClosedTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            size_usd=pos.size_usd,
            confidence=pos.confidence,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            is_win=is_win,
            duration_minutes=duration,
            whale_consensus=pos.whale_consensus,
            venue_consensus=pos.venue_consensus,
            signal_alignment=pos.signal_alignment,
            clean_score=pos.clean_score,
            grok_regime=pos.grok_regime,
            trade_rationale=pos.trade_rationale,
            whale_aligned=whale_aligned,
            venue_aligned=venue_aligned,
        )
        
        # Update capital
        self.capital += pnl_usd
        
        # Record and remove position
        self.closed_trades.append(trade)
        del self.open_positions[pos.id]
        
        emoji = "✅" if is_win else "❌"
        logger.info(
            f"{emoji} Closed {pos.direction} {pos.symbol}: {pnl_pct:+.2%} (${pnl_usd:+.2f}) "
            f"| Whale aligned: {whale_aligned}, Venue aligned: {venue_aligned}"
        )
        
        return trade
    
    def print_stats(self):
        """Print trading statistics with whale analysis."""
        s = self.stats
        
        print("\n" + "=" * 60)
        print("ENHANCED TRADING STATISTICS")
        print("=" * 60)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total Trades: {s.total_trades}")
        print(f"   Win Rate: {s.win_rate:.1%} ({s.winning_trades}W / {s.losing_trades}L)")
        print(f"   Total PnL: ${s.total_pnl_usd:,.2f} ({s.total_pnl_pct:+.2%})")
        print(f"   Avg Win: {s.avg_win_pct:+.2%} | Avg Loss: {s.avg_loss_pct:+.2%}")
        
        print(f"\n🐋 Whale Signal Analysis:")
        print(f"   Whale-Aligned Trades: {s.whale_aligned_trades} ({s.whale_aligned_win_rate:.1%} WR)")
        print(f"   Whale-Opposed Trades: {s.whale_opposed_trades} ({s.whale_opposed_win_rate:.1%} WR)")
        
        print(f"\n🏛️ Venue Signal Analysis:")
        print(f"   Venue-Aligned Trades: {s.venue_aligned_trades} ({s.venue_aligned_win_rate:.1%} WR)")
        
        print(f"\n✨ Quality Analysis:")
        print(f"   High Quality Trades: {s.high_quality_trades} ({s.high_quality_win_rate:.1%} WR)")
        
        if s.stats_by_symbol:
            print(f"\n📈 By Symbol:")
            for sym, data in s.stats_by_symbol.items():
                print(f"   {sym}: {data['trades']} trades, {data['win_rate']:.1%} WR, ${data['pnl_usd']:,.2f}")
        
        if s.stats_by_regime:
            print(f"\n🌡️ By Regime:")
            for regime, data in s.stats_by_regime.items():
                print(f"   {regime}: {data['trades']} trades, {data['win_rate']:.1%} WR")
        
        print("\n" + "=" * 60)
