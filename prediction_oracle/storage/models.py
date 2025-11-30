"""Database models for tracking markets, evaluations, and trades."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MarketSnapshot(Base):
    """Snapshot of market state at a point in time."""

    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True)
    venue = Column(String(50), nullable=False, index=True)
    market_id = Column(String(200), nullable=False, index=True)
    snapshot_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    question = Column(Text)
    prices_json = Column(JSON)  # {outcome_id: price}
    volume_json = Column(JSON)  # {outcome_id: volume}
    metadata_json = Column(JSON)  # Additional market data


class LLMEval(Base):
    """LLM evaluation of a market outcome."""

    __tablename__ = "llm_evals"

    id = Column(Integer, primary_key=True)
    venue = Column(String(50), nullable=False)
    market_id = Column(String(200), nullable=False, index=True)
    outcome_id = Column(String(200), nullable=False)
    model_name = Column(String(100), nullable=False)
    
    p_true = Column(Float, nullable=False)
    confidence = Column(Float)
    implied_p = Column(Float)  # Market price at evaluation time
    edge = Column(Float)  # p_true - implied_p
    
    prompt_hash = Column(String(64))  # Hash of prompt for deduplication
    rule_risks_json = Column(JSON)
    notes = Column(Text)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class Trade(Base):
    """Record of a trade (real or paper)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    venue = Column(String(50), nullable=False)
    market_id = Column(String(200), nullable=False, index=True)
    outcome_id = Column(String(200), nullable=False)
    
    strategy = Column(String(100), nullable=False)
    mode = Column(String(20), nullable=False)  # research/paper/live
    
    direction = Column(String(10))  # BUY/SELL
    size_usd = Column(Float, nullable=False)
    
    # Probabilities and edge
    p_true = Column(Float)
    implied_p = Column(Float)
    edge = Column(Float)
    
    # Execution
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Additional metadata
    order_id = Column(String(200))
    rationale = Column(Text)
    metadata_json = Column(JSON)
