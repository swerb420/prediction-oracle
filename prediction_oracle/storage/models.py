"""Database models for tracking markets, evaluations, and trades."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, Boolean
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


class LLMFineTuningData(Base):
    """
    Stores full prompts and responses for fine-tuning.
    This is the key table for training data!
    """
    
    __tablename__ = "llm_finetuning_data"
    
    id = Column(Integer, primary_key=True)
    
    # Identifiers
    venue = Column(String(50), nullable=False)
    market_id = Column(String(200), nullable=False, index=True)
    question = Column(Text)
    category = Column(String(100))
    
    # The actual training data
    model_name = Column(String(100), nullable=False, index=True)
    system_prompt = Column(Text)  # System message if any
    user_prompt = Column(Text, nullable=False)  # Full prompt sent to LLM
    assistant_response = Column(Text, nullable=False)  # Raw LLM response
    parsed_response_json = Column(JSON)  # Parsed structured output
    
    # Prediction details
    predicted_probability = Column(Float)
    predicted_direction = Column(String(10))  # YES/NO/BUY/SELL
    confidence = Column(Float)
    reasoning = Column(Text)  # Extracted reasoning
    
    # Market context at prediction time
    market_price = Column(Float)  # Price when prediction made
    edge = Column(Float)  # predicted_probability - market_price
    time_to_close_hours = Column(Float)
    volume_24h = Column(Float)
    
    # OUTCOME - The ground truth for training
    actual_outcome = Column(String(20))  # YES/NO/RESOLVED_YES/RESOLVED_NO
    prediction_correct = Column(Boolean)  # Was the prediction right?
    pnl = Column(Float)  # If traded, what was the P&L
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)
    
    # Meta
    tokens_used = Column(Integer)
    cost_usd = Column(Float)


class TrainingExport(Base):
    """Track exports of training data for fine-tuning runs."""
    
    __tablename__ = "training_exports"
    
    id = Column(Integer, primary_key=True)
    export_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    format = Column(String(50))  # jsonl, csv, etc.
    num_samples = Column(Integer)
    filters_json = Column(JSON)  # What filters were applied
    file_path = Column(String(500))
    notes = Column(Text)
