"""
Fine-tuning data logger for capturing all LLM interactions.
This is crucial for building training datasets!
"""

import json
import logging
from datetime import datetime
from typing import Optional, Any

from sqlalchemy.ext.asyncio import AsyncSession

from .models import LLMFineTuningData

logger = logging.getLogger(__name__)


class FineTuningLogger:
    """
    Logs all LLM interactions for fine-tuning purposes.
    Stores full prompts, responses, and outcomes.
    """
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    async def log_prediction(
        self,
        # Identifiers
        venue: str,
        market_id: str,
        question: str,
        category: Optional[str] = None,
        
        # LLM interaction
        model_name: str = "",
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        assistant_response: str = "",
        parsed_response: Optional[dict] = None,
        
        # Prediction
        predicted_probability: Optional[float] = None,
        predicted_direction: Optional[str] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        
        # Market context
        market_price: Optional[float] = None,
        edge: Optional[float] = None,
        time_to_close_hours: Optional[float] = None,
        volume_24h: Optional[float] = None,
        
        # Metadata
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> int:
        """
        Log a prediction for fine-tuning.
        Returns the ID of the created record.
        """
        async with self.session_factory() as session:
            record = LLMFineTuningData(
                venue=venue,
                market_id=market_id,
                question=question,
                category=category,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                parsed_response_json=parsed_response,
                predicted_probability=predicted_probability,
                predicted_direction=predicted_direction,
                confidence=confidence,
                reasoning=reasoning,
                market_price=market_price,
                edge=edge,
                time_to_close_hours=time_to_close_hours,
                volume_24h=volume_24h,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                created_at=datetime.utcnow(),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            
            logger.debug(f"Logged fine-tuning data: {market_id} / {model_name}")
            return record.id
            
    async def update_outcome(
        self,
        record_id: int,
        actual_outcome: str,
        prediction_correct: bool,
        pnl: Optional[float] = None,
    ) -> None:
        """Update a record with the actual outcome."""
        async with self.session_factory() as session:
            from sqlalchemy import update
            stmt = (
                update(LLMFineTuningData)
                .where(LLMFineTuningData.id == record_id)
                .values(
                    actual_outcome=actual_outcome,
                    prediction_correct=prediction_correct,
                    pnl=pnl,
                    resolved_at=datetime.utcnow(),
                )
            )
            await session.execute(stmt)
            await session.commit()
            
            logger.info(f"Updated outcome for record {record_id}: {actual_outcome}")
            
    async def update_outcome_by_market(
        self,
        market_id: str,
        actual_outcome: str,
    ) -> int:
        """
        Update all predictions for a market with the actual outcome.
        Returns the number of records updated.
        """
        async with self.session_factory() as session:
            from sqlalchemy import select, update
            
            # First, get all unresolved records for this market
            stmt = select(LLMFineTuningData).where(
                LLMFineTuningData.market_id == market_id,
                LLMFineTuningData.actual_outcome.is_(None),
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            updated = 0
            for record in records:
                # Determine if prediction was correct
                prediction_correct = False
                if record.predicted_direction:
                    if actual_outcome in ["YES", "RESOLVED_YES"]:
                        prediction_correct = record.predicted_direction.upper() in ["YES", "BUY"]
                    elif actual_outcome in ["NO", "RESOLVED_NO"]:
                        prediction_correct = record.predicted_direction.upper() in ["NO", "SELL"]
                
                record.actual_outcome = actual_outcome
                record.prediction_correct = prediction_correct
                record.resolved_at = datetime.utcnow()
                updated += 1
                
            await session.commit()
            logger.info(f"Updated {updated} records for market {market_id}")
            return updated

    async def export_jsonl(
        self,
        output_path: str,
        only_resolved: bool = True,
        only_correct: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        categories: Optional[list[str]] = None,
    ) -> int:
        """
        Export training data in JSONL format for fine-tuning.
        Returns the number of samples exported.
        
        Format matches OpenAI fine-tuning format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        async with self.session_factory() as session:
            from sqlalchemy import select
            
            query = select(LLMFineTuningData)
            
            if only_resolved:
                query = query.where(LLMFineTuningData.actual_outcome.isnot(None))
            
            if only_correct is True:
                query = query.where(LLMFineTuningData.prediction_correct == True)
            elif only_correct is False:
                query = query.where(LLMFineTuningData.prediction_correct == False)
                
            if min_confidence:
                query = query.where(LLMFineTuningData.confidence >= min_confidence)
                
            if categories:
                query = query.where(LLMFineTuningData.category.in_(categories))
                
            result = await session.execute(query)
            records = result.scalars().all()
            
            with open(output_path, "w") as f:
                for record in records:
                    messages = []
                    
                    if record.system_prompt:
                        messages.append({
                            "role": "system",
                            "content": record.system_prompt
                        })
                    
                    messages.append({
                        "role": "user", 
                        "content": record.user_prompt
                    })
                    
                    messages.append({
                        "role": "assistant",
                        "content": record.assistant_response
                    })
                    
                    # Add metadata as comment
                    entry = {
                        "messages": messages,
                        "_metadata": {
                            "market_id": record.market_id,
                            "category": record.category,
                            "prediction_correct": record.prediction_correct,
                            "actual_outcome": record.actual_outcome,
                            "edge": record.edge,
                        }
                    }
                    
                    f.write(json.dumps(entry) + "\n")
                    
            logger.info(f"Exported {len(records)} samples to {output_path}")
            return len(records)


# Global instance - will be initialized by the runner
finetuning_logger: Optional[FineTuningLogger] = None


def get_finetuning_logger() -> Optional[FineTuningLogger]:
    """Get the global fine-tuning logger instance."""
    return finetuning_logger


def init_finetuning_logger(session_factory) -> FineTuningLogger:
    """Initialize the global fine-tuning logger."""
    global finetuning_logger
    finetuning_logger = FineTuningLogger(session_factory)
    return finetuning_logger
