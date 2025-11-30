"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from ..markets import Market, Venue
from ..risk import BankrollManager


class TradeDecision(BaseModel):
    """A decision to potentially trade on a market."""

    venue: Venue
    market_id: str
    outcome_id: str
    direction: Literal["BUY", "SELL"]
    size_usd: float
    
    # Probabilities and edge
    p_true: float  # LLM-estimated probability
    implied_p: float  # Market price
    edge: float  # p_true - implied_p
    
    # Confidence and risk
    confidence: float
    inter_model_disagreement: float
    rule_risks: list[str]
    
    # Metadata
    strategy_name: str
    rationale: str
    models_used: list[str]


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        """
        Filter markets to those suitable for this strategy.
        
        Args:
            all_markets: All available markets
            
        Returns:
            Filtered list of markets to analyze
        """
        pass

    @abstractmethod
    async def evaluate(
        self,
        markets: list[Market],
        oracle_results: dict,
    ) -> list[TradeDecision]:
        """
        Evaluate markets and generate trade decisions.
        
        Args:
            markets: Markets to evaluate
            oracle_results: Results from LLM oracle
            
        Returns:
            List of trade decisions
        """
        pass


class EnhancedStrategy(BaseStrategy, ABC):
    """Base class for enhanced strategies using the upgraded oracle flow."""

    def __init__(self, name: str, config: dict, bankroll: BankrollManager):
        super().__init__(name, config)
        self.bankroll = bankroll

    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        """Default enhanced behavior: analyze all provided markets."""
        return all_markets

    async def evaluate(
        self,
        markets: list[Market],
        oracle_results: dict,
    ) -> list[TradeDecision]:
        """Delegate to enhanced evaluation and map recommendations to decisions."""
        recommendations = await self.evaluate_markets(markets)

        decisions: list[TradeDecision] = []
        for recommendation in recommendations:
            decision = self._recommendation_to_decision(recommendation)
            if decision:
                decisions.append(decision)

        return decisions

    @abstractmethod
    async def evaluate_markets(self, markets: list[Market]) -> list[dict]:
        """Enhanced evaluation flow producing recommendation dictionaries."""

    @abstractmethod
    def _recommendation_to_decision(self, recommendation: dict) -> TradeDecision | None:
        """Convert an enhanced recommendation into a trade decision."""
