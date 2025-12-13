"""LLM engine for market analysis and crypto trading."""

from .oracle import LLMOracle, OracleResult
from .providers import LLMProvider, LLMQuery, LLMResponse
from .enhanced_oracle import EnhancedOracle
from .providers_enhanced import (
    GrokReasoningProvider,
    FastScreeningProvider,
    EnhancedLLMResponse,
    ReasoningStep,
    create_grok_provider,
    create_fast_provider,
)
from .cost_optimizer import (
    CostTracker,
    SmartRouter,
    ModelProfile,
    MODEL_PROFILES,
    cost_tracker,
    smart_router,
)

# Crypto trading modules
from .crypto_data import (
    CryptoDataFetcher,
    CandleData,
    Candle,
    CryptoSymbol,
    fetch_crypto_data,
    get_fetcher,
)
from .feature_engineering import (
    FeatureSet,
    extract_features,
    extract_all_features,
)
from .ml_predictor import (
    CryptoMLPredictor,
    MLPrediction,
    GradientBoostingPredictor,
    predict_crypto_direction,
    get_predictor,
)
from .grok41_provider import (
    Grok41FastProvider,
    GrokValidation,
    create_grok41_provider,
)
from .hybrid_oracle import (
    HybridCryptoOracle,
    HybridPrediction,
    get_oracle,
    get_trading_signals,
)
from .paper_trading import (
    PaperTradingEngine,
    Position,
    ClosedTrade,
    TradingStats,
    get_trading_engine,
)

# Enhanced multi-venue and whale tracking modules
from .multi_venue_client import (
    MultiVenueClient,
    VenuePrice,
    VenueOrderbook,
    CrossVenueFeatures,
    get_cross_venue_snapshot,
)
from .poly_whale_client import (
    PolyWhalesClient,
    WhaleProfile,
    WhaleTrade,
    WhaleConsensus,
    get_whale_signals,
)
from .enhanced_features import (
    EnhancedFeatureSet,
    FeatureQualityFilter,
    extract_enhanced_features,
    extract_enhanced_features_batch,
)
from .enhanced_grok_provider import (
    EnhancedGrokProvider,
    EnhancedGrokValidation,
    create_enhanced_grok_provider,
)
from .enhanced_ml_predictor import (
    EnhancedCryptoMLPredictor,
    EnhancedMLPrediction,
    EnhancedGradientBoostingPredictor,
    get_enhanced_predictor,
    predict_with_whale_signals,
)
from .enhanced_hybrid_oracle import (
    EnhancedHybridOracle,
    EnhancedHybridPrediction,
    get_enhanced_oracle,
    predict_with_full_context,
)
from .enhanced_paper_trading import (
    EnhancedPaperTradingEngine,
    EnhancedPosition,
    EnhancedClosedTrade,
    EnhancedTradingStats,
)

# New 15M tracking and whale ML modules
from .polymarket_15m_tracker import (
    Polymarket15MTracker,
    Market15MData,
    PriceUpdate,
)
from .top_trader_downloader import (
    TopTraderDownloader,
    TopTrader,
    TraderTrade,
    TraderPosition,
)
from .smart_grok_trigger import (
    SmartGrokTrigger,
    SignalContext,
    TriggerDecision,
    TriggerReason,
)
from .whale_pattern_trainer import (
    WhalePatternTrainer,
    WhaleFeatures,
    TrainingExample,
)

__all__ = [
    # Original exports
    "LLMOracle",
    "OracleResult",
    "LLMProvider",
    "LLMQuery",
    "LLMResponse",
    "EnhancedOracle",
    "GrokReasoningProvider",
    "FastScreeningProvider",
    "EnhancedLLMResponse",
    "ReasoningStep",
    "create_grok_provider",
    "create_fast_provider",
    "CostTracker",
    "SmartRouter",
    "ModelProfile",
    "MODEL_PROFILES",
    "cost_tracker",
    "smart_router",
    # Crypto trading exports
    "CryptoDataFetcher",
    "CandleData",
    "Candle",
    "CryptoSymbol",
    "fetch_crypto_data",
    "get_fetcher",
    "FeatureSet",
    "extract_features",
    "extract_all_features",
    "CryptoMLPredictor",
    "MLPrediction",
    "GradientBoostingPredictor",
    "predict_crypto_direction",
    "get_predictor",
    "Grok41FastProvider",
    "GrokValidation",
    "create_grok41_provider",
    "HybridCryptoOracle",
    "HybridPrediction",
    "get_oracle",
    "get_trading_signals",
    "PaperTradingEngine",
    "Position",
    "ClosedTrade",
    "TradingStats",
    "get_trading_engine",
    # Enhanced multi-venue and whale tracking exports
    "MultiVenueClient",
    "VenuePrice",
    "VenueOrderbook",
    "CrossVenueFeatures",
    "get_cross_venue_snapshot",
    "PolyWhalesClient",
    "WhaleProfile",
    "WhaleTrade",
    "WhaleConsensus",
    "get_whale_signals",
    "EnhancedFeatureSet",
    "FeatureQualityFilter",
    "extract_enhanced_features",
    "extract_enhanced_features_batch",
    "EnhancedGrokProvider",
    "EnhancedGrokValidation",
    "create_enhanced_grok_provider",
    "EnhancedCryptoMLPredictor",
    "EnhancedMLPrediction",
    "EnhancedGradientBoostingPredictor",
    "get_enhanced_predictor",
    "predict_with_whale_signals",
    "EnhancedHybridOracle",
    "EnhancedHybridPrediction",
    "get_enhanced_oracle",
    "predict_with_full_context",
    "EnhancedPaperTradingEngine",
    "EnhancedPosition",
    "EnhancedClosedTrade",
    "EnhancedTradingStats",
    # New 15M tracking and whale ML exports
    "Polymarket15MTracker",
    "Market15MData",
    "PriceUpdate",
    "TopTraderDownloader",
    "TopTrader",
    "TraderTrade",
    "TraderPosition",
    "SmartGrokTrigger",
    "SignalContext",
    "TriggerDecision",
    "TriggerReason",
    "WhalePatternTrainer",
    "WhaleFeatures",
    "TrainingExample",
]
