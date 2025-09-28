"""
Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

This package implements the Compact Attention framework and its enhanced variants
for efficient video generation with transformer models.
"""

from .compact_attention import CompactAttention, CompactAttentionConfig
from .adaptive_compact_attention import (
    AdaptiveCompactAttention, 
    AdaptiveCompactAttentionConfig,
    AdaptiveThresholdController,
    NeuralPatternExtractor,
    MotionAwareTemporalAttention,
    DistributedCompactAttention
)

__version__ = "1.0.0"
__all__ = [
    "CompactAttention",
    "CompactAttentionConfig", 
    "AdaptiveCompactAttention",
    "AdaptiveCompactAttentionConfig",
    "AdaptiveThresholdController",
    "NeuralPatternExtractor", 
    "MotionAwareTemporalAttention",
    "DistributedCompactAttention"
]