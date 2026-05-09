"""
Compatibility wrapper for the unsupervised MLP implementation.

The active experiment runner now selects between:
- `src.ff_snn_mlp_unsup` for unsupervised experiments
- `src.ff_snn_mlp_sup` for supervised experiments

This module stays as a thin re-export so older imports do not break.
"""

from src.ff_snn_mlp_unsup import Layer, Net, OutputLayer, spike_encoder, tdLayerNorm

__all__ = [
    "Layer",
    "Net",
    "OutputLayer",
    "spike_encoder",
    "tdLayerNorm",
]
