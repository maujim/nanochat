"""
Trace storage utilities for model interpretability.

This module provides functionality to store model traces (attention weights,
activations, logits, embeddings) in HDF5 format with JSON metadata.
"""

import json
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


class TraceStorage:
    """
    Storage class for model interpretability traces.

    Stores numeric data (attention weights, activations, logits, embeddings) in HDF5
    format and metadata (prompts, responses, categories) in JSON.
    """

    def __init__(self, base_dir: str = "traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_trace(self,
                     prompt: str,
                     response: str,
                     category: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trace file and return the trace ID.

        Args:
            prompt: Input prompt text
            response: Model response text
            category: Optional category for the trace
            metadata: Additional metadata dictionary

        Returns:
            Trace ID (timestamp-based)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_id = f"trace_{timestamp}"

        # Create metadata
        trace_metadata = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "category": category,
            "metadata": metadata or {}
        }

        # Save metadata as JSON
        metadata_file = self.base_dir / f"{trace_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(trace_metadata, f, indent=2)

        # Create HDF5 file for numeric data
        hdf5_file = self.base_dir / f"{trace_id}_data.h5"
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs["trace_id"] = trace_id
            f.attrs["created_at"] = timestamp

        return trace_id

    def save_activations(self,
                        trace_id: str,
                        token_embeddings: torch.Tensor,
                        output_logits: torch.Tensor,
                        attention_weights: List[torch.Tensor],
                        layer_activations: List[Dict[str, torch.Tensor]]):
        """
        Save model activations to HDF5 file.

        Args:
            trace_id: Trace identifier
            token_embeddings: Input token embeddings [B, T, D]
            output_logits: Output logits [B, T, V]
            attention_weights: List of attention weights per layer [L][B, H, T, T]
            layer_activations: List of activation dicts per layer
        """
        hdf5_file = self.base_dir / f"{trace_id}_data.h5"

        with h5py.File(hdf5_file, 'a') as f:
            # Save embeddings and logits
            if token_embeddings is not None:
                f.create_dataset("token_embeddings", data=token_embeddings.numpy())
            if output_logits is not None:
                f.create_dataset("output_logits", data=output_logits.numpy())

            # Save attention weights
            attention_group = f.create_group("attention_weights")
            for layer_idx, attn_weights in enumerate(attention_weights):
                if attn_weights is not None:
                    attention_group.create_dataset(f"layer_{layer_idx}", data=attn_weights.numpy())

            # Save layer activations
            activations_group = f.create_group("layer_activations")
            for layer_idx, activations in enumerate(layer_activations):
                if activations:
                    layer_group = activations_group.create_group(f"layer_{layer_idx}")
                    for name, activation in activations.items():
                        if activation is not None:
                            layer_group.create_dataset(name, data=activation.numpy())

    def save_tokens(self,
                   trace_id: str,
                   input_tokens: List[int],
                   output_tokens: List[int]):
        """
        Save input and output tokens.

        Args:
            trace_id: Trace identifier
            input_tokens: Input token IDs
            output_tokens: Generated token IDs
        """
        hdf5_file = self.base_dir / f"{trace_id}_data.h5"

        with h5py.File(hdf5_file, 'a') as f:
            f.create_dataset("input_tokens", data=np.array(input_tokens, dtype=np.int32))
            f.create_dataset("output_tokens", data=np.array(output_tokens, dtype=np.int32))

    def load_trace_metadata(self, trace_id: str) -> Dict[str, Any]:
        """Load metadata for a given trace."""
        metadata_file = self.base_dir / f"{trace_id}_metadata.json"
        with open(metadata_file, 'r') as f:
            return json.load(f)

    def load_trace_data(self, trace_id: str) -> Dict[str, np.ndarray]:
        """Load numeric data for a given trace."""
        hdf5_file = self.base_dir / f"{trace_id}_data.h5"
        data = {}

        with h5py.File(hdf5_file, 'r') as f:
            def collect_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
            f.visititems(collect_data)

        return data

    def list_traces(self, category: Optional[str] = None) -> List[str]:
        """List all traces, optionally filtered by category."""
        traces = []

        for metadata_file in self.base_dir.glob("*_metadata.json"):
            trace_id = metadata_file.stem.replace("_metadata", "")

            if category is not None:
                metadata = self.load_trace_metadata(trace_id)
                if metadata.get("category") == category:
                    traces.append(trace_id)
            else:
                traces.append(trace_id)

        return sorted(traces)

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of trace data without loading large arrays."""
        metadata = self.load_trace_metadata(trace_id)

        hdf5_file = self.base_dir / f"{trace_id}_data.h5"
        summary = {"metadata": metadata}

        with h5py.File(hdf5_file, 'r') as f:
            data_info = {}

            def collect_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_info[name] = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype)
                    }
            f.visititems(collect_info)

            summary["data_info"] = data_info

        return summary