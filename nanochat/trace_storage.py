"""
Trace storage utilities for model interpretability.

This module provides functionality to store model traces (attention weights,
activations, logits, embeddings) entirely in JSON format.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


def tensor_to_list(tensor: torch.Tensor) -> List:
    """Convert PyTorch tensor to JSON-serializable list."""
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy().tolist()


def tensor_to_dict(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Convert PyTorch tensor to dictionary with metadata."""
    if tensor is None:
        return None
    return {
        "data": tensor_to_list(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype)
    }


class TraceStorage:
    """
    Storage class for model interpretability traces.

    Stores all data (attention weights, activations, logits, embeddings, metadata)
    in JSON format for easy accessibility and debugging.
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

        # Create complete trace structure
        trace_data = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "category": category,
            "metadata": metadata or {},
            "tokens": {
                "input_tokens": [],
                "output_tokens": []
            },
            "model_data": {
                "token_embeddings": None,
                "output_logits": None,
                "attention_weights": {},
                "layer_activations": {}
            },
            "intermediate_steps": None
        }

        # Save as single JSON file
        trace_file = self.base_dir / f"{trace_id}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)

        return trace_id

    def save_activations(self,
                        trace_id: str,
                        token_embeddings: torch.Tensor,
                        output_logits: torch.Tensor,
                        attention_weights: List[torch.Tensor],
                        layer_activations: List[Dict[str, torch.Tensor]]):
        """
        Save model activations to JSON file.

        Args:
            trace_id: Trace identifier
            token_embeddings: Input token embeddings [B, T, D]
            output_logits: Output logits [B, T, V]
            attention_weights: List of attention weights per layer [L][B, H, T, T]
            layer_activations: List of activation dicts per layer
        """
        trace_file = self.base_dir / f"{trace_id}.json"

        # Load existing trace data
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Save embeddings and logits
        if token_embeddings is not None:
            trace_data["model_data"]["token_embeddings"] = tensor_to_dict(token_embeddings, "token_embeddings")
        if output_logits is not None:
            trace_data["model_data"]["output_logits"] = tensor_to_dict(output_logits, "output_logits")

        # Save attention weights
        for layer_idx, attn_weights in enumerate(attention_weights):
            if attn_weights is not None:
                trace_data["model_data"]["attention_weights"][f"layer_{layer_idx}"] = tensor_to_dict(
                    attn_weights, f"attention_weights_layer_{layer_idx}"
                )

        # Save layer activations
        for layer_idx, activations in enumerate(layer_activations):
            if activations:
                layer_data = {}
                for name, activation in activations.items():
                    if activation is not None:
                        layer_data[name] = tensor_to_dict(activation, f"{name}_layer_{layer_idx}")
                trace_data["model_data"]["layer_activations"][f"layer_{layer_idx}"] = layer_data

        # Save back to file
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)

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
        trace_file = self.base_dir / f"{trace_id}.json"

        # Load existing trace data
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Update tokens
        trace_data["tokens"]["input_tokens"] = input_tokens
        trace_data["tokens"]["output_tokens"] = output_tokens

        # Save back to file
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)

    def save_intermediate_steps(self, trace_id: str, step_captures: List[Dict[str, Any]]):
        """Save intermediate step captures."""
        trace_file = self.base_dir / f"{trace_id}.json"

        # Load existing trace data
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Convert step captures to JSON format
        intermediate_steps = []
        for step_idx, step_capture in enumerate(step_captures):
            step_data = {
                "step": step_idx,
                "token_embeddings": tensor_to_dict(step_capture["token_embeddings"], f"step_{step_idx}_token_embeddings"),
                "output_logits": tensor_to_dict(step_capture["output_logits"], f"step_{step_idx}_output_logits"),
                "attention_weights": {},
                "layer_activations": {}
            }

            # Save attention weights for this step
            for layer_idx, attn_weights in enumerate(step_capture["attention_weights"]):
                if attn_weights is not None:
                    step_data["attention_weights"][f"layer_{layer_idx}"] = tensor_to_dict(
                        attn_weights, f"step_{step_idx}_attention_weights_layer_{layer_idx}"
                    )

            # Save layer activations for this step
            for layer_idx, activations in enumerate(step_capture["layer_activations"]):
                if activations:
                    layer_data = {}
                    for name, activation in activations.items():
                        if activation is not None:
                            layer_data[name] = tensor_to_dict(
                                activation, f"step_{step_idx}_{name}_layer_{layer_idx}"
                            )
                    step_data["layer_activations"][f"layer_{layer_idx}"] = layer_data

            intermediate_steps.append(step_data)

        trace_data["intermediate_steps"] = intermediate_steps

        # Save back to file
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)

    def load_trace(self, trace_id: str) -> Dict[str, Any]:
        """Load complete trace data."""
        trace_file = self.base_dir / f"{trace_id}.json"
        with open(trace_file, 'r') as f:
            return json.load(f)

    def load_trace_metadata(self, trace_id: str) -> Dict[str, Any]:
        """Load metadata for a given trace."""
        trace_data = self.load_trace(trace_id)
        return {
            "trace_id": trace_data["trace_id"],
            "timestamp": trace_data["timestamp"],
            "prompt": trace_data["prompt"],
            "response": trace_data["response"],
            "category": trace_data["category"],
            "metadata": trace_data["metadata"]
        }

    def load_trace_data(self, trace_id: str) -> Dict[str, Any]:
        """Load model data for a given trace."""
        trace_data = self.load_trace(trace_id)
        return trace_data["model_data"]

    def list_traces(self, category: Optional[str] = None) -> List[str]:
        """List all traces, optionally filtered by category."""
        traces = []

        for trace_file in self.base_dir.glob("trace_*.json"):
            trace_id = trace_file.stem

            if category is not None:
                metadata = self.load_trace_metadata(trace_id)
                if metadata.get("category") == category:
                    traces.append(trace_id)
            else:
                traces.append(trace_id)

        return sorted(traces)

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of trace data without loading large arrays."""
        trace_data = self.load_trace(trace_id)

        summary = {
            "metadata": {
                "trace_id": trace_data["trace_id"],
                "timestamp": trace_data["timestamp"],
                "prompt": trace_data["prompt"],
                "response": trace_data["response"],
                "category": trace_data["category"]
            },
            "data_info": {}
        }

        # Add shape information for tensors
        model_data = trace_data.get("model_data", {})
        for key, value in model_data.items():
            if isinstance(value, dict) and "shape" in value:
                summary["data_info"][key] = {
                    "shape": value["shape"],
                    "dtype": value["dtype"]
                }
            elif isinstance(value, dict):
                summary["data_info"][key] = f"Dictionary with {len(value)} items"

        # Add token counts
        tokens = trace_data.get("tokens", {})
        summary["data_info"]["input_token_count"] = len(tokens.get("input_tokens", []))
        summary["data_info"]["output_token_count"] = len(tokens.get("output_tokens", []))

        # Add intermediate step info
        if trace_data.get("intermediate_steps"):
            summary["data_info"]["intermediate_steps_count"] = len(trace_data["intermediate_steps"])

        return summary