"""
InterpretabilityEngine for capturing and analyzing model internals.

This module provides a wrapper around the model that runs inference and captures
attention weights, activations, logits, and embeddings for analysis.
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
from nanochat.engine import Engine
from nanochat.trace_storage import TraceStorage
from nanochat.tokenizer import Tokenizer


class InterpretabilityEngine:
    """
    Engine for capturing model internals during inference.

    Wraps the standard Engine to capture attention weights, layer activations,
    logits, and embeddings, storing them in a structured format for analysis.
    """

    def __init__(self, model, tokenizer: Tokenizer, traces_dir: str = "traces"):
        """
        Initialize the InterpretabilityEngine.

        Args:
            model: The GPT model to instrument
            tokenizer: Tokenizer for encoding/decoding
            traces_dir: Directory to store trace data
        """
        self.model = model
        self.tokenizer = tokenizer
        self.storage = TraceStorage(traces_dir)
        self.engine = Engine(model, tokenizer)

    def clear_model_captures(self):
        """Clear all captured data from the model."""
        # Clear model-level captures
        self.model.token_embeddings = None
        self.model.output_logits = None

        # Clear attention weights
        for block in self.model.transformer.h:
            block.attn.attention_weights = None

        # Clear layer activations
        for block in self.model.transformer.h:
            block.pre_attn_norm = None
            block.post_attn = None
            block.pre_mlp_norm = None
            block.post_mlp = None

    def extract_model_captures(self) -> Dict[str, Any]:
        """Extract all captured data from the model."""
        captures = {
            "token_embeddings": self.model.token_embeddings,
            "output_logits": self.model.output_logits,
            "attention_weights": [],
            "layer_activations": []
        }

        # Extract attention weights from each layer
        for layer_idx, block in enumerate(self.model.transformer.h):
            captures["attention_weights"].append(block.attn.attention_weights)

            # Extract layer activations
            layer_activations = {
                "pre_attn_norm": block.pre_attn_norm,
                "post_attn": block.post_attn,
                "pre_mlp_norm": block.pre_mlp_norm,
                "post_mlp": block.post_mlp
            }
            captures["layer_activations"].append(layer_activations)

        return captures

    @torch.inference_mode()
    def generate_and_capture(self,
                           prompt: str,
                           category: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           max_tokens: int = 256,
                           temperature: float = 0.6,
                           top_k: int = 50,
                           seed: int = 42,
                           capture_intermediate_steps: bool = False) -> Dict[str, Any]:
        """
        Generate text and capture model internals.

        Args:
            prompt: Input prompt text
            category: Optional category for labeling the trace
            metadata: Additional metadata to store
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed
            capture_intermediate_steps: Whether to capture data at each generation step

        Returns:
            Dictionary containing generation results and trace information
        """
        # Encode the prompt
        bos_token_id = self.tokenizer.get_bos_token_id()
        prompt_tokens = [bos_token_id] + self.tokenizer.encode(prompt)

        # Create trace ID and metadata
        trace_id = self.storage.create_trace(
            prompt=prompt,
            response="",  # Will be filled after generation
            category=category,
            metadata=metadata
        )

        # Clear any existing captures
        self.clear_model_captures()

        # Generate tokens and capture intermediate data if requested
        all_generated_tokens = []
        step_captures = []

        for token_column, token_masks in self.engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed
        ):
            token = token_column[0]  # Get first (and only) sample
            all_generated_tokens.append(token)

            # Capture intermediate step data if requested
            if capture_intermediate_steps:
                step_capture = self.extract_model_captures()
                step_captures.append(step_capture)
                # Clear captures for next step
                self.clear_model_captures()

            # Early stop if we hit an end token
            if token == self.tokenizer.encode_special("<|assistant_end|>"):
                break

        # Get final captures
        final_captures = self.extract_model_captures()

        # Decode the response
        response_text = self.tokenizer.decode(all_generated_tokens)

        # Update trace metadata with the actual response
        trace_metadata = self.storage.load_trace_metadata(trace_id)
        trace_metadata["response"] = response_text
        trace_metadata["generation_params"] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed
        }

        # Save metadata back
        import json
        from pathlib import Path
        metadata_file = Path(self.storage.base_dir) / f"{trace_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(trace_metadata, f, indent=2)

        # Save tokens
        self.storage.save_tokens(trace_id, prompt_tokens, all_generated_tokens)

        # Save final activations
        self.storage.save_activations(
            trace_id=trace_id,
            token_embeddings=final_captures["token_embeddings"],
            output_logits=final_captures["output_logits"],
            attention_weights=final_captures["attention_weights"],
            layer_activations=final_captures["layer_activations"]
        )

        # Save intermediate captures if collected
        if capture_intermediate_steps and step_captures:
            self.storage.save_intermediate_steps(trace_id, step_captures)

        return {
            "trace_id": trace_id,
            "prompt": prompt,
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": all_generated_tokens,
            "metadata": trace_metadata
        }

    
    def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Analyze a saved trace and return insights.

        Args:
            trace_id: Trace identifier to analyze

        Returns:
            Analysis results including attention patterns and activation statistics
        """
        trace_data = self.storage.load_trace(trace_id)
        metadata = self.storage.load_trace_metadata(trace_id)
        model_data = trace_data["model_data"]

        analysis = {
            "trace_id": trace_id,
            "metadata": metadata,
            "analysis": {}
        }

        # Analyze attention patterns
        if "attention_weights" in model_data and model_data["attention_weights"]:
            attention_analysis = self._analyze_attention_patterns(model_data["attention_weights"])
            analysis["analysis"]["attention"] = attention_analysis

        # Analyze activation statistics
        if "layer_activations" in model_data and model_data["layer_activations"]:
            activation_analysis = self._analyze_activation_patterns(model_data["layer_activations"])
            analysis["analysis"]["activations"] = activation_analysis

        # Analyze logits
        if "output_logits" in model_data and model_data["output_logits"]:
            logits_analysis = self._analyze_logits(model_data["output_logits"])
            analysis["analysis"]["logits"] = logits_analysis

        return analysis

    def _analyze_attention_patterns(self, attention_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze attention patterns across layers."""
        patterns = {
            "num_layers": len([k for k in attention_data.keys() if k.startswith("layer_")]),
            "layer_stats": {}
        }

        for layer_name, attn_weights in attention_data.items():
            if layer_name.startswith("layer_") and attn_weights and "data" in attn_weights:
                layer_idx = layer_name.split("_")[1]
                attn_list = attn_weights["data"]
                attn_tensor = torch.tensor(attn_list)

                # Compute statistics
                patterns["layer_stats"][layer_idx] = {
                    "mean_attention": float(attn_tensor.mean()),
                    "max_attention": float(attn_tensor.max()),
                    "attention_entropy": self._compute_entropy(attn_tensor),
                    "head_sparsity": self._compute_head_sparsity(attn_tensor)
                }

        return patterns

    def _analyze_activation_patterns(self, activation_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activation patterns across layers."""
        patterns = {
            "layer_stats": {}
        }

        for layer_name, layer_data in activation_data.items():
            if layer_name.startswith("layer_"):
                layer_idx = layer_name.split("_")[1]
                layer_stats = {}

                for activation_type, activation in layer_data.items():
                    if activation and "data" in activation:
                        activation_list = activation["data"]
                        activation_tensor = torch.tensor(activation_list)

                        layer_stats[activation_type] = {
                            "mean": float(activation_tensor.mean()),
                            "std": float(activation_tensor.std()),
                            "max": float(activation_tensor.max()),
                            "min": float(activation_tensor.min()),
                            "sparsity": float((activation_tensor.abs() < 1e-6).float().mean())
                        }

                patterns["layer_stats"][layer_idx] = layer_stats

        return patterns

    def _analyze_logits(self, logits_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze output logits."""
        if not logits_data or "data" not in logits_data:
            return {}

        logits_list = logits_data["data"]
        logits = torch.tensor(logits_list)

        return {
            "mean_logit": float(logits.mean()),
            "std_logit": float(logits.std()),
            "entropy_per_position": self._compute_entropy(logits.softmax(dim=-1)).tolist(),
            "top_tokens_per_position": [
                torch.topk(logits[pos], k=5).indices.tolist()
                for pos in range(logits.shape[0])
            ]
        }

    def _compute_entropy(self, probs: torch.Tensor, eps: float = 1e-8) -> float:
        """Compute entropy of probability distribution."""
        return float(-torch.sum(probs * torch.log(probs + eps), dim=-1).mean())

    def _compute_head_sparsity(self, attention: torch.Tensor, threshold: float = 0.01) -> float:
        """Compute how sparse attention heads are."""
        # attention shape: [B, H, T, T]
        return float((attention < threshold).float().mean())