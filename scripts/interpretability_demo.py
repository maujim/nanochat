#!/usr/bin/env python3
"""
Demo script for using the InterpretabilityEngine.

This script shows how to use the model instrumentation to capture and analyze
model internals during inference.
"""

import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.interpretability_engine import InterpretabilityEngine
from nanochat.checkpoint_manager import load_model


def main():
    parser = argparse.ArgumentParser(description='Model Interpretability Demo')
    parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-p', '--prompt', type=str, default='The capital of France is', help='Prompt for generation')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--category', type=str, default='demo', help='Category for the trace')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--capture-steps', action='store_true', help='Capture intermediate generation steps')
    parser.add_argument('--traces-dir', type=str, default='traces', help='Directory to store traces')
    args = parser.parse_args()

    # Initialize model and tokenizer
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print(f"Loading model from source: {args.source}")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

    # Create interpretability engine
    engine = InterpretabilityEngine(model, tokenizer, args.traces_dir)

    print(f"\nPrompt: {args.prompt}")
    print("Generating response and capturing model internals...")

    # Generate and capture
    with autocast_ctx:
        result = engine.generate_and_capture(
            prompt=args.prompt,
            category=args.category,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            capture_intermediate_steps=args.capture_steps,
            metadata={
                "model_source": args.source,
                "device_type": device_type,
                "dtype": args.dtype
            }
        )

    print(f"\nResponse: {result['response']}")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Generated {len(result['response_tokens'])} tokens")

    # Analyze the trace
    print("\nAnalyzing trace...")
    analysis = engine.analyze_trace(result['trace_id'])

    # Print analysis results
    if 'attention' in analysis['analysis']:
        print(f"\nAttention Analysis ({analysis['analysis']['attention']['num_layers']} layers):")
        for layer_idx, stats in analysis['analysis']['attention']['layer_stats'].items():
            print(f"  Layer {layer_idx}: mean={stats['mean_attention']:.4f}, "
                  f"entropy={stats['attention_entropy']:.4f}, sparsity={stats['head_sparsity']:.4f}")

    if 'activations' in analysis['analysis']:
        print(f"\nActivation Analysis:")
        for layer_idx, stats in analysis['analysis']['activations']['layer_stats'].items():
            print(f"  Layer {layer_idx}:")
            for activation_type, act_stats in stats.items():
                print(f"    {activation_type}: mean={act_stats['mean']:.4f}, "
                      f"sparsity={act_stats['sparsity']:.4f}")

    if 'logits' in analysis['analysis']:
        print(f"\nLogits Analysis:")
        logits_stats = analysis['analysis']['logits']
        print(f"  Mean logit: {logits_stats['mean_logit']:.4f}")
        print(f"  Logit std: {logits_stats['std_logit']:.4f}")

    print(f"\nTrace data saved to: {args.traces_dir}/")
    print("You can load and analyze traces later using the InterpretabilityEngine.load_trace_data() method.")


if __name__ == "__main__":
    main()