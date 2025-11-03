# Model Interpretability Features

This document describes the model instrumentation and interpretability features added to NanoChat.

## Overview

The interpretability system allows you to capture and analyze the internal workings of the model during inference, including:

- **Attention weights**: See what tokens each attention head focuses on
- **Layer activations**: Capture pre/post attention and MLP activations
- **Logits and embeddings**: Store output logits and input token embeddings
- **Step-by-step tracing**: Option to capture data at each generation step

## Components

### 1. Model Instrumentation (`nanochat/gpt.py`)

The model has been instrumented to capture:

- `CausalSelfAttention.attention_weights`: Attention matrices [B, H, T, T]
- `Block.pre_attn_norm`, `Block.post_attn`: Pre/post attention activations
- `Block.pre_mlp_norm`, `Block.post_mlp`: Pre/post MLP activations
- `GPT.token_embeddings`: Input token embeddings [B, T, D]
- `GPT.output_logits`: Output logits [B, T, V]

### 2. Trace Storage (`nanochat/trace_storage.py`)

`TraceStorage` class handles persistence of trace data:

- **JSON format**: All data stored in human-readable JSON files
- **Structured data**: Prompts, responses, categories, and numeric arrays
- **Timestamp-based IDs**: Automatic unique trace identification
- **Categorical organization**: Filter traces by category
- **Single file per trace**: Everything stored in one `{trace_id}.json` file

### 3. Interpretability Engine (`nanochat/interpretability_engine.py`)

`InterpretabilityEngine` provides the main API:

- `generate_and_capture()`: Generate text while capturing model internals
- `analyze_trace()`: Analyze captured traces and compute statistics
- Support for intermediate step-by-step capture
- Automatic attention entropy and sparsity computation
- Layer-wise activation statistics

## Usage Examples

### Basic Usage

```python
from nanochat.interpretability_engine import InterpretabilityEngine
from nanochat.checkpoint_manager import load_model

# Load model and create engine
model, tokenizer, meta = load_model("sft", device, phase="eval")
engine = InterpretabilityEngine(model, tokenizer, "traces")

# Generate and capture
result = engine.generate_and_capture(
    prompt="The capital of France is",
    category="geography",
    max_tokens=50,
    temperature=0.6
)

print(f"Response: {result['response']}")
print(f"Trace ID: {result['trace_id']}")
```

### Analysis

```python
# Analyze the captured trace
analysis = engine.analyze_trace(result['trace_id'])

# Access attention statistics
attention_stats = analysis['analysis']['attention']['layer_stats']
for layer_idx, stats in attention_stats.items():
    print(f"Layer {layer_idx}: entropy={stats['attention_entropy']:.4f}")

# Access activation statistics
activation_stats = analysis['analysis']['activations']['layer_stats']
for layer_idx, stats in activation_stats.items():
    print(f"Layer {layer_idx}: pre_mlp sparsity={stats['pre_mlp_norm']['sparsity']:.4f}")
```

### Step-by-Step Capture

```python
# Capture data at each generation step
result = engine.generate_and_capture(
    prompt="Explain photosynthesis step by step:",
    capture_intermediate_steps=True,  # Enable step capture
    max_tokens=100
)

# The HDF5 file will now contain an "intermediate_steps" group
# with attention weights and activations for each generation step
```

### Loading and Analyzing Existing Traces

```python
from nanochat.trace_storage import TraceStorage

storage = TraceStorage("traces")

# List all traces
traces = storage.list_traces(category="geography")
print(f"Found {len(traces)} geography traces")

# Load complete trace data (JSON)
trace_data = storage.load_trace(traces[0])
print(f"Prompt: {trace_data['prompt']}")
print(f"Response: {trace_data['response']}")

# Access attention weights
attention_weights = trace_data['model_data']['attention_weights']['layer_0']
attention_list = attention_weights['data']  # List of attention values
shape = attention_weights['shape']  # [B, H, T, T]

# Get trace summary
summary = storage.get_trace_summary(traces[0])
print(f"Trace info: {summary['data_info']}")
```

## Command Line Demo

Use the provided demo script:

```bash
python -m scripts.interpretability_demo \
    --prompt "The chemical formula of water is" \
    --category "science" \
    --max-tokens 50 \
    --temperature 0.6 \
    --capture-steps
```

This will:
1. Generate a response
2. Capture all model internals
3. Print analysis statistics
4. Save trace data to `traces/` directory

## Trace File Structure

Each trace creates a single JSON file: **`{trace_id}.json`**

The JSON structure contains all data in one file:

```json
{
  "trace_id": "trace_20241102_143022_123456",
  "timestamp": "2024-11-02T14:30:22.123456",
  "prompt": "The capital of France is",
  "response": " Paris.",
  "category": "geography",
  "metadata": {
    "model_source": "sft",
    "temperature": 0.6
  },
  "tokens": {
    "input_tokens": [1, 464, 437, 322, 310],
    "output_tokens": [220, 13]
  },
  "model_data": {
    "token_embeddings": {
      "data": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
      "shape": [1, 5, 768],
      "dtype": "torch.float32"
    },
    "output_logits": {
      "data": [[-1.2, 0.5, ...], [0.8, -0.3, ...]],
      "shape": [1, 2, 50304],
      "dtype": "torch.float32"
    },
    "attention_weights": {
      "layer_0": {
        "data": [[[0.1, 0.9], [0.8, 0.2]], ...],
        "shape": [1, 12, 5, 5],
        "dtype": "torch.float32"
      },
      "layer_1": {...}
    },
    "layer_activations": {
      "layer_0": {
        "pre_attn_norm": {"data": [...], "shape": [...], "dtype": "..."},
        "post_attn": {"data": [...], "shape": [...], "dtype": "..."},
        "pre_mlp_norm": {"data": [...], "shape": [...], "dtype": "..."},
        "post_mlp": {"data": [...], "shape": [...], "dtype": "..."}
      },
      "layer_1": {...}
    }
  },
  "intermediate_steps": [
    {
      "step": 0,
      "token_embeddings": {...},
      "output_logits": {...},
      "attention_weights": {...},
      "layer_activations": {...}
    },
    {
      "step": 1,
      ...
    }
  ]
}
```

## Analysis Metrics

The system automatically computes several metrics:

### Attention Metrics
- **Mean attention**: Average attention weight per layer
- **Attention entropy**: Information content of attention distributions
- **Head sparsity**: Fraction of attention weights below threshold

### Activation Metrics
- **Mean/std/max/min**: Basic statistics per activation type
- **Sparsity**: Fraction of near-zero activations

### Logits Metrics
- **Mean/std**: Basic statistics
- **Entropy per position**: Information content of output distributions
- **Top tokens per position**: Most likely tokens at each step

## Memory Considerations

- All captured data is moved to CPU to avoid GPU memory issues
- Large traces (especially with step-by-step capture) can use significant disk space as JSON files
- JSON format is human-readable but less storage-efficient than binary formats

## Advantages of JSON Format

- **Human readable**: You can open and inspect trace files in any text editor
- **No external dependencies**: Only requires standard Python json module
- **Easy to process**: Simple to parse and manipulate in any programming language
- **Debugging friendly**: Can easily examine intermediate values during development
- **Version control friendly**: Text files work better with git diffs

## Troubleshooting

**Out of Memory**: If you encounter memory issues:
- Use smaller `max_tokens` values
- Avoid `capture_intermediate_steps=True` for long generations
- Monitor disk space in the traces directory

**Missing Data**: If some captures are None:
- Ensure you're using the instrumented model
- Call `engine.clear_model_captures()` before generation
- Check that inference mode is enabled

**Performance**: The instrumentation adds minimal overhead (~5-10% slower generation) but step-by-step capture can significantly slow down generation.

**Large JSON Files**: For very long generations, JSON files can become large (tens of MB). If this becomes an issue:
- Reduce `max_tokens`
- Disable step-by-step capture
- Consider processing traces in chunks