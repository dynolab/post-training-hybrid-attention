# Task 04 — Retrieval-Head Scoring Report

## Masks (not committed)

Masks are stored externally. Link to location: TBD

| File | Configuration | Description |
|------|--------------|-------------|
| `<path>/retrieval_mask_default.npy` | 14% induction + 1% echo | Default RazorAttention thresholds |
| `<path>/retrieval_mask_medium.npy` | 30% induction + 2% echo | More conservative |
| `<path>/retrieval_mask_safe.npy` | 46% induction + 4% echo | Most conservative |

## Array format

Each mask is a **1D numpy boolean array** of shape `(num_query_heads,)`, where `num_query_heads` is the number of query attention heads in Qwen3-4B.

- Index `h` corresponds to query head `h` in layer order (layer 0 heads 0..H-1, layer 1 heads H..2H-1, etc.).
- `True` = this head is a retrieval head (must keep full attention).
- `False` = this head is non-retrieval (can be linearized / compressed).

**GQA handling**: Qwen3-4B uses grouped query attention (fewer KV heads than query heads). A KV group is marked as retrieval if **any** of its constituent query heads is marked retrieval. The implementation must propagate the per-query-head mask to per-KV-group before applying compression.

## NIAH Results

### Default (14% induction + 1% echo)

| input_len | depth_rel | accuracy | notes |
|----------:|----------:|---------:|------|
| 4k        | 0.00      |          |      |
| 4k        | 0.50      |          |      |
| 4k        | 1.00      |          |      |
| 8k        | 0.00      |          |      |
| 8k        | 0.50      |          |      |
| 8k        | 1.00      |          |      |
| 16k       | 0.00      |          |      |
| 16k       | 0.50      |          |      |
| 16k       | 1.00      |          |      |
| 32k       | 0.00      |          |      |
| 32k       | 0.50      |          |      |
| 32k       | 1.00      |          |      |

### Medium (30% induction + 2% echo)

| input_len | depth_rel | accuracy | notes |
|----------:|----------:|---------:|------|
| 4k        | 0.00      |          |      |
| 4k        | 0.50      |          |      |
| 4k        | 1.00      |          |      |
| 8k        | 0.00      |          |      |
| 8k        | 0.50      |          |      |
| 8k        | 1.00      |          |      |
| 16k       | 0.00      |          |      |
| 16k       | 0.50      |          |      |
| 16k       | 1.00      |          |      |
| 32k       | 0.00      |          |      |
| 32k       | 0.50      |          |      |
| 32k       | 1.00      |          |      |

### Safe (46% induction + 4% echo)

| input_len | depth_rel | accuracy | notes |
|----------:|----------:|---------:|------|
| 4k        | 0.00      |          |      |
| 4k        | 0.50      |          |      |
| 4k        | 1.00      |          |      |
| 8k        | 0.00      |          |      |
| 8k        | 0.50      |          |      |
| 8k        | 1.00      |          |      |
| 16k       | 0.00      |          |      |
| 16k       | 0.50      |          |      |
| 16k       | 1.00      |          |      |
| 32k       | 0.00      |          |      |
| 32k       | 0.50      |          |      |
| 32k       | 1.00      |          |      |

## Summary

Which configurations meet the ≥ 95% criterion:

- **default**: TBD
- **medium**: TBD
- **safe**: TBD

Recommended configuration for downstream tasks: TBD
