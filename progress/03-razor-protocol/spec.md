# Task 03 — RazorAttention Retrieval-Head Protocol for Qwen3

Document the protocol for identifying retrieval heads in Qwen3-4B, derived from the RazorAttention method (RazorAttention paper, `papers/razorattention/main.tex`).

## What "done" means

- `spec.md` exists and fully specifies: golden prompt construction, echo/induction score formulas, head-selection thresholds, and NIAH validation plan.

---

## 1. Golden Prompt

**Source**: RazorAttention paper, Section 3.2 (Identification of Retrieval Heads), line 304.

### Construction

1. Generate **K = 2500** random tokens (vocab-sampled, no semantic meaning).
2. Repeat the K-token block **4 times** consecutively.

**Resulting sequence length**: approximately 10,000 tokens (2500 × 4), which fits comfortably within the 24GB VRAM budget when running Qwen3-4B with full attention and attention-map hooks.

### Why this design

The repeating pattern minimizes semantic dependencies between tokens, making the induction-head and echo-head behavior clearly visible in the attention maps. The paper (line 304) validates that this "repeated sequence" produces stable, convergent scores that match those from a semantically meaningful retrieval prompt.

---

## 2. Echo Score (Definition)

**Source**: RazorAttention paper, Section 3.2, Figure 2(b) caption, line 296.

### Informal definition

An **echo head** tends to attend back to a previous token that is **identical** to the current query token. In the repeated-sequence setting, when the current token is `random_t`, the echo head attends strongly to the previous occurrence of `random_t` in the sequence.

### Formal definition

Let the input sequence be tokens `[x_0, x_1, ..., x_{N-1}]` (N = 4 × K). For a given attention head `h`, let the attention weight from position `m` to position `n` be `A_{h,m,n}` (the softmax output, after RoPE).

For each position `m` that has a previous occurrence at position `n < m` where `x_n = x_m`, define the **echo attention** as:

```
echo_attn_{h,m} = A_{h, m, n}   (where n = max{j < m : x_j = x_m})
```

If the current token has no prior occurrence in the sequence, `echo_attn_{h,m} = 0`.

### Per-head echo score

```
echo_score_h = mean_{m in [0, N)} echo_attn_{h,m}
```

The mean is taken over all positions (including those with no prior match, which contribute 0).

---

## 3. Induction Score (Definition)

**Source**: RazorAttention paper, Section 3.2, line 297, Figure 2(b) caption.

### Informal definition

An **induction head** tends to attend to the **next token** after the copy of the current token in the previous context. In the repeated-sequence setting: when the current token is `random_t`, the induction head attends strongly to the token that immediately **follows** the previous occurrence of `random_t` (i.e., `random_{t+1}` in the same block).

### Formal definition

Let the input sequence be tokens `[x_0, x_1, ..., x_{N-1}]` (N = 4 × K). For a given attention head `h`, let `A_{h,m,n}` be the attention weight from position `m` to position `n`.

For each position `m` that is **not the last token of a block** (i.e., `m mod K ≠ K-1`) and whose current token has a prior occurrence at position `n` where `x_n = x_m`, define the **induction target** as position `n+1` (the token immediately after the prior match):

```
induction_attn_{h,m} = A_{h, m, n+1}
```

If the current token has no prior occurrence, or is the last token of a block, `induction_attn_{h,m} = 0`.

### Per-head induction score

```
induction_score_h = mean_{m in [0, N)} induction_attn_{h,m}
```

The mean is taken over all positions.

---

## 4. Head Selection Thresholds

**Source**: RazorAttention paper, Table 2 (hyperparameters), line 332–333.

### Configurations

Three threshold configurations are evaluated, covering a range from aggressive (fewest retrieval heads) to conservative (most retrieval heads):

| Configuration | Induction heads | Echo heads | Total retrieval heads | Rationale |
|--------------|---------------|------------|----------------------|-----------|
| **default**  | Top 14%       | Top 1%    | ~15%                 | RazorAttention paper defaults (Table 2) |
| **medium**   | Top 30%       | Top 2%    | ~32%                 | More retrieval capacity if default shows quality degradation |
| **safe**     | Top 46%       | Top 4%    | ~50%                 | Conservative upper bound; most heads protected |

### Retrieval heads = union of both sets

For a given configuration, the **retrieval head mask** is:
```
retrieval_mask_h = (rank_induction_h ≤ threshold_induction × num_heads) OR (rank_echo_h ≤ threshold_echo × num_heads)
```

where `rank_induction_h` and `rank_echo_h` are the descending ranks of head `h` by induction score and echo score respectively (1 = highest score).

### Notes for Qwen3-4B

- Qwen3-4B uses GQA (grouped query attention). The number of KV heads may be smaller than the number of query heads. RazorAttention handles this by treating each group as a single head for the purpose of scoring (line 449: "we consider the attention heads in a group as all retrieval if one or more heads satisfy inductive or echoing property"). **For implementation, score each (Q-head, KV-head) pair separately, then propagate retrieval status to the entire KV group.**
- The 14%/1% thresholds are taken directly from the RazorAttention paper. The 30%/2% and 46%/4% values are provided as intermediate and conservative options if the paper's defaults prove insufficient for Qwen3.

---

## 5. NIAH Validation Plan

**Source**: RazorAttention paper, Section 4.2 (Needle In A Haystack), Figure 4, Table 3.

### Objective

Validate that the identified retrieval-head masks (all three configurations) preserve model quality on long-context retrieval before converting heads to linear attention.

### Dataset

Use the standard **Needle-in-a-Haystack** (NIAH) benchmark, specifically the `PaulGrahamEssays` version (consistent with Task 02 baseline).

### Protocol

1. **Full-attention baseline**: Run NIAH at input lengths {4k, 8k, 16k, 32k}, depth_rel ∈ {0, 0.5, 1}. Record accuracy. (This is already done in Task 02.)

2. **Per-configuration retrieval protection**: For each of the three configurations (default / medium / safe), apply the corresponding retrieval-head mask. Keep only the identified retrieval heads as full attention; replace all other heads with a **sliding-window baseline** (keep recent tokens + attention sinks, no compensation token yet). Run the same NIAH sweep for each configuration.

3. **Random-head baseline** (optional ablation): For each configuration, generate a random mask with the same number of heads as the retrieval mask. Run NIAH. This confirms that retrieval-head selection is meaningful, not just the fraction of protected heads.

### Success criterion

Accuracy on NIAH with retrieval-head protection should be ≥ 95% of full-attention accuracy at each (input_len, depth_rel) configuration, for **all three** threshold configurations.

### Reporting

Report per-configuration accuracy tables (rows = configuration, columns = input_len) and note any configurations where the gap vs. full attention exceeds 5%.
