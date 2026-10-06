# ViT Reverse Engineering Lab

## Project Title

**Inside the Vision Transformer: A From-Scratch, Layer-by-Layer Reverse Engineering and Benchmarking Study**

### Short Description

Build a Vision Transformer (ViT) from first principles, reverse-engineer a trusted reference implementation, verify numerical equivalence layer-by-layer, visualize internal attention behavior, benchmark computational efficiency, and perform controlled architectural ablations.

The project should demonstrate genuine understanding of Vision Transformer internals rather than simply wrapping an existing pretrained model.

---

# 1. Project Goals

The project has five primary goals:

1. **Understand ViT mathematically**
2. **Implement ViT from scratch using PyTorch**
3. **Reverse-engineer and reproduce a reference implementation**
4. **Inspect and visualize internal model behavior**
5. **Perform meaningful engineering/research experiments**

The final result should be strong enough for an AI Engineer / Computer Vision / ML Research portfolio.

The project should emphasize:

> Understand → Implement → Verify → Inspect → Benchmark → Experiment

---

# 2. Core Research Question

The central question is:

> **Can we reproduce the behavior of a Vision Transformer from its fundamental components and explain how each architectural decision affects performance, computation, and representation?**

Secondary questions:

- How does patch size affect performance and computation?
- How does the number of attention heads affect representation?
- What does each attention head learn?
- How important are positional embeddings?
- What role does the CLS token play?
- How do residual connections affect optimization?
- How does ViT computational cost scale with image resolution?
- How closely can a from-scratch implementation reproduce a trusted reference implementation?

---

# 3. Reference Architecture

Start with a **ViT-Base/16-style architecture**.

Use the following as the initial target configuration:

```text
Image:
224 × 224 × 3

Patch size:
16 × 16

Number of patches:
14 × 14 = 196

Embedding dimension:
768

CLS token:
1

Sequence length:
197

Transformer blocks:
12

Attention heads:
12

Head dimension:
64

MLP hidden dimension:
3072

Number of classes:
configurable
```

Do NOT assume that every reference implementation has exactly the same operation ordering.

The exact reference model must be inspected and documented.

---

# 4. High-Level Architecture

```text
Input Image
    │
    ▼
Patch Embedding
    │
    ▼
Patch Tokens
    │
    ├── CLS Token
    │
    ▼
Add Positional Embeddings
    │
    ▼
Transformer Block × N
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── MLP
    └── Residual Connection
    │
    ▼
Final Representation
    │
    ▼
CLS Token
    │
    ▼
Classification Head
    │
    ▼
Logits
```

The implementation must make every major component inspectable.

---

# 5. Repository Structure

Create the following structure:

```text
vit-reverse-engineering-lab/
│
├── README.md
├── PROJECT_SPEC.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── vit_base.yaml
│   ├── vit_small.yaml
│   └── experiments/
│
├── src/
│   └── vit_lab/
│       ├── __init__.py
│       │
│       ├── models/
│       │   ├── patch_embedding.py
│       │   ├── positional_embedding.py
│       │   ├── attention.py
│       │   ├── mlp.py
│       │   ├── transformer_block.py
│       │   ├── vision_transformer.py
│       │   └── classification_head.py
│       │
│       ├── reference/
│       │   ├── loader.py
│       │   └── hooks.py
│       │
│       ├── analysis/
│       │   ├── attention.py
│       │   ├── activations.py
│       │   └── representations.py
│       │
│       ├── benchmarking/
│       │   ├── latency.py
│       │   ├── memory.py
│       │   ├── parameters.py
│       │   └── throughput.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           └── config.py
│
├── tests/
│   ├── test_patch_embedding.py
│   ├── test_attention.py
│   ├── test_transformer_block.py
│   ├── test_model_shapes.py
│   ├── test_gradients.py
│   └── test_reference_equivalence.py
│
├── notebooks/
│   ├── 01_patch_embedding.ipynb
│   ├── 02_attention.ipynb
│   ├── 03_model_walkthrough.ipynb
│   ├── 04_attention_visualization.ipynb
│   ├── 05_reference_comparison.ipynb
│   └── 06_ablation_analysis.ipynb
│
├── experiments/
│   ├── patch_size/
│   ├── attention_heads/
│   ├── positional_embedding/
│   ├── cls_token/
│   └── resolution/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── checkpoints/
│   └── logs/
│
└── docs/
    ├── architecture.md
    ├── mathematics.md
    ├── reverse_engineering.md
    ├── benchmarking.md
    └── experiments.md
```

---

# 6. Implementation Requirements

## 6.1 Patch Embedding

Implement patch extraction without relying on a high-level ViT implementation.

Two approaches can be implemented:

### Approach A

Use `nn.Conv2d` with:

```text
kernel_size = patch_size
stride = patch_size
```

Explain why this is mathematically equivalent to applying the same linear projection independently to non-overlapping image patches.

### Approach B

Explicitly extract patches using tensor operations and apply a linear layer.

Use Approach B for educational inspection if practical, and Approach A for an efficient implementation.

The output should be:

```text
[B, C, H, W]
        ↓
[B, N, D]
```

where:

```text
N = (H / P) × (W / P)
```

---

# 7. CLS Token

Create a learnable CLS token:

```text
[1, 1, D]
```

Expand it across the batch:

```text
[B, 1, D]
```

Concatenate:

```text
[CLS] + patch_tokens
```

Result:

```text
[B, N + 1, D]
```

Document why the CLS token exists and how its representation is used for classification.

---

# 8. Positional Embeddings

Implement learnable positional embeddings.

For the standard 224 × 224 / 16 configuration:

```text
[B, 197, 768]
```

The positional embedding must match the token sequence length.

Document:

- Why positional information is needed
- Why self-attention itself does not inherently encode image position
- Learnable vs sinusoidal positional embeddings
- What happens when input resolution changes

---

# 9. Multi-Head Self-Attention

Implement attention manually.

Given:

```text
X ∈ R^(B × N × D)
```

generate:

```text
Q = XWq
K = XWk
V = XWv
```

Compute:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d) V
```

For multi-head attention:

```text
D = heads × head_dim
```

Reshape into:

```text
[B, heads, N, head_dim]
```

Attention matrix:

```text
[B, heads, N, N]
```

The implementation must expose the attention matrix when requested.

Do not hide the entire operation behind `nn.MultiheadAttention`.

The purpose is to understand the actual tensor operations.

---

# 10. MLP Block

Implement:

```text
Linear(D → hidden_dim)
↓
GELU
↓
Linear(hidden_dim → D)
```

The hidden dimension should initially be:

```text
4 × embedding_dimension
```

For ViT-Base:

```text
768 → 3072 → 768
```

Document why the MLP is applied independently to every token.

---

# 11. Transformer Block

Implement the complete Transformer block.

Start by inspecting the reference architecture.

Possible structure:

```text
x
│
├── LayerNorm
│
├── Multi-Head Attention
│
└── Residual
│
├── LayerNorm
│
├── MLP
│
└── Residual
```

Do not blindly assume this ordering.

Record the exact architecture of the reference model.

The project must explicitly explain:

- Pre-Norm vs Post-Norm
- Residual connections
- LayerNorm placement
- Dropout
- Stochastic depth / DropPath if present

---

# 12. Complete Vision Transformer

The model should expose:

```python
model.forward(x)
```

and optionally:

```python
model.forward_features(x)
```

The feature method should return the representation before classification.

Also support an analysis mode:

```python
outputs = model(
    x,
    return_attention=True,
    return_hidden_states=True
)
```

The returned object should make it possible to inspect:

```text
logits
hidden_states
attention_maps
patch_embeddings
final_features
```

without modifying the model source manually.

---

# 13. Reference Model

Use a trusted public implementation as the reference.

`timm` is recommended.

Do not copy the implementation.

The reference is used only for:

- Architecture inspection
- Weight loading
- Intermediate activation comparison
- Parameter comparison
- Output comparison
- Performance comparison

Document:

```text
Reference model
Reference version
Checkpoint
Dataset/pretraining information
Input preprocessing
Library version
```

Pin relevant package versions for reproducibility.

---

# 14. Reverse Engineering Procedure

The most important part of the project.

The reverse-engineering process should follow:

```text
Reference Model
      ↓
Inspect modules
      ↓
Record tensor shapes
      ↓
Identify operations
      ↓
Implement equivalent operation
      ↓
Load reference weights
      ↓
Compare outputs
      ↓
Locate discrepancy
      ↓
Fix implementation
      ↓
Repeat
```

Do this progressively.

---

# 15. Layer-by-Layer Equivalence

Do not only compare final predictions.

Compare:

```text
Input
↓
Patch Embedding
↓
Token + Position Embedding
↓
Block 1
↓
Block 2
↓
...
↓
Block 12
↓
Final Norm
↓
CLS representation
↓
Classification Head
↓
Logits
```

For every stage calculate:

```text
max absolute error
mean absolute error
mean squared error
relative error
cosine similarity
```

Example:

```text
Layer                  Max Error     Cosine Similarity
-------------------------------------------------------
Patch Embedding        ...
Position Embedding     ...
Block 1                ...
Block 2                ...
...
Final Features         ...
Logits                 ...
```

The goal is to get as close as practical to the reference implementation.

---

# 16. Weight Mapping

Create a reproducible weight-mapping procedure.

Document:

```text
Reference parameter name
        ↓
Our parameter name
        ↓
Tensor shape
        ↓
Transformation required
```

If weights need reshaping/transposition, explain why.

Do not silently modify tensors.

Every transformation must be documented.

---

# 17. Unit Tests

Tests are mandatory.

At minimum:

### Patch embedding

Verify:

```text
Input shape
Output shape
Number of patches
Embedding dimension
```

### Attention

Verify:

```text
Q/K/V shapes
Attention matrix shape
Output shape
```

### Transformer block

Verify:

```text
Input shape == Output shape
```

### Gradient test

Confirm that gradients propagate through:

```text
patch embedding
attention
MLP
LayerNorm
classification head
```

### Reference equivalence

Given identical weights and input:

```text
our_model(x)
≈
reference_model(x)
```

Set an appropriate numerical tolerance based on dtype and implementation details.

---

# 18. Attention Visualization

Build tools to visualize attention.

For an input image:

```text
Image
↓
ViT
↓
Attention
↓
Select layer
↓
Select head
↓
CLS → Patch attention
↓
Resize to image resolution
↓
Overlay on original image
```

Allow selection of:

```text
Layer
Head
Sample
```

Produce visualizations showing how attention changes across layers.

Also visualize:

- Early-layer attention
- Middle-layer attention
- Late-layer attention
- Different heads

Important:

Do not claim that an attention heatmap is a definitive explanation of the model's decision.

Call it an **attention visualization / diagnostic**.

---

# 19. Representation Analysis

Extract representations from different layers.

For example:

```text
Block 1
Block 3
Block 6
Block 9
Block 12
```

Analyze:

- Feature similarity
- CLS representation changes
- Token similarity
- Layer-wise representation drift

Optional:

Use PCA or t-SNE/UMAP to visualize representations.

---

# 20. Benchmarking

Measure:

### Model size

```text
Total parameters
Trainable parameters
Parameter memory
```

### Computational cost

Measure or estimate:

```text
FLOPs
MACs
```

Clearly distinguish measured values from estimates.

### Runtime

Measure:

```text
Single-image latency
Batch latency
Throughput
```

### GPU memory

Measure:

```text
Peak allocated memory
Peak reserved memory
```

Run multiple warm-up iterations before timing GPU inference.

Use synchronization where appropriate.

For example:

```text
warmup = 20
timed_iterations = 100
```

Report:

```text
mean
median
standard deviation
```

where practical.

---

# 21. Resolution Experiment

Test multiple resolutions.

Example:

```text
224 × 224
384 × 384
512 × 512
```

Keep the model architecture controlled where possible.

Analyze how the number of tokens changes.

For patch size P:

```text
N = (H/P) × (W/P)
```

Self-attention has approximately quadratic token interaction cost:

```text
O(N²D)
```

Demonstrate this experimentally.

This is especially relevant to medical imaging.

---

# 22. Patch Size Ablation

Compare:

```text
Patch 8
Patch 16
Patch 32
```

Record:

```text
Number of tokens
Parameters
FLOPs
GPU memory
Latency
Accuracy
```

Main question:

> What do we gain and lose when patches become smaller?

Expected trade-off:

Smaller patches:

```text
more spatial detail
more tokens
higher attention cost
```

Larger patches:

```text
less computation
less spatial detail
```

Do not assume the accuracy trend beforehand. Measure it.

---

# 23. Attention Head Ablation

Experiment with different numbers of heads where the architecture remains mathematically valid.

For example:

```text
4
8
12
16
```

Study:

- Performance
- Runtime
- Attention diversity
- Representation similarity

Investigate whether different heads produce visibly different attention patterns.

---

# 24. Positional Embedding Ablation

Compare:

```text
Learnable positional embedding
No positional embedding
```

Optional extension:

```text
Sinusoidal positional embedding
```

Measure:

```text
Accuracy
Training stability
Representation behavior
```

Explain the result rather than merely reporting numbers.

---

# 25. CLS Token Ablation

Compare:

```text
CLS token
```

against an alternative such as:

```text
Global average pooling
```

Study whether classification performance changes.

This is a useful experiment because it directly examines an architectural design decision.

---

# 26. Training Strategy

Do not immediately attempt to train ViT-Base from scratch on a huge dataset.

Use stages.

### Stage 1

Train a small ViT on a small dataset to verify the implementation.

### Stage 2

Compare your implementation with the reference implementation.

### Stage 3

Load pretrained weights.

### Stage 4

Fine-tune on the chosen downstream dataset.

This reduces wasted compute and makes debugging much easier.

---

# 27. Dataset Strategy

Use a simple natural-image dataset first.

Recommended:

```text
CIFAR-10
```

or another small classification dataset.

The purpose is:

```text
debugging
training
ablation
```

not achieving state-of-the-art results.

For the research extension, use a manageable medical-imaging classification dataset.

A retinal fundus dataset is particularly appropriate because it aligns with the broader computer-vision/medical-imaging direction of the portfolio.

Keep the medical experiment clearly separated from the core ViT reverse-engineering work.

---

# 28. Reproducibility

Every experiment must record:

```text
Random seed
Dataset
Dataset split
Model configuration
Image resolution
Patch size
Batch size
Learning rate
Optimizer
Scheduler
Epochs
Weight decay
Precision
GPU
PyTorch version
CUDA version
Git commit
```

Create configuration files rather than hardcoding experiment parameters.

Example:

```yaml
model:
  image_size: 224
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.05
  epochs: 20
```

---

# 29. Engineering Quality

The project should follow professional engineering practices.

Requirements:

- Type hints where useful
- Docstrings for public APIs
- Clear module boundaries
- No giant monolithic Python file
- No hardcoded absolute paths
- Configuration-driven experiments
- Reproducible seeds
- Automated tests
- Logging
- Checkpoint saving
- Error handling
- Clean README
- Reproducible setup instructions

Avoid unnecessary abstractions.

The code should be understandable to another ML engineer.

---

# 30. Documentation

Create:

## `docs/architecture.md`

Explain every ViT component.

## `docs/mathematics.md`

Explain:

```text
Patch projection
Q/K/V
Scaled dot-product attention
Multi-head attention
MLP
LayerNorm
Residual connections
```

Use equations where appropriate.

## `docs/reverse_engineering.md`

Explain exactly how the reference implementation was inspected and reproduced.

## `docs/benchmarking.md`

Explain:

```text
latency methodology
GPU memory methodology
FLOPs methodology
throughput methodology
```

## `docs/experiments.md`

Document every experiment and its findings.

---

# 31. Final README

The README should NOT look like a generic student project.

Opening section:

```text
# Inside the Vision Transformer

A from-scratch implementation and reverse-engineering study of
Vision Transformers, focused on numerical equivalence, internal
representations, attention behavior, and computational efficiency.
```

Then include:

```text
Architecture
↓
Implementation
↓
Reverse Engineering
↓
Numerical Verification
↓
Attention Visualization
↓
Benchmarking
↓
Ablation Studies
↓
Medical Imaging Extension
```

Include architecture diagrams and selected results.

---

# 32. Final Results Table

The README should eventually contain a table similar to:

| Experiment | Accuracy | Params | FLOPs | Latency | GPU Memory |
|---|---:|---:|---:|---:|---:|
| Patch 8 | ... | ... | ... | ... | ... |
| Patch 16 | ... | ... | ... | ... | ... |
| Patch 32 | ... | ... | ... | ... | ... |
| 4 Heads | ... | ... | ... | ... | ... |
| 8 Heads | ... | ... | ... | ... | ... |
| 12 Heads | ... | ... | ... | ... | ... |

Do not fabricate results.

Populate this table only after experiments are actually run.

---

# 33. Reverse Engineering Report

The final report should answer:

### Architecture

> What exactly is happening inside a ViT?

### Tensor flow

> What is the shape of the tensor at every major stage?

### Attention

> How do patches interact with each other?

### Numerical reproduction

> How closely does our implementation reproduce the reference?

### Efficiency

> What dominates computation and memory?

### Resolution

> Why does increasing resolution become expensive?

### Patch size

> What is the trade-off between spatial detail and computational cost?

### Attention heads

> Do different heads learn different relationships?

### Medical imaging

> What happens when ViT is transferred to a medical-imaging task?

---

# 34. Optional Advanced Extensions

Only implement these after the core project is complete.

Possible extensions:

### A. Flash Attention comparison

Compare standard attention against an optimized attention implementation.

Measure:

```text
memory
latency
throughput
```

### B. Mixed precision

Compare:

```text
FP32
FP16
BF16
```

where hardware supports it.

### C. Attention approximation

Investigate whether attention can be approximated to reduce quadratic cost.

### D. Different positional encoding strategies

Compare:

```text
learnable
sinusoidal
relative
rotary
```

Only if the experiment remains coherent with the ViT architecture being studied.

### E. Transfer learning

Compare:

```text
CNN
ViT
```

on the same medical-imaging task.

This can become a strong research extension.

---

# 35. What NOT to Do

Do NOT:

- Copy an existing ViT implementation.
- Claim a model was reverse-engineered without inspecting it.
- Compare only final accuracy.
- Hide important operations behind high-level APIs.
- Fabricate benchmark numbers.
- Claim attention maps are definitive explanations.
- Train massive models unnecessarily.
- Add dozens of meaningless ablations.
- Turn the repository into a collection of disconnected notebooks.
- Optimize for complexity instead of understanding.

The project should prioritize:

```text
Depth > Number of features
Reproducibility > flashy demos
Understanding > abstraction
Evidence > claims
```

---

# 36. Development Order

Codex should implement the project in this exact general order:

```text
1. Repository setup
       ↓
2. Configuration system
       ↓
3. Patch embedding
       ↓
4. Positional embeddings
       ↓
5. CLS token
       ↓
6. Multi-head attention
       ↓
7. MLP
       ↓
8. Transformer block
       ↓
9. Complete ViT
       ↓
10. Unit tests
       ↓
11. Small training experiment
       ↓
12. Reference model loader
       ↓
13. Weight mapping
       ↓
14. Layer-by-layer equivalence
       ↓
15. Attention extraction
       ↓
16. Visualization
       ↓
17. Benchmarking
       ↓
18. Ablations
       ↓
19. Medical-imaging extension
       ↓
20. Documentation
       ↓
21. Final README
```

Do not jump directly to the medical experiment.

The reverse-engineering milestone must be completed first.

---

# 37. Definition of Done

The project is considered complete when all of the following are true:

### Core implementation

- [ ] ViT implemented from scratch
- [ ] Patch embedding works
- [ ] CLS token works
- [ ] Positional embeddings work
- [ ] Multi-head attention implemented
- [ ] MLP implemented
- [ ] Transformer blocks implemented
- [ ] Classification head implemented

### Verification

- [ ] Unit tests pass
- [ ] Gradient flow verified
- [ ] Reference model loaded
- [ ] Reference weights mapped
- [ ] Intermediate tensors compared
- [ ] Final logits compared
- [ ] Numerical differences documented

### Analysis

- [ ] Attention visualization implemented
- [ ] Hidden-state extraction implemented
- [ ] Latency benchmark implemented
- [ ] GPU-memory benchmark implemented
- [ ] Parameter/FLOP analysis implemented

### Experiments

- [ ] Patch-size ablation
- [ ] Attention-head ablation
- [ ] Positional embedding ablation
- [ ] CLS-token/pooling comparison
- [ ] Resolution experiment
- [ ] At least one downstream transfer experiment

### Documentation

- [ ] Architecture documentation
- [ ] Mathematics documentation
- [ ] Reverse-engineering report
- [ ] Benchmark methodology
- [ ] Experiment report
- [ ] Professional README

---

# 38. Portfolio Positioning

The project should be presented on a CV approximately as:

**Vision Transformer Reverse Engineering Lab**  
*PyTorch · Computer Vision · Deep Learning · Model Analysis*

> Reimplemented a Vision Transformer from first principles and reverse-engineered a reference implementation through layer-wise tensor and weight analysis. Verified numerical equivalence, built attention/representation visualization tools, benchmarked latency and GPU memory, and investigated patch size, attention heads, positional embeddings, and input-resolution trade-offs.

Do not claim specific percentages or performance improvements until they have actually been measured.

---

# 39. Core Philosophy

This project is NOT:

> "I implemented ViT."

It is:

> **"I opened a Vision Transformer, traced what happens to every tensor, reproduced the computation, verified it against a real implementation, measured its behavior, and experimentally investigated why the architecture works."**

That distinction is the main reason this project belongs in an AI Engineer portfolio.