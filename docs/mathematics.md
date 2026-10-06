# ViT mathematics and tensor contract

The concrete reference is the pinned timm ViT-Base/16 in [reference_contract.md](reference_contract.md). Symbols: batch `B`, image side `H`, channels `C`, patch side `P`, patch count `N=(H/P)^2`, embedding width `D`, attention heads `h`, head width `d_h=D/h`.

## Patch projection

Split an image `X ∈ R[B,C,H,H]` into `N` non-overlapping patches. Flatten each patch to `p_i ∈ R[C P²]` in channel, row, column order. Apply the **same** learned matrix and bias to every patch:

`z_i = W_patch p_i + b_patch`, with `W_patch ∈ R[D,C P²]`.

`Conv2d(C,D,kernel_size=P,stride=P)` computes the same operation because each kernel sees exactly one patch and shares its weights across grid positions. `unfold` plus a linear layer exposes the flattened patches. Both paths are implemented in `PatchEmbedding`; their values and input gradients are tested against each other.

For a 4×4, one-channel image holding rows `1..4`, `5..8`, `9..12`, `13..16`, `P=2` produces patches `[1,2,5,6]`, `[3,4,7,8]`, `[9,10,13,14]`, `[11,12,15,16]`. With kernel `[[1,2],[3,4]]` and zero bias, the patch outputs are `[44,64,124,144]` in row-major grid order. This hand calculation is a unit test.

## Tokens and position

The reference uses a learnable CLS vector `[1,1,D]`. Expand across the batch and prepend it to patch tokens: `[B,N+1,D]`. Add a learnable positional parameter `[1,N+1,D]` by broadcasting. Attention alone is permutation-equivariant: without positional information, swapping patch tokens swaps their outputs in the same way and leaves no record of where each patch came from. At a new image resolution, the patch grid changes; positional weights require a declared interpolation method rather than an implicit reshape.

## Multi-head self-attention

For token matrix `X ∈ R[B,T,D]`, where `T=N+1`, a packed linear layer produces Q, K, V. Reshape each to `[B,h,T,d_h]`. Each head computes

`A = softmax(Q Kᵀ / sqrt(d_h), axis=keys)` and `Y = A V`.

The probability tensor is `[B,h,T,T]`; each query row sums to approximately one before dropout. Concatenate head outputs to `[B,T,D]` and apply an output linear projection. For the reference, `T=197`, `h=12`, and `d_h=64`.

## MLP, normalization, and residuals

The MLP applies `Linear(D,4D) → GELU → Linear(4D,D)` to **each token independently**; token mixing occurs in attention. The inspected reference block is pre-norm:

`U = X + Attention(LayerNorm1(X))`  
`Y = U + MLP(LayerNorm2(U))`.

Its final LayerNorm precedes CLS-token pooling and the classification head. The installed reference uses LayerNorm epsilon `1e-6`, a detail that must match for numerical equivalence. See [reference_contract.md](reference_contract.md) for the observed dropout, bias, and pooling settings.

## Reference shape trace

| Stage | Shape for `B` images |
|---|---|
| RGB image | `[B,3,224,224]` |
| Patch tokens, `P=16` | `[B,196,768]` |
| CLS plus position | `[B,197,768]` |
| Q, K, V | each `[B,12,197,64]` |
| Attention probabilities | `[B,12,197,197]` |
| Output of each of 12 blocks | `[B,197,768]` |
| Final normalized tokens | `[B,197,768]` |
| CLS representation | `[B,768]` |
| 1,000-class logits | `[B,1000]` |
