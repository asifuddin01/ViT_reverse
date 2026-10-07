# Architecture: observed reference order and module map

This page maps the observed forward path of the pinned reference (`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`, see [reference_contract.md](reference_contract.md)) onto the educational implementation in `src/vit_lab/models/`. Shapes are for ViT-Base/16 at 224×224; the derivations are in [mathematics.md](mathematics.md).

```text
image [B,3,224,224]
  └─ PatchEmbedding: Conv2d(3,768,k=16,s=16) → flatten → [B,196,768]      patch_embedding.py
  └─ prepend CLS [1,1,768] → [B,197,768]                                   vision_transformer.py
  └─ + learned positions [1,197,768] (broadcast)                          positional_embedding.py
  └─ 12 × TransformerBlock (pre-norm)                                      transformer_block.py
       U = X + proj(softmax(QKᵀ/√64) V)  on LayerNorm1(X)                 attention.py
         packed qkv Linear(768,2304,bias) → Q,K,V [B,12,197,64]
       Y = U + fc2(GELU(fc1(LayerNorm2(U))))   fc1: 768→3072, fc2: 3072→768  mlp.py
  └─ final LayerNorm(eps=1e-6) → [B,197,768]                               vision_transformer.py
  └─ CLS pooling → [B,768] → Linear(768,1000) → logits                    classification_head.py
```

| Reference detail (observed in installed timm 1.0.30) | Educational implementation |
|---|---|
| Parameter names and shapes, 152 tensors / 86,567,656 parameters | identical; mapped as identity by `reference/weight_map.py` with `strict=True` |
| Pre-norm block, residual after attention and after MLP | `TransformerBlock.forward` |
| LayerNorm epsilon `1e-6` everywhere | `norm_eps=1e-6` |
| Packed QKV with bias, split Q,K,V on the output dimension | `qkv.reshape(B,N,3,H,d).permute(2,0,3,1,4)` |
| Fused SDPA attention by default | manual `softmax(QKᵀ·scale)V`; reference built with fused attention disabled for diagnostics |
| No pre-block norm, no `fc_norm`, dropout/drop-path zero | omitted / dropout probabilities default to 0 |
| `global_pool='token'` after final norm | `pool_tokens(tokens, "cls")` |

Analysis mode (`model(x, return_attention=True, return_hidden_states=True)`) returns pre-dropout attention probabilities and every block output without hooks. The default path keeps neither. Ablation-only options (`pooling="mean"`, `position_embedding="none"`, `dynamic_image_size=True` with bicubic position interpolation) do not change the reference configuration.

Equivalence evidence: [reverse_engineering.md](reverse_engineering.md) and `results/tables/equivalence.csv` (100/100 stage comparisons passed on CPU/FP32).
