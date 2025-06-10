#!/usr/bin/env python3
import flashinfer
print("Testing FlashInfer...")
print('FlashInfer version', flashinfer.__version__)
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

# prefill attention
qo_len = 2048
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0) # prefill attention
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False) # prefill attention without RoPE on-the-fly, do not apply causal mask

print('FlashInfer prefill attention OK\n')