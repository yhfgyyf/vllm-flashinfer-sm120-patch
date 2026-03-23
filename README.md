# vllm-flashinfer-sm120-patch

Patch package for `flashinfer-python==0.6.6` on SM120 (Blackwell workstation GPUs), intended for vLLM users.

After patching, keep using the original `vllm` command directly.

## Quick Start

```bash
pip install vllm
pip install git+https://github.com/yhfgyyf/vllm-flashinfer-sm120-patch.git
vllm-flashinfer-sm120-apply
```

Then run your original serving command as-is:

```bash
vllm serve <model_path> [original args...]
```

## PR Provenance

This package carries a focused subset of FlashInfer changes aligned with:

- FlashInfer PR: https://github.com/flashinfer-ai/flashinfer/pull/2786

The package applies these 4 files into the installed `flashinfer` location:

- `data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h`
- `data/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp`
- `data/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `jit/gemm/cutlass/generate_kernels.py`

Scope note:

- This repo is a patch distribution helper for vLLM environments.
- It is not an official FlashInfer release artifact.

## Commands

- `vllm-flashinfer-sm120-status`: show whether each target file is `base` / `patched` / `unknown`.
- `vllm-flashinfer-sm120-apply`: apply patch and create `.bak.sm120patch` backup per file.
- `vllm-flashinfer-sm120-apply --force`: apply even if current file hash is unknown.

## Notes

- Target version: `flashinfer-python==0.6.6`
- No local absolute path is embedded in code.
- Chinese README: `README_zh.md`
