# vllm-flashinfer-sm120-patch

Patch package for `flashinfer-python==0.6.6` on SM120 (Blackwell workstation GPUs), intended for vLLM users.

This package applies four patched files (from FlashInfer PR #2786 related changes) into the installed `flashinfer` package directory.

## Goal

After installing `vllm`, make patching as simple as possible with a small extra step:

```bash
pip install git+https://github.com/yhfgyyf/vllm-flashinfer-sm120-patch.git
vllm-flashinfer-sm120-apply
```

Then launch server (optional helper that isolates caches):

```bash
vllm-flashinfer-sm120-serve <model_path> [vllm serve args...]
```

## What is patched

- `data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h`
- `data/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp`
- `data/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `jit/gemm/cutlass/generate_kernels.py`

## Commands

- `vllm-flashinfer-sm120-status`: show whether each target file is base/patched/unknown.
- `vllm-flashinfer-sm120-apply`: apply patch; create `.bak.sm120patch` backup per file.
- `vllm-flashinfer-sm120-apply --force`: apply even if current file hash is unknown.
- `vllm-flashinfer-sm120-serve`: run `vllm serve` with isolated cache env vars to avoid cross-env cache pollution.

## Notes

- This package targets `flashinfer-python==0.6.6`.
- It does not embed any local absolute path in code.
