# vllm-flashinfer-sm120-patch（中文说明）

这是一个面向 `flashinfer-python==0.6.6` 的 SM120（Blackwell）补丁分发包，目标是让 vLLM 用户在安装后以最小步骤应用补丁。

补丁应用后，继续使用原生 `vllm` 命令，不需要额外的 `serve` 包装命令。

## 快速使用

```bash
pip install vllm
pip install git+https://github.com/yhfgyyf/vllm-flashinfer-sm120-patch.git
vllm-flashinfer-sm120-apply
```

然后直接使用你原来的启动命令：

```bash
vllm serve <模型路径> [原有参数...]
```

## 补丁来源（PR 说明）

本仓库打包的是与下面 PR 对齐的一组 FlashInfer 文件修改：

- FlashInfer PR: https://github.com/flashinfer-ai/flashinfer/pull/2786

本包会把以下 4 个文件写入当前环境里已安装的 `flashinfer` 路径：

- `data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h`
- `data/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp`
- `data/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `jit/gemm/cutlass/generate_kernels.py`

说明：

- 本仓库是针对 vLLM 使用场景的“补丁分发与应用工具”，不是 FlashInfer 官方发布包。
- 本仓库不会在代码中写入本地绝对路径。

## 可用命令

- `vllm-flashinfer-sm120-status`：查看目标文件当前状态（`base` / `patched` / `unknown`）。
- `vllm-flashinfer-sm120-apply`：应用补丁，并为每个文件生成 `.bak.sm120patch` 备份。
- `vllm-flashinfer-sm120-apply --force`：当目标文件哈希未知时强制覆盖应用。

## 版本约束

- 目标版本：`flashinfer-python==0.6.6`
