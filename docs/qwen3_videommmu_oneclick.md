# Qwen3-VL-8B-Instruct on VideoMMMU

This repo now has a one-click script for public VLMEvalKit:

```bash
bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh smoke
```

Full run:

```bash
bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh full
```

By default the script creates and uses its own environment, so it can run even if the old `video-o3` environment does not exist:

- Python runtime: `/mnt/output/vlmevalkit/envs/qwen3-videommmu-py312`
- Hugging Face cache: `/mnt/output/cache/hf`
- VideoMMMU data: `/mnt/output/video_bench_datasets/VideoMMMU`
- Output dir: `/mnt/output/vlmevalkit/qwen3_videommmu`

If you want to reuse the environment that was already validated for `video-o3`, set `REUSE_VIDEO_O3_ENV=1`:

```bash
REUSE_VIDEO_O3_ENV=1 \
  bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh smoke
```

That reuse mode points to:

- Python runtime: `/mnt/output/video_o3/eval/py312-runtime`
- Python packages: `/mnt/output/video_o3/eval/.venv312/lib/python3.12/site-packages`

It explicitly sets Qwen3-VL to use FlashAttention:

```bash
ATTN_IMPLEMENTATION=flash_attention_2
MODEL_DTYPE=bfloat16
```

The local `flash_attn` package in this repo is a compatibility shim backed by vLLM's prebuilt FlashAttention kernels. This avoids compiling `flash-attn` again while still making Transformers load Qwen3-VL with `attn_implementation="flash_attention_2"`.

The script installs the Qwen3/VideoMMMU dependency group if needed. It includes `vllm` so the local `flash_attn` shim can use vLLM's prebuilt FlashAttention kernels without compiling `flash-attn`.

Useful overrides:

```bash
# Use two GPUs.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh full

# Run the 64-frame VideoMMMU preset.
DATASET_PRESET=VideoMMMU_64frame \
  bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh full

# Put outputs somewhere else.
WORK_DIR=/mnt/output/my_qwen3_videommmu \
  bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh smoke

# Disable auto-install/bootstrap and require the environment to already exist.
BOOTSTRAP_ENV=0 \
  bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh smoke
```

`smoke` creates `/mnt/output/video_bench_datasets/VideoMMMU_smoke50` with the first row from `VideoMMMU.tsv` and symlinks the existing videos/images. `full` uses the complete dataset root directly.
