# Sourced by every script in local_model_eval/. Paths can be overridden by exporting them beforehand.
export EVAL_ROOT=${EVAL_ROOT:-/purestorage/ailab/yglee/workspace/local_model_eval}
export BENCH_DATA=${BENCH_DATA:-/purestorage/ailab/datasets/share_data/benchmarks}
export LCB_REPO=${LCB_REPO:-/purestorage/ailab/yglee/workspace/benchmarks/livecodebench}
export LCB_DATA_DIR=$BENCH_DATA/livecodebench
export PATH=/purestorage/ailab/yglee/tools/uv:$PATH

# On compute nodes file locks on /purestorage fail (errno 524) and HF/vLLM caches hang forever.
# $HOME is node-local disk there, so keep every lock-using cache under it.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HF_HOME=$HOME/.cache/huggingface
  export VLLM_CACHE_ROOT=$HOME/.cache/vllm
  export TRITON_CACHE_DIR=$HOME/.cache/triton
  export UV_CACHE_DIR=$HOME/.cache/uv
fi
export HF_DATASETS_OFFLINE=1

# Compute nodes have no CUDA toolkit (nvcc); vLLM's FlashInfer sampler and FlashInfer all-reduce (used with
# tensor parallel > 1) JIT-compile kernels and fail without it.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
# Same reason: FlashInfer's fused all-reduce (used with tensor parallel > 1) JIT-compiles with nvcc during CUDA graph
# capture and kills engine startup ("Could not find nvcc"). Fall back to vLLM's custom / NCCL all-reduce.
export VLLM_ALLREDUCE_USE_FLASHINFER=0
