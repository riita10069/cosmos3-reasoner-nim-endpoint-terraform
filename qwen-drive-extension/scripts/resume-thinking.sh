#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_WORK:?set QWEN_WORK to the local work directory (runs, media, .kube)}"
cd "$(dirname "$0")/../benchmark"
export AWS_PROFILE=default KUBECONFIG=${QWEN_WORK}/.kube/config
MEDIA=${QWEN_WORK}/data/media-2fps-eval
.venv/bin/python -m benchmark_tool.cli vlm-run \
  --experiment configs/road-qwen-drive-thinking.json \
  --dataset-lock "$MEDIA/road-dataset-lock.json" --media-lock "$MEDIA/road-media-lock.json" \
  --labeling-config configs/road-labeling.json \
  --runs-dir ${QWEN_WORK}/runs \
  --run-id road-qwen-drive-uniform-24-2fps-thinking-evaluation-v1 \
  --endpoint http://127.0.0.1:18003 --profiles road-uniform-24-reasoned-2fps \
  --concurrency 8 --request-order shuffled-cold \
  --kube-context "${KUBE_CONTEXT:?set KUBE_CONTEXT}" \
  --namespace cosmos --deployment qwen-drive-vllm \
  --server-media-root /benchmark-data/road-24-2fps-evaluation-qwen-v1
