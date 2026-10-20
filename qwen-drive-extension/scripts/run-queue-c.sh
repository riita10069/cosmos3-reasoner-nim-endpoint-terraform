#!/usr/bin/env bash
# Sequential Qwen-Drive ROAD evaluation runs on the EKS endpoint (port-forward :18003).
set -euo pipefail
: "${QWEN_WORK:?set QWEN_WORK to the local work directory (runs, media, .kube)}"
cd "$(dirname "$0")/../benchmark"
export AWS_PROFILE=default KUBECONFIG=${QWEN_WORK}/.kube/config
RUNS=${QWEN_WORK}/runs
MEDIA=${QWEN_WORK}/data/media-2fps-eval
CTX="${KUBE_CONTEXT:?set KUBE_CONTEXT to the cosmos3-vllm-poc kubectl context}"


run() {
  local experiment=$1 labeling=$2 lock=$3 run_id=$4 profile=$5 expected=$6
  .venv/bin/python -m benchmark_tool.cli vlm-run \
    --experiment "$experiment" \
    --dataset-lock "$MEDIA/road-dataset-lock.json" \
    --media-lock "$lock" \
    --labeling-config "$labeling" \
    --runs-dir "$RUNS" --run-id "$run_id" \
    --endpoint http://127.0.0.1:18003 \
    --profiles "$profile" --concurrency 8 --request-order shuffled-cold \
    --kube-context "$CTX" --namespace cosmos --deployment qwen-drive-vllm \
    --server-media-root /benchmark-data/road-24-2fps-evaluation-qwen-v1
  test "$(jq -s 'map(select(.success == true)) | length' "$RUNS/$run_id/results.jsonl")" -eq "$expected"
}

run configs/road-qwen-drive.json configs/road-labeling-single-weak.json "$MEDIA/road-media-lock-single-weak.json" \
  road-qwen-drive-uniform-24-2fps-single-weak-evaluation-v1 road-uniform-24-reasoned-2fps 2625
run configs/road-qwen-drive.json configs/road-labeling.json "$MEDIA/road-media-lock.json" \
  road-qwen-drive-adaptive-24-2fps-evaluation-v2 road-adaptive-24-reasoned-2fps 875
run configs/road-qwen-drive-thinking.json configs/road-labeling.json "$MEDIA/road-media-lock.json" \
  road-qwen-drive-uniform-24-2fps-thinking-evaluation-v1 road-uniform-24-reasoned-2fps 875
echo "queue C complete"
