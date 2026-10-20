#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_WORK:?set QWEN_WORK to the local work directory (runs, media, .kube)}"

PROFILE="${1:-}"
SOURCE_DIR="${2:-}"
DATASET="${3:-road-development-v1}"
KUBE_CONTEXT="${KUBE_CONTEXT:-cosmos3-vllm-poc}"

case "$PROFILE" in
  nano) deployment="cosmos3-nano-vllm" ;;
  super) deployment="cosmos3-super-vllm" ;;
  qwen-drive) deployment="qwen-drive-vllm" ;;
  *)
    echo "Usage: $0 nano|super|qwen-drive SOURCE_DIR [dataset-name]" >&2
    exit 2
    ;;
esac

if [[ -z "$SOURCE_DIR" ]]; then
  echo "SOURCE_DIR is required" >&2
  exit 2
fi
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
if [[ ! -f "$SOURCE_DIR/road-media-lock.json" ]]; then
  echo "Missing ROAD media lock: $SOURCE_DIR/road-media-lock.json" >&2
  exit 1
fi

kubectl --context "$KUBE_CONTEXT" -n cosmos wait \
  --for=condition=available "deployment/$deployment" --timeout=10800s
pod="$(
  kubectl --context "$KUBE_CONTEXT" -n cosmos get pods \
    -l "app=$deployment" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
)"
if [[ -z "$pod" ]]; then
  echo "No running pod for $deployment" >&2
  exit 1
fi

target="/benchmark-data/$DATASET"
kubectl --context "$KUBE_CONTEXT" -n cosmos exec "$pod" -- \
  mkdir -p "$target"
kubectl --context "$KUBE_CONTEXT" -n cosmos cp \
  "$SOURCE_DIR/." "$pod:$target"

local_sha="$(sha256sum "$SOURCE_DIR/road-media-lock.json" | awk '{print $1}')"
remote_sha="$(
  kubectl --context "$KUBE_CONTEXT" -n cosmos exec "$pod" -- \
    sha256sum "$target/road-media-lock.json" | awk '{print $1}'
)"
if [[ "$local_sha" != "$remote_sha" ]]; then
  echo "ROAD media lock checksum mismatch after sync" >&2
  exit 1
fi

local_records="$(
  jq '.records | length' "$SOURCE_DIR/road-media-lock.json"
)"
remote_records="$(
  kubectl --context "$KUBE_CONTEXT" -n cosmos exec "$pod" -- \
    python3 -c \
    "import json; print(len(json.load(open('$target/road-media-lock.json'))['records']))"
)"
if [[ "$local_records" != "$remote_records" ]]; then
  echo "ROAD media record count mismatch after sync" >&2
  exit 1
fi

echo "Synced $local_records ROAD media records to $pod:$target"
echo "Server media root: $target"
