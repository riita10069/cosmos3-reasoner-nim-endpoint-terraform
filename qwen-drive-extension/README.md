# Qwen-Drive-1.0-4B ROAD extension

Extends the Cosmos 3 ROAD scene-labeling study with `Qwen/Qwen-Drive-1.0-4B`
(revision `28484089`) on the same 875 Evaluation clips, prompts and media.

- `report/QWEN_DRIVE_ROAD_REPORT.md`: results and method (Japanese)
- `benchmark/`: ROAD benchmark tool recovered from the original Cosmos 3 work
  (requests reproduce the recorded Cosmos `request_sha256` for all 875 clips)
  plus the Qwen-Drive changes
  (`request_extra`, `max_tokens_override`, reasoning capture, stream-error handling)
  and configs `road-qwen-drive*.json`, `road-sampling-qwen-2fps.json`,
  `road-labeling-single-weak.json`
- `tools/`: complete-GT evaluator, weak-tag analysis, single-tag lock builder,
  VO-prior export
- `scripts/`: media sync and the run queues used on EKS
  (set `QWEN_WORK` for local runs and media, and `KUBE_CONTEXT`)
- `results/`: aggregate metrics only (no per-clip predictions)

The EKS manifest and checkpoint converter live in
`riita10069/cosmos3-nano-vllm-eks-poc` (`qwen-drive/`).

Not included: ROAD videos, encoded media, per-clip predictions (`*.jsonl`),
and the commands of the original tool that served non-ROAD datasets.
Inference runs through the `vlm-run` subcommand. Local paths in result files are replaced with
`${COSMOS_ARTIFACTS}`, `${QWEN_WORK}`, `${REPO_ROOT}` and `${ROAD_CACHE}`.
`make verify` passes with this directory included.
