# 公開自動運転動画に対する時系列シーンラベリング評価

本成果物の品質・性能・費用評価は、すべて公開ROAD dataset上で実施した。

## Dataset

- Annotated frames: 101,780
- Source videos: 18
- Development clips: 169
- Evaluation clips: 875
- Tags: 13

## Evaluation Full Run

| Model / profile | Clips | Precision | Recall | F1 | Accuracy | Balanced Accuracy | MCC | USD/1k clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosmos3-Nano / adaptive 24-reasoned | 875 | 71.3 | 66.7 | 68.9 | 85.6 | 79.1 | 0.596 | 0.66 |
| Cosmos3-Nano / uniform 24-reasoned | 875 | 73.4 | 65.9 | 69.4 | 86.1 | 79.2 | 0.606 | 0.69 |
| Cosmos3-Super / adaptive 24-reasoned | 875 | 81.9 | 68.3 | 74.5 | 88.8 | 81.8 | 0.678 | 13.48 |

## Development Ablation

| Profile | Precision | Recall | F1 | Balanced Accuracy | MCC |
|---|---:|---:|---:|---:|---:|
| road-adaptive-24-grouped | 50.6 | 50.0 | 50.3 | 66.9 | 0.339 |
| road-adaptive-24-reasoned | 73.3 | 67.3 | 70.2 | 79.6 | 0.609 |
| road-adaptive-24-ts | 67.9 | 34.6 | 45.8 | 64.6 | 0.378 |
| road-uniform-2fps-ts | 64.8 | 30.5 | 41.5 | 62.5 | 0.336 |
| road-uniform-4fps-ts | 85.9 | 32.4 | 47.0 | 65.3 | 0.453 |
| road-adaptive-24-reasoned-v2 | 73.6 | 66.2 | 69.7 | 79.1 | 0.605 |
| road-hybrid-24-reasoned | 72.1 | 68.6 | 70.3 | 79.9 | 0.608 |
| road-uniform-24-reasoned | 74.1 | 68.4 | 71.1 | 80.2 | 0.621 |

## ROAD Serving Controlled Sweep

| Model / condition | c | Wall/clip | P95 | Prefill/clip | Decode/clip | MM cache | Prefix cache | Preemptions | USD/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosmos3-Nano / server default (c1) | 1 | 3.62 | 2.58 | 0.96 | 0.38 | 0.0 | 0.6 | 0 | 2.26 |
| Cosmos3-Nano / server default (c2) | 2 | 1.95 | 3.76 | 1.14 | 0.74 | 0.0 | 0.6 | 0 | 1.22 |
| Cosmos3-Nano / server default (c4) | 4 | 1.38 | 5.78 | 1.40 | 1.44 | 0.0 | 0.6 | 0 | 0.86 |
| Cosmos3-Nano / server default (c8) | 8 | 1.23 | 14.50 | 1.59 | 3.48 | 0.0 | 0.6 | 0 | 0.77 |
| Cosmos3-Nano / Chunked Prefill 2,048 (c4) | 4 | 1.41 | 5.77 | 1.36 | 1.45 | 0.0 | 0.6 | 0 | 0.88 |
| Cosmos3-Nano / Chunked Prefill 8,192 (c4) | 4 | 1.60 | 6.02 | 1.29 | 1.78 | 0.0 | 0.6 | 0 | 1.00 |
| Cosmos3-Super / TP4, Chunked Prefill 8,192 (c4) | 4 | 7.38 | 30.47 | 11.52 | 9.52 | 0.0 | 0.6 | 0 | 13.68 |
| Cosmos3-Super / TP4, Chunked Prefill 8,192 (c8) | 8 | 6.97 | 67.36 | 17.51 | 16.49 | 0.0 | 0.6 | 4 | 12.93 |

ROADの同一15 clipsをcold start後に処理したNanoでは、server defaultのconcurrency 1から4でthroughputが2.62倍となり、EC2費用は$2.26から$0.86/1,000 clipsへ61.8\%低下した。一方、P95 latencyは2.58秒から5.78秒へ増加し、c8では14.50秒となった。concurrency 4におけるChunked Prefill 2,048 tokensは8,192 tokens比でthroughputが1.14倍であり、prefillは1.36秒/clip対1.29秒/clipであった。ただしserver default c4の費用は$0.86/1,000 clipsで、2,048-token条件の$0.88を下回り、Chunked Prefillの明確な優位は認められなかった。3-group requestでは同一mediaを連続再利用し、MM cache hit 100.0\%、Prefix cache hit 93.1\%を観測した。Superではc4のpreemptionが0件であったのに対し、c8は4件、P95 67.36秒となったため、Full Runにはc4を採用した。これらの結果は、throughput最大化とtail latency、KV-cache容量および再計算回避を同時に考慮する必要を示す。

## Sampling Coverage

| Profile | Frames | Any hit | 2 frames | Span >=0.5s | Context |
|---|---:|---:|---:|---:|---:|
| road-adaptive-24-reasoned-v2 | 23.3 | 100.0 | 97.4 | 96.1 | 100.0 |
| road-hybrid-24-reasoned | 23.2 | 100.0 | 98.9 | 97.1 | 100.0 |
| road-uniform-24-reasoned | 24.0 | 99.8 | 98.9 | 97.1 | 89.2 |

## False Negative Diagnosis

| Model | FN | Temporal miss | Sparse | Covered | Sampling/sparse |
|---|---:|---:|---:|---:|---:|
| Cosmos3-Nano | 904 | 0 | 38 | 866 | 4.2 |
| Cosmos3-Nano | 926 | 5 | 34 | 887 | 4.2 |
| Cosmos3-Super | 861 | 1 | 41 | 819 | 4.9 |

## Repeated Serving Sweep

| Condition | Runs | Wall/clip mean | P95 mean | USD/1k mean |
|---|---:|---:|---:|---:|
| Chunked Prefill 2,048 (c4) | 3 | 1.382 | 5.728 | 0.861 |
| Chunked Prefill 8,192 (c4) | 3 | 1.535 | 5.914 | 0.956 |
| server default (c1) | 3 | 3.605 | 2.601 | 2.245 |
| server default (c2) | 3 | 1.849 | 3.871 | 1.151 |
| server default (c4) | 3 | 1.384 | 5.750 | 0.862 |
| server default (c8) | 3 | 1.218 | 13.778 | 0.759 |

## Nano-Super Cascade

| Policy | Routed | Precision | Recall | F1 | MCC | USD/1k |
|---|---:|---:|---:|---:|---:|---:|
| nano_only | 0.0 | 73.4 | 65.9 | 69.4 | 0.606 | 0.69 |
| nano_positive_count_at_least_3 | 39.8 | 79.9 | 63.0 | 70.4 | 0.632 | 6.05 |
| nano_pedestrian_positive | 23.7 | 78.4 | 63.8 | 70.3 | 0.628 | 3.88 |
| nano_complex_temporal_positive | 69.1 | 83.1 | 65.8 | 73.4 | 0.670 | 10.01 |
| super_all | 100.0 | 81.9 | 68.3 | 74.5 | 0.678 | 13.48 |

## Public Motion Context and Fusion

公式の生INS/GNSSは使用せず、公開OPR派生軌跡とROAD動画から算出したVOだけを使用した。
- OPR coverage: 341 clips / 6 source videos
- Video-derived VO: 875 clips / 15 source videos

| Method | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|
| Super video baseline | 81.9 | 68.3 | 74.5 | 0.678 |
| Frozen VO motion prior | 82.8 | 70.5 | 76.2 | 0.698 |

Development-only OSM/OPR map-context pass was rejected: F1 76.5 -> 76.4, 3 improved / 7 regressed pairs, additional cost $12.90 per 1,000 target clips. OSM snapshot SHA-256: fb118a8420f6926f515b16e458eb8da7dbbf7d907ab0b879cfd859c4aa512b10.

## Development-only Prompt Selection

| Prompt | Criteria words | Detailed tags | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| road-adaptive-24-reasoned | 213 | 0 | 86.2 | 62.1 | 72.2 | 0.663 |
| road-adaptive-24-contrastive-v2 | 1273 | 13 | 85.6 | 65.8 | 74.4 | 0.683 |
| road-adaptive-24-hybrid-core-v1 (selected) | 603 | 5 | 85.6 | 67.6 | 75.6 | 0.695 |
| road-adaptive-24-hybrid-temporal-v1 | 778 | 7 | 85.8 | 67.8 | 75.8 | 0.697 |
| road-adaptive-24-hybrid-f1-v1 | 850 | 8 | 84.9 | 66.4 | 74.5 | 0.682 |

These values select a prompt on the Development split. They are not Evaluation performance estimates.

## Development-only Recall Diagnostic

| Condition | FN recovered | TP retained | New FP | Frames | E2E | USD/1k proxy |
|---|---:|---:|---:|---:|---:|---:|
| road-recall-baseline-24-core | 0/14 | 7/7 | 5/7 | 22.9 | 17.46 | 16.19 |
| road-recall-oracle-24-core | 2/14 | 7/7 | 4/7 | 24.0 | 18.27 | 16.94 |
| road-recall-oracle-48-core | 5/14 | 7/7 | 4/7 | 48.0 | 21.40 | 19.84 |
| road-recall-oracle-48-multiscale-core | 3/14 | 6/7 | 2/7 | 48.0 | 20.63 | 19.12 |
| road-recall-oracle-48-roi-core | 4/14 | 5/7 | 3/7 | 48.0 | 19.38 | 17.96 |
| road-recall-oracle-48-roi-motion-core | 3/14 | 5/7 | 1/7 | 48.0 | 20.27 | 18.79 |
| road-recall-oracle-48-roi-track-motion-core | 5/14 | 5/7 | 3/7 | 48.0 | 20.75 | 19.24 |

GT event times and GT boxes are diagnostic-only upper bounds. The set intentionally over-samples difficult errors.

## Exploratory Tag-specific Confidence

| Media | Threshold | FN recovered | TP retained | New FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 24 | 0.05 | 10/14 | 7/7 | 6/7 | 73.9 | 81.0 |
| Baseline 24 | 0.20 | 8/14 | 7/7 | 4/7 | 78.9 | 71.4 |
| Baseline 24 | 0.40 | 6/14 | 7/7 | 4/7 | 76.5 | 61.9 |
| Oracle 48 | 0.05 | 11/14 | 7/7 | 5/7 | 78.3 | 85.7 |
| Oracle 48 | 0.20 | 8/14 | 7/7 | 3/7 | 83.3 | 71.4 |
| Oracle 48 | 0.40 | 7/14 | 7/7 | 2/7 | 87.5 | 66.7 |

Thresholds are exploratory and require an independent calibration split before production use.
