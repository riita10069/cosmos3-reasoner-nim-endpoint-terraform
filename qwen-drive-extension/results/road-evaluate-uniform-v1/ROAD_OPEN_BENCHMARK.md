# ROAD complete-GT benchmark

Split: `evaluation`

Unknown GT pairs are excluded. Model abstentions are reported separately, reduce strict recall and overall accuracy, and are guarded by a 99% output-coverage threshold.

| Model | Run | Profile | Serving | Clips | Coverage | Precision | Recall | F1 | Accuracy | Balanced Acc. | MCC | Evidence hit | USD/1k clips |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvidia/Cosmos3-Nano | road-nano-uniform-24-evaluation-full-v1 | road-uniform-24-reasoned | nano-server-default | 875 | 100.0% | 73.4% | 65.9% | 69.4% | 86.1% | 79.2% | 0.606 | 99.8% | 0.686 |
| Qwen/Qwen-Drive-1.0-4B | road-qwen-drive-uniform-24-2fps-evaluation-v1 | road-uniform-24-reasoned-2fps | -- | 875 | 100.0% | 90.0% | 63.9% | 74.7% | 89.6% | 80.8% | 0.700 | 99.8% | 1.488 |
| Qwen/Qwen-Drive-1.0-4B | road-qwen-drive-uniform-24-4fps-evaluation-v1 | road-uniform-24-reasoned | -- | 875 | 100.0% | 91.2% | 62.1% | 73.8% | 89.5% | 80.1% | 0.695 | 99.8% | 0.795 |
| nvidia/Cosmos3-Super | road-super-evaluation-full-v1 | road-adaptive-24-reasoned | super-weights-chunked-8192 | 875 | 100.0% | 81.9% | 68.3% | 74.5% | 88.8% | 81.8% | 0.678 | 100.0% | 13.485 |

## Latency and cache

| Model | Profile | P50 | P95 | P99 | Prefill/clip | Decode/clip | MM cache | Prefix cache | GPU util. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nvidia/Cosmos3-Nano | road-uniform-24-reasoned | 8.639 | 11.443 | 13.016 | 1.808 | 3.934 | 0.0% | 0.6% | 99.434 |
| Qwen/Qwen-Drive-1.0-4B | road-uniform-24-reasoned-2fps | 23.414 | 60.340 | 64.201 | 18.058 | 13.599 | 30.2% | 0.0% | 99.764 |
| Qwen/Qwen-Drive-1.0-4B | road-uniform-24-reasoned | 14.075 | 44.421 | 46.899 | 8.871 | 7.737 | 16.1% | 0.0% | 99.845 |
| nvidia/Cosmos3-Super | road-adaptive-24-reasoned | 29.114 | 30.537 | 30.814 | 12.344 | 10.544 | 0.0% | 0.6% | 94.583 |
