# Reproduction

The paper is generated only from locked ROAD artifacts and complete-GT
evaluation summaries.

## Inputs

- Evaluation summary: `inputs/evaluation-summary.json`
- Development summary: `inputs/development-summary.json`
- Serving smoke summary: `inputs/serving-smoke-summary.json`
- ROAD GT manifest: `inputs/road-benchmark-manifest.json`
- ROAD annotation audit: `inputs/road-annotation-audit.json`
- error-analysis.json: `inputs/error-analysis.json`
- sampling-audit.json: `inputs/sampling-audit.json`
- serving-repeat-summary.json: `inputs/serving-repeat-summary.json`
- cascade-summary.json: `inputs/cascade-summary.json`
- motion-prior-summary.json: `inputs/motion-prior-summary.json`
- composed-motion-summary.json: `inputs/composed-motion-summary.json`
- context-fusion-summary.json: `inputs/context-fusion-summary.json`
- opr-motion-lock.json: `inputs/opr-motion-lock.json`
- visual-motion-lock.json: `inputs/visual-motion-lock.json`
- map-context-lock.json: `inputs/map-context-lock.json`
- prompt-development-summary.json: `inputs/prompt-development-summary.json`
- prompt-selection-lock.json: `inputs/prompt-selection-lock.json`
- prompt-audit.json: `inputs/prompt-audit.json`
- recall-diagnostic-summary.json: `inputs/recall-diagnostic-summary.json`
- recall-confidence-baseline.json: `inputs/recall-confidence-baseline.json`
- recall-confidence-oracle.json: `inputs/recall-confidence-oracle.json`

## Generate the paper

```bash
PATH=/opt/homebrew/bin:$PATH PYTHONPATH=benchmark \
  python3 -m benchmark_tool.road_paper_cli --paper-dir paper
```

PDF generation requires Tectonic and the fonts declared in the TeX source.
The artifact manifest records SHA-256 digests for every input and output.
