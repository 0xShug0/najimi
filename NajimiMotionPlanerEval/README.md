# NajimiMotionPlanerEval

Standalone Swift Package for planner-format evaluation of local MLX model
directories.

This package does not depend on Najimi code. It uses standalone case files in
`cases/`, simulates the planner prompt shape, runs local model directories with
MLX Swift LM, validates output format, and reports:

- valid-format rate
- warm TTFT
- warm prompt token/s
- warm generation token/s
- prompt-cache hit/miss status

## Build

```bash
swift build -c release
```

## Run One Model

```bash
swift run -c release najimi-motion-planner-eval run \
  --model-path /Volumes/Port/Models/qwen3-0.6b-4bit \
  --cases-dir /Volumes/Port/NajimiMotionPlanerEval/cases \
  --output-dir /Volumes/Port/NajimiMotionPlanerEval/results/qwen3-0.6b-4bit
```

## Run A Directory Of Models

```bash
swift run -c release najimi-motion-planner-eval run-dir \
  --models-dir /Volumes/Port/Models \
  --cases-dir /Volumes/Port/NajimiMotionPlanerEval/cases \
  --output-dir /Volumes/Port/NajimiMotionPlanerEval/results/all-models
```

This writes:

- `summary.csv`
- `models/<model-name>/report.json`
- `models/<model-name>/summary.md`

## Console Progress

The CLI prints:

- which model is being loaded
- whether the prompt cache was a hit or miss
- when warmup starts
- how many cases are left for the current model

## Prompt Cache

The evaluator stores a reusable on-disk KV prefix cache per model for the
shared planner system prompt. The cache is written under:

```text
.cache/prefix-cache/<model-name>/
```

Reports and CSV rows include the prompt-cache status and cache file path used by
the run.
