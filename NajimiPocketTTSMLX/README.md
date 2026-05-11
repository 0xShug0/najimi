# NajimiPocketTTSMLX

`NajimiPocketTTSMLX` is a compact Swift package for running PocketTTS-style speech synthesis on Apple platforms with MLX.

The core model code is adapted from [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio), with additional integration work for standalone SwiftPM use, inference-path optimizations, and batch synthesis support.

It provides a lightweight runtime for:

- loading a PocketTTS model from a local directory or Hugging Face repo
- synthesizing speech to a WAV file
- exporting reusable prompt embeddings from reference audio

## Installation

Inside this repository, add it as a local package:

```swift
.package(path: "NajimiPocketTTSMLX")
```

Then depend on the product:

```swift
.product(name: "NajimiPocketTTSMLX", package: "NajimiPocketTTSMLX")
```

If you publish the package independently later, swap the local path for the package URL.

## Model requirements

The runtime expects a PocketTTS model directory containing at least:

- `config.json`
- `.safetensors` weights
- tokenizer / prompt assets referenced by the config

You can pass either:

- a local filesystem path
- a Hugging Face repo identifier such as `"org/model-name"`

For gated Hugging Face repos, set `HF_TOKEN` in the environment before launch.

## Quick start

```swift
import Foundation
import NajimiPocketTTSMLX

let runtime = try await PocketTTSMiniRuntime.load(
    config: PocketTTSMiniConfig(
        modelPathOrRepo: "/path/to/pockettts-model",
        voice: "alba"
    )
)

let outputURL = URL(fileURLWithPath: "/tmp/hello.wav")
try await runtime.synthesize(
    text: "Hello from NajimiPocketTTSMLX.",
    outputURL: outputURL
)
```

## Prompt cloning flow

Create a prompt embedding once:

```swift
let embeddingURL = URL(fileURLWithPath: "/tmp/prompt.safetensors")
try await runtime.exportPromptEmbedding(
    fromAudioAt: URL(fileURLWithPath: "/tmp/reference.wav"),
    outputURL: embeddingURL
)
```

Reuse it for later synthesis:

```swift
let runtime = try await PocketTTSMiniRuntime.load(
    config: PocketTTSMiniConfig(
        modelPathOrRepo: "/path/to/pockettts-model",
        promptEmbeddingURL: embeddingURL
    )
)
```

## Notes

- The package bundles the MLX Metal library helper needed by standalone SwiftPM clients on macOS.
- Output audio is written as PCM WAV.
- Reference audio is trimmed, peak-normalized, and capped before prompt embedding export.
