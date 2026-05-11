# NajimiSileroVADMLX

`NajimiSileroVADMLX` is a small Swift package that runs the Silero voice activity detector with MLX on Apple platforms.

It supports:

- single-chunk inference
- streaming inference with recurrent state
- batched forward passes across longer audio buffers

## Installation

Inside this repository, add it as a local package:

```swift
.package(path: "NajimiSileroVADMLX")
```

Then depend on the product:

```swift
.product(name: "NajimiSileroVADMLX", package: "NajimiSileroVADMLX")
```

If you publish the package independently later, swap the local path for the package URL.

## Model requirements

Load a Silero VAD `.safetensors` checkpoint that matches the bundled MLX graph:

```swift
let model = try SileroVADModel.load(contentsOf: weightsURL)
```

## Quick start

```swift
import MLX
import NajimiSileroVADMLX

let model = try SileroVADModel.load(contentsOf: weightsURL)
let stream = SileroVADStream(model: model)

let probability = try stream.predict(samples: pcmChunk, sampleRate: 16_000)
if probability > 0.5 {
    print("speech detected")
}
```

## Processing a longer buffer

```swift
let model = try SileroVADModel.load(contentsOf: weightsURL)
let stream = SileroVADStream(model: model)
let probabilities = try stream.audioForward(samples: longPCMBuffer, sampleRate: 16_000)
```

## Notes

- The package bundles the MLX Metal library helper needed by standalone SwiftPM clients on macOS.
- Input audio should be 16 kHz PCM, or an integer multiple that can be downsampled to 16 kHz by simple striding.
- Streaming inference keeps a short context window so chunk boundaries remain stable.
