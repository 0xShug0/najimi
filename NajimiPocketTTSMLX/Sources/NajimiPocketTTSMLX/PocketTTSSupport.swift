import Foundation
@preconcurrency import MLX

/// Shared file and audio helpers used by the lightweight PocketTTS runtime.
public enum PocketTTSSupport {
    private static let metalBootstrapLock = NSLock()
    private static nonisolated(unsafe) var didPrepareEmbeddedMetalLibrary = false
    private static let maxReferenceDurationSeconds: Double = 12
    private static let silenceThreshold: Float = 0.008
    private static let silencePaddingSamples = 1200

    @discardableResult
    public static func ensureParentDirectory(for url: URL) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        return url
    }

    public static func ensureEmbeddedMetalLibraryAvailable() {
#if os(macOS)
        metalBootstrapLock.lock()
        defer { metalBootstrapLock.unlock() }

        guard !didPrepareEmbeddedMetalLibrary else { return }
        didPrepareEmbeddedMetalLibrary = true

        guard let bundledLibraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") else {
            return
        }

        let candidateURLs = [
            Bundle.main.resourceURL?.appendingPathComponent("default.metallib"),
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("Resources/default.metallib"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("default.metallib"),
        ].compactMap { $0 }

        if candidateURLs.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return
        }

        for destination in candidateURLs {
            do {
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try FileManager.default.copyItem(at: bundledLibraryURL, to: destination)
                return
            } catch {
                continue
            }
        }
#endif
    }

    @discardableResult
    public static func writeWav(_ audio: MLXArray, sampleRate: Int, to url: URL) throws -> URL {
        let outputURL = try ensureParentDirectory(for: url)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }
        try AudioUtils.writeWavFile(samples: audio.asArray(Float.self), sampleRate: sampleRate, fileURL: outputURL)
        return outputURL
    }

    public static func loadAudio(from url: URL, sampleRate: Int? = nil) throws -> MLXArray {
        let (_, audio) = try loadAudioArray(from: url, sampleRate: sampleRate)
        return audio
    }

    public static func loadReferenceAudio(from url: URL, sampleRate: Int) throws -> MLXArray {
        let (_, audio) = try loadAudioArray(from: url, sampleRate: sampleRate)
        let samples = preprocessReferenceSamples(audio.asArray(Float.self), sampleRate: sampleRate)
        return MLXArray(samples)
    }

    @discardableResult
    public static func writePromptEmbedding(_ embedding: MLXArray, to url: URL) throws -> URL {
        let outputURL = try ensureParentDirectory(for: url)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }
        try PocketTTSModel.savePromptEmbedding(embedding, to: outputURL)
        return outputURL
    }

    private static func preprocessReferenceSamples(_ samples: [Float], sampleRate: Int) -> [Float] {
        guard !samples.isEmpty else {
            return samples
        }

        // Reference conditioning is more stable when silence and clipping are trimmed first.
        let trimmed = trimSilence(samples)
        let clipped = capDuration(trimmed, sampleRate: sampleRate, maxDurationSeconds: maxReferenceDurationSeconds)
        return normalizePeak(clipped)
    }

    private static func trimSilence(_ samples: [Float]) -> [Float] {
        guard let first = samples.firstIndex(where: { abs($0) >= silenceThreshold }),
              let last = samples.lastIndex(where: { abs($0) >= silenceThreshold }) else {
            return samples
        }

        let lowerBound = max(0, first - silencePaddingSamples)
        let upperBound = min(samples.count - 1, last + silencePaddingSamples)
        return Array(samples[lowerBound...upperBound])
    }

    private static func capDuration(_ samples: [Float], sampleRate: Int, maxDurationSeconds: Double) -> [Float] {
        let maxSamples = max(1, Int(Double(sampleRate) * maxDurationSeconds))
        guard samples.count > maxSamples else {
            return samples
        }

        let start = max(0, (samples.count - maxSamples) / 2)
        let end = start + maxSamples
        return Array(samples[start..<end])
    }

    private static func normalizePeak(_ samples: [Float]) -> [Float] {
        let peak = samples.reduce(Float.zero) { max($0, abs($1)) }
        guard peak > 0 else {
            return samples
        }

        let scale = min(Float(0.95) / peak, 8)
        return samples.map { $0 * scale }
    }
}
