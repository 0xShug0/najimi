import Foundation
import OSLog
@preconcurrency import MLX

/// Configuration for loading a PocketTTS model from disk or Hugging Face.
public struct PocketTTSMiniConfig: Sendable {
    public var modelPathOrRepo: String
    public var voice: String?
    public var promptEmbeddingURL: URL?
    public var language: String?

    public init(
        modelPathOrRepo: String,
        voice: String? = "alba",
        promptEmbeddingURL: URL? = nil,
        language: String? = nil
    ) {
        self.modelPathOrRepo = modelPathOrRepo
        self.voice = voice
        self.promptEmbeddingURL = promptEmbeddingURL
        self.language = language
    }
}

/// Small on-device TTS runtime that wraps model loading, synthesis, and prompt export.
public final class PocketTTSMiniRuntime: @unchecked Sendable {
    private static let logger = Logger(subsystem: "com.najimi.PocketTTSMLX", category: "runtime")
    private let model: PocketTTSModel
    private let config: PocketTTSMiniConfig

    private init(model: PocketTTSModel, config: PocketTTSMiniConfig) {
        self.model = model
        self.config = config
    }

    public var sampleRate: Int {
        model.sampleRate
    }

    /// Loads the configured model and prepares the bundled MLX Metal library when needed.
    public static func load(config: PocketTTSMiniConfig) async throws -> PocketTTSMiniRuntime {
        PocketTTSSupport.ensureEmbeddedMetalLibraryAvailable()
        let model = try await PocketTTSModel.fromPretrained(config.modelPathOrRepo)
        return PocketTTSMiniRuntime(model: model, config: config)
    }

    @discardableResult
    public func synthesize(
        text: String,
        outputURL: URL,
        temperature: Float = 0.65
    ) async throws -> URL {
        let audio = try await generateAudio(text: text, temperature: temperature)
        return try PocketTTSSupport.writeWav(audio, sampleRate: model.sampleRate, to: outputURL)
    }

    @discardableResult
    public func synthesize(
        text: String,
        outputURL: URL,
        batchCount: Int,
        temperature: Float = 0.65
    ) async throws -> URL {
        guard batchCount > 1 else {
            return try await synthesize(
                text: text,
                outputURL: outputURL,
                temperature: temperature
            )
        }

        Self.logger.info(
            "POCKETTTS batch_inference batchCount=\(batchCount, privacy: .public) chars=\(text.count, privacy: .public)"
        )

        let audio = try await generateAudio(
            text: text,
            temperature: temperature,
            batchCount: batchCount
        )
        return try PocketTTSSupport.writeWav(audio, sampleRate: model.sampleRate, to: outputURL)
    }

    @discardableResult
    public func exportPromptEmbedding(
        fromAudioAt audioURL: URL,
        outputURL: URL
    ) async throws -> URL {
        // Prompt embeddings can be reused to keep a cloned voice lightweight at inference time.
        let audio = try PocketTTSSupport.loadReferenceAudio(from: audioURL, sampleRate: model.sampleRate)
        let embedding = model.promptEmbedding(fromReferenceAudio: audio)
        return try PocketTTSSupport.writePromptEmbedding(embedding, to: outputURL)
    }

    public func synthesize(
        text: String,
        promptEmbeddingURL: URL,
        outputURL: URL,
        temperature: Float = 0.65
    ) async throws -> URL {
        var parameters = model.defaultGenerationParameters
        parameters.temperature = temperature
        let arrays = try MLX.loadArrays(url: promptEmbeddingURL)
        guard let embedding = arrays["audio_prompt"] else {
            throw NSError(
                domain: "PocketTTSMiniRuntime",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "The embedding file is missing the audio_prompt tensor."]
            )
        }
        let audio = try model.generate(
            text: text,
            promptEmbedding: embedding,
            language: config.language,
            generationParameters: parameters
        )
        return try PocketTTSSupport.writeWav(audio, sampleRate: model.sampleRate, to: outputURL)
    }

    public func synthesize(
        text: String,
        promptEmbedding: MLXArray,
        outputURL: URL,
        temperature: Float = 0.65
    ) async throws -> URL {
        var parameters = model.defaultGenerationParameters
        parameters.temperature = temperature
        let audio = try model.generate(
            text: text,
            promptEmbedding: promptEmbedding,
            language: config.language,
            generationParameters: parameters
        )
        return try PocketTTSSupport.writeWav(audio, sampleRate: model.sampleRate, to: outputURL)
    }

    private func generateAudio(
        text: String,
        temperature: Float,
        batchCount: Int = 1
    ) async throws -> MLXArray {
        var parameters = model.defaultGenerationParameters
        parameters.temperature = temperature

        if let promptEmbeddingURL = config.promptEmbeddingURL {
            let embedding = try loadPromptEmbedding(from: promptEmbeddingURL)
            if batchCount > 1 {
                return try model.generate(
                    text: text,
                    promptEmbedding: embedding,
                    language: config.language,
                    generationParameters: parameters,
                    batchCount: batchCount
                )
            } else {
                return try model.generate(
                    text: text,
                    promptEmbedding: embedding,
                    language: config.language,
                    generationParameters: parameters
                )
            }
        }

        if batchCount > 1 {
            return try await model.generate(
                text: text,
                voice: config.voice,
                refAudio: nil,
                refText: nil,
                language: config.language,
                generationParameters: parameters,
                batchCount: batchCount
            )
        } else {
            return try await model.generate(
                text: text,
                voice: config.voice,
                refAudio: nil,
                refText: nil,
                language: config.language,
                generationParameters: parameters
            )
        }
    }

    private func loadPromptEmbedding(from promptEmbeddingURL: URL) throws -> MLXArray {
        let arrays = try MLX.loadArrays(url: promptEmbeddingURL)
        guard let embedding = arrays["audio_prompt"] else {
            throw NSError(
                domain: "PocketTTSMiniRuntime",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "The embedding file is missing the audio_prompt tensor."]
            )
        }
        return embedding
    }
}
