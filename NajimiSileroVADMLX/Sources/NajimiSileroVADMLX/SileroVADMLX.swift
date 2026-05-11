import Foundation
import MLX
import MLXNN

/// Errors thrown while validating input or running Silero VAD inference.
public enum SileroVADError: LocalizedError {
    case unsupportedInputRank(Int)
    case unsupportedSampleRate(Int)
    case inputTooShort(sampleRate: Int, sampleCount: Int)
    case invalidChunkLength(expected: Int, actual: Int)
    case invalidReflectPadding(length: Int, padding: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedInputRank(let rank):
            return "SileroVADMLX expects audio with rank 1 or 2, received rank \(rank)."
        case .unsupportedSampleRate(let sampleRate):
            return "SileroVADMLX supports 16 kHz input or integer multiples of 16 kHz, received \(sampleRate) Hz."
        case .inputTooShort(let sampleRate, let sampleCount):
            return "SileroVADMLX chunk is too short for \(sampleRate) Hz input (\(sampleCount) samples)."
        case .invalidChunkLength(let expected, let actual):
            return "SileroVADMLX expects \(expected) samples per chunk after resampling, received \(actual)."
        case .invalidReflectPadding(let length, let padding):
            return "SileroVADMLX cannot reflect-pad \(padding) samples from a signal of length \(length)."
        }
    }
}

/// Stateful tensors needed to continue inference across audio chunks.
public struct SileroVADState: @unchecked Sendable {
    public var recurrentState: MLXArray
    public var context: MLXArray
    public var lastSampleRate: Int
    public var lastBatchSize: Int

    public init(
        recurrentState: MLXArray,
        context: MLXArray,
        lastSampleRate: Int = 0,
        lastBatchSize: Int = 0
    ) {
        self.recurrentState = recurrentState
        self.context = context
        self.lastSampleRate = lastSampleRate
        self.lastBatchSize = lastBatchSize
    }
}

/// Streaming wrapper that keeps recurrent state in sync between chunk predictions.
public final class SileroVADStream: @unchecked Sendable {
    public let model: SileroVADModel
    public private(set) var state: SileroVADState

    public init(model: SileroVADModel, batchSize: Int = 1) {
        self.model = model
        self.state = model.initialState(batchSize: batchSize)
    }

    public func reset(batchSize: Int = 1) {
        state = model.initialState(batchSize: batchSize)
    }

    @discardableResult
    public func predict(_ input: MLXArray, sampleRate: Int = 16_000) throws -> MLXArray {
        let (probabilities, nextState) = try model.predict(input, sampleRate: sampleRate, state: state)
        state = nextState
        return probabilities
    }

    @discardableResult
    public func predict(samples: [Float], sampleRate: Int = 16_000) throws -> Float {
        try predict(MLXArray(samples), sampleRate: sampleRate).squeezed().item(Float.self)
    }

    @discardableResult
    public func audioForward(_ input: MLXArray, sampleRate: Int = 16_000) throws -> MLXArray {
        let (probabilities, nextState) = try model.audioForward(input, sampleRate: sampleRate, state: state)
        state = nextState
        return probabilities
    }

    public func audioForward(samples: [Float], sampleRate: Int = 16_000) throws -> [Float] {
        try audioForward(MLXArray(samples), sampleRate: sampleRate).asArray(Float.self)
    }
}

/// Makes the bundled MLX Metal library visible to standalone SwiftPM clients on macOS.
public enum SileroVADMLXSupport {
    private static let bootstrapLock = NSLock()
    private static nonisolated(unsafe) var didPrepareMetalLibrary = false

    public static func ensureEmbeddedMetalLibraryAvailable() {
#if os(macOS)
        bootstrapLock.lock()
        defer { bootstrapLock.unlock() }

        guard !didPrepareMetalLibrary else { return }
        didPrepareMetalLibrary = true

        guard let bundledLibraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") else {
            return
        }

        let candidateDestinations = [
            Bundle.main.resourceURL?.appendingPathComponent("default.metallib"),
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("Resources/default.metallib"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("default.metallib"),
        ].compactMap { $0 }

        for destination in candidateDestinations {
            if FileManager.default.fileExists(atPath: destination.path) {
                return
            }
        }

        for destination in candidateDestinations {
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
}

/// Minimal MLX port of the Silero voice activity detector.
public final class SileroVADModel: Module, @unchecked Sendable {
    public static let supportedSampleRate = 16_000
    public static let hiddenSize = 128
    public static let fftSize = 256
    public static let stride = 128
    public static let reflectPadding = 64
    public static let featureBins = (fftSize / 2) + 1
    public static let contextSize = 64
    public static let chunkSize = 512

    @ModuleInfo(key: "stft_conv") var stftConv: SileroConv1dNCL
    @ModuleInfo var conv1: SileroConv1dNCL
    @ModuleInfo var conv2: SileroConv1dNCL
    @ModuleInfo var conv3: SileroConv1dNCL
    @ModuleInfo var conv4: SileroConv1dNCL
    @ModuleInfo(key: "lstm_cell") var lstmCell: SileroLSTMCell
    @ModuleInfo(key: "final_conv") var finalConv: SileroConv1dNCL

    public override init() {
        self._stftConv = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: 1,
                outputChannels: 258,
                kernelSize: Self.fftSize,
                stride: Self.stride,
                padding: 0,
                bias: false
            ),
            key: "stft_conv"
        )
        self._conv1 = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: Self.featureBins,
                outputChannels: 128,
                kernelSize: 3,
                stride: 1,
                padding: 1
            )
        )
        self._conv2 = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: 128,
                outputChannels: 64,
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
        )
        self._conv3 = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
        )
        self._conv4 = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: 64,
                outputChannels: 128,
                kernelSize: 3,
                stride: 1,
                padding: 1
            )
        )
        self._lstmCell = ModuleInfo(
            wrappedValue: SileroLSTMCell(inputSize: Self.hiddenSize, hiddenSize: Self.hiddenSize),
            key: "lstm_cell"
        )
        self._finalConv = ModuleInfo(
            wrappedValue: SileroConv1dNCL(
                inputChannels: Self.hiddenSize,
                outputChannels: 1,
                kernelSize: 1,
                stride: 1,
                padding: 0
            ),
            key: "final_conv"
        )
    }

    public static func load(contentsOf weightsURL: URL) throws -> SileroVADModel {
        SileroVADMLXSupport.ensureEmbeddedMetalLibraryAvailable()
        let model = SileroVADModel()
        let weights = try MLX.loadArrays(url: weightsURL)
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    public func initialState(batchSize: Int = 1) -> SileroVADState {
        SileroVADState(
            recurrentState: MLXArray.zeros([2, batchSize, Self.hiddenSize], type: Float.self),
            context: MLXArray.zeros([batchSize, Self.contextSize], type: Float.self)
        )
    }

    public func predict(
        _ input: MLXArray,
        sampleRate: Int = 16_000,
        state: SileroVADState? = nil
    ) throws -> (MLXArray, SileroVADState) {
        let (validatedInput, resolvedSampleRate) = try validateInput(input, sampleRate: sampleRate)
        let sampleCount = validatedInput.shape[1]
        guard sampleCount == Self.chunkSize else {
            throw SileroVADError.invalidChunkLength(expected: Self.chunkSize, actual: sampleCount)
        }

        let batchSize = validatedInput.shape[0]
        var workingState = prepareState(
            state ?? initialState(batchSize: batchSize),
            batchSize: batchSize,
            sampleRate: resolvedSampleRate
        )

        // The model keeps a short context tail so chunk boundaries do not lose speech onset information.
        let withContext = concatenated([workingState.context, validatedInput], axis: 1)
        workingState.context = withContext[.ellipsis, (withContext.shape[1] - Self.contextSize)...]

        let stftInput = try reflectPadRight(withContext, count: Self.reflectPadding).expandedDimensions(axis: 1)
        let stftOutput = stftConv(stftInput)

        let real = stftOutput[.ellipsis, ..<Self.featureBins, 0...]
        let imag = stftOutput[.ellipsis, Self.featureBins..., 0...]
        var features = sqrt(real * real + imag * imag)

        features = relu(conv1(features))
        features = relu(conv2(features))
        features = relu(conv3(features))
        features = relu(conv4(features))

        let lstmInput = features.squeezed(axis: -1)
        let hidden = workingState.recurrentState[0]
        let cell = workingState.recurrentState[1]
        let (nextHidden, nextCell) = lstmCell(lstmInput, hidden: hidden, cell: cell)

        workingState.recurrentState = stacked([nextHidden, nextCell], axis: 0)
        workingState.lastSampleRate = resolvedSampleRate
        workingState.lastBatchSize = batchSize

        var probabilities = relu(nextHidden).expandedDimensions(axis: -1)
        probabilities = finalConv(probabilities)
        probabilities = sigmoid(probabilities)
        probabilities = mean(probabilities.squeezed(axis: 1), axis: 1, keepDims: true)

        return (probabilities, workingState)
    }

    public func audioForward(
        _ input: MLXArray,
        sampleRate: Int = 16_000,
        state: SileroVADState? = nil
    ) throws -> (MLXArray, SileroVADState) {
        let (validatedInput, resolvedSampleRate) = try validateInput(input, sampleRate: sampleRate)
        let batchSize = validatedInput.shape[0]
        let totalSamples = validatedInput.shape[1]

        var workingState = prepareState(
            state ?? initialState(batchSize: batchSize),
            batchSize: batchSize,
            sampleRate: resolvedSampleRate
        )

        var outputs: [MLXArray] = []
        var offset = 0
        while offset < totalSamples {
            let end = min(offset + Self.chunkSize, totalSamples)
            var chunk = validatedInput[.ellipsis, offset ..< end]
            let chunkSamples = end - offset
            if chunkSamples < Self.chunkSize {
                let zeros = MLXArray.zeros([batchSize, Self.chunkSize - chunkSamples], type: Float.self)
                chunk = concatenated([chunk, zeros], axis: 1)
            }

            let (probabilities, nextState) = try predict(
                chunk,
                sampleRate: resolvedSampleRate,
                state: workingState
            )
            outputs.append(probabilities)
            workingState = nextState
            offset += Self.chunkSize
        }

        if outputs.isEmpty {
            return (MLXArray.zeros([batchSize, 0], type: Float.self), workingState)
        }

        return (concatenated(outputs, axis: 1), workingState)
    }

    private func prepareState(_ state: SileroVADState, batchSize: Int, sampleRate: Int) -> SileroVADState {
        if state.lastBatchSize == 0 || state.lastBatchSize != batchSize {
            return initialState(batchSize: batchSize)
        }
        if state.lastSampleRate != 0 && state.lastSampleRate != sampleRate {
            return initialState(batchSize: batchSize)
        }
        return state
    }

    private func validateInput(_ input: MLXArray, sampleRate: Int) throws -> (MLXArray, Int) {
        var resolvedInput = input
        if resolvedInput.ndim == 1 {
            resolvedInput = resolvedInput.expandedDimensions(axis: 0)
        }
        guard resolvedInput.ndim == 2 else {
            throw SileroVADError.unsupportedInputRank(resolvedInput.ndim)
        }

        var resolvedSampleRate = sampleRate
        if sampleRate != Self.supportedSampleRate && sampleRate % Self.supportedSampleRate == 0 {
            let step = sampleRate / Self.supportedSampleRate
            resolvedInput = resolvedInput[.ellipsis, .stride(by: step)]
            resolvedSampleRate = Self.supportedSampleRate
        }

        guard resolvedSampleRate == Self.supportedSampleRate else {
            throw SileroVADError.unsupportedSampleRate(sampleRate)
        }

        if Float(resolvedSampleRate) / Float(resolvedInput.shape[1]) > 31.25 {
            throw SileroVADError.inputTooShort(
                sampleRate: resolvedSampleRate,
                sampleCount: resolvedInput.shape[1]
            )
        }

        return (resolvedInput.asType(.float32), resolvedSampleRate)
    }

    private func reflectPadRight(_ input: MLXArray, count: Int) throws -> MLXArray {
        guard count > 0 else { return input }
        let length = input.shape[1]
        guard length > count else {
            throw SileroVADError.invalidReflectPadding(length: length, padding: count)
        }

        let reflectedSource = input[.ellipsis, (length - count - 1) ..< (length - 1)]
        let reflected = reflectedSource[.ellipsis, .stride(by: -1)]
        return concatenated([input, reflected], axis: 1)
    }
}

final class SileroConv1dNCL: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let stride: Int
    let padding: Int
    let groups: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._weight.wrappedValue = MLXArray.zeros([outputChannels, inputChannels / groups, kernelSize], type: Float.self)
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels], type: Float.self) : nil
        self.stride = stride
        self.padding = padding
        self.groups = groups
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var output = conv1d(
            input.transposed(0, 2, 1),
            weight.transposed(0, 2, 1),
            stride: stride,
            padding: padding,
            groups: groups
        )
        if let bias {
            output = output + bias
        }
        return output.transposed(0, 2, 1)
    }
}

final class SileroLSTMCell: Module {
    @ParameterInfo(key: "weight_ih") var weightIH: MLXArray
    @ParameterInfo(key: "weight_hh") var weightHH: MLXArray
    @ParameterInfo(key: "bias_ih") var biasIH: MLXArray
    @ParameterInfo(key: "bias_hh") var biasHH: MLXArray

    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self._weightIH.wrappedValue = MLXArray.zeros([4 * hiddenSize, inputSize], type: Float.self)
        self._weightHH.wrappedValue = MLXArray.zeros([4 * hiddenSize, hiddenSize], type: Float.self)
        self._biasIH.wrappedValue = MLXArray.zeros([4 * hiddenSize], type: Float.self)
        self._biasHH.wrappedValue = MLXArray.zeros([4 * hiddenSize], type: Float.self)
    }

    func callAsFunction(_ input: MLXArray, hidden: MLXArray, cell: MLXArray) -> (MLXArray, MLXArray) {
        let inputProjection = addMM(biasIH, input, weightIH.T)
        let hiddenProjection = addMM(biasHH, hidden, weightHH.T)
        let gates = inputProjection + hiddenProjection
        let chunks = split(gates, parts: 4, axis: -1)

        let inputGate = sigmoid(chunks[0])
        let forgetGate = sigmoid(chunks[1])
        let candidate = tanh(chunks[2])
        let outputGate = sigmoid(chunks[3])

        let nextCell = forgetGate * cell + inputGate * candidate
        let nextHidden = outputGate * tanh(nextCell)
        return (nextHidden, nextCell)
    }
}
