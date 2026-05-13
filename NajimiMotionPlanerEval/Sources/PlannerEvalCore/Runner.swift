import CryptoKit
import Foundation
import MLXHuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

private struct PromptCacheArtifact {
    let cache: [any KVCache]
    let status: String
    let path: URL
}

public struct SingleRunConfiguration: Sendable {
    public let modelPath: URL
    public let casesDirectory: URL
    public let outputDirectory: URL
    public let maxTokens: Int
    public let caseLimit: Int?
    public let maxCasesPerSuite: Int?
    public let suiteFilter: Set<String>

    public init(
        modelPath: URL,
        casesDirectory: URL,
        outputDirectory: URL,
        maxTokens: Int,
        caseLimit: Int?,
        maxCasesPerSuite: Int?,
        suiteFilter: Set<String>
    ) {
        self.modelPath = modelPath
        self.casesDirectory = casesDirectory
        self.outputDirectory = outputDirectory
        self.maxTokens = maxTokens
        self.caseLimit = caseLimit
        self.maxCasesPerSuite = maxCasesPerSuite
        self.suiteFilter = suiteFilter
    }
}

public struct DirectoryRunConfiguration: Sendable {
    public let modelsDirectory: URL
    public let casesDirectory: URL
    public let outputDirectory: URL
    public let modelFilter: Set<String>
    public let maxTokens: Int
    public let caseLimit: Int?
    public let maxCasesPerSuite: Int?
    public let suiteFilter: Set<String>

    public init(
        modelsDirectory: URL,
        casesDirectory: URL,
        outputDirectory: URL,
        modelFilter: Set<String>,
        maxTokens: Int,
        caseLimit: Int?,
        maxCasesPerSuite: Int?,
        suiteFilter: Set<String>
    ) {
        self.modelsDirectory = modelsDirectory
        self.casesDirectory = casesDirectory
        self.outputDirectory = outputDirectory
        self.modelFilter = modelFilter
        self.maxTokens = maxTokens
        self.caseLimit = caseLimit
        self.maxCasesPerSuite = maxCasesPerSuite
        self.suiteFilter = suiteFilter
    }
}

public enum PlannerEvalRunner {
    public static func runSingle(_ configuration: SingleRunConfiguration) async throws -> ModelReport {
        let evaluationPlan = try CaseLoader.loadPlan(
            from: configuration.casesDirectory,
            suiteFilter: configuration.suiteFilter,
            caseLimit: configuration.caseLimit,
            maxCasesPerSuite: configuration.maxCasesPerSuite
        )
        return try await runPreparedSingle(
            modelPath: configuration.modelPath,
            outputDirectory: configuration.outputDirectory,
            evaluationPlan: evaluationPlan,
            maxTokens: configuration.maxTokens
        )
    }

    public static func runDirectory(_ configuration: DirectoryRunConfiguration) async throws -> URL {
        let evaluationPlan = try CaseLoader.loadPlan(
            from: configuration.casesDirectory,
            suiteFilter: configuration.suiteFilter,
            caseLimit: configuration.caseLimit,
            maxCasesPerSuite: configuration.maxCasesPerSuite
        )
        let modelDirectories = try discoverModelDirectories(
            in: configuration.modelsDirectory,
            modelFilter: configuration.modelFilter
        )
        let modelsOutputDirectory = configuration.outputDirectory.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsOutputDirectory, withIntermediateDirectories: true)

        var rows: [BatchCSVRow] = []
        for modelPath in modelDirectories {
            do {
                let report = try await runPreparedSingle(
                    modelPath: modelPath,
                    outputDirectory: modelsOutputDirectory.appendingPathComponent(modelPath.lastPathComponent, isDirectory: true),
                    evaluationPlan: evaluationPlan,
                    maxTokens: configuration.maxTokens
                )
                rows.append(
                    BatchCSVRow(
                        modelName: report.modelName,
                        modelPath: report.modelPath,
                        status: "ok",
                        error: "",
                        totalCases: report.summary.totalCases,
                        validCases: report.summary.validCases,
                        validRate: report.summary.validRate,
                        promptCacheStatus: report.summary.promptCacheStatus,
                        promptCachePath: report.summary.promptCachePath,
                        warmupCompleted: report.summary.warmupCompleted,
                        warmCaseCount: report.summary.warmCaseCount,
                        warmMeanTTFTMilliseconds: report.summary.warmMeanTTFTMilliseconds,
                        warmMeanPromptTokensPerSecond: report.summary.warmMeanPromptTokensPerSecond,
                        warmMeanGenerationTokensPerSecond: report.summary.warmMeanGenerationTokensPerSecond,
                        suiteSummary: report.summary.suiteSummary
                    )
                )
                print("model=\(report.modelName) status=ok valid_rate=\(String(format: "%.2f%%", report.summary.validRate * 100))")
            } catch {
                rows.append(
                    BatchCSVRow(
                        modelName: modelPath.lastPathComponent,
                        modelPath: modelPath.path,
                        status: "error",
                        error: error.localizedDescription,
                        totalCases: 0,
                        validCases: 0,
                        validRate: 0,
                        promptCacheStatus: "",
                        promptCachePath: "",
                        warmupCompleted: false,
                        warmCaseCount: 0,
                        warmMeanTTFTMilliseconds: 0,
                        warmMeanPromptTokensPerSecond: 0,
                        warmMeanGenerationTokensPerSecond: 0,
                        suiteSummary: [:]
                    )
                )
                print("model=\(modelPath.lastPathComponent) status=error error=\(error.localizedDescription)")
            }
        }

        let csvURL = configuration.outputDirectory.appendingPathComponent("summary.csv")
        try Reporter.writeBatchCSV(rows: rows, to: csvURL)
        return csvURL
    }

    private static func discoverModelDirectories(in directory: URL, modelFilter: Set<String>) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw PlannerEvalError.invalidModelsDirectory(directory)
        }

        let children = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        .filter { !$0.lastPathComponent.hasPrefix(".") }
        .filter { child in
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: child.path, isDirectory: &isDirectory) && isDirectory.boolValue
        }
        .filter { modelFilter.isEmpty || modelFilter.contains($0.lastPathComponent) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !children.isEmpty else {
            throw PlannerEvalError.noModelDirectories(directory)
        }
        return children
    }

    private static func runPreparedSingle(
        modelPath: URL,
        outputDirectory: URL,
        evaluationPlan: LoadedEvaluationPlan,
        maxTokens: Int
    ) async throws -> ModelReport {
        let modelContainer = try await loadModel(modelPath)
        let evaluation = try await evaluateModel(
            modelName: modelPath.lastPathComponent,
            modelContainer: modelContainer,
            actionSet: evaluationPlan.actionSet,
            suites: evaluationPlan.suites,
            maxTokens: maxTokens
        )

        let report = Reporter.buildModelReport(
            modelName: modelPath.lastPathComponent,
            modelPath: modelPath,
            caseReports: evaluation.caseReports,
            promptCacheStatus: evaluation.promptCache.status,
            promptCachePath: evaluation.promptCache.path,
            warmupCompleted: true
        )
        try Reporter.writeModelReport(report, to: outputDirectory)
        return report
    }

    private static func loadModel(_ modelPath: URL) async throws -> ModelContainer {
        PlannerEvalMetalSupport.ensureEmbeddedMetalLibraryAvailable()
        print("loading_model=\(modelPath.lastPathComponent) path=\(modelPath.path)")
        return try await loadModelContainer(
            from: modelPath,
            using: #huggingFaceTokenizerLoader()
        )
    }

    private struct ModelEvaluation {
        let promptCache: PromptCacheArtifact
        let caseReports: [CaseReport]
    }

    private static func evaluateModel(
        modelName: String,
        modelContainer: ModelContainer,
        actionSet: ActionSetFile,
        suites: [PreparedPlannerSuite],
        maxTokens: Int
    ) async throws -> ModelEvaluation {
        let systemPrompt = PlannerPromptBuilder.systemPrompt(actions: actionSet.actions)
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let promptCache = try await loadOrCreatePrefixCache(
            modelName: modelName,
            modelContainer: modelContainer,
            systemPrompt: systemPrompt,
            parameters: parameters
        )
        try await runWarmup(
            modelName: modelName,
            modelContainer: modelContainer,
            prefixCache: promptCache.cache,
            parameters: parameters
        )

        var reports: [CaseReport] = []
        let plannedTotalCases = suites.reduce(0) { $0 + $1.cases.count }
        var seen = 0

        for suite in suites {
            for testCase in suite.cases {
                let current = seen + 1
                let remaining = max(0, plannedTotalCases - current)
                print(
                    "testing_model=\(modelName) case_index=\(current)/\(plannedTotalCases) " +
                    "suite=\(suite.suiteID) suite_case_index=\(testCase.suiteCaseIndex) remaining=\(remaining)"
                )

                let session = ChatSession(
                    modelContainer,
                    instructions: nil,
                    cache: clonedCache(promptCache.cache),
                    generateParameters: parameters,
                    additionalContext: ["enable_thinking": false]
                )

                var rawOutput = ""
                var completionInfo: GenerateCompletionInfo?
                for try await event in session.streamDetails(
                    to: testCase.prompt,
                    images: [],
                    videos: []
                ) {
                    switch event {
                    case .chunk(let chunk):
                        rawOutput += chunk
                    case .info(let info):
                        completionInfo = info
                    case .toolCall:
                        break
                    }
                }

                guard let completionInfo else {
                    throw PlannerEvalError.generationDidNotReturnCompletionInfo
                }

                let validation = PlannerValidator.validate(
                    rawOutput: rawOutput,
                    allowedActions: actionSet.actions,
                    expectedLineCount: testCase.expectedLineCount
                )

                let metrics = CaseMetrics(
                    measurementMode: "warm",
                    promptTokenCount: completionInfo.promptTokenCount,
                    generationTokenCount: completionInfo.generationTokenCount,
                    ttftMilliseconds: completionInfo.promptTime * 1000,
                    promptTokensPerSecond: completionInfo.promptTokensPerSecond,
                    generationTokensPerSecond: completionInfo.tokensPerSecond,
                    generationTimeSeconds: completionInfo.generateTime
                )

                reports.append(
                    CaseReport(
                        suiteID: suite.suiteID,
                        caseID: testCase.caseID,
                        plannerInstruction: testCase.plannerInstruction,
                        assistantReply: testCase.assistantReply,
                        expectedLineCount: testCase.expectedLineCount,
                        rawOutput: rawOutput,
                        validation: validation,
                        metrics: metrics
                    )
                )

                seen += 1
            }
        }

        return ModelEvaluation(
            promptCache: promptCache,
            caseReports: reports
        )
    }

    private static func loadOrCreatePrefixCache(
        modelName: String,
        modelContainer: ModelContainer,
        systemPrompt: String,
        parameters: GenerateParameters
    ) async throws -> PromptCacheArtifact {
        let cacheURL = prefixCacheURL(modelName: modelName, systemPrompt: systemPrompt, parameters: parameters)
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            print("prefix_cache=hit path=\(cacheURL.path)")
            let (cache, _) = try loadPromptCache(url: cacheURL)
            return PromptCacheArtifact(cache: cache, status: "hit", path: cacheURL)
        }

        print("prefix_cache=miss path=\(cacheURL.path)")
        try FileManager.default.createDirectory(
            at: cacheURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        try await modelContainer.perform { context in
            let prefixInput = try await context.processor.prepare(
                input: UserInput(
                    chat: [.system(systemPrompt)],
                    additionalContext: ["enable_thinking": false]
                )
            )
            let cache = try prefillCache(
                input: prefixInput,
                model: context.model,
                parameters: parameters
            )
            try savePromptCache(
                url: cacheURL,
                cache: cache,
                metadata: [
                    "model_name": modelName,
                    "system_prompt_sha256": sha256(systemPrompt),
                ]
            )
        }

        let (cache, _) = try loadPromptCache(url: cacheURL)
        return PromptCacheArtifact(cache: cache, status: "miss", path: cacheURL)
    }

    private static func runWarmup(
        modelName: String,
        modelContainer: ModelContainer,
        prefixCache: [any KVCache],
        parameters: GenerateParameters
    ) async throws {
        print("warming_model=\(modelName)")
        let warmupSession = ChatSession(
            modelContainer,
            instructions: nil,
            cache: clonedCache(prefixCache),
            generateParameters: parameters,
            additionalContext: ["enable_thinking": false]
        )
        let warmupPrompt = """
        Planner payload:

        User message:
        Warmup.

        Sentences:
        1. Warmup.

        Output:
        1. [bodyAction, expression]
        2. [bodyAction, expression]
        3. [bodyAction, expression]
        """
        for try await _ in warmupSession.streamDetails(to: warmupPrompt, images: [], videos: []) {
            // discard
        }
    }

    private static func clonedCache(_ cache: [any KVCache]) -> [any KVCache] {
        cache.map { $0.copy() }
    }

    private static func prefixCacheURL(
        modelName: String,
        systemPrompt: String,
        parameters: GenerateParameters
    ) -> URL {
        let digest = sha256(
            """
            model=\(modelName)
            prompt=\(systemPrompt)
            prefill=\(parameters.prefillStepSize)
            maxTokens=\(parameters.maxTokens ?? -1)
            """
        )
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("prefix-cache", isDirectory: true)
            .appendingPathComponent(modelName, isDirectory: true)
            .appendingPathComponent("\(digest).safetensors")
    }

    private static func sha256(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func prefillCache(
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> [any KVCache] {
        let cache = model.newCache(parameters: parameters)
        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            let result = model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            eval(result.logits)
        case .logits(let result):
            eval(result.logits)
        }
        return cache
    }

}
