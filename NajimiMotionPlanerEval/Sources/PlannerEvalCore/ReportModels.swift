import Foundation

public struct CaseMetrics: Codable, Sendable {
    public let measurementMode: String
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let ttftMilliseconds: Double
    public let promptTokensPerSecond: Double
    public let generationTokensPerSecond: Double
    public let generationTimeSeconds: Double

    enum CodingKeys: String, CodingKey {
        case measurementMode = "measurement_mode"
        case promptTokenCount = "prompt_token_count"
        case generationTokenCount = "generation_token_count"
        case ttftMilliseconds = "ttft_milliseconds"
        case promptTokensPerSecond = "prompt_tokens_per_second"
        case generationTokensPerSecond = "generation_tokens_per_second"
        case generationTimeSeconds = "generation_time_seconds"
    }
}

public struct CaseReport: Codable, Sendable {
    public let suiteID: String
    public let caseID: String
    public let plannerInstruction: String
    public let assistantReply: String
    public let expectedLineCount: Int
    public let rawOutput: String
    public let validation: ValidationResult
    public let metrics: CaseMetrics

    enum CodingKeys: String, CodingKey {
        case suiteID = "suite_id"
        case caseID = "case_id"
        case plannerInstruction = "planner_instruction"
        case assistantReply = "assistant_reply"
        case expectedLineCount = "expected_line_count"
        case rawOutput = "raw_output"
        case validation
        case metrics
    }
}

public struct SuiteSummary: Codable, Sendable {
    public let totalCases: Int
    public let validCases: Int
    public let validRate: Double

    enum CodingKeys: String, CodingKey {
        case totalCases = "total_cases"
        case validCases = "valid_cases"
        case validRate = "valid_rate"
    }
}

public struct ModelSummary: Codable, Sendable {
    public let totalCases: Int
    public let validCases: Int
    public let validRate: Double
    public let promptCacheStatus: String
    public let promptCachePath: String
    public let warmupCompleted: Bool
    public let warmCaseCount: Int
    public let warmMeanTTFTMilliseconds: Double
    public let warmMeanPromptTokensPerSecond: Double
    public let warmMeanGenerationTokensPerSecond: Double
    public let suiteSummary: [String: SuiteSummary]

    enum CodingKeys: String, CodingKey {
        case totalCases = "total_cases"
        case validCases = "valid_cases"
        case validRate = "valid_rate"
        case promptCacheStatus = "prompt_cache_status"
        case promptCachePath = "prompt_cache_path"
        case warmupCompleted = "warmup_completed"
        case warmCaseCount = "warm_case_count"
        case warmMeanTTFTMilliseconds = "warm_mean_ttft_milliseconds"
        case warmMeanPromptTokensPerSecond = "warm_mean_prompt_tokens_per_second"
        case warmMeanGenerationTokensPerSecond = "warm_mean_generation_tokens_per_second"
        case suiteSummary = "suite_summary"
    }
}

public struct ModelReport: Codable, Sendable {
    public let generatedAt: String
    public let modelName: String
    public let modelPath: String
    public let summary: ModelSummary
    public let results: [CaseReport]

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case modelName = "model_name"
        case modelPath = "model_path"
        case summary
        case results
    }
}

public struct BatchCSVRow: Sendable {
    public let modelName: String
    public let modelPath: String
    public let status: String
    public let error: String
    public let totalCases: Int
    public let validCases: Int
    public let validRate: Double
    public let promptCacheStatus: String
    public let promptCachePath: String
    public let warmupCompleted: Bool
    public let warmCaseCount: Int
    public let warmMeanTTFTMilliseconds: Double
    public let warmMeanPromptTokensPerSecond: Double
    public let warmMeanGenerationTokensPerSecond: Double
    public let suiteSummary: [String: SuiteSummary]
}
