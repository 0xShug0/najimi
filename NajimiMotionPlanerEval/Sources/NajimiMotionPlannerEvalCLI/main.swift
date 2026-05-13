import ArgumentParser
import Foundation
import PlannerEvalCore

@main
struct NajimiMotionPlannerEvalCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "najimi-motion-planner-eval",
        abstract: "Standalone planner format evaluator for local MLX model directories.",
        subcommands: [Run.self, RunDirectory.self]
    )
}

struct SharedArguments: ParsableArguments {
    @Option(name: .long, help: "Directory containing standalone cases.")
    var casesDir: String = "cases"

    @Option(name: .long, help: "Maximum generated tokens per case.")
    var maxTokens: Int = 128

    @Option(name: .long, help: "Stop after this many total cases.")
    var caseLimit: Int?

    @Option(name: .long, help: "Maximum cases to run from each suite.")
    var maxCasesPerSuite: Int?

    @Option(name: .long, parsing: .upToNextOption, help: "Restrict to one or more suite IDs.")
    var suite: [String] = []
}

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run planner format validation on one local model directory."
    )

    @Option(name: .long, help: "Absolute path to one local model directory.")
    var modelPath: String

    @Option(name: .long, help: "Directory to write report.json and summary.md.")
    var outputDir: String

    @OptionGroup var shared: SharedArguments

    mutating func run() async throws {
        let report = try await PlannerEvalRunner.runSingle(
            .init(
                modelPath: URL(fileURLWithPath: modelPath, isDirectory: true),
                casesDirectory: URL(fileURLWithPath: shared.casesDir, isDirectory: true),
                outputDirectory: URL(fileURLWithPath: outputDir, isDirectory: true),
                maxTokens: shared.maxTokens,
                caseLimit: shared.caseLimit,
                maxCasesPerSuite: shared.maxCasesPerSuite,
                suiteFilter: Set(shared.suite)
            )
        )

        print("model=\(report.modelName)")
        print("total_cases=\(report.summary.totalCases)")
        print("valid_cases=\(report.summary.validCases)")
        print("valid_rate=\(String(format: "%.2f%%", report.summary.validRate * 100))")
        print("prompt_cache_status=\(report.summary.promptCacheStatus)")
        print("prompt_cache_path=\(report.summary.promptCachePath)")
        print("warmup_completed=\(report.summary.warmupCompleted)")
        print("warm_case_count=\(report.summary.warmCaseCount)")
        print("warm_mean_ttft_ms=\(String(format: "%.2f", report.summary.warmMeanTTFTMilliseconds))")
        print("warm_mean_prompt_tokens_per_second=\(String(format: "%.2f", report.summary.warmMeanPromptTokensPerSecond))")
        print("warm_mean_generation_tokens_per_second=\(String(format: "%.2f", report.summary.warmMeanGenerationTokensPerSecond))")
        print("report=\(URL(fileURLWithPath: outputDir).appendingPathComponent("report.json").path)")
        print("summary=\(URL(fileURLWithPath: outputDir).appendingPathComponent("summary.md").path)")
    }
}

struct RunDirectory: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run-dir",
        abstract: "Run planner format validation on every model directory under a root and emit a CSV summary."
    )

    @Option(name: .long, help: "Directory whose child directories are model directories.")
    var modelsDir: String

    @Option(name: .long, help: "Directory to write summary.csv and per-model subdirectories.")
    var outputDir: String

    @Option(name: .long, parsing: .upToNextOption, help: "Restrict the batch to specific child model directory names.")
    var model: [String] = []

    @OptionGroup var shared: SharedArguments

    mutating func run() async throws {
        let csvURL = try await PlannerEvalRunner.runDirectory(
            .init(
                modelsDirectory: URL(fileURLWithPath: modelsDir, isDirectory: true),
                casesDirectory: URL(fileURLWithPath: shared.casesDir, isDirectory: true),
                outputDirectory: URL(fileURLWithPath: outputDir, isDirectory: true),
                modelFilter: Set(model),
                maxTokens: shared.maxTokens,
                caseLimit: shared.caseLimit,
                maxCasesPerSuite: shared.maxCasesPerSuite,
                suiteFilter: Set(shared.suite)
            )
        )
        print("csv=\(csvURL.path)")
    }
}
