import Foundation

public enum Reporter {
    public static func buildModelReport(
        modelName: String,
        modelPath: URL,
        caseReports: [CaseReport],
        promptCacheStatus: String,
        promptCachePath: URL,
        warmupCompleted: Bool
    ) -> ModelReport {
        let totalCases = caseReports.count
        let validCases = caseReports.filter(\.validation.isValid).count
        let ttfts = caseReports.map(\.metrics.ttftMilliseconds)
        let promptTPS = caseReports.map(\.metrics.promptTokensPerSecond)
        let generationTPS = caseReports.map(\.metrics.generationTokensPerSecond)

        let grouped = Dictionary(grouping: caseReports, by: \.suiteID)
        let suiteSummary = grouped.mapValues { reports in
            let valid = reports.filter(\.validation.isValid).count
            return SuiteSummary(
                totalCases: reports.count,
                validCases: valid,
                validRate: reports.isEmpty ? 0 : Double(valid) / Double(reports.count)
            )
        }

        let summary = ModelSummary(
            totalCases: totalCases,
            validCases: validCases,
            validRate: totalCases == 0 ? 0 : Double(validCases) / Double(totalCases),
            promptCacheStatus: promptCacheStatus,
            promptCachePath: promptCachePath.path,
            warmupCompleted: warmupCompleted,
            warmCaseCount: caseReports.count,
            warmMeanTTFTMilliseconds: mean(ttfts),
            warmMeanPromptTokensPerSecond: mean(promptTPS),
            warmMeanGenerationTokensPerSecond: mean(generationTPS),
            suiteSummary: suiteSummary
        )

        return ModelReport(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            modelName: modelName,
            modelPath: modelPath.path,
            summary: summary,
            results: caseReports
        )
    }

    public static func writeModelReport(_ report: ModelReport, to outputDirectory: URL) throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let reportURL = outputDirectory.appendingPathComponent("report.json")
        try encoder.encode(report).write(to: reportURL)

        let summaryURL = outputDirectory.appendingPathComponent("summary.md")
        var lines: [String] = [
            "# Planner Format Eval",
            "",
            "- Model: `\(report.modelName)`",
            "- Total cases: `\(report.summary.totalCases)`",
            "- Valid cases: `\(report.summary.validCases)`",
            "- Valid rate: `\(percent(report.summary.validRate))`",
            "- Prompt cache: `\(report.summary.promptCacheStatus)`",
            "- Warmup completed: `\(report.summary.warmupCompleted)`",
            "- Warm cases: `\(report.summary.warmCaseCount)`",
            "- Warm mean TTFT: `\(String(format: "%.2f", report.summary.warmMeanTTFTMilliseconds)) ms`",
            "- Warm mean prompt token/s: `\(String(format: "%.2f", report.summary.warmMeanPromptTokensPerSecond))`",
            "- Warm mean generation token/s: `\(String(format: "%.2f", report.summary.warmMeanGenerationTokensPerSecond))`",
            "",
            "## Suites",
            "",
        ]
        for suiteID in report.summary.suiteSummary.keys.sorted() {
            if let suite = report.summary.suiteSummary[suiteID] {
                lines.append(
                    "- `\(suiteID)`: `\(suite.validCases)/\(suite.totalCases)` valid (`\(percent(suite.validRate))`)"
                )
            }
        }
        try lines.joined(separator: "\n").appending("\n").write(to: summaryURL, atomically: true, encoding: .utf8)
    }

    public static func writeBatchCSV(rows: [BatchCSVRow], to outputURL: URL) throws {
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let suiteIDs = Array(Set(rows.flatMap { $0.suiteSummary.keys })).sorted()

        var header = [
            "model_name",
            "model_path",
            "status",
            "error",
            "total_cases",
            "valid_cases",
            "valid_rate",
            "prompt_cache_status",
            "prompt_cache_path",
            "warmup_completed",
            "warm_case_count",
            "warm_mean_ttft_milliseconds",
            "warm_mean_prompt_tokens_per_second",
            "warm_mean_generation_tokens_per_second",
        ]
        for suiteID in suiteIDs {
            header.append("\(suiteID)_total_cases")
            header.append("\(suiteID)_valid_cases")
            header.append("\(suiteID)_valid_rate")
        }

        var rowsText = [csvLine(header)]
        for row in rows {
            var values = [
                row.modelName,
                row.modelPath,
                row.status,
                row.error,
                "\(row.totalCases)",
                "\(row.validCases)",
                String(format: "%.6f", row.validRate),
                row.promptCacheStatus,
                row.promptCachePath,
                row.warmupCompleted ? "true" : "false",
                "\(row.warmCaseCount)",
                String(format: "%.6f", row.warmMeanTTFTMilliseconds),
                String(format: "%.6f", row.warmMeanPromptTokensPerSecond),
                String(format: "%.6f", row.warmMeanGenerationTokensPerSecond),
            ]
            for suiteID in suiteIDs {
                if let suite = row.suiteSummary[suiteID] {
                    values.append("\(suite.totalCases)")
                    values.append("\(suite.validCases)")
                    values.append(String(format: "%.6f", suite.validRate))
                } else {
                    values.append("")
                    values.append("")
                    values.append("")
                }
            }
            rowsText.append(csvLine(values))
        }

        try rowsText.joined(separator: "\n").appending("\n").write(to: outputURL, atomically: true, encoding: .utf8)
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func percent(_ value: Double) -> String {
        String(format: "%.2f%%", value * 100)
    }

    private static func csvLine(_ fields: [String]) -> String {
        fields
            .map { field in
                let escaped = field.replacingOccurrences(of: "\"", with: "\"\"")
                return "\"\(escaped)\""
            }
            .joined(separator: ",")
    }
}
