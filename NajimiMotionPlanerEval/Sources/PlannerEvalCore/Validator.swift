import Foundation

public struct ParsedPlannerLine: Codable, Sendable {
    public let bodyAction: String
    public let expression: String
}

public struct ValidationResult: Codable, Sendable {
    public let isValid: Bool
    public let expectedLineCount: Int
    public let actualLineCount: Int
    public let parsedLines: [ParsedPlannerLine]
    public let errors: [String]
}

public enum PlannerValidator {
    private static let linePattern = try! NSRegularExpression(
        pattern: #"^(?:\d+\.\s*)?\[([^,\]]+)\s*,\s*([^\]]+)\]$"#
    )

    public static func validate(
        rawOutput: String,
        allowedActions: [String],
        expectedLineCount: Int
    ) -> ValidationResult {
        let actionMap = Dictionary(uniqueKeysWithValues: allowedActions.map { ($0.lowercased(), $0) })
        let expressionMap = Dictionary(
            uniqueKeysWithValues: PlannerPromptBuilder.allowedExpressions.map { ($0.lowercased(), $0) }
        )

        let lines = rawOutput
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var errors: [String] = []
        if lines.count != expectedLineCount {
            errors.append("expected \(expectedLineCount) non-empty output lines, found \(lines.count)")
        }

        var parsed: [ParsedPlannerLine] = []
        for (index, line) in lines.enumerated() {
            let range = NSRange(line.startIndex..<line.endIndex, in: line)
            guard let match = linePattern.firstMatch(in: line, range: range),
                  let bodyRange = Range(match.range(at: 1), in: line),
                  let expressionRange = Range(match.range(at: 2), in: line) else {
                errors.append("line \(index + 1) is not valid planner syntax: \(line)")
                continue
            }

            let bodyRaw = String(line[bodyRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            let expressionRaw = String(line[expressionRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let bodyAction = actionMap[bodyRaw.lowercased()] else {
                errors.append("line \(index + 1) bodyAction is not in the allowed action set: \(bodyRaw)")
                continue
            }
            guard let expression = expressionMap[expressionRaw.lowercased()] else {
                errors.append("line \(index + 1) expression is not in the allowed expression set: \(expressionRaw)")
                continue
            }

            parsed.append(ParsedPlannerLine(bodyAction: bodyAction, expression: expression))
        }

        return ValidationResult(
            isValid: errors.isEmpty,
            expectedLineCount: expectedLineCount,
            actualLineCount: lines.count,
            parsedLines: parsed,
            errors: errors
        )
    }
}
