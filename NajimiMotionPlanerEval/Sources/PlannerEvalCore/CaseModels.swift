import Foundation

public struct ActionSetFile: Decodable, Sendable {
    public let actionSetID: String
    public let actions: [String]

    enum CodingKeys: String, CodingKey {
        case actionSetID = "action_set_id"
        case actions
    }
}

public struct PlannerCase: Decodable, Sendable {
    public let caseID: String
    public let plannerInstruction: String
    public let assistantReply: String

    enum CodingKeys: String, CodingKey {
        case caseID = "case_id"
        case plannerInstruction = "planner_instruction"
        case assistantReply = "assistant_reply"
    }
}

public struct PlannerSuite: Decodable, Sendable {
    public let suiteID: String
    public let promptContract: String
    public let actionSetID: String
    public let cases: [PlannerCase]

    enum CodingKeys: String, CodingKey {
        case suiteID = "suite_id"
        case promptContract = "prompt_contract"
        case actionSetID = "action_set_id"
        case cases
    }
}

public struct LoadedSuites: Sendable {
    public let actionSet: ActionSetFile
    public let suites: [PlannerSuite]
}

public struct PreparedPlannerCase: Sendable {
    public let suiteID: String
    public let suiteCaseIndex: Int
    public let caseID: String
    public let plannerInstruction: String
    public let assistantReply: String
    public let prompt: String
    public let expectedLineCount: Int
}

public struct PreparedPlannerSuite: Sendable {
    public let suiteID: String
    public let cases: [PreparedPlannerCase]
}

public struct LoadedEvaluationPlan: Sendable {
    public let actionSet: ActionSetFile
    public let suites: [PreparedPlannerSuite]

    public var totalCases: Int {
        suites.reduce(0) { $0 + $1.cases.count }
    }
}

public enum CaseLoader {
    public static func load(from casesDirectory: URL, suiteFilter: Set<String> = []) throws -> LoadedSuites {
        let decoder = JSONDecoder()
        let actionSetURL = casesDirectory
            .appendingPathComponent("action_sets", isDirectory: true)
            .appendingPathComponent("vrma_builtin_50.json")
        let suitesDirectory = casesDirectory.appendingPathComponent("suites", isDirectory: true)

        let actionSet = try decoder.decode(ActionSetFile.self, from: Data(contentsOf: actionSetURL))

        let suiteURLs = try FileManager.default.contentsOfDirectory(
            at: suitesDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "json" && !$0.lastPathComponent.hasPrefix("._") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let suites = try suiteURLs.map { try decoder.decode(PlannerSuite.self, from: Data(contentsOf: $0)) }
        let filteredSuites =
            suiteFilter.isEmpty
            ? suites
            : suites.filter { suiteFilter.contains($0.suiteID) }

        if filteredSuites.isEmpty {
            throw PlannerEvalError.noSuitesMatched(Array(suiteFilter).sorted())
        }

        return LoadedSuites(actionSet: actionSet, suites: filteredSuites)
    }

    public static func loadPlan(
        from casesDirectory: URL,
        suiteFilter: Set<String> = [],
        caseLimit: Int? = nil,
        maxCasesPerSuite: Int? = nil
    ) throws -> LoadedEvaluationPlan {
        let loadedSuites = try load(from: casesDirectory, suiteFilter: suiteFilter)
        return compilePlan(
            actionSet: loadedSuites.actionSet,
            suites: loadedSuites.suites,
            caseLimit: caseLimit,
            maxCasesPerSuite: maxCasesPerSuite
        )
    }

    private static func compilePlan(
        actionSet: ActionSetFile,
        suites: [PlannerSuite],
        caseLimit: Int?,
        maxCasesPerSuite: Int?
    ) -> LoadedEvaluationPlan {
        var compiledSuites: [PreparedPlannerSuite] = []
        var totalSeen = 0

        for suite in suites {
            if let caseLimit, totalSeen >= caseLimit {
                break
            }

            var compiledCases: [PreparedPlannerCase] = []
            for (offset, testCase) in suite.cases.enumerated() {
                if let caseLimit, totalSeen >= caseLimit {
                    break
                }
                if let maxCasesPerSuite, compiledCases.count >= maxCasesPerSuite {
                    break
                }

                let prompt = PlannerPromptBuilder.userPrompt(for: testCase)
                compiledCases.append(
                    PreparedPlannerCase(
                        suiteID: suite.suiteID,
                        suiteCaseIndex: offset + 1,
                        caseID: testCase.caseID,
                        plannerInstruction: testCase.plannerInstruction,
                        assistantReply: testCase.assistantReply,
                        prompt: prompt.prompt,
                        expectedLineCount: prompt.expectedLineCount
                    )
                )
                totalSeen += 1
            }

            if !compiledCases.isEmpty {
                compiledSuites.append(
                    PreparedPlannerSuite(
                        suiteID: suite.suiteID,
                        cases: compiledCases
                    )
                )
            }
        }

        return LoadedEvaluationPlan(
            actionSet: actionSet,
            suites: compiledSuites
        )
    }
}
