import Foundation

public enum PlannerEvalError: LocalizedError {
    case noSuitesMatched([String])
    case invalidModelsDirectory(URL)
    case noModelDirectories(URL)
    case generationDidNotReturnCompletionInfo

    public var errorDescription: String? {
        switch self {
        case .noSuitesMatched(let suiteIDs):
            return "No suites matched: \(suiteIDs.joined(separator: ", "))"
        case .invalidModelsDirectory(let url):
            return "Invalid models directory: \(url.path)"
        case .noModelDirectories(let url):
            return "No model directories found in: \(url.path)"
        case .generationDidNotReturnCompletionInfo:
            return "Generation finished without completion metrics."
        }
    }
}
