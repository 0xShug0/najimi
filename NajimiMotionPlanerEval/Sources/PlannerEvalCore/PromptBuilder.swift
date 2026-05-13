import Foundation
import NaturalLanguage

public enum PlannerPromptBuilder {
    public static let allowedExpressions = [
        "neutral",
        "blink",
        "joy",
        "angry",
        "sorrow",
        "fun",
        "lookup",
        "lookdown",
        "lookleft",
        "lookright",
        "blink_l",
        "blink_r",
    ]

    public static func systemPrompt(actions: [String]) -> String {
        let exampleA = actions.indices.contains(0) ? actions[0] : "Relax"
        let exampleB = actions.indices.contains(1) ? actions[1] : exampleA
        let exampleC = actions.indices.contains(2) ? actions[2] : exampleA
        return """
        You are an internal animation planner for a speaking avatar.
        Never answer with prose, JSON, markdown, code fences, explanations, labels, or sentence text.
        Build one compact tag for each sentence in the assistant reply.
        Choose one VRMA body action name from the allowed list and one expression from the allowed list for each sentence.
        Do not invent new action names.
        Use only the provided VRMA body actions. Do not use generic built-in actions.
        Prefer subtle movement for factual or summary content and use the calmest suitable action when unsure.
        Use higher-energy motion only when the sentence genuinely feels playful, excited, or celebratory.
        Output exactly one line per sentence in this format:
        [bodyAction, expression]
        Repeat one line per sentence in the same order as the reply.

        Available VRMA body actions:
        \(actions.joined(separator: ", "))

        Available expressions:
        \(allowedExpressions.joined(separator: ", "))

        Choose each bodyAction token exactly from the allowed VRMA body actions list.
        Choose each expression token exactly from the allowed expressions list.
        Do not paraphrase, translate, explain, or invent labels.
        The user prompt will provide a compact planner payload in this exact order:
        User message:
        Sentences:
        Output:
        The Sentences block is authoritative and must be planned in order.
        Example output for three sentences:
        1. [\(exampleA), neutral]
        2. [\(exampleB), joy]
        3. [\(exampleC), neutral]
        """
    }

    public static func splitSentences(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = trimmed
        var sentences: [String] = []
        tokenizer.enumerateTokens(in: trimmed.startIndex..<trimmed.endIndex) { range, _ in
            let sentence = String(trimmed[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }
        return sentences.isEmpty ? [trimmed] : sentences
    }

    public static func userPrompt(for testCase: PlannerCase) -> (prompt: String, expectedLineCount: Int) {
        let sentences = splitSentences(testCase.assistantReply)
        let sentenceBlock = sentences.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        let prompt = """
        Planner payload:

        User message:
        \(testCase.plannerInstruction)

        Sentences:
        \(sentenceBlock)

        Output:
        1. [bodyAction, expression]
        2. [bodyAction, expression]
        3. [bodyAction, expression]
        """
        return (prompt, sentences.count)
    }
}
