import Foundation

public enum PlannerEvalMetalSupport {
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
