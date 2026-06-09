import Combine
import Foundation

@MainActor
final class WorkerProcess: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Worker stopped"
    @Published private(set) var health: HealthResponse?
    @Published private(set) var lastLog = ""

    private var process: Process?
    private let client = WorkerClient()
    private let port = WorkerEndpoint.port

    func start() async {
        if process?.isRunning == true {
            await refreshHealth()
            return
        }

        let root = Self.resolveProjectRoot()
        let workerSource = root.appendingPathComponent("worker/src").path
        let workerRoot = root.appendingPathComponent("worker").path
        let pythonExecutable = Self.resolvePythonExecutable(root: root)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable.executable)
        process.arguments = [
            "-m",
            "worker.main",
            "--host",
            "127.0.0.1",
            "--port",
            "\(port)",
        ]
        process.currentDirectoryURL = URL(fileURLWithPath: workerRoot)

        var environment = ProcessInfo.processInfo.environment
        let existingPythonPath = environment["PYTHONPATH"]
        environment["PYTHONPATH"] = [workerSource, existingPythonPath].compactMap { $0 }.joined(separator: ":")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        process.environment = environment

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.lastLog = text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        do {
            try process.run()
            self.process = process
            isRunning = true
            status = "Starting worker with \(pythonExecutable.label) on port \(port)..."
        } catch {
            status = "Could not start worker: \(error.localizedDescription)"
            isRunning = false
            return
        }

        for _ in 0..<40 {
            try? await Task.sleep(nanoseconds: 250_000_000)
            do {
                let health: HealthResponse = try await client.get("health")
                self.health = health
                status = health.ok ? "Worker ready" : "Worker responded with an error"
                return
            } catch {
                continue
            }
        }
        status = "Worker started, but health check is not responding yet"
    }

    func refreshHealth() async {
        do {
            let health: HealthResponse = try await client.get("health")
            self.health = health
            status = health.ok ? "Worker ready" : "Worker responded with an error"
            isRunning = true
        } catch {
            status = "Worker health unavailable: \(error.localizedDescription)"
            health = nil
            isRunning = process?.isRunning == true
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
        status = "Worker stopped"
    }

    private static func resolveProjectRoot() -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let explicit = environment["MOKU_PROJECT_ROOT"] {
            return URL(fileURLWithPath: explicit)
        }

        if let resources = Bundle.main.resourceURL,
           FileManager.default.fileExists(atPath: resources.appendingPathComponent("worker/src").path) {
            return resources
        }

        let current = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        if FileManager.default.fileExists(atPath: current.appendingPathComponent("worker/src").path) {
            return current
        }

        var candidate = Bundle.main.bundleURL
        for _ in 0..<6 {
            if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("worker/src").path) {
                return candidate
            }
            candidate.deleteLastPathComponent()
        }
        return current
    }

    private static func resolvePythonExecutable(root: URL) -> (executable: String, label: String) {
        let bundledPython = root.appendingPathComponent("python/bin/python")
        if FileManager.default.isExecutableFile(atPath: bundledPython.path) {
            return (bundledPython.path, "bundled Python")
        }

        let venvPython = root.appendingPathComponent("worker/.venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return (venvPython.path, "worker/.venv")
        }

        for candidate in ["/opt/homebrew/bin/python3.11", "/usr/local/bin/python3.11", "/usr/bin/python3"] {
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return (candidate, (candidate as NSString).lastPathComponent)
            }
        }
        return ("/usr/bin/python3", "python3")
    }

}
