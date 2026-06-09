import Foundation

enum WorkerEndpoint {
    static let port: Int = {
        if let raw = ProcessInfo.processInfo.environment["MOKU_WORKER_PORT"],
           let value = Int(raw),
           value > 0 {
            return value
        }
        return 18765
    }()

    static var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }
}
