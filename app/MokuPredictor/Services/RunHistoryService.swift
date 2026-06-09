import CryptoKit
import Foundation

struct RunAuditRecord: Codable {
    var timestamp: String
    var databaseType: String
    var maskedConnection: String?
    var userPrompt: String
    var schemaHash: String?
    var selectedTables: [String]
    var llmProviderType: String
    var llmModelName: String
    var generatedSQL: String
    var sqlValidator: String?
    var sqlWasSafe: Bool
    var targetColumn: String
    var taskType: String
    var dataset: JSONValue?
    var model: JSONValue?
    var metrics: JSONValue?
    var warnings: [String]
}

enum RunHistoryService {
    static func save(_ record: RunAuditRecord) throws -> URL {
        let directory = try runsDirectory()
        let fileName = "run-\(record.timestamp.replacingOccurrences(of: ":", with: "-")).json"
        let url = directory.appendingPathComponent(fileName)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(record)
        try data.write(to: url, options: [.atomic])
        return url
    }

    static func schemaHash(_ schema: JSONValue?) -> String? {
        guard let schema else { return nil }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(schema) else { return nil }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func runsDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base.appendingPathComponent("MokuML/runs", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

