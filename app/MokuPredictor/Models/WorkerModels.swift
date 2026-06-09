import Foundation

struct HealthResponse: Decodable {
    var ok: Bool
    var version: String?
    var capabilities: [String: Bool]?
    var warnings: [String]?
    var python_executable: String?
    var python_version: String?
    var port_hint: String?
}

struct ConnectionTestResponse: Decodable {
    var ok: Bool
    var database_type: String?
    var server_version: String?
    var message: String?
    var technical_details: String?
    var suggested_fix: String?
    var masked_connection: String?
}

struct SchemaIntrospectResponse: Decodable {
    var ok: Bool
    var schema: JSONValue?
    var message: String?
    var technical_details: String?
    var suggested_fix: String?
}

struct LLMGenerateResponse: Decodable {
    var ok: Bool
    var status: String?
    var task_type: String?
    var target_column: String?
    var entity_id_columns: [String]?
    var sql: String?
    var reasoning_summary: String?
    var leakage_warnings: [String]?
    var assumptions: [String]?
    var message: String?
    var technical_details: String?
    var suggested_fix: String?
}

struct OllamaStatusResponse: Decodable {
    var ok: Bool
    var server_running: Bool?
    var model_available: Bool?
    var message: String?
    var installed_models: [String]?
    var technical_details: String?
    var suggested_fix: String?
}

struct LLMChatMessage: Codable, Hashable {
    var role: String
    var content: String
}

struct LLMBuildPromptResponse: Decodable {
    var ok: Bool
    var messages: [LLMChatMessage]?
    var message: String?
    var technical_details: String?
}

struct SQLValidateResponse: Decodable {
    var ok: Bool
    var is_safe: Bool?
    var errors: [String]?
    var normalized_sql: String?
    var warnings: [String]?
    var validator: String?
}

struct ColumnInfo: Decodable, Identifiable, Hashable {
    var id: String { name }
    var name: String
    var dtype: String
}

struct PreviewResponse: Decodable {
    var ok: Bool
    var columns: [ColumnInfo]?
    var row_count: Int?
    var preview: [JSONValue]?
    var warnings: [String]?
    var normalized_sql: String?
    var message: String?
    var technical_details: String?
    var suggested_fix: String?
    var errors: [String]?
}

struct MLRunResponse: Decodable {
    var ok: Bool
    var task_type: String?
    var target_column: String?
    var entity_id_columns: [String]?
    var model: JSONValue?
    var dataset: JSONValue?
    var metrics: JSONValue?
    var predictions: [JSONValue]?
    var warnings: [String]?
    var message: String?
    var technical_details: String?
    var suggested_fix: String?
    var errors: [String]?
}

struct TableSummary: Identifiable, Hashable {
    var id: String { qualifiedName }
    var schema: String
    var name: String
    var columnCount: Int
    var foreignKeyCount: Int
    var indexCount: Int

    var qualifiedName: String {
        schema.isEmpty ? name : "\(schema).\(name)"
    }
}

enum WorkflowStep: String, CaseIterable, Identifiable, Hashable {
    case connect
    case describe
    case schema
    case sql
    case preview
    case predict
    case export

    var id: String { rawValue }

    var title: String {
        switch self {
        case .connect: "Connect Database"
        case .describe: "Describe Prediction"
        case .schema: "Inspect Schema"
        case .sql: "Generate SQL"
        case .preview: "Preview ML Table"
        case .predict: "Run Prediction"
        case .export: "Export"
        }
    }

    var symbol: String {
        switch self {
        case .connect: "cylinder.split.1x2"
        case .describe: "text.bubble"
        case .schema: "tablecells"
        case .sql: "chevron.left.forwardslash.chevron.right"
        case .preview: "doc.text.magnifyingglass"
        case .predict: "chart.line.uptrend.xyaxis"
        case .export: "square.and.arrow.down"
        }
    }
}

enum LLMProviderChoice: String, CaseIterable, Identifiable {
    case localOllama = "local_ollama"
    case offlineTemplate = "offline_template"
    case openAICompatible = "openai_compatible"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .localOllama: "Local Ollama"
        case .offlineTemplate: "Offline Template"
        case .openAICompatible: "OpenAI-Compatible API"
        }
    }
}

enum TaskTypeChoice: String, CaseIterable, Identifiable {
    case unknown
    case classification
    case regression

    var id: String { rawValue }

    var title: String {
        rawValue.capitalized
    }
}
