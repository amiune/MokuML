import AppKit
import Combine
import Foundation
import UniformTypeIdentifiers

@MainActor
final class AppViewModel: ObservableObject {
    @Published var selectedStep: WorkflowStep = .connect
    @Published var connectionString = ""
    @Published var saveConnection = false
    @Published var predictionPrompt = "Predict which customers are likely to churn next month."
    @Published var selectedExample = "Predict which customers are likely to churn next month."
    @Published var includeSchemas = "public"
    @Published var llmProvider: LLMProviderChoice = .localOllama
    @Published var localEndpoint = "http://127.0.0.1:11434"
    @Published var localModel = "sqlcoder"
    @Published var installedOllamaModels: [String] = []
    @Published var apiBaseURL = "https://api.openai.com/v1"
    @Published var apiModel = "gpt-4.1-mini"
    @Published var apiKey = ""
    @Published var schema: JSONValue?
    @Published var generatedSQL = ""
    @Published var targetColumn = ""
    @Published var entityIDColumns = ""
    @Published var taskType: TaskTypeChoice = .unknown
    @Published var connectionResult: ConnectionTestResponse?
    @Published var ollamaStatus: OllamaStatusResponse?
    @Published var sqlValidation: SQLValidateResponse?
    @Published var previewColumns: [ColumnInfo] = []
    @Published var previewRows: [JSONValue] = []
    @Published var previewRowCount: Int?
    @Published var predictions: [JSONValue] = []
    @Published var metrics: JSONValue?
    @Published var dataset: JSONValue?
    @Published var modelInfo: JSONValue?
    @Published var assumptions: [String] = []
    @Published var leakageWarnings: [String] = []
    @Published var warnings: [String] = []
    @Published var statusMessage = "Start the worker, then connect to Postgres."
    @Published var technicalDetails = ""
    @Published var isWorking = false

    let examples = [
        "Predict which customers are likely to churn next month.",
        "Predict the expected order value for each open quote.",
        "Predict which invoices are likely to be paid late.",
    ]

    private let client = WorkerClient()
    private let savedConnectionAccount = "default-postgres-connection"

    init() {
        if let savedConnection = KeychainService.read(account: savedConnectionAccount) {
            connectionString = savedConnection
            saveConnection = true
        }
    }

    var schemaTables: [TableSummary] {
        guard let schemaObject = schema?.objectValue else { return [] }
        return (schemaObject["schemas"]?.arrayValue ?? []).flatMap { schemaValue -> [TableSummary] in
            guard let schemaDict = schemaValue.objectValue else { return [] }
            let schemaName = schemaDict["name"]?.displayString ?? ""
            return (schemaDict["tables"]?.arrayValue ?? []).compactMap { tableValue in
                guard let table = tableValue.objectValue else { return nil }
                return TableSummary(
                    schema: schemaName,
                    name: table["name"]?.displayString ?? "",
                    columnCount: table["columns"]?.arrayValue?.count ?? 0,
                    foreignKeyCount: table["foreign_keys"]?.arrayValue?.count ?? 0,
                    indexCount: table["indexes"]?.arrayValue?.count ?? 0
                )
            }
        }
    }

    func useSelectedExample() {
        predictionPrompt = selectedExample
    }

    func testConnection() async {
        await run("Testing database connection...") {
            let response: ConnectionTestResponse = try await client.post(
                "connect/test",
                payload: ["connection_string": .string(connectionString)]
            )
            connectionResult = response
            statusMessage = response.message ?? (response.ok ? "Connection successful" : "Connection failed")
            technicalDetails = [
                response.technical_details,
                response.suggested_fix.map { "Suggested fix: \($0)" },
            ]
            .compactMap { $0 }
            .joined(separator: "\n\n")
            if response.ok {
                if saveConnection {
                    try KeychainService.save(connectionString, account: savedConnectionAccount)
                }
                selectedStep = .describe
            }
        }
    }

    func introspectSchema() async {
        await run("Inspecting schema metadata locally...") {
            let schemas = includeSchemas
                .split(separator: ",")
                .map { JSONValue.string($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            let response: SchemaIntrospectResponse = try await client.post(
                "schema/introspect",
                payload: [
                    "connection_string": .string(connectionString),
                    "include_schemas": .array(schemas),
                    "exclude_tables": .array([]),
                ]
            )
            if response.ok, let schema = response.schema {
                self.schema = schema
                statusMessage = "Schema loaded: \(schemaTables.count) tables"
                selectedStep = .schema
            } else {
                statusMessage = response.message ?? "Schema introspection failed"
                technicalDetails = response.technical_details ?? ""
            }
        }
    }

    func generateSQL() async {
        guard let schema else {
            statusMessage = "Inspect schema before generating SQL."
            return
        }
        if llmProvider == .localOllama {
            let status = await checkOllamaStatus(silent: true)
            guard status?.ok == true else {
                statusMessage = status?.message ?? "Ollama is not ready."
                technicalDetails = ollamaSetupDetails(status)
                return
            }
            await generateSQLWithLocalOllama(schema: schema)
            return
        }
        await run("Generating SQL from schema metadata and prompt...") {
            let response: LLMGenerateResponse = try await client.post(
                "llm/generate-sql",
                payload: [
                    "prediction_prompt": .string(predictionPrompt),
                    "schema": schema,
                    "llm_provider": .object(providerPayload()),
                ]
            )
            if response.ok {
                generatedSQL = response.sql ?? generatedSQL
                targetColumn = response.target_column ?? targetColumn
                entityIDColumns = (response.entity_id_columns ?? []).joined(separator: ", ")
                if let task = response.task_type, let choice = TaskTypeChoice(rawValue: task) {
                    taskType = choice
                }
                assumptions = response.assumptions ?? []
                leakageWarnings = response.leakage_warnings ?? []
                statusMessage = response.reasoning_summary ?? "SQL generated"
                selectedStep = .sql
            } else {
                statusMessage = response.message ?? "SQL generation failed"
                technicalDetails = response.technical_details ?? ""
            }
        }
    }

    @discardableResult
    func checkOllamaStatus(silent: Bool = false) async -> OllamaStatusResponse? {
        if !silent {
            isWorking = true
            statusMessage = "Checking Ollama..."
            technicalDetails = ""
        }
        defer {
            if !silent {
                isWorking = false
            }
        }
        do {
            let response = try await fetchOllamaStatus()
            ollamaStatus = response
            if !silent {
                statusMessage = response.message ?? (response.ok ? "Ollama is ready" : "Ollama needs setup")
                technicalDetails = ollamaSetupDetails(response)
            }
            return response
        } catch {
            let response = OllamaStatusResponse(
                ok: false,
                server_running: false,
                model_available: false,
                message: "Could not check Ollama status.",
                installed_models: nil,
                technical_details: String(describing: error),
                suggested_fix: "Start Ollama, then run: ollama pull \(localModel)"
            )
            ollamaStatus = response
            if !silent {
                statusMessage = response.message ?? "Ollama needs setup"
                technicalDetails = ollamaSetupDetails(response)
            }
            return response
        }
    }

    func useOfflineTemplateProvider() {
        llmProvider = .offlineTemplate
        statusMessage = "Offline Template selected"
        technicalDetails = ""
    }

    func selectLocalOllamaModel(_ modelName: String) {
        localModel = modelName
        guard let status = ollamaStatus, let installedModels = status.installed_models else { return }
        let modelAvailable = installedModels.contains { Self.ollamaModelName($0, matches: modelName) }
        ollamaStatus = OllamaStatusResponse(
            ok: status.server_running == true && modelAvailable,
            server_running: status.server_running,
            model_available: modelAvailable,
            message: modelAvailable
                ? "Ollama is ready with \(modelName)."
                : "Ollama is running, but \(modelName) is not installed.",
            installed_models: installedModels,
            technical_details: status.technical_details,
            suggested_fix: modelAvailable ? nil : "Run: ollama pull \(modelName)"
        )
    }

    func openOllamaDownload() {
        if let url = URL(string: "https://ollama.com/download") {
            NSWorkspace.shared.open(url)
        }
    }

    func openOllamaApp() {
        let appURL = URL(fileURLWithPath: "/Applications/Ollama.app")
        if FileManager.default.fileExists(atPath: appURL.path) {
            NSWorkspace.shared.openApplication(
                at: appURL,
                configuration: NSWorkspace.OpenConfiguration()
            )
        } else {
            openOllamaDownload()
        }
    }

    private func generateSQLWithLocalOllama(schema: JSONValue) async {
        await run("Generating SQL with local Ollama...") {
            let prompt: LLMBuildPromptResponse = try await client.post(
                "llm/build-prompt",
                payload: [
                    "prediction_prompt": .string(predictionPrompt),
                    "schema": schema,
                ]
            )
            guard prompt.ok, let messages = prompt.messages else {
                statusMessage = prompt.message ?? "Could not build LLM prompt."
                technicalDetails = prompt.technical_details ?? ""
                return
            }
            let rawContent = try await callLocalOllama(messages: messages)
            let response: LLMGenerateResponse = try await client.post(
                "llm/parse-sql",
                payload: ["raw_content": .string(rawContent)]
            )
            applyGenerateSQLResponse(response)
        }
    }

    private func applyGenerateSQLResponse(_ response: LLMGenerateResponse) {
        if response.ok {
            generatedSQL = response.sql ?? generatedSQL
            targetColumn = response.target_column ?? targetColumn
            entityIDColumns = (response.entity_id_columns ?? []).joined(separator: ", ")
            if let task = response.task_type, let choice = TaskTypeChoice(rawValue: task) {
                taskType = choice
            }
            assumptions = response.assumptions ?? []
            leakageWarnings = response.leakage_warnings ?? []
            statusMessage = response.reasoning_summary ?? "SQL generated"
            selectedStep = .sql
        } else {
            statusMessage = response.message ?? "SQL generation failed"
            technicalDetails = response.technical_details ?? ""
        }
    }

    func validateSQL() async {
        guard let schema else {
            statusMessage = "Inspect schema before validating SQL."
            return
        }
        await run("Validating SQL safety...") {
            let response: SQLValidateResponse = try await client.post(
                "sql/validate",
                payload: [
                    "sql": .string(generatedSQL),
                    "dialect": .string("postgres"),
                    "schema": schema,
                    "max_rows": .number(100000),
                ]
            )
            sqlValidation = response
            warnings = response.warnings ?? []
            if response.is_safe == true {
                generatedSQL = response.normalized_sql ?? generatedSQL
                statusMessage = "SQL is safe using \(response.validator ?? "validator")"
                selectedStep = .preview
            } else {
                statusMessage = "SQL was rejected"
                technicalDetails = (response.errors ?? []).joined(separator: "\n")
            }
        }
    }

    func previewTable() async {
        guard let schema else {
            statusMessage = "Inspect schema before previewing."
            return
        }
        await run("Executing preview locally in a read-only transaction...") {
            let response: PreviewResponse = try await client.post(
                "sql/execute-preview",
                payload: [
                    "connection_string": .string(connectionString),
                    "sql": .string(generatedSQL),
                    "dialect": .string("postgres"),
                    "schema": schema,
                    "preview_rows": .number(20),
                    "max_rows": .number(100000),
                ]
            )
            if response.ok {
                previewColumns = response.columns ?? []
                previewRows = response.preview ?? []
                previewRowCount = response.row_count
                warnings = response.warnings ?? []
                generatedSQL = response.normalized_sql ?? generatedSQL
                statusMessage = "Preview loaded locally"
                selectedStep = .preview
            } else {
                statusMessage = response.message ?? "Preview failed"
                technicalDetails = response.technical_details ?? (response.errors ?? []).joined(separator: "\n")
            }
        }
    }

    func runPrediction() async {
        guard let schema else {
            statusMessage = "Inspect schema before running predictions."
            return
        }
        let trimmedTargetColumn = targetColumn.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTargetColumn.isEmpty else {
            statusMessage = "Choose the target column before running prediction."
            technicalDetails = "The target column is the value the model learns to predict. After executing preview, choose one of the returned columns in the Target Column control."
            return
        }
        await run("Running local tabular prediction...") {
            let ids = entityIDColumns
                .split(separator: ",")
                .map { JSONValue.string($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            let response: MLRunResponse = try await client.post(
                "ml/run-tabpfn",
                payload: [
                    "connection_string": .string(connectionString),
                    "sql": .string(generatedSQL),
                    "schema": schema,
                    "target_column": .string(trimmedTargetColumn),
                    "entity_id_columns": .array(ids),
                    "task_type": .string(taskType.rawValue),
                    "mode": .string("auto"),
                    "max_rows": .number(100000),
                    "test_size": .number(0.2),
                    "random_seed": .number(42),
                ]
            )
            if response.ok {
                predictions = response.predictions ?? []
                metrics = response.metrics
                dataset = response.dataset
                modelInfo = response.model
                warnings = response.warnings ?? []
                statusMessage = "Prediction run complete"
                saveAuditRecord()
                selectedStep = .predict
            } else {
                statusMessage = response.message ?? "Prediction failed"
                technicalDetails = response.technical_details ?? (response.errors ?? []).joined(separator: "\n")
            }
        }
    }

    func chooseTargetColumn(_ column: ColumnInfo) {
        targetColumn = column.name
        if taskType == .unknown {
            taskType = likelyClassificationColumn(column) ? .classification : .regression
        }
        statusMessage = "Target column selected: \(column.name)"
        technicalDetails = ""
    }

    func addEntityIDColumn(_ column: ColumnInfo) {
        var columns = entityIDColumns
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !columns.contains(column.name) else { return }
        columns.append(column.name)
        entityIDColumns = columns.joined(separator: ", ")
        statusMessage = "Entity ID column added: \(column.name)"
        technicalDetails = ""
    }

    func copySQL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(generatedSQL, forType: .string)
        statusMessage = "SQL copied"
    }

    func exportCSV() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "mokuml-predictions.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try csvString().write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Predictions exported"
        } catch {
            statusMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    func exportMarkdownReport() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.text]
        panel.nameFieldStringValue = "mokuml-run-report.md"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try markdownReport().write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Run report exported"
        } catch {
            statusMessage = "Report export failed: \(error.localizedDescription)"
        }
    }

    private func providerPayload() -> [String: JSONValue] {
        switch llmProvider {
        case .localOllama:
            return [
                "type": .string(llmProvider.rawValue),
                "endpoint": .string(localEndpoint),
                "model": .string(localModel),
            ]
        case .offlineTemplate:
            return ["type": .string(llmProvider.rawValue)]
        case .openAICompatible:
            return [
                "type": .string(llmProvider.rawValue),
                "base_url": .string(apiBaseURL),
                "model": .string(apiModel),
                "api_key": .string(apiKey),
            ]
        }
    }

    private func fetchOllamaStatus() async throws -> OllamaStatusResponse {
        let endpoint = normalizedOllamaEndpoint()
        let url = endpoint.appendingPathComponent("api/tags")
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            return OllamaStatusResponse(
                ok: false,
                server_running: false,
                model_available: false,
                message: "Ollama returned HTTP \(http.statusCode).",
                installed_models: nil,
                technical_details: String(data: data, encoding: .utf8),
                suggested_fix: "Open Ollama, then install the configured model."
            )
        }
        let tags = try JSONDecoder().decode(OllamaTagsResponse.self, from: data)
        let installedModels = Array(Set(tags.models.flatMap { [$0.name, $0.model] }
            .filter { !$0.isEmpty }))
            .sorted()
        installedOllamaModels = installedModels
        if !installedModels.contains(where: { Self.ollamaModelName($0, matches: localModel) }),
           (localModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || localModel == "sqlcoder"),
           let firstModel = installedModels.first {
            localModel = firstModel
        }
        let modelAvailable = installedModels.contains { name in
            Self.ollamaModelName(name, matches: localModel)
        }
        let message: String
        if modelAvailable {
            message = "Ollama is ready with \(localModel)."
        } else if installedModels.isEmpty {
            message = "Ollama is running, but no models are installed."
        } else {
            message = "Ollama is running, but \(localModel) is not installed."
        }
        return OllamaStatusResponse(
            ok: modelAvailable,
            server_running: true,
            model_available: modelAvailable,
            message: message,
            installed_models: installedModels,
            technical_details: nil,
            suggested_fix: modelAvailable ? nil : "Run: ollama pull \(localModel)"
        )
    }

    private func callLocalOllama(messages: [LLMChatMessage]) async throws -> String {
        let url = normalizedOllamaEndpoint().appendingPathComponent("api/chat")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = OllamaChatRequest(
            model: localModel,
            messages: messages,
            stream: false,
            format: "json",
            options: ["temperature": 0.0, "top_p": 1.0]
        )
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw WorkerClientError.http(status: http.statusCode, body: body)
        }
        let decoded = try JSONDecoder().decode(OllamaChatResponse.self, from: data)
        return decoded.message.content
    }

    private func normalizedOllamaEndpoint() -> URL {
        let trimmed = localEndpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        return URL(string: trimmed.hasSuffix("/") ? String(trimmed.dropLast()) : trimmed)
            ?? URL(string: "http://127.0.0.1:11434")!
    }

    private func ollamaSetupDetails(_ status: OllamaStatusResponse?) -> String {
        guard let status else {
            return "Start Ollama, then run: ollama pull \(localModel)"
        }
        var lines: [String] = []
        if let suggestedFix = status.suggested_fix, !suggestedFix.isEmpty {
            lines.append(suggestedFix)
        } else if status.ok != true {
            lines.append("Start Ollama, then run: ollama pull \(localModel)")
        }
        if let technicalDetails = status.technical_details, !technicalDetails.isEmpty {
            lines.append("Details: \(technicalDetails)")
        }
        if let installedModels = status.installed_models, !installedModels.isEmpty {
            lines.append("Installed models: \(installedModels.joined(separator: ", "))")
        }
        return lines.joined(separator: "\n")
    }

    private func saveAuditRecord() {
        let formatter = ISO8601DateFormatter()
        let record = RunAuditRecord(
            timestamp: formatter.string(from: Date()),
            databaseType: connectionResult?.database_type ?? "postgres",
            maskedConnection: connectionResult?.masked_connection,
            userPrompt: predictionPrompt,
            schemaHash: RunHistoryService.schemaHash(schema),
            selectedTables: schemaTables.map(\.qualifiedName),
            llmProviderType: llmProvider.rawValue,
            llmModelName: llmProvider == .openAICompatible ? apiModel : localModel,
            generatedSQL: generatedSQL,
            sqlValidator: sqlValidation?.validator,
            sqlWasSafe: sqlValidation?.is_safe == true,
            targetColumn: targetColumn,
            taskType: taskType.rawValue,
            dataset: dataset,
            model: modelInfo,
            metrics: metrics,
            warnings: warnings
        )
        do {
            _ = try RunHistoryService.save(record)
        } catch {
            warnings.append("Run finished, but audit record could not be saved: \(error.localizedDescription)")
        }
    }

    private func run(_ message: String, operation: () async throws -> Void) async {
        isWorking = true
        statusMessage = message
        technicalDetails = ""
        defer { isWorking = false }
        do {
            try await operation()
        } catch {
            statusMessage = error.localizedDescription
            technicalDetails = String(describing: error)
        }
    }

    private func csvString() -> String {
        let rows = predictions.compactMap(\.objectValue)
        let headers = Array(Set(rows.flatMap { $0.keys })).sorted()
        let body = rows.map { row in
            headers.map { escapeCSV(row[$0]?.displayString ?? "") }.joined(separator: ",")
        }
        return ([headers.joined(separator: ",")] + body).joined(separator: "\n")
    }

    private func markdownReport() -> String {
        """
        # MokuML Run Report

        ## Prompt
        \(predictionPrompt)

        ## Generated SQL
        ```sql
        \(generatedSQL)
        ```

        ## Target
        - Target column: \(targetColumn)
        - Entity ID columns: \(entityIDColumns)
        - Task type: \(taskType.rawValue)

        ## Metrics
        \(metrics?.displayString ?? "No metrics")

        ## Dataset
        \(dataset?.displayString ?? "No dataset summary")

        ## Warnings
        \(warnings.isEmpty ? "None" : warnings.map { "- \($0)" }.joined(separator: "\n"))
        """
    }

    private func escapeCSV(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") {
            return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        return value
    }

    private func likelyClassificationColumn(_ column: ColumnInfo) -> Bool {
        let name = column.name.lowercased()
        let dtype = column.dtype.lowercased()
        return dtype.contains("bool")
            || name.hasPrefix("is_")
            || name.hasPrefix("has_")
            || name.contains("churn")
            || name.contains("late")
            || name.contains("paid")
            || name.contains("status")
    }

    private static func ollamaModelName(_ installedName: String, matches selectedName: String) -> Bool {
        let selected = selectedName.trimmingCharacters(in: .whitespacesAndNewlines)
        return installedName == selected || installedName.split(separator: ":").first.map(String.init) == selected
    }
}

private struct OllamaTagsResponse: Decodable {
    var models: [OllamaTagModel]
}

private struct OllamaTagModel: Decodable {
    var name: String
    var model: String
}

private struct OllamaChatRequest: Encodable {
    var model: String
    var messages: [LLMChatMessage]
    var stream: Bool
    var format: String
    var options: [String: Double]
}

private struct OllamaChatResponse: Decodable {
    var message: OllamaChatMessage
}

private struct OllamaChatMessage: Decodable {
    var content: String
}
