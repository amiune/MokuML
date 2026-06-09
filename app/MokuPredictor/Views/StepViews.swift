import SwiftUI

struct ConnectStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Postgres Connection", systemImage: "cylinder.split.1x2")
            VStack(alignment: .leading, spacing: 10) {
                EditableLineField(
                    text: $model.connectionString,
                    placeholder: "postgresql://user:password@host:5432/database"
                )
                .frame(height: 28)
                Toggle("Save connection in Keychain", isOn: $model.saveConnection)
                HStack {
                    Button {
                        Task { await model.testConnection() }
                    } label: {
                        Label("Test Connection", systemImage: "bolt.horizontal")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.connectionString.isEmpty || model.isWorking)

                    if let result = model.connectionResult {
                        StatusBadge(text: result.ok ? "Connected" : "Failed", isPositive: result.ok)
                        if let masked = result.masked_connection {
                            Text(masked)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                if let result = model.connectionResult {
                    ConnectionFeedbackView(result: result)
                } else if !model.technicalDetails.isEmpty {
                    InlineDiagnosticView(
                        title: "Connection Test Error",
                        message: model.statusMessage,
                        technicalDetails: model.technicalDetails,
                        isPositive: false
                    )
                }
            }

            WorkerRequirementsView()
        }
    }
}

struct DescribeStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Prediction Request", systemImage: "text.bubble")
            Picker("Example", selection: $model.selectedExample) {
                ForEach(model.examples, id: \.self) { example in
                    Text(example).tag(example)
                }
            }
            HStack {
                Button("Use Example") {
                    model.useSelectedExample()
                }
                Spacer()
            }
            EditableTextArea(text: $model.predictionPrompt)
                .frame(minHeight: 140)

            SectionTitle("LLM Provider", systemImage: "cpu")
            Picker("Provider", selection: $model.llmProvider) {
                ForEach(LLMProviderChoice.allCases) { provider in
                    Text(provider.title).tag(provider)
                }
            }
            .pickerStyle(.segmented)

            providerSettings

            HStack {
                EditableLineField(text: $model.includeSchemas, placeholder: "public")
                    .frame(width: 260, height: 28)
                Button {
                    Task { await model.introspectSchema() }
                } label: {
                    Label("Inspect Schema", systemImage: "tablecells")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.connectionString.isEmpty || model.isWorking)
            }
        }
    }

    @ViewBuilder
    private var providerSettings: some View {
        switch model.llmProvider {
        case .localOllama:
            HStack(alignment: .center, spacing: 10) {
                EditableLineField(text: $model.localEndpoint, placeholder: "http://127.0.0.1:11434")
                    .frame(height: 28)
                if !model.installedOllamaModels.isEmpty {
                    Picker("Model", selection: Binding(
                        get: { model.localModel },
                        set: { model.selectLocalOllamaModel($0) }
                    )) {
                        ForEach(model.installedOllamaModels, id: \.self) { installedModel in
                            Text(installedModel).tag(installedModel)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 240)
                }
                EditableLineField(text: $model.localModel, placeholder: "Ollama model")
                    .frame(width: 220, height: 28)
            }
            OllamaSetupView(model: model)
        case .offlineTemplate:
            WarningList(["Offline template mode drafts a local placeholder query and cannot infer a target column."])
        case .openAICompatible:
            WarningList(["API mode sends schema metadata and the prediction prompt to the configured provider. Raw rows are still never sent."])
            HStack {
                EditableLineField(text: $model.apiBaseURL, placeholder: "https://api.openai.com/v1")
                    .frame(height: 28)
                EditableLineField(text: $model.apiModel, placeholder: "gpt-4.1-mini")
                    .frame(width: 220, height: 28)
            }
            EditableLineField(text: $model.apiKey, placeholder: "API key")
                .frame(height: 28)
        }
    }
}

private struct OllamaSetupView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: statusIcon)
                    .foregroundStyle(statusColor)
                    .frame(width: 20)
                VStack(alignment: .leading, spacing: 4) {
                    Text(statusTitle)
                        .font(.headline)
                    Text(statusMessage)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
            }

            HStack {
                Button {
                    Task { await model.checkOllamaStatus() }
                } label: {
                    Label("Check Ollama", systemImage: "arrow.clockwise")
                }
                Button {
                    model.openOllamaApp()
                } label: {
                    Label("Open Ollama", systemImage: "play.fill")
                }
                Button {
                    model.openOllamaDownload()
                } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
                Button {
                    model.useOfflineTemplateProvider()
                } label: {
                    Label("Use Offline", systemImage: "doc.text")
                }
                Spacer()
            }
            .disabled(model.isWorking)

            if let detail = setupDetail, !detail.isEmpty {
                Text(detail)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(statusColor.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var statusTitle: String {
        guard let status = model.ollamaStatus else { return "Ollama status unknown" }
        if status.ok { return "Ollama ready" }
        if status.server_running == true { return "Model missing" }
        return "Ollama not running"
    }

    private var statusMessage: String {
        model.ollamaStatus?.message
            ?? "Check Ollama before generating SQL with the local provider."
    }

    private var setupDetail: String? {
        guard let status = model.ollamaStatus else {
            return "If Ollama is installed, open it and install the configured model: ollama pull \(model.localModel)"
        }
        if status.ok { return status.installed_models?.isEmpty == false ? "Installed models: \(status.installed_models!.joined(separator: ", "))" : nil }
        return status.suggested_fix ?? "Install the configured model: ollama pull \(model.localModel)"
    }

    private var statusIcon: String {
        guard let status = model.ollamaStatus else { return "questionmark.circle.fill" }
        return status.ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
    }

    private var statusColor: Color {
        guard let status = model.ollamaStatus else { return .secondary }
        return status.ok ? .green : .orange
    }
}

struct SchemaStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Schema Metadata", systemImage: "tablecells")
            PrivacyNotice()

            HStack {
                Button {
                    Task { await model.introspectSchema() }
                } label: {
                    Label("Refresh Schema", systemImage: "arrow.clockwise")
                }
                Button {
                    Task { await model.generateSQL() }
                } label: {
                    Label("Generate SQL", systemImage: "wand.and.stars")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.schema == nil || model.predictionPrompt.isEmpty)
                Spacer()
                Text("\(model.schemaTables.count) tables")
                    .foregroundStyle(.secondary)
            }

            if model.schemaTables.isEmpty {
                EmptyState("No schema loaded yet", systemImage: "tray")
            } else {
                Table(model.schemaTables) {
                    TableColumn("Table", value: \.qualifiedName)
                    TableColumn("Columns") { table in
                        Text("\(table.columnCount)")
                    }
                    TableColumn("Foreign Keys") { table in
                        Text("\(table.foreignKeyCount)")
                    }
                    TableColumn("Indexes") { table in
                        Text("\(table.indexCount)")
                    }
                }
                .frame(minHeight: 320)
            }
        }
    }
}

struct SQLStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Generated SQL", systemImage: "chevron.left.forwardslash.chevron.right")
            HStack {
                Button {
                    Task { await model.generateSQL() }
                } label: {
                    Label("Generate", systemImage: "wand.and.stars")
                }
                Button {
                    Task { await model.validateSQL() }
                } label: {
                    Label("Validate", systemImage: "checkmark.shield")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.generatedSQL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    model.copySQL()
                } label: {
                    Label("Copy SQL", systemImage: "doc.on.doc")
                }
                .disabled(model.generatedSQL.isEmpty)
                Spacer()
                if let validation = model.sqlValidation {
                    StatusBadge(text: validation.is_safe == true ? "Safe" : "Rejected", isPositive: validation.is_safe == true)
                }
            }

            CodeEditor(text: $model.generatedSQL)
                .frame(minHeight: 260)

            if !model.assumptions.isEmpty {
                DetailList(title: "Assumptions", items: model.assumptions)
            }
            if !model.leakageWarnings.isEmpty {
                WarningList(model.leakageWarnings)
            }
            if let errors = model.sqlValidation?.errors, !errors.isEmpty {
                WarningList(errors)
            }
            if !model.warnings.isEmpty {
                WarningList(model.warnings)
            }
        }
    }
}

struct PreviewStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("ML Table Preview", systemImage: "doc.text.magnifyingglass")
            HStack {
                Button {
                    Task { await model.previewTable() }
                } label: {
                    Label("Execute Preview", systemImage: "play.rectangle")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.connectionString.isEmpty || model.generatedSQL.isEmpty)
                Spacer()
                if let rowCount = model.previewRowCount {
                    Text("\(rowCount) local rows loaded")
                        .foregroundStyle(.secondary)
                }
            }

            HStack(alignment: .top) {
                VStack(alignment: .leading) {
                    Text("Target Column")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        EditableLineField(text: $model.targetColumn, placeholder: "churned_next_month")
                            .frame(height: 28)
                        if !model.previewColumns.isEmpty {
                            Menu {
                                ForEach(model.previewColumns) { column in
                                    Button(column.name) {
                                        model.chooseTargetColumn(column)
                                    }
                                }
                            } label: {
                                Label("Choose", systemImage: "list.bullet")
                            }
                            .menuStyle(.borderlessButton)
                        }
                    }
                }
                VStack(alignment: .leading) {
                    Text("Entity ID Columns")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        EditableLineField(text: $model.entityIDColumns, placeholder: "customer_id, invoice_id")
                            .frame(height: 28)
                        if !model.previewColumns.isEmpty {
                            Menu {
                                ForEach(model.previewColumns) { column in
                                    Button(column.name) {
                                        model.addEntityIDColumn(column)
                                    }
                                }
                            } label: {
                                Label("Add", systemImage: "plus")
                            }
                            .menuStyle(.borderlessButton)
                        }
                    }
                }
                VStack(alignment: .leading) {
                    Text("Task")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("Task", selection: $model.taskType) {
                        ForEach(TaskTypeChoice.allCases) { task in
                            Text(task.title).tag(task)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .frame(width: 180)
            }

            JSONTable(rows: model.previewRows, emptyTitle: "No preview rows")
                .frame(minHeight: 260)

            HStack {
                Button {
                    Task { await model.runPrediction() }
                } label: {
                    Label("Run Prediction", systemImage: "chart.line.uptrend.xyaxis")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.generatedSQL.isEmpty || model.isWorking)
                Spacer()
            }

            if model.targetColumn.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                WarningList(["Choose the target column from the preview before running prediction."])
            }

            if !model.warnings.isEmpty {
                WarningList(model.warnings)
            }
        }
    }
}

struct PredictStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Predictions", systemImage: "chart.line.uptrend.xyaxis")
            HStack {
                Button {
                    Task { await model.runPrediction() }
                } label: {
                    Label("Run Again", systemImage: "arrow.clockwise")
                }
                .disabled(model.targetColumn.isEmpty)
                Spacer()
                Text("\(model.predictions.count) predictions")
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .top, spacing: 16) {
                JSONSummary(title: "Model", value: model.modelInfo)
                JSONSummary(title: "Dataset", value: model.dataset)
                JSONSummary(title: "Metrics", value: model.metrics)
            }

            JSONTable(rows: model.predictions, emptyTitle: "No predictions yet")
                .frame(minHeight: 320)

            if !model.warnings.isEmpty {
                WarningList(model.warnings)
            }
        }
    }
}

struct ExportStepView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionTitle("Export", systemImage: "square.and.arrow.down")
            HStack {
                Button {
                    model.exportCSV()
                } label: {
                    Label("Predictions CSV", systemImage: "tablecells")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.predictions.isEmpty)
                Button {
                    model.exportMarkdownReport()
                } label: {
                    Label("Run Report", systemImage: "doc.text")
                }
                Button {
                    model.copySQL()
                } label: {
                    Label("Copy SQL", systemImage: "doc.on.doc")
                }
                .disabled(model.generatedSQL.isEmpty)
            }
            JSONSummary(title: "Last Run", value: model.dataset)
            JSONTable(rows: model.predictions, emptyTitle: "No predictions to export")
                .frame(minHeight: 320)
        }
    }
}
