import AppKit
import SwiftUI

struct MainView: View {
    @StateObject private var worker = WorkerProcess()
    @StateObject private var model = AppViewModel()

    var body: some View {
        NavigationSplitView {
            SidebarView(selectedStep: $model.selectedStep, worker: worker)
                .navigationSplitViewColumnWidth(min: 240, ideal: 260)
        } detail: {
            VStack(spacing: 0) {
                HeaderView(worker: worker, model: model)
                Divider()
                ScrollView {
                    StepContentView(model: model)
                        .padding(24)
                        .frame(maxWidth: 1120, alignment: .topLeading)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                Divider()
                FooterView(model: model)
            }
        }
        .task {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
            await worker.start()
        }
    }
}

private struct SidebarView: View {
    @Binding var selectedStep: WorkflowStep
    @ObservedObject var worker: WorkerProcess

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("MokuML")
                    .font(.title2.weight(.semibold))
                Text("Local SQL predictions")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.top, 18)

            VStack(spacing: 4) {
                ForEach(WorkflowStep.allCases) { step in
                    Button {
                        selectedStep = step
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: step.symbol)
                                .frame(width: 22)
                            Text(step.title)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 9)
                        .background(selectedStep == step ? Color.accentColor.opacity(0.14) : Color.clear)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)

            Spacer()

            VStack(alignment: .leading, spacing: 8) {
                Label(worker.status, systemImage: worker.isRunning ? "checkmark.circle" : "xmark.circle")
                    .font(.caption)
                    .foregroundStyle(worker.isRunning ? .green : .secondary)
                if let warnings = worker.health?.warnings, !warnings.isEmpty {
                    Text("\(warnings.count) worker warning\(warnings.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Text("Port \(WorkerEndpoint.port)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if let python = worker.health?.python_executable {
                    Text((python as NSString).lastPathComponent)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(16)
        }
    }
}

private struct HeaderView: View {
    @ObservedObject var worker: WorkerProcess
    @ObservedObject var model: AppViewModel

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.selectedStep.title)
                    .font(.title3.weight(.semibold))
                Text(model.statusMessage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer()
            if model.isWorking {
                ProgressView()
                    .controlSize(.small)
            }
            Button {
                Task { await worker.refreshHealth() }
            } label: {
                Label("Health", systemImage: "heart.text.square")
            }
            Button {
                Task { await worker.start() }
            } label: {
                Label("Start Worker", systemImage: "play.fill")
            }
            Button {
                worker.stop()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }
}

private struct StepContentView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        switch model.selectedStep {
        case .connect:
            ConnectStepView(model: model)
        case .describe:
            DescribeStepView(model: model)
        case .schema:
            SchemaStepView(model: model)
        case .sql:
            SQLStepView(model: model)
        case .preview:
            PreviewStepView(model: model)
        case .predict:
            PredictStepView(model: model)
        case .export:
            ExportStepView(model: model)
        }
    }
}

private struct FooterView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        HStack {
            Label("LLM boundary: prompt + schema only", systemImage: "lock.shield")
                .foregroundStyle(.secondary)
            Spacer()
            if !model.technicalDetails.isEmpty {
                DisclosureGroup("Technical details") {
                    Text(model.technicalDetails)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: 680, alignment: .leading)
                }
                .frame(maxWidth: 760)
            }
        }
        .font(.caption)
        .padding(.horizontal, 24)
        .padding(.vertical, 10)
    }
}
