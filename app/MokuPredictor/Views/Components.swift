import AppKit
import SwiftUI

struct SettingsView: View {
    @State private var message = ""

    var body: some View {
        Form {
            Section("Worker") {
                Text("Default port: \(WorkerEndpoint.port)")
                Text("Project root can be overridden with MOKU_PROJECT_ROOT.")
            }
            Section("Storage") {
                Text("Run audits are saved in Application Support without raw rows or credentials.")
                Button {
                    do {
                        try KeychainService.deleteAll()
                        message = "Saved credentials forgotten."
                    } catch {
                        message = error.localizedDescription
                    }
                } label: {
                    Label("Forget Saved Connections", systemImage: "trash")
                }
                if !message.isEmpty {
                    Text(message)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}

struct SectionTitle: View {
    var title: String
    var systemImage: String

    init(_ title: String, systemImage: String) {
        self.title = title
        self.systemImage = systemImage
    }

    var body: some View {
        Label(title, systemImage: systemImage)
            .font(.title3.weight(.semibold))
    }
}

struct StatusBadge: View {
    var text: String
    var isPositive: Bool

    var body: some View {
        Label(text, systemImage: isPositive ? "checkmark.circle.fill" : "xmark.circle.fill")
            .font(.caption.weight(.semibold))
            .foregroundStyle(isPositive ? .green : .red)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background((isPositive ? Color.green : Color.red).opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct PrivacyNotice: View {
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "lock.shield")
                .foregroundStyle(.green)
            VStack(alignment: .leading, spacing: 3) {
                Text("Only schema metadata is sent to the LLM.")
                    .font(.headline)
                Text("Rows, previews, values, passwords, and API keys stay outside the LLM request.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Color.green.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct WorkerRequirementsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionTitle("Worker Environment", systemImage: "terminal")
            Text("Python 3.11+ with FastAPI, SQLAlchemy, psycopg, sqlglot, pandas, scikit-learn, and TabPFN enables the full local workflow.")
                .foregroundStyle(.secondary)
            Text("The built-in fallback server keeps health, validation, and basic ML scaffolding available before setup.")
                .foregroundStyle(.secondary)
        }
    }
}

struct ConnectionFeedbackView: View {
    var result: ConnectionTestResponse

    var body: some View {
        InlineDiagnosticView(
            title: result.ok ? "Connection Successful" : "Connection Failed",
            message: result.message ?? (result.ok ? "Postgres responded successfully." : "The worker could not connect to Postgres."),
            technicalDetails: [result.technical_details, result.suggested_fix.map { "Suggested fix: \($0)" }]
                .compactMap { $0 }
                .joined(separator: "\n\n"),
            isPositive: result.ok
        )
    }
}

struct InlineDiagnosticView: View {
    var title: String
    var message: String
    var technicalDetails: String
    var isPositive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: isPositive ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(isPositive ? .green : .red)
                Text(title)
                    .font(.headline)
            }
            Text(message)
            if !technicalDetails.isEmpty {
                DisclosureGroup("Details and suggested fix") {
                    Text(technicalDetails)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((isPositive ? Color.green : Color.red).opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct WarningList: View {
    var items: [String]

    init(_ items: [String]) {
        self.items = items
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(items, id: \.self) { item in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(item)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct DetailList: View {
    var title: String
    var items: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "checkmark")
                        .foregroundStyle(.secondary)
                    Text(item)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct EmptyState: View {
    var title: String
    var systemImage: String

    init(_ title: String, systemImage: String) {
        self.title = title
        self.systemImage = systemImage
    }

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(title)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
    }
}

struct CodeEditor: View {
    @Binding var text: String

    var body: some View {
        TextField("Generated SQL", text: $text, axis: .vertical)
            .font(.system(.body, design: .monospaced))
            .textFieldStyle(.plain)
            .textSelection(.enabled)
            .lineLimit(8...24)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.secondary.opacity(0.25))
            )
    }
}

struct EditableLineField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    func makeNSView(context: Context) -> NSTextField {
        let field = FocusableTextField(string: text)
        field.placeholderString = placeholder
        field.focusDelegate = context.coordinator
        field.isEditable = true
        field.isSelectable = true
        field.isEnabled = true
        field.isBordered = true
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.drawsBackground = true
        field.delegate = context.coordinator
        field.lineBreakMode = .byTruncatingMiddle
        field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        nsView.placeholderString = placeholder
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        @Binding var text: String

        init(text: Binding<String>) {
            _text = text
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSTextField else { return }
            text = field.stringValue
        }
    }
}

final class FocusableTextField: NSTextField {
    weak var focusDelegate: NSTextFieldDelegate?

    override var acceptsFirstResponder: Bool {
        true
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if currentEditor() == nil {
            selectText(nil)
        }
        super.mouseDown(with: event)
    }

    override func becomeFirstResponder() -> Bool {
        let accepted = super.becomeFirstResponder()
        delegate = focusDelegate
        return accepted
    }
}

struct EditableTextArea: NSViewRepresentable {
    @Binding var text: String
    var isMonospaced = false

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.borderType = .bezelBorder
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = true
        scrollView.backgroundColor = .textBackgroundColor

        let textView = FocusableTextView()
        textView.string = text
        textView.isEditable = true
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = true
        textView.font = isMonospaced
            ? .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
            : .systemFont(ofSize: NSFont.systemFontSize)
        textView.textColor = .textColor
        textView.backgroundColor = .textBackgroundColor
        textView.textContainerInset = NSSize(width: 8, height: 8)
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.delegate = context.coordinator

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? NSTextView else { return }
        textView.font = isMonospaced
            ? .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
            : .systemFont(ofSize: NSFont.systemFontSize)
        if textView.string != text {
            textView.string = text
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        @Binding var text: String

        init(text: Binding<String>) {
            _text = text
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            text = textView.string
        }
    }
}

final class FocusableTextView: NSTextView {
    override var acceptsFirstResponder: Bool {
        true
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.mouseDown(with: event)
    }
}

struct JSONSummary: View {
    var title: String
    var value: JSONValue?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text(value?.displayString.isEmpty == false ? value?.displayString ?? "" : "No data")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(8)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct JSONTable: View {
    var rows: [JSONValue]
    var emptyTitle: String

    var body: some View {
        let objects = rows.compactMap(\.objectValue)
        let columns = Array(Set(objects.flatMap { $0.keys })).sorted()

        if objects.isEmpty || columns.isEmpty {
            EmptyState(emptyTitle, systemImage: "tablecells")
        } else {
            ScrollView([.horizontal, .vertical]) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                    GridRow {
                        ForEach(columns, id: \.self) { column in
                            Text(column)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .frame(minWidth: 120, alignment: .leading)
                        }
                    }
                    ForEach(Array(objects.prefix(100).enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(columns, id: \.self) { column in
                                Text(row[column]?.displayString ?? "")
                                    .lineLimit(2)
                                    .truncationMode(.tail)
                                    .frame(minWidth: 120, maxWidth: 240, alignment: .leading)
                            }
                        }
                    }
                }
                .padding(12)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.separator))
        }
    }
}
