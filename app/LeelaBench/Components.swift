import SwiftUI

/// A labelled integer field.
struct NumberRow: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>

    var body: some View {
        LabeledContent(title) {
            TextField(title, value: Binding(
                get: { value },
                set: { value = min(max($0, range.lowerBound), range.upperBound) }
            ), format: .number.grouping(.never))
            .keyboardType(.numbersAndPunctuation)
            .multilineTextAlignment(.trailing)
            .frame(maxWidth: 140)
        }
    }
}

/// A big number with a caption, for result summaries.
struct StatTile: View {
    let title: String
    let value: String
    var unit: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value).font(.title3.bold()).monospacedDigit()
                if let unit { Text(unit).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Scrollable, selectable console output.
struct LogView: View {
    let title: String
    let text: String
    /// Keep scrolled to the end as `text` grows.
    var follow = false

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(text.isEmpty ? "No output yet." : text)
                    .font(.caption.monospaced())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding()
                Color.clear.frame(height: 1).id("end")
            }
            .onAppear { proxy.scrollTo("end", anchor: .bottom) }
            .onChange(of: text) { _ in
                if follow { proxy.scrollTo("end", anchor: .bottom) }
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                Button {
                    UIPasteboard.general.string = text
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                ShareLink(item: text)
            }
        }
    }
}

struct DeviceInfoView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Device") {
                    LabeledContent("Model", value: DeviceInfo.modelIdentifier)
                    LabeledContent("OS", value: DeviceInfo.osVersion)
                    LabeledContent("GPU", value: DeviceInfo.gpuName)
                    LabeledContent("CPU cores", value: "\(DeviceInfo.cpuCores)")
                    LabeledContent("RAM", value: DeviceInfo.physicalMemory)
                    LabeledContent("Available to app", value: DeviceInfo.availableMemory)
                }
                Section {
                    LabeledContent("Thermal state", value: DeviceInfo.thermalState)
                    LabeledContent("Low Power Mode", value: DeviceInfo.lowPowerMode ? "on" : "off")
                } header: {
                    Text("Performance")
                } footer: {
                    Text("For comparable numbers, start runs at \"nominal\" with Low Power Mode off. \"serious\" and \"critical\" mean iOS is slowing the chip down to cool it.")
                }
                Section("Build") {
                    LabeledContent("lc0", value: BuildInfo.lc0Version)
                    LabeledContent("ONNX Runtime / Core ML", value: BuildInfo.hasOnnx ? "included" : "not included")
                    LabeledContent("App", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?")
                }
            }
            .navigationTitle("Device")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button("Done") { dismiss() }
            }
        }
    }
}

/// Colour for a thermal state string from `DeviceInfo.thermalState`.
func thermalColor(_ state: String?) -> Color {
    switch state {
    case "fair": return .yellow
    case "serious": return .orange
    case "critical": return .red
    default: return .secondary
    }
}
