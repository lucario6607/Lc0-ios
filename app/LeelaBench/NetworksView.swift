import SwiftUI
import UniformTypeIdentifiers

struct NetworksView: View {
    @EnvironmentObject private var nets: NetStore
    @State private var showImporter = false
    @State private var downloadURL = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(nets.nets, id: \.self) { url in
                        NavigationLink {
                            NetworkDetailView(url: url)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(url.lastPathComponent).lineLimit(2)
                                Text(fileSize(url)).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { offsets in
                        offsets.map { nets.nets[$0] }.forEach(nets.delete)
                    }
                    if nets.nets.isEmpty {
                        Text("No networks yet.").foregroundStyle(.secondary)
                    }
                } header: {
                    Text("On this device")
                } footer: {
                    Text("You can also copy .pb.gz files into LeelaBench's folder with the Files app.")
                }

                Section("Add a network") {
                    Button {
                        showImporter = true
                    } label: {
                        Label("Import from Files…", systemImage: "folder")
                    }
                }

                Section {
                    TextField("https://storage.lczero.org/files/networks/…", text: $downloadURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    if let progress = nets.downloadProgress {
                        ProgressView(value: progress) {
                            Text("Downloading… \(Int(progress * 100))%")
                        }
                    } else {
                        Button("Download") {
                            guard let url = URL(string: downloadURL.trimmingCharacters(in: .whitespaces)) else { return }
                            Task {
                                await nets.download(url)
                                downloadURL = ""
                            }
                        }
                        .disabled(URL(string: downloadURL)?.scheme?.hasPrefix("http") != true)
                    }
                } header: {
                    Text("Download from URL")
                } footer: {
                    Link("Browse recommended networks on lczero.org",
                         destination: URL(string: "https://lczero.org/play/networks/bestnets/")!)
                }
            }
            .navigationTitle("Networks")
            .refreshable { nets.refresh() }
            .onAppear { nets.refresh() }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item],
                          allowsMultipleSelection: true) { result in
                if case .success(let urls) = result {
                    urls.forEach(nets.importFile)
                }
            }
            .alert("Error", isPresented: Binding(
                get: { nets.lastError != nil },
                set: { if !$0 { nets.lastError = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(nets.lastError ?? "")
            }
        }
    }

    private func fileSize(_ url: URL) -> String {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }
}

/// File details plus lc0's `describenet` output.
struct NetworkDetailView: View {
    @EnvironmentObject private var runner: EngineRunner
    @EnvironmentObject private var nets: NetStore
    @Environment(\.dismiss) private var dismiss
    let url: URL

    @State private var description: String?
    @State private var loading = false
    @State private var confirmDelete = false

    private var attributes: URLResourceValues? {
        try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
    }

    var body: some View {
        List {
            Section("File") {
                LabeledContent("Name", value: url.lastPathComponent)
                if let size = attributes?.fileSize {
                    LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
                }
                if let date = attributes?.contentModificationDate {
                    LabeledContent("Added", value: date.formatted(date: .abbreviated, time: .shortened))
                }
            }

            Section("Network info") {
                if let description {
                    Text(description)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                } else if loading {
                    HStack { ProgressView(); Text("Reading network…").foregroundStyle(.secondary) }
                } else {
                    Button("Show architecture and training info") { describe() }
                        .disabled(runner.isRunning)
                    if runner.isRunning {
                        Text("Available when the current benchmark finishes.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }

            Section {
                Button("Delete network", role: .destructive) { confirmDelete = true }
            }
        }
        .navigationTitle(url.deletingPathExtension().deletingPathExtension().lastPathComponent)
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Delete \(url.lastPathComponent)?", isPresented: $confirmDelete,
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                nets.delete(url)
                dismiss()
            }
        }
    }

    private func describe() {
        loading = true
        runner.run(["describenet", "--weights=\(url.path)"]) { outcome in
            loading = false
            // Drop the lc0 banner and our own bookkeeping lines.
            let lines = outcome.output.split(separator: "\n", omittingEmptySubsequences: false)
                .drop { !$0.contains(":") }
            description = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
}
