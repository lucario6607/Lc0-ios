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
                        VStack(alignment: .leading, spacing: 2) {
                            Text(url.lastPathComponent).lineLimit(2)
                            Text(fileSize(url)).font(.caption).foregroundStyle(.secondary)
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
                    Text("You can also copy .pb.gz files into LeelaBench's folder with the Files app, or with Finder / iTunes file sharing from a computer.")
                }

                Section {
                    Button {
                        showImporter = true
                    } label: {
                        Label("Import from Files…", systemImage: "folder")
                    }
                } header: {
                    Text("Add a network")
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
                            Task { await nets.download(url) }
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
