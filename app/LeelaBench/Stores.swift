import Foundation
import Metal
import UIKit

// MARK: - Results on disk

@MainActor
final class ResultsStore: ObservableObject {
    @Published private(set) var results: [BenchResult] = []

    private let url = FileLocations.documents.appendingPathComponent("results.json")

    init() {
        if let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode([BenchResult].self, from: data) {
            results = decoded
        }
    }

    func result(id: UUID) -> BenchResult? {
        results.first { $0.id == id }
    }

    func add(_ result: BenchResult) {
        results.insert(result, at: 0)
        save()
    }

    func delete(ids: Set<UUID>) {
        results.removeAll { ids.contains($0.id) }
        save()
    }

    func deleteAll() {
        results.removeAll()
        save()
    }

    private func save() {
        if let data = try? JSONEncoder().encode(results) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// One row per data point (batch size or searched position), for spreadsheets.
    static func csv(for results: [BenchResult]) -> String {
        var csv = "date,device,mode,network,backend,backend_opts,threads,batch,position,nps,mean_ms,thermal_start,thermal_end\n"
        let formatter = ISO8601DateFormatter()
        func q(_ s: String) -> String { "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        for r in results {
            let prefix = [formatter.string(from: r.date), q(r.device), r.mode.rawValue, q(r.network),
                          q(r.backend), q(r.backendOpts), String(r.threads)].joined(separator: ",")
            let suffix = [r.thermalStart ?? "", r.thermalEnd ?? ""].joined(separator: ",")
            if !r.points.isEmpty {
                for p in r.points {
                    csv += "\(prefix),\(p.batch),,\(Int(p.nps)),\(p.meanMs.map { String($0) } ?? ""),\(suffix)\n"
                }
            } else if !r.positionResults.isEmpty {
                for s in r.positionResults { csv += "\(prefix),,\(s.position),\(s.nps),,\(suffix)\n" }
            } else {
                csv += "\(prefix),,,\(r.searchNps.map(String.init) ?? ""),,\(suffix)\n"
            }
        }
        return csv
    }

    func exportCSV(_ subset: [BenchResult]? = nil, name: String = "leelabench-results") -> URL {
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).csv")
        try? Self.csv(for: subset ?? results).write(to: out, atomically: true, encoding: .utf8)
        return out
    }
}

// MARK: - Networks on disk

enum FileLocations {
    static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    static let nets: URL = {
        let url = documents.appendingPathComponent("nets", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
}

@MainActor
final class NetStore: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published private(set) var nets: [URL] = []
    @Published private(set) var downloadProgress: Double?
    @Published var lastError: String?

    private var downloadContinuation: CheckedContinuation<URL, Error>?

    override init() {
        super.init()
        refresh()
    }

    /// Nets live in Documents/nets; files dropped into Documents via the Files
    /// app or Finder are picked up too.
    func refresh() {
        let fm = FileManager.default
        var found: [URL] = []
        for dir in [FileLocations.nets, FileLocations.documents] {
            let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
            for url in items {
                let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if !isDir && !["results.json", "last-run.log"].contains(url.lastPathComponent)
                    && !url.lastPathComponent.hasPrefix(".") {
                    found.append(url)
                }
            }
        }
        nets = found.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    func url(named name: String) -> URL? {
        nets.first { $0.lastPathComponent == name }
    }

    func importFile(_ source: URL) {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        let dest = FileLocations.nets.appendingPathComponent(source.lastPathComponent)
        do {
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: source, to: dest)
        } catch {
            lastError = "Import failed: \(error.localizedDescription)"
        }
        refresh()
    }

    func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        refresh()
    }

    func download(_ remote: URL) async {
        guard downloadProgress == nil else { return }
        downloadProgress = 0
        defer { downloadProgress = nil }
        do {
            let session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
            let tempURL: URL = try await withCheckedThrowingContinuation { continuation in
                downloadContinuation = continuation
                session.downloadTask(with: remote).resume()
            }
            session.finishTasksAndInvalidate()
            var name = remote.lastPathComponent
            if name.isEmpty || !name.contains(".") { name += ".pb.gz" }
            let dest = FileLocations.nets.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tempURL, to: dest)
        } catch {
            lastError = "Download failed: \(error.localizedDescription)"
        }
        refresh()
    }

    // URLSessionDownloadDelegate — delegateQueue is .main.

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didWriteData _: Int64, totalBytesWritten written: Int64,
                                totalBytesExpectedToWrite expected: Int64) {
        Task { @MainActor in
            self.downloadProgress = expected > 0 ? Double(written) / Double(expected) : 0
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // The file is deleted when this returns, so move it somewhere stable first.
        let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 200
        let result: Result<URL, Error>
        if !(200..<300).contains(status) {
            result = .failure(URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(status)"]))
        } else {
            do {
                try FileManager.default.moveItem(at: location, to: keep)
                result = .success(keep)
            } catch {
                result = .failure(error)
            }
        }
        Task { @MainActor in
            self.downloadContinuation?.resume(with: result)
            self.downloadContinuation = nil
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }
        Task { @MainActor in
            self.downloadContinuation?.resume(throwing: error)
            self.downloadContinuation = nil
        }
    }
}

// MARK: - Device / build info

enum BuildInfo {
    static var lc0Version: String {
        Bundle.main.object(forInfoDictionaryKey: "LC0Version") as? String ?? "unknown"
    }
    static var hasOnnx: Bool {
        (Bundle.main.object(forInfoDictionaryKey: "LC0Onnx") as? String) == "true"
    }
}

enum DeviceInfo {
    static let modelIdentifier: String = {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }()

    static let gpuName = MTLCreateSystemDefaultDevice()?.name ?? "unknown"

    static var osVersion: String { "iOS \(UIDevice.current.systemVersion)" }

    static var summary: String { "\(modelIdentifier), \(osVersion)" }

    static var physicalMemory: String {
        ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)
    }

    static var availableMemory: String {
        ByteCountFormatter.string(fromByteCount: Int64(os_proc_available_memory()), countStyle: .memory)
    }

    static var cpuCores: Int { ProcessInfo.processInfo.activeProcessorCount }

    static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static var isThrottling: Bool {
        ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical
    }

    static var lowPowerMode: Bool { ProcessInfo.processInfo.isLowPowerModeEnabled }
}
