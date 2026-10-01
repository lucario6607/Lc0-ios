import AppleArchive
import Foundation
import Metal
import System
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

    /// Replace the result with the same id, or add it.
    func upsert(_ result: BenchResult) {
        if let i = results.firstIndex(where: { $0.id == result.id }) {
            results[i] = result
            save()
        } else {
            add(result)
        }
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
    /// Converted Core ML models: one *.lc0coreml folder each.
    static let coreml: URL = {
        let url = documents.appendingPathComponent("coreml", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
}

/// A model made by the "Convert net to Core ML" workflow.
struct CoreMLModelInfo: Identifiable, Hashable {
    var id: String { folderName }
    let url: URL
    let displayName: String
    let batchSizes: [Int]
    let precision: String
    let bytes: Int64

    var folderName: String { url.lastPathComponent }

    init?(url: URL) {
        guard let data = try? Data(contentsOf: url.appendingPathComponent("lc0coreml.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        self.url = url
        displayName = json["name"] as? String ?? url.deletingPathExtension().lastPathComponent
        batchSizes = (json["batch_sizes"] as? [Int] ?? []).sorted()
        precision = json["precision"] as? String ?? "?"
        var total: Int64 = 0
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        while let file = files?.nextObject() as? URL {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        bytes = total
    }
}

@MainActor
final class NetStore: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published private(set) var nets: [URL] = []
    @Published private(set) var coremlModels: [CoreMLModelInfo] = []
    @Published private(set) var downloadProgress: Double?
    @Published private(set) var unpacking = false
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
                    && !url.lastPathComponent.hasPrefix(".") && url.pathExtension != "aar" {
                    found.append(url)
                }
            }
        }
        nets = found.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }

        let folders = (try? fm.contentsOfDirectory(at: FileLocations.coreml, includingPropertiesForKeys: nil)) ?? []
        coremlModels = folders.filter { $0.pathExtension == "lc0coreml" }
            .compactMap(CoreMLModelInfo.init(url:))
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }

    func url(named name: String) -> URL? {
        nets.first { $0.lastPathComponent == name }
    }

    func coremlModel(named folder: String) -> CoreMLModelInfo? {
        coremlModels.first { $0.folderName == folder }
    }

    /// Unpacks a *.lc0coreml.aar from the conversion workflow into Documents/coreml.
    func installArchive(_ archive: URL) async {
        unpacking = true
        defer { unpacking = false }
        do {
            try await Task.detached(priority: .userInitiated) {
                try NetStore.extract(archive, to: FileLocations.coreml)
            }.value
        } catch {
            lastError = "Couldn't unpack \(archive.lastPathComponent): \(error.localizedDescription)"
        }
        refresh()
    }

    private struct ArchiveError: LocalizedError {
        var errorDescription: String? { "not a valid .aar archive" }
    }

    nonisolated private static func extract(_ archive: URL, to dir: URL) throws {
        guard let file = ArchiveByteStream.fileStream(
                path: FilePath(archive.path), mode: .readOnly, options: [],
                permissions: FilePermissions(rawValue: 0o644)),
              let decompress = ArchiveByteStream.decompressionStream(readingFrom: file),
              let decode = ArchiveStream.decodeStream(readingFrom: decompress),
              let extract = ArchiveStream.extractStream(extractingTo: FilePath(dir.path),
                                                        flags: [.ignoreOperationNotPermitted])
        else { throw ArchiveError() }
        defer {
            try? extract.close()
            try? decode.close()
            try? decompress.close()
            try? file.close()
        }
        _ = try ArchiveStream.process(readingFrom: decode, writingTo: extract)
    }

    func importFile(_ source: URL) {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        if source.pathExtension == "aar" {
            // Copy first: the security-scoped URL may not outlive this call.
            let temp = FileManager.default.temporaryDirectory.appendingPathComponent(source.lastPathComponent)
            try? FileManager.default.removeItem(at: temp)
            do {
                try FileManager.default.copyItem(at: source, to: temp)
            } catch {
                lastError = "Import failed: \(error.localizedDescription)"
                return
            }
            Task {
                await installArchive(temp)
                try? FileManager.default.removeItem(at: temp)
            }
            return
        }
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
            if name.hasSuffix(".aar") {
                await installArchive(tempURL)
                try? FileManager.default.removeItem(at: tempURL)
                refresh()
                return
            }
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
    /// e.g. "Stockfish 19 (abc1234, nn-....nnue)", or nil if the build has no Stockfish.
    static var stockfishVersion: String? {
        guard let v = Bundle.main.object(forInfoDictionaryKey: "LC0Stockfish") as? String,
              !v.isEmpty, !v.hasPrefix("$(") else { return nil }
        return v
    }
    static var hasCoreML: Bool {
        (Bundle.main.object(forInfoDictionaryKey: "LC0CoreML") as? String) == "true"
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

    /// Performance cores (perflevel0); falls back to all cores if unknown.
    static let performanceCores: Int = sysctlInt("hw.perflevel0.physicalcpu") ?? cpuCores

    /// Efficiency cores (perflevel1), 0 if unknown.
    static let efficiencyCores: Int = sysctlInt("hw.perflevel1.physicalcpu") ?? 0

    private static func sysctlInt(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0, value > 0 else { return nil }
        return Int(value)
    }

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
