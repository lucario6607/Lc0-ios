import Foundation
import Metal
import UIKit

// MARK: - Benchmark configuration

enum BenchMode: String, CaseIterable, Codable, Identifiable {
    case backendbench, benchmark, bench, describenet
    var id: String { rawValue }

    var title: String {
        switch self {
        case .backendbench: return "Backend"
        case .benchmark: return "Search"
        case .bench: return "Quick"
        case .describenet: return "Net info"
        }
    }

    var explanation: String {
        switch self {
        case .backendbench: return "Raw NN evaluation speed per batch size (lc0 backendbench). Best for comparing backends."
        case .benchmark: return "Full MCTS search over test positions (lc0 benchmark)."
        case .bench: return "Short search benchmark: 10 positions × 500 ms (lc0 bench)."
        case .describenet: return "Print the network's architecture and training info."
        }
    }
}

enum Backend: String, CaseIterable, Codable, Identifiable {
    case metal, blas, eigen
    case onnxCoreML = "onnx-coreml"
    case onnxCPU = "onnx-cpu"
    case random
    var id: String { rawValue }

    var title: String {
        switch self {
        case .metal: return "Metal (GPU, MPSGraph)"
        case .blas: return "BLAS (CPU, Accelerate)"
        case .eigen: return "Eigen (CPU)"
        case .onnxCoreML: return "ONNX → Core ML"
        case .onnxCPU: return "ONNX Runtime (CPU)"
        case .random: return "Random (no NN, search only)"
        }
    }

    var optionsHint: String {
        switch self {
        case .metal: return "e.g. batch=64,max_batch=1024"
        case .blas, .eigen: return "e.g. batch_size=256"
        case .onnxCoreML, .onnxCPU: return "e.g. batch=64,fp16=true"
        case .random: return ""
        }
    }

    var isOnnx: Bool { self == .onnxCoreML || self == .onnxCPU }

    static var available: [Backend] {
        allCases.filter { !$0.isOnnx || BuildInfo.hasOnnx }
    }
}

/// Core ML compute units, passed to lc0's onnx-coreml backend as `gpu=N`.
enum CoreMLUnits: Int, CaseIterable, Codable, Identifiable {
    case cpuAndGPU = 0, cpuAndNeuralEngine = 1, all = 2
    var id: Int { rawValue }
    var title: String {
        switch self {
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .all: return "All"
        }
    }
}

struct BenchConfig: Codable, Equatable {
    var network = ""
    var backend = Backend.metal
    var backendOpts = ""
    var coreMLUnits = CoreMLUnits.cpuAndNeuralEngine
    var mode = BenchMode.backendbench
    var threads = 1

    // backendbench
    var batches = 50
    var startBatch = 16
    var maxBatch = 256
    var batchStep = 16

    // benchmark
    var numPositions = 10
    var movetimeMs = 5000
    var nodes = -1

    var extraArgs = ""

    var effectiveBackendOpts: String {
        var parts: [String] = []
        if backend == .onnxCoreML { parts.append("gpu=\(coreMLUnits.rawValue)") }
        let trimmed = backendOpts.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { parts.append(trimmed) }
        return parts.joined(separator: ",")
    }

    var backendLabel: String {
        backend == .onnxCoreML ? "\(backend.rawValue) (\(coreMLUnits.title))" : backend.rawValue
    }

    func arguments(networkPath: String) -> [String] {
        var args = [mode.rawValue, "--weights=\(networkPath)"]
        if mode != .describenet {
            args.append("--backend=\(backend.rawValue)")
            let opts = effectiveBackendOpts
            if !opts.isEmpty { args.append("--backend-opts=\(opts)") }
            args.append("--threads=\(threads)")
        }
        switch mode {
        case .backendbench:
            args += ["--batches=\(batches)", "--start-batch-size=\(startBatch)",
                     "--max-batch-size=\(maxBatch)", "--batch-step=\(batchStep)"]
        case .benchmark:
            args += ["--num-positions=\(numPositions)", "--movetime=\(movetimeMs)"]
            if nodes > 0 { args.append("--nodes=\(nodes)") }
        case .bench, .describenet:
            break
        }
        args += extraArgs.split(whereSeparator: \.isWhitespace).map(String.init)
        return args
    }

    private static let key = "BenchConfig"

    static func load() -> BenchConfig {
        guard let data = UserDefaults.standard.data(forKey: key),
              let config = try? JSONDecoder().decode(BenchConfig.self, from: data)
        else { return BenchConfig() }
        return config
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }
}

// MARK: - Results

struct BatchPoint: Codable, Hashable {
    var batch: Int
    var nps: Double
}

struct BenchResult: Codable, Identifiable {
    var id = UUID()
    var date = Date()
    var device: String
    var mode: BenchMode
    var network: String
    var backend: String
    var backendOpts: String
    var threads: Int
    var arguments: [String]
    var exitCode: Int32
    var points: [BatchPoint] = []
    var searchNps: Int?
    var output: String

    var peak: BatchPoint? { points.max { $0.nps < $1.nps } }

    var headline: String {
        if let searchNps { return "\(searchNps.formatted()) nps" }
        if let peak { return "peak \(Int(peak.nps).formatted()) nps @ batch \(peak.batch)" }
        return exitCode == 0 ? "done" : "failed (exit \(exitCode))"
    }

    var label: String { "\(backend) · \(network)" }

    /// Pulls numbers out of lc0's console output.
    mutating func parseOutput() {
        for line in output.split(separator: "\n") {
            // backendbench rows: "  64,     1234,  51.87, ..."
            let cols = line.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            if cols.count >= 5, let batch = Int(cols[0]), let nps = Double(cols[1]) {
                points.append(BatchPoint(batch: batch, nps: nps))
            }
            // benchmark summary: "Nodes/second    : 1234"
            if line.hasPrefix("Nodes/second"), let value = line.split(separator: ":").last {
                searchNps = Int(value.trimmingCharacters(in: .whitespaces))
            }
        }
    }
}

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

    func add(_ result: BenchResult) {
        results.insert(result, at: 0)
        save()
    }

    func delete(ids: Set<UUID>) {
        results.removeAll { ids.contains($0.id) }
        save()
    }

    private func save() {
        if let data = try? JSONEncoder().encode(results) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// One row per data point, for spreadsheets.
    func exportCSV() -> URL {
        var csv = "date,device,mode,network,backend,backend_opts,threads,batch,nps\n"
        let formatter = ISO8601DateFormatter()
        func q(_ s: String) -> String { "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        for r in results {
            let prefix = [formatter.string(from: r.date), q(r.device), r.mode.rawValue, q(r.network),
                          q(r.backend), q(r.backendOpts), String(r.threads)].joined(separator: ",")
            if r.points.isEmpty {
                csv += prefix + ",," + (r.searchNps.map(String.init) ?? "") + "\n"
            } else {
                for p in r.points { csv += prefix + ",\(p.batch),\(Int(p.nps))\n" }
            }
        }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("leelabench-results.csv")
        try? csv.write(to: out, atomically: true, encoding: .utf8)
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
                if !isDir && url.lastPathComponent != "results.json" && !url.lastPathComponent.hasPrefix(".") {
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
        case .serious: return "serious (throttling)"
        case .critical: return "critical (throttling)"
        @unknown default: return "unknown"
        }
    }

    static var lowPowerMode: Bool { ProcessInfo.processInfo.isLowPowerModeEnabled }
}
