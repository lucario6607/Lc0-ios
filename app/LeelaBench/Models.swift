import Foundation

// MARK: - Benchmark configuration

enum BenchMode: String, CaseIterable, Codable, Identifiable {
    case backendbench, benchmark, bench
    var id: String { rawValue }

    var title: String {
        switch self {
        case .backendbench: return "Backend"
        case .benchmark: return "Search"
        case .bench: return "Quick"
        }
    }

    var explanation: String {
        switch self {
        case .backendbench: return "Raw network speed at each batch size (lc0 backendbench). Best for comparing backends."
        case .benchmark: return "Full search over test positions (lc0 benchmark)."
        case .bench: return "Short search: 10 positions × 500 ms (lc0 bench)."
        }
    }

    var isSearch: Bool { self != .backendbench }
}

enum Backend: String, CaseIterable, Codable, Identifiable {
    case metal, blas, eigen
    case onnxCoreML = "onnx-coreml"
    case onnxCPU = "onnx-cpu"
    case random
    var id: String { rawValue }

    var title: String {
        switch self {
        case .metal: return "Metal (GPU)"
        case .blas: return "BLAS (CPU, Accelerate)"
        case .eigen: return "Eigen (CPU)"
        case .onnxCoreML: return "Core ML (via ONNX)"
        case .onnxCPU: return "ONNX Runtime (CPU)"
        case .random: return "Random (no network)"
        }
    }

    var optionsHint: String {
        switch self {
        case .metal: return "batch=64"
        case .blas, .eigen: return "batch_size=256"
        case .onnxCoreML, .onnxCPU: return "batch=64"
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
    var shortTitle: String {
        switch self {
        case .cpuAndGPU: return "GPU"
        case .cpuAndNeuralEngine: return "ANE"
        case .all: return "all"
        }
    }
}

/// ONNX model precision. lc0 defaults to fp16 for Core ML and fp32 for CPU.
enum Precision: String, CaseIterable, Codable, Identifiable {
    case auto, fp16, fp32
    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "Default"
        case .fp16: return "FP16"
        case .fp32: return "FP32"
        }
    }
}

struct BenchConfig: Codable, Equatable {
    var network = ""
    var backend = Backend.metal
    var backendOpts = ""
    var coreMLUnits = CoreMLUnits.cpuAndNeuralEngine
    var precision = Precision.auto
    /// ONNX sessions (lc0 `steps`): each compiles its own copy of the model for
    /// one batch size. 0 = lc0's default (4 for Core ML).
    var onnxSessions = 0
    /// Batch size of the smallest session (lc0 `batch`). 0 = lc0's default (16 for Core ML).
    var onnxBatch = 0
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

    var effectivePrecision: String {
        switch precision {
        case .fp16: return "fp16"
        case .fp32: return "fp32"
        case .auto:
            switch backend {
            case .onnxCoreML: return "fp16"
            case .random: return "-"
            default: return "fp32"
            }
        }
    }

    var effectiveBackendOpts: String {
        var parts: [String] = []
        if backend == .onnxCoreML { parts.append("gpu=\(coreMLUnits.rawValue)") }
        if backend.isOnnx && precision != .auto { parts.append("fp16=\(precision == .fp16)") }
        if backend.isOnnx && onnxSessions > 0 { parts.append("steps=\(onnxSessions)") }
        if backend.isOnnx && onnxBatch > 0 { parts.append("batch=\(onnxBatch)") }
        let trimmed = backendOpts.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { parts.append(trimmed) }
        return parts.joined(separator: ",")
    }

    var backendLabel: String {
        let sessions = onnxSessions > 0 ? " ×\(onnxSessions)" : ""
        switch backend {
        case .onnxCoreML: return "coreml-\(coreMLUnits.shortTitle) \(effectivePrecision)\(sessions)"
        case .onnxCPU: return "onnx-cpu \(effectivePrecision)\(sessions)"
        default: return backend.rawValue
        }
    }

    func arguments(networkPath: String) -> [String] {
        var args = [mode.rawValue, "--weights=\(networkPath)", "--backend=\(backend.rawValue)"]
        let opts = effectiveBackendOpts
        if !opts.isEmpty { args.append("--backend-opts=\(opts)") }
        args.append("--threads=\(threads)")
        switch mode {
        case .backendbench:
            args += ["--batches=\(batches)", "--start-batch-size=\(startBatch)",
                     "--max-batch-size=\(maxBatch)", "--batch-step=\(batchStep)"]
        case .benchmark:
            args += ["--num-positions=\(numPositions)", "--movetime=\(movetimeMs)"]
            if nodes > 0 { args.append("--nodes=\(nodes)") }
        case .bench:
            break
        }
        args += extraArgs.split(whereSeparator: \.isWhitespace).map(String.init)
        return args
    }

    init() {}

    /// Field-by-field so settings survive app updates that add new fields.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ key: CodingKeys, _ fallback: T) -> T {
            (try? c.decodeIfPresent(T.self, forKey: key)) ?? fallback
        }
        let d = BenchConfig()
        network = value(.network, d.network)
        backend = value(.backend, d.backend)
        backendOpts = value(.backendOpts, d.backendOpts)
        coreMLUnits = value(.coreMLUnits, d.coreMLUnits)
        precision = value(.precision, d.precision)
        onnxSessions = value(.onnxSessions, d.onnxSessions)
        onnxBatch = value(.onnxBatch, d.onnxBatch)
        mode = value(.mode, d.mode)
        threads = value(.threads, d.threads)
        batches = value(.batches, d.batches)
        startBatch = value(.startBatch, d.startBatch)
        maxBatch = value(.maxBatch, d.maxBatch)
        batchStep = value(.batchStep, d.batchStep)
        numPositions = value(.numPositions, d.numPositions)
        movetimeMs = value(.movetimeMs, d.movetimeMs)
        nodes = value(.nodes, d.nodes)
        extraArgs = value(.extraArgs, d.extraArgs)
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

// MARK: - Parsed output

/// One row of `lc0 backendbench`.
struct BatchPoint: Codable, Hashable {
    var batch: Int
    var nps: Double
    var meanMs: Double?
}

/// One progress line of `lc0 benchmark`.
struct SearchSample: Codable, Hashable {
    var position: Int
    var timeMs: Int
    var nodes: Int
    var nps: Int
}

/// Reads lc0's console output incrementally, line by line.
struct OutputParser {
    private(set) var points: [BatchPoint] = []
    private(set) var samples: [SearchSample] = []
    private(set) var searchNps: Int?
    private(set) var currentPosition = 0
    private(set) var totalPositions: Int?
    private var partialLine = ""

    mutating func feed(_ chunk: String) {
        var lines = (partialLine + chunk).components(separatedBy: "\n")
        partialLine = lines.removeLast()
        lines.forEach { parse($0) }
    }

    mutating func finish() {
        parse(partialLine)
        partialLine = ""
    }

    private mutating func parse(_ raw: String) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return }

        // backendbench: "  64,     1234,   51.87, ..."
        let cols = line.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        if cols.count >= 5, let batch = Int(cols[0]), let nps = Double(cols[1]) {
            points.append(BatchPoint(batch: batch, nps: nps, meanMs: Double(cols[2])))
            return
        }
        // benchmark: "Position: 3/10 <fen>"
        if line.hasPrefix("Position:") {
            let fraction = line.dropFirst("Position:".count).split(separator: " ").first ?? ""
            let parts = fraction.split(separator: "/").compactMap { Int($0) }
            if parts.count == 2 {
                currentPosition = parts[0]
                totalPositions = parts[1]
            }
            return
        }
        // benchmark: "Benchmark time 123 ms, 4567 nodes, 890 nps, move e2e4"
        if line.hasPrefix("Benchmark time") {
            let numbers = line.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            if numbers.count >= 3 {
                samples.append(SearchSample(position: max(currentPosition, 1), timeMs: numbers[0],
                                            nodes: numbers[1], nps: numbers[2]))
            }
            return
        }
        // benchmark summary: "Nodes/second    : 1234"
        if line.hasPrefix("Nodes/second"), let value = line.split(separator: ":").last {
            searchNps = Int(value.trimmingCharacters(in: .whitespaces))
        }
    }
}

// MARK: - Results

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
    var points: [BatchPoint]
    var searchNps: Int?
    var output: String
    // Added later; optional so older saved results still load.
    var samples: [SearchSample]?
    var durationSeconds: Double?
    var thermalStart: String?
    var thermalEnd: String?
    var lowestFreeMemory: Int?

    var peak: BatchPoint? { points.max { $0.nps < $1.nps } }

    /// The final progress sample of each searched position.
    var positionResults: [SearchSample] {
        var last: [Int: SearchSample] = [:]
        for sample in samples ?? [] { last[sample.position] = sample }
        return last.values.sorted { $0.position < $1.position }
    }

    var succeeded: Bool { exitCode == 0 && (!points.isEmpty || searchNps != nil) }

    var headlineValue: String {
        if let searchNps { return searchNps.formatted() }
        if let peak { return Int(peak.nps).formatted() }
        return "—"
    }

    var headline: String {
        if let searchNps { return "\(searchNps.formatted()) nps" }
        if let peak { return "\(Int(peak.nps).formatted()) nps peak @ batch \(peak.batch)" }
        return exitCode == 0 ? "No results" : "Failed (exit \(exitCode))"
    }

    var networkShortName: String {
        network.replacingOccurrences(of: ".pb.gz", with: "").replacingOccurrences(of: ".pb", with: "")
    }

    var label: String { "\(backend) · \(networkShortName)" }

    /// The values plotted for this result, whichever kind of run it was.
    var sparkline: [Double] {
        if !points.isEmpty { return points.map(\.nps) }
        return positionResults.map { Double($0.nps) }
    }
}

extension BenchResult {
    init(config: BenchConfig, arguments: [String], outcome: RunOutcome) {
        self.init(
            date: outcome.started,
            device: DeviceInfo.summary,
            mode: config.mode,
            network: config.network,
            backend: config.backendLabel,
            backendOpts: config.effectiveBackendOpts,
            threads: config.threads,
            arguments: arguments,
            exitCode: outcome.exitCode,
            points: outcome.parsed.points,
            searchNps: outcome.parsed.searchNps,
            output: String(outcome.output.suffix(60_000)),
            samples: outcome.parsed.samples.isEmpty ? nil : outcome.parsed.samples,
            durationSeconds: outcome.duration,
            thermalStart: outcome.thermalStart,
            thermalEnd: outcome.thermalEnd,
            lowestFreeMemory: outcome.lowestFreeMemory)
    }
}
