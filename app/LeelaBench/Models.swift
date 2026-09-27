import Foundation

// MARK: - Benchmark configuration

enum BenchMode: String, CaseIterable, Codable, Identifiable {
    case backendbench, sweep, benchmark, bench
    var id: String { rawValue }

    var title: String {
        switch self {
        case .backendbench: return "Backend"
        case .sweep: return "Sweep"
        case .benchmark: return "Search"
        case .bench: return "Quick"
        }
    }

    var explanation: String {
        switch self {
        case .backendbench: return "Raw network speed at each batch size (lc0 backendbench). Best for comparing backends."
        case .sweep: return "Finds the best session batch size: for each size, compiles the model as one session of exactly that size and measures full batches only. Each size is a separate compile, so big nets take a while."
        case .benchmark: return "Full search over test positions (lc0 benchmark)."
        case .bench: return "Short search: 10 positions × 500 ms (lc0 bench)."
        }
    }

    var isSearch: Bool { self == .benchmark || self == .bench }
}

enum Backend: String, CaseIterable, Codable, Identifiable {
    case metal, blas, eigen
    case onnxCoreML = "onnx-coreml"
    case onnxCPU = "onnx-cpu"
    case coreml
    case random
    var id: String { rawValue }

    var title: String {
        switch self {
        case .metal: return "Metal (GPU)"
        case .blas: return "BLAS (CPU, Accelerate)"
        case .eigen: return "Eigen (CPU)"
        case .onnxCoreML: return "Core ML (via ONNX)"
        case .onnxCPU: return "ONNX Runtime (CPU)"
        case .coreml: return "Core ML (native)"
        case .random: return "Random (no network)"
        }
    }

    var optionsHint: String {
        switch self {
        case .metal: return "batch=64"
        case .blas, .eigen: return "batch_size=256"
        case .onnxCoreML, .onnxCPU: return "batch=64"
        case .coreml, .random: return ""
        }
    }

    var isOnnx: Bool { self == .onnxCoreML || self == .onnxCPU }

    /// Runs a model converted by the "Convert net to Core ML" workflow instead of a net.
    var usesCoreMLModel: Bool { self == .coreml }

    /// Backends where a batch-size sweep makes sense (each size is its own compiled model).
    var supportsSweep: Bool { isOnnx || self == .coreml }

    static var available: [Backend] {
        allCases.filter { (!$0.isOnnx || BuildInfo.hasOnnx) && ($0 != .coreml || BuildInfo.hasCoreML) }
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
    /// Value of the native coreml backend's `units` option.
    var unitsOption: String {
        switch self {
        case .cpuAndGPU: return "gpu"
        case .cpuAndNeuralEngine: return "ne"
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
    /// Converted Core ML model (a *.lc0coreml folder name) for the coreml backend.
    var coremlModel = ""
    /// coreml backend batch size; 0 = pick per call from the model's sizes.
    var coremlBatch = 0
    var mode = BenchMode.backendbench
    var threads = 1

    // backendbench
    var batches = 50
    var startBatch = 16
    var maxBatch = 256
    var batchStep = 16

    // sweep
    var sweepSizes = "8, 16, 32, 64, 128, 256"

    // benchmark
    var numPositions = 10
    var movetimeMs = 5000
    var nodes = -1
    /// Search batch size (lc0 --minibatch-size); 0 = what the backend suggests.
    var minibatch = 0

    var extraArgs = ""

    var parsedSweepSizes: [Int] {
        let sizes = sweepSizes.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        return Array(Set(sizes.filter { (1...1024).contains($0) })).sorted()
    }

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

    /// The net or Core ML model this run uses, by file name.
    var selectedName: String { backend.usesCoreMLModel ? coremlModel : network }

    var effectiveBackendOpts: String {
        backend.usesCoreMLModel ? backendOpts(sessions: 0, batch: coremlBatch)
                                : backendOpts(sessions: onnxSessions, batch: onnxBatch)
    }

    /// `sessions`/`batch` of 0 leave lc0's defaults. `modelPath` is the coreml
    /// backend's model folder (left out for display).
    func backendOpts(sessions: Int, batch: Int, modelPath: String? = nil) -> String {
        var parts: [String] = []
        if backend == .coreml {
            if let modelPath { parts.append("model='\(modelPath)'") }
            parts.append("units=\(coreMLUnits.unitsOption)")
            if batch > 0 { parts.append("batch=\(batch)") }
        }
        if backend == .onnxCoreML { parts.append("gpu=\(coreMLUnits.rawValue)") }
        if backend.isOnnx && precision != .auto { parts.append("fp16=\(precision == .fp16)") }
        if backend.isOnnx && sessions > 0 { parts.append("steps=\(sessions)") }
        if backend.isOnnx && batch > 0 { parts.append("batch=\(batch)") }
        let trimmed = backendOpts.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { parts.append(trimmed) }
        return parts.joined(separator: ",")
    }

    var backendLabel: String {
        let sessions = mode == .sweep ? " sweep" : onnxSessions > 0 ? " ×\(onnxSessions)" : ""
        switch backend {
        case .onnxCoreML: return "coreml-\(coreMLUnits.shortTitle) \(effectivePrecision)\(sessions)"
        case .onnxCPU: return "onnx-cpu \(effectivePrecision)\(sessions)"
        case .coreml:
            let batch = mode == .sweep ? " sweep" : coremlBatch > 0 ? " b\(coremlBatch)" : ""
            return "coreml-native-\(coreMLUnits.shortTitle)\(batch)"
        default: return backend.rawValue
        }
    }

    /// One sweep step: a single session of `batch`, measured at exactly `batch`.
    func sweepArguments(networkPath: String, batch: Int) -> [String] {
        let coreml = backend.usesCoreMLModel
        let opts = coreml ? backendOpts(sessions: 0, batch: batch, modelPath: networkPath)
                          : backendOpts(sessions: 1, batch: batch)
        var args = ["backendbench", "--weights=\(coreml ? "" : networkPath)", "--backend=\(backend.rawValue)",
                    "--backend-opts=\(opts)",
                    "--threads=\(threads)", "--batches=\(batches)",
                    "--start-batch-size=\(batch)", "--max-batch-size=\(batch)", "--batch-step=1"]
        args += extraArgs.split(whereSeparator: \.isWhitespace).map(String.init)
        return args
    }

    func arguments(networkPath: String) -> [String] {
        if mode == .sweep {
            return sweepArguments(networkPath: networkPath, batch: parsedSweepSizes.first ?? 64)
        }
        // The coreml backend loads its model folder itself; no lc0 weights file.
        let coreml = backend.usesCoreMLModel
        var args = [mode.rawValue, "--weights=\(coreml ? "" : networkPath)", "--backend=\(backend.rawValue)"]
        let opts = coreml ? backendOpts(sessions: 0, batch: coremlBatch, modelPath: networkPath)
                          : effectiveBackendOpts
        if !opts.isEmpty { args.append("--backend-opts=\(opts)") }
        args.append("--threads=\(threads)")
        switch mode {
        case .backendbench, .sweep:
            args += ["--batches=\(batches)", "--start-batch-size=\(startBatch)",
                     "--max-batch-size=\(maxBatch)", "--batch-step=\(batchStep)"]
        case .benchmark:
            args += ["--num-positions=\(numPositions)", "--movetime=\(movetimeMs)"]
            if nodes > 0 { args.append("--nodes=\(nodes)") }
        case .bench:
            break
        }
        // Only the search tests have this flag; backendbench rejects it.
        if mode.isSearch && minibatch > 0 { args.append("--minibatch-size=\(minibatch)") }
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
        coremlModel = value(.coremlModel, d.coremlModel)
        coremlBatch = value(.coremlBatch, d.coremlBatch)
        mode = value(.mode, d.mode)
        threads = value(.threads, d.threads)
        batches = value(.batches, d.batches)
        startBatch = value(.startBatch, d.startBatch)
        maxBatch = value(.maxBatch, d.maxBatch)
        batchStep = value(.batchStep, d.batchStep)
        sweepSizes = value(.sweepSizes, d.sweepSizes)
        numPositions = value(.numPositions, d.numPositions)
        movetimeMs = value(.movetimeMs, d.movetimeMs)
        nodes = value(.nodes, d.nodes)
        minibatch = value(.minibatch, d.minibatch)
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
            network: config.selectedName,
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
