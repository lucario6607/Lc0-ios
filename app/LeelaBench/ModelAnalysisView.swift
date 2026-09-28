import CoreML
import SwiftUI

/// One operation of a Core ML program, as Core ML plans to run it.
struct PlannedOp: Identifiable {
    let id: Int
    let type: String
    let output: String
    let device: String
    let supported: [String]
    let cost: Double
}

/// Where Core ML will run each operation of one function of a model.
struct ModelAnalysis {
    let units: CoreMLUnits
    let batch: Int
    let ops: [PlannedOp]

    struct Group: Identifiable {
        var id: String { name }
        let name: String
        var count = 0
        var cost = 0.0
        var devices: [String: Int] = [:]
    }

    var totalCost: Double { ops.reduce(0) { $0 + $1.cost } }

    var byDevice: [Group] { grouped(by: \.device) }
    var byType: [Group] { grouped(by: \.type) }

    /// Ops that could run on the Neural Engine setting but don't.
    var offNeuralEngine: [PlannedOp] {
        guard units != .cpuAndGPU else { return [] }
        return ops.filter { $0.device != "ANE" }
    }

    private func grouped(by key: KeyPath<PlannedOp, String>) -> [Group] {
        var groups: [String: Group] = [:]
        for op in ops {
            var g = groups[op[keyPath: key]] ?? Group(name: op[keyPath: key])
            g.count += 1
            g.cost += op.cost
            g.devices[op.device, default: 0] += 1
            groups[g.name] = g
        }
        return groups.values.sorted { $0.cost > $1.cost }
    }

    func share(_ cost: Double) -> String {
        totalCost > 0 ? String(format: "%.1f%%", 100 * cost / totalCost) : "-"
    }

    /// Plain-text summary for pasting elsewhere.
    func report(modelName: String) -> String {
        var lines = ["LeelaBench model analysis: \(modelName)",
                     "device \(DeviceInfo.summary), units \(units.title), batch \(batch), \(ops.count) ops", "",
                     "By device (share of estimated cost, op count):"]
        for g in byDevice { lines.append("  \(g.name): \(share(g.cost)), \(g.count) ops") }
        lines += ["", "By op type (cost share, count, devices):"]
        for g in byType {
            let devices = g.devices.sorted { $0.value > $1.value }.map { "\($0.key) \($0.value)" }.joined(separator: " ")
            lines.append("  \(g.name): \(share(g.cost)), \(g.count) [\(devices)]")
        }
        let off = offNeuralEngine
        lines += ["", "Not on the Neural Engine (\(off.count)):"]
        for op in off.prefix(60) {
            lines.append("  \(op.type) \(op.output) -> \(op.device) (supported: \(op.supported.joined(separator: ",")), cost \(share(op.cost)))")
        }
        lines += ["", "Most expensive ops:"]
        for op in ops.sorted(by: { $0.cost > $1.cost }).prefix(25) {
            lines.append("  \(share(op.cost)) \(op.type) \(op.output) on \(op.device)")
        }
        return lines.joined(separator: "\n")
    }
}

@available(iOS 18.0, *)
enum ModelAnalyzer {
    static func analyze(_ model: CoreMLModelInfo, units: CoreMLUnits, batch: Int) async throws -> ModelAnalysis {
        let config = MLModelConfiguration()
        config.computeUnits = units.mlComputeUnits
        config.functionName = "b\(batch)"
        let url = model.url.appendingPathComponent("model.mlmodelc")
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        guard case .program(let program) = plan.modelStructure,
              let function = program.functions["b\(batch)"] else {
            throw CocoaError(.featureUnsupported)
        }

        var ops: [PlannedOp] = []
        func visit(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                op.blocks.forEach(visit)
                // Constants have no device.
                guard let usage = plan.deviceUsage(for: op) else { continue }
                ops.append(PlannedOp(
                    id: ops.count,
                    type: op.operatorName.replacingOccurrences(of: #"^ios\d+\."#, with: "", options: .regularExpression),
                    output: op.outputs.first?.name ?? "",
                    device: name(usage.preferred),
                    supported: usage.supported.map(name),
                    cost: plan.estimatedCost(of: op)?.weight ?? 0))
            }
        }
        visit(function.block)
        return ModelAnalysis(units: units, batch: batch, ops: ops)
    }

    private static func name(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "ANE"
        @unknown default: return "?"
        }
    }
}

extension CoreMLUnits {
    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}

struct ModelAnalysisView: View {
    @EnvironmentObject private var runner: EngineRunner
    let model: CoreMLModelInfo

    @State private var units = CoreMLUnits.cpuAndNeuralEngine
    @State private var batch = 16
    @State private var analysis: ModelAnalysis?
    @State private var error: String?
    @State private var working = false

    var body: some View {
        List {
            Section {
                Picker("Compute units", selection: $units) {
                    ForEach(CoreMLUnits.allCases) { Text($0.title).tag($0) }
                }
                Picker("Batch size", selection: $batch) {
                    ForEach(model.batchSizes, id: \.self) { Text("\($0)").tag($0) }
                }
                Button {
                    analyze()
                } label: {
                    if working {
                        HStack { ProgressView(); Text("Analyzing… (the first time per size can take a while)") }
                    } else {
                        Label("Analyze", systemImage: "magnifyingglass")
                    }
                }
                .disabled(working || runner.isRunning)
            } footer: {
                Text("Asks Core ML where it will run each operation and roughly what each costs. Operations that fall off the Neural Engine are what to rewrite in the converter.")
            }

            if let error {
                Section { Text(error).foregroundStyle(.red) }
            }

            if let a = analysis {
                Section("Where the cost goes") {
                    ForEach(a.byDevice) { g in
                        HStack {
                            Text(g.name).bold().frame(width: 44, alignment: .leading)
                            ProgressView(value: a.totalCost > 0 ? g.cost / a.totalCost : 0)
                            Text(a.share(g.cost)).monospacedDigit().frame(width: 56, alignment: .trailing)
                            Text("\(g.count) ops").font(.caption).foregroundStyle(.secondary)
                                .frame(width: 60, alignment: .trailing)
                        }
                    }
                }
                Section("By operation type") {
                    ForEach(a.byType) { g in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(g.name).font(.callout.monospaced())
                                Text(g.devices.sorted { $0.value > $1.value }
                                    .map { "\($0.key) \($0.value)" }.joined(separator: " · "))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text(a.share(g.cost)).monospacedDigit()
                                Text("\(g.count)×").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                if units != .cpuAndGPU {
                    Section("Not on the Neural Engine (\(a.offNeuralEngine.count))") {
                        if a.offNeuralEngine.isEmpty {
                            Label("Everything runs on the Neural Engine", systemImage: "checkmark.circle")
                                .foregroundStyle(.green)
                        }
                        ForEach(a.offNeuralEngine.prefix(100)) { op in
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(op.type) → \(op.device)").font(.callout.monospaced())
                                Text(op.output).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                                Text("could run on: \(op.supported.joined(separator: ", "))")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Section("Most expensive operations") {
                    ForEach(a.ops.sorted { $0.cost > $1.cost }.prefix(25)) { op in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(op.type).font(.callout.monospaced())
                                Text(op.output).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer()
                            Text(op.device).font(.caption.bold())
                            Text(a.share(op.cost)).monospacedDigit().frame(width: 56, alignment: .trailing)
                        }
                    }
                }
            }
        }
        .navigationTitle("Model analysis")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let a = analysis {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button {
                        UIPasteboard.general.string = a.report(modelName: model.displayName)
                    } label: {
                        Label("Copy report", systemImage: "doc.on.doc")
                    }
                    ShareLink(item: a.report(modelName: model.displayName))
                }
            }
        }
        .onAppear {
            if !model.batchSizes.contains(batch) { batch = model.batchSizes.first ?? 1 }
        }
    }

    private func analyze() {
        guard #available(iOS 18.0, *) else {
            error = "Needs iOS 18 or later."
            return
        }
        working = true
        error = nil
        let (model, units, batch) = (self.model, self.units, self.batch)
        Task {
            do {
                analysis = try await ModelAnalyzer.analyze(model, units: units, batch: batch)
            } catch {
                self.error = "Analysis failed: \(error.localizedDescription)"
            }
            working = false
        }
    }
}
