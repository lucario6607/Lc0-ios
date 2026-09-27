import SwiftUI

struct BenchmarkView: View {
    @EnvironmentObject private var runner: EngineRunner
    @EnvironmentObject private var nets: NetStore
    @EnvironmentObject private var results: ResultsStore

    @State private var config = BenchConfig.load()
    @State private var lastResult: BenchResult?

    private var networkPath: String? { nets.url(named: config.network)?.path }

    var body: some View {
        NavigationStack {
            Form {
                if let crashLog = runner.crashedRunLog { crashSection(crashLog) }
                networkSection
                if config.mode != .describenet { backendSection }
                modeSection
                runSection
                if let lastResult { lastResultSection(lastResult) }
                outputSection
                deviceSection
            }
            .navigationTitle("LeelaBench")
            .onAppear {
                nets.refresh()
                if nets.url(named: config.network) == nil {
                    config.network = nets.nets.first?.lastPathComponent ?? ""
                }
                if !Backend.available.contains(config.backend) { config.backend = .metal }
            }
            .onChange(of: config) { $0.save() }
        }
    }

    // MARK: Sections

    private var networkSection: some View {
        Section("Network") {
            if nets.nets.isEmpty {
                Text("No networks yet. Add one in the Networks tab.")
                    .foregroundStyle(.secondary)
            } else {
                Picker("Weights", selection: $config.network) {
                    ForEach(nets.nets, id: \.lastPathComponent) { url in
                        Text(url.lastPathComponent).tag(url.lastPathComponent)
                    }
                }
            }
        }
    }

    private var backendSection: some View {
        Section {
            Picker("Backend", selection: $config.backend) {
                ForEach(Backend.available) { Text($0.title).tag($0) }
            }
            if config.backend == .onnxCoreML {
                Picker("Compute units", selection: $config.coreMLUnits) {
                    ForEach(CoreMLUnits.allCases) { Text($0.title).tag($0) }
                }
            }
            LabeledContent("Options") {
                TextField(config.backend.optionsHint, text: $config.backendOpts)
                    .multilineTextAlignment(.trailing)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .font(.body.monospaced())
            }
            Stepper("Threads: \(config.threads)", value: $config.threads, in: 1...16)
        } header: {
            Text("Backend")
        } footer: {
            if !BuildInfo.hasOnnx {
                Text("This build has no ONNX Runtime, so the onnx-* backends are unavailable.")
            }
        }
    }

    private var modeSection: some View {
        Section {
            Picker("Mode", selection: $config.mode) {
                ForEach(BenchMode.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)

            switch config.mode {
            case .backendbench:
                NumberRow(title: "Batches per size", value: $config.batches, range: 2...10_000)
                NumberRow(title: "Start batch size", value: $config.startBatch, range: 1...1024)
                NumberRow(title: "Max batch size", value: $config.maxBatch, range: 1...1024)
                NumberRow(title: "Batch step", value: $config.batchStep, range: 1...256)
            case .benchmark:
                NumberRow(title: "Positions (max 34)", value: $config.numPositions, range: 1...34)
                NumberRow(title: "Time per position (ms)", value: $config.movetimeMs, range: 100...600_000)
                NumberRow(title: "Node limit (-1 = none)", value: $config.nodes, range: -1...1_000_000_000)
            case .bench, .describenet:
                EmptyView()
            }

            LabeledContent("Extra args") {
                TextField("--minibatch-size=64", text: $config.extraArgs)
                    .multilineTextAlignment(.trailing)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .font(.body.monospaced())
            }
        } header: {
            Text("Mode")
        } footer: {
            Text(config.mode.explanation)
        }
    }

    private var runSection: some View {
        Section {
            Button {
                run()
            } label: {
                HStack {
                    Spacer()
                    if runner.isRunning {
                        ProgressView().padding(.trailing, 6)
                        Text("Running…")
                    } else {
                        Label("Run", systemImage: "play.fill").bold()
                    }
                    Spacer()
                }
            }
            .disabled(runner.isRunning || networkPath == nil)
        } footer: {
            VStack(alignment: .leading, spacing: 6) {
                if let networkPath {
                    Text("lc0 " + config.arguments(networkPath: (networkPath as NSString).lastPathComponent)
                        .joined(separator: " "))
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
                Text("A run can't be cancelled, and iOS stops GPU work in the background, so keep the app open until it finishes.")
            }
        }
    }

    private func lastResultSection(_ result: BenchResult) -> some View {
        Section("Last result") {
            VStack(alignment: .leading, spacing: 4) {
                Text(result.headline).font(.title3.bold())
                Text("\(result.backend) · \(result.network)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if result.points.count > 1 {
                NpsChart(series: [result])
                    .frame(height: 200)
            }
            if let lowest = runner.lowestAvailableMemory {
                LabeledContent("Lowest free memory during run",
                               value: ByteCountFormatter.string(fromByteCount: Int64(lowest), countStyle: .memory))
            }
        }
    }

    private func crashSection(_ crashLog: String) -> some View {
        Section {
            Text("The app was closed while lc0 was running. The last lines usually say why: if \"available memory\" drops toward 0 MB, iOS killed it for using too much memory.")
                .font(.callout)
            ScrollViewReader { proxy in
                ScrollView {
                    Text(crashLog)
                        .font(.caption2.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Color.clear.frame(height: 1).id("crashBottom")
                }
                .frame(height: 220)
                .onAppear { proxy.scrollTo("crashBottom", anchor: .bottom) }
            }
            HStack {
                Button("Copy log") { UIPasteboard.general.string = crashLog }
                Spacer()
                Button("Dismiss", role: .destructive) { runner.crashedRunLog = nil }
            }
            .buttonStyle(.borderless)
        } header: {
            Label("Previous run crashed", systemImage: "exclamationmark.octagon.fill")
                .foregroundStyle(.red)
        }
    }

    private var outputSection: some View {
        Section {
            ScrollViewReader { proxy in
                ScrollView {
                    Text(runner.log.isEmpty ? "Output will appear here." : runner.log)
                        .font(.caption2.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(height: 280)
                .onChange(of: runner.log) { _ in proxy.scrollTo("bottom", anchor: .bottom) }
            }
        } header: {
            HStack {
                Text("Output")
                Spacer()
                Button("Copy") { UIPasteboard.general.string = runner.log }
                    .font(.caption)
                    .disabled(runner.log.isEmpty)
            }
        }
    }

    private var deviceSection: some View {
        Section("Device") {
            LabeledContent("Model", value: DeviceInfo.modelIdentifier)
            LabeledContent("OS", value: DeviceInfo.osVersion)
            LabeledContent("GPU", value: DeviceInfo.gpuName)
            LabeledContent("CPU cores", value: "\(DeviceInfo.cpuCores)")
            LabeledContent("RAM", value: DeviceInfo.physicalMemory)
            LabeledContent("Available to app", value: DeviceInfo.availableMemory)
            LabeledContent("Thermal state", value: DeviceInfo.thermalState)
            if DeviceInfo.lowPowerMode {
                Label("Low Power Mode is on: results will be slower", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            LabeledContent("lc0", value: BuildInfo.lc0Version)
        }
    }

    // MARK: Running

    private func run() {
        guard let path = networkPath else { return }
        let snapshot = config
        let args = snapshot.arguments(networkPath: path)
        runner.run(args) { output, code in
            var result = BenchResult(
                device: DeviceInfo.summary,
                mode: snapshot.mode,
                network: snapshot.network,
                backend: snapshot.mode == .describenet ? "-" : snapshot.backendLabel,
                backendOpts: snapshot.effectiveBackendOpts,
                threads: snapshot.threads,
                arguments: args,
                exitCode: code,
                output: String(output.suffix(60_000)))
            result.parseOutput()
            lastResult = result
            if snapshot.mode != .describenet {
                results.add(result)
            }
        }
    }
}

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
