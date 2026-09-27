import SwiftUI

struct BenchmarkView: View {
    @EnvironmentObject private var runner: EngineRunner
    @EnvironmentObject private var nets: NetStore
    @EnvironmentObject private var results: ResultsStore

    @State private var config = BenchConfig.load()
    @State private var lastResultID: UUID?
    @State private var showDevice = false
    @State private var showCrashLog = false

    private var networkPath: String? { nets.url(named: config.network)?.path }
    private var lastResult: BenchResult? { lastResultID.flatMap { results.result(id: $0) } }

    var body: some View {
        NavigationStack {
            Form {
                if runner.crashedRunLog != nil { crashSection }
                if runner.isRunning {
                    liveSection
                } else if let lastResult {
                    lastResultSection(lastResult)
                }
                networkSection
                backendSection
                modeSection
                runSection
            }
            .navigationTitle("LeelaBench")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showDevice = true
                    } label: {
                        Label("Device", systemImage: DeviceInfo.isThrottling ? "thermometer.high" : "info.circle")
                    }
                }
            }
            .sheet(isPresented: $showDevice) { DeviceInfoView() }
            .sheet(isPresented: $showCrashLog) {
                NavigationStack {
                    LogView(title: "Crashed run", text: runner.crashedRunLog ?? "")
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Done") { showCrashLog = false }
                            }
                        }
                }
            }
            .onAppear {
                nets.refresh()
                if nets.url(named: config.network) == nil {
                    config.network = nets.nets.first?.lastPathComponent ?? ""
                }
                if !Backend.available.contains(config.backend) { config.backend = .metal }
            }
            .onChange(of: config) { $0.save() }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    // MARK: Status sections

    private var crashSection: some View {
        Section {
            Text("LeelaBench was closed while lc0 was running, usually because iOS killed it for using too much memory. The log's last lines show the free memory just before.")
                .font(.callout)
            Button("View log") { showCrashLog = true }
            Button("Dismiss", role: .destructive) { runner.crashedRunLog = nil }
        } header: {
            Label("Previous run crashed", systemImage: "exclamationmark.octagon.fill")
                .foregroundStyle(.red)
        }
    }

    private var liveSection: some View {
        Section {
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                HStack {
                    StatTile(title: "Elapsed", value: elapsed())
                    StatTile(title: "Free memory", value: DeviceInfo.availableMemory)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Thermal").font(.caption).foregroundStyle(.secondary)
                        Text(DeviceInfo.thermalState).font(.title3.bold())
                            .foregroundStyle(DeviceInfo.isThrottling ? thermalColor(DeviceInfo.thermalState) : .primary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            liveProgress
            NavigationLink("Live output") { LiveLogView() }
        } header: {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Running \(config.mode.title.lowercased()) benchmark")
            }
        } footer: {
            Text("Keep LeelaBench open: iOS stops GPU work in the background, and a run can't be cancelled.")
        }
    }

    @ViewBuilder
    private var liveProgress: some View {
        let parsed = runner.parsed
        if !parsed.points.isEmpty {
            BatchNpsChart(series: [BatchSeries(name: "live", points: parsed.points)])
                .frame(height: 200)
                .padding(.vertical, 6)
            if let last = parsed.points.last {
                LabeledContent("Batch \(last.batch)", value: "\(Int(last.nps).formatted()) nps")
                    .monospacedDigit()
            }
        } else if !parsed.samples.isEmpty {
            let positions = Dictionary(grouping: parsed.samples, by: \.position)
                .compactMap { $0.value.last }
                .sorted { $0.position < $1.position }
            PositionNpsChart(positions: positions)
                .frame(height: 180)
                .padding(.vertical, 6)
            if let last = parsed.samples.last {
                LabeledContent("Position \(parsed.currentPosition)/\(parsed.totalPositions ?? 0)",
                               value: "\(last.nps.formatted()) nps")
                    .monospacedDigit()
            }
        } else {
            Text(config.backend == .onnxCoreML
                 ? "Loading the network. Core ML compiles the model on first use, which can take a few minutes for big nets."
                 : "Loading the network…")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private func lastResultSection(_ result: BenchResult) -> some View {
        Section("Last run") {
            NavigationLink {
                ResultDetailView(result: result)
            } label: {
                ResultRow(result: result)
            }
        }
    }

    // MARK: Settings sections

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
            if config.backend.isOnnx {
                Picker("Precision", selection: $config.precision) {
                    ForEach(Precision.allCases) { p in
                        Text(p == .auto ? "Default (\(config.backend == .onnxCoreML ? "FP16" : "FP32"))" : p.title).tag(p)
                    }
                }
            }
            if config.backend != .random {
                LabeledContent("Options") {
                    TextField(config.backend.optionsHint, text: $config.backendOpts)
                        .multilineTextAlignment(.trailing)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.body.monospaced())
                }
            }
            Stepper("Threads: \(config.threads)", value: $config.threads, in: 1...16)
        } header: {
            Text("Backend")
        } footer: {
            switch config.backend {
            case .metal:
                Text("Runs on the GPU in FP32 (lc0's Metal backend has no FP16 mode).")
            case .onnxCoreML:
                Text("Core ML decides per layer where to run within the allowed units, and always keeps the CPU as a fallback. The first run per network is slow while Core ML compiles it.")
            case .blas, .eigen, .onnxCPU:
                Text("Runs on the CPU. Try more threads.")
            case .random:
                Text("No network is evaluated; measures search overhead only.")
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
            case .bench:
                EmptyView()
            }

            DisclosureGroup("Advanced") {
                LabeledContent("Extra args") {
                    TextField("--minibatch-size=64", text: $config.extraArgs)
                        .multilineTextAlignment(.trailing)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.body.monospaced())
                }
                if let networkPath {
                    Text("lc0 " + config.arguments(networkPath: (networkPath as NSString).lastPathComponent)
                        .joined(separator: " "))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        } header: {
            Text("Test")
        } footer: {
            Text(config.mode.explanation)
        }
    }

    private var runSection: some View {
        Section {
            Button {
                run()
            } label: {
                Label(runner.isRunning ? "Running…" : "Run benchmark", systemImage: "play.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .disabled(runner.isRunning || networkPath == nil)
        } footer: {
            if DeviceInfo.lowPowerMode {
                Label("Low Power Mode is on, so results will be slower.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            } else if DeviceInfo.isThrottling {
                Label("The device is hot (\(DeviceInfo.thermalState)); let it cool for comparable results.",
                      systemImage: "thermometer.high")
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: Running

    private func elapsed() -> String {
        guard let start = runner.startedAt else { return "0:00" }
        return Duration.seconds(Date().timeIntervalSince(start)).formatted(.time(pattern: .minuteSecond))
    }

    private func run() {
        guard let path = networkPath else { return }
        let snapshot = config
        let args = snapshot.arguments(networkPath: path)
        lastResultID = nil
        runner.run(args) { outcome in
            let result = BenchResult(config: snapshot, arguments: args, outcome: outcome)
            results.add(result)
            lastResultID = result.id
        }
    }
}

/// The current run's output, updating as lc0 writes.
private struct LiveLogView: View {
    @EnvironmentObject private var runner: EngineRunner

    var body: some View {
        LogView(title: "Output", text: runner.log, follow: true)
    }
}
