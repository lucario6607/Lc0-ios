import SwiftUI

struct BenchmarkView: View {
    @EnvironmentObject private var runner: EngineRunner
    @EnvironmentObject private var nets: NetStore
    @EnvironmentObject private var results: ResultsStore

    @State private var config = BenchConfig.load()
    @State private var lastResultID: UUID?
    @State private var showDevice = false
    @State private var showCrashLog = false
    @State private var sweep: SweepState?

    /// A batch-size sweep in progress: one lc0 run per size.
    private struct SweepState {
        var config: BenchConfig
        var networkPath: String
        var sizes: [Int]
        var index = 0
        var resultID = UUID()
        var points: [BatchPoint] = []
        var failed: [Int] = []
        var output = ""
        var started = Date()
        var thermalStart = DeviceInfo.thermalState
        var lowestFreeMemory: Int?

        var currentSize: Int? { index < sizes.count ? sizes[index] : nil }
    }

    private var isBusy: Bool { runner.isRunning || sweep != nil }

    /// The selected net's file, or the Core ML model folder for the native backend.
    private var networkPath: String? {
        config.backend.usesCoreMLModel ? selectedCoreMLModel?.url.path : nets.url(named: config.network)?.path
    }

    private var selectedCoreMLModel: CoreMLModelInfo? { nets.coremlModel(named: config.coremlModel) }

    /// Sweep sizes, limited to those compiled into the model for the native backend.
    private var sweepSizes: [Int] {
        guard config.backend.usesCoreMLModel else { return config.parsedSweepSizes }
        let available = Set(selectedCoreMLModel?.batchSizes ?? [])
        return config.parsedSweepSizes.filter(available.contains)
    }

    /// Core ML with several sessions on a large net will likely exceed the
    /// per-app memory limit.
    private var bigNetWarning: Bool {
        guard config.backend == .onnxCoreML, config.onnxSessions != 1,
              let url = nets.url(named: config.network),
              let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        else { return false }
        return size > 100_000_000
    }
    private var lastResult: BenchResult? { lastResultID.flatMap { results.result(id: $0) } }

    var body: some View {
        NavigationStack {
            Form {
                if runner.crashedRunLog != nil { crashSection }
                if isBusy {
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
                if nets.coremlModel(named: config.coremlModel) == nil {
                    config.coremlModel = nets.coremlModels.first?.folderName ?? ""
                }
                if nets.url(named: config.network) == nil {
                    config.network = nets.nets.first?.lastPathComponent ?? ""
                }
                if !Backend.available.contains(config.backend) { config.backend = .metal }
            }
            .onChange(of: config) { newConfig in
                if newConfig.mode == .sweep && !newConfig.backend.supportsSweep {
                    config.mode = .backendbench
                }
                newConfig.save()
            }
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
                if let sweep, let size = sweep.currentSize {
                    Text("Sweep: batch \(size) (\(sweep.index + 1) of \(sweep.sizes.count))")
                } else {
                    Text("Running \(config.mode.title.lowercased()) benchmark")
                }
            }
        } footer: {
            Text("Keep LeelaBench open: iOS stops GPU work in the background, and a run can't be cancelled.")
        }
    }

    @ViewBuilder
    private var liveProgress: some View {
        let parsed = runner.parsed
        if let sweep {
            if sweep.points.isEmpty {
                Text("Compiling and measuring batch \(sweep.currentSize ?? 0). Core ML compiles the model for every size, which can take a few minutes for big nets.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                BatchNpsChart(series: [BatchSeries(name: "sweep", points: sweep.points)])
                    .frame(height: 200)
                    .padding(.vertical, 6)
                LabeledContent("Measuring batch \(sweep.currentSize ?? 0)",
                               value: parsed.points.isEmpty ? "compiling…" : "measuring…")
            }
            if !sweep.failed.isEmpty {
                Text("Failed: batch \(sweep.failed.map(String.init).joined(separator: ", "))")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        } else if !parsed.points.isEmpty {
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

    @ViewBuilder
    private var networkSection: some View {
        if config.backend.usesCoreMLModel {
            Section {
                if nets.coremlModels.isEmpty {
                    Text("No Core ML models yet. Convert a net with the \"Convert net to Core ML\" workflow, then add it in the Networks tab.")
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $config.coremlModel) {
                        ForEach(nets.coremlModels) { Text($0.displayName).tag($0.folderName) }
                    }
                    if let model = selectedCoreMLModel {
                        LabeledContent("Batch sizes", value: model.batchSizes.map(String.init).joined(separator: ", "))
                        LabeledContent("Precision", value: model.precision.uppercased())
                    }
                }
            } header: {
                Text("Core ML model")
            }
        } else {
            netPickerSection
        }
    }

    private var netPickerSection: some View {
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
            if config.backend == .onnxCoreML || config.backend == .coreml {
                Picker("Compute units", selection: $config.coreMLUnits) {
                    ForEach(CoreMLUnits.allCases) { Text($0.title).tag($0) }
                }
            }
            if config.backend == .coreml && config.mode != .sweep {
                NumberRow(title: "Batch (0 = auto)", value: $config.coremlBatch, range: 0...1024)
                if config.coremlBatch > 0, let model = selectedCoreMLModel,
                   !model.batchSizes.contains(config.coremlBatch) {
                    Label("This model has batch sizes \(model.batchSizes.map(String.init).joined(separator: ", ")) only.",
                          systemImage: "exclamationmark.triangle")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
            }
            if config.backend.isOnnx {
                Picker("Precision", selection: $config.precision) {
                    ForEach(Precision.allCases) { p in
                        Text(p == .auto ? "Default (\(config.backend == .onnxCoreML ? "FP16" : "FP32"))" : p.title).tag(p)
                    }
                }
                if config.mode != .sweep {
                    Picker("Sessions", selection: $config.onnxSessions) {
                        Text("Default (\(config.backend == .onnxCoreML ? 4 : 1))").tag(0)
                        Text("1 (least memory)").tag(1)
                        Text("2").tag(2)
                        Text("4").tag(4)
                    }
                    NumberRow(title: "Session batch (0 = default)", value: $config.onnxBatch, range: 0...1024)
                }
                if bigNetWarning && config.mode != .sweep {
                    Label("Big network: each session keeps its own copy of the model, and iOS limits the app to about \(DeviceInfo.availableMemory). Set Sessions to 1 (with a session batch of 64 or so).",
                          systemImage: "memorychip")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
            }
            if config.backend != .random && config.backend != .coreml {
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
            case .coreml:
                Text("Runs a model converted ahead of time on a Mac, directly with Core ML: no ONNX Runtime and no on-phone conversion, so it needs far less memory. Batch 0 picks the smallest compiled size that fits each call; a fixed batch pads to that size.")
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
                ForEach(BenchMode.allCases.filter { $0 != .sweep || config.backend.supportsSweep }) {
                    Text($0.title).tag($0)
                }
            }
            .pickerStyle(.segmented)

            switch config.mode {
            case .sweep:
                LabeledContent("Batch sizes") {
                    TextField("8, 16, 32, 64", text: $config.sweepSizes)
                        .multilineTextAlignment(.trailing)
                        .keyboardType(.numbersAndPunctuation)
                        .font(.body.monospaced())
                }
                NumberRow(title: "Batches per size", value: $config.batches, range: 2...10_000)
                if config.backend.usesCoreMLModel {
                    Text("Each size uses the model's precompiled function for that batch size, measured only at that size. Sizes not in the model are skipped\(sweepSizes.isEmpty ? "" : "; this sweep runs " + sweepSizes.map(String.init).joined(separator: ", ")).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Each size runs as one session of exactly that size (lc0 steps=1, batch=N) and is measured only at that size, so there's no padding. Sessions / Session batch above are ignored.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .backendbench:
                NumberRow(title: "Batches per size", value: $config.batches, range: 2...10_000)
                NumberRow(title: "Start batch size", value: $config.startBatch, range: 1...1024)
                NumberRow(title: "Max batch size", value: $config.maxBatch, range: 1...1024)
                NumberRow(title: "Batch step", value: $config.batchStep, range: 1...256)
            case .benchmark:
                NumberRow(title: "Positions (max 34)", value: $config.numPositions, range: 1...34)
                NumberRow(title: "Time per position (ms)", value: $config.movetimeMs, range: 100...600_000)
                NumberRow(title: "Node limit (-1 = none)", value: $config.nodes, range: -1...1_000_000_000)
                NumberRow(title: "Minibatch (0 = backend)", value: $config.minibatch, range: 0...1024)
            case .bench:
                NumberRow(title: "Minibatch (0 = backend)", value: $config.minibatch, range: 0...1024)
            }

            DisclosureGroup("Advanced") {
                LabeledContent("Extra args") {
                    TextField("--flag=value (all tests)", text: $config.extraArgs)
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
                Label(isBusy ? "Running…" : config.mode == .sweep ? "Run sweep" : "Run benchmark",
                      systemImage: "play.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .disabled(isBusy || networkPath == nil
                      || (config.mode == .sweep && sweepSizes.isEmpty))
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
        guard let start = sweep?.started ?? runner.startedAt else { return "0:00" }
        return Duration.seconds(Date().timeIntervalSince(start)).formatted(.time(pattern: .minuteSecond))
    }

    private func run() {
        guard let path = networkPath else { return }
        let snapshot = config
        lastResultID = nil
        if snapshot.mode == .sweep {
            sweep = SweepState(config: snapshot, networkPath: path, sizes: sweepSizes)
            runSweepStep()
            return
        }
        let args = snapshot.arguments(networkPath: path)
        runner.run(args) { outcome in
            let result = BenchResult(config: snapshot, arguments: args, outcome: outcome)
            results.add(result)
            lastResultID = result.id
        }
    }

    /// Runs the current sweep size, records it, then moves on to the next.
    private func runSweepStep() {
        guard let state = sweep, let size = state.currentSize else {
            lastResultID = sweep?.resultID
            sweep = nil
            return
        }
        let args = state.config.sweepArguments(networkPath: state.networkPath, batch: size)
        runner.run(args) { outcome in
            guard var state = sweep else { return }
            if let point = outcome.parsed.points.last(where: { $0.batch == size }) {
                state.points.append(point)
            } else {
                state.failed.append(size)
            }
            state.output += "===== session batch \(size) (exit \(outcome.exitCode)) =====\n" + outcome.output + "\n"
            state.output = String(state.output.suffix(60_000))
            if let low = outcome.lowestFreeMemory {
                state.lowestFreeMemory = min(state.lowestFreeMemory ?? low, low)
            }
            state.index += 1
            sweep = state
            // Save after every size, so a crash later in the sweep keeps what's done.
            results.upsert(sweepResult(state, lastArguments: args, thermalEnd: outcome.thermalEnd))
            runSweepStep()
        }
    }

    private func sweepResult(_ state: SweepState, lastArguments: [String], thermalEnd: String) -> BenchResult {
        BenchResult(
            id: state.resultID,
            date: state.started,
            device: DeviceInfo.summary,
            mode: .sweep,
            network: state.config.selectedName,
            backend: state.config.backendLabel,
            backendOpts: state.config.backendOpts(sessions: 1, batch: 0) + " (batch swept)",
            threads: state.config.threads,
            arguments: lastArguments,
            exitCode: state.points.isEmpty ? 1 : 0,
            points: state.points,
            searchNps: nil,
            output: state.output,
            durationSeconds: Date().timeIntervalSince(state.started),
            thermalStart: state.thermalStart,
            thermalEnd: thermalEnd,
            lowestFreeMemory: state.lowestFreeMemory)
    }
}

/// The current run's output, updating as lc0 writes.
private struct LiveLogView: View {
    @EnvironmentObject private var runner: EngineRunner

    var body: some View {
        LogView(title: "Output", text: runner.log, follow: true)
    }
}
