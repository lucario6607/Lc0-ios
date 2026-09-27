import SwiftUI

struct ResultDetailView: View {
    @EnvironmentObject private var results: ResultsStore
    let result: BenchResult

    private var positions: [SearchSample] { result.positionResults }

    var body: some View {
        List {
            summarySection

            if !result.points.isEmpty {
                Section {
                    BatchNpsChart(series: [BatchSeries(name: result.backend, points: result.points)])
                        .frame(height: 240)
                        .padding(.vertical, 8)
                } header: {
                    Text(result.mode == .sweep ? "Speed by session batch size" : "Speed by batch size")
                } footer: {
                    if result.mode == .sweep {
                        Text("Each point is the model compiled as one session of that batch size and measured with full batches only, so there's no padding.")
                    }
                }
                if result.points.contains(where: { $0.meanMs != nil }) {
                    Section("Time per batch") {
                        BatchLatencyChart(points: result.points)
                            .frame(height: 180)
                            .padding(.vertical, 8)
                    }
                }
            }

            if !positions.isEmpty {
                Section("Speed by position") {
                    PositionNpsChart(positions: positions)
                        .frame(height: 220)
                        .padding(.vertical, 8)
                }
                if let samples = result.samples, samples.count > 2 {
                    Section("Speed over time") {
                        SearchTimelineChart(samples: samples)
                            .frame(height: 200)
                            .padding(.vertical, 8)
                    }
                }
            }

            dataSection
            runSection

            Section {
                NavigationLink("Output log") {
                    LogView(title: "Output", text: result.output)
                }
                DisclosureGroup("Command line") {
                    Text("lc0 " + result.arguments.joined(separator: " "))
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
            }
        }
        .navigationTitle(result.backend)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ShareLink(item: results.exportCSV([result], name: "leelabench-\(result.backend)-\(result.networkShortName)")) {
                Label("Export CSV", systemImage: "square.and.arrow.up")
            }
        }
    }

    // MARK: Sections

    private var summarySection: some View {
        Section {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(result.mode.title.uppercased())
                        .font(.caption2.bold())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15), in: Capsule())
                    Text(result.networkShortName).font(.subheadline).lineLimit(1)
                }
                if !result.succeeded {
                    Label(result.headline, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                }
                HStack(alignment: .top) {
                    if let peak = result.peak {
                        StatTile(title: "Peak", value: Int(peak.nps).formatted(), unit: "nps")
                        StatTile(title: "At batch", value: "\(peak.batch)")
                        StatTile(title: "Sizes tested", value: "\(result.points.count)")
                    } else if let nps = result.searchNps {
                        StatTile(title: "Search speed", value: nps.formatted(), unit: "nps")
                        StatTile(title: "Positions", value: "\(positions.count)")
                        StatTile(title: "Nodes", value: positions.map(\.nodes).reduce(0, +).formatted())
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private var dataSection: some View {
        if !result.points.isEmpty {
            Section {
                DisclosureGroup("All batch sizes") {
                    ForEach(result.points, id: \.batch) { p in
                        HStack {
                            Text("batch \(p.batch)")
                            Spacer()
                            if let ms = p.meanMs {
                                Text(String(format: "%.2f ms", ms)).foregroundStyle(.secondary)
                            }
                            Text("\(Int(p.nps).formatted()) nps").frame(minWidth: 100, alignment: .trailing)
                        }
                        .font(.callout.monospacedDigit())
                    }
                }
            }
        } else if !positions.isEmpty {
            Section {
                DisclosureGroup("All positions") {
                    ForEach(positions, id: \.position) { s in
                        HStack {
                            Text("position \(s.position)")
                            Spacer()
                            Text("\(s.nodes.formatted()) nodes").foregroundStyle(.secondary)
                            Text("\(s.nps.formatted()) nps").frame(minWidth: 90, alignment: .trailing)
                        }
                        .font(.callout.monospacedDigit())
                    }
                }
            }
        }
    }

    private var runSection: some View {
        Section("Run") {
            LabeledContent("Network", value: result.network)
            LabeledContent("Backend", value: result.backend)
            if !result.backendOpts.isEmpty {
                LabeledContent("Backend options", value: result.backendOpts)
            }
            LabeledContent("Threads", value: "\(result.threads)")
            LabeledContent("Device", value: result.device)
            LabeledContent("Date", value: result.date.formatted(date: .abbreviated, time: .shortened))
            if let duration = result.durationSeconds {
                LabeledContent("Duration", value: Duration.seconds(duration).formatted(.time(pattern: .minuteSecond)))
            }
            if let start = result.thermalStart, let end = result.thermalEnd {
                LabeledContent("Thermal state") {
                    HStack(spacing: 4) {
                        Text(start).foregroundStyle(thermalColor(start))
                        Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.secondary)
                        Text(end).foregroundStyle(thermalColor(end))
                    }
                }
            }
            if let lowest = result.lowestFreeMemory {
                LabeledContent("Lowest free memory",
                               value: ByteCountFormatter.string(fromByteCount: Int64(lowest), countStyle: .memory))
            }
            if result.exitCode != 0 {
                LabeledContent("Exit code", value: "\(result.exitCode)")
            }
        }
    }
}
