import Charts
import SwiftUI

struct ResultsView: View {
    @EnvironmentObject private var results: ResultsStore
    @State private var selection = Set<UUID>()
    @State private var editMode = EditMode.inactive
    @State private var confirmDelete = false

    private var selected: [BenchResult] {
        results.results.filter { selection.contains($0.id) }
    }

    var body: some View {
        NavigationStack {
            List(selection: $selection) {
                if results.results.isEmpty {
                    Text("Run a benchmark to see results here.")
                        .foregroundStyle(.secondary)
                }
                ForEach(results.results) { result in
                    NavigationLink(value: result.id) {
                        ResultRow(result: result)
                    }
                }
                .onDelete { offsets in
                    results.delete(ids: Set(offsets.map { results.results[$0].id }))
                }
            }
            .navigationTitle("Results")
            .navigationDestination(for: UUID.self) { id in
                if let result = results.results.first(where: { $0.id == id }) {
                    ResultDetailView(result: result)
                }
            }
            .environment(\.editMode, $editMode)
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarLeading) {
                    if editMode.isEditing {
                        Button("Delete", role: .destructive) { confirmDelete = true }
                            .disabled(selection.isEmpty)
                    } else {
                        ShareLink(item: results.exportCSV()) {
                            Label("Export CSV", systemImage: "square.and.arrow.up")
                        }
                        .disabled(results.results.isEmpty)
                    }
                }
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    if editMode.isEditing {
                        NavigationLink("Compare (\(selection.count))") {
                            CompareView(results: selected)
                        }
                        .disabled(selection.count < 1)
                    }
                    Button(editMode.isEditing ? "Done" : "Select") {
                        editMode = editMode.isEditing ? .inactive : .active
                        if !editMode.isEditing { selection.removeAll() }
                    }
                }
            }
            .confirmationDialog("Delete \(selection.count) result(s)?", isPresented: $confirmDelete,
                                titleVisibility: .visible) {
                Button("Delete", role: .destructive) {
                    results.delete(ids: selection)
                    selection.removeAll()
                }
            }
        }
    }
}

struct ResultRow: View {
    let result: BenchResult

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(result.headline).font(.headline)
            Text(result.label).font(.subheadline).lineLimit(1)
            Text("\(result.mode.title) · \(result.threads) thr · \(result.date.formatted(date: .abbreviated, time: .shortened))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

struct ResultDetailView: View {
    let result: BenchResult

    var body: some View {
        List {
            Section {
                Text(result.headline).font(.title2.bold())
                if result.points.count > 1 {
                    NpsChart(series: [result]).frame(height: 240)
                }
            }
            if !result.points.isEmpty {
                Section("Batch size → nps") {
                    ForEach(result.points, id: \.self) { point in
                        LabeledContent("\(point.batch)", value: Int(point.nps).formatted())
                            .monospacedDigit()
                    }
                }
            }
            Section("Run") {
                LabeledContent("Network", value: result.network)
                LabeledContent("Backend", value: result.backend)
                if !result.backendOpts.isEmpty {
                    LabeledContent("Backend opts", value: result.backendOpts)
                }
                LabeledContent("Threads", value: "\(result.threads)")
                LabeledContent("Device", value: result.device)
                LabeledContent("Date", value: result.date.formatted())
                LabeledContent("Exit code", value: "\(result.exitCode)")
            }
            Section("Output") {
                Text(result.output)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
            }
        }
        .navigationTitle(result.mode.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ShareLink(item: result.output)
        }
    }
}

/// nps against batch size, one line per run.
struct NpsChart: View {
    let series: [BenchResult]

    var body: some View {
        Chart {
            ForEach(Array(series.enumerated()), id: \.element.id) { index, result in
                let name = series.count > 1 ? "\(index + 1). \(result.backend)" : result.backend
                ForEach(result.points, id: \.batch) { point in
                    LineMark(x: .value("Batch", point.batch), y: .value("nps", point.nps))
                        .foregroundStyle(by: .value("Run", name))
                    PointMark(x: .value("Batch", point.batch), y: .value("nps", point.nps))
                        .foregroundStyle(by: .value("Run", name))
                        .symbolSize(16)
                }
            }
        }
        .chartXAxisLabel("batch size")
        .chartYAxisLabel("nps")
        .chartLegend(series.count > 1 ? .visible : .hidden)
    }
}

struct CompareView: View {
    let results: [BenchResult]

    private var curves: [BenchResult] { results.filter { $0.points.count > 1 } }
    private var searches: [BenchResult] { results.filter { $0.searchNps != nil } }

    var body: some View {
        List {
            if !curves.isEmpty {
                Section("Backend benchmark") {
                    NpsChart(series: curves).frame(height: 300)
                    ForEach(Array(curves.enumerated()), id: \.element.id) { index, result in
                        VStack(alignment: .leading) {
                            Text("\(index + 1). \(result.headline)").font(.subheadline.bold())
                            Text(result.label).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            if !searches.isEmpty {
                Section("Search benchmark") {
                    Chart(searches) { result in
                        BarMark(x: .value("nps", result.searchNps ?? 0),
                                y: .value("Run", "\(result.backend)\n\(result.network)"))
                    }
                    .frame(height: CGFloat(60 * searches.count + 40))
                }
            }
            if curves.isEmpty && searches.isEmpty {
                Text("The selected runs have no numbers to compare.")
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Compare")
        .navigationBarTitleDisplayMode(.inline)
    }
}
