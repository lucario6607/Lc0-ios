import Charts
import SwiftUI

struct ResultsView: View {
    @EnvironmentObject private var results: ResultsStore

    private enum Filter: String, CaseIterable, Identifiable {
        case all = "All", backend = "Backend", search = "Search"
        var id: String { rawValue }
    }

    @State private var filter = Filter.all
    @State private var showCompare = false
    @State private var confirmDeleteAll = false

    private var filtered: [BenchResult] {
        switch filter {
        case .all: return results.results
        case .backend: return results.results.filter { $0.mode == .backendbench }
        case .search: return results.results.filter { $0.mode.isSearch }
        }
    }

    var body: some View {
        NavigationStack {
            List {
                if results.results.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "chart.xyaxis.line").font(.largeTitle).foregroundStyle(.secondary)
                        Text("No results yet").font(.headline)
                        Text("Runs from the Benchmark tab show up here.")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 40)
                    .listRowBackground(Color.clear)
                } else {
                    Picker("Show", selection: $filter) {
                        ForEach(Filter.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets())

                    ForEach(filtered) { result in
                        NavigationLink(value: result.id) {
                            ResultRow(result: result)
                        }
                    }
                    .onDelete { offsets in
                        results.delete(ids: Set(offsets.map { filtered[$0].id }))
                    }
                }
            }
            .navigationTitle("Results")
            .navigationDestination(for: UUID.self) { id in
                if let result = results.result(id: id) {
                    ResultDetailView(result: result)
                }
            }
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button {
                        showCompare = true
                    } label: {
                        Label("Compare", systemImage: "square.stack.3d.down.right")
                    }
                    .disabled(results.results.count < 2)

                    Menu {
                        ShareLink(item: results.exportCSV()) {
                            Label("Export all as CSV", systemImage: "square.and.arrow.up")
                        }
                        Button(role: .destructive) {
                            confirmDeleteAll = true
                        } label: {
                            Label("Delete all results", systemImage: "trash")
                        }
                    } label: {
                        Label("More", systemImage: "ellipsis.circle")
                    }
                    .disabled(results.results.isEmpty)
                }
            }
            .confirmationDialog("Delete all \(results.results.count) results?",
                                isPresented: $confirmDeleteAll, titleVisibility: .visible) {
                Button("Delete all", role: .destructive) { results.deleteAll() }
            }
            .sheet(isPresented: $showCompare) {
                ComparePickerView().environmentObject(results)
            }
        }
    }
}

struct ResultRow: View {
    let result: BenchResult

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(result.backend).font(.headline).lineLimit(1)
                    Text(result.mode.title)
                        .font(.caption2.bold())
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                }
                Text(result.networkShortName)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 4) {
                    Text(result.date.formatted(date: .abbreviated, time: .shortened))
                    if let end = result.thermalEnd, end != "nominal" {
                        Image(systemName: "thermometer.medium").foregroundStyle(thermalColor(end))
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            VStack(alignment: .trailing, spacing: 2) {
                if result.succeeded {
                    Text(result.headlineValue).font(.headline.monospacedDigit())
                    Text(result.peak != nil ? "peak nps" : "nps").font(.caption2).foregroundStyle(.secondary)
                } else {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                }
            }
            Sparkline(values: result.sparkline)
                .frame(width: 56, height: 30)
        }
        .padding(.vertical, 2)
    }
}

/// Pick results to overlay.
struct ComparePickerView: View {
    @EnvironmentObject private var results: ResultsStore
    @Environment(\.dismiss) private var dismiss
    @State private var selected: [UUID] = []

    private var chosen: [BenchResult] {
        selected.compactMap { results.result(id: $0) }
    }

    var body: some View {
        NavigationStack {
            List(results.results) { result in
                Button {
                    if let i = selected.firstIndex(of: result.id) {
                        selected.remove(at: i)
                    } else {
                        selected.append(result.id)
                    }
                } label: {
                    HStack {
                        Image(systemName: selected.contains(result.id) ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(selected.contains(result.id) ? Color.accentColor : .secondary)
                            .font(.title3)
                        ResultRow(result: result)
                    }
                }
                .buttonStyle(.plain)
            }
            .navigationTitle("Compare")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    NavigationLink("Compare (\(selected.count))") {
                        CompareView(results: chosen)
                    }
                    .disabled(selected.count < 2)
                }
            }
        }
    }
}

struct CompareView: View {
    let results: [BenchResult]

    private var curves: [BenchResult] { results.filter { !$0.points.isEmpty } }
    private var searches: [BenchResult] { results.filter { $0.searchNps != nil } }

    /// Series names must be unique for the chart legend.
    private func seriesName(_ result: BenchResult, index: Int) -> String {
        "\(index + 1). \(result.backend)"
    }

    var body: some View {
        List {
            if !curves.isEmpty {
                Section("Speed by batch size") {
                    BatchNpsChart(series: curves.enumerated().map { i, r in
                        BatchSeries(name: seriesName(r, index: i), points: r.points)
                    })
                    .frame(height: 300)
                    .padding(.vertical, 8)
                }
                Section("Peak") {
                    ForEach(Array(curves.enumerated()), id: \.element.id) { i, r in
                        CompareRow(name: seriesName(r, index: i), detail: r.networkShortName,
                                   value: r.peak.map { Int($0.nps) } ?? 0,
                                   best: curves.compactMap { $0.peak.map { Int($0.nps) } }.max() ?? 1,
                                   suffix: r.peak.map { "@ \($0.batch)" } ?? "")
                    }
                }
            }
            if !searches.isEmpty {
                Section("Search speed") {
                    Chart(Array(searches.enumerated()), id: \.element.id) { i, r in
                        BarMark(x: .value("nps", r.searchNps ?? 0),
                                y: .value("Run", seriesName(r, index: i)))
                            .foregroundStyle(Color.accentColor.gradient)
                            .annotation(position: .trailing) {
                                Text((r.searchNps ?? 0).formatted()).font(.caption2)
                            }
                    }
                    .frame(height: CGFloat(44 * searches.count + 30))
                    .padding(.vertical, 8)
                    ForEach(Array(searches.enumerated()), id: \.element.id) { i, r in
                        CompareRow(name: seriesName(r, index: i), detail: r.networkShortName,
                                   value: r.searchNps ?? 0,
                                   best: searches.compactMap(\.searchNps).max() ?? 1, suffix: "")
                    }
                }
            }
        }
        .navigationTitle("Comparison")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct CompareRow: View {
    let name: String
    let detail: String
    let value: Int
    let best: Int
    let suffix: String

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(name).font(.subheadline.bold())
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing) {
                Text("\(value.formatted()) nps \(suffix)").font(.subheadline.monospacedDigit())
                if value < best, best > 0 {
                    Text(String(format: "%.0f%% of best", 100 * Double(value) / Double(best)))
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("fastest").font(.caption).foregroundStyle(.green)
                }
            }
        }
    }
}
