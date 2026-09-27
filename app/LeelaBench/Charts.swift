import Charts
import SwiftUI

/// A named line of backendbench points.
struct BatchSeries: Identifiable {
    var id: String { name }
    var name: String
    var points: [BatchPoint]
}

/// nps against batch size; one line per series. Marks the peak when there is
/// a single series.
struct BatchNpsChart: View {
    let series: [BatchSeries]

    private var peak: BatchPoint? {
        series.count == 1 ? series[0].points.max { $0.nps < $1.nps } : nil
    }

    var body: some View {
        Chart {
            ForEach(series) { s in
                ForEach(s.points, id: \.batch) { point in
                    LineMark(x: .value("Batch size", point.batch), y: .value("nps", point.nps))
                        .foregroundStyle(by: .value("Run", s.name))
                        .interpolationMethod(.monotone)
                    if series.count == 1 {
                        AreaMark(x: .value("Batch size", point.batch), y: .value("nps", point.nps))
                            .foregroundStyle(.linearGradient(colors: [.accentColor.opacity(0.25), .clear],
                                                             startPoint: .top, endPoint: .bottom))
                            .interpolationMethod(.monotone)
                    }
                }
            }
            if let peak {
                PointMark(x: .value("Batch size", peak.batch), y: .value("nps", peak.nps))
                    .symbolSize(60)
                    .annotation(position: .top, alignment: .center) {
                        Text("\(Int(peak.nps).formatted()) @ \(peak.batch)")
                            .font(.caption2.bold())
                            .padding(.horizontal, 4)
                            .background(.background.opacity(0.8), in: Capsule())
                    }
            }
        }
        .chartXAxisLabel("batch size")
        .chartYAxisLabel("nodes / s")
        .chartLegend(series.count > 1 ? .visible : .hidden)
        .chartLegend(position: .bottom, alignment: .leading)
    }
}

/// Mean time per batch against batch size.
struct BatchLatencyChart: View {
    let points: [BatchPoint]

    var body: some View {
        Chart(points.filter { $0.meanMs != nil }, id: \.batch) { point in
            LineMark(x: .value("Batch size", point.batch), y: .value("ms", point.meanMs ?? 0))
                .interpolationMethod(.monotone)
                .foregroundStyle(.orange)
            PointMark(x: .value("Batch size", point.batch), y: .value("ms", point.meanMs ?? 0))
                .symbolSize(14)
                .foregroundStyle(.orange)
        }
        .chartXAxisLabel("batch size")
        .chartYAxisLabel("ms per batch")
    }
}

/// Final nps of each searched position.
struct PositionNpsChart: View {
    let positions: [SearchSample]

    private var average: Double {
        positions.isEmpty ? 0 : Double(positions.map(\.nps).reduce(0, +)) / Double(positions.count)
    }

    var body: some View {
        Chart {
            ForEach(positions, id: \.position) { sample in
                BarMark(x: .value("Position", "\(sample.position)"), y: .value("nps", sample.nps))
                    .foregroundStyle(Color.accentColor.gradient)
            }
            if positions.count > 1 {
                RuleMark(y: .value("Average", average))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .trailing) {
                        Text("avg \(Int(average).formatted())").font(.caption2).foregroundStyle(.secondary)
                    }
            }
        }
        .chartXAxisLabel("position")
        .chartYAxisLabel("nodes / s")
    }
}

/// nps over the course of a search benchmark, positions laid end to end.
struct SearchTimelineChart: View {
    let samples: [SearchSample]

    private struct TimedSample: Identifiable {
        let id: Int
        let seconds: Double
        let nps: Int
        let position: Int
    }

    private var timeline: [TimedSample] {
        var offsetMs = 0
        var lastPosition = samples.first?.position ?? 0
        var lastTime = 0
        var out: [TimedSample] = []
        for (i, s) in samples.enumerated() {
            if s.position != lastPosition {
                offsetMs += lastTime
                lastPosition = s.position
            }
            lastTime = s.timeMs
            out.append(TimedSample(id: i, seconds: Double(offsetMs + s.timeMs) / 1000, nps: s.nps, position: s.position))
        }
        return out
    }

    var body: some View {
        Chart(timeline) { s in
            LineMark(x: .value("Time", s.seconds), y: .value("nps", s.nps),
                     series: .value("Position", s.position))
                .foregroundStyle(Color.accentColor)
                .interpolationMethod(.monotone)
        }
        .chartXAxisLabel("seconds")
        .chartYAxisLabel("nodes / s")
    }
}

/// A tiny axis-less line for list rows.
struct Sparkline: View {
    let values: [Double]

    var body: some View {
        if values.count > 1 {
            Chart(Array(values.enumerated()), id: \.offset) { item in
                LineMark(x: .value("i", item.offset), y: .value("v", item.element))
                    .interpolationMethod(.monotone)
                    .foregroundStyle(Color.accentColor)
                AreaMark(x: .value("i", item.offset), y: .value("v", item.element))
                    .interpolationMethod(.monotone)
                    .foregroundStyle(.linearGradient(colors: [.accentColor.opacity(0.3), .clear],
                                                     startPoint: .top, endPoint: .bottom))
            }
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            .chartLegend(.hidden)
        } else {
            Color.clear
        }
    }
}
