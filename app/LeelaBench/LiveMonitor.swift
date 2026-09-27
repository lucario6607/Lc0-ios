import Charts
import SwiftUI

/// Elapsed time, free memory and thermal state, refreshed every second.
struct RunStatsBar: View {
    @EnvironmentObject private var runner: EngineRunner
    /// Start of the whole run (a sweep spans several lc0 runs).
    var started: Date?

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            HStack(alignment: .top, spacing: 8) {
                stat("Elapsed", elapsed, color: .primary)
                stat("Free memory", ByteCountFormatter.string(
                        fromByteCount: Int64(os_proc_available_memory()), countStyle: .memory),
                     detail: lowest.map { "low \($0)" }, color: memoryColor)
                stat("Thermal", DeviceInfo.thermalState,
                     color: DeviceInfo.thermalState == "nominal" ? .primary : thermalColor(DeviceInfo.thermalState))
            }
        }
    }

    private var elapsed: String {
        guard let start = started ?? runner.startedAt else { return "0:00" }
        return Duration.seconds(Date().timeIntervalSince(start)).formatted(.time(pattern: .minuteSecond))
    }

    private var lowest: String? {
        runner.lowestAvailableMemory.map {
            ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory)
        }
    }

    private var memoryColor: Color {
        let freeMB = os_proc_available_memory() / 1_048_576
        return freeMB < 300 ? .red : freeMB < 800 ? .orange : .primary
    }

    private func stat(_ title: String, _ value: String, detail: String? = nil, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.headline.monospacedDigit()).foregroundStyle(color)
                .lineLimit(1).minimumScaleFactor(0.7)
            if let detail {
                Text(detail).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Free memory over the current run.
struct MemoryChart: View {
    @EnvironmentObject private var runner: EngineRunner

    var body: some View {
        Chart(runner.memorySamples) { sample in
            AreaMark(x: .value("s", sample.seconds), y: .value("MB", sample.freeMB))
                .foregroundStyle(.linearGradient(colors: [.green.opacity(0.35), .green.opacity(0.05)],
                                                 startPoint: .top, endPoint: .bottom))
            LineMark(x: .value("s", sample.seconds), y: .value("MB", sample.freeMB))
                .foregroundStyle(.green)
        }
        .chartYScale(domain: 0...max(runner.memorySamples.first?.freeMB ?? 1, 1))
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let mb = value.as(Int.self) { Text(mb >= 1024 ? String(format: "%.1fG", Double(mb) / 1024) : "\(mb)M") }
                }
            }
        }
        .chartXAxis(.hidden)
    }
}

/// The last part of the live output, kept scrolled to the bottom.
struct LogTail: View {
    @EnvironmentObject private var runner: EngineRunner

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(String(runner.log.suffix(8_000)))
                    .font(.caption2.monospaced())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                Color.clear.frame(height: 1).id("tail")
            }
            .onAppear { proxy.scrollTo("tail", anchor: .bottom) }
            .onChange(of: runner.log) { _ in proxy.scrollTo("tail", anchor: .bottom) }
        }
    }
}

/// Current thermal state and free memory, shown when nothing is running.
struct IdleStatusRow: View {
    var body: some View {
        TimelineView(.periodic(from: .now, by: 2)) { _ in
            HStack(spacing: 14) {
                Label(DeviceInfo.thermalState, systemImage: "thermometer.medium")
                    .foregroundStyle(DeviceInfo.thermalState == "nominal" ? .secondary : thermalColor(DeviceInfo.thermalState))
                Label(DeviceInfo.availableMemory + " free", systemImage: "memorychip")
                    .foregroundStyle(.secondary)
                if DeviceInfo.lowPowerMode {
                    Label("Low Power", systemImage: "battery.25").foregroundStyle(.orange)
                }
            }
            .font(.caption)
        }
    }
}
