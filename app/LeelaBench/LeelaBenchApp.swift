import SwiftUI

@main
struct LeelaBenchApp: App {
    @StateObject private var runner: EngineRunner
    @StateObject private var nets = NetStore()
    @StateObject private var results = ResultsStore()

    init() {
        // Must happen before lc0 first writes anything.
        OutputCapture.shared.install()
        _runner = StateObject(wrappedValue: EngineRunner())
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                BenchmarkView()
                    .tabItem { Label("Benchmark", systemImage: "speedometer") }
                ResultsView()
                    .tabItem { Label("Results", systemImage: "chart.xyaxis.line") }
                NetworksView()
                    .tabItem { Label("Networks", systemImage: "square.stack.3d.up") }
            }
            .environmentObject(runner)
            .environmentObject(nets)
            .environmentObject(results)
        }
    }
}
