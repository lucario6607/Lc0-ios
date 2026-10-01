import Foundation
import UIKit

/// Redirects this process's stdout/stderr into a pipe so lc0's console output
/// can be shown in the app. Installed once; output is also echoed to the
/// original stderr so it still shows up in Xcode's console.
final class OutputCapture {
    static let shared = OutputCapture()

    private let pipe = Pipe()
    private var originalStderr: Int32 = -1
    private let lock = NSLock()
    private var handler: ((String) -> Void)?

    func install() {
        originalStderr = dup(STDERR_FILENO)
        setvbuf(stdout, nil, _IOLBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)
        let writeFD = pipe.fileHandleForWriting.fileDescriptor
        dup2(writeFD, STDOUT_FILENO)
        dup2(writeFD, STDERR_FILENO)
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self, !data.isEmpty else { return }
            if self.originalStderr >= 0 {
                data.withUnsafeBytes { _ = write(self.originalStderr, $0.baseAddress, data.count) }
            }
            let text = String(decoding: data, as: UTF8.self)
            self.lock.lock()
            let handler = self.handler
            self.lock.unlock()
            handler?(text)
        }
    }

    func setHandler(_ handler: ((String) -> Void)?) {
        lock.lock()
        self.handler = handler
        lock.unlock()
    }
}

/// Accumulates captured output off the main thread.
final class LogBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    private var file: FileHandle?

    func append(_ chunk: String) {
        lock.lock()
        text += chunk
        lock.unlock()
        writeToFile(chunk)
    }

    /// Mirror output to `url` as it arrives, so it survives the app being killed.
    func startFile(at url: URL, header: String) {
        FileManager.default.createFile(atPath: url.path, contents: Data(header.utf8))
        let handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
        lock.lock()
        file = handle
        lock.unlock()
    }

    func closeFile() {
        lock.lock()
        try? file?.close()
        file = nil
        lock.unlock()
    }

    func writeToFile(_ chunk: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let file else { return }
        try? file.write(contentsOf: Data(chunk.utf8))
        try? file.synchronize()
    }

    func drain() -> String {
        lock.lock()
        defer { text = ""; lock.unlock() }
        return text
    }
}

/// Everything known about a finished lc0 invocation.
struct RunOutcome {
    var output: String
    var exitCode: Int32
    var parsed: OutputParser
    var started: Date
    var duration: TimeInterval
    var thermalStart: String
    var thermalEnd: String
    var lowestFreeMemory: Int?
}

/// Runs lc0 in-process, one invocation at a time.
@MainActor
final class EngineRunner: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var log = ""
    /// Numbers parsed from the current (or last) run's output so far.
    @Published private(set) var parsed = OutputParser()
    @Published private(set) var startedAt: Date?
    /// Arguments of the current (or last) run, without argv[0].
    @Published private(set) var arguments: [String] = []
    /// Output of a run that never finished — the app was killed mid-run.
    @Published var crashedRunLog: String?
    @Published private(set) var lowestAvailableMemory: Int?

    nonisolated static let exitMarker = "\u{1}LC0_EXIT "
    private static let maxLogLength = 400_000
    private static let runLogURL = FileLocations.documents.appendingPathComponent("last-run.log")
    private static let runInProgressKey = "runInProgress"

    private let buffer = LogBuffer()
    private var flushTimer: Timer?
    private var memoryTimer: Timer?
    private var memoryWarning: NSObjectProtocol?
    private var runLog = ""
    private var thermalStart = ""
    private var completion: ((RunOutcome) -> Void)?

    init() {
        let buffer = self.buffer
        OutputCapture.shared.setHandler { text in buffer.append(text) }

        if UserDefaults.standard.bool(forKey: Self.runInProgressKey) {
            UserDefaults.standard.set(false, forKey: Self.runInProgressKey)
            let text = (try? String(contentsOf: Self.runLogURL, encoding: .utf8)) ?? "(no log was saved)"
            crashedRunLog = String(text.suffix(100_000))
        }
    }

    nonisolated private static func availableMemoryMB() -> String {
        "\(os_proc_available_memory() / 1_048_576) MB"
    }

    /// Runs `lc0 <args...>` and calls `completion` when lc0 returns.
    func run(_ args: [String], engine: ChessEngine = .lc0, completion: @escaping (RunOutcome) -> Void) {
        guard !isRunning else { return }
        isRunning = true
        arguments = args
        log = "$ lc0 " + args.joined(separator: " ") + "\n"
        runLog = ""
        parsed = OutputParser()
        startedAt = Date()
        thermalStart = DeviceInfo.thermalState
        self.completion = completion
        UIApplication.shared.isIdleTimerDisabled = true

        buffer.startFile(at: Self.runLogURL, header: """
            LeelaBench run \(Date().formatted(.iso8601))
            device: \(DeviceInfo.summary), RAM \(DeviceInfo.physicalMemory), thermal \(DeviceInfo.thermalState)
            available memory at start: \(Self.availableMemoryMB())
            \(log)
            """)
        UserDefaults.standard.set(true, forKey: Self.runInProgressKey)
        lowestAvailableMemory = os_proc_available_memory()

        flushTimer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.flush() }
        }
        // Memory trail in the saved log only: if iOS kills the app for memory,
        // the last lines show how close it got.
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                let available = os_proc_available_memory()
                self.lowestAvailableMemory = min(self.lowestAvailableMemory ?? available, available)
                self.buffer.writeToFile("[LeelaBench] available memory: \(available / 1_048_576) MB\n")
            }
        }
        memoryWarning = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main
        ) { [buffer] _ in
            buffer.writeToFile("[LeelaBench] iOS memory warning, available: \(Self.availableMemoryMB())\n")
        }

        let argv0 = Bundle.main.executablePath ?? "lc0"
        let thread = Thread {
            var cArgs: [UnsafeMutablePointer<CChar>?] = ([argv0] + args).map { strdup($0) }
            cArgs.append(nil)
            let argc = Int32(cArgs.count - 1)
            let code: Int32
            switch engine {
            case .lc0:
                code = cArgs.withUnsafeMutableBufferPointer { buffer in
                    buffer.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self,
                                                          capacity: buffer.count) { lc0_main(argc, $0) }
                }
            case .stockfish:
                code = stockfish_main_c(argc, &cArgs)
            }
            for arg in cArgs { free(arg) }
            fflush(stderr)
            fputs("\n\(EngineRunner.exitMarker)\(code)\n", stdout)
            fflush(stdout)
        }
        // lc0 expects a main-thread-sized stack.
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.name = "lc0"
        thread.start()
    }

    private func flush() {
        let text = buffer.drain()
        guard !text.isEmpty else { return }

        var chunk = text.replacingOccurrences(
            of: "\u{1B}\\[[0-9;]*m", with: "", options: .regularExpression)
        var exitCode: Int32?
        if let range = chunk.range(of: Self.exitMarker) {
            let rest = chunk[range.upperBound...]
            exitCode = Int32(rest.prefix { $0 != "\n" }) ?? -1
            chunk = String(chunk[..<range.lowerBound])
        }

        runLog += chunk
        parsed.feed(chunk)
        log += chunk
        if log.count > Self.maxLogLength {
            log = "…\n" + log.suffix(Self.maxLogLength / 2)
        }

        if let exitCode { finish(exitCode) }
    }

    private func finish(_ code: Int32) {
        flushTimer?.invalidate()
        flushTimer = nil
        memoryTimer?.invalidate()
        memoryTimer = nil
        if let memoryWarning { NotificationCenter.default.removeObserver(memoryWarning) }
        memoryWarning = nil
        buffer.writeToFile("\n[lc0 exited with code \(code)]\n")
        buffer.closeFile()
        UserDefaults.standard.set(false, forKey: Self.runInProgressKey)
        isRunning = false
        UIApplication.shared.isIdleTimerDisabled = false
        log += "\n[lc0 exited with code \(code)]\n"
        parsed.finish()
        let started = startedAt ?? Date()
        let outcome = RunOutcome(
            output: runLog, exitCode: code, parsed: parsed, started: started,
            duration: Date().timeIntervalSince(started), thermalStart: thermalStart,
            thermalEnd: DeviceInfo.thermalState, lowestFreeMemory: lowestAvailableMemory)
        let completion = self.completion
        self.completion = nil
        completion?(outcome)
    }
}
