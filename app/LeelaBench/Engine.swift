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

    func append(_ chunk: String) {
        lock.lock()
        text += chunk
        lock.unlock()
    }

    func drain() -> String {
        lock.lock()
        defer { text = ""; lock.unlock() }
        return text
    }
}

/// Runs lc0 in-process, one invocation at a time.
@MainActor
final class EngineRunner: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var log = ""

    nonisolated static let exitMarker = "\u{1}LC0_EXIT "
    private static let maxLogLength = 400_000

    private let buffer = LogBuffer()
    private var flushTimer: Timer?
    private var runLog = ""
    private var completion: ((String, Int32) -> Void)?

    init() {
        let buffer = self.buffer
        OutputCapture.shared.setHandler { text in buffer.append(text) }
    }

    /// Runs `lc0 <args...>`. `completion` gets the full output of this run and
    /// lc0's return code.
    func run(_ args: [String], completion: @escaping (String, Int32) -> Void) {
        guard !isRunning else { return }
        isRunning = true
        log = "$ lc0 " + args.joined(separator: " ") + "\n"
        runLog = ""
        self.completion = completion
        UIApplication.shared.isIdleTimerDisabled = true

        flushTimer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.flush() }
        }

        let argv0 = Bundle.main.executablePath ?? "lc0"
        let thread = Thread {
            var cArgs: [UnsafePointer<CChar>?] = ([argv0] + args).map { UnsafePointer(strdup($0)) }
            cArgs.append(nil)
            let code = lc0_main(Int32(cArgs.count - 1), &cArgs)
            for arg in cArgs { free(UnsafeMutableRawPointer(mutating: arg)) }
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
        log += chunk
        if log.count > Self.maxLogLength {
            log = "…\n" + log.suffix(Self.maxLogLength / 2)
        }

        if let exitCode { finish(exitCode) }
    }

    private func finish(_ code: Int32) {
        flushTimer?.invalidate()
        flushTimer = nil
        isRunning = false
        UIApplication.shared.isIdleTimerDisabled = false
        log += "\n[lc0 exited with code \(code)]\n"
        let completion = self.completion
        self.completion = nil
        completion?(runLog, code)
    }
}
