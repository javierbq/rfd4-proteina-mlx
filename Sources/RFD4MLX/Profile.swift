import Foundation
import MLX

/// A section timer for finding where a run's time actually goes.
///
/// Two things make naive timing useless here and both are handled below.
///
/// MLX is **lazy**: a line that builds a graph returns immediately, so timing it
/// measures graph construction. Every section therefore ends with an `eval`
/// barrier, which is also why profiling is off by default — those barriers are
/// precisely what the sampler avoids between layers, and leaving them in would
/// change the thing being measured.
///
/// And a first pass is not representative: Metal compiles kernels on first use, so
/// the opening step of a run carries compilation that no later step pays. Warm-up
/// passes are excluded from the totals rather than averaged into them.
public enum Profile {
    public nonisolated(unsafe) static var enabled =
        ProcessInfo.processInfo.environment["RFD4_PROFILE"] != nil

    /// Passes to discard before counting, so Metal's kernel compilation does not
    /// land in the averages.
    public nonisolated(unsafe) static var warmupPasses = 2

    private nonisolated(unsafe) static var totals: [String: Double] = [:]
    private nonisolated(unsafe) static var counts: [String: Int] = [:]
    private nonisolated(unsafe) static var order: [String] = []
    private nonisolated(unsafe) static var passIndex = 0
    private nonisolated(unsafe) static let lock = NSLock()

    /// Time one section, forcing its result so the number means something.
    @discardableResult
    public static func section<T>(_ name: String, _ body: () -> T, force: (T) -> [MLXArray])
        -> T {
        guard enabled else { return body() }
        let start = DispatchTime.now().uptimeNanoseconds
        let out = body()
        MLX.eval(force(out))
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        record(name, elapsed)
        return out
    }

    public static func section(_ name: String, _ body: () -> MLXArray) -> MLXArray {
        section(name, body, force: { [$0] })
    }

    private static func record(_ name: String, _ seconds: Double) {
        lock.lock(); defer { lock.unlock() }
        guard passIndex >= warmupPasses else { return }
        if totals[name] == nil { order.append(name) }
        totals[name, default: 0] += seconds
        counts[name, default: 0] += 1
    }

    /// Mark the end of a trunk pass, so warm-up can be excluded.
    public static func endPass() {
        guard enabled else { return }
        lock.lock(); defer { lock.unlock() }
        passIndex += 1
    }

    public static func report() -> String {
        lock.lock(); defer { lock.unlock() }
        guard enabled, !order.isEmpty else { return "" }
        // Sections nest, so the sum of all of them is not the wall-clock. Share is
        // taken against the outermost section rather than against that sum, which
        // would double-count and read as a nonsense percentage.
        let root = order.first.flatMap { totals[$0] } ?? 1
        // `%s` in String(format:) takes a C string, not a Swift String — passing one
        // is undefined behaviour and segfaults. Build the header by padding instead.
        var lines = ["  " + "section".padding(toLength: 26, withPad: " ", startingAt: 0)
                     + "   total s   per call    share"]
        for name in order {
            let total = totals[name] ?? 0
            let n = counts[name] ?? 1
            lines.append(String(
                format: "  %@ %9.2f %10.4f %6.1f%%",
                name.padding(toLength: 26, withPad: " ", startingAt: 0),
                total, total / Double(n), 100 * total / root))
        }
        lines.append(String(format: "  (%d passes counted, %d discarded as warm-up)",
                            max(0, passIndex - warmupPasses), min(passIndex, warmupPasses)))
        return lines.joined(separator: "\n")
    }

    public static func reset() {
        lock.lock(); defer { lock.unlock() }
        totals = [:]; counts = [:]; order = []; passIndex = 0
    }
}
