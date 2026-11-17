import Foundation

/// What one generation will peak at, and the largest one that fits a given ceiling.
///
/// This exists so a HOST can refuse an over-large design before touching the GPU. That
/// is not a nicety: when MLX cannot wire what it asks Metal for, it throws
/// `std::runtime_error` from inside a command-buffer completion handler, on a thread
/// with no handler on its stack. The process reaches `std::terminate` and dies with
/// SIGABRT — a `do`/`catch` around `generate` does not see it. In an application host
/// that costs the user their whole session, so refusing the input is the only defence
/// available in-process.
///
/// ## The curve lives here, once
///
/// `validation/benchmark/peak-memory.md` has the measurement and the method. The fit is
/// quadratic in TOKEN count, which is what the pair track implies — it is `[n, n, c]`
/// and every triangle operation reads and rewrites it.
///
/// Token count, not atom count, and the distinction matters when porting policy from a
/// host that sizes RFdiffusion3: there a design is sized in atoms. Here one token is one
/// residue, target or designed, so `tokens = target residues + designed residues` and
/// nothing has to count atoms to ask this question.
///
/// ## Why a second fit is not allowed
///
/// A host that copies these coefficients gets a second fit that drifts from this one the
/// first time the trunk's memory profile changes — and it is already scheduled to change,
/// because `validation/benchmark/peak-memory.md` records which commit to revert if memory
/// ever matters more than speed. A copied constant would then refuse designs that fit.
/// Hosts should carry POLICY (what fraction of a machine one design may claim) and call
/// in here for arithmetic.
///
/// ## Validated end to end, not only on the bench
///
/// The coefficients come from `BatchBench`, which builds the real trunk types against a
/// synthetic checkpoint. The obvious objection is that a bench is not a generation: a
/// generation also holds the real weights, runs 60 sampling steps and carries the
/// sampler's state. Checked rather than argued — the 100-design albumin campaign
/// (`~/albumin-run/designs/stats.csv`, fp32, 210–230 tokens) reports a high-water mark of
/// **2.79 GB**, against this curve's 2.84 GB at its largest design. 1.7% high, which is
/// the right direction for a guard and within the fit's own error.
public enum RFD4Budget {

    /// Which precision the trunk computes in. The peak differs by exactly 2.000x, so
    /// this is not a correction factor applied to one curve — it is two measured fits
    /// that happen to stand in that ratio, which is itself the check that nothing is
    /// left in float32 inside the trunk.
    public enum Precision: String, Sendable {
        case fp32, fp16

        /// `(a, b, c)` of `peak(n) = a·n² + b·n + c`, in GIGABYTES.
        var coefficients: (a: Double, b: Double, c: Double) {
            switch self {
            case .fp32: return (3.12637e-05, -1.62935e-03, 1.559)
            case .fp16: return (1.56318e-05, -8.14674e-04, 0.780)
            }
        }
    }

    /// What ships, and therefore what a host's guard should ask about unless it has
    /// deliberately set `RFD4_DTYPE=fp32`. See `validation/quality/fp16.md`.
    public static let shippingPrecision: Precision = .fp16

    /// Bytes per gigabyte as the fit means it: 1e9, not 2^30.
    ///
    /// The coefficients were fitted against `MLX.GPU.peakMemory / 1e9`, so converting
    /// with 2^30 here would shrink every prediction by 7.4% — in the UNSAFE direction,
    /// since a guard that under-predicts admits a design that aborts the process.
    private static let bytesPerGB = 1_000_000_000.0

    /// Predicted peak GPU allocation for a generation of `tokens` tokens, in bytes.
    ///
    /// Clamped at zero below, because the quadratic's vertex is at a few dozen tokens
    /// and a tiny system would otherwise predict a negative peak. Nothing real is that
    /// small — the smallest useful design is tens of residues — but a guard that can
    /// return a negative budget is a guard with a sign bug waiting in it.
    public static func predictedPeakBytes(
        tokens: Int, precision: Precision = shippingPrecision
    ) -> Int {
        let (a, b, c) = precision.coefficients
        let n = Double(max(0, tokens))
        return Int(max(0, (a * n * n + b * n + c) * bytesPerGB))
    }

    /// The largest token count whose predicted peak fits `budgetBytes`.
    ///
    /// Solved rather than searched: `a·n² + b·n + (c − budget) = 0` has one positive
    /// root for every budget above the vertex, and `floor` of it is the answer. Returns
    /// 0 when even an empty system would not fit, which is a real answer on a machine
    /// too small to run this at all rather than a sentinel.
    public static func maxTokens(
        budgetBytes: Int, precision: Precision = shippingPrecision
    ) -> Int {
        let (a, b, c) = precision.coefficients
        let budget = Double(budgetBytes) / bytesPerGB
        let discriminant = b * b - 4 * a * (c - budget)
        guard discriminant > 0 else { return 0 }
        let root = (-b + discriminant.squareRoot()) / (2 * a)
        guard root.isFinite, root > 0 else { return 0 }
        // `floor`, then verified against `predictedPeakBytes` rather than trusted. The
        // two are the same arithmetic in opposite directions, and float rounding at the
        // boundary can leave the root's floor one token over — which would make the
        // guard's own advice ("at most N tokens fit") name a size it would then refuse.
        var n = Int(root.rounded(.down))
        while n > 0 && predictedPeakBytes(tokens: n, precision: precision) > budgetBytes {
            n -= 1
        }
        return n
    }
}
