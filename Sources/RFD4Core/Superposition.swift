import Foundation

/// Rigid-body superposition by Horn's quaternion method.
///
/// ## Why this is needed at all
///
/// A conditioned design does not come back in the frame it was given. `DesignProblem`
/// enters the target as a `CoordinateGroup`, which is centred on its own centroid and
/// then **randomly rotated** — the model is told the target's shape and must place it.
/// So the generated complex is in an arbitrary frame, and putting the design back beside
/// the structure it was designed against means recovering the rigid motion that maps the
/// REBUILT target onto the SUPPLIED one, then carrying the designed chain through it.
///
/// Measured on the albumin campaign: a design's own target chain starts at
/// (15.0, 8.3, 7.8) where the input target's first CA is at (33.5, 26.4, 20.3). Not a
/// translation — a rotation as well — which is why this is a full superposition and not
/// the mean-offset recovery that suffices for a sampler working in a translated frame.
///
/// ## Why not LAPACK
///
/// Accelerate's LAPACK symbol set changed with `ACCELERATE_NEW_LAPACK`, and binding it
/// from Swift is a portability wager taken for a 4×4 eigenproblem. A cyclic Jacobi sweep
/// is exact to ~1e-14 here (``selfTestResidual()``) and has no external surface at all.
public enum Superposition {

    /// A rigid motion: centre on the mobile centroid, rotate, translate to the
    /// reference centroid.
    public struct Transform: Sendable {
        /// Row-major 3×3, applied AFTER centring.
        public let rotation: [[Double]]
        public let mobileCentroid: SIMD3<Double>
        public let referenceCentroid: SIMD3<Double>

        public init(rotation: [[Double]], mobileCentroid: SIMD3<Double>,
                    referenceCentroid: SIMD3<Double>) {
            self.rotation = rotation
            self.mobileCentroid = mobileCentroid
            self.referenceCentroid = referenceCentroid
        }

        public func apply(_ p: SIMD3<Double>) -> SIMD3<Double> {
            let c = p - mobileCentroid
            let x = rotation[0][0] * c.x + rotation[0][1] * c.y + rotation[0][2] * c.z
            let y = rotation[1][0] * c.x + rotation[1][1] * c.y + rotation[1][2] * c.z
            let z = rotation[2][0] * c.x + rotation[2][1] * c.y + rotation[2][2] * c.z
            return SIMD3<Double>(x, y, z) + referenceCentroid
        }

        /// The same motion applied to a `Vec3`, which is what the structure types carry.
        public func apply(_ p: Vec3) -> Vec3 {
            let out = apply(SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)))
            return Vec3(Float(out.x), Float(out.y), Float(out.z))
        }
    }

    /// The transform that best maps `mobile` onto `reference`, plus the RMSD it achieves.
    ///
    /// Counts must match; the two are paired index-wise, and pairing them correctly is
    /// the caller's problem. `nil` for mismatched counts or fewer than three points —
    /// three is the floor at which a rotation is determined at all, and below it the
    /// "best fit" is a family rather than an answer.
    public static func fit(mobile: [SIMD3<Double>], reference: [SIMD3<Double>])
        -> (transform: Transform, rmsd: Double)? {
        guard mobile.count == reference.count, mobile.count >= 3 else { return nil }
        let n = Double(mobile.count)
        let cm = mobile.reduce(SIMD3<Double>.zero, +) / n
        let cr = reference.reduce(SIMD3<Double>.zero, +) / n
        let p = mobile.map { $0 - cm }
        let q = reference.map { $0 - cr }

        var s = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        for k in 0 ..< p.count {
            let a = p[k], b = q[k]
            s[0][0] += a.x * b.x; s[0][1] += a.x * b.y; s[0][2] += a.x * b.z
            s[1][0] += a.y * b.x; s[1][1] += a.y * b.y; s[1][2] += a.y * b.z
            s[2][0] += a.z * b.x; s[2][1] += a.z * b.y; s[2][2] += a.z * b.z
        }

        // Horn's key matrix: the eigenvector of its largest eigenvalue IS the optimal
        // rotation as a quaternion, and the eigenvalue itself gives the RMSD without
        // applying it.
        let n4: [[Double]] = [
            [s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
            [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
            [s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]],
            [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]],
        ]

        let (values, vectors) = jacobiEigen(n4)
        var best = 0
        for i in 1 ..< 4 where values[i] > values[best] { best = i }
        let lambda = values[best]
        let quat = (0 ..< 4).map { vectors[$0][best] }   // column `best`

        let w = quat[0], x = quat[1], y = quat[2], z = quat[3]
        let rotation: [[Double]] = [
            [w*w + x*x - y*y - z*z, 2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),         w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),         w*w - x*x - y*y + z*z],
        ]

        let sumSq = p.reduce(0.0) { $0 + ($1 * $1).sum() }
                  + q.reduce(0.0) { $0 + ($1 * $1).sum() }
        // CLAMPED, and not defensively: at a perfect fit the two terms cancel to a value
        // that lands a few ulps below zero, and `sqrt` of that is NaN — which downstream
        // reads as a failed run rather than a perfect one.
        let rmsd = (Swift.max(0, sumSq - 2 * lambda) / n).squareRoot()
        return (Transform(rotation: rotation, mobileCentroid: cm, referenceCentroid: cr),
                rmsd)
    }

    /// RMSD between two already-paired point sets with NO fitting.
    ///
    /// The thing to measure when the question is "did it land where it was put", as
    /// opposed to "is this the same shape" — fitting first answers the second question
    /// and silently discards the first.
    public static func rmsdInPlace(_ a: [SIMD3<Double>], _ b: [SIMD3<Double>]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var acc = 0.0
        for i in 0 ..< a.count { let d = a[i] - b[i]; acc += (d * d).sum() }
        return (acc / Double(a.count)).squareRoot()
    }

    /// Recover a known rotation and report the error, so the solver can be checked
    /// without a test target importing its internals.
    ///
    /// Exposed rather than left to the test suite because this is the one routine whose
    /// silent failure mode is a plausible wrong answer: a design placed by a bad rotation
    /// is a well-formed complex in the wrong pose, and nothing downstream can tell.
    public static func selfTestResidual() -> Double {
        let reference: [SIMD3<Double>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 2, 0), SIMD3(0, 0, 3),
            SIMD3(1, 1, 1), SIMD3(-2, 1, 0.5), SIMD3(3, -1, 2),
        ]
        // 30 degrees about z, then a translation. Recovering this exactly is the test.
        let c = (3.0).squareRoot() / 2, s = 0.5
        let mobile = reference.map { p in
            SIMD3<Double>(c * p.x - s * p.y + 10, s * p.x + c * p.y - 4, p.z + 7)
        }
        guard let (transform, _) = fit(mobile: mobile, reference: reference) else {
            return .infinity
        }
        var worst = 0.0
        for (m, r) in zip(mobile, reference) {
            let d = transform.apply(m) - r
            worst = Swift.max(worst, (d * d).sum().squareRoot())
        }
        return worst
    }

    /// Cyclic Jacobi for a small symmetric matrix. Returns (eigenvalues, eigenvectors
    /// as COLUMNS).
    private static func jacobiEigen(_ input: [[Double]]) -> ([Double], [[Double]]) {
        let n = input.count
        var a = input
        var v = (0 ..< n).map { i in (0 ..< n).map { j in i == j ? 1.0 : 0.0 } }
        for _ in 0 ..< 100 {
            var off = 0.0
            for i in 0 ..< n { for j in (i + 1) ..< n { off += a[i][j] * a[i][j] } }
            if off < 1e-24 { break }
            for p in 0 ..< (n - 1) {
                for q in (p + 1) ..< n where abs(a[p][q]) > 1e-18 {
                    let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    let t = (theta >= 0 ? 1.0 : -1.0)
                        / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c
                    for k in 0 ..< n {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0 ..< n {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0 ..< n {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        return ((0 ..< n).map { a[$0][$0] }, v)
    }
}
