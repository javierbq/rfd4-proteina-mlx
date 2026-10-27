import Foundation
import MLX
import MLXRandom
import Testing
@testable import RFD4MLX

/// The initial state and the per-step injection are both drawn here, so their
/// distribution is part of the model's behaviour, not an implementation detail.
/// A generator that is not standard normal changes the sample without failing any
/// parity test, because every parity test supplies its noise explicitly.
@Suite("noise sanity")
struct NoiseSanityTests {
    @Test("MLXRandom.normal is standard normal")
    func standardNormal() {
        MLXRandom.seed(0)
        let x = MLXRandom.normal([1, 20000, 3])
        let mean = x.mean().item(Float.self)
        let sd = MLX.sqrt(MLX.mean((x - x.mean()) * (x - x.mean()))).item(Float.self)
        print(String(format: "     MLXRandom.normal: mean %.4f  sd %.4f", mean, sd))
        #expect(abs(mean) < 0.05, "mean \(mean) is not ~0")
        #expect(abs(sd - 1) < 0.05, "sd \(sd) is not ~1")
    }
}
