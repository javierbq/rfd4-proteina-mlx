import Foundation
import MLX
import RFD4Core

/// Placeholder so the MLX target exists and `mlx-swift` resolution is proven from
/// the first commit, per the port playbook's Phase 0 ("confirm every dependency has
/// a real tag before you build anything on top"). The trunk, autoencoder decoder and
/// sampler land here in Phase 2; nothing in this file is load-bearing.
public enum RFD4MLX {
    /// The vocabulary is owned by `RFD4Core` and carries no MLX types. Re-exported
    /// here only so this target has a reason to link both and a build failure in
    /// either shows up immediately.
    public static var vocabularyTokenCount: Int { Vocabulary.unifiedAtom37.tokenCount }

    /// Smoke check that MLX is linked and can evaluate. Not a test of anything real.
    public static func mlxIsLive() -> Bool {
        let a = MLXArray([1, 2, 3] as [Int32])
        return (a + a).sum().item(Int32.self) == 12
    }
}
