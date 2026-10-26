import Testing
@testable import RFD4MLX

@Suite("MLX link smoke")
struct SmokeTests {
    @Test("mlx-swift is linked and evaluates")
    func mlxLinks() {
        #expect(RFD4MLX.mlxIsLive())
    }

    @Test("core vocabulary is visible from the MLX target")
    func coreVisible() {
        #expect(RFD4MLX.vocabularyTokenCount == 147)
    }
}
