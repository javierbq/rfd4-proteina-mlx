// swift-tools-version: 6.0
import PackageDescription

// SPM has no subdirectory support, so this manifest lives at the repo ROOT.
//
// Two libraries, deliberately split:
//
//   RFD4Core  -- vocabulary, structure handling, and the featurizer. NO MLX.
//   RFD4MLX   -- the flow trunk, autoencoder decoder, and sampler. Needs MLX.
//
// The split is not cosmetic. Roughly 80% of this port is featurization, and the
// upstream suite that specifies it (233 ported tests across 44 files) asserts on
// index arrays and masks, not on GPU tensors. Keeping that work MLX-free means
// those tests build and run in seconds, on CI, with no Metal device and no
// 300 MB of checkouts -- which is the whole point of driving the port from the
// test suite.
//
// mlx-swift is pinned `exact: 0.31.6` to match MPNNKit, BoltzMLX and RFD3MLX.
// `exact:` has no consumer-side override, so if any one of the four bumps
// independently, resolution fails outright and RayMol will not build at all.
// Bump them together and re-tag.
let package = Package(
    name: "rfd4-proteina-mlx",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "RFD4Core", targets: ["RFD4Core"]),
        .library(name: "RFD4MLX", targets: ["RFD4MLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    ],
    targets: [
        .target(
            name: "RFD4Core",
            resources: [.process("Resources")]
        ),
        .target(
            name: "RFD4MLX",
            dependencies: [
                "RFD4Core",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "BenchTmp",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "RFD4MLXCLI",
            dependencies: ["RFD4Core", "RFD4MLX"]
        ),
        .testTarget(
            name: "RFD4CoreTests",
            dependencies: ["RFD4Core"],
            resources: [.process("Fixtures")]
        ),
        .testTarget(
            name: "RFD4MLXTests",
            dependencies: ["RFD4MLX"],
            resources: [.process("Fixtures")]
        ),
    ]
)
