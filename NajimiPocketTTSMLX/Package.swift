// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "NajimiPocketTTSMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "NajimiPocketTTSMLX", targets: ["NajimiPocketTTSMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.31.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "NajimiPocketTTSMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/NajimiPocketTTSMLX",
            resources: [
                .copy("Resources/default.metallib"),
            ]
        ),
    ]
)
