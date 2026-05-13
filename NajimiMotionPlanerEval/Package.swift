// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "NajimiMotionPlanerEval",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "PlannerEvalCore",
            targets: ["PlannerEvalCore"]
        ),
        .executable(
            name: "najimi-motion-planner-eval",
            targets: ["NajimiMotionPlannerEvalCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "PlannerEvalCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            resources: [
                .copy("Resources/default.metallib"),
            ]
        ),
        .executableTarget(
            name: "NajimiMotionPlannerEvalCLI",
            dependencies: [
                "PlannerEvalCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
