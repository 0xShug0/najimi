// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "NajimiSileroVADMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "NajimiSileroVADMLX",
            targets: ["NajimiSileroVADMLX"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
    ],
    targets: [
        .target(
            name: "NajimiSileroVADMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/NajimiSileroVADMLX",
            resources: [
                .copy("Resources/default.metallib"),
            ]
        ),
    ]
)
