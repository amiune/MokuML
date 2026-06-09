// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MokuML",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MokuPredictor", targets: ["MokuPredictor"])
    ],
    targets: [
        .executableTarget(
            name: "MokuPredictor",
            path: "app/MokuPredictor"
        )
    ]
)

