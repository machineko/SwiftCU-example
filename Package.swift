// swift-tools-version: 6.0

import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuIncludePath = "-I\(cuPath)\\include"
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCU-example",
    dependencies: 
    [
        .package(path: "/home/machineko/swiftStuff/SwiftCU")
    ],
    targets: [
        .target(
            name: "cudaKernel",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ]
        ),
        .executableTarget(
            name: "SwiftCU-example",
            dependencies: [
                "cudaKernel",
                .product(name: "SwiftCU", package: "SwiftCU"),
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    ["-Xcc", cuIncludePath]
                )
            ]
        ),
    ]
)
