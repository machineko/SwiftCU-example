import cxxCU
import SwiftCU
import cudaKernel

/// run with -Xswiftc -L/$(pwd)/Sources/cudaKernel/lib -Xswiftc -lcuADD -Xswiftc -DaddKernel
#if addKernel
    func addKernel() {
        let arraySize: Int = 32
        let arrayBytes: Int = arraySize * MemoryLayout<Float32>.size
        let deviceStatus = CUDevice(index: 0).setDevice()
        assert(deviceStatus, "Can't set device \(deviceStatus)")

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?

        var allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        assert(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")
        allocateStatus = bPointer.cudaMemoryAllocate(arrayBytes)
        assert(allocateStatus.isSuccessful, "Can't allocate memory for bPointer \(allocateStatus)")
        allocateStatus = cPointer.cudaMemoryAllocate(arrayBytes)
        assert(allocateStatus.isSuccessful, "Can't allocate memory for cPointer \(allocateStatus)")

        var (a, b) = (
            [Float32](repeating: 0.0, count: arraySize), [Float32](repeating: 0.0, count: arraySize)
        )
        for i in 0..<32 {
            a[i] = Float32(i)
            b[i] = Float32(i * 2)
        }
        var copyStatus = aPointer.cudaMemoryCopy(fromRawPointer: &a, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyHostToDevice)
        assert(copyStatus.isSuccessful, "Can't copy memory for aPointer \(copyStatus)")
        copyStatus = bPointer.cudaMemoryCopy(fromRawPointer: &b, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyHostToDevice)
        assert(copyStatus.isSuccessful, "Can't copy memory for bPointer \(copyStatus)")

        var kernelArgs: CUDAKernelArguments = CUDAKernelArguments()
        kernelArgs.addRawPointers([aPointer, bPointer, cPointer])

        var someValue = Int32(arraySize)
        kernelArgs.addPointer(&someValue)
        let threadsPerBlock = 32
        let blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock
        let blockDim = dim3(make_uint3(UInt32(threadsPerBlock), 1, 1))
        let gridDim = dim3(make_uint3(UInt32(blocksPerGrid), 1, 1))
        var syncStatus = cudaDeviceSynchronize()
        assert(syncStatus.asSwift.isSuccessful, "Can't sync device \(syncStatus)")
        let kernel = CUDAKernel.init(functionPointer: getKernelPointer(ADD_F32))
        let launchStatus = kernel.launch(arguments: kernelArgs, blockDim: blockDim, gridDim: gridDim)

        assert(launchStatus.isSuccessful, "Can't launch kernel \(launchStatus)")

        syncStatus = cudaDeviceSynchronize()
        assert(syncStatus.asSwift.isSuccessful, "Can't sync device \(syncStatus)")

        var outputData = [Float32](repeating: 0, count: arraySize)
        outputData.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyDeviceToHost)
            assert(outStatus.isSuccessful, "Can't copy memory from device \(syncStatus)")
        }
        defer {
            var deallocateStatus = aPointer.cudaAndHostDeallocate()
            assert(aPointer == nil)
            assert(deallocateStatus.isSuccessful, "Can't dealocate aPointer \(deallocateStatus)")
            deallocateStatus = bPointer.cudaAndHostDeallocate()
            assert(deallocateStatus.isSuccessful, "Can't dealocate bPointer \(deallocateStatus)")
            deallocateStatus = cPointer.cudaAndHostDeallocate()
            assert(deallocateStatus.isSuccessful, "Can't dealocate cPointer \(deallocateStatus)")
        }

        for i in 0..<32 {
            let expectedValue = a[i] + b[i]
            let cudaValue = outputData[i]
            assert(
                outputData[i] == expectedValue, "results of addition not equal to expected value \(outputData[i]) != \(expectedValue)"
            )
            print("CUDA output \(cudaValue), CPU output \(expectedValue)")
        }
    }
#endif