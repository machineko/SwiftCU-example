import SwiftCU
import cxxCU

extension size_t {
    var asGB: Int {
        Int(round(Double(self) / (1024 * 1024 * 1024)))
    }
    var asMB: Int {
        Int(round(Double(self) / (1024 * 1024)))
    }
}


func getCUDAMemory() -> CUMemory {
    var memory = CUMemory()
    let updateStatus = memory.updateCUDAMemory()
    assert(updateStatus.isSuccessful)
    // print("GPU Free memory \(memory.free.asGB) GB | GPU Total memory \(memory.total.asGB) GB")
    return memory
}

func allocateMemory(allocationSize: Int) -> Bool {
    var cuPointer: UnsafeMutableRawPointer?
    print("trying to allocate \(allocationSize.asMB) MB")
    var memory = getCUDAMemory()
    print("GPU Free memory before allocation \(memory.free.asMB) MB")
    let status = cuPointer.cudaMemoryAllocate(allocationSize)
    assert(status.isSuccessful, "Can't allocate \(allocationSize) error \(status)")
    memory = getCUDAMemory()
    print("GPU Free memory after  allocation \(memory.free.asMB) MB")
    return status.isSuccessful
}