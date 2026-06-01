import subprocess, numpy as np, os, sys

def vk_gemm(A, B):
    M, K = A.shape
    N = B.shape[0]
    A.astype(np.float32).tofile("/data/local/tmp/A.bin")
    B.astype(np.float32).tofile("/data/local/tmp/B.bin")
    subprocess.run(["/data/local/tmp/vk_gemm_test"], cwd="/data/local/tmp", capture_output=True)
    return np.fromfile("/data/local/tmp/C.bin", dtype=np.float32).reshape(M, N)

# Test
A = np.eye(4, dtype=np.float32)
B = np.ones((4, 4), dtype=np.float32)
C = vk_gemm(A, B)
print("GPU:", C)
print("CPU:", A @ B.T)
