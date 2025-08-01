import torch
import time
import tqdm

# 確認 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("GPU not available. Please ensure CUDA is enabled and GPU is accessible.")
print(f"Using device: {device}")

# 測試參數
matrix_size = 1000  # 單個矩陣大小
num_matrices = 1000  # 同時運算的矩陣對數
num_runs = 10  # 總運行次數

# 初始化多個矩陣
matrices_a = [torch.randn(matrix_size, matrix_size, device=device) for _ in range(num_matrices)]
matrices_b = [torch.randn(matrix_size, matrix_size, device=device) for _ in range(num_matrices)]

total_time = 0

# 開始測試
for _ in tqdm.tqdm(range(num_runs), desc="GPU Load Test"):
    start_time = time.time()

    # 並行運算多個矩陣乘法
    results = [torch.matmul(a, b) for a, b in zip(matrices_a, matrices_b)]

    # 同步等待所有 GPU 任務完成
    torch.cuda.synchronize()
    end_time = time.time()

    # 記錄運行時間
    total_time += (end_time - start_time)


# 計算平均時間
average_time = total_time / num_runs
print(f"Average matrix multiplication with {num_matrices} pairs of {matrix_size}x{matrix_size} matrices took {average_time:.2f} seconds over {num_runs} runs")
