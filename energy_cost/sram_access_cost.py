import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 原始数据（单位统一成 KB 和 pJ）
MEMORY_KB = np.array([8, 32, 1024])  # 内存容量，单位：KB
ACCESS_ENERGY_PJ = np.array([10, 20, 100])  # 对应能耗，单位：pJ
# 创建线性插值函数
interp_func = interp1d(MEMORY_KB, ACCESS_ENERGY_PJ, kind='linear', fill_value='extrapolate')
def sram_access_cost(memory_size):
    """
    计算SRAM访问能耗
    :param memory_size: 内存容量，单位：KB
    :return: 返回对应的访问能耗，单位：pJ
    """
    access_cost = interp_func(memory_size)
    return access_cost


if __name__ == "__main__":
    # 例子：插值计算某些中间值
    test_sizes = [16, 64, 512]  # 待插值内存大小
    for size in test_sizes:
        energy = interp_func(size)
        print(f"{size}KB 的能耗估计为 {energy:.2f} pJ")

    # 可视化插值曲线
    x_dense = np.linspace(8, 1024, 500)
    y_interp = sram_access_cost(x_dense)

    plt.plot(MEMORY_KB , ACCESS_ENERGY_PJ, 'ro', label='原始数据')
    plt.plot(x_dense, y_interp, 'b--', label='线性插值')
    plt.xlabel('SRAM 容量 (KB)')
    plt.ylabel('访问能耗 (pJ)')
    plt.title('SRAM访问能耗线性插值拟合')
    plt.legend()
    plt.grid(True)
    plt.show()