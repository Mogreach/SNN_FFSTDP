import numpy as np
from scipy.interpolate import interp1d
# 假设能耗模型（单位：pJ）
ENERGY_PER_MAC = 3.2    # ANN中一次乘加
ENERGY_PER_ADD = 0.1    # SNN中一次加法（加权脉冲）
# 网络结构配置：每层是一个 dict
NETWORK = [
    {
        "name": "conv1",
        "type": "conv",
        "in_channels": 3,
        "out_channels": 40,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_size": 32 * 32,
        "output_size": 32 * 32,
        "input_feature": (3, 32, 32),
        "output_feature": (40, 32, 32),
        "timesteps": 20,
        "spike_in_count": int(3 * 32 * 32 * 0.1 * 20),
        "spike_out_count": int(40 * 32 * 32 * 0.1 * 20),
        "weight_shape": (40, 3, 3, 3)
    },
    {
        "name": "conv2",
        "type": "conv",
        "in_channels": 40,
        "out_channels": 80,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "input_size": 32 * 32,
        "output_size": 16 * 16,
        "input_feature": (40, 32, 32),
        "output_feature": (80, 16, 16),
        "timesteps": 20,
        "spike_in_count": int(40 * 32 * 32 * 0.1 * 20),
        "spike_out_count": int(80 * 16 * 16 * 0.1 * 20),
        "weight_shape": (80, 40, 3, 3)
    },
    {
        "name": "conv3",
        "type": "conv",
        "in_channels": 80,
        "out_channels": 120,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_size": 16 * 16,
        "output_size": 16 * 16,
        "input_feature": (80, 16, 16),
        "output_feature": (120, 16, 16),
        "timesteps": 20,
        "spike_in_count": int(80 * 16 * 16 * 0.1 * 20),
        "spike_out_count": int(120 * 16 * 16 * 0.1 * 20),
        "weight_shape": (120, 80, 3, 3)
    },
    {
        "name": "conv4",
        "type": "conv",
        "in_channels": 120,
        "out_channels": 240,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "input_size": 16 * 16,
        "output_size": 8 * 8,
        "input_feature": (120, 16, 16),
        "output_feature": (240, 8, 8),
        "timesteps": 20,
        "spike_in_count": int(120 * 16 * 16 * 0.1 * 20),
        "spike_out_count": int(240 * 8 * 8 * 0.1 * 20),
        "weight_shape": (240, 120, 3, 3)
    }
]

# NETWORK =[
#     {
#         "name": "conv1",
#         "type": "conv",
#         "in_channels": 1,
#         "out_channels": 40,
#         "kernel_size": 3,
#         "stride": 1,
#         "padding": 1,
#         "input_size": 784,
#         "output_size": 784,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 0,
#         "spike_out_count": 6760,
#         "weight_shape": (40, 1, 3, 3)
#     },
#     {
#         "name": "conv2",
#         "type": "conv",
#         "in_channels": 40,
#         "out_channels": 160,
#         "kernel_size": 3,
#         "stride": 2,
#         "padding": 1,
#         "input_size": 784,
#         "output_size": 196,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 6760,
#         "spike_out_count": 23455,
#         "weight_shape": (160, 40, 3, 3)
#     },
#     {
#         "name": "conv3",
#         "type": "conv",
#         "in_channels": 160,
#         "out_channels": 80,
#         "kernel_size": 3,
#         "stride": 1,
#         "padding": 1,
#         "input_size": 196,
#         "output_size": 196,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 23455,
#         "spike_out_count": 22212,
#         "weight_shape": (80, 160, 3, 3)
#     },
#     {
#         "name": "conv4",
#         "type": "conv",
#         "in_channels": 80,
#         "out_channels": 200,
#         "kernel_size": 3,
#         "stride": 2,
#         "padding": 1,
#         "input_size": 196,
#         "output_size": 49,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 22212,
#         "spike_out_count": 27617,
#         "weight_shape": (200, 80, 3, 3)
#     },
#     {
#         "name": "conv5",
#         "type": "conv",
#         "in_channels": 200,
#         "out_channels": 120,
#         "kernel_size": 3,
#         "stride": 1,
#         "padding": 1,
#         "input_size": 49,
#         "output_size": 49,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 27617,
#         "spike_out_count": 4957,
#         "weight_shape": (120, 200, 3, 3)
#     },
#     {
#         "name": "conv6",
#         "type": "conv",
#         "in_channels": 120,
#         "out_channels": 40,
#         "kernel_size": 3,
#         "stride": 1,
#         "padding": 1,
#         "input_size": 49,
#         "output_size": 49,
#         "input_feature": None,
#         "output_feature": None,
#         "timesteps": 10,
#         "spike_in_count": 4957,
#         "spike_out_count": 24529,
#         "weight_shape": (40, 120, 3, 3)
#     },
#         {
#         "name": "fc1",
#         "type": "fc",
#         "in_channels": None,
#         "out_channels": None,
#         "kernel_size": None,
#         "stride": None,
#         "padding": None,
#         "input_size": None,
#         "output_size": None,
#         "in_features": None,  # 自动推导
#         "out_features": 5000,
#         "timesteps": 10,
#         "spike_in_count": 5000,
#         "spike_out_count": 15000,
#         "weight_shape": (10, 1000)
#     }
# ]
# {
#     "name": "fc1",
#     "type": "fc",
#     "in_channels": None,
#     "out_channels": None,
#     "kernel_size": None,
#     "stride": None,
#     "padding": None,
#     "input_size": None,
#     "output_size": None,
#     "in_features": 1000,
#     "out_features": 10,
#     "timesteps": 10,
#     "spike_in_count": 5000,
#     "spike_out_count": 120,
#     "weight_shape": (10, 1000)
# }


# 原始数据（单位统一成 KB 和 pJ）
MEMORY_KB = np.array([8, 32, 1024])  # 内存容量，单位：KB
ACCESS_ENERGY_PJ = np.array([10, 20, 100])  # 对应能耗，单位：pJ
# 创建线性插值函数
INTERP_FUNC = interp1d(MEMORY_KB, ACCESS_ENERGY_PJ, kind='linear', fill_value='extrapolate')
