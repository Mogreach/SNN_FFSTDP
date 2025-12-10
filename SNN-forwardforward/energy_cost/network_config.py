import numpy as np
from scipy.interpolate import interp1d
# 假设能耗模型（单位：pJ）
ENERGY_PER_MAC = 3.1    # ANN中一次乘加
ENERGY_PER_ADD = 0.1    # SNN中一次加法（加权脉冲）
# 原始数据（单位统一成 KB 和 pJ）
MEMORY_KB = np.array([8, 32, 1024])  # 内存容量，单位：KB
ACCESS_ENERGY_PJ = np.array([10, 20, 100])  # 对应能耗，单位：pJ
# 创建线性插值函数
INTERP_FUNC = interp1d(MEMORY_KB, ACCESS_ENERGY_PJ, kind='linear', fill_value='extrapolate')
SRAM_SIZE = 4  # SRAM大小，单位：KB
# 网络结构配置：每层是一个 dict
timesteps = 20
spike_rate = 0.1


NETWORK ={
"VGG-16" :
    [
        # Block 1
        {
            "name": "conv1_1",
            "type": "conv",
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 32 * 32,
            "output_size": 32 * 32,
            "input_feature": (3, 32, 32),
            "output_feature": (64, 32, 32),
            "timesteps": timesteps,
            "spike_in_count": int(3 * 32 * 32 * spike_rate * timesteps),
            "spike_out_count": int(64 * 32 * 32 * spike_rate * timesteps),
            "weight_shape": (64, 3, 3, 3)
        },
        {
            "name": "conv1_2",
            "type": "conv",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 32 * 32,
            "output_size": 32 * 32,
            "input_feature": (64, 32, 32),
            "output_feature": (64, 32, 32),
            "timesteps": timesteps,
            "spike_in_count": int(64 * 32 * 32 * spike_rate * timesteps),
            "spike_out_count": int(64 * 32 * 32 * spike_rate * timesteps),
            "weight_shape": (64, 64, 3, 3)
        },

        # Block 2
        {
            "name": "conv2_1",
            "type": "conv",
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 16 * 16,
            "output_size": 16 * 16,
            "input_feature": (64, 16, 16),
            "output_feature": (128, 16, 16),
            "timesteps": timesteps,
            "spike_in_count": int(64 * 16 * 16 * spike_rate * timesteps),
            "spike_out_count": int(128 * 16 * 16 * spike_rate * timesteps),
            "weight_shape": (128, 64, 3, 3)
        },
        {
            "name": "conv2_2",
            "type": "conv",
            "in_channels": 128,
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 16 * 16,
            "output_size": 16 * 16,
            "input_feature": (128, 16, 16),
            "output_feature": (128, 16, 16),
            "timesteps": timesteps,
            "spike_in_count": int(128 * 16 * 16 * spike_rate * timesteps),
            "spike_out_count": int(128 * 16 * 16 * spike_rate * timesteps),
            "weight_shape": (128, 128, 3, 3)
        },

        # Block 3
        {
            "name": "conv3_1",
            "type": "conv",
            "in_channels": 128,
            "out_channels": 256,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 8 * 8,
            "output_size": 8 * 8,
            "input_feature": (128, 8, 8),
            "output_feature": (256, 8, 8),
            "timesteps": timesteps,
            "spike_in_count": int(128 * 8 * 8 * spike_rate * timesteps),
            "spike_out_count": int(256 * 8 * 8 * spike_rate * timesteps),
            "weight_shape": (256, 128, 3, 3)
        },
        {
            "name": "conv3_2",
            "type": "conv",
            "in_channels": 256,
            "out_channels": 256,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 8 * 8,
            "output_size": 8 * 8,
            "input_feature": (256, 8, 8),
            "output_feature": (256, 8, 8),
            "timesteps": timesteps,
            "spike_in_count": int(256 * 8 * 8 * spike_rate * timesteps),
            "spike_out_count": int(256 * 8 * 8 * spike_rate * timesteps),
            "weight_shape": (256, 256, 3, 3)
        },
        {
            "name": "conv3_3",
            "type": "conv",
            "in_channels": 256,
            "out_channels": 256,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 8 * 8,
            "output_size": 8 * 8,
            "input_feature": (256, 8, 8),
            "output_feature": (256, 8, 8),
            "timesteps": timesteps,
            "spike_in_count": int(256 * 8 * 8 * spike_rate * timesteps),
            "spike_out_count": int(256 * 8 * 8 * spike_rate * timesteps),
            "weight_shape": (256, 256, 3, 3)
        },

        # Block 4
        {
            "name": "conv4_1",
            "type": "conv",
            "in_channels": 256,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 4 * 4,
            "output_size": 4 * 4,
            "input_feature": (256, 4, 4),
            "output_feature": (512, 4, 4),
            "timesteps": timesteps,
            "spike_in_count": int(256 * 4 * 4 * spike_rate * timesteps),
            "spike_out_count": int(512 * 4 * 4 * spike_rate * timesteps),
            "weight_shape": (512, 256, 3, 3)
        },
        {
            "name": "conv4_2",
            "type": "conv",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 4 * 4,
            "output_size": 4 * 4,
            "input_feature": (512, 4, 4),
            "output_feature": (512, 4, 4),
            "timesteps": timesteps,
            "spike_in_count": int(512 * 4 * 4 * spike_rate * timesteps),
            "spike_out_count": int(512 * 4 * 4 * spike_rate * timesteps),
            "weight_shape": (512, 512, 3, 3)
        },
        {
            "name": "conv4_3",
            "type": "conv",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 4 * 4,
            "output_size": 4 * 4,
            "input_feature": (512, 4, 4),
            "output_feature": (512, 4, 4),
            "timesteps": timesteps,
            "spike_in_count": int(512 * 4 * 4 * spike_rate * timesteps),
            "spike_out_count": int(512 * 4 * 4 * spike_rate * timesteps),
            "weight_shape": (512, 512, 3, 3)
        },

        # Block 5
        {
            "name": "conv5_1",
            "type": "conv",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 2 * 2,
            "output_size": 2 * 2,
            "input_feature": (512, 2, 2),
            "output_feature": (512, 2, 2),
            "timesteps": timesteps,
            "spike_in_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "spike_out_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "weight_shape": (512, 512, 3, 3)
        },
        {
            "name": "conv5_2",
            "type": "conv",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 2 * 2,
            "output_size": 2 * 2,
            "input_feature": (512, 2, 2),
            "output_feature": (512, 2, 2),
            "timesteps": timesteps,
            "spike_in_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "spike_out_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "weight_shape": (512, 512, 3, 3)
        },
        {
            "name": "conv5_3",
            "type": "conv",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 2 * 2,
            "output_size": 2 * 2,
            "input_feature": (512, 2, 2),
            "output_feature": (512, 2, 2),
            "timesteps": timesteps,
            "spike_in_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "spike_out_count": int(512 * 2 * 2 * spike_rate * timesteps),
            "weight_shape": (512, 512, 3, 3)
        }
    ],
# ANN
# The Forward-Forward Algorithm: Some Preliminary Investigations
"FF":
    [
        {
            "name": "fc1",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 32 * 32 *3,
            "output_feature": 3072,
            "timesteps": 10,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 1024*3)
        },   
        {
            "name": "fc2",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 3072,
            "output_feature": 3072,
            "timesteps": 10,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 3072)
        }  
    ],
# ANN
"SymBa":
    [
        {
            "name": "fc1",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 32 * 32 * 3,
            "output_feature": 3072,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 1024 * 3)
        },   
        {
            "name": "fc2",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 3072,
            "output_feature": 3072,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 3072)
        },   
        {
            "name": "fc3",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 3072,
            "output_feature": 3072,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 3072)
        }    
    ],
# SNN
# Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm
"Ghader":
    [
        {
            "name": "fc1",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 32 * 32 *3,
            "output_feature": 2000,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (2000, 1024*3)
        },   
        {
            "name": "fc2",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 2000,
            "output_feature": 2000,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (2000, 2000)
        }  
    ],
# SNN
# CONTRASTIVE-SIGNAL-DEPENDENT PLASTICITY:  SELF-SUPERVISED LEARNING IN SPIKING NEURAL CIRCUITS 
"Ororbia":
    [
        {
            "name": "fc1",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 32 * 32 *3,
            "output_feature": 3072,
            "timesteps": 10,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 1024*3)
        },   
        {
            "name": "fc2",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 3072,
            "output_feature": 3072,
            "timesteps": 10,
            "spike_in_count": 0,
            "spike_out_count": 0,
            "weight_shape": (3072, 3072)
        }  
    ],
# ANN
# THE CASCADED FORWARD ALGORITHM FOR NEURAL NETWORK  TRAINING
"CaFo" :
    [
        {
            "name": "conv1",
            "type": "conv",
            "in_channels": 3,
            "out_channels": 40,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": timesteps * 10 * spike_rate,
            "weight_shape": (32, 3, 3, 3)
        },
        {
            "name": "conv2",
            "type": "conv",
            "in_channels": 32,
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 6760,
            "spike_out_count": 23455,
            "weight_shape": (128, 32, 3, 3)
        },
        {
            "name": "conv3",
            "type": "conv",
            "in_channels": 128,
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 23455,
            "spike_out_count": 22212,
            "weight_shape": (512, 128, 3, 3)
        }
    ],
# ANN

"CwComp" :
    [
        {
            "name": "conv1",
            "type": "conv",
            "in_channels": 3,
            "out_channels": 20,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": timesteps * 10 * spike_rate,
            "weight_shape": (20, 3, 3, 3)
        },
        {
            "name": "conv2",
            "type": "conv",
            "in_channels": 20,
            "out_channels": 80,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 6760,
            "spike_out_count": 23455,
            "weight_shape": (80, 20, 3, 3)
        },
        {
            "name": "conv3",
            "type": "conv",
            "in_channels": 80,
            "out_channels": 240,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 23455,
            "spike_out_count": 22212,
            "weight_shape": (240, 80, 3, 3)
        },
        {
            "name": "conv4",
            "type": "conv",
            "in_channels": 240,
            "out_channels": 480,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 23455,
            "spike_out_count": 22212,
            "weight_shape": (480, 240, 3, 3)
        },
        {
            "name": "fc1",
            "type": "fc",
            "in_channels": None,
            "out_channels": None,
            "kernel_size": None,
            "stride": None,
            "padding": None,
            "input_size": None,
            "output_size": None,
            "input_feature": 480 * 1024,
            "output_feature": 10,
            "timesteps": 10,
            "spike_in_count": 5000,
            "spike_out_count": 120,
            "weight_shape": (10, 480 * 1024)
        }
    ],
# SNN
"CSNN" :
    [
        {
            "name": "conv1",
            "type": "conv",
            "in_channels": 3,
            "out_channels": 40,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "input_size": 1024,
            "output_size": 1024,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 0,
            "spike_out_count": timesteps * 10 * spike_rate,
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
            "input_size": 1024,
            "output_size": 256,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 6760,
            "spike_out_count": 23455,
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
            "input_size": 256,
            "output_size": 256,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 23455,
            "spike_out_count": 22212,
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
            "input_size": 256,
            "output_size": 64,
            "input_feature": None,
            "output_feature": None,
            "timesteps": timesteps,
            "spike_in_count": 22212,
            "spike_out_count": 27617,
            "weight_shape": (240, 120, 3, 3)
        }
    ],
}
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
#     "input_feature": 1000,
#     "output_feature": 10,
#     "timesteps": 10,
#     "spike_in_count": 5000,
#     "spike_out_count": 120,
#     "weight_shape": (10, 1000)
# }

