import numpy as np

# 网络结构
NETWORK =[
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
        "spike_in_count": 0,
        "spike_out_count": None,  # to be filled based on timesteps and spike_rate
        "weight_shape": (40, 3, 3, 3)
    },
    {
        "name": "conv2",
        "type": "conv",
        "in_channels": 40,
        "out_channels": 120,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "input_size": 1024,
        "output_size": 256,
        "spike_in_count": 6760,
        "spike_out_count": 23455,
        "weight_shape": (120, 40, 3, 3)
    },
    {
        "name": "conv3",
        "type": "conv",
        "in_channels": 120,
        "out_channels": 80,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_size": 256,
        "output_size": 256,
        "spike_in_count": 23455,
        "spike_out_count": 22212,
        "weight_shape": (80, 120, 3, 3)
    },
    {
        "name": "conv4",
        "type": "conv",
        "in_channels": 80,
        "out_channels": 200,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "input_size": 256,
        "output_size": 64,
        "spike_in_count": 22212,
        "spike_out_count": 27617,
        "weight_shape": (200, 80, 3, 3)
    },
    {
        "name": "conv5",
        "type": "conv",
        "in_channels": 200,
        "out_channels": 80,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_size": 64,
        "output_size": 64,
        "spike_in_count": 27617,
        "spike_out_count": 4957,
        "weight_shape": (80, 200, 3, 3)
    }
]

# 超参数
timesteps = 20
spike_rate = 0.1

# 硬件能耗模型（单位：pJ）
E_MAC = 0.1     # 每次突触加法操作
E_WMEM = 6      # 每次权重访问（SRAM）
E_AMEM = 2      # 每次激活访问（SRAM）

# 计算推理能耗
def estimate_snn_energy(network, timesteps, spike_rate):
    total_energy = 0
    energy_breakdown = []
    for i, layer in enumerate(network):
        # 填充 spike_out_count for conv1
        if layer["name"] == "conv1":
            layer["spike_out_count"] = timesteps * 10 * spike_rate

        spike_in = layer["spike_in_count"]
        spike_out = layer["spike_out_count"]
        out_channels, in_channels, k_h, k_w = layer["weight_shape"]
        kernel_ops_per_spike = in_channels * k_h * k_w

        # MACs = 每个输入脉冲产生这么多MAC操作（忽略零权重）
        mac_ops = spike_in * kernel_ops_per_spike

        # 能耗计算
        mac_energy = mac_ops * E_MAC
        wmem_energy = out_channels * in_channels * k_h * k_w * E_WMEM  # 假设每层T个时间步只加载一次权重
        amem_energy = (spike_in + spike_out) * E_AMEM

        layer_energy = mac_energy + wmem_energy + amem_energy
        total_energy += layer_energy

        energy_breakdown.append({
            "layer": layer["name"],
            "mac_energy": mac_energy,
            "wmem_energy": wmem_energy,
            "amem_energy": amem_energy,
            "total_energy": layer_energy
        })

    return total_energy, energy_breakdown

total_energy, breakdown = estimate_snn_energy(NETWORK, timesteps, spike_rate)
print(total_energy/1e9)
