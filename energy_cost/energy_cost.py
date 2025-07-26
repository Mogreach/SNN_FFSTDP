"""
====================================================================
File          : energy_cost.py
Description   : 计算SNN、ANN的推理能耗。目前支持卷积层和全连接层的推理计算能耗，修改NETWORK可以添加更多层。
                主函数代码根据network_config计算相同结构下ANN和SNN的能耗，输出结果。
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-06-03
contact       : 1245598043@qq.com
License       : 
====================================================================
"""
import math
import matplotlib.pyplot as plt
import csv
from FNN_cost import FNNEnergyCost
from SNN_cost import SNNEnergyCost
from network_config import *
def conv_output_size(input_size, kernel_size, stride, padding):
    h = math.sqrt(input_size)
    o = math.floor((h - kernel_size + 2 * padding) / stride) + 1
    return o**2

def main():
    layer_names = []
    spike_in_rate_list = []
    spike_out_rate_list = []
    energy_ann_list = []
    energy_snn_list = []

    print("===== 各层详细能耗分析 =====")
    prev_output_size = None
    prev_out_channels = None

    for i, layer in enumerate(NETWORK):
        layer_type = layer['type']
        in_channels = layer.get('in_channels')
        out_channels = layer.get('out_channels')
        kernel_size = layer.get('kernel_size')
        stride = layer.get('stride')
        padding = layer.get('padding')
        input_size = layer.get('input_size', prev_output_size)
        output_size = conv_output_size(input_size, kernel_size, stride, padding) if layer_type == 'conv' else layer.get('output_size')
        input_feature = layer.get('in_features')
        output_feature = layer.get('out_features')
        timesteps = layer.get('timesteps') * 3
        spike_in_count = layer.get('spike_in_count')
        spike_out_count = layer.get('spike_out_count')
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size) if layer_type == 'conv' else (output_feature, input_feature)
        # 自动推导卷积层到全连接层的输入输出特征
        if input_feature is None and layer_type == 'fc':
            if prev_output_size is not None and prev_out_channels is not None:
                input_feature = prev_output_size * prev_out_channels
            else:
                raise ValueError("缺少 FC 层 input_feature 参数 定义，且无法从前层推导")
        ann_estimator = FNNEnergyCost(
                input_size=input_size,
                output_size=output_size,
                channel_in=in_channels,
                channel_out=out_channels,
                kernel_size=kernel_size,
                input_feature=input_feature,
                output_feature=output_feature
        )
        snn_estimator = SNNEnergyCost(
                input_size=input_size,
                output_size=output_size,
                channel_in=in_channels,
                channel_out=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                timesteps=timesteps,
                spike_in_count=spike_in_count,
                spike_out_count=spike_out_count,
                input_feature=input_feature,
                output_feature=output_feature
        )

        # 自动推导 input_size/output_size
        if layer_type == 'conv':
            spike_in_rate = spike_in_count / (input_size * timesteps * in_channels)
            spike_out_rate = spike_out_count / (output_size * timesteps * out_channels)
            print(f"\n--- Layer {layer['name']} (Conv) ---")
            print(f"Input: {input_size}x{input_size}, Output: {output_size}x{output_size}")
            print(f"Channels: in={layer['in_channels']}, out={layer['out_channels']}")
            print(f"Spikes: in={spike_in_count}, out={spike_out_count}")
            energy_ann = ann_estimator.calculate_cost('conv')
            energy_snn = snn_estimator.calculate_cost('conv')

        elif layer_type == 'fc':
            spike_in_rate = spike_in_count / (input_feature * timesteps)
            spike_out_rate = spike_out_count / (output_feature * timesteps)
            print(f"\n--- Layer {layer['name']} (FC) ---")
            print(f"Input: {input_feature}, Output: {output_feature}")
            print(f"Spikes: in={spike_in_count}, out={spike_out_count}")
            energy_ann = ann_estimator.calculate_cost('fc')
            energy_snn = snn_estimator.calculate_cost('fc')

        else:
            print(f"\n跳过暂不支持的层类型: {layer['type']}")
            continue

        print(f"ANN Energy Cost: {sum(energy_ann):.2f} mJ")
        print(f"SNN Energy Cost: {sum(energy_snn):.2f} mJ")

        # 记录当前层输出作为下一层输入
        prev_output_size = output_size if layer_type == 'conv' else None
        prev_out_channels = layer.get('out_channels', None)

        #绘图数据记录

        spike_in_rate_list.append(spike_in_rate)
        spike_out_rate_list.append(spike_out_rate)
        layer_names.append(layer["name"])

        # ANN、SNN 访存、计算能耗
        energy_ann_list.append(energy_ann)
        energy_snn_list.append(energy_snn)
        # 提取总能耗
        energy_ann_vals = [sum(v) for v in energy_ann_list]
        energy_snn_vals = [sum(v) for v in energy_snn_list]


    # 绘图
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 能耗条形图
    width = 0.35
    x = range(len(layer_names))
    ax1.bar([i - width/2 for i in x], energy_ann_vals, width, label='ANN Energy', color='skyblue')
    ax1.bar([i + width/2 for i in x], energy_snn_vals, width, label='SNN Energy', color='salmon')
    ax1.set_ylabel("Energy (mJ)")
    ax1.set_title("ANN vs SNN Energy per Layer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names)
    ax1.legend(loc='upper left')

    # 添加右侧平均脉冲率折线图
    ax2 = ax1.twinx()
    ax2.plot(x, spike_out_rate_list, color='green', marker='o', label='Output Spike Rate')
    ax2.set_ylabel("Spike Rate (Avg per Neuron)")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # 保存每层参数总量、访存能耗、计算总能耗
    layer_energy_data = []
    total_params = 0
    total_comp_energy_ann = 0
    total_comp_energy_snn = 0
    total_memory_energy_ann = 0
    total_memory_energy_snn = 0
    total_energy_ann = 0
    total_energy_snn = 0

    for i, layer in enumerate(NETWORK):
        layer_type = layer["type"]
        layer_name = layer["name"]
        weight_shape = layer.get("weight_shape")
        if layer_type not in ['conv', 'fc']:
            continue
        param_count = np.prod(weight_shape)
        total_params += param_count

        # 获取 ANN / SNN 能耗
        ann_energy = energy_ann_list[i]
        snn_energy = energy_snn_list[i]

        total_comp_energy_ann += ann_energy[0]  # 计算能耗
        total_comp_energy_snn += snn_energy[0]   
        total_memory_energy_ann += ann_energy[1]  # 访存能耗
        total_memory_energy_snn += snn_energy[1]

        total_energy_ann += sum(ann_energy)
        total_energy_snn += sum(snn_energy)

        layer_energy_data.append({
            "layer": layer_name,
            "type": layer_type,
            "params": int(param_count),
            "ANN_comp_energy_mJ": ann_energy[0],
            "SNN_comp_energy_mJ": snn_energy[0],
            "ANN_memory_energy_mJ": ann_energy[1],
            "SNN_memory_energy_mJ": snn_energy[1],
            "ANN_total_energy_mJ": total_energy_ann,
            "SNN_total_energy_mJ": total_energy_snn
        })

    # ===== 写入 CSV =====
    csv_file = "energy_report.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "type", "params", 
            "ANN_comp_energy_mJ","SNN_comp_energy_mJ",
            "ANN_memory_energy_mJ","SNN_memory_energy_mJ",
            "ANN_total_energy_mJ","SNN_total_energy_mJ"
        ])
        writer.writeheader()
        writer.writerows(layer_energy_data)

    # ===== 打印汇总信息 =====
    print("\n===== 网络总能耗统计 =====")
    print(f"总参数量：{total_params:,} 个")
    print(f"ANN 访存能耗：{total_memory_energy_ann:.2f} mJ")
    print(f"ANN 计算能耗：{total_comp_energy_ann:.2f} mJ")
    print(f"ANN 总推理能耗：{total_energy_ann:.2f} mJ")

    print(f"SNN 访存能耗：{total_memory_energy_snn:.2f} mJ")
    print(f"SNN 计算能耗：{total_comp_energy_snn:.2f} mJ")
    print(f"SNN 总推理能耗：{total_energy_snn:.2f} mJ")
    print(f"已写入结果至：{csv_file}")

    print("\n===== 统计分析完成 =====")
if __name__ == "__main__":
    main()

