import sys
sys.path.append('D:/OneDrive/SNN_FFSTDP/SNN-forwardforward')
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import math
from src.ff_snn_net import Net
from config import ConfigParser
from hardware_sim_config import *
def quantize_to_int(x_fp, scale, bits=16):
    """
    将浮点数张量 x_fp 按 scale 量化为有符号整型 (int16 / int8)
    支持饱和裁剪
    """
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    x_int = np.round(x_fp * scale)
    x_int = np.clip(x_int, qmin, qmax).astype(np.int16)
    return x_int

def quantize_tensor_int(tensor_fp: np.ndarray, max_abs, num_bits=16):
    qmax = 2**(num_bits - 1) - 1

    scale = max_abs / 2**(num_bits-1) if max_abs > 0 else 1.0
    x_int = torch.clamp(torch.round(tensor_fp / scale), -qmax - 1, qmax).to(torch.int32)
    return x_int.numpy()
def float_to_fixed_bin(val, I, F):
    """
    将浮点数转换为补码形式的有符号定点数二进制字符串（用于硬件表示）。

    参数:
        val (float): 输入的浮点数（支持正负数）。
        I (int): 定点数整数部分的位数（不包括符号位）。
        F (int): 定点数小数部分的位数。

    返回:
        str: 长度为 (1 + I + F) 的二进制字符串，采用补码形式，适合用于 RTL、Verilog 等。

    功能:
        - 将浮点数 `val` 按照 QI.F 定点格式进行缩放（val × 2^F）；
        - 对正数进行二进制转换；
        - 对负数使用补码编码（two's complement）；
        - 输出最终二进制字符串，作为一个完整的（1 + I + F）位的补码定点表示。

    示例:
        >>> float_to_fixed_bin(1.625, 2, 5)
        '00110100'  # Q2.5 格式，正数
        
        >>> float_to_fixed_bin(-1.625, 2, 5)
        '11001100'  # Q2.5 格式，负数补码表示

    注意事项:
        - 总位宽为 (1 + I + F)，表示范围为 [-2^I, 2^I - 2^-F]
        - 若数值超出该范围，可能出现截断或溢出（需提前处理）
    """
    # 计算总位宽
    total_bits = I + F + 1  # 包含符号位
    hex_bits = int(((total_bits+ 4-1) / 4))


    # 转为有符号二进制字符串
    mask = (1 << total_bits) - 1
    scaled_twos = val & mask
    bin_str = format(scaled_twos, f'0{total_bits}b')
    hex_str = f"{int(bin_str, 2):0{hex_bits}X}"
    return bin_str, hex_str
def bin_list_to_hex_list(packed_data_bin, zero_pad=True):
    """
    packed_data_bin: List[str]，每个元素是二进制字符串，如 '10101100'
    zero_pad: 是否保持位宽（4bit 对齐补 0）

    return: List[str]，对应的 16 进制字符串（不含 0x）
    """
    packed_data_hex = []

    for bin_str in packed_data_bin:
        # 去掉可能的空格 / 换行
        bin_str = bin_str.strip()

        # 位宽不是 4 的整数倍时，前面补 0
        if zero_pad and len(bin_str) % 4 != 0:
            pad_len = 4 - (len(bin_str) % 4)
            bin_str = '0' * pad_len + bin_str

        hex_str = hex(int(bin_str, 2))[2:]  # 去掉 '0x'
        packed_data_hex.append(hex_str)

    return packed_data_hex

def pack_to_nbit(q_weights, max_val, num_bits, pack_bits=32):
    """
    将定点量化后的数据按 pack_bits 位拼接（可为 32 / 64 / 96 / 128 ...）
    :param q_weights: 量化后 int8 或其它 bit-width 的数组
    :param max_val: 用于定点转换
    :param num_bits: 每个数的 bit width（你这里是 int8，所以 8）
    :param pack_bits: 拼接后的总 bit width（默认 32，可以改成 64）
    :return: 拼接好的字符串数组，每个元素 pack_bits 位
    """

    q_weights = q_weights.flatten()
    vals_per_pack = pack_bits // num_bits  # 每多少个数拼成一个 pack
    num_packed = len(q_weights) // vals_per_pack

    packed_data_bin = []
    packed_data_hex = []

    # 计算 il / fl
    fl = int(-np.log2(max_val / (2 ** (num_bits - 1))))
    il = int(num_bits - fl - 1)

    for i in range(num_packed):
        s_bin_list = []
        s_hex_list = []
        for j in range(vals_per_pack):
            bin_s, hex_s  = float_to_fixed_bin(q_weights[i * vals_per_pack + j], il, fl)
            s_bin_list.append(bin_s)
            # s_hex_list.append(hex_s)

        # 小端：低位在前
        bin_str = "".join(s_bin_list[::-1])  # reverse for little endian
        packed_data_bin.append(bin_str)
        # hex_str = "".join(s_hex_list[::-1])  # reverse for little endian
        # packed_data_hex.append(hex_str)
    packed_data_hex = bin_list_to_hex_list(packed_data_bin, zero_pad=False)
    return packed_data_bin, packed_data_hex


def save_coe_file(packed_data, filename="weights.coe"):
    """
    生成 COE 文件 (16 进制格式)
    :param packed_data: 32-bit 8位16进制 字符串列表
    :param filename: COE 文件名
    """
    with open(filename, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")

        for i, hex_str in enumerate(packed_data):
            if i == len(packed_data) - 1:
                f.write(f"{hex_str};\n")  # 结尾加 ";"
            else:
                f.write(f"{hex_str},\n")  # 每行写入 32-bit 数据
def save_txt_file(packed_data, filename="weights.txt"):
    """
    生成 txt 文件 (16 进制格式)
    :param packed_data: 32-bit 8位16进制 字符串列表
    :param filename: COE 文件名
    """
    with open(filename, "w") as f:
        for i, hex_str in enumerate(packed_data):
            f.write(f"{hex_str}\n")  #
if __name__ == "__main__":
    out_dir = "Gen_out/" + TASK
    os.makedirs(out_dir, exist_ok=True)
    config = ConfigParser()
    args = config.parse()
    device = torch.device("cuda")
    net = Net(dims=[784,256, 10],tau=args.tau, epoch=args.epochs, T=T, lr=args.lr,
              v_threshold_pos=V_THRESHOLD_POS,v_threshold_neg=V_THRESHOLD_NEG, opt=args.opt, loss_threshold=THETA)
    net.load(NET_PATH)
    layer_weights = {}
    layer_weights_int8 = {}
    for layer_idx, layer in enumerate(net.layers):
            # 遍历命名子模块，找到真正的可学习层（你模型中 name=="layer" 的模块）
            for name, module in layer.named_modules():
                if name == "layer":
                    for p_name, param in module[1].named_parameters():
                        w = param.detach().cpu()
                        layer_weights[p_name] = w.transpose(1,0) 
            break
    
    for layer_name, w in layer_weights.items():
        weight_int8 = quantize_tensor_int(w, WEIGHT_MAX, num_bits=WEIGHT_WIDTH)
        layer_weights_int8[layer_name] = weight_int8
        error = np.mean(np.abs(w.numpy() - (weight_int8 * (WEIGHT_MAX / 2**(WEIGHT_WIDTH-1))) ))
        print(f"{layer_name} 量化INT{WEIGHT_WIDTH}误差: {error:.4f}")
        packed_data_bin, packed_data_hex = pack_to_nbit(q_weights=weight_int8, max_val=WEIGHT_MAX, num_bits=WEIGHT_WIDTH, pack_bits=int(POST_PARALLEL * WEIGHT_WIDTH))
        # 2. 4 个 8-bit 合并为 32-bit
        save_coe_file(packed_data_hex, f"./{out_dir}/weights_{layer_name}.coe")
        save_txt_file(packed_data_bin, f"./{out_dir}/weights_{layer_name}.txt")
    print("COE 文件已生成！")



    DATA_WIDTH = int(POST_PARALLEL * WEIGHT_WIDTH)
    BLOCK_DEPTH = 2048

    with open(f"./{out_dir}/weights_{layer_name}.txt") as f:
        data = [line.strip() for line in f if line.strip()]

    bank_num = math.ceil(len(data) / BLOCK_DEPTH)

    for b in range(bank_num):
        with open(f"./{out_dir}/weights_bank_{b}.txt", "w") as f:
            for i in range(BLOCK_DEPTH):
                idx = b * BLOCK_DEPTH + i
                if idx < len(data):
                    f.write(data[idx] + "\n")
                else:
                    f.write("0\n")

    

