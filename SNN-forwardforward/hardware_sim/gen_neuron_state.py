import sys
sys.path.append('D:/OneDrive/SNN_FFSTDP/SNN-forwardforward')
import os
from hardware_sim_config import *

def pack_pre_state_to_hex(spike_cnt, depth, pack_bits=32):
    """
    将 spike_cnt(7bit) 单元按 pack_bits 位宽拼接，最终输出 hex 字符串列表。

    字段（从高到低）：
        [spike_cnt(7b)]

    :param spike_cnt: int（0~127）
    :param depth: 输出数组长度
    :param pack_bits: 拼接后的总 bit 宽度（32/64/128/...）
    :return: hex 字符串数组
    """

    # ---- Step 1: 拼成 7-bit 单元 ----
    unit_bin = f"{spike_cnt & 0x7F:07b}"  # 7 bits

    # ---- Step 2: 每 pack 需要多少个 7-bit 单元 ----
    N = pack_bits // 7

    # ---- Step 3: 复制 N 次，构造大 bit 串 ----
    full_bits = unit_bin * N
    full_bits = full_bits[:pack_bits]  # 精准截断 pack_bits

    packed_data = []

    # ---- Step 4: 根据 depth 复制输出 ----
    for _ in range(depth):
        hex_str = f"{int(full_bits, 2):0{pack_bits//4}X}"
        packed_data.append(hex_str)

    return packed_data

def pack_post_state_to_hex(enable, spike_cnt, v_th, v_mem, depth, pack_bits=32):
    """
    将单元(1b + 7b + 12b + 12b = 32bit)复制 N 次，并按 pack_bits 拼接成 hex 字符串。

    字段（从高到低拼接）：
        [enable(1b)][spike_cnt(7b)][v_th(12b)][v_mem(12b)]

    :param enable: int (0/1)
    :param spike_cnt: int 0~127
    :param v_th: int 0~4095
    :param v_mem: int 0~4095
    :param N: 单元复制次数
    :param pack_bits: 输出拼接位宽（32/64/128/...）
    :return: hex 字符串列表，每个元素 pack_bits 位
    """

    # ---- Step 1: 构造单元 32-bit ----
    enable_bin   = f"{enable:01b}"
    spike_bin    = f"{spike_cnt:06b}"
    # vth_bin      = f"{v_th:013b}"
    vmem_bin     = f"{v_mem:013b}"

    # 拼成 32-bit（大端格式：高位在左）
    # unit_bin = enable_bin + spike_bin + vth_bin + vmem_bin  # len = 32
    unit_bin = enable_bin + spike_bin + vmem_bin  # len = 32

    N = int(pack_bits / len(unit_bin))
    # ---- Step 2: 复制 N 次 ----
    full_bits = unit_bin * N


    packed_data = []

    for i in range(depth):
        chunk = full_bits[i * pack_bits:(i + 1) * pack_bits]
        # 转 hex（自动补齐为 pack_bits/4 位）
        hex_str = f"{int(full_bits, 2):0{pack_bits//4}X}"
        packed_data.append(hex_str)

    return packed_data


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
    out_dir = "Gen_out"
    os.makedirs(out_dir, exist_ok=True)
    
    vth = int(1.3 / WEIGHT_SCALE)
    post_state = pack_post_state_to_hex(enable=1, spike_cnt=0, v_th=vth, v_mem=0, depth = POST_SRAM_DEPTH, pack_bits=POST_SRAM_DATA_WIDTH)
    pre_state = pack_pre_state_to_hex(spike_cnt=0, depth = PRE_SRAM_DEPTH, pack_bits=PRE_SRAM_DATA_WIDTH)

    save_txt_file(post_state, filename=f"./{out_dir}/post_neuron_state.txt")
    save_txt_file(pre_state, filename=f"./{out_dir}/pre_neuron_state.txt")

    print("TXT 文件已生成！")
    

