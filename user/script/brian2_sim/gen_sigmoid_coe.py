import numpy as np
import os
import matplotlib.pyplot as plt

T = 32
theta = 0.25*T
N = 8               # 8位定点数
FL = 5			    # 表示小数部分位宽
IL = N - FL - 1		# 表示符号位 + 整数部分位宽
def sigmoid(x):
    # 计算 Sigmoid 函数
    sigmoid = 1 / (1 + np.exp(-x))

    return sigmoid
def to_signed_8bit(value):
    # 将值从 [0, 1] 范围映射到 [-128, 127] 范围
    scaled_value = (value) * (2**FL)
    fix_list=[]
    for i in scaled_value:
        fix_list.append(int(i))
    return fix_list
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
    # 定点缩放
    scaled = int(round(val * (2 ** F)))
    # 计算总位宽
    total_bits = I + F + 1  # 包含符号位
    # 对超出范围的进行裁剪
    min_val = -2 ** I
    max_val = 2 ** I - 2 ** (-F)
    if val < min_val or val > max_val:
        print(f"Warning: {val} 超出 Q{I}.{F} 表示范围 [{min_val}, {max_val}]，会被裁剪")
    # 如果是负数，使用补码
    if scaled < 0:
        scaled = (1 << total_bits) + scaled
    # 转为二进制字符串并截取
    bin_str = format(scaled, f'0{total_bits}b')
    hex_str = hex(int(bin_str, 2))
    return bin_str, hex_str

def float_to_fixed_val(data,I,F):
    """
    将浮点数组进行定点量化。

    参数:
        data (ndarray): 原始浮点数组
        I (int): 整数位数（不包含符号位）
        F (int): 小数位数

    返回:
        ndarray: 量化后的数组
    """
    int_max = 2**(I) - 2**(-F)
    int_min = -2**I
    step = 2 ** (-F)
    # 进行量化
    quantized = np.clip(np.round(data / step) * step, int_min, int_max)
    return quantized
def write_coe(file_name,data):
    f = open(file_name, 'w')
    f.write('MEMORY_INITIALIZATION_RADIX=16;\n')
    f.write('MEMORY_INITIALIZATION_VECTOR=\n')
    for i in data:
        bin_str, hex_str = float_to_fixed_bin(i, IL, FL)
        f.write(hex_str[2:] + ',\n')
    f.close()
def cal_Q8_value(q8_fix):
    value = 0
    bin_str = bin(q8_fix)[2:].zfill(FL)
    for i,b in enumerate(bin_str):
        if b=='1':
            value +=(2**(-i-1))
    return value
def analyze_error(v_float,v_fix,is_pos):
    for i in range(len(v_fix)):
        if is_pos:
            v_fix_ = -1*cal_Q8_value(v_fix[i])
        else:
            v_fix_ = cal_Q8_value(v_fix[i])
        error = abs(v_float[i] - v_fix_)
        print(f"derivative_float{i} = {v_float[i]:.4f}  derivative_fix{i} = {v_fix[i]}    derivative_fix_bin{i} = {bin(v_fix[i])} erro = {error:.4f}")
def generate_coe(pos_delta_list,neg_delta_list):
    # pos_fix = to_signed_8bit(pos_derivative)
    # neg_fix = to_signed_8bit(neg_derivative)
    # analyze_error(pos_derivative,pos_fix,True)
    # analyze_error(neg_derivative,neg_fix,False)
    write_coe("pos_derivative_coe.coe",pos_delta_list)
    write_coe("neg_derivative_coe.coe",neg_delta_list)
def find_fix_erro_min(delta_weight):
    # 尝试所有可能的格式：I + F = 7, I from 0~7
    min_error = float('inf')
    best_format = (0, 0)
    for I in range(1, N-1):
        F = N - 1 - I
        quantized = float_to_fixed_val(delta_weight, I, F)
        error = np.mean(np.abs(delta_weight - quantized))
        print(f"Q{I}.{F} format -> Avg Abs Error: {error:.6f}")
        if error < min_error:
            min_error = error
            best_format = (I, F)
    # 输出最佳定点格式
    print(f"\n✅ 最佳定点格式: Q{best_format[0]}.{best_format[1]}，平均绝对误差: {min_error:.6f}")
def delta_weight_vis(pos_delta_list, neg_delta_list):
    # 计算 delta weight 的最大值、最小值和平均值
    print("pos_delta:", pos_delta_list)
    print("neg_delta:", neg_delta_list)
    print("pos_delta_max:", np.abs(pos_delta_list).max())
    print("pos_delta_min:", np.abs(pos_delta_list).min())
    print("pos_delta_mean:", np.abs(pos_delta_list).mean())
    print("neg_delta_max:", np.abs(neg_delta_list).max())
    print("neg_delta_min:", np.abs(neg_delta_list).min())
    print("neg_delta_mean:", np.abs(neg_delta_list).mean())

    pos_delta_list = np.array(pos_delta_list)
    neg_delta_list = np.array(neg_delta_list)
    distance = pos_delta_list + neg_delta_list

    find_fix_erro_min(pos_delta_list)
    find_fix_erro_min(neg_delta_list)
    pos_delta_quantized = float_to_fixed_val(pos_delta_list, IL, FL)
    neg_delta_quantized = float_to_fixed_val(neg_delta_list, IL, FL)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(pos_delta_quantized, label='Positive Δw', color='blue')
    plt.plot(neg_delta_quantized, label='Negative Δw', color='red')
    # plt.plot(distance, label='Distance Δw', color='green')
    plt.xlabel("Flattened (pre, post) index")
    plt.ylabel("Delta Weight")
    plt.title("STDP-like Δw Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sigmoid_vis():
    # 参数设置
    x = np.arange(T+1)
    goodness = (x**2)/T - theta

    # goodness 和 sigmoid 值
    sigmoid_pos = 2*(x/T)*sigmoid(-goodness)
    sigmoid_neg = 2*(x/T)*sigmoid(goodness)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(x, sigmoid_pos, label='sigmoid(goodness)', color='blue')
    plt.plot(x, sigmoid_neg, label='sigmoid(-goodness)', color='red')
    plt.xlabel('x (spike count)')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid(goodness) vs Sigmoid(-goodness)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    x = np.arange(T+1)
    goodness = (x**2)/T - theta
    pos_derivative = 2*(x/T)*sigmoid(-goodness)
    neg_derivative = -2*(x/T)*sigmoid(goodness)
    pos_delta_list = []
    neg_delta_list = []
    for pre in range(T):
        for post in range(T):
            pos_delta = (pre+1)  * pos_derivative[post+1] /32
            neg_delta = (pre+1)  * neg_derivative[post+1] /64
            pos_delta_list.append(pos_delta)
            neg_delta_list.append(neg_delta)
    
    sigmoid_vis()
    delta_weight_vis(pos_delta_list, neg_delta_list)

    # generate_coe(pos_delta_list, neg_delta_list)
