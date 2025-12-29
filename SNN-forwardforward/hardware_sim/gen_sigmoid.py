import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import math


from hardware_sim_config import *

def pos_derivative(x, theta):
    return -1 / (1 + torch.exp(x - theta))

def neg_derivative(y, theta):
    return 1 / (1 + torch.exp(theta - y))

def cal_goodness(T, freq):
    return T * freq.abs().pow(2) * freq.sign()

def quantize_tensor_int(tensor_fp: np.ndarray, max_abs, num_bits=16):
    qmax = 2**(num_bits - 1) - 1
    scale = max_abs / 2**(num_bits-1) if max_abs > 0 else 1.0
    x_int = torch.clamp(torch.round(tensor_fp / scale), -qmax - 1, qmax).to(torch.int32)
    return x_int.numpy()

def float_to_fixed_bin(val, I, F):
    total_bits = I + F + 1
    mask = (1 << total_bits) - 1
    scaled_twos = val & mask
    bin_str = format(scaled_twos, f'0{total_bits}b')
    hex_str = f"{int(bin_str, 2):02X}"
    return bin_str, hex_str

def write_coe(file_name, data, num_bits, max_val):
    f = open(file_name, 'w')
    f.write('MEMORY_INITIALIZATION_RADIX=16;\n')
    f.write('MEMORY_INITIALIZATION_VECTOR=\n')
    fl = int(-np.log2(max_val / (2 ** (num_bits - 1))))
    il = int(num_bits - fl -1)
    for i in data:
        bin_str, hex_str = float_to_fixed_bin(i, il, fl)
        f.write(hex_str + ',\n')
    f.close()

def rom_vis(x,pos,neg):
    plt.figure(figsize=(10, 5))
    plt.plot(x, pos, label='ROM POS Derivative')
    plt.plot(x, neg, label='ROM NEG Derivative')
    plt.xlabel('x')
    plt.ylabel('ROM Output')
    plt.legend()
    plt.grid(True)
    plt.show()

#############################
# 🔥 自动生成 Verilog ROM 模块
#############################
def generate_rom_verilog_reg(file_name, module_name, data_list, data_width):
    """
    自动生成 可综合、同步读、寄存输出的 ROM
    dout 在下一个周期输出
    """
    depth = len(data_list)
    addr_width = int(math.log2(depth))    

    with open(file_name, 'w') as f:
        f.write(f"// Automatically generated ROM with registered output\n")
        f.write(f"module {module_name} #(\n")
        f.write(f"    parameter DATA_WIDTH = {data_width},\n")
        f.write(f"    parameter ADDR_WIDTH = $clog2({depth})\n")
        f.write(f")(\n")
        f.write(f"    input  wire                  clk,\n")
        f.write(f"    input  wire [ADDR_WIDTH-1:0] addr,\n")
        f.write(f"    output reg  [DATA_WIDTH-1:0] dout\n")
        f.write(f");\n\n")

        # 定义内部中间寄存器
        f.write(f"reg [DATA_WIDTH-1:0] rom_data;\n\n")

        # 组合逻辑 ROM
        f.write(f"always @(*) begin\n")
        f.write(f"    case(addr)\n")

        for i, val in enumerate(data_list):
            hex_val = f"{(val & ((1<<data_width)-1)):0{data_width//4}X}"
            f.write(f"        {addr_width}'d{i}: rom_data = {data_width}'h{hex_val};\n")

        f.write(f"        default: rom_data = {{DATA_WIDTH{{1'b0}}}};\n")
        f.write(f"    endcase\n")
        f.write(f"end\n\n")

        # 时序寄存输出
        f.write(f"always @(posedge clk) begin\n")
        f.write(f"    dout <= rom_data;\n")
        f.write(f"end\n\n")

        f.write(f"endmodule\n")

    print(f"Generated ROM Verilog with registered output: {file_name}")


##################################
# 主流程
##################################
if __name__ == '__main__':
    out_dir = "Gen_out" + "/" + TASK
    os.makedirs(out_dir, exist_ok=True)
    theta = THETA
    lr = 0.015625 / 8

    x = torch.tensor(np.arange(T+1))
    freq = x / T
    goodness = cal_goodness(T, freq)

    pos_der = -2 * freq * pos_derivative(goodness, theta)
    neg_der = -2 * freq * neg_derivative(goodness, theta)

    pos_delta_list = []
    neg_delta_list = []

    for pre in range(16):
        for post in range(16):
            if pre + 1 > T or post + 1 > T:
                pos_delta = 0
                neg_delta = 0
            else:
                pos_delta =  lr * (pre+1) * pos_der[post+1]
                neg_delta =  lr * (pre+1) * neg_der[post+1]
            pos_delta_list.append(pos_delta)
            neg_delta_list.append(neg_delta)
    _ = torch.tensor(np.arange(16*16))
    pos_delta_list = torch.tensor(pos_delta_list)
    neg_delta_list = torch.tensor(neg_delta_list)
    rom_vis(_, pos_delta_list, neg_delta_list)

    # 量化
    pos_delta_int_list = quantize_tensor_int(pos_delta_list, WEIGHT_MAX, num_bits=WEIGHT_WIDTH)
    neg_delta_int_list = quantize_tensor_int(neg_delta_list, WEIGHT_MAX, num_bits=WEIGHT_WIDTH)
    rom_vis(_, pos_delta_int_list, neg_delta_int_list)
    # 🔥 生成 Verilog ROM
    generate_rom_verilog_reg("./Hardware/rtl/snn_ff/rom/pos_derivative_rom.v", "pos_derivative_rom", pos_delta_int_list, WEIGHT_WIDTH)
    generate_rom_verilog_reg("./Hardware/rtl/snn_ff/rom/neg_derivative_rom.v", "neg_derivative_rom", neg_delta_int_list, WEIGHT_WIDTH)

    write_coe(f"./{out_dir}/pos_derivative.coe",pos_delta_int_list, num_bits=WEIGHT_WIDTH, max_val=WEIGHT_MAX) 
    write_coe(f"./{out_dir}/neg_derivative.coe",neg_delta_int_list, num_bits=WEIGHT_WIDTH, max_val=WEIGHT_MAX)
