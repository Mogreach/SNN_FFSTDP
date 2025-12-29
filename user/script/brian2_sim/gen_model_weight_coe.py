from brian2 import *
import brian2.numpy_ as np
import h5py
import time
import os

T = 16
theta = 2.5
N = 8               # 8位定点数
FL = 5			    # 表示小数部分位宽
IL = N - FL - 1		# 表示符号位 + 整数部分位宽
class SNN():
    def __init__(self):
        self.input_layer_size = 784
        self.hidden_layer_size = 256
        self.output_layer_size = 100
        self.cita_h = 1.6
        self.cita_o = 2.5
        self.reset = 0
        self.theta = 0.8
        # define the neuron model
        self.eqs = '''
                v:1
                cita_h:1
                cita_o:1
                reset:1
                '''
        self.base_frequency = 250
        self.learn_rate = 2 * 0.0078125
        self.time_step = 16
        self.sim_time = 16
        # make sure "test_steps * time_step = 20.0"
        self.test_steps = 1

        self.rates = zeros(self.input_layer_size)

        # input neurons
        inp = NeuronGroup(self.input_layer_size, 'v:1', method='exact', threshold='v>=1',
                          reset='v=0',name="input")
        def update_volt():
            inp.v += self.rates
        network_op = NetworkOperation(update_volt, dt=1.0 * ms)
        # hidden neurons
        hidden = NeuronGroup(self.hidden_layer_size, self.eqs, threshold="v>cita_h",
                             reset='v=reset', method="exact",name="hidden")
        hidden.cita_h = self.cita_h
        hidden.reset = self.reset
        output = NeuronGroup(self.output_layer_size, self.eqs, threshold='v>cita_o',
                             reset='v=reset', method='exact',name="output")
        output.cita_o = self.cita_o
        output.reset = self.reset


        conn_ih = Synapses(inp, hidden, model='w:1', on_pre='v_post += w',name="conn_ih")
        conn_ih.connect(p=1)
        conn_ih.w = np.random.randn(self.input_layer_size*self.hidden_layer_size)
        conn_ho = Synapses(hidden, output, model='w:1', on_pre='v_post += w',name="conn_ho")
        conn_ho.connect(p=1)
        conn_ho.w = np.random.randn(self.hidden_layer_size*self.output_layer_size)
        self.net = Network(conn_ih, conn_ho, network_op,
                           inp, hidden, output)
        # 定义脉冲变量
        self.spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        self.spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        self.spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')

        self.input_spike_count = array(self.spikemon_input.count).copy()
        self.hidden_spike_count = array(self.spikemon_hidden.count).copy()
        self.output_spike_count = array(self.spikemon_output.count).copy()
        # self.net.store("initial_weight")

    def set_input(self,img_array):
        self.rates = img_array / 255.0
        self.net.set_states({"input":{"v":zeros(self.input_layer_size)}})
        self.net.set_states({"hidden":{"v":zeros(self.hidden_layer_size)}})
        self.net.set_states({"output":{"v":zeros(self.output_layer_size)}})

    def save_weight(self,file_name):
        print("# Saving weights")
        f = h5py.File(file_name,'w')
        f["weight_1"] = self.net.get_states()["conn_ih"]["w"]
        f["weight_2"] = self.net.get_states()["conn_ho"]["w"]
        f.close()

    def load_weight(self,file_name):
        print("# Loading weights from %s" % file_name)
        f = h5py.File(file_name, 'r')
        self.net.set_states({"conn_ih":{"w":f["weight_1"][:]}})
        self.net.set_states({"conn_ho":{"w":f["weight_2"][:]}})
        f.close()


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
    # 对超出范围的进行裁剪
    min_val = -2 ** I
    max_val = 2 ** I - 2 ** (-F)
    if val < min_val or val > max_val:
        print(f"Warning: {val} 超出 Q{I}.{F} 表示范围 [{min_val}, {max_val}]，会被裁剪")
        val = np.clip(val, min_val, max_val)
    # 定点缩放
    scaled = int(round(val * (2 ** F)))

    
    # 如果是负数，使用补码
    if scaled < 0:
        scaled = (1 << total_bits) + scaled
    # 转为二进制字符串并截取
    bin_str = format(scaled, f'0{total_bits}b')
    hex_str = f"{int(bin_str, 2):02X}"
    return bin_str, hex_str
def pack_to_32bit(q_weights):
    """
    将 8-bit 有符号数据 (-128 ~ 127) 按 4 个一组打包为 32-bit
    :param q_weights: 量化后的 int8 数组 (784*256)
    :return: 32-bit 拼接的 int 数组
    """
    q_weights = q_weights.flatten()  # 展平为 1D 数组
    num_packed = len(q_weights) // 4  # 计算 32-bit 数据个数
    packed_data = [] # 预分配数组

    for i in range(num_packed):
        # 读取 4 个 int8 值
        _, s0 = float_to_fixed_bin(q_weights[i * 4 + 0], IL, FL)
        _, s1 = float_to_fixed_bin(q_weights[i * 4 + 1], IL, FL)
        _, s2 = float_to_fixed_bin(q_weights[i * 4 + 2], IL, FL)
        _, s3 = float_to_fixed_bin(q_weights[i * 4 + 3], IL, FL)
        # b0 = np.int8(q_weights[i * 4 + 0])
        # b1 = np.int8(q_weights[i * 4 + 1])
        # b2 = np.int8(q_weights[i * 4 + 2])
        # b3 = np.int8(q_weights[i * 4 + 3])

        # # 转换为无符号数（补码表示），确保拼接正确
        # ub0 = np.uint8(b0) & 0xFF
        # ub1 = np.uint8(b1) & 0xFF
        # ub2 = np.uint8(b2) & 0xFF
        # ub3 = np.uint8(b3) & 0xFF
        hex_str = f"{s3}{s2}{s1}{s0}"  # 拼接为 32-bit 字符串
        # 采用 **小端模式** 存储：低字节在前，高字节在后
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

if __name__ == "__main__":
    snn = SNN()
    snn.load_weight(f"./4-10_acc_93.h5")
    conn_ih = snn.net.get_states()["conn_ih"]["w"]
    conn_ho = snn.net.get_states()["conn_ho"]["w"]
    # save_quantized_weights("quantized_weights.h5", conn_ih, conn_ho)
    # 测试
    # original_weights = np.array([-4.0, -3.5, -2.75, -1.25, 0, 1.25, 2.75, 3.5, 4.0])  # 示例权重
    quantized_weights = float_to_fixed_val(conn_ih, 2, 5)
    # quantized_weights = quantize_to_q2_5(conn_ih)

    print("原始权重:", conn_ih)
    print("量化后权重:", quantized_weights)


    # 2. 4 个 8-bit 合并为 32-bit
    packed_data = pack_to_32bit(conn_ih)

    # 3. 生成 COE 文件
    save_coe_file(packed_data, "weights_93.coe")
    print("COE 文件已生成！")

