import math
from network_config import *
class FNNEnergyCost:
    """
    前馈神经网络操作成本估计类（用于ANN）
    :param channel_in: 输入通道数
    :param channel_out: 输出通道数
    :param kernel_size: 卷积核大小
    :param input_size: 输入特征大小，HxW
    :param output_size: 输出特征大小，HxW
    :param input_feature: 输入神经元数量
    :param output_feature: 输出神经元数量
    """
    def __init__(self, input_size, output_size, channel_in, channel_out, kernel_size, 
                 input_feature=None, output_feature=None):
        self.input_size = input_size
        self.output_size = output_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.input_feature = input_feature if input_feature is not None else input_size * channel_in
        self.output_feature = output_feature if output_feature is not None else output_size * channel_out
    def operational_cost_Conv(self, channel_in, channel_out, kernel_size, output_size):
        """
        计算FNN卷积层的操作成本
        :return: MACs (乘加操作数) 以及 ACCs (加法操作数)
        """
        mac = channel_out * output_size * channel_in * kernel_size * kernel_size
        acc = channel_out * output_size
        return mac, acc
    def operational_cost_FC(self, input_feature, output_feature):
        """
        计算FNN全连接层的操作成本
        :return: MACs (乘加操作数) 以及 ACCs (加法操作数)
        """
        mac = input_feature * output_feature
        acc = output_feature
        return mac, acc
    def Rd_Input_memory_access_Conv(self, channel_in, channel_out, output_size, kernel_size):
        """
        计算FNN卷积层的输入变量内存访问成本
        :return: 输入内存访问次数
        """
        return channel_in * channel_out * output_size * kernel_size * kernel_size
    def Rd_Input_memory_access_FC(self, input_feature):
        """
        计算FNN全连接层的输入变量内存访问成本
        :return: 输入内存访问次数
        """
        return input_feature
    def Rd_Param_memory_access_Conv(self, channel_in, channel_out, kernel_size, output_size):
        """
        计算FNN卷积层的参数变量内存访问成本
        :return: 参数内存访问次数
        """
        return (channel_in * kernel_size * kernel_size + 1) * channel_out * output_size

    def Rd_Param_memory_access_FC(self, input_feature, output_feature):
        """
        计算FNN全连接层的参数变量内存访问成本
        :return: 参数内存访问次数
        """
        return (input_feature + 1) * output_feature
    
    def Wr_output_memory_access_Conv(self, channel_out, output_size):
        """
        计算FNN卷积层的输出变量内存访问成本
        :return: 输出内存访问次数
        """
        return channel_out * output_size
    def Wr_output_memory_access_FC(self, output_feature):
        """
        计算FNN全连接层的输出变量内存访问成本
        :return: 输出内存访问次数
        """
        return output_feature
    def Addressing_cost_Conv(self, channel_in, channel_out, input_size, output_size, kernel_size):
        """
        计算FNN卷积层的地址计算成本
        :return: 地址计算量 mac 和 acc
        """
        mac = 0
        acc = channel_in * input_size + channel_out * output_size + channel_out * kernel_size * kernel_size
        return mac, acc
    def Addressing_cost_FC(self, input_feature, output_feature):
        """
        计算FNN全连接层的地址计算成本
        :return: 地址计算量 mac 和 acc
        """
        mac = 0
        acc = input_feature + output_feature  
        return mac, acc
    def calculate_cost(self, layer_type):
        if layer_type == 'conv':
            Rd_count = self.Rd_Input_memory_access_Conv(self.channel_in, self.channel_out, self.output_size, self.kernel_size) + \
                        self.Rd_Param_memory_access_Conv(self.channel_in, self.channel_out, self.kernel_size, self.output_size)
            Wr_count = self.Wr_output_memory_access_Conv(self.channel_out, self.output_size)
            op_mac, op_acc = self.operational_cost_Conv(self.channel_in, self.channel_out, self.kernel_size, self.output_size)
            addr_mac, addr_acc = self.Addressing_cost_Conv(self.channel_in, self.channel_out, self.input_size, self.output_size, self.kernel_size)
        elif layer_type == 'fc':
            Rd_count = self.Rd_Input_memory_access_FC(self.input_feature) + \
                        self.Rd_Param_memory_access_FC(self.input_feature, self.output_feature)
            Wr_count = self.Wr_output_memory_access_FC(self.output_feature)
            op_mac, op_acc = self.operational_cost_FC(self.input_feature, self.output_feature)
            addr_mac, addr_acc = self.Addressing_cost_FC(self.input_feature, self.output_feature)
        else:
            raise ValueError('Invalid layer type')
        # 假设单次读写的内存访问成本相同
        # Rd_count 、 Wr_count 是内存访问次数，单次访问8bit即1Byte
        memory_access_cost = INTERP_FUNC((Rd_count + Wr_count)/1000)  # 转换为KB
        mac_cost = (op_mac + addr_mac) * ENERGY_PER_MAC
        acc_cost = (op_acc + addr_acc) * ENERGY_PER_ADD
        total_cost = memory_access_cost + mac_cost + acc_cost
        return ((mac_cost+acc_cost)/1000000000, memory_access_cost/1000000000)  # 转换为mJ