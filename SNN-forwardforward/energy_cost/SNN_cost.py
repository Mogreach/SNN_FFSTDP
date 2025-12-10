import math
from network_config import *
class SNNEnergyCost:
    """
    脉冲神经网络操作成本估计类（用于SNN）
    :param channel_out: 输出通道数
    :param kernel_size: 卷积核大小
    :param input_size: 输入特征图大小，HxW
    :param output_size: 输出特征图大小，HxW
    :param input_feature: 输入神经元数量
    :param output_feature: 输出神经元数量
    :param stride: 步长
    :param timesteps: 时间步数
    :param spike_in_count: 输入脉冲数
    :param spike_out_count: 输出脉冲数
    """
    def __init__(self, input_size, output_size, channel_in, channel_out, kernel_size, stride,
                 timesteps, spike_in_count, spike_out_count, input_feature=None, output_feature=None):
        self.input_size = input_size
        self.output_size = output_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.timesteps = timesteps
        self.spike_in_count = spike_in_count
        self.spike_out_count = spike_out_count
        self.input_feature = input_feature if input_feature is not None else input_size * channel_in
        self.output_feature = output_feature if output_feature is not None else output_size * channel_out




    def operational_cost_Conv(self, channel_out, kernel_size, output_size, stride, timesteps, spike_in_count, spike_out_count):
        """
        计算SNN卷积层的操作成本
        :return: MACs (乘加操作数) 以及 ACCs (加法操作数)
        """
        mac = timesteps * channel_out * output_size
        acc = spike_in_count * math.floor(kernel_size/stride) * math.floor(kernel_size/stride) * \
                channel_out + timesteps * channel_out * output_size + spike_out_count
        return mac, acc

    def operational_cost_FC(self, input_feature, output_feature, timesteps, spike_in_count):
        """
        计算SNN全连接层的操作成本
        :return: MACs (乘加操作数) 以及 ACCs (加法操作数)
        """
        mac = timesteps * output_feature
        acc = spike_in_count  * output_feature + timesteps * output_feature
        return mac, acc

    def Rd_Input_memory_access(self, spike_in_count):
        """
        计算SNN的输入变量内存访问成本(FC和Conv层相同)
        :return: 输入内存访问次数
        """
        return spike_in_count
    
    def Rd_Param_memory_access_Conv(self, channel_out, kernel_size, output_size, spike_in_count, channel_in):
        """
        计算SNN卷积层的参数变量内存访问成本
        :return: 参数内存访问次数
        """
        return channel_in * channel_out * kernel_size * kernel_size + channel_out * output_size
        # return   spike_in_count * channel_out * kernel_size * kernel_size + channel_out * output_size

    def Rd_Param_memory_access_FC(self, output_feature, spike_in_count):
        """
        计算SNN全连接层的参数变量内存访问成本
        :return: 参数内存访问次数
        """
        # return output_feature + output_feature
        return   spike_in_count * output_feature + output_feature

    def Rd_mem_memory_access_Conv(self, spike_in_count, channel_out, kernel_size, output_size, timesteps):
        """
        计算SNN卷积层的膜电位内存访问成本
        :return: 内存访问次数
        """
        return timesteps * channel_out * output_size
        # return (spike_in_count * channel_out * kernel_size * kernel_size)/2 + channel_out * output_size

    def Rd_mem_memory_access_FC(self, spike_in_count, output_feature, timesteps):
        """
        计算SNN全连接层的膜电位内存访问成本
        :return: 内存访问次数
        """
        return timesteps * output_feature
        # return spike_in_count * output_feature + output_feature
    def Wr_output_memory_access(self, spike_out_count):
        """
        计算SNN的输出变量内存访问成本(FC和Conv层相同)
        :return: 输出内存访问次数
        """
        return spike_out_count
    def Wr_mem_memory_access_Conv(self, channel_out, kernel_size, output_size, spike_in_count, timesteps):
        """
        计算SNN卷积层的膜电位输出内存访问成本
        :return: 内存访问次数
        """
        return timesteps * channel_out * output_size
        # return (spike_in_count * channel_out * kernel_size * kernel_size)/2 + channel_out * output_size

    def Wr_mem_memory_access_FC(self, output_feature, spike_in_count, timesteps):
        """
        计算SNN全连接层膜电位输出内存访问成本
        :return: 内存访问次数
        """
        return timesteps * output_feature
        # return spike_in_count * output_feature + output_feature

    def Addressing_cost_Conv(self, channel_out, kernel_size, spike_in_count):
        """
        计算SNN卷积层的地址计算成本
        :return: 地址计算量 mac 和 acc
        """
        mac = 2 * spike_in_count
        acc = spike_in_count * channel_out * kernel_size * kernel_size
        return mac, acc

    def Addressing_cost_FC(self, output_feature, spike_in_count):
        """
        计算SNN全连接层的地址计算成本
        :return: 地址计算量 mac 和 acc
        """
        mac = 0
        acc = spike_in_count * output_feature
        return mac, acc
    def calculate_cost(self, layer_type):
        if layer_type == 'conv':
            Rd_count = self.Rd_Input_memory_access(self.spike_in_count) + \
                        self.Rd_Param_memory_access_Conv(self.channel_out, self.kernel_size, self.output_size, self.spike_in_count, self.channel_in) + \
                        self.Rd_mem_memory_access_Conv(self.spike_in_count, self.channel_out, self.kernel_size, self.output_size, self.timesteps)
            Wr_count = self.Wr_output_memory_access(self.spike_out_count) + \
                        self.Wr_mem_memory_access_Conv(self.channel_out, self.kernel_size, self.output_size, self.spike_in_count, self.timesteps)
            
            op_mac, op_acc = self.operational_cost_Conv(self.channel_out, self.kernel_size, self.output_size, self.stride, self.timesteps, self.spike_in_count, self.spike_out_count)
            addr_mac, addr_acc = self.Addressing_cost_Conv(self.channel_out, self.kernel_size, self.spike_in_count)
        elif layer_type == 'fc':
            Rd_count = self.Rd_Input_memory_access(self.spike_in_count) + \
                        self.Rd_Param_memory_access_FC(self.output_feature, self.spike_in_count) + \
                        self.Rd_mem_memory_access_FC(self.spike_in_count, self.output_feature, self.timesteps)
            Wr_count = self.Wr_output_memory_access(self.spike_out_count) + \
                        self.Wr_mem_memory_access_FC(self.output_feature, self.spike_in_count, self.timesteps)
            
            op_mac, op_acc = self.operational_cost_FC(self.input_feature, self.output_feature, self.timesteps, self.spike_in_count)
            addr_mac, addr_acc = self.Addressing_cost_FC(self.output_feature, self.spike_in_count)
        else:
            raise ValueError('Invalid layer type')
        # 假设单次读写的内存访问成本相同
        # Rd_count 、 Wr_count 是内存访问次数，单次访问SRAM_SIZE的SRAM
        memory_access_cost =(Rd_count + Wr_count) * INTERP_FUNC(SRAM_SIZE) #单位pJ
        print(f"SNN:{Rd_count, Wr_count}")
        print(f"SNN:{op_mac, op_acc, addr_mac, addr_acc}")
        mac_cost = (op_mac + addr_mac) * ENERGY_PER_MAC #单位pJ
        acc_cost = (op_acc + addr_acc) * ENERGY_PER_ADD #单位pJ
        total_cost = memory_access_cost + mac_cost + acc_cost
        return ((mac_cost+acc_cost)*1e-9, memory_access_cost*1e-9)  # 转换为mJ


        class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

