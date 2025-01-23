import numpy as np
import torch
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
def plot_x():
    plt.rcParams['figure.dpi'] = 200
    if_node = neuron.IFNode(v_reset=None)
    T = 128
    x = torch.arange(-0.2, 1.2, 0.04)
    plt.scatter(torch.arange(x.shape[0]), x)
    plt.title('Input $x_{i}$ to IF neurons')
    plt.xlabel('Neuron index $i$')
    plt.ylabel('Input $x_{i}$')
    plt.grid(linestyle='-.')
    plt.show()
    s_list = []
    for t in range(T):
        s_list.append(if_node(x).unsqueeze(0))

    out_spikes = np.asarray(torch.cat(s_list))
    visualizing.plot_1d_spikes(out_spikes, 'IF neurons\' spikes and firing rates', 't', 'Neuron index $i$')
    plt.show()
    plt.subplot(1, 2, 1)
    firing_rate = np.mean(out_spikes, axis=0)
    plt.plot(x, firing_rate)
    plt.title('Input $x_{i}$ and firing rate')
    plt.xlabel('$x_{i}$')
    plt.ylabel('$f_r$')
    plt.grid(linestyle='-.')

    plt.subplot(1, 2, 2)
    plt.plot(x, x.relu())
    plt.title('Input $x_{i}$ and ReLU($x_{i}$)')
    plt.xlabel('$x_{i}$')
    plt.ylabel('ReLU($x_{i}$)')
    plt.grid(linestyle='-.')
    plt.show()

def plot_x_vs_firing_rate():
    plt.rcParams['figure.dpi'] = 200
    # 设置参数
    T = 128  # 总时间步
    x = torch.arange(-0.2, 1.2, 0.04)  # 输入值
    z = torch.zeros(x.shape)
    if_node = neuron.IFNode(v_reset=None)  # 定义脉冲神经元

    # 定义频率变量 f_in 的范围
    f_in_values = np.linspace(0, 1, 10)  # 在 0 到 1 之间取 6 个频率值

    # 准备绘图
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(f_in_values)))

    for i, f_in in enumerate(f_in_values):
        s_list = []
        active_steps = int(f_in * T)  # 有效时间步数
        for t in range(T):
            if t < active_steps:  # 有效时间步，使用输入 x
                s_list.append(if_node(x).unsqueeze(0))
            else:  # 非有效时间步，使用输入 z (即全 0)
                s_list.append(if_node(z).unsqueeze(0))
        
        # 计算输出脉冲
        out_spikes = np.asarray(torch.cat(s_list))
        firing_rate = np.mean(out_spikes, axis=0)  # 计算发放率

        # 绘制输入 x 和发放率的关系
        plt.plot(x, firing_rate, label=f'f_in = {f_in:.2f}', color=colors[i])

    # 添加图例和标签
    plt.title('Relationship between $x$, $f_{in}$, and $f_r$')
    plt.xlabel('$x_{i}$')
    plt.ylabel('$f_r$')
    plt.legend(loc='upper left')
    plt.grid(linestyle='-.')
    plt.show()
def plot_x_vs_firing_rate_n_f_in():
    # 设置参数
    T = 128  # 总时间步
    x = torch.arange(-0.2, 1.2, 0.04)  # 输入值范围
    f_in_values = np.linspace(0, 1, 50)  # f_in 的取值范围
    if_node = neuron.IFNode(v_reset=None)  # 定义脉冲神经元

    # 创建网格
    X, F_IN = np.meshgrid(x, f_in_values)
    Z = np.zeros_like(X)  # 存储 firing_rate 的值

    # 计算 firing_rate
    for i, f_in in enumerate(f_in_values):
        s_list = []
        active_steps = int(f_in * T)  # 有效时间步数
        for t in range(T):
            if t < active_steps:  # 有效时间步，使用输入 x
                s_list.append(if_node(torch.tensor(X[i, :], dtype=torch.float32)).unsqueeze(0))
            else:  # 非有效时间步，输入为 0
                s_list.append(if_node(torch.zeros_like(torch.tensor(X[i, :]))).unsqueeze(0))
        
        # 计算发放率
        out_spikes = torch.cat(s_list).numpy()
        Z[i, :] = np.mean(out_spikes, axis=0)  # 平均发放率

    # 绘制 3D 图像
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(F_IN, X, Z, cmap='viridis', edgecolor='k', alpha=0.8)

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # 设置轴标签和标题
    ax.set_title('3D Plot: $f_{in}$, $x_i$, and $f_r$')
    ax.set_xlabel('$f_{in}$')
    ax.set_ylabel('$x_i$')
    ax.set_zlabel('$f_r$')

    plt.show()

def plot_f_in_vs_gd():
    # 设置参数
    T = 128  # 总时间步
    x = torch.arange(-0.2, 1.2, 0.04)  # 输入值范围
    f_in_values = np.linspace(0, 1, 50)  # f_in 的取值范围
    if_node = neuron.IFNode(v_reset=None)  # 定义脉冲神经元

    # 创建网格
    X, F_IN = np.meshgrid(x, f_in_values)
    Z = np.zeros_like(X)  # 存储 firing_rate 的值

    # 计算 firing_rate
    for i, f_in in enumerate(f_in_values):
        s_list = []
        active_steps = int(f_in * T)  # 有效时间步数
        for t in range(T):
            if t < active_steps:  # 有效时间步，使用输入 x
                s_list.append(if_node(torch.tensor(X[i, :], dtype=torch.float32)).unsqueeze(0))
            else:  # 非有效时间步，输入为 0
                s_list.append(if_node(torch.zeros_like(torch.tensor(X[i, :]))).unsqueeze(0))
        
        # 计算发放率
        out_spikes = torch.cat(s_list).numpy()
        Z[i, :] = np.mean(out_spikes, axis=0)  # 平均发放率

    # 计算 firing_rate 关于 x 的导数
    Z_grad = (np.gradient(Z, x.numpy(), axis=1)).mean(axis = 1)
    # 绘制图像
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制导数曲线
    ax.plot(f_in_values, Z_grad, alpha=0.5)

    # 设置轴标签和标题
    ax.set_title('Derivative with Respect to Input Fequency')
    ax.set_xlabel('$f_{in}$')
    ax.set_ylabel('$\\frac{\\partial f_r}{\\partial x_{i}}$')
    ax.grid(linestyle='--', alpha=0.6)

    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    plt.tight_layout()
    plt.show()
# plot_x()
# plot_x_vs_firing_rate()
# plot_x_vs_firing_rate_n_f_in()
plot_f_in_vs_gd()