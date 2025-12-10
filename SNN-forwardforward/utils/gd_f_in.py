import numpy as np
import torch
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron
from scipy.interpolate import interp1d

def compute_f_in_to_grad_step_function(step_size=0.1):
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
    Z_grad = np.gradient(Z, x.numpy(), axis=1).mean(axis=1)

    # 对 f_in 分段
    step_bins = np.arange(0, 1 + step_size, step_size)  # 分段区间
    step_values = []  # 存储每段的平均梯度

    for i in range(len(step_bins) - 1):
        start, end = step_bins[i], step_bins[i + 1]
        mask = (f_in_values >= start) & (f_in_values < end)  # 找到对应的 f_in 范围
        avg_grad = Z_grad[mask].mean() if mask.any() else 0  # 计算该段平均梯度
        step_values.append(avg_grad)

    # 定义阶梯函数
    def grad_step_function(f_in):
        idx = np.digitize(f_in, step_bins) - 1  # 找到 f_in 对应的区间索引
        idx = np.clip(idx, 0, len(step_values) - 1)  # 防止越界
        return step_values[idx]

    return grad_step_function, step_bins, step_values

def compute_f_in_to_grad_function():
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
    Z_grad = np.gradient(Z, x.numpy(), axis=1).mean(axis=1)

    # 构造插值函数
    grad_interpolator = interp1d(f_in_values, Z_grad, kind='cubic', fill_value="extrapolate")

    return grad_interpolator

def plot_approximate_gradient_function():
    # 获取近似函数
    grad_func = compute_f_in_to_grad_function()

    # 给定某些 f_in 值，计算对应的梯度
    f_in_sample = np.array([0.1, 0.5, 0.8])
    gradient_values = grad_func(f_in_sample)

    # 打印结果
    for f_in, grad in zip(f_in_sample, gradient_values):
        print(f"f_in: {f_in}, Gradient: {grad}")

    # 绘制梯度近似函数
    f_in_range = np.linspace(0, 1, 100)
    approx_grad = grad_func(f_in_range)

    plt.plot(f_in_range, approx_grad, label="Approximate Gradient")
    plt.xlabel("$f_{in}$")
    plt.ylabel("$\\frac{\\partial f_r}{\\partial x_{i}}$")
    plt.title("Approximate Gradient Function")
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()
def plot_approximate_step_gradient_function():
    # 获取阶梯函数
    step_func, step_bins, step_values = compute_f_in_to_grad_step_function(step_size=0.05)

    # 打印每段的区间和对应的梯度值
    for i in range(len(step_bins) - 1):
        print(f"f_in in [{step_bins[i]:.2f}, {step_bins[i+1]:.2f}): Gradient = {step_values[i]:.4f}")

    # 可视化阶梯函数
    f_in_range = np.linspace(0, 1, 100)
    approx_grad = [step_func(f) for f in f_in_range]

    plt.step(f_in_range, approx_grad, where='post', label='Step Function Approximation')
    plt.xlabel('$f_{in}$')
    plt.ylabel('$\\frac{\\partial f_r}{\\partial x_{i}}$')
    plt.title('Step Function Approximation of Gradient')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()
plot_approximate_gradient_function()