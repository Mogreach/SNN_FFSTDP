import torch
import matplotlib.pyplot as plt

def pos_derivative(x, theta):
    """
    计算 log(1 + exp(-x + theta)) 关于 x 的导数。
    """
    sigmoid = -1 / (1 + torch.exp(x - theta))
    return sigmoid

def neg_derivative(y, theta):
    """
    计算 log(1 + exp(y - theta)) 关于 y 的导数。
    """
    sigmoid = 1 / (1 + torch.exp(theta - y))
    return sigmoid

# 参数设置
T = 8  # 时间步数
theta = 0  # 参数 theta

# 生成 T 个整数向量并归一化
y = torch.arange(-T,T,step=0.01)
x = torch.arange(-T, T, dtype=torch.float32) / T
x = x**2 * T * torch.sign(x)
# 计算 pos_derivative 和 neg_derivative
pos_ideal = -pos_derivative(y, theta)
neg_ideal = -neg_derivative(y, theta)

pos_values = -pos_derivative(x, theta)
neg_values = -neg_derivative(x, theta)

# 绘制对比图
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), pos_values.numpy(), label="pos_derivative", color="blue")
plt.plot(x.numpy(), neg_values.numpy(), label="neg_derivative", color="red")
plt.plot(y.numpy(), pos_ideal.numpy(), label="pos_ideal", color="cyan", linestyle="--")
plt.plot(y.numpy(), neg_ideal.numpy(), label="neg_ideal", color="orange", linestyle="--")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # 添加零线
plt.title("Comparison of pos_derivative and neg_derivative")
plt.xlabel("x (normalized)")
plt.ylabel("Derivative Value")
plt.legend()
plt.grid(True)
plt.show()

# 脉冲总数范围
pulse_counts = torch.arange(-8, 9)  # 从 -8 到 8 的整数

# 创建一个包含两个子图的图形
fig, axes = plt.subplots(2, 1, figsize=(10, 16))

# 绘制正goodness的对比图
for pulse in pulse_counts:  
    weighted_pos = pos_values * pulse
    axes[0].plot(x.numpy(), weighted_pos.numpy(), label=f"pos_derivative (Input Spike={pulse})", alpha=0.5)

axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)  # 添加零线
axes[0].set_title("Weighted pos_derivative (Positive Pulses)")
axes[0].set_xlabel("Output Goodness")
axes[0].set_ylabel("Weighted Derivative Value")
axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
axes[0].grid(True)

# 绘制负goodness的对比图
for pulse in pulse_counts: 
    weighted_neg = neg_values * pulse
    axes[1].plot(x.numpy(), weighted_neg.numpy(), label=f"neg_derivative (Input Spike={pulse})", alpha=0.5)

axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)  # 添加零线
axes[1].set_title("Weighted neg_derivative (Negative Pulses)")
axes[1].set_xlabel("Output Goodness")
axes[1].set_ylabel("Weighted Derivative Value")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
axes[1].grid(True)

# 调整布局并显示图形
plt.tight_layout()
plt.show()