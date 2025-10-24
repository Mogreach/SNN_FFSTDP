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
T = 16  # 时间步数
theta = 1.8  # 参数 theta

# 生成 T 个整数向量并归一化
y = torch.arange(0,T,step=0.01)
x = torch.arange(0, T, dtype=torch.float32) / T
x = x**2 * T
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