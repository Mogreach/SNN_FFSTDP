import numpy as np
import matplotlib.pyplot as plt
"""
左图
"""
# 定义目标函数
def target_function(x, y):
    return np.log(1 + np.exp(1.5 - x)) + np.log(1 + np.exp(y - 1.5))

# 定义二阶导数函数
def second_order_derivatives(x, y):
    d2f_dx2 = np.exp(1.5 - x) / (1 + np.exp(1.5 - x))**2
    d2f_dy2 = np.exp(y - 1.5) / (1 + np.exp(y - 1.5))**2
    return d2f_dx2, d2f_dy2

# 创建网格数据
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = target_function(X, Y)

# 计算二阶导数
d2f_dx2, d2f_dy2 = second_order_derivatives(X, Y)

# 创建图像
fig = plt.figure(figsize=(14, 6))

# 绘制 d2f/dx2 曲面图
ax1 = fig.add_subplot(121)
contour_dx2 = ax1.contourf(X, Y, d2f_dx2, levels=50, cmap='plasma')
# surf_dx2 = ax1.plot_surface(X, Y, d2f_dx2, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_title(r"$\frac{\partial^2 f}{\partial x^2}$", fontsize=14)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
# ax1.set_zlabel(r"$\frac{\partial^2 f}{\partial x^2}$", fontsize=12)
fig.colorbar(contour_dx2, ax=ax1, shrink=0.5, aspect=10, label="Value")

# 绘制 d2f/dy2 等高线图
ax2 = fig.add_subplot(122)
contour_dy2 = ax2.contourf(X, Y, d2f_dy2, levels=50, cmap='plasma')
# surf_dy2 = ax2.plot_surface(X, Y, d2f_dy2, cmap='viridis', edgecolor='none', alpha=0.9)
ax2.set_title(r"$\frac{\partial^2 f}{\partial y^2}$", fontsize=14)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("y", fontsize=12)
fig.colorbar(contour_dy2, ax=ax2, label="Value")
ax2.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
