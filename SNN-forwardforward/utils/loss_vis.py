import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 定义目标函数
def target_function(x, y):
    return np.log(1 + np.exp(1.5 - x)) + np.log(1 + np.exp(y - 1.5))

# 定义梯度函数
def func_gradient(x, y):
    grad_x = -np.exp(1.5 - x) / (1 + np.exp(1.5 - x))
    grad_y = np.exp(y - 1.5) / (1 + np.exp(y - 1.5))
    return grad_x, grad_y

# 创建网格数据
x = np.linspace(-5, 5, 200)  # 增加分辨率
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = target_function(X, Y)

# 计算梯度
grad_x, grad_y = func_gradient(X, Y)
magnitude = np.sqrt(grad_x**2 + grad_y**2)  # 梯度大小

# 创建图像
fig = plt.figure(figsize=(14, 6))

# 绘制 3D 曲面图
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.9)
ax1.view_init(elev=45, azim=135)  # 调整观察角度
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label="Function Value")
ax1.set_title("3D Plot of f(x, y)", fontsize=14)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.set_zlabel("z", fontsize=12)

# 绘制梯度场图
ax2 = fig.add_subplot(122)
norm = Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))
sm = ScalarMappable(cmap='cool', norm=norm)
sm.set_array(magnitude)
ax2.quiver(X[::10, ::10], Y[::10, ::10], grad_x[::10, ::10], grad_y[::10, ::10],  # 调整密度
           magnitude[::10, ::10], cmap='cool', scale=50, pivot='middle')
fig.colorbar(sm, ax=ax2, label="Gradient Magnitude")
ax2.set_title("Gradient Field of f(x, y)", fontsize=14)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("y", fontsize=12)
ax2.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
