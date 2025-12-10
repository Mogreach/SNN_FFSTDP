import numpy as np
import matplotlib.pyplot as plt

# 加载保存的脉冲数据
v_t_array = np.load("v_t_array.npy")  # 神经元电压随时间变化
s_t_array = np.load("s_t_array.npy")  # 神经元脉冲随时间变化

# 设置图像大小
plt.figure(figsize=(10, 6))

# 绘制神经元电压图像
plt.subplot(2, 1, 1)  # 创建上下两个子图，1代表电压图
plt.imshow(v_t_array, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='Voltage')
plt.title('Neuron Membrane Potential Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Neurons')

# 绘制脉冲图像
plt.subplot(2, 1, 2)  # 2代表脉冲图
plt.imshow(s_t_array, aspect='auto', cmap='binary', interpolation='nearest')
plt.colorbar(label='Spike (0 or 1)')
plt.title('Neuron Spikes Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Neurons')

# 调整布局
plt.tight_layout()

# 保存图像到本地
plt.savefig("neuron_activity.png", dpi=300)

# 显示图像
plt.show()
