from brian2 import *
import brian2.numpy_ as np
import h5py
from bitarray import bitarray
def get_label_neg(label):
    # 生成0-10的所有可能值
    possible_values = np.arange(10)
    # 排除输入的值
    possible_values = possible_values[possible_values != label]
    fake_label = np.random.choice(possible_values)
    # 随机选择一个不同的值
    return fake_label
def overlay_label_on_img(img, label):
    # 找到数组的最大值
    max_value = np.max(img)
    # 将第一行第y列的值设置为最大值
    img[0, label] = max_value
    return img
class SNN():
    def __init__(self):
        self.input_size = 784
        self.rates = zeros(self.input_size)
        # input neurons
        inp = NeuronGroup(self.input_size, 'v:1', method='exact', threshold='v>=1',
                          reset='v=0',name="input")
        def update_volt():
            inp.v += self.rates
        network_op = NetworkOperation(update_volt, dt=1.0 * ms,name="network_op")
        # hidden neurons
        self.net = Network(network_op,inp)
        self.net.store()

    def set_input(self,img_array):
        self.rates = img_array / 255.0
        self.net.set_states({"input":{"v":zeros(self.input_size)}})
    def get_input_spike(self,img,T):
        # 定义输入神经元组
        input_size = self.input_size    
        # 添加脉冲监视器
        spike_monitor = SpikeMonitor(self.net["input"], name='input_spikes')
        spikemon_list = [spike_monitor]
        self.net.add(spikemon_list)
        # 运行模拟
        sim_time = T * ms  # 16 个时间步
        self.set_input(img)
        self.net.run(sim_time)
        # 获取脉冲数据
        spike_times = spike_monitor.t   #对应时刻
        spike_indices = spike_monitor.i #神经元序号

        # 将脉冲数据整理为每个时间步的脉冲情况
        time_steps = int(sim_time / self.net["network_op"].clock.dt)
        spike_data = np.zeros((time_steps, input_size), dtype=bool)

        for t, i in zip(spike_times, spike_indices):
            time_step = int(t / self.net["network_op"].clock.dt)
            spike_data[time_step, i] = True
        self.net.remove(spikemon_list)
        self.net.restore()
        return spike_data
def sort_dataset():
    f = h5py.File("./HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()

    f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()
    # 按 0-9 的顺序分组排列
    def group_by_label(images, labels):
        # 初始化一个字典，用于存储每个类别的样本
        label_to_images = {i: [] for i in range(10)}
        
        # 遍历数据，将每个类别的样本存储到字典中
        for idx, lbl in enumerate(labels):
            label_to_images[lbl].append(images[idx])
        a = []
        for lbl in range(10):
            a.append(len(label_to_images[lbl])) 
        # 找到最小样本数量，确保每组都包含 10 个类别
        min_samples = min(a)
        
        # 初始化存储分组后的数据
        grouped_images = []
        grouped_labels = []
        
        # 按顺序构建完整组
        for i in range(min_samples):
            for lbl in range(10):
                grouped_images.append(label_to_images[lbl][i])
                grouped_labels.append(lbl)
        
        # 转换为 numpy 数组
        grouped_images = np.array(grouped_images)
        grouped_labels = np.array(grouped_labels)
        
        return grouped_images, grouped_labels
    # 按标签 0-9 重新排序
    def sort_by_label(images, labels):
        # 初始化一个字典，用于存储每个类别的索引
        label_to_indices = {i: [] for i in range(10)}
        
        # 遍历标签，记录每个类别的索引
        for idx, lbl in enumerate(labels):
            label_to_indices[lbl].append(idx)
        
        # 按类别顺序合并索引
        sorted_indices = []
        for lbl in range(10):
            sorted_indices.extend(label_to_indices[lbl])
        
        # 根据排序后的索引重新排列图像和标签
        sorted_images = images[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        return sorted_images, sorted_labels

    # 对训练集进行排序
    sorted_img, sorted_label = sort_by_label(img, label)
    # 打印结果
    print("排序后的标签:", sorted_label[:100])  # 打印前100个标签，检查是否按类别排序
    train_type_index=[]
    cnt_num = np.array(0)
    # 验证排序是否正确
    for lbl in range(10):
        train_type_index.append(cnt_num.item())
        num = np.sum(sorted_label == lbl)
        print(f"类别 {lbl} 的数量:", num)
        cnt_num += num
    # 保存排序后的数据到新的 HDF5 文件
    with h5py.File("./HDF5_MNIST_TRAIN_SORTED.h5", 'w') as f:
        f.create_dataset("img", data=sorted_img)
        f.create_dataset("label", data=sorted_label)
    
    
    # 对训练集进行分组排列
    grouped_img, grouped_label = group_by_label(img, label)
    # 验证分组是否正确
    for i in range(0, len(grouped_label), 10):
        print(f"第 {i//10 + 1} 组的标签:", grouped_label[i:i+10])
    # 保存分组后的数据到新的 HDF5 文件
    with h5py.File("./HDF5_MNIST_TRAIN_GROUPED.h5", 'w') as f:
        f.create_dataset("img", data=grouped_img)
        f.create_dataset("label", data=grouped_label)
def save_spike_data(spike_data, filename):
    """
    按照时间步长顺序保存脉冲数据到二进制文件，每个值占 1 位。

    参数:
        spike_data (np.ndarray): 形状为 (N, 16, 784) 的数组。
        filename (str): 保存的文件名。
    """
    # 将布尔数组转换为 uint8 类型
    # spike_data = spike_data.astype(np.uint8)
    # 将数据打包为二进制
    # packed_data = np.packbits(spike_data, axis=-1)
    # 将数据保存为二进制文件
    bits = bitarray()
    with open(filename, "wb") as f:
        for sample in spike_data:  # 遍历每个样本
            for time_step in range(sample.shape[0]):  # 遍历每个时间步
                # 将当前时间步的所有脉冲数据转换为二进制位
                bits.extend(sample[time_step, :])

    # 将 bitarray 保存为二进制文件
    with open(filename, "wb") as f:
        bits.tofile(f)

    print(f"Saved spike data to {filename}")

def gen_test_label(test_pics,test_label):
    # 选取前 n 个标签
    test_label_n = test_label[:test_pics]

    # 生成 C 语言数组的字符串
    c_array = ", ".join(map(str, test_label_n))  # 转换为 "1, 2, 3, ..." 格式

    # 定义 C 语言头文件内容
    c_code = f"""#ifndef TEST_LABELS_H
    #define TEST_LABELS_H

    #define NUM_TEST_LABELS 100

    const int test_labels[NUM_TEST_LABELS] = {{ {c_array} }};

    #endif // TEST_LABELS_H
    """
    # 写入文本文件
    with open("test_labels.h", "w") as f:
        f.write(c_code)
    print("C header file 'test_labels.h' generated successfully!")
def gen_dataset_spike(train_pics,test_pics,T):
    f = h5py.File("./HDF5_MNIST_TRAIN_GROUPED.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()

    f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()

    snn = SNN()
    all_spikes = []
    gen_test_label(test_pics,test_label)

    for index in range(train_pics):
        img_neg = img[index].copy()
        img_pos = overlay_label_on_img(img[index], label[index])
        label_neg = get_label_neg(label[index])
        img_neg = overlay_label_on_img(img_neg, label_neg)
        # 获取脉冲数据
        img_pos_spike = snn.get_input_spike(img_pos.flatten(), T)
        img_neg_spike = snn.get_input_spike(img_neg.flatten(), T)

        # 将脉冲数据添加到列表中
        all_spikes.append(img_pos_spike)
        all_spikes.append(img_neg_spike)

        print(f"Processed Train image {index}")
    
    for i in range(test_pics):
        for k in range(10):
            img_test = test_img[i].copy()
            img_encoded = overlay_label_on_img(img_test, k)
            # 获取脉冲数据
            img_encoded_spike = snn.get_input_spike(img_encoded.flatten(), T)
            # 将脉冲数据添加到列表中
            all_spikes.append(img_encoded_spike)
        print(f"Processed Test image {i}")

    # 将列表转换为 numpy 数组
    all_spikes = np.array(all_spikes)  # 形状为 (2 * train_pics, 16, 784)

    # 保存为单个二进制文件
    save_spike_data(all_spikes, "all_spikes.bin")
    print("Saved all spike data to binary files")




def main():
    # sort_dataset()
    train_pics = 1000
    test_pics = 100
    T = 16
    size = 784
    gen_dataset_spike(train_pics,test_pics,T)

if __name__ == "__main__":
    main()