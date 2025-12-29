import sys
sys.path.append('D:/OneDrive/SNN_FFSTDP/SNN-forwardforward')
import matplotlib.pyplot as plt
import torch
from spikingjelly.datasets.n_mnist import NMNIST
import os
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision
from src.ff_snn_net import Net, spike_encoder
from src.generate_neg_sample import *
import torch.nn.functional as F
from config import ConfigParser
import numpy as np
from bitarray import bitarray
from hardware_sim_config import *
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

    #define NUM_TEST_LABELS {test_pics}

    const int test_labels[NUM_TEST_LABELS] = {{ {c_array} }};

    #endif // TEST_LABELS_H
    """
    # 写入文本文件
    with open(f"./Gen_out/{TASK}/test_labels.h", "w") as f:
        f.write(c_code)
    print("C header file 'test_labels.h' generated successfully!")
def gen_dataset_spike(train_pics,test_pics,T):
    config = ConfigParser()
    args = config.parse()
###########################################################################################
####################################前向学习的代码结构######################################
    # 初始化数据加载器
    # 加载训练集和测试集
    if TASK == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    elif TASK == "N-MNIST":
        train_dataset = NMNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        test_dataset = NMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    elif TASK == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    elif TASK == "CIFAR10":

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    else:
        raise ValueError("Unsupported dataset. Please choose either 'MNIST' or 'CIFAR10'.")

    # 划分训练集和验证集
    train_size = int(0.95 * len(train_dataset))  # 80% 用于训练
    val_size = len(train_dataset) - train_size  # 20% 用于验证
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=train_pics,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=test_pics,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    device = torch.device("cuda")
    net = Net(dims=[784,256, 10],tau=args.tau, epoch=args.epochs, T=T, lr=args.lr,
              v_threshold_pos=V_THRESHOLD_POS,v_threshold_neg=V_THRESHOLD_NEG, opt=args.opt, loss_threshold=THETA)
    net.load(NET_PATH)
    all_spikes = []
    test_acc = 0
    test_count = 0
    with torch.no_grad():
        for x, y in train_data_loader:
            test_count += 1 
            x = x.to(device)
            y = y.to(device)
            x_pos = overlay_y_on_x(x, y)
            y_neg = get_y_neg(y, device)
            x_neg = overlay_y_on_x(x, y_neg)
            predict_result , goodness,freq = net.predict_analyze(x)
            test_acc += predict_result.eq(y).cpu().float().mean().item()
            img_pos_spike = spike_encoder(x_pos, T)
            img_neg_spike = spike_encoder(x_neg, T)

            for b in range(x.shape[0]):
                all_spikes.append(img_pos_spike[:,b,:,:,:].cpu().flatten(1).numpy())
                all_spikes.append(img_neg_spike[:,b,:,:,:].cpu().flatten(1).numpy())
            if test_count == 50:
                break

    print("test Acc:", 100 * test_acc / test_count, "%")
    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            test_img = x_te
            test_label = y_te.numpy()
            # break
    for i in range(test_pics):
        for k in range(10):
            img_test = test_img[i:i+1]
            k = torch.tensor([k])
            img_encoded = overlay_y_on_x(img_test, k)
            # 获取脉冲数据
            img_encoded_spike = spike_encoder(img_encoded, T)
            # 将脉冲数据添加到列表中
            all_spikes.append(img_encoded_spike[:,0,:,:,:].cpu().flatten(1).numpy())
        print(f"Processed Test image {i}")
    gen_test_label(test_pics,test_label)
    

    # 将列表转换为 numpy 数组
    all_spikes = np.array(all_spikes).astype(int)  # 形状为 (2 * train_pics, 16, 784)

    # 保存为单个二进制文件
    save_spike_data(all_spikes, f"./Gen_out/{TASK}/all_spikes.bin")
    print("Saved all spike data to binary files")




def main():
    # sort_dataset()
    train_pics = 1000
    test_pics = 10000

    size = 784
    gen_dataset_spike(train_pics,test_pics,T)

if __name__ == "__main__":
    main()