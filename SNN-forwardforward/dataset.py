from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader, SubsetRandomSampler
from spikingjelly.datasets import play_frame
import numpy as np
import random
random.seed(123) # 保证多次实验，采用的val是相同的
batch_size = 1000


def normalize_frame(x):
    x = x.astype(np.float32)
    # N-MNIST frame 通常是 [T, C, H, W]，先沿时间维求和 -> [C, H, W]
    if x.ndim == 4:
        x = x.sum(axis=0)
    # 对整体 C/H/W 做统一归一化（全局 max）
    max_v = x.max()
    if max_v > 0:
        x = x / max_v
    return x

def nmnist():
    nmnist_train = NMNIST(
        root='./SNN-forwardforward/data/NMNIST',
        train=True,
        data_type='frame',
        frames_number=20,
        split_by='number',
        transform=normalize_frame
    )
    nmnist_test = NMNIST(
        root='./SNN-forwardforward/data/NMNIST',
        train=False,
        data_type='frame',
        frames_number=20,
        split_by='number',
        transform=normalize_frame
    )

    indices = [x for x in range(len(nmnist_train))]
    random.shuffle(indices)
    # np.save('indices_for_nmnist', indices)
    train_indices = indices[0:50000]
    val_indices = indices[50000:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, sampler=train_sampler)
    val_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, sampler=val_sampler)
    test_data_loader = DataLoader(dataset=nmnist_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
def dvs_gesture():

    print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
    print('resource, url, md5/n', CIFAR10DVS.resource_url_md5())

    print('DVS128Gesture downloadable', DVS128Gesture.downloadable())
    print('resource, url, md5/n', DVS128Gesture.resource_url_md5())

    root_dir = './SNN-forwardforward/data/DVSgesture'
    train_set = DVS128Gesture(root_dir, train=True, data_type='event')
    event, label = train_set[0]
    for k in event.keys():
        print(k, event[k])
    print('label', label)
    train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=32, split_by='number')
    frame, label = train_set[0]
    print(frame.shape)
dvs_gesture()