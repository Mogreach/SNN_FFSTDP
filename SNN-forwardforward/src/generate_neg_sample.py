import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import torch
def stdnorm (x, dims = [1,2,3]):
    
    x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

    return x
def minmax_norm(x, dims=[1,2,3], eps=1e-10):
    x_min = torch.amin(x, dim=dims, keepdim=True)
    x_max = torch.amax(x, dim=dims, keepdim=True)
    x = (x - x_min) / (x_max - x_min + eps)
    return x
def generate_pos_n_neg_sample(x, y, num_classes=10):

    # One-hot编码标签叠加在输入向量前10个像素位置
    # x_pos = overlay_y_on_x(x, y, classes=num_classes)
    # y_neg = get_y_neg(y, x.device)
    # x_neg = overlay_y_on_x(x, y_neg, classes=num_classes)

    # 负样本独0码标签
    # x_pos = overlay_label_on_x(x)
    # y_neg = get_y_neg(y, device)
    # x_neg = overlay_y_on_x(x, y_neg)
    # x_neg = overlay_zero_on_x(x,y)

    # Mask掩码
    # x_pos = x
    # x_neg = generate_negative_samples_continuous(x_pos, y, train_dataset.dataset, device=device, visualize=False)
    # x_pos = overlay_label_on_x(x_pos)
    # x_neg = overlay_zero_on_x(x_neg,y)

    # SCFF方式
    p = 1
    batch_size = x.shape[0]
    x_pos = x + x
    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)
    batch_negs = []
    for i in random_indices:
        x_neg = x[(labeles+i)%batch_size]
        batch_neg = x + x_neg
        batch_negs.append(batch_neg)
    x_neg = torch.cat(batch_negs)

    x_pos = minmax_norm(x_pos, dims = [2,3])
    x_neg = minmax_norm(x_neg, dims = [2,3])
    return x_pos, x_neg


def get_y_neg(y, device):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(10))
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)
def overlay_y_on_x(x, y, classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, :, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        label = y[i].item()  # y[i]是该样本的标签
        # 确保标签在0到9之间（根据设置的 classes）
        # 将第一通道前10个像素位置中对应标签的像素赋值为最大值
        x_[i, :, 0, label] = (
            x_.max()
        )  # 将每个样本前10个像素中，对应标签类别序号赋为当前矩阵最大值
    return x_
def overlay_label_on_x(x, classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, :, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    x_[:, :, 0, :classes] += 1
    return x_

def overlay_zero_on_x(x, y, classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, :, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    x_[:, :, 0, :classes] += 1.0
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        label = y[i].item()  # y[i]是该样本的标签
        # 确保标签在0到9之间（根据设置的 classes）
        # 将第一通道前10个像素位置中对应标签的像素赋值为最大值
        x_[i, :, 0, label] = 0  # 将每个样本前10个像素中，对应标签类别序号赋为当前矩阵最大值
    return x_
def generate_continuous_mask(shape, block_scale=8, smooth=True, device='cpu'):
    """
    生成连续大块的二值掩码。
    参数:
        shape: [1, 1, H, W] 图像形状
        block_scale: 块的粗糙度，越大 → 块越大、越连续
        smooth: 是否使用模糊平滑
    返回:
        mask: [1, 1, H, W] 二值掩码（0或1）
    """
    B, C, H, W = shape
    # 1️⃣ 低分辨率随机噪声
    low_H, low_W = H // block_scale, W // block_scale
    noise = torch.rand((B, C, low_H, low_W), device=device)

    # 2️⃣ 上采样放大 → 产生大片区域
    mask = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)

    # 3️⃣ 平滑模糊（可选）
    if smooth:
        kernel = torch.ones((1, 1, 5, 5), device=device) / 25.0
        mask = F.conv2d(mask, kernel, padding=2)

    # 4️⃣ 阈值化
    threshold = random.uniform(0.4, 0.6)
    # threshold = np.random.uniform(0.4, 0.6)

    mask = (mask > threshold).float()

    return mask

def generate_negative_samples_continuous(x, y, dataset, num_classes=10, device='cpu', visualize=False,
                                         block_scale=3, smooth=False):
    """
    向量化生成负样本（保证标签对齐且不同）
    输入/输出接口保持不变：
      x: (B,1,H,W)
      y: (B,)
      dataset: MNIST dataset (dataset.data, dataset.targets) or equivalent
    返回:
      neg_samples: (B,1,H,W)
    """
    B = x.size(0)
    H, W = x.size(2), x.size(3)

    # 把dataset.targets和dataset.data准备成tensor（注意内存，MNIST可以直接放GPU；若数据很大请在CPU上索引）
    # 我把targets放device上，但 data 先保持在 cpu，再按索引一次性送到 device（避免一次性移动全部数据造成不必要内存占用）
    targets = torch.tensor(dataset.targets, device=device)  # (N,)
    data_cpu = dataset.data.float() / 255.0  # still on cpu, shape (N,28,28)

    # === 1) 为每个样本生成 neg_label，保证 != y ===
    # 用随机偏移 1..(num_classes-1)：
    offsets = torch.randint(1, num_classes, (B,), device=device)
    neg_labels = (y.to(device) + offsets) % num_classes  # shape (B,), guaranteed != y

    # === 2) 为每个 batch 位置选择一个该类别的随机索引（并保证顺序对应） ===
    neg_indices = torch.empty(B, dtype=torch.long, device=device)

    # 对每个类别挑选需要填充的位置并从该类别索引数组中随机选取
    for lbl in range(num_classes):
        pos_mask = (neg_labels == lbl)  # 哪些 batch 需要该类别样本
        cnt = int(pos_mask.sum().item())
        if cnt == 0:
            continue
        # 数据集中属于该类别的所有索引（在 cpu 上）
        class_indices_cpu = (dataset.targets == lbl).nonzero(as_tuple=True)[0]
        if len(class_indices_cpu) == 0:
            raise RuntimeError(f"No samples for class {lbl} in dataset.")
        # 在 class_indices_cpu 中随机采样 cnt 个索引（允许重复）
        sel = torch.randint(0, len(class_indices_cpu), (cnt,), device=device)
        chosen_cpu = class_indices_cpu[sel.cpu()].long()  # bring to cpu idx values
        # 将它们放到 neg_indices 的对应 batch 位置
        neg_indices[pos_mask] = chosen_cpu.to(device)

    # === 3) 用 neg_indices 在 data_cpu 上索引并转移到 device ===
    # 注意：data_cpu[neg_indices.cpu()] 得到 tensor (B, H, W) 在 cpu，再 unsqueeze -> move to device
    neg_img = data_cpu[neg_indices.cpu()].unsqueeze(1).to(device)  # (B,1,H,W)

    # === 4) 生成连续掩码（批量化）并混合 ===
    mask = generate_continuous_mask((B, 1, H, W), block_scale=block_scale, smooth=smooth, device=device)
    neg_samples = x.to(device) * mask + neg_img * (1.0 - mask)

    # === 5) 可视化（保持原先 show 行为）===
    if visualize:
        import matplotlib.pyplot as plt
        n_vis = min(3, B)
        for i in range(n_vis):
            fig, axs = plt.subplots(1, 4, figsize=(8, 2))
            axs[0].imshow(x[i].cpu().squeeze(), cmap='gray')
            axs[0].set_title(f'Positive ({int(y[i].item())})')
            axs[1].imshow(neg_img[i].cpu().squeeze(), cmap='gray')
            axs[1].set_title(f'NegClass ({int(neg_labels[i].item())})')
            axs[2].imshow(mask[i].cpu().squeeze(), cmap='gray')
            axs[2].set_title('Mask')
            axs[3].imshow(neg_samples[i].cpu().squeeze(), cmap='gray')
            axs[3].set_title('Mixed (Negative)')
            for ax in axs: ax.axis('off')
            plt.show()

    return neg_samples

if __name__ == "__main__":
    # 加载MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 随机取一个batch
    x, y = next(iter(torch.utils.data.DataLoader(mnist_train, batch_size=8, shuffle=True)))

    # 生成负样本
    neg_samples = generate_negative_samples_continuous(x, y, mnist_train, device='cpu', visualize=True)
