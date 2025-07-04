import torch
import matplotlib.pyplot as plt
import seaborn as sns


def get_bin_indices(coords, bins, N):
    indices = torch.bucketize(coords, bins) - 1
    indices = torch.clamp(indices, 0, N - 1)
    return indices


def softmax_with_temperature(probs, temperature=10.0):
    logits = torch.log(probs + 1e-8)  # 防止 log(0)
    logits /= temperature
    exp_logits = torch.exp(logits)
    softmax_probs = exp_logits / torch.sum(exp_logits)
    return softmax_probs


def local2dis(boxes, N, temperature):
    bins = torch.linspace(0, 1, N + 1).to(boxes.device)
    x1_bins = get_bin_indices(boxes[:, 0], bins, N)
    y1_bins = get_bin_indices(boxes[:, 1], bins, N)
    x2_bins = get_bin_indices(boxes[:, 2], bins, N)
    y2_bins = get_bin_indices(boxes[:, 3], bins, N)

    hist_x1 = torch.zeros(N, dtype=torch.float32)
    hist_y1 = torch.zeros(N, dtype=torch.float32)
    hist_x2 = torch.zeros(N, dtype=torch.float32)
    hist_y2 = torch.zeros(N, dtype=torch.float32)

    for i in range(boxes.shape[0]):
        hist_x1[x1_bins[i]] += 1
        hist_y1[y1_bins[i]] += 1
        hist_x2[x2_bins[i]] += 1
        hist_y2[y2_bins[i]] += 1

    prob_x1 = hist_x1
    prob_y1 = hist_y1
    prob_x2 = hist_x2
    prob_y2 = hist_y2

    smoothed_prob_x1 = softmax_with_temperature(prob_x1, temperature)
    smoothed_prob_y1 = softmax_with_temperature(prob_y1, temperature)
    smoothed_prob_x2 = softmax_with_temperature(prob_x2, temperature)
    smoothed_prob_y2 = softmax_with_temperature(prob_y2, temperature)

    # probs = torch.stack([smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2], dim=0)
    return smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2
    return probs


def local2dis_joint(boxes, N, temperature):
    bins = torch.linspace(0, 1, N + 1).to(boxes.device)
    x1_bins = get_bin_indices(boxes[:, 0], bins, N)
    y1_bins = get_bin_indices(boxes[:, 1], bins, N)
    x2_bins = get_bin_indices(boxes[:, 2], bins, N)
    y2_bins = get_bin_indices(boxes[:, 3], bins, N)

    indices_x = x1_bins * N + x2_bins
    hist_x_joint = torch.bincount(indices_x, minlength=N * N).view(N, N).float().to(boxes.device)
    prob_x_joint = hist_x_joint / hist_x_joint.sum() if hist_x_joint.sum() != 0 else hist_x_joint

    indices_y = y1_bins * N + y2_bins
    hist_y_joint = torch.bincount(indices_y, minlength=N * N).view(N, N).float().to(boxes.device)
    prob_y_joint = hist_y_joint / hist_y_joint.sum() if hist_y_joint.sum() != 0 else hist_y_joint

    smoothed_joint_x = softmax_with_temperature(prob_x_joint.view(-1), temperature).view(N, N)
    smoothed_joint_y = softmax_with_temperature(prob_y_joint.view(-1), temperature).view(N, N)

    probs = torch.stack([smoothed_joint_x, smoothed_joint_y], dim=0)

    return smoothed_joint_x, smoothed_joint_y
    # return probs


if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    torch.manual_seed(0)

    # 定义样本数量 n 和 bins 数量 N
    n = 1000
    N = 10  # 将区间 [0, 1] 分成 N 份

    # 生成大小为 n x 4 的检测框 tensor (x1, y1, x2, y2)，值在 [0, 1] 之间
    boxes = torch.rand(n, 4)
    # 温度参数 T
    temperature = 1
    bins = torch.linspace(0, 1, N + 1)
    smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2 = local2dis(boxes, N, temperature)
    smoothed_joint_x, smoothed_joint_y = local2dis_joint(boxes, N, temperature)

    # 可视化结果
    plt.figure(figsize=(15, 8))

    # 绘制每个坐标的平滑后概率分布
    for i, (name, smoothed_prob) in enumerate(
            zip(['x1', 'y1', 'x2', 'y2'], [smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2])):
        plt.subplot(1, 4, i + 1)
        sns.barplot(x=bins[:-1].numpy(), y=smoothed_prob.numpy(), color='green')
        # plt.title(f'Smoothed Distribution of {name} (T={temperature})')
        # plt.xlabel('Bin')
        # plt.ylabel('Smoothed Probability')


    plt.figure(figsize=(15, 8))
    for i, (name, smoothed_prob) in enumerate(
            zip(['x', 'y'], [smoothed_joint_x, smoothed_joint_y])):
        plt.subplot(1, 2, i + 1)
        # plt.plot(smoothed_prob.numpy(), color='green')
        # plt.title(f'Smoothed Distribution of {name} (T={temperature})')
        # plt.xlabel('Bin')
        # plt.ylabel('Smoothed Probability')
        plt.imshow(smoothed_prob.numpy(),cmap="viridis")
    # 调整布局
    plt.tight_layout()
    plt.show()


