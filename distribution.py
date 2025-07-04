import numpy as np

def softmax_with_temperature(logits, temperature):
    """
    带温度参数的 softmax 函数。

    参数:
    logits (numpy.ndarray): 输入 logits（可以是 one-hot 编码或其他形式）。
    temperature (float): 温度参数，控制 softmax 的平滑程度。

    返回:
    numpy.ndarray: 经过 softmax 处理后的概率分布。
    """
    # 调整 logits 的温度
    logits = logits / temperature
    # 计算指数
    exp_logits = np.exp(logits - np.max(logits))  # 减去最大值以提高数值稳定性
    # 计算 softmax
    softmax_output = exp_logits / np.sum(exp_logits)
    return softmax_output

# 示例输入：one-hot 编码
one_hot_input = np.array([0, 0, 1, 0])  # 假设类别为 2（索引从 0 开始）
logits = one_hot_input.astype(float)  # 转换为浮点数

# 加入随机噪声（例如高斯噪声）
noise_scale = 0.1  # 噪声的尺度
noise = np.random.normal(0, noise_scale, size=logits.shape)  # 生成高斯噪声
logits_with_noise = logits + noise  # 加入噪声

# 温度参数
temperature = 5  # 温度越小，输出分布越尖锐；温度越大，输出分布越平滑

# 计算带温度的 softmax
softmax_output = softmax_with_temperature(logits_with_noise, temperature)

print("输入 (one-hot 编码):", one_hot_input)
print("加入噪声后的 logits:", logits_with_noise)
print("带温度的 softmax 输出:", softmax_output)