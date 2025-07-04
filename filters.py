import torch
import torch.nn as nn
import torch.nn.functional as F


class StateNet(nn.Module):
    def __init__(self):
        super(StateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        # x1 = torch.mean(x, dim=(2, 3))
        # x2 = torch.std(x, dim=(2, 3))
        # x = torch.cat([x1, x2], dim=1)
        return x


class WB(nn.Module):
    def __init__(self):
        super(WB, self).__init__()

    def forward(self, x, params):
        device = x.device
        params = torch.unsqueeze(torch.unsqueeze(params, -1), -1).to(device)
        params = params.expand_as(x)
        x = x * params
        x = torch.clamp(x, 0, 1)
        # print("wb:",torch.max(x),torch.min(x))
        return x


class GAMMA(nn.Module):
    def __init__(self):
        super(GAMMA, self).__init__()

    def forward(self, x, param):
        device = x.device  # Get the device of the input tensor x
        param = param.to(device)  # Move param to the same device as x
        param = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(param, -1), -1), -1)
        x = x ** param + 1e-3
        x = torch.clamp(x, 0, 1)
        # print("gamma:",torch.max(x),torch.min(x))

        return x


class Contrast(nn.Module):
    def __init__(self):
        super(Contrast, self).__init__()

    def forward(self, x, alpha):
        device = x.device
        alpha = alpha.to(device, dtype=x.dtype).view(-1, 1, 1, 1)
        lum = torch.sum(
            x * torch.tensor([0.06, 0.67, 0.27], device=device, dtype=x.dtype).view(1, -1, 1, 1).repeat(x.shape[0], 1,
                                                                                                        x.shape[2],
                                                                                                        x.shape[3]),
            dim=1, keepdim=True)
        en_lum = 0.5 * (1 - torch.cos(torch.pi * lum))
        en_pi = x * (en_lum / (lum + 1e-3))
        p_o = alpha * en_pi + (1 - alpha) * x
        p_o = torch.clamp(p_o, 0.001, 1)

        return p_o


class SHAPEN(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(SHAPEN, self).__init__()

        # Create a 1D Gaussian kernel
        kernel_1d = self.create_gaussian_kernel(kernel_size, sigma)

        # Convert the 1D kernel to a 2D kernel by outer product
        kernel_2d = torch.ger(kernel_1d, kernel_1d)

        # Expand the 2D kernel to be compatible with convolution operation
        # Adjust dimensions for xltiple input channels
        kernel_2d = kernel_2d.unsqueeze(0).repeat(3, 1, 1)  # Assuming 3 input channels

        # Register the kernel as a buffer so it is not trained
        self.register_buffer('weight', kernel_2d.unsqueeze(1))  # Add channel dimension

    def create_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
        phi_x = torch.exp(-0.5 * x.pow(2) / (sigma ** 2))
        phi_x /= phi_x.sum() + 1e-3
        return phi_x

    def forward(self, x, lamda):
        device = x.device
        lamda = lamda.to(device)
        self.weight = self.weight.to(device)
        lamda = lamda.view(-1, 1, 1, 1)
        # Determine the padding based on the kernel size
        padding = self.weight.shape[2] // 2

        # Apply the 2D convolution using the Gaussian kernel
        # Use 'groups' to ensure each channel is filtered independently
        x1 = torch.clamp(F.conv2d(x, self.weight, padding=padding, groups=x.shape[1]), 0, 1)
        x = x + lamda * (x - x1)
        x = torch.clamp(x, 0, 1)
        # print("shapen:",torch.max(x),torch.min(x))
        return x


class Tone(torch.nn.Module):
    def __init__(self, L=8):
        super(Tone, self).__init__()
        self.L = L

    def forward(self, P_i, t_k):
        device = P_i.device
        t_k - t_k.to(device)
        T_L = torch.sum(t_k, dim=1, keepdim=True).to(device)
        P_o = torch.zeros_like(P_i)

        for j in range(len(t_k[0])):
            clip_range = (self.L * P_i - j).to(device)
            clipped_P_i = F.relu(clip_range, True)
            P_o += clipped_P_i / (T_L.view(-1, 1, 1, 1) * t_k[:, j].view(-1, 1, 1, 1) + 1e-3)

        return P_o


class Tone1(torch.nn.Module):
    def __init__(self, L=8):
        super(Tone1, self).__init__()

    def forward(self, x, t):
        device = x.device
        t = t.to(device)
        t = t.view(-1, 1, 1, 1)
        t = t.expand_as(x)
        x = x + t * x * (1 - x)

        return x


class Dehazing(nn.Module):
    def __init__(self):
        super(Dehazing, self).__init__()

    def dark_channel(self, img, kernel_size=15):
        min_channel = torch.min(img, dim=1, keepdim=True)[0]
        dark_channel = F.avg_pool2d(min_channel, kernel_size, stride=1, padding=kernel_size // 2)
        return dark_channel

    def atmospheric_light(self, img, dark_channel):
        N, C, H, W = img.shape
        num_pixels = H * W
        top_k = int(num_pixels * 0.05)

        flat_dark_channel = dark_channel.view(N, -1)

        _, top_indices = flat_dark_channel.topk(top_k, dim=1)

        flat_img = img.view(N, C, -1)
        atmospheric_light = []

        for i in range(N):
            selected_pixels = flat_img[i, :, top_indices[i]]
            mean_rgb = selected_pixels.mean(dim=1)
            atmospheric_light.append(mean_rgb)
        atmospheric_light = torch.stack(atmospheric_light)

        return atmospheric_light

    def forward(self, img, w):
        device = img.device
        w = w.to(device)
        w = w.view(-1, 1, 1, 1)
        dark_channel = self.dark_channel(img)
        atmospheric_light = self.atmospheric_light(img, dark_channel)
        atmospheric_light = torch.unsqueeze(torch.unsqueeze(atmospheric_light, -1), -1)
        atmospheric_light = atmospheric_light.expand_as(img)
        # Estimate transmission map
        transmission = 1 - w * dark_channel
        transmission = torch.where(transmission > 0.1, transmission, torch.tensor(0.1).to(img.device, dtype=img.dtype))

        # Recover the scene radiance
        x = (img - atmospheric_light) / (transmission + 1e-3) + atmospheric_light
        x = torch.clamp(x, 0, 1)
        # print("dehaze:",torch.max(x),torch.min(x))

        return x


class CNN_PP(nn.Module):
    def __init__(self):
        super(CNN_PP, self).__init__()

        # Convolutional blocks
        self.conv1_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2_blocks = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv3_blocks = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv4_blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv5_blocks = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Fully-connected layers
        self.fc5_layers = nn.Sequential(
            nn.Linear(256, 7),
            nn.Sigmoid()
        )
        self.fc4_layers = nn.Sequential(
            nn.Linear(128, 7),
            nn.Sigmoid()
        )
        self.fc3_layers = nn.Sequential(
            nn.Linear(64, 7),
            nn.Sigmoid()
        )
        self.fc2_layers = nn.Sequential(
            nn.Linear(32, 7),
            nn.Sigmoid()
        )
        self.fc1_layers = nn.Sequential(
            nn.Linear(16, 7),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def tanh01(self, x):
        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x, left, right):
        return self.tanh01(x) * (right - left) + left

    def clip_range(self, x):
        x1 = torch.zeros_like(x)
        x1[:, 0] = self.tanh_range(x[:, 0], 0.8, 1.5)
        x1[:, 1] = self.tanh_range(x[:, 1], 0.1, 1.0)
        x1[:, 2] = self.tanh_range(x[:, 2], 0.1, 5.0)
        x1[:, 3] = self.tanh_range(x[:, 3], 0.8, 1.0)
        x1[:, 4:7] = self.tanh_range(x[:, 4:7], 0.90, 1.1)
        x1[:, 7:] = self.softmax(self.tanh_range(x[:, 7:], -10.0, 10.0))
        return x1

    def forward(self, x):
        x = self.conv1_blocks(x)
        x1 = torch.mean(x, dim=(2, 3))  # x1 = x1.view(x1.si,ze(0), -1)
        x1 = self.fc1_layers(x1)
        x1 = self.clip_range(x1)

        x = self.conv2_blocks(x)
        x2 = torch.mean(x, dim=(2, 3))
        x2 = self.fc2_layers(x2)
        x2 = self.clip_range(x2)

        x = self.conv3_blocks(x)
        x3 = torch.mean(x, dim=(2, 3))
        x3 = self.fc3_layers(x3)
        x3 = self.clip_range(x3)

        x = self.conv4_blocks(x)
        x4 = torch.mean(x, dim=(2, 3))
        x4 = self.fc4_layers(x4)
        x4 = self.clip_range(x4)

        x = self.conv5_blocks(x)
        x5 = torch.mean(x, dim=(2, 3))
        x5 = self.fc5_layers(x5)
        x5 = self.clip_range(x5)
        x_avg = (x3 + x4 + x5) / 3.0
        # [x1, x2, x3, x4, x5]
        return x_avg


class CNN_PPFC(nn.Module):
    def __init__(self):
        super(CNN_PPFC, self).__init__()

        # Fully-connected layers
        self.fc2_layers = nn.Sequential(
            nn.Linear(64, 7),
        )
        self.fc1_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.softmax = nn.Softmax()

    def tanh01(self, x):
        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x, left, right):
        return self.tanh01(x) * (right - left) + left

    def clip_range(self, x):
        x1 = torch.zeros_like(x)
        x1[:, 0] = self.tanh_range(x[:, 0], 0.7, 1.1)
        x1[:, 1] = self.tanh_range(x[:, 1], 0.1, 1.0)
        x1[:, 2] = self.tanh_range(x[:, 2], 0.1, 2.0)
        x1[:, 3] = self.tanh_range(x[:, 3], 0.1, 1)
        x1[:, 4:7] = self.tanh_range(x[:, 4:7], 0.90, 1.1)
        # x1[:, 7:] = self.softmax(self.tanh_range(x[:, 7:], -10.0, 10.0))
        return x1

    def forward(self, x):
        x = self.avg(x).view(x.shape[0], -1)
        x = self.fc1_layers(x)
        x = self.fc2_layers(x)
        x = self.clip_range(x)
        return x


class Power(nn.Module):
    def __init__(self, out_channels=3):
        super(Power, self).__init__()
        self.alphas = nn.Parameter(torch.ones(out_channels // 3))

    def forward(self, x):
        return x ** self.alphas


class ISP(nn.Module):
    def __init__(self):
        super(ISP, self).__init__()
        self.wb = WB().cuda()
        self.gamma = GAMMA().cuda()
        self.contrast = Contrast().cuda()
        self.shapen = SHAPEN().cuda()
        self.dehaze = Dehazing().cuda()
        # self.tone1 = Tone1().cuda()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def tanh01(self, x):
        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x, left, right):
        return self.tanh01(x) * (right - left) + left

    def clip_range(self, x):
        x1 = torch.zeros_like(x)
        x1[:, 0] = self.tanh_range(x[:, 0], 1.0, 1.25)
        x1[:, 1] = self.tanh_range(x[:, 1], 0.1, 1.0)
        x1[:, 2] = self.tanh_range(x[:, 2], 0.0, 2.0)
        x1[:, 3] = self.tanh_range(x[:, 3], 0.0, 1.0)
        x1[:, 4:7] = self.tanh_range(x[:, 4:7], 0.95, 1.05)
        # x1[:, 7:] = self.tanh_range(x[:, 7:], 0.0, 1.0)
        return x1

    # def forward(self, x, output1, output2):
    #     device = x.device  # Geot the device of input tensor x
    #     # for output1 in mparams:
    #     output1 = output1.to(device)
    #     output2 = output2.to(device)
    #     # print(output1[0])
    #     # Ensure all params and tensors are moved to the same device
    #     # param = (0.5 + output1[:, 0]).to(device)  # 0.5-1.5
    #     output1 = self.clip_range(output1)
    #     output2 = self.softmax(output2)
    #     # gamma_list = torch.tensor([0.5, 0.7, 1.0, 1.5, 2.0], device=x.device, dtype=x.dtype)
    #     # output1 = self.softmax(output1)
    #     # sorted_tensor, sorted_indices = torch.sort(output1)
    #     # index = sorted_indices[:, 0]
    #     # gamma_sel = []
    #     # for i in index:
    #     #     g = gamma_list[i]
    #     #     gamma_sel.append(g)
    #     # gamma = torch.tensor(gamma_sel)
    #     # print(gamma)
    #     # x = self.gamma(x, gamma)
    #
    #     param = output1[:, 0]
    #     # param =  torch.clamp(output1[:, 0],0.8,1.0).to(device)
    #     # pro=output[:, 1].view(-1,1,1,1).to(device,dtype=x.dtype)
    #     # mask = pro > 0.5
    #     # mask= mask.expand_as(x)
    #     alpha = output1[:, 1]  # 0-1
    #     # alpha =  torch.clamp(output1[:, 1],0.5,1.0).to(device)
    #     # lamda = (5 * output1[:, 2]).to(device)  # 0-5
    #     lamda = output1[:, 2]
    #     # lamda =  torch.clamp(output1[:, 2],0.0,5.0).to(device)
    #     # t = (-1+output[:, 3]*2).to(device)
    #     # w = (0.1 + 0.9 * output1[:, 3]).to(device)  # 0.5-1.0
    #     w = output1[:, 3]
    #     # w =  torch.clamp(output1[:, 3],0.5,1.0).to(device)
    #     # params = (0.9 + 0.2 * output1[:, 4:7]).to(device)  # 0.9-1.1
    #     # params = torch.clamp(output1[:, 4:7], 0.9, 1.0).to(device)
    #     params = output1[:, 4:7]
    #
    #     # print(param[0].item(), alpha[0].item(), lamda[0].item(), w[0].item(), params[0][0].item(), params[0][1].item(),
    #     #      params[0][2].item())
    #     # pros =output1[:, 7:].to(device, dtype=x.dtype)
    #     # output2 = output1[:, 7:]
    #     # print(output2[0])
    #     # print(pros)
    #     # idx = torch.argsort(pros, descending=True)
    #     # print(idx)
    #     # for ide in id_:
    #     #    if pros[ide]>0.5:
    #     # weights = output1[:, 4:7].view(x.shape[0],-1,1,1).to(device)
    #     # if len(weights)>5:
    #     #    print(weights[0],weights[5])
    #     # weights=weights.repeat(1, 1, x.shape[2], x.shape[3])
    #
    #     pros, idx = torch.max(output2, dim=1, keepdim=True)
    #     x1 = self.wb(x, params)
    #     x2 = self.gamma(x, param)
    #     x3 = self.contrast(x, alpha)
    #     x4 = self.shapen(x, lamda)
    #     x5 = self.dehaze(x, w)
    #     x6 = x
    #     #
    #     x_list = [x1, x2, x3, x4, x5, x6]
    #     x_all = torch.zeros_like(x)
    #     for i, id in enumerate(idx):
    #         # print(id, i)
    #         x_tmp = x_list[id][i]
    #         x_all[i] = x_tmp
    #     x = x_all
    #     #
    #     # max_x, max_id = torch.max(output2, dim=1)
    #     # print("------")
    #     # for _ in x_list:
    #     #     print(torch.isnan(_).any())
    #
    #     # print(self.p1.item(),self.p2.item(),self.p3.item(),self.p4.item(),self.p5.item(),self.p6.item())
    #     # x_sum = torch.zeros_like(x)
    #     # for _ in x_list:
    #     #    x_sum += _
    #     # x = (x_sum / (len(x_list)))
    #     # x_min = torch.min(x)
    #     # x_max = torch.max(x)
    #     # x = (x - x_min) / ((x_max - x_min) + 1e-3)
    #     # x_list=[self.dehaze(x, w),self.gamma(x, param),x]
    #     # tl = []
    #     # for i,id in enumerate(max_id):
    #     #     tl.append(x_list[id][i])
    #     # x = torch.stack(tl)
    #     # output2 = torch.where(output2 > 0.1, output2, -10.0)
    #     # output2 = F.softmax(output2, dim=1)
    #     # output2 = torch.where(output2 < 0.1, 0.0, output2).to(x.device, dtype=x.dtype)
    #     # sum_output2 = torch.sum(output2, dim=1)
    #     # # print("parameters：" + str(output1[0]))
    #     # # print("actions:" + str(output2[0]))
    #     # x_sum = torch.zeros_like(x)
    #     # for i, x_t in enumerate(x_list):
    #     #     x_sum += (output2[:, i].view(-1, 1, 1, 1) / (sum_output2.view(-1, 1, 1, 1) + 1e-3)).repeat(1, 3, x.shape[2],
    #     #                                                                                                x.shape[3]) * x_t
    #     #     # x_sum += x_t
    #     # x = x_sum
    #     # # print("x_min:{},x_max:{}".format(torch.min(x), torch.max(x)))
    #     # x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-3)
    #     # print(torch.max(x))
    #     # print(torch.min(x))
    #     # if torch.rand(1).item()<0.8:
    #     # print(torch.argsort(torch.mean(output2,0), descending=True))
    #     # idx = torch.argsort(torch.mean(output2,0), descending=True)[0].item()
    #     # idx = torch.argsort(torch.mean(output2,dim=0,keepdim=True),dim=1,descending=True)[0].item()
    #     # else:
    #     # idx = torch.randint(0, len(x_list), (1,)).item()
    #     # print(idx)
    #     # x=x_list[idx]
    #     # x = torch.clamp(self.wb(x, params),0,1)
    #     # x = torch.where(mask, self.gamma(x, param), x)
    #     # x = self.dehaze(x, w)
    #     # x = self.wb(x,params)
    #     # x = self.gamma(x, param)
    #     # x = self.contrast(x, alpha)
    #     # x = self.shapen(x, lamda)
    #     # x = self.tone1(x, t)
    #     # print(param[0].item(),alpha[0].item(),lamda[0].item(),w[0].item(),params[0])
    #     # output2 = output2.to(x.device, dtype=x.dtype)
    #     #
    #     # probs, orders = torch.sort(torch.mean(output2, dim=0), descending=True)
    #     # for prob, order in zip(probs, orders):
    #     #     if prob.item() > 0.1:
    #     #         if order.item() == 0:
    #     #             x = self.wb(x, params)
    #     #         elif order.item() == 1:
    #     #             x = self.gamma(x, param)
    #     #         elif order.item() == 2:
    #     #             x = self.contrast(x, alpha)
    #     #         elif order.item() == 3:
    #     #             x = self.shapen(x, lamda)
    #     #         elif order.item() == 4:
    #     #             x = self.dehaze(x, w)
    #     return x

    def forward(self, x, output1, output2):
        device = x.device  # Get the device of input tensor x
        output1 = output1.to(device)
        output2 = output2.to(device)

        # Ensure all parameters are within valid ranges
        output1 = self.clip_range(output1)
        output2 = self.tanh_range(output2, 0.0, 1.0)
        output2 = self.softmax(output2)
        # output2 = self.softmax(output2)
        # print(output1[0])

        param = output1[:, 0]
        alpha = output1[:, 1]
        lamda = output1[:, 2]
        w = output1[:, 3]
        params = output1[:, 4:7]

        # Apply image processing steps
        x1 = self.wb(x, params)
        x2 = self.gamma(x, param)
        x3 = self.contrast(x, alpha)
        x4 = self.shapen(x, lamda)
        x5 = self.dehaze(x, w)

        x_list = [x1, x2, x4, x5]

        # # Select the best enhancement based on output2
        # pros, idx = torch.max(output2, dim=1, keepdim=True)
        # x_all = torch.zeros_like(x)
        # for i, id in enumerate(idx):
        #     x_all[i] = x_list[id][i]
        # Compute weighted sum of the processed images
        # output2 = torch.where(output2 < 0.1, torch.tensor(0.0, device=x.device, dtype=x.dtype), output2)
        # print(output2)
        x_sum = torch.zeros_like(x)
        for i, x_t in enumerate(x_list):
            weight = output2[:, i].view(-1, 1, 1, 1)
            # x_sum += x_t
            x_sum += weight * x_t
        x_sum = x + 0.1 * x_sum
        # Normalize the weighted sum
        x_sum = x_sum / (torch.sum(output2, dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-3)
        # x_sum = x + x_sum
        min_vals, _ = torch.min(torch.min(x_sum, dim=3)[0], dim=2)
        max_vals, _ = torch.max(torch.max(x_sum, dim=3)[0], dim=2)

        min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)

        epsilon = 1e-7
        range_vals = max_vals - min_vals + epsilon

        x_sum = (x_sum - min_vals) / range_vals

        return x_sum


class MLP_PP(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(MLP_PP, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
        self.act = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b, n_states] --> [b, n_hiddens]
        x = self.fc2(x)
        x = 1 - 0.2 * self.act(x)
        print(x)
        return x


if __name__ == "__main__":
    # Test case
    model = CNN_PP()
    isp = ISP()
    input_image = torch.rand(16, 3, 640, 640)  # B, C, H, W
    input_image1 = F.interpolate(input_image, size=(256, 256), mode='bilinear', align_corners=False)

    output = model(input_image1)
    output1 = isp(input_image, output)
    print(output1.shape)
