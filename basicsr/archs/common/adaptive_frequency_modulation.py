import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

class AFM(nn.Module):#adaptive frequency modulation for highlighting the high frequence by the adaptive guassian-laplace operator
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        # afm_window_size=4,#make sure h and w can be divided
        num_filter=8,
        filter_size="(3,3)",
        use_gaussian=True,
        gaussian_kernel_size="(3,3)",
        gaussian_sigma=0.5,# 0.5-2
        range=4,
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None
    ):
        super(AFM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filter = num_filter

        self.use_gaussian = use_gaussian
        self.gaussian_kernel_size = ast.literal_eval(gaussian_kernel_size)
        self.gaussian_sigma = gaussian_sigma

        self.conv_first = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=3, stride=1, padding=1)
        self.filter_size = ast.literal_eval(filter_size)
        self.conv = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=num_filter)
        self.fc = nn.Linear(num_filter, num_filter)
        self.range = range
        self.conv_last = nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def apply_afm(self, x):
        B, C, H, W = x.shape
        # auto calculate gaussian kernel size based on input size and sigma
        if self.use_gaussian:
            gaussian_kernel_size_h = int(torch.ceil(2 * torch.tensor(self.gaussian_kernel_size[0]).float() * self.gaussian_sigma))
            if gaussian_kernel_size_h % 2 == 0:  
                gaussian_kernel_size_h += 1
            gaussian_kernel_size_w = int(torch.ceil(2 * torch.tensor(self.gaussian_kernel_size[1]).float() * self.gaussian_sigma))
            if gaussian_kernel_size_w % 2 == 0:  
                gaussian_kernel_size_w += 1

            # Generate Gaussian kernel ij
            gaussian_kernel = torch.tensor([
                [torch.exp(-(i ** 2 + j ** 2) / (2 * torch.tensor(self.gaussian_sigma) ** 2)) for j in range(-gaussian_kernel_size_w // 2, gaussian_kernel_size_w // 2 )]
                for i in range(-gaussian_kernel_size_h // 2, gaussian_kernel_size_h // 2)
            ]).to(x.device)

            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum().to(x.device)
            # Apply Gaussian kernel to input
            x = F.conv2d(x, gaussian_kernel.view(1, 1, gaussian_kernel_size_h, gaussian_kernel_size_w),
                        padding=((gaussian_kernel_size_h-1)//2, (gaussian_kernel_size_w-1)//2))

        # caculate filter
        kernel_size_h = H - self.filter_size[0] + 1 
        kernel_size_w = W - self.filter_size[1] + 1 
        avg_pool_1 = nn.AvgPool2d(kernel_size=(kernel_size_h, kernel_size_w), stride=1)#pool_size == filter_size

        x1 = self.conv_first(x)
        l_h = avg_pool_1(x1)
        l_h = self.conv(l_h) #b c hf wf
        l_h = self.bn(l_h)  # laplace
        l_h = self.relu(l_h) #tanh can be wrong?
        l_h = self.range * l_h
        # l_h = self.range * torch.tanh(l_h)

        x_h = F.conv2d(x1, l_h.repeat(self.num_filter, 1, 1, 1), stride=1, padding=((self.filter_size[0]-1)//2, (self.filter_size[1]-1)//2))

        #caculate weights of the different high frequency layers 
        w_h = F.avg_pool2d(x_h, kernel_size=(H, W)).view(B, self.num_filter)
        w_h = self.fc(w_h)
        # w_h = self.fc(w_h).view(B, self.out_channels, 1, 1)
        w_h = w_h.view(B, self.num_filter, 1, 1)
        ln = nn.LayerNorm((self.num_filter, 1, 1)).to(x.device)
        w_h = ln(w_h.to(x.device))
        w_h = F.softmax(w_h, dim=1)
        x_ = self.conv_last(x_h * w_h)
        return x_

    def forward(self, x):
        _, C, _, _  = x.shape
        x_afm = []
        for i in range(C):
            x_afm.append(self.apply_afm(x[:, i, :, :].unsqueeze(1))) 
        x = torch.cat(x_afm, dim=1)
        
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True)

        x = (x - mean) / (std + 1e-6) 
        
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 2000, 2000)
    model_AFM = AFM(
        in_channels = 3,
        out_channels = 3,
        num_filter=9,
        filter_size="(7,7)",
        use_gaussian=True,
        gaussian_kernel_size="(3,3)",
        gaussian_sigma=2,# 0.5-2
        range=4,
    )
    x = model_AFM(x)
    print(x.shape)


