import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import torchvision.models as models
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class TIE(nn.Module):#Text_Information_Enhance
    def __init__(
        self,
        in_channels,
        out_channels,
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None
    ):
        super(TIE, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3_r = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2_r = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv1_r = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.offload_to_cpu = offload_to_cpu
        if self.offload_to_cpu:
            self.to('cpu')
        
    def forward(self, tensor):
        x = tensor
        x = self.conv_first(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = self.conv3_r(x3)
        x2 = self.conv2_r(x3 + x2)
        # x2 = self.conv2_r(x2)
        x1 = self.conv1_r(x2 + x1)
        x = self.conv_last(x1 + x)
        norm = nn.LayerNorm(x.size()[1:]).to(x.device)
        x = norm(x) + tensor
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(residual.shape)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            # print(residual.shape)
        # print(out.shape)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, strides=[2,1,2,1,1]):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[3])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[4])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

def resnet45():
    return ResNet(BasicBlock, [3, 4, 6, 6, 3])# strides: [(2, 1),1,(2, 1),1,1]


class STM(nn.Module):#Shape Transform Module 
    def __init__(self, in_channels, out_channels):
        super(STM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class STN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim
        ):
        super(STN, self).__init__()
        
        # bonenet
        self.bonenet = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim*2, 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(-1, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 3 * 2)
        # )
        
        # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    # Spatial transformer network forward function
    def stn(self, x):
        x1 = self.bonenet(x)
        batch_size = x1.size(0)
        in_features = x1.size(1) * x1.size(2) * x1.size(3)
        x1 = x1.view(batch_size, -1)
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features, 8),#64
            nn.ReLU(True),
            nn.Linear(8, 3 * 2)
        )
        self.fc_loc.to(next(self.parameters()).device)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        theta = self.fc_loc(x1)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
    
    def forward(self, x):
        # Perform the spatial transformation
        x = self.stn(x)
        return x

class PRE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PRE, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim, 
        num_heads,
        anchor_use = "light_kind",
        anchor_size = 2,
        mlp_ratio = 1.0,
        res_scale = 1.0
        ):

        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."
        self.anchor_use = anchor_use
        assert self.anchor_use == anchor_use, "Use anchor to limit the attention calculation!" 
        self.anchor_size = anchor_size
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.anchor = nn.Linear(embed_dim, embed_dim//(anchor_size**2))

        self.out = nn.Linear(embed_dim, embed_dim)
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation for query, key, and value
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, N, L, D)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, N, L, D)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, N, L, D)

        if self.anchor_use == "light_kind":
            A = self.anchor(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)# (B, N, L//A, D)
            k_a = torch.matmul(K, A.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, N, L, L//A)
            q_a = torch.matmul(A, Q.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, N, L//A, L)
            z = torch.matmul(V.transpose(-2, -1), k_a) / math.sqrt(self.head_dim) # (B, N, D, L//A)
            output = torch.matmul(z, q_a).transpose(-2, -1) #(B, N, L, D)
        elif self.anchor_use == "strong_kind":
            A = self.anchor(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)# (B, N, L//A, D)
            k_a = torch.matmul(K, A.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, N, L, L//A)
            q_a = torch.matmul(A, Q.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, N, L//A, L)
            attn = torch.matmul(k_a, q_a) #(B, N, L, L)
            attn_weights = F.softmax(attn, dim=-1)  # (B, N, L, L)
            output = torch.matmul(attn_weights, V)  # (B, N, L, D)
        else:
            attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, N, L, L)
            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn, dim=-1)  # (B, N, L, L)
            # Apply attention weights to values
            output = torch.matmul(attn_weights, V)  # (B, N, L, D)

        # Concatenate and linearly transform output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (B, N*L, D)
        output = self.out(output) 

        return output

class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        embed_dim,
        num_heads,
        anchor_use = False,
        anchor_size = 2,
        ):
        super(CrossAttentionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PRE(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, anchor_use, anchor_size)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.embedding(x)
        # x = self.pos_encoding(x)
        cross_attn_output = self.attn(query = self.pos_encoding(x), key = self.pos_encoding(x), value = x)      
        x = self.fc_out(cross_attn_output)
        return x

class Transform_Text_Block(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        in_features,
        embed_dim,
        num_heads,
        anchor_use,
        anchor_size,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        res_scale=1.0,
        norm_layer=nn.LayerNorm,
        conv_type="1conv",
        act_layer=nn.GELU,
        ):
        super(Transform_Text_Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.res_scale = res_scale

        self.attn = CrossAttentionTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            anchor_use=anchor_use,
            anchor_size=anchor_size
        )
        self.norm1 = norm_layer(input_dim)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.norm2 = norm_layer(output_dim)

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        # print(blc_to_bchw((self.res_scale * self.drop_path(self.norm1(self.attn(x)))), (x.shape[2], x.shape[3])).shape)
        x = bchw_to_blc(x) + (self.res_scale * self.drop_path(self.norm1(self.attn(x))))#res_norm + x
        x = self.res_scale * self.drop_path(self.norm2(self.mlp(x)))#MLP
        x = blc_to_bchw(x, (H, W))
        return x


from typing import Tuple
def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)

def build_last_conv(conv_type, dim):
    if conv_type == "1conv":
        block = nn.Conv2d(dim, dim, 3, 1, 1)
    elif conv_type == "3conv":
    # to save parameters and memory
        block = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    elif conv_type == "1conv1x1":
        block = nn.Conv2d(dim, dim, 1, 1, 0)#-1+1
    elif conv_type == "linear":
        block = Linear(dim, dim)
    return block

class Transform_Text_Layer(nn.Module):
    def __init__(
        self,
        depth,
        input_dim,
        output_dim,
        in_features,
        embed_dim,
        num_heads,
        anchor_use,
        anchor_size,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        init_method="n",
        res_scale=1.0,
        conv_type="1conv",
        fairscale_checkpoint=False, 
        offload_to_cpu=False,
        ):
        super(Transform_Text_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = in_features
        self.init_method = init_method
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.attn_drop = attn_drop

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Transform_Text_Block(
                input_dim=input_dim,
                output_dim=output_dim,
                in_features=in_features,
                embed_dim=embed_dim,
                num_heads=num_heads,
                anchor_use=anchor_use,
                anchor_size=anchor_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                res_scale = 1.0 if init_method == "r" else 1.0,
                conv_type=conv_type
            )
            if fairscale_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=offload_to_cpu)
            self.blocks.append(block)
        self.conv_last = build_last_conv(conv_type, output_dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size):
        res = x
        for blk in self.blocks:
            res = blk(res)
        res = self.conv_last(res)############
        return (res + x)



if __name__ == "__main__":
    x = torch.randn((1, 3, 64, 64))
    model_TIE = TIE(
        in_channels = 3,
        out_channels = 3,    
    )
    model_STN = STN(
        in_channels = 3,
        out_channels = 3,
        embed_dim = 32
    )
    model_CrossAttention = CrossAttentionTransformer(
        input_dim = 3,
        output_dim = 3,
        embed_dim = 96,
        num_heads = 4,
        anchor_use = "light_kind",#2 ways to use anchor
        # anchor_use = "strong_kind",
        anchor_size = 2
    )
    x = model_TIE(x)
    print(x.shape)
    x = model_STN(x)
    print(x.shape)
    # x = model_CrossAttention(x)
    # print(x.shape)
    B, C, H, W = x.shape
    TTL = Transform_Text_Layer(
        depth = 6,
        input_dim = 3,
        output_dim = 3,
        in_features = B*C*H*W,
        embed_dim = 32,
        num_heads = 4,
        anchor_use = "light_kind",
        anchor_size = 2
    )
    x = TTL(x, (x.shape[2], x.shape[3]))
    print(x.shape)


