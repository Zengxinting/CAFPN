from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import typing as t
from torch.jit.annotations import Tuple, List, Dict
from einops import rearrange
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
def xavier_init(m, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(m.weight, gain=gain)
    else:
        nn.init.xavier_normal_(m.weight, gain=gain)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, bias)
class DCE(nn.Module):
    def __init__(self, in_channels):
        super(DCE, self).__init__()

        # ----------------------------------------------------- #
        # 第一个分支  w, h, C --> w, h, C/2 --> SSF --> 2w, 2h, C
        # ----------------------------------------------------- #
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.dp = DySample_UP(in_channels // 4)
        self.conv1x1_1 = nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=1)
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        # ----------------------------------------------------- #
        # 第二个分支  w, h, C --> w/2, h/2, 2C --> SSF --> 2w, 2h, C
        # ----------------------------------------------------- #
        self.channel_attention = nn.Sequential(
            # 降维，减少参数数量和计算复杂度
            nn.Linear(in_channels, int(in_channels / 4)),
            nn.ReLU(inplace=True),  # 非线性激活
            # 升维，恢复到原始通道数
            nn.Linear(int(in_channels / 4), in_channels)
        )
        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.dp2 = DySample_UP(in_channels // 8)
        # ----------------------------------------------------- #
        # 第三个分支  w, h, C --> 1, 1, C --> broadcast
        # ----------------------------------------------------- #
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_3 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        out_size = x.shape[-2:]
        out_size = [x * 2 for x in out_size]

        m=self.conv3x3(x)
        m=self.dp(m)
        branch1=self.conv1x1_1(m)



        b, c, h, w = x.shape  # 输入张量的维度信息
        # 调整张量形状以适配通道注意力处理
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        # 应用通道注意力，并恢复原始张量形状
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        # 应用通道注意力图进行特征加权
        branch2 = x * x_channel_att
        branch2 = self.conv1x1_2(branch2)
        branch2 = self.dp2(branch2)
        branch2 = F.interpolate(branch2, size=out_size, mode="nearest")

        qq = self.globalpool(x)
        branch3 = self.conv1x1_3(qq)

        out = (branch1 + branch2 + branch3)
        return out
class CSAG(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int = 1,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(CSAG, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(self.dim, self.dim, 1)
        self.fc2 = nn.Conv2d(self.dim, self.dim, 1)
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Channel attention based on self attention
        # reduce calculations
        fc1 = self.fc1(self.avgpool(x))
        fc2 = self.fc2(self.maxpool(x))
        y = fc1 + fc2
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class DySample_UP(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample_UP, self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        x1 = x.reshape(B * self.groups, -1, H, W)
        ss = F.grid_sample(x1, coords, mode='bilinear',
                             align_corners=False, padding_mode="border")
        return ss.view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class CAFPN(nn.Module):
    def __init__(self, in_channels=512):
        super(CAFPN, self).__init__()
        self.dy_up5 = DySample_UP(in_channels=in_channels // 2)
        self.dy_up4 = DySample_UP(in_channels=in_channels // 4)
        self.conv3x3_5 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv3x3_4 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1)

        # ------------------------------- #
        # 定义SCE模块
        # ------------------------------- #
        self.DCE = DCE(in_channels=in_channels)

        # ------------------------------- #
        # 定义CAG模块
        # ------------------------------- #
        self.CAG = CSAG(dim=in_channels // 8)

        # ------------------------------- #
        # 定义1x1卷积
        # ------------------------------- #

        # 经过SSF后的1x1卷积
        self.SSF_C5 = nn.Conv2d(in_channels // 2, in_channels // 8, 1)
        self.SSF_C4 = nn.Conv2d(in_channels // 4, in_channels // 8, 1)

        # ------------------------------- #
        # 定义Ci --> Fi 的1x1卷积
        # ------------------------------- #

        self.conv_1x1_4 = nn.Conv2d(in_channels // 2, in_channels // 8, 1)
        self.conv_1x1_3 = nn.Conv2d(in_channels // 4, in_channels //8, 1)
        self.conv_1x1_2 = nn.Conv2d(in_channels // 8, in_channels // 8, 1)

        self.mp = nn.MaxPool2d(kernel_size=1, stride=2)
        
        self.conv_1x1 = nn.Conv2d(in_channels // 4+in_channels // 8, in_channels //8, 1)

        self.r2= Conv(in_channels // 8, in_channels // 8, 1)
        self.r3 = Conv(in_channels // 8, in_channels // 4, 1)
        self.r4 = Conv(in_channels // 8, in_channels // 2, 1)
        self.r5 = Conv(in_channels // 8, in_channels, 1)

        # self.convc5 = Conv(in_channels//2, in_channels, 3,2)



    def forward(self, x):
        # ------------------------------- #
        # get Ci
        # ------------------------------- #
        C2, C3, C4, C5 = x
        # C5 = self.convc5(C4)

        # ------------------------------- #
        # DCE
        # ------------------------------- #
        SCE_out = self.DCE(C5)

        # ------------------------------- #
        # get Fi
        # ------------------------------- #
        CC5 = self.dy_up5(self.conv3x3_5(C5))
        F4 = self.SSF_C5(CC5) + self.conv_1x1_4(C4)
        F3 = self.SSF_C4(self.dy_up4(self.conv3x3_4(C4))) + self.conv_1x1_3(C3)
        F2 = self.conv_1x1_2(C2)
        #
        # # ------------------------------- #
        # # DUF
        # # ------------------------------- #
        P4 = self.conv_1x1_3(torch.cat((F4,F.adaptive_max_pool2d(F3, output_size=F4.shape[-2:])),dim=1))
        P3_mid = self.conv_1x1_3(torch.cat((F.interpolate(P4, size=F3.shape[-2:], mode='nearest'), F3), dim=1))

        P4 = self.conv_1x1(torch.cat((P4, F.adaptive_max_pool2d(P3_mid, output_size=F4.shape[-2:]), F.adaptive_max_pool2d(F2,
                                                                                                   output_size=F4.shape[
                                                                                                               -2:])), dim=1))
        P3 = self.conv_1x1(torch.cat((F.interpolate(P4, size=F3.shape[-2:], mode='nearest') , P3_mid ,F.adaptive_max_pool2d(F2,
                                                                                                    output_size=F3.shape[
                                                                                                                -2:])),dim=1))
        P2 = self.conv_1x1(torch.cat((F.interpolate(P4, size=F2.shape[-2:], mode='nearest') ,F.interpolate(P3_mid, size=F2.shape[-2:],
                                                                                   mode='nearest') , F2),dim=1))

        # ------------------------------- #
        # get feature map I
        # ------------------------------- #
        out_size = P4.shape[-2:]
        SCE_out = F.interpolate(SCE_out, size=out_size, mode="nearest")
        I_P4 = F.interpolate(P4, size=out_size, mode="nearest")
        I_P3 = F.adaptive_max_pool2d(P3, output_size=out_size)
        I_P2 = F.adaptive_max_pool2d(P2, output_size=out_size)

        I = (I_P4 + I_P3 + I_P2 + SCE_out) / 4

        # ------------------------------- #
        # get Ri and CA fusion
        # ------------------------------- #
        outs = []
        CA = self.CSAG(I)
        R5 = F.adaptive_max_pool2d(I, output_size=C5.shape[-2:])
        R5 = (R5 + F.adaptive_max_pool2d(I, output_size=C5.shape[-2:])) * CA
        residual_R4 = F.adaptive_max_pool2d(I, output_size=C4.shape[-2:])
        R4 = (residual_R4 + F.adaptive_max_pool2d(I, output_size=C4.shape[-2:])) * CA
        residual_R3 = F.interpolate(I, size=C3.shape[-2:], mode="nearest")
        R3 = (residual_R3 + F.interpolate(I, size=C3.shape[-2:], mode="nearest")) * CA
        residual_R2 = F.interpolate(I, size=C2.shape[-2:], mode="nearest")
        R2 = (residual_R2 + F.interpolate(I, size=C2.shape[-2:], mode="nearest")) * CA

        R2 = self.r2(R2)
        R3 = self.r3(R3)
        R4 = self.r4(R4)
        R5 = self.r5(R5)



        for i in [R2,R3,R4,R5]:
        # for i in [ C2, C3, C4, C5 ]:
            outs.append(i)

        r6 = self.mp(R5)
        outs.append(r6)

        return outs


if __name__ == "__main__":
    # 初始化模块，使用单一整数作为输入通道数
    ela = CAFPN(in_channels=1024)  # 假设我们使用最深层的特征图作为输入

    # 创建一个有序字典作为输入
    input1 = torch.rand(3, 128, 80,80)
    input2 = torch.rand(3, 256, 40,40)
    input3 = torch.rand(3, 512, 20,20)
    # input4 = torch.rand(3, 512, 4, 4)

    # 前向传播
    output =  ela([input1, input2, input3])
    # 打印出输出张量的形状
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)