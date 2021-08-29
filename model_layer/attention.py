import torch
import torch.nn as nn
import torch.nn.functional as F



"""
class StandAloneSelfAttention
    With Q, K, V mapping layer (This is convolution operation)
class StandAloneAttention
    Only compue attention (not mapping layer)
"""
class StandAloneSelfAttention(nn.Module):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: tuple or list, 
            stride = 1, 
            padding = 1,
            heads = 4,
            dim_head = 4,
            bias = False):
        super(StandAloneSelfAttention, self).__init__()
        assert out_channels == heads * dim_head, "out_channels must be equal (heads * dim_head)"

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.heads        = heads
        self.dim_head     = dim_head
        self.bias         = bias

        self.rel_h = nn.Parameter(
            torch.randn(out_channels // 2, 1, 1, kernel_size[0], 1), requires_grad = True)
        self.rel_w = nn.Parameter(
            torch.randn(out_channels // 2, 1, 1, 1, kernel_size[1]), requires_grad = True)

        self.q_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias)
        self.k_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias)
        self.v_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias)

    
    def forward(self, inputs):
        """
        Args:
            inputs: [B, C, H, W]
        """
        B, _, H, W = inputs.shape # 2, 3, 32, 32
        padded_x   = F.pad(inputs, [self.padding, self.padding, self.padding, self.padding]) # 2, 3, 34, 34

        q_out = self.q_conv(inputs)
        k_out = self.k_conv(padded_x)
        v_out = self.v_conv(padded_x)
        # print("q_out shape :", q_out.shape) # torch.Size([2, 16, 32, 32])
        # print("k_out shape :", k_out.shape) # torch.Size([2, 16, 34, 34])
        # print("v_out shape :", v_out.shape) # torch.Size([2, 16, 34, 34])

        # (3, 3) 커널, 스트라이드 1로 unfold 하는 코드 (컨볼루션 필터 형태로 바꾸어 주는 것)
        k_out = k_out.unfold(2, self.kernel_size[0], self.stride).unfold(3, self.kernel_size[1], self.stride)
        v_out = v_out.unfold(2, self.kernel_size[0], self.stride).unfold(3, self.kernel_size[1], self.stride)
        # print("unfold k_out shape: ", k_out.shape) # torch.Size([2, 16, 32, 32, 3, 3])
        # print("unfold v_out shape: ", v_out.shape) # torch.Size([2, 16, 32, 32, 3, 3])

        # dim = 1에서 self.out_channels = 16이므로 각각 반씩 나눔
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim = 1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim = 1)

        # torch.Size([2, 16, 32, 32]) -> torch.Size([2, 1, 16, 32, 32, 1])
        q_out = q_out.view(B, self.heads, self.dim_head, H, W, 1)

        # torch.Size([2, 16, 32, 32, 3, 3]) -> # torch.Size([2, 1, 16, 32, 32, 3, 9])
        k_out = k_out.contiguous().view(B, self.heads, self.dim_head, H, W, -1)
        v_out = v_out.contiguous().view(B, self.heads, self.dim_head, H, W, -1)
        # print(q_out.shape, k_out.shape, v_out.shape)

        out = q_out * k_out
        out = F.softmax(out, dim = -1)
        out = torch.einsum('a b c d e f, a b c d e f  -> a c b d e ', out, v_out)
        out = out.view(B, -1, H, W)
        return out



class StandAloneAttention(nn.Module):
    def __init__(self, 
            in_channels: int, kernel_size: tuple or list, stride = 1, padding = 1, heads = 8, dim_head = 8):
        super(StandAloneAttention, self).__init__()
        """
        It is not use embbeding convolution or other trianable operator.
        Only compute attentio mechanism.

        kernel_size = (3, 3) ~ padding = 1
        kernel_size = (5, 5) ~ padding = 2
        kernel_size = (7, 7) ~ padding = 3
        """
        assert in_channels % heads == 0, "in_channels should be divided by heads."
        assert kernel_size[0] % padding == 1, "kernel_size = 2*padding + 1"

        self.in_channels  = in_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.heads        = heads
        self.dim_head     = dim_head

        self.rel_h = nn.Parameter(
            torch.randn(in_channels // 2, 1, 1, kernel_size[0], 1), requires_grad = True)
        self.rel_w = nn.Parameter(
            torch.randn(in_channels // 2, 1, 1, 1, kernel_size[1]), requires_grad = True)


    def forward(self, features1, features2):
        """
        Args
            features1: [B, C, H, W]
            features2: [B, C, H, W]
        
        returns
            out: [B, C, H, W]
        """
        B, C, H, W      = features1.shape
        padded_features = F.pad(
            features2, [self.padding, self.padding, self.padding, self.padding]) # 2, 3, 34, 34

        q = features1
        k = padded_features
        v = padded_features

        k = k.unfold(2, self.kernel_size[0], self.stride).unfold(3, self.kernel_size[1], self.stride)
        v = v.unfold(2, self.kernel_size[0], self.stride).unfold(3, self.kernel_size[1], self.stride)

        kh, kw = k.split(self.in_channels // 2, dim = 1)
        k      = torch.cat((kh + self.rel_h, kw + self.rel_w), dim = 1)

        q = q.view(B, self.heads, self.dim_head, H, W, 1) # 쿼리의 각 element마다 차원을 주어서 expand_dims
        k = k.contiguous().view(B, self.heads, self.dim_head, H, W, -1) # 로컬 영역마다의 픽셀을 옆으로 flatten
        v = v.contiguous().view(B, self.heads, self.dim_head, H, W, -1) # 로컬 영역마다의 픽셀을 옆으로 flatten

        out = q * k
        out = F.softmax(out, dim = -1)
        out = torch.einsum('a b c d e f, a b c d e f  -> a c b d e ', out, v)
        out = out.view(B, -1, H, W)
        return out



if __name__ == "__main__":
    temp = torch.randn((4, 8, 192, 640))
    attention = StandAloneAttention(
        in_channels = 8, kernel_size = (3, 3), stride = 1, padding = 1, heads = 4, dim_head = 2)
    attention(temp, temp)