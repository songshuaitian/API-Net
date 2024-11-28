from einops import rearrange
import torch
from torch.nn import nn
class refine_att(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv=nn.Conv2d(
                cur_head_split * Ch*2,
                cur_head_split,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split,
            )



            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch*2 for x in self.head_splits]

    def forward(self, q,k, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        k_img = k
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        q_img = rearrange(q_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        k_img = rearrange(k_img, "B h Ch (H W) -> B h Ch H W", H=H, W=W)
        qk_concat=torch.cat((q_img,k_img),2)
        qk_concat= rearrange(qk_concat, "B h Ch H W -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        qk_concat_list = torch.split(qk_concat, self.channel_splits, dim=1)
        qk_att_list = [
            conv(x) for conv, x in zip(self.conv_list, qk_concat_list)
        ]

        qk_att = torch.cat(qk_att_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        qk_att = rearrange(qk_att, "B (h Ch) H W -> B h (H W) Ch", h=h)
        # print("qk_att:",qk_att.shape)
        return qk_att
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,shared_refine_att=None,qk_norm=1):
        super(Attention, self).__init__()
        self.norm=qk_norm
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.Leakyrelu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if num_heads == 8:
            crpe_window = {
                3: 2,
                5: 3,
                7: 3
            }
        elif num_heads == 1:
            crpe_window = {
                3: 1,
            }
        elif num_heads == 2:
            crpe_window = {
                3: 2,
            }
        elif num_heads == 4:
            crpe_window = {
                3: 2,
                5: 2,
            }
        self.refine_att = refine_att(Ch=dim // num_heads,
                                     h=num_heads,
                                     window=crpe_window)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        q_norm=torch.norm(q,p=2,dim=-1,keepdim=True)/self.norm+1e-6#dim=-1沿着最后一个维度，p=2表示使用L2范数
        q=torch.div(q,q_norm)
        k_norm=torch.norm(k,p=2,dim=-2,keepdim=True)/self.norm+1e-6
        k=torch.div(k,k_norm)
        #k = torch.nn.functional.normalize(k, dim=-2)

        refine_weight = self.refine_att(q,k, v, size=(h, w))
        #refine_weight=self.Leakyrelu(refine_weight)
        refine_weight = self.sigmoid(refine_weight)
        attn = k@v
        #attn = attn.softmax(dim=-1)

        #print(torch.sum(k, dim=-1).unsqueeze(3).shape)
        out_numerator = torch.sum(v, dim=-2).unsqueeze(2)+(q@attn)
        out_denominator = torch.full((h*w,c//self.num_heads),h*w).to(q.device)\
                          +q@torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads)+1e-6

        #out=torch.div(out_numerator,out_denominator)*self.temperature*refine_weight
        out = torch.div(out_numerator, out_denominator) * self.temperature
        out = out* refine_weight
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        # print("----------------------------------------------------------------")
        # print("out:",out.shape)
        out = self.project_out(out)
        # print("-----------------------------")
        return out + x  #原来是out