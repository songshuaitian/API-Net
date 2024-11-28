import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange
# from option import opt, model_name, log_dir

class ResBlock(nn.Module):#输入维度64，输出维度64

    def __init__(self,dim):
        super(ResBlock,self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(dim,dim*2,kernel_size=3,padding=1),
            # nn.BatchNorm2d(dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2,dim,3,padding=1)
        )

    def forward(self,x):
        x1 = self.rb(x)
        return x1 + x


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

class FEM(nn.Module):######################################linear的错
    def __init__(self,dim,dropout=0.1,batch_size=1):
        super(FEM,self).__init__()
        self.dim = dim

        # self.Fem = nn.Sequential(
        #     nn.LayerNorm([dim, 256, 256]),
        #     nn.Flatten(),
        #     nn.Linear(dim, dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim * 2, dim),
        # )

        # self.norm = nn.LayerNorm([dim,256,256])#????????????
        # self.norm = nn.BatchNorm2d(dim)#++++++++++++++++++++++++++
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(dim,dim,1)
        self.conv2 = nn.Conv2d(dim,dim,1)
        self.relu = nn.ReLU(inplace=True)
        self.droput = nn.Dropout(dropout)

    def forward(self,x):
        # norm = nn.LayerNorm(self.dim,x.shape[2],x.shape[3])
        # x = self.norm(x)#+++++++++++++++++++++
        # x1 = self.flatten(x1)
        # print("flatten_size:",x1.shape,type(x1))
        x2 = self.conv1(x)
        x2 = self.relu(x2)
        x2 = self.droput(x2)
        x2 = self.conv2(x2)
        # y =self.Fem(x)
        # y = torch.Tensor(x3)
        # ouput = torch.reshape(y,[1,self.dim,256,256])# 后期需要调size
        return x + x2
#######################################
#MSA多头自注意力
class MSA(nn.Module):
    def __init__(self,dim,num_heads,bias):
        super(MSA,self).__init__()
        self.att = Attention(dim=dim,num_heads=num_heads,bias=bias)#------需要传参数
        self.fem = FEM(dim=dim)#-----需要传入参数

    def forward(self,x):
        x = self.att(x)
        x = self.fem(x)
        return x
######################################
#噪声预测模块
class NPM(nn.Module):
    def __init__(self,dim,num_heads,bias):
        super(NPM,self).__init__()
        self.conv1 = nn.Conv2d(dim,dim*2,3,padding=1,bias=True)#---------需要传参
        # self.batch_norm = nn.BatchNorm2d(dim*2)
        self.norm = nn.LayerNorm(dim*2)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim*2,dim,3,padding=1)
        self.msa1 = MSA(dim=dim,num_heads=num_heads,bias=bias)#---------需要传参
        self.msa2 = MSA(dim=dim,num_heads=num_heads,bias=bias)#---------需要传参

    def forward(self,x):
        x = self.conv1(x)
        # x1 = self.batch_norm(x1)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.act(x)
        # print("-------x2:",x2)
        x = self.conv2(x)
        x = self.msa1(x)
        y = self.msa2(x)
        return y
'''
    DownSanple
'''
# class PatchEmbed(nn.Module):
#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
#         super().__init__()
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         if kernel_size is None:
#             kernel_size = patch_size
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
#                               padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')
#
#     def forward(self, x):
#         # x = x.to(torch.float32)
#         x = self.proj(x)
#         return x
# '''
#     Upsample
# '''
# class PatchUnEmbed(nn.Module):
#     def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
#         super().__init__()
#         self.out_chans = out_chans
#         self.embed_dim = embed_dim
#
#         if kernel_size is None:
#             kernel_size = 1
#
#         self.proj = nn.Sequential(
#             nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
#                       padding=kernel_size // 2, padding_mode='reflect'),
#             nn.PixelShuffle(patch_size)
#         )
#
#     def forward(self, x):
#         x = self.proj(x)
#         return x
class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x

'''
    part1 
'''
class Denoisy(nn.Module):
    # def __init__(self,num_heads,bias,dim = 64):
    def __init__(self,num_heads,bias,dim,in_chans=3,embed_dim=[24,48,92,48,24]):
        super(Denoisy,self).__init__()
        # self.conv1 = nn.Conv2d(3,dim,3,padding=1,bias=True)
        # self.patch_embed = PatchEmbed(
		# 	patch_size=1, in_chans=in_chans, embed_dim=dim, kernel_size=3)
        # self.res = ResBlock(dim=dim)#输入，输出维度都是32
        self.msa1 = MSA(dim=dim,num_heads=num_heads,bias=bias)#传参32->32
        # self.npm1 = NPM(dim=dim,num_heads=num_heads,bias=bias)#传参64->128    32->32
        # self.npm2 = NPM(dim=dim,num_heads=num_heads,bias=bias)#传参128->256   32->32
        # self.conv_post = nn.Conv2d(dim,3,kernel_size=3,padding=1,bias=True)

    def forward(self,x):
        # x = self.conv1(x)
        # x = self.patch_embed(x)
        # x = self.res(x)
        x = self.msa1(x)
        # x = self.npm1(x)
        # x1 = self.npm2(x)
        # x1 = self.conv_post(x)
        return x


class Downsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
           # nn.BatchNorm2d(n_feat // 2),
           # nn.Hardswish(),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
           # nn.BatchNorm2d(n_feat*2),
           # nn.Hardswish(),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

# class SKFusion(nn.Module):
#     def __init__(self, dim, height=2, reduction=8):
#         super(SKFusion, self).__init__()
#
#         self.height = height
#         d = max(int(dim / reduction), 4)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, d, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(d, dim * height, 1, bias=False)
#         )
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, in_feats):
#         B, C, H, W = in_feats[0].shape
#
#         in_feats = torch.cat(in_feats, dim=1)
#         in_feats = in_feats.view(B, self.height, C, H, W)
#
#         feats_sum = torch.sum(in_feats, dim=1)
#         attn = self.mlp(self.avg_pool(feats_sum))
#         attn = self.softmax(attn.view(B, self.height, C, 1, 1))
#
#         out = torch.sum(in_feats * attn, dim=1)
#         return out

class Denoisy_Unet(nn.Module):
    def __init__(self,num_heads,bias,in_channel=[24,48,96]):
        super(Denoisy_Unet,self).__init__()
        self.conv1 = nn.Conv2d(3,in_channel[0],3,padding=1,bias=True)

        self.layer1 = Denoisy(num_heads=num_heads,bias=bias,dim=in_channel[0])#24

        self.down1_2 = Downsample(in_channel[0],in_channel[1])

        self.layer2 = Denoisy(num_heads=num_heads,bias=bias,dim=in_channel[1])

        self.down2_3 = Downsample(in_channel[1], in_channel[2])


        self.layer3 = Denoisy(num_heads=num_heads,bias=bias,dim=in_channel[2])#96
        # self.down3_4 = Downsample(in_channel[2], in_channel[3])
        #
        #
        #
        # self.layer4 = Denoisy(num_heads=num_heads,bias=bias,dim=in_channel[3])#48
        #
        # self.up4_3 = Upsample(in_channel[3],in_channel[2])
        # self.layer5 = Denoisy(num_heads=num_heads,bias=bias,dim=in_channel[2])
        self.up3_2 = Upsample(in_channel[2], in_channel[1])
        self.layer6 = Denoisy(num_heads=num_heads, bias=bias, dim=in_channel[1])
        self.up2_1 = Upsample(in_channel[1], in_channel[0])
        self.layer7 = Denoisy(num_heads=num_heads, bias=bias, dim=in_channel[0])

        self.conv2 = nn.Conv2d(in_channel[0],3,kernel_size=3,bias=True,padding=1)
        self.level1 = nn.Conv2d(in_channel[2]*2,in_channel[2],kernel_size=1,bias=True)
        self.level2 = nn.Conv2d(in_channel[1]*2,in_channel[1],kernel_size=1,bias=True)
        self.level3 = nn.Conv2d(in_channel[0]*2,in_channel[0],kernel_size=1,bias=True)


    # def check_image_size(self, x):
    #     # NOTE: for I2I test
    #     _, _, h, w = x.size()
    #     mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
    #     mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
    #     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     return x

    # def forward_feature(self,x):
    def forward(self,x):
        x = self.conv1(x) #3->24
        x0 = self.layer1(x) #24->24
        res_x = x + x0
        # del x,x0
        # x1 = self.patch_embed1(x)
        x1 = self.down1_2(res_x) #24->48
        x11 = self.layer2(x1) #48->48
        res1_x = x1 + x11
        # del x1, x11

        # x2 = self.patch_embed2(x1)
        x2 = self.down2_3(res1_x)#48->72
        x22 = self.layer3(x2)#72->72
        # x2 = self.patch_upembed(x2)
        res2_x = x2 + x22
        # del x2, x22

        # x3 = self.down3_4(res2)#72->96
        # # x3 = self.fusion1([x2,self.skip2(skip2)]) + x2
        # x33 = self.layer4(x3)#96->96
        # # x3 = self.patch_upembed1(x3)
        # res3 = x3 + x33
        #
        # x4 = self.up4_3(res3)#96->72
        # # x4 = self.fusion2([x3,self.skip1(skip1)]) + x3
        # x4 = torch.cat([x4,res2],dim=1)#72->72*2
        # x4_3 = self.level1(x4) #72*2->72
        # x44 = self.layer5(x4_3)#72
        # res4 = x44 + x4_3

        # x3 = self.layer5(res2)#72->72
        # res3 = x3 + res2
        # del x3

        x5 = self.up3_2(res2_x)#72-》48
        # x5 = torch.cat([x5,res1_x],dim=1)#48->48*2
        # x3_2 = self.level2(x5)#48*2->48
        x55 = self.layer6(x5)#48
        res5_x = x55 + x5
        # del x5, x3_2,x55

        x6 = self.up2_1(res5_x)#48->24
        # x6 = torch.cat([x6,res_x],dim=1)#24->24*2
        # x2_1 = self.level3(x6)#24*2->24
        x66 = self.layer7(x6)#24
        res6_x = x66 + x6
        x = self.conv2(res6_x)
        # del x6, x2_1,x66
        return x

    # def forward(self, x):
    #     H, W = x.shape[2:]
    #     x = self.check_image_size(x)
    #
    #     feat = self.forward_feature(x)
    #     # 2022/11/26
    #     K, B = torch.split(feat, (1, 3), dim=1)
    #
    #     x = K * x - B + x
    #     x = x[:, :, :H, :W]
    #     return x


# if __name__ == "__main__":
#     model = Denoisy(num_heads=4,bias=True,dim=64).cuda()
#
#     image = torch.randn(1,3,256,256).cuda()
#     # model = Denoisy_Unet(4,True)
#     # result = model(image)
#     result_model = model(image)
#     print(result_model)



    # att = Attention(64, 8, True).cuda()
    # image = torch.randn(1,64,256,256).cuda()
    # res = att(image)
    # print(res.shape)
    # print(res)
    # resul_att = attention(image)
    # print(result_att.shape)


