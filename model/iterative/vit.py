import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import sys

from timm.layers import Mlp
from einops import rearrange
from peft import LoraConfig, get_peft_model
from model.layers.dpt import UpsampleFeats, ProjFeats
from model.layers.block import resconv, conv3x3

class VitIter(nn.Module):
    def __init__(self, model_name, input_dim, patch_size=8, alpha=None, r=None, res_layers=4):
        super(VitIter, self).__init__()
        self.dpt_configs = {
            'vitl': {'encoder': 'vit_large_patch16_224', 'n_layers': 24, 'dim': 1024},
            'vitb': {'encoder': 'vit_base_patch16_224', 'n_layers': 12, 'dim': 768},
            'vits': {'encoder': 'vit_small_patch16_224', 'n_layers': 12, 'dim': 384},
            'vitt': {'encoder': 'vit_tiny_patch16_224', 'n_layers': 12, 'dim': 192}
        }
        self.dim = self.dpt_configs[model_name]['dim']
        self.n_idx = (int)(math.log2(patch_size)) + 1
        self.idx = [(i + 1) * self.dpt_configs[model_name]['n_layers'] // self.n_idx - 1 for i in range(self.n_idx)]
        self.out_c = [(int)(self.dim * (2 ** (i - self.n_idx + 1))) for i in range(self.n_idx)]
        vit = timm.create_model(
            self.dpt_configs[model_name]['encoder'],
            pretrained=True
        )
        if r is not None:
            self.blks = nn.ModuleList()
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["qkv", "proj"]
            )
            for blk in vit.blocks:
                self.blks.append(get_peft_model(blk, lora_config))
        else:
            self.blks = vit.blocks

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.init = Mlp(input_dim, input_dim, self.out_c[0], use_conv=True)
        patch_layers = (int)(math.log2(patch_size))
        # self.patch_embed = nn.Conv2d(self.out_c[0], self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embed = nn.Sequential(
            *[resconv(self.out_c[i], self.out_c[i+1], k=3, s=2) for i in range(patch_layers)]
        )
        self.res_convs = nn.Sequential(
            *[resconv(self.out_c[0], self.out_c[0], k=3, s=1) for _ in range(res_layers)]
        )
        self.proj = ProjFeats(self.dim, self.out_c, lvl=-self.n_idx+1)
        self.upsample = UpsampleFeats(self.out_c[0], self.out_c)
        self.final_mlp = Mlp(2*self.out_c[0]+input_dim, input_dim, input_dim, use_conv=True)

    def forward(self, inp):
        x = self.init(inp)

        vit_x = self.patch_embed(x)
        h, w = vit_x.shape[-2:]
        vit_x = rearrange(vit_x, 'b c h w -> b (h w) c')
        vit_feats = []
        for i in range(len(self.blks)):
            vit_x = self.blks[i](vit_x)
            if i in self.idx:
                vit_feats.append(rearrange(vit_x, 'b (h w) c -> b c h w', h=h, w=w))

        vit_feats = self.proj(vit_feats)
        vit_feats = self.upsample(vit_feats)

        res_x = self.res_convs(x)

        final = self.final_mlp(torch.cat([vit_feats[0], res_x, inp], dim=1))
        return final

if __name__ == '__main__':
    model = VitIter('vitt', 95, patch_size=8, alpha=16, r=8)
    print(model.blks)
    # input = torch.randn(1, 95, 512, 768)
    # output = model(input)
    # print(output.shape)
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA
    #     ],
    #     with_flops=True) as prof:
    #         output = model(input)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    # events = prof.events()
    # forward_MACs = sum([int(evt.flops) for evt in events])
    # print("forward MACs: ", forward_MACs / 2 / 1e9, "G")