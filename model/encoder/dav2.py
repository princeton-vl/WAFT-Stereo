import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import torch.nn.functional as F

import sys
import os
from peft import LoraConfig, get_peft_model
from einops import rearrange
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from model.layers.dpt import UpsampleFeats, ProjFeats

DEPTH_ANYTHING_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
class DAv2Encoder(nn.Module):
    def __init__(self, model_name='vits', alpha=None, r=None):
        super().__init__()
        self.model_name = model_name
        depth_anything = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[model_name])
        depth_anything.load_state_dict(torch.load(f'depth-anything-ckpts/depth_anything_v2_{model_name}.pth', map_location='cpu'))
        self.dpt_configs = {
            'vitl': {'n_layers': 24, "dim": 1024},
            'vitb': {'n_layers': 12, "dim": 768},
            'vits': {'n_layers': 12, "dim": 384},
        }
        self.dim = self.dpt_configs[model_name]['dim']
        self.idx = [(i + 1) * self.dpt_configs[model_name]['n_layers'] // 4 - 1 for i in range(4)]
        self.out_c = [(int)(self.dim * (2 ** (i - 3))) for i in range(4)]
        self.output_dim = self.out_c[0]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"]
        )
        self.encoder = get_peft_model(depth_anything.pretrained, lora_config)
        self.fmap_proj = ProjFeats(self.dim, self.out_c, lvl=-3)
        self.fmap_upsample = UpsampleFeats(self.output_dim, self.out_c)
        self.hidden_proj = ProjFeats(self.dim*2, self.out_c, lvl=-3)
        self.hidden_upsample = UpsampleFeats(self.output_dim, self.out_c) 

    def forward(self, imgs):
        B, N, _, H, W = imgs.shape
        # encode by dinov2
        h, w = H // 16, W // 16
        imgs = imgs.reshape(B*N, _, H, W)
        
        imgs = F.interpolate(imgs, (h*14, w*14), mode='bilinear', align_corners=True)
        feats = self.encoder.get_intermediate_layers(imgs, self.idx, return_class_token=True)
        fmap_feats = [rearrange(x[0], 'bn (h w) c -> bn c h w', h=h, w=w) for x in feats]
        fmap_feats = self.fmap_proj(fmap_feats)
        fmap_feats = self.fmap_upsample(fmap_feats)
        fmaps = rearrange(fmap_feats[0], '(b n) c h w -> b n c h w', n=2)
        fmap1 = fmaps[:, 0]
        fmap2 = fmaps[:, 1]

        hidden_feats = [rearrange(x[0], '(b n) (h w) c -> b (n c) h w', n=2, h=h, w=w) for x in feats]
        hidden_feats = self.hidden_proj(hidden_feats)
        hidden_feats = self.hidden_upsample(hidden_feats)
        hidden = hidden_feats[0]

        return fmap1, fmap2, hidden

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model = DAv2Encoder(alpha=16, r=8).cuda()
    input = torch.randn(1, 2, 3, 504, 756).cuda()
    fmap1, fmap2, net = model(input)
    print(fmap1.shape, fmap2.shape, net.shape)
    # for name, module in model.named_modules():
    #     if isinstance(module, PeftModel):
    #         print(f"{name} is a PeftModel with {count_parameters(module)} trainable parameters.")
    #         module.merge_and_unload()
    # with torch.no_grad():
    #     with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA
    #         ],
    #         with_flops=True) as prof:
    #             output = model(input)

    #     print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    #     events = prof.events()
    #     forward_MACs = sum([int(evt.flops) for evt in events])
    #     print("forward MACs: ", forward_MACs / 2 / 1e9, "G")
    #     def count_parameters(model):
    #         return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("Number of parameters: ", count_parameters(model) / 1e6, "M")