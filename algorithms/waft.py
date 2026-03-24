import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from timm.layers import Mlp
from einops import rearrange

from model.iterative import fetch_iterative_module
from model.encoder import fetch_feature_encoder
from model.utils import Padder, disp_warp, gaussian_weights

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    for p in module.buffers():
        p.requires_grad = False

class WAFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.WAFT.ITERATIVE_MODULE.TASK
        self.iters = len(self.task)
        self.n_bins = (int)(cfg.WAFT.LOSS[0].split('_')[-1]) + 1
        self.encoder, self.enc_dim, self.factor = fetch_feature_encoder(cfg.WAFT.FEATURE_ENCODER)
        self.hidden_dim = self.enc_dim
        self.prop_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.PROP_ITER, input_dim=self.hidden_dim)
        self.prop_proj = Mlp(self.enc_dim*2, self.hidden_dim, self.hidden_dim, use_conv=True)
        self.delta_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER, input_dim=self.hidden_dim)
        self.delta_proj = Mlp(self.enc_dim*2+self.hidden_dim+1, self.hidden_dim, self.hidden_dim, use_conv=True)
        
        self.max_disp = cfg.WAFT.MAX_DISP
        self.delta_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4*9, use_conv=True)
        self.delta_dist_head = Mlp(self.hidden_dim, self.hidden_dim, 4, use_conv=True)
        self.delta_disp_head = Mlp(self.hidden_dim, self.hidden_dim, 1, use_conv=True)
        self.prop_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4*9, use_conv=True)
        self.prop_bins_head = Mlp(self.hidden_dim, self.hidden_dim, self.n_bins, use_conv=True)

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def convex_upsample(self, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        return up_info.reshape(N, C, 2*H, 2*W)
    
    def forward(self, sample, disp_init=None):
        """ Estimate disparity between pair of frames """
        output = {}
        image1 = self.normalize_image(sample['img1'])
        image2 = self.normalize_image(sample['img2'])
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        fmap1, fmap2, net = self.encoder(torch.stack([image1, image2], dim=1))
        n, _, h, w = fmap1.shape

        idx_bins_2x = torch.linspace(0, self.max_disp/2, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)
        idx_bins_1x = torch.linspace(0, self.max_disp/1, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)

        prop_hidden = self.prop_proj(torch.cat([fmap1, fmap2], dim=1))
        prop_hidden = self.prop_decoder(prop_hidden)
        prob_mask = .25 * self.prop_mask_head(prop_hidden)
        prob_bins = self.prop_bins_head(prop_hidden)
        prob_up = self.convex_upsample(prob_bins, prob_mask)
        output['init'] = padder.unpad(prob_up)
        prob_bins = F.softmax(prob_bins, dim=1)
        disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

        if disp_init is not None:
            disp = padder.pad(disp_init.unsqueeze(1))
            disp = F.interpolate(disp, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

        delta_disp_preds = []
        delta_info_preds = []
        for itr in range(self.iters):
            disp = disp.detach()
            warped_fmap2 = disp_warp(fmap2, disp, padding_mode='zeros')
            net = self.delta_proj(torch.cat([fmap1, warped_fmap2, net, disp], dim=1))
            net = self.delta_decoder(net)
            info = self.delta_dist_head(net)
            delta_disp = self.delta_disp_head(net)
            mask = .25 * self.delta_mask_head(net)
            disp = disp + delta_disp
            disp_up = self.convex_upsample(disp * 2, mask)
            info_up = self.convex_upsample(info, mask)
            delta_disp_preds.append(disp_up)
            delta_info_preds.append(info_up)

        output['delta_disp_preds'] = [padder.unpad(disp) for disp in delta_disp_preds]
        output['delta_info_preds'] = [padder.unpad(info) for info in delta_info_preds]
        if self.iters > 0:
            disp_final = output['delta_disp_preds'][-1].squeeze(1)
            output['disp_pred'] = disp_final
        else:
            disp_final = torch.sum(F.softmax(output['init'], dim=1) * idx_bins_1x, dim=1)
            output['disp_pred'] = disp_final
        
        return output

    def inference(self, sample, size=None, factor=1.0, disp_init=None): 
        sample = {
            'img1': F.interpolate(sample['img1'], scale_factor=factor, mode='bilinear', align_corners=True),
            'img2': F.interpolate(sample['img2'], scale_factor=factor, mode='bilinear', align_corners=True)
        }
        disp_init = None if disp_init is None else F.interpolate(disp_init.unsqueeze(1), scale_factor=factor, mode='bilinear', align_corners=True).squeeze(1) * factor

        if size is None:
            output = self.forward(sample, disp_init=disp_init)
            for k in output.keys():
                if 'disp' in k:
                    ratio = 1/factor
                else:
                    ratio = 1
                if isinstance(output[k], list):
                    for i in range(len(output[k])):
                        output[k][i] = F.interpolate(output[k][i], scale_factor=1/factor, mode='bilinear', align_corners=True) * ratio
                else:
                    if output[k].dim() == 4:
                        output[k] = F.interpolate(output[k], scale_factor=1/factor, mode='bilinear', align_corners=True)
                    else:
                        output[k] = F.interpolate(output[k][:, None], scale_factor=1/factor, mode='bilinear', align_corners=True).squeeze(1) * ratio
            return output
        
        img1 = sample['img1']
        img2 = sample['img2']
        padder = Padder(img1.shape, size=size)
        img1 = padder.pad(img1)
        img2 = padder.pad(img2)
        disp_init = None if disp_init is None else padder.pad(disp_init)
        hstep = size[0] - 16
        wstep = size[1] - 16
        gau = gaussian_weights(size[0], size[1], device=img1.device).view(1, size[0], size[1])
        b, _, h, w = img1.shape
        weights = torch.zeros((b, h, w), device=img1.device, dtype=img1.dtype)
        output = {}
        for idx_h in range(0, h, hstep):
            for idx_w in range(0, w, wstep):
                rh = min(idx_h + size[0], h)
                rw = min(idx_w + size[1], w)
                lh = rh - size[0]
                lw = rw - size[1]
                sample_patch = {}
                sample_patch['img1'] = img1[:, :, lh:rh, lw:rw]
                sample_patch['img2'] = img2[:, :, lh:rh, lw:rw]
                disp_init_patch = None if disp_init is None else disp_init[:, lh:rh, lw:rw]
                output_patch = self.forward(sample_patch, disp_init=disp_init_patch)
                for k in output_patch.keys():
                    if k not in output:
                        if isinstance(output_patch[k], list):
                            output[k] = []
                            for t in output_patch[k]:
                                if t.dim() == 4:
                                    output[k].append(torch.zeros((b, t.shape[1], h, w), device=t.device, dtype=t.dtype))
                                else:
                                    output[k].append(torch.zeros((b, h, w), device=t.device, dtype=t.dtype))
                        else:
                            t = output_patch[k]
                            if t.dim() == 4:
                                output[k] = torch.zeros((b, t.shape[1], h, w), device=t.device, dtype=t.dtype)
                            else:
                                output[k] = torch.zeros((b, h, w), device=t.device, dtype=t.dtype)
                    if isinstance(output_patch[k], list):
                        for i in range(len(output_patch[k])):
                            output[k][i][..., lh:rh, lw:rw] += output_patch[k][i] * gau
                    else:
                        output[k][..., lh:rh, lw:rw] += output_patch[k] * gau
                weights[:, lh:rh, lw:rw] += gau
        
        for k in output.keys():
            if 'disp' in k:
                ratio = 1/factor
            else:
                ratio = 1
            if isinstance(output[k], list):
                for i in range(len(output[k])):
                    output[k][i] = padder.unpad(output[k][i] / weights)
                    output[k][i] = F.interpolate(output[k][i], scale_factor=1/factor, mode='bilinear', align_corners=True) * ratio
            else:
                output[k] = padder.unpad(output[k] / weights)
                if output[k].dim() == 4:
                    output[k] = F.interpolate(output[k], scale_factor=1/factor, mode='bilinear', align_corners=True)
                else:
                    output[k] = F.interpolate(output[k][:, None], scale_factor=1/factor, mode='bilinear', align_corners=True).squeeze(1) * ratio
        
        return output
    
    def heirarchical_inference(self, sample, size=None, factor_list=None):
        output = {}
        disp_init = None
        for i in range(len(factor_list)):
            output = self.inference(sample, size=size, factor=factor_list[i], disp_init=disp_init)
            disp_init = output['disp_pred']

        return output
