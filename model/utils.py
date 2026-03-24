import torch
import torch.nn as nn
import torch.nn.functional as F

class Padder:
    """ Pads images such that dimensions are divisible by factor """
    def __init__(self, dims, factor=None, size=None):
        self.ht, self.wd = dims[-2:]
        if factor is not None:
            pad_ht = (((self.ht - 1) // factor) + 1) * factor - self.ht
            pad_wd = (((self.wd - 1) // factor) + 1) * factor - self.wd
        else:
            pad_ht = (size[0] - self.ht) if self.ht < size[0] else 0
            pad_wd = (size[1] - self.wd) if self.wd < size[1] else 0
        
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, x):
        return F.pad(x, self._pad, mode='constant', value=0)

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
def gaussian_weights(h, w, sigma_y=None, sigma_x=None, device=None):
    if sigma_y is None:
        sigma_y = h / 4
    if sigma_x is None:
        sigma_x = w / 4

    y = torch.arange(h, device=device)
    x = torch.arange(w, device=device)

    yy, xx = torch.meshgrid(y, x, indexing='ij')

    cy = (h - 1) / 2
    cx = (w - 1) / 2

    weights = torch.exp(
        -((yy - cy)**2 / (2 * sigma_y**2) +
        (xx - cx)**2 / (2 * sigma_x**2))
    )
    return weights

def normalize_coords(grid):
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img):
    b, _, h, w = img.size()
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)
    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]
    return grid

def disp_warp(feature, disp, padding_mode='border'):
    grid = meshgrid(feature)  # [B, C, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_feature = F.grid_sample(feature, sample_grid, mode='bilinear', padding_mode=padding_mode)
    return warped_feature