import torch
import math
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

def mixlap_loss(output, target, loss_gamma=0.9, max_disp=192):
    disp_gt = target['disp']
    valid = target['valid']
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    if valid.sum() == 0:
        return torch.sum(output['delta_disp_preds'][0]) * 0.0
    
    lap_loss = 0.0
    n_predictions = len(output['delta_disp_preds'])
    for i in range(n_predictions):
        i_weight = loss_gamma**(n_predictions - i - 1)
        disp_pred = output['delta_disp_preds'][i].squeeze(1)
        weights = output['delta_info_preds'][i][:, :2]
        # laplace likelihood loss
        log_b = torch.clamp(output['delta_info_preds'][i][:, 2], min=0, max=10)
        term0 = (disp_gt - disp_pred).abs() * torch.exp(-log_b) + log_b + math.log(2)
        term1 = (disp_gt - disp_pred).abs() + math.log(2)
        lap_term = torch.logsumexp(weights[:, :2], dim=1) - torch.logsumexp(weights[:, :2] - torch.stack([term0, term1], dim=1), dim=1)
        lap_loss += i_weight * lap_term[valid.bool() & ~torch.isnan(lap_term)].mean()

    return lap_loss

def init_loss(output, target, max_disp=192):
    prob = output['init']
    disp_gt = target['disp']
    valid = target['valid']
    n_bins = prob.shape[1]
    # Keep tensors on the same device/dtype for scatter_reduce_ under mixed precision.
    disp_gt = disp_gt.to(prob.device)
    valid = valid.to(prob.device)
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    if valid.sum() == 0:
        return torch.tensor(0.0, device=disp_gt.device)
    disp_gt = torch.clamp(disp_gt, min=0, max=max_disp-1)
    idx_bins = torch.linspace(0, max_disp, n_bins, device=disp_gt.device, dtype=disp_gt.dtype).view(1, n_bins, 1, 1)
    label = F.softmax(-(idx_bins - disp_gt.unsqueeze(1)).abs(), dim=1)
    # softmax before kl
    prob = F.softmax(prob, dim=1)
    kl_loss = -(torch.log(torch.clamp(prob, min=1e-6)) * label).sum(dim=1)
    return kl_loss[valid.bool() & ~torch.isnan(kl_loss)].mean()

class WAFTCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight_dict = {}
        if 'mixlap' in self.cfg.WAFT.LOSS[0]:
            self.weight_dict.update({'mixlap': 1.0})
            self.weight_dict.update({'init': 1.0})
        elif 'wprob' in self.cfg.WAFT.LOSS[0]:
            self.weight_dict.update({'mixlap': 1.0})
            self.weight_dict.update({'kl': 1.0})
        
    def forward(self, outputs, targets, log):
        loss_dict = {}
        epe = (outputs['disp_pred'] - targets['disp']).abs()
        valid = (targets['valid'] >= 0.5).to(epe.device)
        metrics = {
            'EPE': epe[valid].mean(),
            'BP-1': (epe[valid] > 1).float().mean().item(),
            'BP-2': (epe[valid] > 2).float().mean().item(),
            'D1': ((epe[valid] > 3) & (epe[valid] / targets['disp'][valid] > 0.05)).float().mean().item()
        }
        
        if 'mixlap' in self.cfg.WAFT.LOSS[0]:
            loss = init_loss(outputs, targets, max_disp=self.cfg.WAFT.MAX_DISP)
            loss_dict.update({'init': loss})
            metrics.update({'init': loss.item()}) 
            loss = mixlap_loss(outputs, targets, loss_gamma=0.9, max_disp=self.cfg.WAFT.MAX_DISP)
            loss_dict.update({'mixlap': loss})
            metrics.update({'mixlap': loss.item()})
        elif 'wprob' in self.cfg.WAFT.LOSS[0]:
            n_bins = (int)(self.cfg.WAFT.LOSS[0].split('_')[-1]) + 1
            tasks = self.cfg.WAFT.ITERATIVE_MODULE.TASK
            lap_loss, kl_loss = wprob_loss(outputs, targets, n_bins, loss_gamma=0.9, max_disp=self.cfg.WAFT.MAX_DISP)
            loss_dict.update({'mixlap': lap_loss})
            loss_dict.update({'kl': kl_loss})
            metrics.update({'mixlap': lap_loss})
            metrics.update({'kl': kl_loss})
        else:
            raise ValueError(f"Unknown WAFT loss type: {self.cfg.WAFT.LOSS[0]}")
        
        return loss_dict, metrics
        
def build_criterion(cfg):
    if cfg.ALGORITHM == "waft":
        return WAFTCriterion(cfg)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.ALGORITHM}")
    
