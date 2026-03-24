import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import torch
import copy

from peft import PeftModel
from algorithms.waft import WAFT
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils import visualization
from bridgedepth.loss import build_criterion

def setup(args):
    """
    Create config and perform basic setups.
    """
    from bridgedepth.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def macs_profiler(model, criterion, enable_bf16=False):
    b = 1
    input = torch.randn(b, 3, 540, 960).cuda()
    sample = {
        "img1": input,
        "img2": input,
        "disp": torch.ones(b, 540, 960).cuda() * 100,
        "valid": torch.ones(b, 540, 960).cuda()
    }

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True) as prof:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bf16):
                output = model(sample)
                loss_dict, metrics = criterion(output, sample, log=True)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    if enable_bf16:
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
        print("Note: BF16 profiling may not be accurate for MACs due to potential optimizations. Consider using FP32 for more precise measurements.")
    else:
        events = prof.events()
        forward_MACs = sum([int(evt.flops) for evt in events])
        print("forward MACs: ", forward_MACs / 2 / 1e9, "G")
        print("Number of parameters: ", count_parameters(model) / 1e6, "M")

    print("===== Loss Results =====")
    print("Total Loss: ", losses.item())
    for k, _ in weight_dict.items():
        print(f"Loss component: {k}, Weight: {weight_dict[k]}, Value: {loss_dict[k]}")
                    
def profile_peak_mem(model, warmup=10, iters=30):
    device = next(model.parameters()).device
    assert device.type == "cuda"

    b = 1
    input = torch.randn(b, 3, 540, 960).cuda()
    sample = {
        "img1": input,
        "img2": input,
        "disp": torch.ones(b, 540, 960).cuda() * 100,
        "valid": torch.ones(b, 540, 960).cuda()
    }
    model.eval()
    with torch.inference_mode():
        # Warmup (important: allocators + kernels stabilize)
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                output = model(sample)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        # (optional) clear any cached blocks so "reserved" starts lower:
        # torch.cuda.empty_cache()

        for _ in range(iters):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                output = model(sample)

        torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device)     # bytes actually allocated by tensors
    peak_reserved = torch.cuda.max_memory_reserved(device)   # bytes held by caching allocator
    return peak_alloc, peak_reserved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
    
    cfg = setup(args)
    model = WAFT(cfg)
    for name, module in model.named_modules():
        if isinstance(module, PeftModel):
            print(f"{name} is a PeftModel with {count_parameters(module)} trainable parameters.")
            module.merge_and_unload()
            
    model = model.to(torch.device("cuda")).eval()
    criterion = build_criterion(cfg)
    macs_profiler(model, criterion, enable_bf16=False)
    # macs_profiler(model, criterion, enable_bf16=True)
    # # Example:
    # peak_alloc, peak_reserved = profile_peak_mem(model)
    # print(f"Peak allocated: {peak_alloc/1024**3:.2f} GiB")
    # print(f"Peak reserved : {peak_reserved/1024**3:.2f} GiB")