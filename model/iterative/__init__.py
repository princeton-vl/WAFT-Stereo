from .vit import VitIter

def fetch_iterative_module(cfg, input_dim=3):
    if cfg.TYPE == 'vit':
        decoder_lora_rank = cfg.LORA_RANK
        decoder_lora_alpha = cfg.LORA_ALPHA
        iter_decoder = VitIter(
            cfg.ARCH, 
            input_dim,
            patch_size=cfg.PATCH_SIZE,
            alpha=decoder_lora_alpha,
            r=decoder_lora_rank,
        )
    else:
        raise ValueError(f"Unknown iterative module: {cfg.TYPE}")
    
    return iter_decoder