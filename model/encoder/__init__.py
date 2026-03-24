from .dinov3 import DINOv3Encoder
from .dav2 import DAv2Encoder

def fetch_feature_encoder(cfg):
    if cfg.TYPE == 'dinov3':
        factor = 16
        encoder_lora_rank = cfg.LORA_RANK
        encoder_lora_alpha = cfg.LORA_ALPHA
        encoder = DINOv3Encoder(
            model_name=cfg.ARCH,
            alpha=encoder_lora_alpha,
            r=encoder_lora_rank,
        )
        encoder_dim = encoder.output_dim
    
    elif cfg.TYPE == 'dav2':
        factor = 16
        encoder_lora_rank = cfg.LORA_RANK
        encoder_lora_alpha = cfg.LORA_ALPHA
        encoder = DAv2Encoder(
            model_name=cfg.ARCH,
            alpha=encoder_lora_alpha,
            r=encoder_lora_rank,
        )
        encoder_dim = encoder.output_dim
    else:
        raise ValueError(f"Unknown feature encoder: {cfg.TYPE}")

    return encoder, encoder_dim, factor