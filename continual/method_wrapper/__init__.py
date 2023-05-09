from .maskformer import MaskFormerDistillation
from .per_pixel import PerPixelDistillation


def build_wrapper(cfg, model, model_old):
    if cfg.MODEL.MASK_FORMER.PER_PIXEL:
        return PerPixelDistillation(cfg, model, model_old)
    else:
        return MaskFormerDistillation(cfg, model, model_old)
