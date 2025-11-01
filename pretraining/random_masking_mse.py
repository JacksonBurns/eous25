import torch
from chemprop.nn.metrics import LossFunctionRegistry, MetricRegistry, ChempropMetric

@LossFunctionRegistry.register("random-mse")
@MetricRegistry.register("random-mse")
class RandomMaskingMSE(ChempropMetric):
    def __init__(self, mask_percent: float = 0.85):
        super().__init__()
        self.mask_percent = mask_percent
    
    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, *args) -> torch.Tensor:
        mask = (torch.rand_like(targets) > self.mask_percent).float()
        loss =  torch.nn.functional.mse_loss(preds, targets, reduction="none")
        return loss * mask
