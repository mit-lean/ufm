import torch
import torch.nn as nn

class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super(HeteroscedasticLoss, self).__init__()
    def forward(self, pred, target):
        log_var = torch.reshape(pred[:,1,:,:],(pred.shape[0],1,224,224))
        pred = torch.reshape(pred[:,0,:,:],(pred.shape[0],1,224,224))
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        log_var = log_var[valid_mask]
        precision = torch.exp(-log_var)
        self.loss = (0.5*precision * diff**2).mean() + (0.5*log_var).mean()
        return self.loss

