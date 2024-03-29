import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input + eps)
    return loss

class ConfidentialIntervalLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, min_value, max_value):
        mask = torch.where((pred <= max_value) & (pred >= min_value), target, pred)
        loss = cross_entropy_loss(mask, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds

class GaussianNoise(nn.Module):
    """add gasussian noise into feature"""

    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros_ = torch.zeros_like(x)
        n = torch.normal(zeros_, std=self.std)
        return x + n


class PiModelLoss(nn.Module):
    def __init__(self, std=0.15):
        super(PiModelLoss, self).__init__()
        self.gn = GaussianNoise(std)

    def forward(self, model, x):
        logits1 = model(x)
        probs1 = F.softmax(logits1, dim=1)
        with torch.no_grad():
            logits2 = model(self.gn(x))
            probs2 = F.softmax(logits2, dim=1)
        loss = F.mse_loss(probs1, probs2, reduction="sum") / x.size(0)
        return loss

def consistency_loss_function(consistency, consistency_criterion, model, train_loader, img, epoch, batch_size):
    if consistency != "none":
        consistency_loss = consistency_criterion(model, img)
        consistency_rampup = 0.4 * epoch * len(train_loader) / batch_size
        alpha = get_rampup_weight(0.05, epoch, consistency_rampup)
        consistency_loss = alpha * consistency_loss
    else:
        consistency_loss = torch.tensor(0.0)
    return consistency_loss