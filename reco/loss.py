import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class QualityFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(QualityFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        modulator = (targets - predictions) ** self.gamma
        return (modulator * bce_loss).sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
        Mean binary focal loss

        predictions: a torch tensor containing the predictions, 0s and 1s.
        targets: a torch tensor containing the ground truth, 0s and 1s.

        gamma: focal loss power parameter, a float scalar
            - how much importance is given to misclassified examples
            - 2 is a good start
        alpha: weight of the class indicated by 1, a float scalar.
            - foreground term
            - In practice may be set by inverse class frequency to begin with
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()


def batch_sum(x, batch):
    size = int(batch.max().item() + 1)
    return scatter(x, batch, dim=0, dim_size=size, reduce='sum')


class GraphClassificationLoss(nn.Module):
    def __init__(self):
        super(GraphClassificationLoss, self).__init__()

    def forward(self, preds, data):
        """
        Input is (foreground, background) probabilities (softmax) per sample
        """
        foreground_shared_e = 1 + batch_sum(data.e, data.batch)
        background_shared_e = 1 + batch_sum(data.e - data.shared_e, data.batch)

        foreground = batch_sum(data.shared_e * (preds[:,0] - data.y)**2, data.batch)
        background = batch_sum((data.e - data.shared_e) * (preds[:,1] - (1 - data.y))**2, data.batch)

        batches = foreground / foreground_shared_e + background / background_shared_e

        return batches.sum()


class GraphClassificationLossSingleClass(nn.Module):
    def __init__(self):
        super(GraphClassificationLossSingleClass, self).__init__()

    def forward(self, preds, data):
        """
        Input is (0-1: background-foreground)
        """
        foreground_shared_e = 1 + batch_sum(data.e, data.batch)
        background_shared_e = 1 + batch_sum(data.e - data.shared_e, data.batch)

        foreground = batch_sum(data.shared_e * (preds - data.y)**2, data.batch)
        background = batch_sum((data.e - data.shared_e) * ((1 - preds) - (1 - data.y))**2, data.batch)

        batches = foreground / foreground_shared_e + background / background_shared_e

        return batches.sum()
