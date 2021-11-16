import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self, weight=100):
        super(MaskedMSELoss, self).__init__()
        self.loss = weighted_masked_mse_loss
        self.weight = weight

    def forward(self, pred_t, target_t,
                pred_r, target_r, dataset_idx):
        return self.loss(pred_t, target_t,
                         pred_r, target_r,
                         dataset_idx, self.weight)


def weighted_masked_mse_loss(pred_t, target_t, pred_r, target_r,
                      dataset_idx, k):
    mask = dataset_idx.bool().unsqueeze(-1)
    criterion = nn.MSELoss()
    t_loss = criterion(pred_t.squeeze(), target_t)
    pred_r = torch.masked_select(pred_r.squeeze(), mask)
    target_r = torch.masked_select(target_r, mask)
    # to prevent nan which is caused from all vicon room data in a batch
    if pred_r.nelement() == 0:
        r_loss = torch.zeros_like(t_loss)
    else:
        r_loss = criterion(pred_r, target_r)
    print(t_loss)
    print(r_loss)
    loss = t_loss + k * r_loss
    return loss


class MSELoss(nn.Module):
    def __init__(self, weight=100):
        super(MSELoss, self).__init__()
        self.loss = weighted_mse_loss
        self.weight = weight

    def forward(self, pred_t, target_t,
                pred_r, target_r):
        return self.loss(pred_t, target_t,
                         pred_r, target_r, self.weight)


def weighted_mse_loss(pred_t, target_t, pred_r, target_r, k):
    criterion = nn.MSELoss()
    t_loss = criterion(pred_t.squeeze(), target_t)
    r_loss = criterion(pred_r.squeeze(), target_r)
    # print(t_loss)
    # print(r_loss)
    loss = t_loss + k * r_loss
    return loss, t_loss, k * r_loss
