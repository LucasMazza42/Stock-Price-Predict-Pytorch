import torch
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self):
        super(MyCustomLoss, self).__init__()

    def forward(self, pred, target):
        pred_mean, pred_log_var = pred[:, 0], pred[:, 1]
        true_price = target

        mse_loss = nn.MSELoss()
        mse = mse_loss(pred_mean, true_price)

        # Compute negative log-likelihood component (Gaussian likelihood)
        nll_loss = 0.5 * (pred_log_var + torch.exp(-pred_log_var) * (true_price - pred_mean)**2)
        nll = torch.mean(nll_loss)

        total_loss = mse + nll

        return total_loss