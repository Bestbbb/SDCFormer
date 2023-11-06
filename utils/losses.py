# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import pytorch_lightning as pl
import torch.nn.functional as F

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class adaptive_loss(nn.Module):
    def __init__(self) -> None:
        super(adaptive_loss).__init__()
    

    def forward(self,x):
        return x

class KL_distance_loss(nn.Module):
    def __init__(self):
        super(KL_distance_loss,self).__init__()
        self.epsilon = 1e-8

    def _cal_kl_distance(self,p,q):
        p = F.softmax(p,dim=-1)
        q = F.softmax(q,dim=-1)
        return t.sum(p * t.log(p / q), dim=-1)

    def _softmax(self, x):
        x = t.exp(x)
        return x / t.sum(x, dim=-1, keepdim=True)

    def forward(self,batch_x,pred,patch_len):
        batch_size, seq_len, feature_len = batch_x.size()
        _, pred_len, _ = pred.size()

        x_num_segments = seq_len // patch_len
        pred_num_segments = pred_len // patch_len

        x_segments = t.split(batch_x.reshape(-1,seq_len), patch_len, dim=1)
        pred_segments = t.split(pred.reshape(-1,pred_len), patch_len, dim=1)
        # x_segments = t.stack(x_segments, dim=1) # (batch_size, x_num_segments, patch_len, 21)
        # pred_segments = t.stack(pred_segments, dim=1)
        # kl_divs = self._cal_kl_distance(x_segments.unsqueeze(2),pred_segments.unsqueeze(1))
        kl_divs = t.zeros(batch_size*feature_len, x_num_segments, pred_num_segments).to(pred.device)

        kl_divs = kl_divs.clone() + self.epsilon
        # Calculate KL divergence for each segment
        for i, x_seg in enumerate(x_segments):
            for j, pred_seg in enumerate(pred_segments):
                kl_divs[:, i, j] = F.kl_div(F.log_softmax(x_seg, dim=-1),
                                            pred_seg.softmax(dim=-1),
                                            reduction='none').sum(dim=-1)

        weights = F.softmax(kl_divs, dim=-1)

        loss = t.mean(t.sum(weights * kl_divs, dim=(1, 2)))
        return loss


        # kl_divs = t.zeros((batch_size, x_num_segments, pred_num_segments)).to(pred.device)+self.epsilon
   
        # x_segments = t.chunk(batch_x, patch_len, dim=1)
        # pred_segments = t.chunk(pred, patch_len, dim=1)

        # # calculate KL divergence for each segment
        # for i, x_seg in enumerate(x_segments):
        #     for j, pred_seg in enumerate(pred_segments):
        #         print(i,j)
        #         kl_divs[:, i, j] = self._cal_kl_distance(x_seg[:,i], pred_seg[:,j])

        # weights = self._softmax(kl_divs)
        # loss = t.mean(t.sum(kl_divs * weights, dim=(1, 2)))

        # return loss
