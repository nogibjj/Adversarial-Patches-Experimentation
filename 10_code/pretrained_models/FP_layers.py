import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit, symmetric=False):
        """
        symmetric: True for symmetric quantization, False for asymmetric quantization
        """
        if bit is None:
            wq = w
        elif bit == 0:
            wq = w * 0
        else:
            # Build a mask to record position of zero weights

            weight_mask = torch.where(w != 0, 1, 0)

            # Lab3 (a), Your code here:
            if symmetric == False:
                # Compute alpha (scale) for dynamic scaling - QUESTION: do we perform global or local scaling?

                w_copy_min = w.clone()  # for calculating minimum value
                w_copy_min[
                    weight_mask == 0
                ] = (
                    w.max()
                )  # to ensure that these values will not be picked as the minimum

                alpha = (
                    w.max() - w_copy_min.min()
                )  # only applied to non-zero weights. - take it for the entire weight matrix.
                # Compute beta (bias) for dynamic scaling
                beta = w_copy_min.min()
                # Scale w with alpha and beta so that all elements in ws are between 0 and 1
                ws = (w - beta) / alpha

                step = 2 ** (bit) - 1

                # Quantize ws with a linear quantizer to "bit" bits
                R = (1 / step) * torch.round(step * ws)

                # Scale the quantized weight R back with alpha and beta
                wq = alpha * R + beta

            # Lab3 (e), Your code here:
            else:
                max_value = torch.abs(w).max()
                min_value = -1 * max_value
                # calculating alpha
                alpha = max_value - min_value

                # calculating beta
                beta = min_value

                # scale w with alpha and beta so that all elements in ws are between -min_value and max_value
                ws = (w - beta) / alpha

                step = 2 ** (bit) - 1

                # Quantize ws with a linear quantizer to "bit" bits
                R = (1 / step) * torch.round(step * ws)

                # Scale the quantized weight R back with alpha and beta
                wq = alpha * R + beta
            # Restore zero elements in wq
            wq = wq * weight_mask

        return wq

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class FP_Linear(nn.Module):
    def __init__(self, in_features, out_features, Nbits=None, symmetric=False):
        super(FP_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.Nbits = Nbits
        self.symmetric = symmetric

        # Initailization
        m = self.in_features
        n = self.out_features
        self.linear.weight.data.normal_(0, math.sqrt(2.0 / (m + n)))

    def forward(self, x):
        return F.linear(
            x,
            STE.apply(self.linear.weight, self.Nbits, self.symmetric),
            self.linear.bias,
        )


class FP_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        Nbits=None,
        symmetric=False,
    ):
        super(FP_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.Nbits = Nbits
        self.symmetric = symmetric

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2.0 / (n + m)))
        self.sparsity = 1.0

    def forward(self, x):
        return F.conv2d(
            x,
            STE.apply(self.conv.weight, self.Nbits, self.symmetric),
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
