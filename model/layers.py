import torch


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



"""
Adv classifier & Grad reverse codes from: 
https://github.com/thuhcsi/tacotron/tree/186d5b89f7c73c62aa87937460e7556bcc2a8997
"""

class AdversarialClassifier(torch.nn.Module):
    """
    AdversarialClassifier
        - 1 gradident reversal layer
        - n hidden linear layers with ReLU activation
        - 1 output linear layer with Softmax activation
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[256], rev_scale=1):
        """
        Args:
             in_dim: input dimension
            out_dim: number of units of output layer (number of classes)
        hidden_dims: number of units of hidden layers
          rev_scale: gradient reversal scale
        """
        super(AdversarialClassifier, self).__init__()

        self.gradient_rev = GradientReversal(rev_scale)

        in_sizes = [in_dim] + hidden_dims[:]
        out_sizes = hidden_dims[:] + [out_dim]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.activations = [torch.nn.ReLU()] * len(hidden_dims) + [torch.nn.Softmax(dim=-1)]

    def forward(self, x):
        x = self.gradient_rev(x)
        for (linear, activate) in zip(self.layers, self.activations):
            x = activate(linear(x))
        return x


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function.
    In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer
    Code from:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)