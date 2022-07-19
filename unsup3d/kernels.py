import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor
import torch.nn as nn


# class PositionKernel(Kernel):
#     r"""Position Kernel (non-diffenretiable)

#         inputs are inverse order/permutations
#     """

#     def __init__(self, lamda=1.0):
#         super().__init__()
#         self.lamda = nn.Parameter(torch.ones(1) * lamda)

#     def forward(
#         self,
#         x1: Tensor,
#         x2: Tensor,
#         diag: bool = False,
#         last_dim_is_batch: bool = False,
#         **kwargs
#     ) -> Tensor:

#         distance = torch.abs(x1.unsqueeze(1) - x2.unsqueeze(0)).sum(-1)
#         return torch.exp(-self.lamda.pow(2) * distance)

class PositionKernel(Kernel):
    r"""Position Kernel (non-diffenretiable)

        inputs are inverse order/permutations
    """

    def __init__(self, lamda=1.0):
        super().__init__()
        self.lamda = nn.Parameter(torch.ones(1) * lamda)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs
    ) -> Tensor:

        distance = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)).sum(-1)
        return torch.exp(-self.lamda.pow(2) * distance)


class DAGKernel(Kernel):
    r"""Position Kernel (non-diffenretiable)

        inputs are inverse order/permutations
    """

    def __init__(self, lamda=1.0):
        super().__init__()
        self.lamda = nn.Parameter(torch.ones(1) * lamda)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs
    ) -> Tensor:

        distance = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)).sum(-1)
        return torch.exp(-self.lamda.pow(2) * distance)
