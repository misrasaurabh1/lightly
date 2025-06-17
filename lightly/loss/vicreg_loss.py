import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from lightly.utils.dist import gather


class VICRegLoss(torch.nn.Module):
    """Implementation of the VICReg loss [0].

    This implementation is based on the code published by the authors [1].

    - [0] VICReg, 2022, https://arxiv.org/abs/2105.04906
    - [1] https://github.com/facebookresearch/vicreg/

    Attributes:
        lambda_param:
            Scaling coefficient for the invariance term of the loss.
        mu_param:
            Scaling coefficient for the variance term of the loss.
        nu_param:
            Scaling coefficient for the covariance term of the loss.
        gather_distributed:
            If True, the cross-correlation matrices from all GPUs are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = VICRegLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps: float = 0.0001,
    ):
        """Initializes the VICRegLoss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(VICRegLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).

        Returns:
            The computed VICReg loss.

        Raises:
            AssertionError: If z_a or z_b have a batch size <= 1.
            AssertionError: If z_a and z_b do not have the same shape.
        """
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # Invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # Gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        # Variance and covariance terms of the loss
        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps)
        )
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        # Total VICReg loss
        loss = (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )
        return loss


def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).

    Returns:
        The computed VICReg invariance loss.
    """
    return F.mse_loss(x, y)


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.

    Returns:
        The computed VICReg variance loss.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x: Tensor with shape (batch_size, ..., dim).

    Returns:
          The computed VICReg covariance loss.
    """
    # Use keepdim to prevent large repeated broadcasting allocations
    x = x - x.mean(dim=0, keepdim=True)
    batch_size = x.size(0)
    dim = x.size(-1)

    # Precompute mask only once per dim per device
    nondiag_mask = _get_nondiag_mask(dim, x.device)

    # ---------- OPTIMIZED COVARIANCE COMPUTATION -------------
    # Instead of einsum, use reshape + matmul for batch matmul; much faster.
    other_dims = x.shape[1:-1]
    if other_dims:
        # flatten all non-batch dimensions for efficient matmul
        x_flat = x.reshape(batch_size, -1, dim)
        # x_flat: (batch, any, dim)
        # want output: (..., dim, dim): that is, treat each "slot" independently
        # We can do a transpose (0,2,1): (batch, dim, any)
        # cov for each "slot": (any, dim, dim), treat (any=prod(other_dims))
        x_flat_trans = x_flat.permute(1, 0, 2)  # (any, batch, dim)
        # batch_matmul: (any, batch, dim)T @ (any, batch, dim) --> (any, dim, dim)
        cov = torch.matmul(x_flat_trans.transpose(1, 2), x_flat_trans) / (
            batch_size - 1
        )
        # (any, dim, dim). Now reshape to original shape (..., dim, dim)
        cov = cov.reshape(*other_dims, dim, dim)
    else:
        # No extra dims, just (batch, dim)
        cov = (x.T @ x) / (batch_size - 1)

    # ------------ OPTIMIZED OFFDIAGONAL LOSS -------------
    # Instead of masking and then sum(-1), use .sum(-2, -1) for contiguous compute
    # Masking creates a 1D view of all off-diagonal elements; we can achieve sum faster:
    cov2 = cov.pow(2)
    offdiag_sum = cov2.sum(dim=(-2, -1)) - cov2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # Divide by dim to average (as before)
    loss = offdiag_sum / dim

    return loss.mean()


def _get_nondiag_mask(dim: int, device) -> torch.Tensor:
    # Create or retrieve the non-diagonal mask efficiently and move to device only once
    eye = torch.eye(dim, device=device, dtype=torch.bool)
    return ~eye
