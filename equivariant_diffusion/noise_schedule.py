import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.typing import Tensor


class NoiseSchedule:
    """
    Noise schedule used by Hoogeboom et al.

    See Appendix B of their paper for mathematical description, and I looked at their
    code for implementation details:
    https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py # noqa: E501


    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps.
    precision_val : float, optional
        Precision value to use in alpha calculations, by default 1e-5
    device : torch.device, optional
        PyTorch device to put gammas on, by default torch.device("cpu")
    """

    def __init__(
        self,
        num_timesteps: int,
        precision_val: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):

        self.num_timesteps = num_timesteps

        self.precomputed_t = torch.arange(0, num_timesteps + 1, dtype=torch.float)

        # Appendix B
        alphas2 = (1.0 - (self.precomputed_t / float(num_timesteps)) ** 2) ** 2
        alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
        # "To avoid numerical instabilities during sampling"
        alphas_step = alphas2[1:] / alphas2[:-1]
        alphas_step = np.clip(alphas_step, a_min=0.001, a_max=1.0)
        alphas2 = np.cumprod(alphas_step, axis=0)

        precision = 1 - 2 * precision_val
        self.alphas2 = precision * alphas2 + precision_val

        self.sigmas2 = 1 - self.alphas2

        log_alphas2 = np.log(self.alphas2)
        log_sigmas2 = np.log(self.sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gammas = torch.from_numpy(-log_alphas2_to_sigmas2).float().to(device)

    def gamma(self, t: int):
        """Get gamma value at t."""
        return self.gammas[t]

    def alpha(self, gamma: Tensor):
        """Get alpha value at gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def sigma(self, gamma: Tensor):
        """Get sigma value at gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def sigma2_t_s(self, t: int, s: int):
        r"""Compute $\sigma_{t|s}^2$. See Appendix B + their code for implementation."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)

        sigma2_t_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        return sigma2_t_s

    def alpha_t_s(self, t: int, s: int):
        r"""Compute $\alpha_{t|s}^2$. See Appendix B + their code for implementation."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_s = log_alpha2_t - log_alpha2_s

        alpha_t_s = torch.exp(0.5 * log_alpha2_t_s)

        return alpha_t_s

    def sigma_t_to_s(self, t: int, s: int):
        r"""Compute $\sigma_{t \to s}^2$. See just after Eq. 4."""
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)
        sigma_t, sigma_s = self.sigma(gamma_t), self.sigma(gamma_s)

        sigma2_t_s = self.sigma2_t_s(t, s)
        sigma_t_s = torch.sqrt(sigma2_t_s)

        sigma_t_to_s = sigma_t_s * sigma_s / sigma_t

        return sigma_t_to_s
