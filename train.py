from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, rc
from rich.progress import track
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.typing import Tensor
from torch_geometric.utils import to_dense_batch, to_undirected

from equivariant_diffusion.egnn import EquivariantGNN
from equivariant_diffusion.noise_schedule import NoiseSchedule
from equivariant_diffusion.straight_line_dataset import StraightLineDataLoader

CONFIG = {
    "hidden_embedding_dimension": 128,
    "message_dimension": 128,
    "num_layers": 4,
    "num_timesteps": 1000,
    "precision_val": 1e-5,
    "num_epochs": 1000,
    "learning_rate": 1e-3,
    "num_data_examples": 1000,
    "min_num_nodes": 4,
    "max_num_nodes": 10,
    "min_line_length": 5.0,
    "max_line_length": 10.0,
    "batch_size": 32,
    "plot_every_n_epochs": 25,
}


def sample_random_noise(batch: Batch, max_timestep: int) -> Tuple[Tensor, Tensor]:
    """Sample mean zero random noise of the same shape as the batch."""
    device = batch.x.device
    batch_size = batch.batch.unique().shape[0]

    # TODO: Should this be from 0? All algorithms seem to suggest so
    t = torch.randint(1, max_timestep, (batch_size,), device=device)
    t = t[batch.batch]

    epsilon = torch.randn_like(batch.x)
    # Subtract center of gravity
    dense_batch_epsilon, _ = to_dense_batch(epsilon, batch=batch.batch, fill_value=0.0)
    dense_epsilon_CoMs = dense_batch_epsilon.mean(axis=1)
    epsilon -= dense_epsilon_CoMs[batch.batch]

    return t, epsilon


def compute_z(
    x: Tensor, t: Tensor, epsilon: Tensor, noise_schedule: NoiseSchedule
) -> Tensor:
    """Compute the noisy vector z_t. See algorithm 1 from the paper."""
    gamma_t = noise_schedule.gamma(t)
    alpha_t = noise_schedule.alpha(gamma_t)
    sigma_t = noise_schedule.sigma(gamma_t)
    z_t = (alpha_t * x.T).T + (sigma_t * epsilon.T).T

    return z_t


def sample_from_model(model, device, noise_schedule, num_nodes: int = 5):
    """
    Sample z_T and return a sequence of increasingly denoised z_t vectors.

    Parameters
    ----------
    model : _type_
        _description_
    device : _type_
        _description_
    noise_schedule : _type_
        _description_
    num_nodes : int, optional
        _description_, by default 5

    Returns
    -------
    _type_
        _description_
    """
    with torch.no_grad():
        z_T = torch.randn((num_nodes, 2), device=device)
        z_T -= z_T.mean(axis=0)
        z_t = z_T

        fully_connected_edge_indices = torch.tensor(
            [
                (u, v)
                for u, v in product(
                    torch.arange(0, num_nodes), torch.arange(0, num_nodes)
                )
                if u != v
            ]
        ).T

        edge_index = to_undirected(fully_connected_edge_indices).to(device)

        all_z_values = [z_T]

        for s in track(
            reversed(range(0, noise_schedule.num_timesteps)),
            description="Sampling from model",
        ):
            s = torch.tensor([s])
            t = s + 1

            gamma_t = noise_schedule.gamma(t)
            sigma_t = noise_schedule.sigma(gamma_t)

            sigma2_t_s = noise_schedule.sigma2_t_s(t, s)
            alpha_t_s = noise_schedule.alpha_t_s(t, s)

            sigma_t_to_s = noise_schedule.sigma_t_to_s(t, s)

            epsilon = torch.randn((num_nodes, 2), device=device)
            epsilon_CoM = epsilon.mean(axis=0)
            epsilon -= epsilon_CoM

            predicted_epsilon = model(
                z_t,
                edge_index,
                torch.tensor([t for _ in z_t], device=device),
                torch.zeros(len(z_t), device=device, dtype=torch.int64),
            )

            z_s = (
                z_t / alpha_t_s
                - (sigma2_t_s / (alpha_t_s * sigma_t)) * predicted_epsilon
                + sigma_t_to_s * epsilon
            )
            # Ensure zero CoM to "avoid numerical runaway of the center of gravity."
            z_s -= z_s.mean(axis=0)
            all_z_values.append(z_s)
            z_t = z_s

    return all_z_values


def save_sample_gif(model, device, noise_schedule, epoch, num_nodes: int = 5):
    """Save a gif of this reverse diffusion process."""
    z_values = sample_from_model(model, device, noise_schedule, num_nodes=num_nodes)

    subsampled_z_values = z_values[::10] + [z_values[-1]]
    print(f"Final z values = {z_values[-1]}")

    rc("animation", html="html5")
    fig, ax = plt.subplots(figsize=(20, 20))

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.axis("equal")

    scatter = ax.scatter([], [])

    def init():
        return (scatter,)

    def animate(i):
        scatter.set_offsets(subsampled_z_values[i].cpu().numpy())
        return (scatter,)

    print("Animating")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=101, interval=20, blit=True
    )
    anim.save(f"./plots/diffusion_samples/{epoch}.gif", writer="ffmpeg", fps=20)
    print("Finished animating")


if __name__ == "__main__":
    model_params = {
        CONFIG["hidden_embedding_dimension"],
        CONFIG["message_dimension"],
        CONFIG["num_layers"],
        CONFIG["num_timesteps"],
    }
    noise_schedule_params = {
        CONFIG["num_timesteps"],
        CONFIG["precision_val"],
    }
    loader_params = {
        CONFIG["num_data_examples"],
        CONFIG["min_num_nodes"],
        CONFIG["max_num_nodes"],
        CONFIG["min_line_length"],
        CONFIG["max_line_length"],
        CONFIG["batch_size"],
    }

    num_timesteps = CONFIG["num_timesteps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EquivariantGNN(**model_params)
    model.to(device)

    ns = NoiseSchedule(
        **noise_schedule_params,
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    loader = StraightLineDataLoader(**loader_params)

    for epoch in range(CONFIG["num_epochs"]):
        epoch_losses = []
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            batch = batch.to(device)
            t, epsilon = sample_random_noise(batch, num_timesteps)

            z_t = compute_z(batch.x, t, epsilon, ns)

            predicted_epsilon = model(z_t, batch.edge_index, t, batch.batch)

            loss = loss_fn(predicted_epsilon, epsilon)

            loss.backward()

            epoch_losses.append(loss.item())

            optimizer.step()

        print(f"Average epoch loss: {np.mean(epoch_losses)}")

        if epoch % CONFIG["plot_every_n_epochs"] == 0:
            save_sample_gif(
                model,
                device,
                ns,
                epoch,
                num_nodes=10,
            )
