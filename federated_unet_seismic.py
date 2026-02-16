#!/usr/bin/env python3
"""Federated seismic denoising with U-Net using Flower (FLWR).

Implements 3 clients and 4 FL server strategies:
- FedAvg
- FedAdagrad
- FedAdam
- Secure FedAvg (simulated; clipped + noised global parameters)

Data expectations:
- a_3.mat: noisy=d0, clean=dh
- a_2.mat: clean only (synthetic noisy generated)
- a_4.mat: high-noisy input (d0 if present)

Normalization rule (training inputs only):
    mean = X_train.mean()
    std = X_train.std() + 1e-8
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import flwr as fl
    from flwr.common import NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
except ImportError as exc:
    raise ImportError(
        "flwr is required for this script. Install with: pip install flwr"
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_numeric_array(mat_dict: Dict[str, np.ndarray]) -> np.ndarray:
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 2 and np.issubdtype(value.dtype, np.number):
            return value.astype(np.float32)
    raise ValueError("No numeric matrix found in MAT file.")


def ensure_2d(img: np.ndarray) -> np.ndarray:
    out = np.squeeze(img)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {out.shape}")
    return out.astype(np.float32)


def load_client_pairs(data_dir: Path, synthetic_noise_sigma: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    m3 = sio.loadmat(data_dir / "a_3.mat")
    if "d0" not in m3 or "dh" not in m3:
        raise KeyError("a_3.mat must contain d0 (noisy) and dh (clean)")
    pairs["0"] = (ensure_2d(m3["d0"]), ensure_2d(m3["dh"]))

    m2 = sio.loadmat(data_dir / "a_2.mat")
    clean2 = ensure_2d(m2["dh"] if "dh" in m2 else first_numeric_array(m2))
    noisy2 = clean2 + np.random.normal(0.0, synthetic_noise_sigma, size=clean2.shape).astype(np.float32)
    pairs["1"] = (noisy2, clean2)

    m4 = sio.loadmat(data_dir / "a_4.mat")
    noisy4 = ensure_2d(m4["d0"] if "d0" in m4 else first_numeric_array(m4))
    if "dh" in m4:
        clean4 = ensure_2d(m4["dh"])
    else:
        ref_clean = pairs["0"][1]
        h, w = noisy4.shape
        clean4 = ref_clean[:h, :w]
    pairs["2"] = (noisy4, clean4)

    return pairs


def extract_patches(img: np.ndarray, patch_size: int = 64, stride: int = 32) -> np.ndarray:
    h, w = img.shape
    patches = []
    for i in range(0, max(1, h - patch_size + 1), stride):
        for j in range(0, max(1, w - patch_size + 1), stride):
            p = img[i : i + patch_size, j : j + patch_size]
            if p.shape == (patch_size, patch_size):
                patches.append(p)
    if not patches:
        raise ValueError(f"No patches extracted for image shape={img.shape}")
    return np.stack(patches, axis=0)[:, None, :, :]


def split_train_test(x: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int((1 - test_ratio) * len(x))
    tr, te = idx[:split], idx[split:]
    return x[tr], x[te], y[tr], y[te]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def get_parameters(model: nn.Module) -> NDArrays:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, params: NDArrays) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


def psnr_from_mse(mse: float, max_pixel: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_pixel**2) / mse)


def train_local(model: nn.Module, loader: DataLoader, epochs: int, device: torch.device, lr: float) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    losses: List[float] = []
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device):
    model.eval()
    xt = torch.from_numpy(x).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    pred = model(xt)
    mse = torch.mean((pred - yt) ** 2).item()
    psnr = psnr_from_mse(mse)
    return mse, psnr, pred.detach().cpu().numpy()


def build_clients_data(
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    patch_size: int,
    stride: int,
    batch_size: int,
    test_ratio: float,
    seed: int,
):
    clients = {}
    all_train = []

    for idx, (cid, (noisy2d, clean2d)) in enumerate(pairs.items()):
        x = extract_patches(noisy2d, patch_size, stride)
        y = extract_patches(clean2d, patch_size, stride)
        x_tr, x_te, y_tr, y_te = split_train_test(x, y, test_ratio, seed + idx)
        clients[cid] = {
            "x_train": x_tr,
            "y_train": y_tr,
            "x_test": x_te,
            "y_test": y_te,
        }
        all_train.append(x_tr)

    # Compute normalization from TRAINING data ONLY
    X_train = np.concatenate(all_train, axis=0)
    mean = X_train.mean()
    std = X_train.std() + 1e-8

    for cid in clients:
        clients[cid]["x_train"] = (clients[cid]["x_train"] - mean) / std
        clients[cid]["x_test"] = (clients[cid]["x_test"] - mean) / std

        train_ds = TensorDataset(
            torch.from_numpy(clients[cid]["x_train"]).float(),
            torch.from_numpy(clients[cid]["y_train"]).float(),
        )
        clients[cid]["train_loader"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return clients, float(mean), float(std)


class SeismicClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data: dict, base_channels: int, local_epochs: int, local_lr: float, device: torch.device):
        self.cid = cid
        self.data = data
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.device = device
        self.model = SmallUNet(base=base_channels).to(device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = train_local(self.model, self.data["train_loader"], self.local_epochs, self.device, self.local_lr)
        return get_parameters(self.model), len(self.data["x_train"]), {"loss": loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        mse, psnr, _ = evaluate_model(self.model, self.data["x_test"], self.data["y_test"], self.device)
        return float(mse), len(self.data["x_test"]), {"mse": float(mse), "psnr": float(psnr)}


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_parameters = aggregated
        return aggregated, metrics


class TrackingFedAdagrad(fl.server.strategy.FedAdagrad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_parameters = aggregated
        return aggregated, metrics


class TrackingFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_parameters = aggregated
        return aggregated, metrics


class SecureFedAvg(TrackingFedAvg):
    """Simulated secure FedAvg: clip and noise aggregated parameters."""

    def __init__(self, clip_value: float = 3.0, noise_std: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_value = clip_value
        self.noise_std = noise_std

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated, metrics

        nds = parameters_to_ndarrays(aggregated)
        secured = []
        for arr in nds:
            clipped = np.clip(arr, -self.clip_value, self.clip_value)
            noisy = clipped + np.random.normal(0.0, self.noise_std, size=arr.shape).astype(arr.dtype)
            secured.append(noisy)

        secured_params = ndarrays_to_parameters(secured)
        self.latest_parameters = secured_params
        return secured_params, metrics


def fit_metrics_agg(metrics):
    if not metrics:
        return {}
    losses = [m[1].get("loss", 0.0) for m in metrics]
    return {"loss": float(np.mean(losses))}


def evaluate_metrics_agg(metrics):
    if not metrics:
        return {}
    mses = [m[1].get("mse", 0.0) for m in metrics]
    psnrs = [m[1].get("psnr", 0.0) for m in metrics]
    return {"mse": float(np.mean(mses)), "psnr": float(np.mean(psnrs))}


def make_strategy(name: str, num_clients: int, server_lr: float, clip_value: float, noise_std: float):
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_metrics_agg,
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg,
    )
    if name == "fedavg":
        return TrackingFedAvg(**common)
    if name == "fedadagrad":
        return TrackingFedAdagrad(eta=server_lr, eta_l=1.0, tau=1e-9, **common)
    if name == "fedadam":
        return TrackingFedAdam(eta=server_lr, eta_l=1.0, beta_1=0.9, beta_2=0.99, tau=1e-9, **common)
    if name == "secure_fedavg":
        return SecureFedAvg(clip_value=clip_value, noise_std=noise_std, **common)
    raise ValueError(f"Unknown strategy: {name}")


def save_outputs(
    out_dir: Path,
    strategy_name: str,
    round_losses: List[float],
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    mse: float,
    psnr: float,
):
    strategy_dir = out_dir / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(round_losses) + 1), round_losses, marker="o")
    plt.title(f"Federated Training Loss - {strategy_name} (FLWR)")
    plt.xlabel("Round")
    plt.ylabel("Avg Client Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = strategy_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    idx = 0
    noisy_img = x_test[idx, 0]
    clean_img = y_test[idx, 0]
    denoised_img = pred[idx, 0]

    vmin = min(noisy_img.min(), clean_img.min(), denoised_img.min())
    vmax = max(noisy_img.max(), clean_img.max(), denoised_img.max())

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Seismic Section Denoising - U-Net ({strategy_name}, PSNR: {psnr:.2f} dB)", fontsize=16, fontweight="bold")

    im0 = axs[0].imshow(noisy_img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Noisy Input", fontweight="bold")
    axs[0].set_xlabel("Trace Number")
    axs[0].set_ylabel("Time Sample")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(clean_img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Clean (Ground Truth)", fontweight="bold")
    axs[1].set_xlabel("Trace Number")
    axs[1].set_ylabel("Time Sample")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(denoised_img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    axs[2].set_title("Denoised Output (U-Net)", fontweight="bold")
    axs[2].set_xlabel("Trace Number")
    axs[2].set_ylabel("Time Sample")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    comp_path = strategy_dir / "comparison.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()

    metrics_path = strategy_dir / "metrics.txt"
    metrics_path.write_text(f"strategy={strategy_name}\nmse={mse:.8f}\npsnr={psnr:.4f}\n")

    return str(loss_path), str(comp_path)


def parse_args():
    p = argparse.ArgumentParser(description="FLWR federated U-Net seismic denoising")
    p.add_argument("--data_dir", type=Path, default=Path("."))
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--local_lr", type=float, default=1e-3)
    p.add_argument("--server_lr", type=float, default=0.1)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--clip_value", type=float, default=3.0)
    p.add_argument("--noise_std", type=float, default=1e-4)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["fedavg", "fedadagrad", "fedadam", "secure_fedavg"],
        choices=["fedavg", "fedadagrad", "fedadam", "secure_fedavg"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = load_client_pairs(args.data_dir)
    clients, mean, std = build_clients_data(
        pairs,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Normalization (training-only): mean={mean:.6f}, std={std:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    x_global_test = np.concatenate([clients[cid]["x_test"] for cid in sorted(clients.keys())], axis=0)
    y_global_test = np.concatenate([clients[cid]["y_test"] for cid in sorted(clients.keys())], axis=0)

    num_clients = len(clients)

    for strategy_name in args.strategies:
        print(f"\n=== Running {strategy_name} with FLWR ===")

        strategy = make_strategy(
            strategy_name,
            num_clients=num_clients,
            server_lr=args.server_lr,
            clip_value=args.clip_value,
            noise_std=args.noise_std,
        )

        def client_fn(cid: str):
            return SeismicClient(
                cid=cid,
                data=clients[cid],
                base_channels=args.base_channels,
                local_epochs=args.local_epochs,
                local_lr=args.local_lr,
                device=device,
            )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1},
        )

        if strategy.latest_parameters is None:
            raise RuntimeError(f"No aggregated parameters were produced for {strategy_name}.")

        final_model = SmallUNet(base=args.base_channels).to(device)
        set_parameters(final_model, parameters_to_ndarrays(strategy.latest_parameters))

        mse, psnr, pred = evaluate_model(final_model, x_global_test, y_global_test, device)

        round_losses = [loss for _, loss in history.losses_distributed]
        if not round_losses:
            round_losses = [loss for _, loss in history.losses_centralized]

        loss_path, comp_path = save_outputs(
            args.out_dir,
            strategy_name,
            round_losses,
            x_global_test,
            y_global_test,
            pred,
            mse,
            psnr,
        )

        summary.append(
            {
                "strategy": strategy_name,
                "mse": mse,
                "psnr": psnr,
                "loss_curve": loss_path,
                "comparison": comp_path,
            }
        )

        print(f"{strategy_name}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    summary_path = args.out_dir / "summary.csv"
    lines = ["strategy,mse,psnr,loss_curve,comparison"]
    for row in summary:
        lines.append(
            f"{row['strategy']},{row['mse']:.8f},{row['psnr']:.4f},{row['loss_curve']},{row['comparison']}"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
