# Federated Seismic Denoising (U-Net + FLWR)

This repository runs a **single-model federated learning baseline** for 2D seismic denoising using **Flower (FLWR)** with 3 clients.

Implemented FL strategies:
- FedAvg
- FedAdagrad
- FedAdam
- Secure FedAvg (simulated: clipped + noised aggregated parameters)

## Data assumptions

Expected files in repo root:

- `a_3.mat`
  - `d0`: noisy
  - `dh`: clean
- `a_2.mat`
  - clean matrix (`dh` if present, otherwise first numeric matrix)
  - synthetic noisy version is generated in code
- `a_4.mat`
  - high-noise matrix (`d0` if present)
  - `dh` clean if available, else fallback proxy clean is used

## Normalization (as requested)

Normalization is computed from **training inputs only**:

```python
mean = X_train.mean()
std = X_train.std() + 1e-8
```

Then each client train/test input is normalized with this same train-only mean/std.

## Install

```bash
pip install numpy scipy matplotlib torch flwr
```

> Note: `flwr.simulation.start_simulation` may require additional runtime dependencies depending on your FLWR version/environment.

## Run

```bash
python federated_unet_seismic.py \
  --data_dir . \
  --out_dir outputs \
  --rounds 8 \
  --local_epochs 2
```

## Outputs

For each strategy under `outputs/<strategy>/`:

- `metrics.txt` (MSE + PSNR)
- `loss_curve.png`
- `comparison.png` (Noisy / Clean / Denoised)

Global summary:
- `outputs/summary.csv`
