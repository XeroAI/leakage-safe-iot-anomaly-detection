import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import get_dataloaders
from model import LST_TFDN


def get_saliency(model, data):
    """Absolute gradient of the logit w.r.t. input [1, sensors, time]."""
    model.eval()
    data = data.clone().detach().requires_grad_(True)
    output = model(data)
    model.zero_grad()
    output.backward()
    return data.grad.squeeze(0).abs().numpy()


def find_example_windows(test_loader):
    anomaly_data, normal_data = None, None
    for data, target in test_loader:
        if target.item() == 1.0 and anomaly_data is None:
            anomaly_data = data
        elif target.item() == 0.0 and normal_data is None:
            normal_data = data
        if anomaly_data is not None and normal_data is not None:
            break
    return anomaly_data, normal_data


def generate_xai_figure(
    dataset_name="MSL",
    checkpoint_path="lst_tfdn_MSL_best.pth",
    output_path="xai_3panel.png",
    window_size=100,
    stride=10,
    d_model=32,
    top_k=10,
):
    device = torch.device("cpu")
    print(f"Loading data and model for 3-panel XAI ({dataset_name})...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train MSL first: set DATASET_NAME='MSL' in train.py and run python train.py"
        )

    _, _, test_loader = get_dataloaders(
        dataset_name, batch_size=1, window_size=window_size, stride=stride
    )

    sample_data, _ = next(iter(test_loader))
    num_sensors = sample_data.shape[1]

    model = LST_TFDN(
        num_sensors=num_sensors,
        window_size=window_size,
        d_model=d_model,
        heads=4,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    anomaly_data, normal_data = find_example_windows(test_loader)
    if anomaly_data is None or normal_data is None:
        raise RuntimeError("Could not find both an anomaly and a normal window in the test set.")

    anom_saliency = get_saliency(model, anomaly_data)
    norm_saliency = get_saliency(model, normal_data)

    sensor_attr = np.sum(anom_saliency, axis=1)
    top_indices = np.argsort(sensor_attr)[-top_k:]
    top_values = sensor_attr[top_indices]

    vmax = max(anom_saliency.max(), norm_saliency.max(), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(anom_saliency, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[0].set_title("A) Anomalous Window: Sensor Attribution", fontsize=12)
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel(f"Sensors (0-{num_sensors - 1})")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(norm_saliency, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[1].set_title("B) Normal Window: Sensor Attribution", fontsize=12)
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel(f"Sensors (0-{num_sensors - 1})")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].barh(range(top_k), top_values, color="darkred")
    axes[2].set_yticks(range(top_k))
    axes[2].set_yticklabels([f"Sensor {i}" for i in top_indices])
    axes[2].set_title(f"C) Top {top_k} Contributing Sensors (Anomaly)", fontsize=12)
    axes[2].set_xlabel("Summed Gradient Magnitude")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"3-panel XAI figure saved: {output_path}")
    print(f"PDF version saved: {output_path.replace('.png', '.pdf')}")
    print(f"Top sensors: {list(top_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3-panel XAI saliency figure.")
    parser.add_argument("--dataset", default="MSL")
    parser.add_argument("--checkpoint", default="lst_tfdn_MSL_best.pth")
    parser.add_argument("--output", default="xai_3panel.png")
    args = parser.parse_args()

    generate_xai_figure(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
