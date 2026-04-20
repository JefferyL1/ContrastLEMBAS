"""
Module for visualizing training and validation losses from ContrastLEMBAS runs.

Workflow:
1. Load training and validation JSONL log files from a given output directory.
2. Plot each loss type (total, drug-protein, drug-drug, protein-protein) over
   epochs, using solid lines for training and dashed lines for validation, with
   distinct colors per loss type.
3. Optionally save the resulting figure to a specified output path.
"""

import json
import matplotlib.pyplot as plt


def load_jsonl(filepath):
    """
    Load a JSONL file and return its contents as a list of dicts.

    Arguments:
        filepath (str): Path to the JSONL file to load.

    Returns:
        list[dict]: A list where each element is a parsed JSON record
                    (one per line in the file).
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_losses(train_log_path, val_log_path, output_path=None):
    """
    Plot training and validation losses over epochs.

    Each loss type is assigned a distinct color. Training losses are drawn
    with solid lines and validation losses with dashed lines. A legend
    distinguishes all series. The figure is displayed and optionally saved.

    Arguments:
        train_log_path (str): Path to the training JSONL log file.
        val_log_path (str): Path to the validation JSONL log file.
        output_path (str, optional): File path at which to save the figure.
                                     If None, the figure is only displayed.

    Returns:
        None
    """
    train_records = load_jsonl(train_log_path)
    val_records = load_jsonl(val_log_path)

    loss_keys = [
        ("total_loss", "Total"),
        ("drug-protein_loss", "Drug-Protein"),
        ("drug-drug_loss", "Drug-Drug"),
        ("protein-protein_loss", "Protein-Protein"),
    ]
    colors = {
        "total_loss": "#e41a1c",
        "drug-protein_loss": "#377eb8",
        "drug-drug_loss": "#4daf4a",
        "protein-protein_loss": "#ff7f00",
    }

    train_epochs = [r["epoch"] for r in train_records]
    val_epochs = [r["epoch"] for r in val_records]

    fig, ax = plt.subplots(figsize=(9, 6))

    for key, label in loss_keys:
        color = colors[key]

        train_values = [r[key] for r in train_records if key in r]
        if train_values:
            ax.plot(
                train_epochs[: len(train_values)],
                train_values,
                color=color,
                linestyle="-",
                linewidth=1.8,
                label=f"{label} (train)",
            )

        val_values = [r[key] for r in val_records if key in r]
        if val_values:
            ax.plot(
                val_epochs[: len(val_values)],
                val_values,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{label} (val)",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Losses over Epochs")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    plt.show()
