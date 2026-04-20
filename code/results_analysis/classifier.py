"""
Module for training and evaluating a drug-protein interaction classifier.

Workflow:
    1. Build a labeled dataset of (drug_embedding, protein_embedding, label)
       triples from pre-generated embeddings and a known interaction dict.
       Positive pairs come from known interactions; negatives are randomly sampled.
    2. Train a small MLP classifier that concatenates the drug and protein
       embeddings as input and outputs an interaction probability.
    3. Evaluate accuracy and ROC-AUC on a held-out validation split each epoch.
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score


class PairDataset(Dataset):
    """
    Dataset of (drug_embedding, protein_embedding, label) triples.

    Args:
        pairs: List of (drug_name, protein_name, label) tuples where label is
               1 for a known interaction and 0 for a non-interaction.
        drug_embeddings: Dict mapping drug name -> embedding tensor.
        protein_embeddings: Dict mapping protein name -> embedding tensor.
    """

    def __init__(self, pairs: list, drug_embeddings: dict, protein_embeddings: dict):
        self.pairs = pairs
        self.drug_embeddings = drug_embeddings
        self.protein_embeddings = protein_embeddings

    def __len__(self):
        """Return the total number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return (drug_embedding, protein_embedding, label) for a single pair.

        Returns:
            Tuple of (drug_emb tensor, protein_emb tensor, label float tensor).
        """
        drug, protein, label = self.pairs[idx]
        return (
            self.drug_embeddings[drug],
            self.protein_embeddings[protein],
            torch.tensor(label, dtype=torch.float32),
        )


class InteractionClassifier(nn.Module):
    """
    Two-hidden-layer MLP that predicts drug-protein interaction from concatenated
    embeddings. Outputs a single logit (use BCEWithLogitsLoss during training).

    Args:
        embedding_dim: Dimensionality of each embedding vector. The input to the
                       MLP is the concatenation of drug and protein embeddings,
                       so the first layer receives 2 * embedding_dim features.
        hidden_dim:    Width of the two hidden layers. Defaults to 256.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        """
        Concatenate drug and protein embeddings and compute an interaction logit.

        Args:
            drug_emb:    Tensor of shape [batch, embedding_dim].
            protein_emb: Tensor of shape [batch, embedding_dim].

        Returns:
            Tensor of shape [batch, 1] containing raw logits.
        """
        x = torch.cat([drug_emb, protein_emb], dim=-1)
        return self.net(x)


def prepare_data(
    drug_embeddings: dict,
    protein_embeddings: dict,
    known_interactions: dict,
    neg_ratio: int = 1,
) -> PairDataset:
    """
    Build a PairDataset with positive pairs from known_interactions and randomly
    sampled negative pairs.

    Args:
        drug_embeddings:   Dict mapping drug name -> embedding tensor.
        protein_embeddings: Dict mapping protein name -> embedding tensor.
        known_interactions: Dict mapping drug name -> set of known interacting
                            protein names (e.g. dataset.drug_to_protein_inter).
        neg_ratio:         Number of negative pairs to sample per positive pair.
                           Defaults to 1 (balanced dataset).

    Returns:
        A PairDataset containing all positive pairs and randomly sampled
        negatives, filtered to drugs and proteins present in the embedding dicts.
    """
    all_drugs = list(drug_embeddings.keys())
    all_proteins = list(protein_embeddings.keys())
    protein_set = set(all_proteins)

    positives = []
    for drug, targets in known_interactions.items():
        if drug not in drug_embeddings:
            continue
        for protein in targets:
            if protein in protein_set:
                positives.append((drug, protein, 1))

    negatives = []
    pos_set = {(d, p) for d, p, _ in positives}
    needed = neg_ratio * len(positives)
    attempts = 0
    while len(negatives) < needed and attempts < needed * 20:
        drug = random.choice(all_drugs)
        protein = random.choice(all_proteins)
        if (drug, protein) not in pos_set:
            negatives.append((drug, protein, 0))
        attempts += 1

    return PairDataset(positives + negatives, drug_embeddings, protein_embeddings)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Compute accuracy and ROC-AUC over a DataLoader.

    Args:
        model:  The InteractionClassifier to evaluate.
        loader: DataLoader yielding (drug_emb, protein_emb, label) batches.
        device: Torch device for inference.

    Returns:
        Dict with keys "accuracy" (float) and "auc" (float).
    """
    model.eval()
    all_labels, all_probs, correct, total = [], [], 0, 0
    with torch.no_grad():
        for drug_emb, protein_emb, labels in loader:
            drug_emb, protein_emb, labels = (
                drug_emb.to(device),
                protein_emb.to(device),
                labels.to(device),
            )
            logits = model(drug_emb, protein_emb).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": accuracy, "auc": auc}


def train_classifier(
    drug_embeddings: dict,
    protein_embeddings: dict,
    known_interactions: dict,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    neg_ratio: int = 1,
    device: str = "cpu",
) -> InteractionClassifier:
    """
    Train an InteractionClassifier on drug-protein embedding pairs.

    Builds a dataset from the provided embeddings and interaction labels, splits
    it into train/validation subsets, and runs a standard training loop. Prints
    accuracy and AUC at the end of each epoch.

    Args:
        drug_embeddings:   Dict mapping drug name -> embedding tensor.
        protein_embeddings: Dict mapping protein name -> embedding tensor.
        known_interactions: Dict mapping drug name -> set of known interacting
                            protein names.
        hidden_dim:        Width of the classifier's hidden layers. Defaults to 256.
        epochs:            Number of training epochs. Defaults to 20.
        batch_size:        Batch size for training and evaluation. Defaults to 256.
        lr:                Learning rate for Adam optimizer. Defaults to 1e-3.
        val_fraction:      Fraction of data to reserve for validation. Defaults to 0.2.
        neg_ratio:         Negative pairs sampled per positive pair. Defaults to 1.
        device:            Torch device string ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        The trained InteractionClassifier (in eval mode).
    """
    dev = torch.device(device)

    dataset = prepare_data(drug_embeddings, protein_embeddings, known_interactions, neg_ratio)
    n_val = max(1, int(val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    embedding_dim = next(iter(drug_embeddings.values())).shape[0]
    model = InteractionClassifier(embedding_dim, hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for drug_emb, protein_emb, labels in train_loader:
            drug_emb, protein_emb, labels = (
                drug_emb.to(dev),
                protein_emb.to(dev),
                labels.to(dev),
            )
            optimizer.zero_grad()
            logits = model(drug_emb, protein_emb).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = _evaluate(model, val_loader, dev)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={total_loss / len(train_loader):.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

    model.eval()
    return model
