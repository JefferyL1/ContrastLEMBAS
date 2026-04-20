"""
Module for generating embeddings from a trained contrastive model.

Workflow:
    1. Load a saved FFNNContrastiveModel checkpoint from disk.
    2. Pass drug and protein input tensors through their respective encoders
       in batches.
    3. L2-normalize the resulting embeddings so they live on the unit hypersphere.
    4. Return two dicts mapping each drug/protein name to its normalized embedding.
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from contrast_model import FFNNContrastiveModel


def get_embeddings(
    checkpoint_path: str,
    drug_input_dim: int,
    protein_input_dim: int,
    hidden_dim: int,
    output_dim: int,
    drug_inputs: dict,
    protein_inputs: dict,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple:
    """
    Load a trained contrastive model and encode drug and protein inputs into a
    shared, L2-normalized embedding space.

    Args:
        checkpoint_path:   Path to a saved model state dict (.pt file) produced
                           by contrast_train.py (e.g. model_epoch_100.pt).
        drug_input_dim:    Dimensionality of the raw drug input vectors.
        protein_input_dim: Dimensionality of the raw protein input vectors.
        hidden_dim:        Hidden layer width used when the model was trained.
        output_dim:        Embedding dimensionality used when the model was trained.
        drug_inputs:       Dict mapping drug name/SMILES -> raw input tensor
                           (shape [drug_input_dim]).
        protein_inputs:    Dict mapping UniProt accession -> raw input tensor
                           (shape [protein_input_dim]).
        device:            Torch device string, e.g. "cpu" or "cuda".
                           Defaults to "cpu".
        batch_size:        Number of inputs to encode in a single forward pass.
                           Defaults to 512.

    Returns:
        A 2-tuple (drug_embeddings, protein_embeddings) where:
          - drug_embeddings: dict mapping drug name -> L2-normalized embedding
            tensor (shape [output_dim], on CPU).
          - protein_embeddings: dict mapping UniProt accession -> L2-normalized
            embedding tensor (shape [output_dim], on CPU).
    """
    dev = torch.device(device)

    model = FFNNContrastiveModel(drug_input_dim, protein_input_dim, hidden_dim, output_dim)
    state_dict = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    def _encode(encoder, inputs_dict: dict) -> dict:
        """
        Run a single encoder sub-network over all entries in inputs_dict and
        return L2-normalized embeddings.

        Args:
            encoder:     The encoder sub-network (model.drug_encoder or
                         model.protein_encoder).
            inputs_dict: Dict mapping name -> raw input tensor.

        Returns:
            Dict mapping name -> L2-normalized embedding tensor (on CPU).
        """
        if not inputs_dict:
            return {}

        names = list(inputs_dict.keys())
        tensors = torch.stack([inputs_dict[n] for n in names])
        loader = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)

        chunks = []
        with torch.no_grad():
            for (batch,) in loader:
                emb = encoder(batch.to(dev))
                chunks.append(F.normalize(emb, p=2, dim=-1).cpu())

        all_emb = torch.cat(chunks, dim=0)
        return {name: all_emb[i] for i, name in enumerate(names)}

    drug_embeddings = _encode(model.drug_encoder, drug_inputs)
    protein_embeddings = _encode(model.protein_encoder, protein_inputs)

    return drug_embeddings, protein_embeddings
