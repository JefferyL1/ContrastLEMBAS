"""Top-k analysis utilities for the ContrastLEMBAS drug-protein interaction model.

This module provides functions to generate projected, L2-normalized embeddings
from a trained FFNNContrastiveModel and to query those embeddings for the
top-k most similar proteins for one or more drugs, or to retrieve the rank of
specific proteins in the full ranked list for a given drug.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EmbeddingDict = Dict[str, torch.Tensor]  # str key -> 1-D normalized embedding


# ---------------------------------------------------------------------------
# 1. generate_embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[EmbeddingDict, EmbeddingDict]:
    """Generate L2-normalized projected embeddings for all drugs and proteins.

    Iterates over every unique drug (keyed by SMILES) and every unique protein
    (keyed by UniProt accession) present in the dataset, runs them through the
    model in batches without gradient tracking, L2-normalizes the outputs, and
    returns CPU-resident embedding dicts.

    The model is set to eval mode inside this function and restored to its
    original training/eval state afterward.

    Args:
        model: A trained FFNNContrastiveModel (or any nn.Module with the same
            forward signature: ``(drug_input, protein_input) ->
            (drug_embedding, protein_embedding)``).
        dataset: A MultiTaskContrastiveDataset instance.  Must expose
            ``smiles_to_embedding`` and ``uniprot_to_embedding`` dicts.
        device: The torch device on which to run inference.
        batch_size: Number of items to process per forward pass.  Defaults to
            512.

    Returns:
        A 2-tuple ``(drug_embeddings, protein_embeddings)`` where:

        - ``drug_embeddings``: dict mapping drug SMILES -> normalized
          projected embedding tensor (shape ``[output_dim]``, on CPU).
        - ``protein_embeddings``: dict mapping UniProt accession -> normalized
          projected embedding tensor (shape ``[output_dim]``, on CPU).

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size}."
        )

    was_training = model.training
    model.eval()
    model.to(device)

    # ------------------------------------------------------------------
    # Helper: encode one modality in batches
    # ------------------------------------------------------------------
    def _encode_items(
        key_to_raw: dict,
        encoder: torch.nn.Module,
    ) -> EmbeddingDict:
        """Encode a dict of raw embeddings through *encoder* in batches.

        Args:
            key_to_raw: Mapping from identifier string -> raw embedding tensor.
            encoder: The sub-network (drug_encoder or protein_encoder) to
                apply.

        Returns:
            Mapping from identifier string -> L2-normalized projected
            embedding (1-D, on CPU).
        """
        keys = list(key_to_raw.keys())
        result: EmbeddingDict = {}

        with torch.no_grad():
            for start in range(0, len(keys), batch_size):
                batch_keys = keys[start : start + batch_size]
                # Stack raw embeddings into a 2-D tensor [B, input_dim]
                raw_batch = torch.stack(
                    [key_to_raw[k] for k in batch_keys], dim=0
                ).to(device)

                projected_batch = encoder(raw_batch)  # [B, output_dim]
                normalized_batch = F.normalize(projected_batch, p=2, dim=-1)  # [B, output_dim]
                normalized_batch = normalized_batch.cpu()

                for i, key in enumerate(batch_keys):
                    result[key] = normalized_batch[i]

        return result

    drug_embeddings = _encode_items(
        dataset.smiles_to_embedding, model.drug_encoder
    )
    protein_embeddings = _encode_items(
        dataset.uniprot_to_embedding, model.protein_encoder
    )

    # Restore original training state
    if was_training:
        model.train()

    return drug_embeddings, protein_embeddings


# ---------------------------------------------------------------------------
# 2. get_top_k_proteins
# ---------------------------------------------------------------------------

def get_top_k_proteins(
    drug_smiles: str,
    drug_embeddings: EmbeddingDict,
    protein_embeddings: EmbeddingDict,
    k: int = 10,
) -> List[Dict]:
    """Return the top-k most similar proteins for a single drug.

    Similarity is measured as cosine similarity between the (already
    L2-normalized) drug and protein embeddings produced by
    :func:`generate_embeddings`.  Because the embeddings are L2-normalized,
    cosine similarity equals the dot product.

    Args:
        drug_smiles: The SMILES string identifying the query drug.
        drug_embeddings: Dict mapping drug SMILES -> normalized projected
            embedding (as returned by :func:`generate_embeddings`).
        protein_embeddings: Dict mapping UniProt accession -> normalized
            projected embedding (as returned by :func:`generate_embeddings`).
        k: Number of top proteins to return.  Must be a positive integer.

    Returns:
        A list of ``k`` dicts (or fewer if there are fewer proteins than
        ``k``), sorted by descending cosine similarity.  Each dict contains:

        - ``rank`` (int): 1-indexed rank (1 = most similar).
        - ``uniprot`` (str): UniProt accession of the protein.
        - ``similarity`` (float): Cosine similarity score.

    Raises:
        ValueError: If ``drug_smiles`` is not found in ``drug_embeddings``.
        ValueError: If ``k`` is not a positive integer.
        ValueError: If ``protein_embeddings`` is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if drug_smiles not in drug_embeddings:
        raise ValueError(
            f"Drug SMILES '{drug_smiles}' not found in drug_embeddings. "
            "Ensure generate_embeddings was called with the correct dataset."
        )
    if not protein_embeddings:
        raise ValueError("protein_embeddings is empty; cannot rank proteins.")

    drug_emb = drug_embeddings[drug_smiles]  # [output_dim]

    uniprot_keys = list(protein_embeddings.keys())
    # Stack all protein embeddings: [n_proteins, output_dim]
    protein_matrix = torch.stack(
        [protein_embeddings[u] for u in uniprot_keys], dim=0
    )

    # Cosine similarity = dot product (embeddings already L2-normalized)
    similarity_scores = protein_matrix @ drug_emb  # [n_proteins]

    effective_k = min(k, len(uniprot_keys))
    top_values, top_indices = torch.topk(similarity_scores, effective_k)

    results = []
    for rank_idx, (tensor_idx, sim_val) in enumerate(
        zip(top_indices.tolist(), top_values.tolist()), start=1
    ):
        results.append(
            {
                "rank": rank_idx,
                "uniprot": uniprot_keys[tensor_idx],
                "similarity": float(sim_val),
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3. get_top_k_proteins_batch
# ---------------------------------------------------------------------------

def get_top_k_proteins_batch(
    drug_smiles_list: List[str],
    drug_embeddings: EmbeddingDict,
    protein_embeddings: EmbeddingDict,
    k: int = 10,
) -> Dict[str, List[Dict]]:
    """Return the top-k most similar proteins for each drug in a list.

    Uses vectorized matrix multiplication to compute the full
    ``[n_drugs x n_proteins]`` cosine-similarity matrix in a single pass
    (embeddings are already L2-normalized, so cosine similarity == dot
    product), then applies ``torch.topk`` row-wise.

    Args:
        drug_smiles_list: List of drug SMILES strings to query.
        drug_embeddings: Dict mapping drug SMILES -> normalized projected
            embedding (as returned by :func:`generate_embeddings`).
        protein_embeddings: Dict mapping UniProt accession -> normalized
            projected embedding (as returned by :func:`generate_embeddings`).
        k: Number of top proteins to return per drug.  Must be a positive
            integer.

    Returns:
        A dict mapping each drug SMILES -> list of ``k`` result dicts (or
        fewer if there are fewer proteins than ``k``).  Each result dict
        contains:

        - ``rank`` (int): 1-indexed rank (1 = most similar).
        - ``uniprot`` (str): UniProt accession.
        - ``similarity`` (float): Cosine similarity score.

    Raises:
        ValueError: If ``k`` is not a positive integer.
        ValueError: If any drug SMILES in ``drug_smiles_list`` is not found in
            ``drug_embeddings``.
        ValueError: If ``protein_embeddings`` is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if not protein_embeddings:
        raise ValueError("protein_embeddings is empty; cannot rank proteins.")

    missing_drugs = [s for s in drug_smiles_list if s not in drug_embeddings]
    if missing_drugs:
        raise ValueError(
            f"The following drug SMILES were not found in drug_embeddings: "
            f"{missing_drugs}"
        )

    if not drug_smiles_list:
        return {}

    uniprot_keys = list(protein_embeddings.keys())
    # [n_proteins, output_dim]
    protein_matrix = torch.stack(
        [protein_embeddings[u] for u in uniprot_keys], dim=0
    )

    # [n_drugs, output_dim]
    drug_matrix = torch.stack(
        [drug_embeddings[s] for s in drug_smiles_list], dim=0
    )

    # Full similarity matrix: [n_drugs, n_proteins]
    sim_matrix = drug_matrix @ protein_matrix.T

    effective_k = min(k, len(uniprot_keys))
    # top_values, top_indices: each [n_drugs, effective_k]
    top_values, top_indices = torch.topk(sim_matrix, effective_k, dim=1)

    batch_results: Dict[str, List[Dict]] = {}
    for drug_row_idx, drug_smiles in enumerate(drug_smiles_list):
        drug_top_results = []
        for rank_idx in range(effective_k):
            protein_idx = top_indices[drug_row_idx, rank_idx].item()
            sim_val = top_values[drug_row_idx, rank_idx].item()
            drug_top_results.append(
                {
                    "rank": rank_idx + 1,
                    "uniprot": uniprot_keys[protein_idx],
                    "similarity": float(sim_val),
                }
            )
        batch_results[drug_smiles] = drug_top_results

    return batch_results


# ---------------------------------------------------------------------------
# 4. get_protein_ranks
# ---------------------------------------------------------------------------

def get_protein_ranks(
    drug_smiles: str,
    protein_uniprots: List[str],
    drug_embeddings: EmbeddingDict,
    protein_embeddings: EmbeddingDict,
) -> List[Dict]:
    """Return the rank of each specified protein in the full ranking for a drug.

    Computes cosine similarity between the drug and every protein in
    ``protein_embeddings``, then determines the rank of each queried protein
    in the resulting sorted order.

    Ties are resolved so that tied entries share the *lowest* rank among them
    (i.e., ``rank = number_of_proteins_with_strictly_higher_similarity + 1``).

    Args:
        drug_smiles: The SMILES string identifying the query drug.
        protein_uniprots: List of UniProt accessions whose ranks are to be
            retrieved.
        drug_embeddings: Dict mapping drug SMILES -> normalized projected
            embedding (as returned by :func:`generate_embeddings`).
        protein_embeddings: Dict mapping UniProt accession -> normalized
            projected embedding (as returned by :func:`generate_embeddings`).

    Returns:
        A list of dicts in the same order as ``protein_uniprots``.  Each dict
        contains:

        - ``uniprot`` (str): UniProt accession.
        - ``rank`` (int | None): 1-indexed rank (1 = most similar).  ``None``
          if the UniProt accession is not found in ``protein_embeddings``.
        - ``similarity`` (float | None): Cosine similarity score.  ``None``
          if the UniProt accession is not found in ``protein_embeddings``.
        - ``n_proteins`` (int): Total number of proteins in the library (i.e.,
          ``len(protein_embeddings)``), allowing the caller to compute a
          percentile.

    Raises:
        ValueError: If ``drug_smiles`` is not found in ``drug_embeddings``.
        ValueError: If ``protein_embeddings`` is empty.
    """
    if not protein_uniprots:
        return []

    if drug_smiles not in drug_embeddings:
        raise ValueError(
            f"Drug SMILES '{drug_smiles}' not found in drug_embeddings. "
            "Ensure generate_embeddings was called with the correct dataset."
        )
    if not protein_embeddings:
        raise ValueError("protein_embeddings is empty; cannot compute ranks.")

    drug_emb = drug_embeddings[drug_smiles]  # [output_dim]
    n_proteins = len(protein_embeddings)

    uniprot_keys = list(protein_embeddings.keys())
    # [n_proteins, output_dim]
    protein_matrix = torch.stack(
        [protein_embeddings[u] for u in uniprot_keys], dim=0
    )
    # Build a lookup: uniprot -> position in uniprot_keys list
    uniprot_to_pos = {u: i for i, u in enumerate(uniprot_keys)}

    # Full similarity vector for the drug against all proteins: [n_proteins]
    all_scores = protein_matrix @ drug_emb

    results = []
    for uniprot in protein_uniprots:
        if uniprot not in uniprot_to_pos:
            results.append(
                {
                    "uniprot": uniprot,
                    "rank": None,
                    "similarity": None,
                    "n_proteins": n_proteins,
                }
            )
            continue

        pos = uniprot_to_pos[uniprot]
        query_score = all_scores[pos]  # scalar tensor

        # Rank = number of proteins with strictly higher similarity + 1
        rank = int((all_scores > query_score).sum().item()) + 1

        results.append(
            {
                "uniprot": uniprot,
                "rank": rank,
                "similarity": float(query_score.item()),
                "n_proteins": n_proteins,
            }
        )

    return results
