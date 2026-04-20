"""
ChemBERTa-100M-MLM Embedding Generator
Generates molecular embeddings from SMILES strings.

Usage:
    # From a CSV file:
    python chemberta_embeddings.py --input molecules.csv --smiles_col smiles --output embeddings.pkl

    # With a label column (saved alongside embeddings):
    python chemberta_embeddings.py --input molecules.csv --smiles_col smiles --label_col activity --output embeddings.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from tqdm import tqdm


MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"


class SMILESDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles = smiles_list

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


def collate_fn(tokenizer, max_length=512):
    def _collate(batch):
        return tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    return _collate


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(device):
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    print("Model loaded.")
    return tokenizer, model


def generate_embeddings(smiles_list, tokenizer, model, device, batch_size=64, max_length=512):
    """
    Generate CLS token embeddings for a list of SMILES strings.
    Returns a dict mapping SMILES string -> embedding tensor (cpu, shape: [hidden_size]).
    """
    dataset = SMILESDataset(smiles_list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, max_length),
        num_workers=0,
    )

    embeddings_dict = {}
    smiles_iter = iter(smiles_list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            batch_smiles = [next(smiles_iter) for _ in range(len(batch["input_ids"]))]

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.roberta(**batch)

            # CLS token embedding (position 0) — from base roberta, bypassing MLM head
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            for smi, emb in zip(batch_smiles, cls_embeddings):
                embeddings_dict[smi] = emb.cpu()

    return embeddings_dict


def validate_smiles(smiles_list):
    """
    Optionally filter invalid SMILES using RDKit if available.
    Returns (valid_smiles, valid_indices).
    """
    try:
        from rdkit import Chem
        valid_smiles, valid_indices = [], []
        invalid_count = 0
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_indices.append(i)
            else:
                invalid_count += 1
                print(f"  Invalid SMILES at index {i}: {smi}")
        if invalid_count > 0:
            print(f"Filtered {invalid_count} invalid SMILES.")
        return valid_smiles, valid_indices
    except ImportError:
        print("RDKit not available — skipping SMILES validation.")
        return smiles_list, list(range(len(smiles_list)))


def main():
    parser = argparse.ArgumentParser(description="Generate ChemBERTa-100M-MLM embeddings from SMILES.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Column name containing SMILES.")
    parser.add_argument("--label_col", type=str, default=None, help="Optional column name for labels.")
    parser.add_argument("--output", type=str, default="embeddings.pkl", help="Output path for embeddings pickle (.pkl).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for SMILES.")
    parser.add_argument("--no_validate", action="store_true", help="Skip RDKit SMILES validation.")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_col].tolist()
    print(f"  {len(smiles_list)} molecules loaded.")

    # Validate SMILES
    if not args.no_validate:
        smiles_list, valid_indices = validate_smiles(smiles_list)
        df = df.iloc[valid_indices].reset_index(drop=True)

    # Load model
    device = get_device()
    tokenizer, model = load_model(device)

    # Generate embeddings -> {smiles: tensor}
    embeddings_dict = generate_embeddings(
        smiles_list, tokenizer, model, device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Generated embeddings for {len(embeddings_dict)} molecules.")

    # Save embeddings dict as pickle
    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
    print(f"Embeddings dict saved to {output_path}")

    # Save labels if provided
    if args.label_col and args.label_col in df.columns:
        label_path = output_path.with_name(output_path.stem + "_labels.npy")
        labels = df[args.label_col].to_numpy()
        np.save(label_path, labels)
        print(f"Labels saved to {label_path}")

    # Save metadata (SMILES + index mapping)
    meta_path = output_path.with_name(output_path.stem + "_meta.csv")
    df[[args.smiles_col] + ([args.label_col] if args.label_col else [])].to_csv(meta_path, index=False)
    print(f"Metadata saved to {meta_path}")

    print("\nDone.")

    # Usage hint
    print("\nTo load embeddings:")
    print("  import pickle, torch")
    print(f"  with open('{output_path}', 'rb') as f:")
    print("      embeddings = pickle.load(f)  # dict[smiles_str -> torch.Tensor]")


if __name__ == "__main__":
    main()