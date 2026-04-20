import argparse
import pickle
import torch
import h5py
import sys

sys.path.append('/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/code')
from data_processing.data_object_v2 import MultiTaskContrastiveDataset
from model.contrast_model import FFNNContrastiveModel
from model.contrast_train import train_contrastive, train_contrastive_limited, base_parameters

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="Path to output directory")
parser.add_argument("--drug_embeddings", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/drug_data/smiles_to_ecfp4.pkl',
    help="Path to smiles -> embedding pickle file")
parser.add_argument("--train_interactions", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/broad_repurposing_data/filt_train_interaction_data.pkl',
    help="Path to training interaction data pickle")
parser.add_argument("--val_interactions", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/broad_repurposing_data/filt_val_interaction_data.pkl',
    help="Path to validation interaction data pickle")
parser.add_argument("--protein_similarity", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/protein_data/protein_similarity.pt',
    help="Path to protein similarity .pt file with similarity and uniprot_to_index")
parser.add_argument("--drug_similarity", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/drug_data/drug_similarity.pt',
    help="Path to drug similarity .pt file with similarity and smiles_to_index")
parser.add_argument("--protein_embeddings", type=str,
    default='/orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/data/protein_data/uniprot_to_embedding.pkl',
    help="Path to uniprot->embedding pickle file")
parser.add_argument("--limited", action="store_true", help="Enable limited loss - only contrastive loss")
args = parser.parse_args()

# Loading all data in
with open(args.train_interactions, 'rb') as f:
    training_interaction_data = pickle.load(f)

with open(args.val_interactions, 'rb') as f:
    valid_interaction_data = pickle.load(f)

with open(args.drug_embeddings, 'rb') as f:
    smiles_to_embedding = pickle.load(f)

data = torch.load(args.protein_similarity, map_location="cpu")
protein_similarity_adj = data["similarity"]
uniprot_to_index = data["uniprot_to_index"]

data = torch.load(args.drug_similarity, map_location="cpu")
drug_similarity_adj = data["similarity"]
smiles_to_index = data["smiles_to_index"]

with open(args.protein_embeddings, 'rb') as f:
    uniprot_to_embedding = pickle.load(f)

# Build dataset
print('Making dataset')
full_data = MultiTaskContrastiveDataset(
    training_known_target_interactions=training_interaction_data,
    val_known_target_interactions=valid_interaction_data,
    smiles_to_embedding=smiles_to_embedding,
    smiles_to_index=smiles_to_index,
    drug_similarity_adj=drug_similarity_adj,
    uniprot_to_embedding=uniprot_to_embedding,
    uniprot_to_index=uniprot_to_index,
    protein_similarity_adj=protein_similarity_adj,
)
print('Completed making dataset')

# Initialize model
print('Initializing model')

model = FFNNContrastiveModel(
    drug_input_dim=768,
    protein_input_dim=1024,
    hidden_dim=1024,
    output_dim=1024,
)
print('Completed initializing model')

# Train
print('Training model')

if args.limited:
    model = train_contrastive_limited(model, full_data, output_directory=args.output_dir, params=base_parameters)
else:
    model = train_contrastive(model, full_data, output_directory=args.output_dir, params=base_parameters)