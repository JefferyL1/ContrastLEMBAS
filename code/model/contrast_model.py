import torch
import torch.nn as nn   
import torch.nn.functional as F
import math

class FFNNContrastiveModel(nn.Module):
    """ Simple feedforward neural network w. batch norm and dropout for contrastive 
    learning of drug and protein embeddings using embeddings derived from
    pre-trained models / other methods. """

    def __init__(self, drug_input_dim, protein_input_dim, hidden_dim, output_dim):
        super(FFNNContrastiveModel, self).__init__()
        
        # 3 layer NN with batch norm, relu, and dropout for drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dim, bias = True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim, bias = True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim, bias = True),
        )
        
        # 5 layer NN with batch norm, relu, and dropout for protein encoder
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_input_dim, hidden_dim, bias = True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim, bias = True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim, bias = True)
        )

        # three parameters for each loss: (drug-protein, drug, protein)
        self.temps = nn.Parameter(torch.full((3,), math.log(10), dtype = torch.float32)) # temperature for scaling cosine similarity 
        self.biases = nn.Parameter(torch.full((3,), -10, dtype = torch.float32)) # biases for handling imbalance by upweighting positive
    def forward(self, drug_input, protein_input):

        # getting embeddings for drug and protein inputs
        drug_embedding = self.drug_encoder(drug_input)
        protein_embedding = self.protein_encoder(protein_input)

        return drug_embedding, protein_embedding
