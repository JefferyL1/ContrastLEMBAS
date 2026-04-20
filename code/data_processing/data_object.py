import random
import torch
import torch.utils.data as data
from collections import defaultdict

class MultiTaskContrastiveDataset(data.Dataset):
    """ A PyTorch Dataset class for multi-task contrastive learning, learning drug-protein interactions while maintaining sequence similarity
    and drug similarity. """

    def __init__(self, training_known_target_interactions, val_known_target_interactions, smiles_to_embedding, smiles_to_index, drug_similarity_adj,
                 uniprot_to_embedding, uniprot_to_index, protein_similarity_adj):
        """ Initializes the dataset with known drug-protein interactions, SMILES and UniProt embeddings and similarity graphs. 
        
        Args:
            training_known_target_interactions (dict): A dictionary mapping drug SMILES to set of known protein interactions (as UniProt accessions) for training.
            val_known_target_interactions (dict): A dictionary mapping drug SMILES to set of known protein interactions (as UniProt accessions) for validation.
            smiles_to_embedding (dict): A dictionary mapping drug SMILES strings to their corresponding embeddings.
            smiles_to_index (dict): A dictionary mapping drug SMILES strings to their corresponding indices in the drug similarity adjacency matrix.
            drug_similarity_adj (torch.Tensor): A tensor representing the drug similarity adjacency matrix (undirected, symmetric).
            uniprot_to_embedding (dict): A dictionary mapping gene UniProt accessions to their corresponding embeddings.
            uniprot_to_index (dict): A dictionary mapping gene UniProt accessions to their corresponding indices in the protein similarity adjacency matrix.
            protein_similarity_adj (torch.Tensor): A tensor representing the protein similarity adjacency matrix (undirected, symmetric).
        """

        # saving everything 
        known_target_interactions = defaultdict(set, {**training_known_target_interactions, **val_known_target_interactions})
        self.val_known_target_interactions = val_known_target_interactions
        self.training_known_target_interactions = training_known_target_interactions
        self.drug_to_protein_inter= known_target_interactions
        self.protein_to_drug_inter = defaultdict(set)
        for drug, proteins in known_target_interactions.items():
            for protein in proteins:
                self.protein_to_drug_inter[protein].add(drug)
        self.smiles_to_embedding = smiles_to_embedding
        self.smiles_to_index = smiles_to_index
        self.drug_similarity_adj = drug_similarity_adj
        self.uniprot_to_embedding = uniprot_to_embedding
        self.uniprot_to_index = uniprot_to_index
        self.protein_similarity_adj = protein_similarity_adj
        self.index_to_smiles = {idx: smi for smi, idx in smiles_to_index.items()}
        self.index_to_uniprot = {idx: uni for uni, idx in uniprot_to_index.items()}
        
        # creating dataset of positive drug-gene pairs for training
        self.drug_gene_pairs = []
        for drug_smiles, target_genes in known_target_interactions.items():
            for gene in target_genes:
                self.drug_gene_pairs.append((drug_smiles, gene))
        random.shuffle(self.drug_gene_pairs)  # shuffle the pairs for training

        # getting validation and training indices
        train_indices = []
        val_indices = []

        for idx, (drug_smiles, gene) in enumerate(self.drug_gene_pairs):
            if drug_smiles in self.val_known_target_interactions:  # your validation dictionary
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        self.train_indices = train_indices
        self.val_indices = val_indices

        # checking that all drugs and genes in the pairs have corresponding embeddings and indices
        self.check_inputs()
        
        # creating weights for drug-drug, protein-protein, and drug-protein pairs for contrastive learning
        self.drug_target_mat = self.create_drug_target_mat()
        self.drug_weights = self.create_drug_weights()
        self.protein_weights = self.create_protein_weights()
        self.drug_protein_weights = self.create_drug_protein_weights()
    
    def get_label_matrix(self, drug_indices, gene_indices, type = "drug-protein"):
        """ Gets the label matrix for a batch of drug and gene indices. The label matrix is of shape (len(drug_indices), len(gene_indices)) 
        where entry (i, j) is the weight for the pair of drug i and gene j. 
        
        Args:
            drug_indices (list): A list of unique drug indices in the batch.
            gene_indices (list): A list of unique gene indices in the batch.
            type (str): The type of pairs to get labels for - "drug-protein", "drug", or "protein". Determines which weight matrix to use. """

        if type == "drug-protein":
            return self.drug_protein_weights[torch.ix_(drug_indices, gene_indices)]
        elif type == "drug":
            return self.drug_weights[torch.ix_(drug_indices, drug_indices)]
        elif type == "protein":
            return self.protein_weights[torch.ix_(gene_indices, gene_indices)]
        else:
            raise ValueError(f"Invalid type {type} for label matrix. Must be one of 'drug-protein', 'drug', or 'protein'.")
    
    def embedding_dict_errors(self):
        """ Checks that all drugs and genes in the pairs have corresponding embeddings. Returns 
        a list of errors if any drugs or genes are missing embeddings. """

        errors = []
        # check that all drugs and genes in the pairs have corresponding embeddings and indices
        all_drugs = set(drug for drug, gene in self.drug_gene_pairs)
        all_genes = set(gene for drug, gene in self.drug_gene_pairs)
        if not all_drugs.issubset(set(self.smiles_to_embedding.keys())):
            missing_drugs = all_drugs - set(self.smiles_to_embedding.keys())
            errors.append(f"The following drugs are missing embeddings: {missing_drugs}")
        if not all_drugs.issubset(set(self.smiles_to_index.keys())):
            missing_drugs = all_drugs - set(self.smiles_to_index.keys())
            errors.append(f"The following drugs are missing indices: {missing_drugs}")
        if not all_genes.issubset(set(self.uniprot_to_embedding.keys())):
            missing_genes = all_genes - set(self.uniprot_to_embedding.keys())
            errors.append(f"The following genes are missing embeddings: {missing_genes}")
        if not all_genes.issubset(set(self.uniprot_to_index.keys())):
            missing_genes = all_genes - set(self.uniprot_to_index.keys())
            errors.append(f"The following genes are missing indices: {missing_genes}")
        return errors
    
    def index_dict_errors(self):
        """ Checks that all drugs and genes in the pairs have valid indices. Returns a list of errors if any drugs or genes have invalid indices. """

        errors = []
        # check that drug and protein indices are valid (within bound and no duplicates)
        drug_inverse = defaultdict(list)
        for key, idx in self.smiles_to_index.items():
            drug_inverse[idx].append(key)

        gene_inverse = defaultdict(list)
        for key, idx in self.uniprot_to_index.items():
            gene_inverse[idx].append(key)

        drug_collisions = {idx: keys for idx, keys in drug_inverse.items() if len(keys) > 1}
        gene_collisions = {idx: keys for idx, keys in gene_inverse.items() if len(keys) > 1}

        if drug_collisions:
            errors.append(f"Index collisions detected in drugs: {drug_collisions}")
        drug_min_idx, drug_max_idx = 0, self.drug_similarity_adj.shape[0]
        drug_out_of_range_keys = [idx for idx in drug_inverse.keys() if not (drug_min_idx <= idx < drug_max_idx)]
        if drug_out_of_range_keys:
            errors.append(f"Drug indices out of range: {drug_out_of_range_keys}")

        if gene_collisions:
            errors.append(f"Index collisions detected in genes: {gene_collisions}")
        gene_min_idx, gene_max_idx = 0, self.protein_similarity_adj.shape[0]
        gene_out_of_range_keys = [idx for idx in gene_inverse.keys() if not (gene_min_idx <= idx < gene_max_idx)]
        if gene_out_of_range_keys:
            errors.append(f"Gene indices out of range: {gene_out_of_range_keys}")
        
        return errors

    def check_inputs(self):
        """ Checks that the inputs are correct - all of the drugs and proteins have valid embeddings and indices. """

        errors = []
        errors.extend(self.embedding_dict_errors())
        errors.extend(self.index_dict_errors())
        if errors:
            raise ValueError(f"Input validation errors found: {errors}")

        return None

    def create_drug_target_mat(self):
        """ Creates a binary drug-target interaction matrix of shape (num_drugs, num_proteins) where entry (i, j) 
        is 1 if drug i is known to target protein j and 0 otherwise. """

        n_drugs = self.drug_similarity_adj.shape[0]
        n_proteins = self.protein_similarity_adj.shape[0]
        drug_target_mat = torch.zeros(n_drugs, n_proteins, dtype=torch.float32) # (num_drugs, num_proteins)

        for drug_smiles, targets in self.drug_to_protein_inter.items():
            i = self.smiles_to_index[drug_smiles]
            for target in targets:
                j = self.uniprot_to_index[target]
                drug_target_mat[i, j] = 1.0

        return drug_target_mat

    def create_drug_weights(self, upper_drug_threshold = 0.7, lower_drug_threshold = 0.6, upper_target_threshold = 0.7, lower_target_threshold = 0.6):
        """ Weights for (drug, drug) pairs for positive and negative pairs based on drug similarity and known target similarity.

        Weights are:
        +1 if any targets are the same
        +0.5 if drug similarity >= upper_drug_threshold AND any target similarity >= upper_target_threshold
        0 if they are the same drug
        -0.5 if lower_drug_threshold <= drug similarity < upper_drug_threshold AND target similarity < lower_target_threshold
        -1 if drug similarity < lower_drug_threshold AND target similarity < lower_target_threshold.

        All other pairs are uncertain and labeled 0 (not used for contrastive learning).

        Args:
            upper_drug_threshold (float): upper similarity threshold for drugs
            lower_drug_threshold (float): lower similarity threshold for drugs
            upper_target_threshold (float): upper similarity threshold for targets
            lower_target_threshold (float): lower similarity threshold for targets

        Returns:
            torch.Tensor: A tensor of shape (num_drugs, num_drugs) where entry (i, j) is 1 if drugs i and j are similar (positive pair), -1 if they are dissimilar (negative pair), and 0 if they are the same drug or if similarity is not defined.
        
        """
        n_drugs = self.drug_similarity_adj.shape[0]

        # boolean matrix of if any drugs share a target 
        shared_targets = (self.drug_target_mat @ self.drug_target_mat.T) > 0  # (num_drugs, num_drugs)

        # calculating max target similarity for each drug-protein - max similarity between protein in target space of drug i to protein j
        max_drug_protein_sim = (self.drug_target_mat.unsqueeze(2) * self.protein_similarity_adj.unsqueeze(0)).max(dim=1).values  # (n, m)

        # calculating max target similarity for each drug pair - max similarity of protein in target space of drug i to any protein in target space of drug j
        max_target_sim = (max_drug_protein_sim.unsqueeze(1) * self.drug_target_mat.unsqueeze(0)).max(dim=2).values  # (n, n)

        drug_weights = torch.zeros(n_drugs, n_drugs, dtype=torch.float32)  # (num_drugs, num_drugs)
        # condition 1: if any targets are the same, weight is +1
        cond_pos1 = shared_targets
        drug_weights[cond_pos1] = 1.0

        # condition 2: if drug similarity >= upper_drug_threshold AND any target similarity >= upper_target_threshold, weight is +0.5
        cond_pos_half = (~shared_targets 
                     & (self.drug_similarity_adj >= upper_drug_threshold) 
                     & (max_target_sim >= upper_target_threshold))
        drug_weights[cond_pos_half] = 0.5

        # condition 3: if lower_drug_threshold <= drug similarity < upper_drug_threshold AND target similarity < lower_target_threshold, weight is -0.5
        cond_neg_half = (~shared_targets
                & (self.drug_similarity_adj >= lower_drug_threshold)
                & (self.drug_similarity_adj <  upper_drug_threshold)
                & (max_target_sim < lower_target_threshold))
        drug_weights[cond_neg_half] = -0.5

        # condition 4: if drug similarity < lower_drug_threshold AND target similarity < lower_target_threshold, weight is -1
        cond_neg_1 = (~shared_targets
                     & (self.drug_similarity_adj < lower_drug_threshold)
                     & (max_target_sim < lower_target_threshold))
        drug_weights[cond_neg_1] = -1

        drug_weights.fill_diagonal_(0) # set diagonal to 0 (same drug) - not used for contrastive learning

        return drug_weights

    def create_protein_weights(self, upper_protein_threshold = 0.7, lower_protein_threshold = 0.6, upper_drug_threshold = 0.7, lower_drug_threshold = 0.6):
        """ Weights for (protein, protein) pairs for positive and negative pairs based on protein similarity and known drugs similarity.

        Weights are:
        +1 if any known drugs are the same
        +0.5 if protein similarity >= upper_protein_threshold AND max drug similarity >= upper_drug_threshold
        0 if they are the same protein
        -0.5 if lower_protein_threshold <= protein similarity < upper_protein_threshold AND max drug similarity < lower_drug_threshold
        -1 if protein similarity < lower_protein_threshold AND max drug similarity < lower_drug_threshold.

        All other pairs are uncertain and labeled 0 (not used for contrastive learning).

        Args:
            upper_protein_threshold (float): upper similarity threshold for proteins
            lower_protein_threshold (float): lower similarity threshold for proteins
            upper_drug_threshold (float): upper similarity threshold for drugs
            lower_drug_threshold (float): lower similarity threshold for drugs

        Returns:
            torch.Tensor: A tensor of shape (num_drugs, num_drugs) where entry (i, j) is 1 if drugs i and j are similar (positive pair), -1 if they are dissimilar (negative pair), and 0 if they are the same drug or if similarity is not defined.
        
        """
        m = self.protein_similarity_adj.shape[0]
        target_drug_mat = self.drug_target_mat.T  # (num_proteins, num_drugs)

        # boolean matrix of if any targets share a drug
        shared_drugs = (target_drug_mat @ target_drug_mat.T) > 0  # (num_proteins, num_proteins)

        # calculating max drug similarity for each protein-drug - max similarity between drug in drug space of protein i to drug j
        max_protein_drug_sim = (target_drug_mat.unsqueeze(2) * self.drug_similarity_adj.unsqueeze(0)).max(dim=1).values  # (m, n)

        # calculating max drug similarity for each protein pair - max similarity of drug in drug space of protein i to any drug in drug space of protein j
        max_drug_sim = (max_protein_drug_sim.unsqueeze(1) * target_drug_mat.unsqueeze(0)).max(dim=2).values  # (m, m)

        protein_weights = torch.zeros(self.protein_similarity_adj.shape, dtype=torch.float32)  # (num_proteins, num_proteins)

        # condition 1: if any drugs in drug space are the same, weight is +1
        cond_pos1 = shared_drugs
        protein_weights[cond_pos1] = 1.0

        # condition 2: if protein similarity >= upper_protein_threshold AND max drug similarity >= upper_drug_threshold, weight is +0.5
        cond_pos_half = (~shared_drugs
                    & (self.protein_similarity_adj >= upper_protein_threshold)
                    & (max_drug_sim >= upper_drug_threshold))
        protein_weights[cond_pos_half] = 0.5

        # condition 3: if lower_protein_threshold <= protein similarity < upper_protein_threshold AND max drug similarity < lower_drug_threshold, weight is -0.5
        cond_neg_half = (~shared_drugs
                    & (self.protein_similarity_adj >= lower_protein_threshold)  
                    & (self.protein_similarity_adj <  upper_protein_threshold)
                    & (max_drug_sim < lower_drug_threshold))
        protein_weights[cond_neg_half] = -0.5

        # condition 4: if protein similarity < lower_protein_threshold AND max drug similarity < lower_drug_threshold, weight is -1
        cond_neg_1 = (~shared_drugs
                    & (self.protein_similarity_adj < lower_protein_threshold)
                    & (max_drug_sim < lower_drug_threshold))
        protein_weights[cond_neg_1] = -1

        protein_weights.fill_diagonal_(0) # set diagonal to 0 (same protein) - not used for contrastive learning

        return protein_weights

    def create_drug_protein_weights(self, upper_drug_threshold = 0.7, lower_drug_threshold = 0.6, upper_target_threshold = 0.7, lower_target_threshold = 0.6):
        """ Weights for (drug, protein) pairs for positive and negative pairs based on known drug-protein interactions and drug and protein similarity.
        
        Weights are:
        +1 if drug and protein are known to interact
        +0.5 if average drug similarity to drugs that target protein is >= upper_drug_threshold AND average protein similarity to known targets of that drug is >= upper_target_threshold
        -0.5 if lower_drug_threshold <= avg_drug_sim < upper_drug_threshold and avg_protein_sim < lower_target_threshold
        -1 if average drug similarity < lower_drug_threshold AND average protein similarity < lower_protein_threshold
        0 otherwise (interaction is uncertain and not used for contrastive learning).
        """

        n_drugs = self.drug_similarity_adj.shape[0]
        n_proteins = self.protein_similarity_adj.shape[0]

        known_interactions = self.drug_target_mat > 0

        # getting average drug similarity to drugs that target the protein
        drug_counts = self.drug_target_mat.sum(dim=0, keepdim=True).clamp(min=1)  # (1, num_proteins) - number of drugs that target each protein
        avg_drug_sim = (self.drug_similarity_adj @ self.drug_target_mat) / drug_counts  # (num_drugs, num_proteins)

        # getting average protein similarity to proteins that are targeted by the drug
        protein_counts = self.drug_target_mat.sum(dim=1, keepdim=True).clamp(min=1)  # (num_drugs, 1) - number of proteins targeted by each drug
        avg_protein_sim = (self.drug_target_mat @ self.protein_similarity_adj.T) / protein_counts  # (num_drugs, num_proteins)

        drug_protein_weights = torch.zeros(n_drugs, n_proteins, dtype=torch.float32)  # (num_drugs, num_proteins)

        # condition 1: known interaction → +1
        drug_protein_weights[known_interactions] = 1.0

        # condition 2: if average drug similarity to drugs that target protein is >= upper_drug_threshold AND average protein similarity to known targets of that drug is >= upper_target_threshold, weight is +0.5
        cond_pos_half = (~known_interactions
            & (avg_drug_sim >= upper_drug_threshold)
            & (avg_protein_sim >= upper_target_threshold))
        drug_protein_weights[cond_pos_half] = 0.5

        # condition 3: if lower_drug_threshold <= avg_drug_sim <= upper_drug_threshold and avg_protein_sim < lower_target_threshold, weight is -0.5
        cond_neg_half = (~known_interactions
            & (avg_drug_sim >= lower_drug_threshold)
            & (avg_drug_sim < upper_drug_threshold)
            & (avg_protein_sim < lower_target_threshold))
        drug_protein_weights[cond_neg_half] = -0.5

        # condition 4: if average drug similarity < lower_drug_threshold AND average protein similarity < lower_target_threshold, weight is -1
        cond_neg_1 = (~known_interactions
            & (avg_drug_sim < lower_drug_threshold)
            & (avg_protein_sim < lower_target_threshold))
        drug_protein_weights[cond_neg_1] = -1

        return drug_protein_weights

    def __len__(self):
        return len(self.drug_gene_pairs)

    def __getitem__(self, idx):
        drug, gene = self.drug_gene_pairs[idx]
        drug_embedding = self.smiles_to_embedding[drug]
        gene_embedding = self.uniprot_to_embedding[gene]
        drug_index = self.smiles_to_index[drug]
        gene_index = self.uniprot_to_index[gene]

        return drug_index, drug_embedding, gene_index, gene_embedding