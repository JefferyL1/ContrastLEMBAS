import random
import torch
import torch.utils.data as data
from collections import defaultdict

class MultiTaskContrastiveDataset(data.Dataset):
    """ A PyTorch Dataset class for multi-task contrastive learning, learning drug-protein interactions while maintaining sequence similarity
    and drug similarity. """

    def __init__(self, known_target_interactions, smiles_to_embedding, smiles_to_index, drug_similarity_adj,
                 uniprot_to_embedding, uniprot_to_index, protein_similarity_adj):
        """ Initializes the dataset with known drug-protein interactions, SMILES and UniProt embeddings and similarity graphs. 
        
        Args:
            known_target_interactions (dict): A dictionary mapping drug SMILES to set of known protein interactions (as UniProt accessions).
            smiles_to_embedding (dict): A dictionary mapping drug SMILES strings to their corresponding embeddings.
            smiles_to_index (dict): A dictionary mapping drug SMILES strings to their corresponding indices in the drug similarity adjacency matrix.
            drug_similarity_adj (torch.Tensor): A tensor representing the drug similarity adjacency matrix (undirected, symmetric).
            uniprot_to_embedding (dict): A dictionary mapping gene UniProt accessions to their corresponding embeddings.
            uniprot_to_index (dict): A dictionary mapping gene UniProt accessions to their corresponding indices in the protein similarity adjacency matrix.
            protein_similarity_adj (torch.Tensor): A tensor representing the protein similarity adjacency matrix (undirected, symmetric).
        """

        # saving everything 
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
        
        # checking that all drugs and genes in the pairs have corresponding embeddings and indices
        self.check_inputs()
        
        # creating weights for drug-drug, protein-protein, and drug-protein pairs for contrastive learning
        self.drug_weights = self.create_drug_weights()
        self.protein_weights = self.create_protein_weights()
        self.drug_protein_weights = self.create_drug_protein_weights()
        
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
        drug_out_of_range_keys = [idx for idx in self.drug_inverse.keys() if not (drug_min_idx <= idx < drug_max_idx)]
        if drug_out_of_range_keys:
            errors.append(f"Drug indices out of range: {drug_out_of_range_keys}")

        if gene_collisions:
            errors.append(f"Index collisions detected in genes: {gene_collisions}")
        gene_min_idx, gene_max_idx = 0, self.protein_similarity_adj.shape[0]
        gene_out_of_range_keys = [idx for idx in self.gene_inverse.keys() if not (gene_min_idx <= idx < gene_max_idx)]
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

        drug_weights = torch.zeros(self.drug_similarity_adj.shape, dtype=torch.float16)  # (num_drugs, num_drugs)
        for i in range(self.drug_similarity_adj.shape[0]):
            for j in range(i+1, self.drug_similarity_adj.shape[1]):
                
                drug_similarity = self.drug_similarity_adj[i, j]
                drug1_target = self.known_target_interactions[self.index_to_smiles[i]]
                drug2_target = self.known_target_interactions[self.index_to_smiles[j]]
                target_pairs = [(x,y) for x in drug1_target for y in drug2_target]
                target_similarities = [self.protein_similarity_adj[self.uniprot_to_index[x], self.uniprot_to_index[y]] for x, y in target_pairs]
                max_target_sim = max(target_similarities) if target_similarities else 0

                # weight is +1 if any targets are the same (target similarity = 1)
                if drug1_target.intersection(drug2_target):
                    drug_weights[i, j] = 1
                    drug_weights[j, i] = 1

                # weight is +0.5 if drug similarity >= upper_drug_threshold AND any target similarity >= upper_target_threshold
                elif drug_similarity >= upper_drug_threshold and max_target_sim >= upper_target_threshold:
                    drug_weights[i, j] = 1
                    drug_weights[j, i] = 1
                
                # weight is -0.5 if lower_drug_threshold <= drug similarity < upper_drug_threshold AND target similarity < lower_target_threshold
                elif lower_drug_threshold <= drug_similarity < upper_drug_threshold and max_target_sim < lower_target_threshold:
                    drug_weights[i, j] = -0.5
                    drug_weights[j, i] = -0.5

                # weight is -1 if drug similarity < lower_drug_threshold AND target similarity < lower_target_threshold
                elif drug_similarity < lower_drug_threshold and max_target_sim < lower_target_threshold:
                    drug_weights[i, j] = -1
                    drug_weights[j, i] = -1

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

        protein_weights = torch.zeros(self.protein_similarity_adj.shape, dtype=torch.float16)  # (num_drugs, num_drugs)
        for i in range(self.protein_similarity_adj.shape[0]):
            for j in range(i+1, self.protein_similarity_adj.shape[1]):

                protein_similarity = self.protein_similarity_adj[i, j]
                prot1_drugs = self.protein_to_drug_inter[self.index_to_uniprot[i]]
                prot2_drugs = self.protein_to_drug_inter[self.index_to_uniprot[j]]
                drug_pairs = [(x,y) for x in prot1_drugs for y in prot2_drugs]
                drug_similarities = [self.drug_similarity_adj[self.smiles_to_index[x], self.smiles_to_index[y]] for x, y in drug_pairs]
                max_drug_sim = max(drug_similarities) if drug_similarities else 0

                # weight is +1 if any targets are the same
                if prot1_drugs.intersection(prot2_drugs):
                    protein_weights[i, j] = 1
                    protein_weights[j, i] = 1

                # weight is +0.5 if protein similarity >= upper_protein_threshold AND max drug similarity >= upper_drug_threshold
                elif protein_similarity >= upper_protein_threshold and max_drug_sim >= upper_drug_threshold:
                    protein_weights[i, j] = 0.5
                    protein_weights[j, i] = 0.5

                # weight is -0.5 if lower_protein_threshold <= protein similarity < upper_protein_threshold AND max drug similarity < lower_drug_threshold
                elif lower_protein_threshold <= protein_similarity < upper_protein_threshold and max_drug_sim < lower_drug_threshold:
                    protein_weights[i, j] = -0.5
                    protein_weights[j, i] = -0.5

                # weight is -1 if protein similarity < lower_protein_threshold AND max drug similarity < lower_drug_threshold
                elif protein_similarity < lower_protein_threshold and max_drug_sim < lower_drug_threshold:
                    protein_weights[i, j] = -1
                    protein_weights[j, i] = -1

        protein_weights.fill_diagonal_(0) # set diagonal to 0 (same protein) - not used for contrastive learning

        return protein_weights

    def create_drug_protein_weights(self, upper_drug_threshold = 0.7, lower_drug_threshold = 0.6, upper_target_threshold = 0.7, lower_target_threshold = 0.6):
        """ Weights for (drug, protein) pairs for positive and negative pairs based on known drug-protein interactions and drug and protein similarity.
        
        Weights are:
        +1 if drug and protein are known to interact
        +0.5 if average drug similarity to drugs that target protein is >= upper_drug_threshold AND average protein similarity to known targets of that drug is >= upper_target_threshold
        -0.5 if lower_drug_threshold <= avg_drug_sim <= upper_drug_threshold and avg_protein_sim < lower_target_threshold
        -1 if average drug similarity < lower_drug_threshold AND average protein similarity < lower_protein_threshold
        0 otherwise (interaction is uncertain and not used for contrastive learning).
        """

        drug_protein_weights = torch.zeros((self.drug_similarity_adj.shape[0], self.protein_similarity_adj.shape[0]), dtype=torch.float16) # (num_drugs, num_proteins)

        for i in range(self.drug_similarity_adj.shape[0]):
            for j in range(self.protein_similarity_adj.shape[0]):

                # if drug and protein are known to interact, weight is +1
                if self.index_to_uniprot[j] in self.drug_to_protein_inter[self.index_to_smiles[i]]:
                    drug_protein_weights[i, j] = 1
                
                else:
                    drug_sim_to_known_drugs = [self.drug_similarity_adj[i, self.smiles_to_index[other_drug]] for other_drug in self.protein_to_drug_inter[self.index_to_uniprot[j]]]
                    protein_sim_to_known_proteins = [self.protein_similarity_adj[j, self.uniprot_to_index[other_protein]] for other_protein in self.drug_to_protein_inter[self.index_to_smiles[i]]]
                    avg_drug_sim = sum(drug_sim_to_known_drugs) / len(drug_sim_to_known_drugs) if drug_sim_to_known_drugs else 0
                    avg_protein_sim = sum(protein_sim_to_known_proteins) / len(protein_sim_to_known_proteins) if protein_sim_to_known_proteins else 0

                    # weight is +0.5 if average drug similarity >= upper_drug_threshold AND average protein similarity >= upper_target_threshold
                    if avg_drug_sim >= upper_drug_threshold and avg_protein_sim >= upper_target_threshold:
                        drug_protein_weights[i, j] = 0.5
                    
                    # weight is -0.5 if lower_drug_threshold < average drug similarity < upper_drug_threshold AND average protein similarity < lower_protein_threshold
                    elif lower_drug_threshold <= avg_drug_sim <= upper_drug_threshold and avg_protein_sim < lower_target_threshold:
                        drug_protein_weights[i, j] = -0.5

                    # weight is -1 if average drug similarity < lower_drug_threshold AND average protein similarity < lower_protein_threshold
                    elif avg_drug_sim < lower_drug_threshold and avg_protein_sim < lower_target_threshold:
                        drug_protein_weights[i, j] = -1

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