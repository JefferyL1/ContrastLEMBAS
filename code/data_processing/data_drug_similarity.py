# generating similarity graph
from rdkit import DataStructs
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np
import h5py
import pickle
from tqdm import tqdm
import networkx as nx
import os 
import torch

# Convert Numpy arrays to RDKit ExplicitBitVect objects
def numpy_row_to_fp(row):
    """Convert 1D numpy array (0/1) to RDKit ExplicitBitVect"""
    fp = DataStructs.ExplicitBitVect(len(row))
    on_bits = np.flatnonzero(row)
    for i in on_bits:
        fp.SetBit(int(i))
    return fp

def generate_similarity_mat(drug_smiles, ecfp4_dict, save = False, save_path = None, type = 'torch'):
    """ Generates a similarity matrix by calculating pairwise Tanimoto similarities.

    Args:
        drug_smiles (list): A list of drug SMILES strings.
        ecfp4_dict (dict): A dictionary mapping SMILES strings to their ECFP4 fingerprints.
        save (bool): Whether to save the similarity matrix to disk.
        save_path (str): The path where the similarity matrix should be saved if save is True.
        type (str): The type of the output matrix, either "torch" for PyTorch tensors or "numpy" for numpy arrays. Default is "torch".

    Returns:
        np.ndarray: A 2D array representing the similarity matrix.
        smiles_to_index (dict): A dictionary mapping SMILES strings to their corresponding indices in the similarity matrix.
    """

    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(drug_smiles), len(drug_smiles)), dtype = np.float32)

    # Create a mapping from SMILES to indices for easy lookup
    smiles_to_index = {smi: idx for idx, smi in enumerate(drug_smiles)}
    ecfp4_fingerprint = [ecfp4_dict[smi] for smi in drug_smiles]
    
    fps = [numpy_row_to_fp(ecfp4) for ecfp4 in ecfp4_fingerprint]

    # Calculate pairwise Tanimoto similarities
    for i, smi in tqdm(enumerate(drug_smiles)):
        assert smiles_to_index[smi] == i, f"Index mismatch for {smi}: expected {i}, got {smiles_to_index[smi]}"
        assert np.array_equal(ecfp4_dict[smi], ecfp4_fingerprint[i]), f"Fingerprint mismatch for {smi}"

        query_fp = fps[i]
        fps_subset = fps[i+1:]  # Only compute upper triangle to avoid redundant calculations
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps_subset)
        similarity_matrix[i, i+1:] = sims
        similarity_matrix[i+1:, i] = sims  # Symmetric matrix
    
    similarity_matrix += np.eye(len(drug_smiles))  # Set diagonal to 1
    
    if save and save_path:
        os.makedirs(f"{save_path}", exist_ok=True)

        if type == "torch":
            drug_sim_tensor = torch.from_numpy(similarity_matrix)
            save_data = {"similarity" : drug_sim_tensor,
                         "smiles_to_index": smiles_to_index}
            torch.save(save_data, f"{save_path}/drug_similarity.pt")
        
        else:
            np.savez(f"{save_path}/drug_similarity.npz", similarity=similarity_matrix, smiles_to_index=smiles_to_index)

    return similarity_matrix, smiles_to_index

def turn_similarity_to_graph(similarity_matrix, smiles_to_index, threshold = 0.5):
    """ Converts a similarity matrix into a graph representation. 

    Args:
        similarity_matrix (np.ndarray): A 2D array representing pairwise similarities between drugs.
        smiles_to_index (dict): A dictionary mapping SMILES strings to their corresponding indices in the similarity matrix.
        threshold (float): The similarity threshold above which an edge is created between two drugs.

    Returns:
        networkx.Graph: A graph where nodes represent drugs and edges represent similarities above the threshold.
    """
    index_to_smiles = {idx: smi for smi, idx in smiles_to_index.items()}   

    G = nx.Graph()
    
    # Add nodes
    for smi in smiles_to_index.keys():
        G.add_node(smi)

    # Add edges based on similarity threshold
    for i in tqdm(range(similarity_matrix.shape[0])):
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= threshold:
                smi_i = index_to_smiles[i]
                smi_j = index_to_smiles[j]
                G.add_edge(smi_i, smi_j, weight=similarity_matrix[i, j])
    
    return G

def get_similar_drugs(all_list_drugs, ecfp4_dict, filter_list_drugs, threshold = 0.4):
    """ Finds similar drugs in all_list_drugs that are similar to any drug in filter_list_drugs based on Tanimoto similarity of ECFP4 fingerprints. 

    Args:
        all_list_drugs (list): A list of all drug SMILES strings.
        ecfp4_dict (dict): A dictionary mapping drug SMILES strings to their ECFP4 fingerprints (for all_list_drugs).
        filter_list_drugs (list): A list of drug SMILES strings to filter by.
        threshold (float): The similarity threshold above which drugs are considered similar.

    Returns:
        set: A set of drug SMILES from all_list_drugs that are considered similar to any drug in filter_list_drugs.
    """
    ecfp4_fingerprint = [ecfp4_dict[smi] for smi in all_list_drugs]
    fps = [numpy_row_to_fp(ecfp4) for ecfp4 in ecfp4_fingerprint]

    similar_drugs = set()
    for drug in filter_list_drugs:
        mol = Chem.MolFromSmiles(drug)
        target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(target_fp, fps)
        arr = np.array(sims)
        indices = np.where(arr > threshold)[0]
        similar_drugs.update(all_list_drugs[idx] for idx in indices)

    return similar_drugs

def remove_similar_drugs(processed_interaction_data, drug_list, smiles_to_index, ecfp4_dict, similarity_threshhold = 0.6, save = False, save_path = None):
    """
    Args:
        processed_interaction_data (dict): A dictionary mapping SMILES to set of protein targets (uniprot accessions).
        drug_list (list): A list of drug SMILES strings to remove similar drugs.
        smiles_to_index (dict): A dictionary mapping SMILES strings to their corresponding indices in the similarity matrix.
        ecfp4_dict (dict): A dictionary mapping drug SMILES strings to their ECFP4 fingerprints.
        similarity_threshold (float): The similarity threshold above which drugs are considered similar and should be removed.
        save (bool): Whether to save the resulting training and testing data to disk.
        save_path (str): The path where the training and testing data should be saved if save is True.

    Returns:
        dict: A dictionary mapping SMILES to set of protein targets (uniprot accessions) for training data (with similar drugs removed).
        dict: A dictionary mapping SMILES to set of protein targets (uniprot accessions) for testing data (only similar drugs).

    """

    # finding similar drugs to drug_list
    drugs_to_remove = get_similar_drugs(list(smiles_to_index.keys()), ecfp4_dict, drug_list, threshold = similarity_threshhold)

    # removing similar drugs from processed interaction data (training)
    training_data = {smi: targets for smi, targets in processed_interaction_data.items() if smi not in drugs_to_remove}

    # adding similar drugs to separate (testing)
    testing_data = {smi: targets for smi, targets in processed_interaction_data.items() if smi in drugs_to_remove}

    # optionally save training and testing data
    if save and save_path:
        os.makedirs(f"{save_path}", exist_ok=True)

        with open(f"{save_path}/training_interaction_data.pkl", "wb") as f:
            pickle.dump(training_data, f)
        
        with open(f"{save_path}/validation_interaction_data.pkl", "wb") as f:
            pickle.dump(testing_data, f)

    return training_data, testing_data