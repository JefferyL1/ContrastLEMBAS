import pickle
import pandas as pd
import torch
import numpy as np
import os

def process_uniprot_to_fasta(uniprot_to_fasta_dict, output_path):
    """ Processes a dictionary mapping UniProt accession numbers to FASTA sequences by removing the > ---- line and only getting string of FASTA.
    
    Args:
        uniprot_to_fasta_dict (dict): A dictionary mapping UniProt accession numbers to their corresponding FASTA sequences.
    """
    for accession in uniprot_to_fasta_dict:
        uniprot_to_fasta_dict[accession] = uniprot_to_fasta_dict[accession].split("\n", 1)[1].replace(" ", "").replace("\n", "")

    with open(f"{output_path}/uniprot_to_fasta.pkl", "wb") as f:
        pickle.dump(uniprot_to_fasta_dict, f)
    return uniprot_to_fasta_dict

def write_mmseqs_fasta(uniprot_to_fasta_dict, output_path):
    """ Writes a FASTA file in the format required for MMseqs2 from a dictionary mapping UniProt accession numbers to FASTA sequences.
    
    Args:
        uniprot_to_fasta_dict (dict): A dictionary mapping UniProt accession numbers to their corresponding FASTA sequences.
        output_path (str): The path where the output FASTA file should be saved.
    """

    with open(output_path, "w") as f:
        for accession, fasta in uniprot_to_fasta_dict.items():
            f.write(f">{accession}\n")
            f.write(f"{fasta}\n")

def load_mmseqs_results(mmseqs_results_path):
    """ Loads MMseqs2 results from a file and returns them as a dataframe.
    
    Args:
        mmseqs_results_path (str): The path to the MMseqs2 results file.
    """
    mmseqs_results = pd.read_csv(mmseqs_results_path, sep="\t", header=None, names=["query_id", "target_id", "seq_identity", "alignment_length", "mismatches", "gap_opens", "q.start", "q.end", "t.start", "t.end", "e_value", "bit_score"])
    return mmseqs_results

def mmseqs_results_to_adj(mmseqs_results, uniprot_ids, type = "torch", save = False, save_path = None):
    """ Converts MMseqs2 results to an adjacency matrix of similarity. MMseqs2 calculated similarity are stored in adjacency matrix and uncalculated pairs
    are assumed to have similarity of 0. The resulting adjacency matrix can be saved to disk as a PyTorch tensor or a NumPy array.

    Args:
        mmseqs_results (df): A dataframe containing the results of MMSeqs2 alignment.
        uniprot_ids (list): A list of UniProt IDs to include in the adjacency matrix.
        output_path (str): The path where the output adjacency matrix file should be saved.
        type (str): The type of adjacency matrix to create - "torch" or "numpy".
        save (bool): Whether to save the resulting adjacency matrix to disk.
        save_path (str): The path where the adjacency matrix should be saved if save is True.
    """
    if type == "torch":
        adj = torch.zeros((len(uniprot_ids), len(uniprot_ids)))
    else:
        adj = np.zeros((len(uniprot_ids), len(uniprot_ids)))

    uniprot_to_index = {uniprot_id: i for i, uniprot_id in enumerate(uniprot_ids)}

    for index, row in mmseqs_results.iterrows():
        accession1, accession2 = row["query_id"], row["target_id"]
        similarity = row["seq_identity"]

        # set similarity in adjacency matrix (undirected)
        adj[uniprot_to_index[accession1], uniprot_to_index[accession2]] = similarity
        adj[uniprot_to_index[accession2], uniprot_to_index[accession1]] = similarity

    if save and save_path:

        if type == "torch":
            save_data = {"similarity" : adj,
                         "uniprot_to_index": uniprot_to_index}
            torch.save(save_data, f"{save_path}/protein_similarity.pt")
        
        else:
            np.savez(f"{save_path}/protein_similarity.npz", similarity=adj, uniprot_to_index=uniprot_to_index)
    return adj
