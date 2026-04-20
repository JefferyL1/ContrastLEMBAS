from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import networkx as nx
import pandas as pd
import numpy as np
import re
import pubchempy as pcp
import time
import random
from tqdm import tqdm
import requests
import pickle
import os 
import torch

def read_broad_data(path_to_repurposing_txt_file):
    """ Reads in the Broad Drug Repurposing txt file and returns a dictionary mapping each drug to a list of its target genes.
    Parses through lines of the text and extracting the first pattern matching drug name and any patterns matching gene names 
    (all capital letters or numbers).
    
    Args:
        path_to_repurposing_txt_file (str): Path to the Broad Drug Repurposing txt file.

    Returns:
        dict: A dictionary mapping each drug to a list of its target genes.
    """

    pattern = re.compile(r'^[^\s]+|\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]+\b') # regex pattern to match 

    broad_repurposing_dict = {}
    with open(path_to_repurposing_txt_file) as f:
        for line in f:
            matches = pattern.findall(line)
            if matches:
                matches[0] = matches[0].replace('"', '')
            if len(matches) > 1:
                broad_repurposing_dict[matches[0]] = matches[1:]
    
    return broad_repurposing_dict

def get_smiles_for_drugs(drug_list, save = False, save_path = None):
    """ Takes a list of drug names and returns a dictionary mapping each drug to its SMILES string. 
    Uses PubChemPy to fetch the SMILES strings, with retries and exponential backoff in case of errors.
    Removes those genes without a valid SMILES string. Throttles requests every 50 drugs to avoid hitting rate limits.
    
    Args:
        drug_list (list): A list of drug names.
        
    Returns:
        dict: A dictionary mapping each drug to its SMILES string.
    """

    smiles_dict = {}
    for idx, drug in enumerate(tqdm(drug_list)):
        retries = 0
        while retries < 5:
            try:
                compounds = pcp.get_compounds(drug, namespace='name')
                if compounds:
                    smiles = compounds[0].canonical_smiles
                    smiles_dict[drug] = smiles
                else:
                    smiles = None
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries + random.random()  # exponential backoff
                # print(f"Error fetching {drug}: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)

        # Throttle every 50 drugs
        if (idx + 1) % 50 == 0:
            time.sleep(5)

    if save and save_path:
        with open(f"{save_path}/drug_to_smiles.pkl", "wb") as f:
            pickle.dump(smiles_dict, f)

    return smiles_dict

def get_uniprot_accession(gene_name):
    """ Takes a gene name and returns the corresponding UniProt accession number searching in humans (9606).
    Args:
        gene_name (str): The name of the gene.

    Returns:
        str: The UniProt accession number corresponding to the gene, or None if not found.
    """

    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f'gene_exact:{gene_name} AND organism_id:9606'
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,gene_names,protein_name"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    if not results:
        return None
    
    # Return first matching accession
    return results[0]["primaryAccession"]

def process_gene_names(gene_list, save = False, save_path = None):
    """ Takes a list of gene names and returns a dict of corresponding UniProt accession numbers. 
    Uses get_uniprot_accession to fetch the accession numbers. Removes those without a valid UniProt accession number.
    
    Args:
        gene_list (list): A list of gene names.
    """

    uniprot_dict = {}
    for gene in tqdm(gene_list):
        uniprot_id = get_uniprot_accession(gene)
        if uniprot_id:
            uniprot_dict[gene] = uniprot_id

    if save and save_path:
        with open(f"{save_path}/gene_to_uniprot.pkl", "wb") as f:
            pickle.dump(uniprot_dict, f)

    return uniprot_dict

def get_fasta_for_uniprot_accession(uniprot_accession_list, save = False, save_path = None):
    """ Takes a list of UniProt accession numbers and returns a dictionary mapping each accession number to its corresponding FASTA sequence.
    Uses the UniProt REST API to fetch the FASTA sequences. Removes those without a valid FASTA sequence. 
    
    Args:
        uniprot_accession_list (list): A list of UniProt accession numbers.

    Returns:
        dict: A dictionary mapping each UniProt accession number to its corresponding FASTA sequence.
    """

    fasta_dict = {}
    for accession in tqdm(uniprot_accession_list):
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            fasta_dict[accession] = response.text

    if save and save_path:
        with open(f"{save_path}/uniprot_to_fasta.pkl", "wb") as f:
            pickle.dump(fasta_dict, f)

    return fasta_dict

def get_ecfp4_fingerprint(smiles_list, save = False, save_path = None, type = "torch"):
    """ Takes a list of SMILES strings and returns a dictionary mapping each SMILES string to its ECFP4 fingerprint represented as a numpy array.
    Uses RDKit to compute the ECFP4 fingerprint for each SMILES string. If a SMILES string cannot be parsed, it is skipped and not included in the 
    resulting dictionary. Optionally saves the resulting dictionary into save_path.

    Args:
        smiles_list (list): A list of SMILES strings.
        save (bool): Whether to save the resulting dictionary as a pickle file.
        save_path (str): Path to save the pickle file. 
        type (str): The type of the output fingerprint, either "torch" for PyTorch tensors or "numpy" for numpy arrays. Default is "torch".

    Returns:
        dict: A dictionary mapping each SMILES string to its ECFP4 fingerprint represented as a numpy array.
    """

    n_bits = 2048

    smiles_dict = {}

    for smi in tqdm(smiles_list):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Invalid SMILES: {smi}, skipping")
            continue
        
        # old API
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        
        # convert to numpy array
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        
        if type == "torch":
            torch_arr = torch.from_numpy(arr)
            smiles_dict[smi] = torch_arr
        elif type == "numpy":
            smiles_dict[smi] = arr

    if save and save_path:
        with open(f"{save_path}/smiles_to_ecfp4.pkl", "wb") as f:
            pickle.dump(smiles_dict, f)
    
    return smiles_dict

def process_broad_repurposing_data(path_to_repurposing_txt_file, save = False, save_dir = None):
    """ 
    1. Loads data from Broad Drug Repurposing txt file by . 
    2. Matches drug name to SMILES and filters out unknown drugs. 
    3. Matches gene name to UniProt accession number and filters out unknown genes.
    4. Matches UniProt accession number to FASTA sequenceand filters out unknown genes.
    5. Creates dictionary mapping each drug (SMILES) to a list of its target genes. 

    Optionally saves to save_dir/broad_repurposing_data.

    Args:
        path_to_repurposing_txt_file (str): Path to the Broad Drug Repurposing txt file.
        save (bool): Whether to save the resulting dictionary as a CSV file.
        save_dir (str): Directory to save files.

    Returns:
        dict: A dictionary mapping each drug (SMILES) to a list of its target genes.
    """

    # parsing file 
    broad_repurposing_dict = read_broad_data(path_to_repurposing_txt_file)
    
    # creating drug name to SMILES dictionary
    drug_list = list(broad_repurposing_dict.keys())
    smiles_dict = get_smiles_for_drugs(drug_list)

    # creating gene name to UniProt accession dictionary
    all_genes = set().union(*broad_repurposing_dict.values())
    uniprot_dict = process_gene_names(all_genes)

    # getting FASTA sequences for target proteins
    fasta_dict = get_fasta_for_uniprot_accession(list(uniprot_dict.values()))

    # creating final dictionary mapping drug SMILES to list of target genes (UniProt accessions)
    processed_data = {}
    for drug, genes in broad_repurposing_dict.items():
        if drug in smiles_dict:
            drug_smiles = smiles_dict[drug]
            target_genes = set(uniprot_dict[gene] for gene in genes if gene in fasta_dict)  # only include genes with valid UniProt accessions and FASTA sequences
            if target_genes:  # only include drugs with valid target genes
                processed_data[drug_smiles] = target_genes

    # get ecfp4 fingerprints for drugs in processed_data
    smiles_list = list(processed_data.keys())
    ecfp4_dict = get_ecfp4_fingerprint(smiles_list)

    if save and save_dir:

        os.makedirs(f"{save_dir}/broad_repurposing_data", exist_ok=True)
        with open(f"{save_dir}/broad_repurposing_data/drug_to_smiles.pkl", "wb") as f:
            pickle.dump(smiles_dict, f)

        with open(f"{save_dir}/broad_repurposing_data/gene_to_uniprot.pkl", "wb") as f:
            pickle.dump(uniprot_dict, f)
        
        with open(f"{save_dir}/broad_repurposing_data/uniprot_to_fasta.pkl", "wb") as f:
            pickle.dump(fasta_dict, f)

        with open(f"{save_dir}/broad_repurposing_data/processed_interaction_data.pkl", "wb") as f:
            pickle.dump(processed_data, f)

        with open(f"{save_dir}/broad_repurposing_data/smiles_to_ecfp4.pkl", "wb") as f:
            pickle.dump(ecfp4_dict, f)

    return processed_data


def split_data_into_splits(processed_data, val_set_drugs, save_dir = None):
    """ Splits the processed data into training, validation, and testing sets based on the specified fractions. 
    The split is done at the drug level, meaning that all interactions for a given drug will be in the same split. 
    Optionally saves the resulting splits as pickle files in save_dir.

    Args:
        processed_data (dict): A dictionary mapping each drug (SMILES) to a list of its target genes.
        val_set_drugs (set): A set of drug SMILES that should be included in the validation set.
        save_dir (str): Directory to save files.

    Returns:
        dict: A dictionary mapping each drug (SMILES) to a list of its target genes for the specified split.
    """

    train_data = {}
    val_data = {}

    for drug, targets in processed_data.items():
        if drug in val_set_drugs:
            val_data[drug] = targets
        else:
            train_data[drug] = targets

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/train_interaction_data.pkl", "wb") as f:
            pickle.dump(train_data, f)
        with open(f"{save_dir}/val_interaction_data.pkl", "wb") as f:
            pickle.dump(val_data, f)

    return train_data, val_data